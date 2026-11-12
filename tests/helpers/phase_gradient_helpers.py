# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pytest helper functions for testing phase gradient decomposition rules.
"""

from functools import partial

import numpy as np

import pennylane as qp
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.composite2 import CompositeOp2


def bits_to_angles(decimals, unit):
    """Angle(s) with the given binary decimals (along the last axis), in units of ``unit``."""
    decimals = np.array(decimals)
    return decimals @ (2.0 ** -np.arange(1, decimals.shape[-1] + 1)) * unit


def leaf_ops(ops):
    """Recursively unpack composite operators (change of basis, products) and adjoints of
    composite or adjoint operators."""
    for op in ops:
        if isinstance(op, CompositeOp2) or (
            isinstance(op, Adjoint2) and isinstance(op.base, (CompositeOp2, Adjoint2))
        ):
            yield from leaf_ops(op.decomposition())
        else:
            yield op


def count_ands(ops):
    """Number of ``TemporaryAND`` gates and number of ANDs in ``SemiAdder`` gates and their
    adjoints (``len(y_wires) - 1`` each) among the :func:`leaf_ops` of ``ops``. Adjoint
    ``TemporaryAND`` gates are measurement-based and not counted."""
    leaves = list(leaf_ops(ops))
    ands = sum(isinstance(op, qp.TemporaryAND) for op in leaves)
    adders = [op.base if isinstance(op, Adjoint2) else op for op in leaves]
    adder_ands = sum(len(op.y_wires) - 1 for op in adders if isinstance(op, qp.SemiAdder))
    return ands, adder_ands


def work_wire_spec_total(rule, op):
    """Total number of work wires that ``rule`` requests to decompose ``op``. Catalyst excludes
    rules for which this is positive; like Catalyst, pass the arguments the rule is called with.
    This matches Catalyst for Operator2 ops with ``wire_argnames != ("wires",)``, to which Catalyst
    also passes all arguments as keywords, if the spec only depends on the lengths of the
    arguments: Catalyst passes dummy values for the wires and dynamic arguments."""
    return rule.get_work_wire_spec(**op.arguments).total


def _final_state(prepare, ops, wires):
    """State of default.qubit on ``wires`` after the state preparation that ``prepare()`` queues
    and the :func:`leaf_ops` of ``ops``."""
    with qp.capture.pause():
        # default.qubit simulates composite blocks as dense matrices on all their wires
        leaves = list(leaf_ops(ops))

        # Captured operations carry jax data, which is slow to simulate
        @qp.transforms.convert_to_numpy_parameters
        @qp.qnode(qp.device("default.qubit", wires=wires), interface="numpy")
        def circuit():
            prepare()
            for op in leaves:
                qp.apply(op)
            return qp.state()

        return circuit()


def assert_decomp_realizes_op(op, tape, phase_grad_wires, seed, *, zeroed_wires=()):
    """Assert that the operations of ``tape`` realize ``op`` on a random input state, if a phase
    gradient state is prepared on ``phase_grad_wires``, returning all other wires to zero.

    The ``zeroed_wires`` of ``op``, such as work wires, start in zero instead of the random state
    and must return to zero as well."""
    zeroed_wires = qp.wires.Wires(zeroed_wires)
    phase_grad_wires = qp.wires.Wires(phase_grad_wires)
    in_wires = qp.wires.Wires([w for w in op.wires if w not in zeroed_wires])
    aux_wires = qp.wires.Wires(
        [w for w in op.wires + tape.wires if w not in in_wires and w not in phase_grad_wires]
    )
    rng = np.random.default_rng(seed)
    in_state = rng.random(2 ** len(in_wires)) + 1j * rng.random(2 ** len(in_wires))
    in_state /= np.linalg.norm(in_state)

    with qp.capture.pause():
        # default.qubit decomposes a phase gradient state preparation that is not the first
        # operation into hundreds of gates, so prepare it together with the input state
        phase_grad_state = qp.PhaseGradientStatePrep(phase_grad_wires).state_vector().flatten()

    # ``in_wires + op.wires`` puts the zeroed wires last, so the first column holds the amplitudes
    # with all of them in zero
    op_state = _final_state(partial(qp.StatePrep, in_state, in_wires), [op], in_wires + op.wires)
    op_state = op_state.reshape(2 ** len(in_wires), -1)[:, 0]

    out_state = _final_state(
        partial(qp.StatePrep, np.kron(in_state, phase_grad_state), in_wires + phase_grad_wires),
        tape.operations,
        in_wires + phase_grad_wires + aux_wires,
    )
    expected = np.kron(op_state, phase_grad_state)
    expected = np.kron(expected, np.eye(2 ** len(aux_wires), 1)[:, 0])

    assert np.allclose(out_state, expected)
    # np.allclose misses global phase errors below about 1e-5
    overlap = np.vdot(expected, out_state)
    assert np.isclose(overlap, 1, rtol=0, atol=1e-9), f"Overlap with the expected state: {overlap}"


def record(qfunc, *args):
    """Operations of ``qfunc(*args)``. Under program capture, the arguments are traced as jax
    arrays, as in decomposition rules, and the operations are extracted from the captured plxpr."""
    if not qp.capture.enabled():
        return qp.tape.make_qscript(qfunc)(*args).operations
    import jax.numpy as jnp

    args = [jnp.array(arg) for arg in args]
    plxpr = qp.capture.make_plxpr(qfunc, autograph=False)(*args)
    return qp.testing.plxpr_to_tape(plxpr.jaxpr, plxpr.consts, *args).operations


def basis_outputs(ops, num_wires, inputs):
    """Big-endian bits of the states that ``ops`` map the basis states ``inputs`` of the wires
    ``0, ..., num_wires - 1`` to, one row per input, asserting that each of them is a basis state
    with phase 1."""
    wires = list(range(num_wires))
    indices = []
    for x in inputs:
        state = _final_state(partial(qp.BasisState, x, wires), ops, wires)
        index = int(np.argmax(np.abs(state)))
        assert np.isclose(state[index], 1)
        indices.append(index)
    return qp.math.int_to_binary(np.array(indices), num_wires)
