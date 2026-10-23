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

import numpy as np

import pennylane as qp
from pennylane.decomposition.utils import _get_decomp_args
from pennylane.ops.op_math.composite2 import CompositeOp2
from pennylane.testing.decompositions import _capture_decomp_rule_to_tape, _decomp_rule_to_tape


def bits_to_angles(bits, unit):
    """Angle(s) with the given binary decimals (along the last axis), in units of ``unit``."""
    bits = np.array(bits)
    return bits @ (2.0 ** -np.arange(1, bits.shape[-1] + 1)) * unit


def leaf_ops(ops):
    """Recursively unpack composite operators (change of basis, products)."""
    for op in ops:
        if isinstance(op, CompositeOp2):
            yield from leaf_ops(op.decomposition())
        else:
            yield op


def decomp_to_tape(op, rule):
    """Decompose ``op`` with ``rule`` into a tape, tracing with program capture if it is enabled,
    so that the operator data are abstract during the decomposition."""
    if qp.capture.enabled():
        return _capture_decomp_rule_to_tape(rule, op)
    _, args, kwargs = _get_decomp_args(op)
    return _decomp_rule_to_tape(rule, args, kwargs)


def assert_decomp_realizes_op(op, tape, phase_grad_wires, seed):
    """Assert that the operations of ``tape`` realize ``op`` on a random input state, if a phase
    gradient state is prepared on ``phase_grad_wires``, returning all other wires to zero."""
    aux_wires = [
        w for w in qp.wires.Wires.all_wires([phase_grad_wires, tape.wires]) if w not in op.wires
    ]
    rng = np.random.default_rng(seed)
    in_state = rng.random(2 ** len(op.wires)) + 1j * rng.random(2 ** len(op.wires))
    in_state /= np.linalg.norm(in_state)

    with qp.capture.pause():
        # Captured decompositions carry jax data, which is slow to simulate
        [tape], _ = qp.transforms.convert_to_numpy_parameters(tape)
        circuit = qp.tape.QuantumScript(
            [
                qp.StatePrep(in_state, op.wires),
                qp.PhaseGradientStatePrep(phase_grad_wires),
                *tape.operations,
                qp.adjoint(qp.PhaseGradientStatePrep(phase_grad_wires)),
            ],
            [qp.state()],
        )
        dev = qp.device("default.qubit", wires=op.wires + qp.wires.Wires(aux_wires))
        out_state = qp.execute([circuit], dev)[0]
        expected = np.kron(
            qp.matrix(op, wire_order=op.wires) @ in_state, np.eye(2 ** len(aux_wires), 1)[:, 0]
        )

    assert np.allclose(out_state, expected)
