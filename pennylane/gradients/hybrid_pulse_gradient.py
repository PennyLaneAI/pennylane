# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains functions for computing the stochastic parameter-shift gradient
of pulse sequences in a qubit-based quantum tape.
"""
from collections.abc import Sequence
import numpy as np

import pennylane as qml

from pennylane._device import _get_num_copies
from pennylane.pulse import ParametrizedEvolution
from pennylane.ops.qubit.special_unitary import pauli_basis_matrices, pauli_basis_strings

from .finite_difference import _all_zero_grad_new, _no_trainable_grad_new
from .parameter_shift import (
    _make_zero_rep,
    _reorder_grads,
)
from .stoch_pulse_gradient import _assert_has_jax, _split_evol_tapes
from .gradient_transform import (
    assert_active_return,
    assert_no_state_returns,
    assert_no_variance,
    choose_grad_methods,
    gradient_analysis_and_validation,
    gradient_transform,
)
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    # Handling the case where JAX is not installed is done via _assert_has_jax
    pass


def _get_one_parameter_generators(op):
    r"""Compute the generators of one-parameter groups that reproduce
    the partial derivatives of a parameterized evolution gate.

    Args:
        op (): TODO

    Returns:
        tensor_like: The generators for one-parameter groups that reproduce the
        partial derivatives of the special unitary gate.
        The output shape is ``(num_params, 2**num_wires, 2**num_wires)`` where
        ``num_params`` is the number of parameters of the parameterized evolution.

    TODO:
    Consider a special unitary gate parametrized in the following way:

    .. math::

        U(\theta) &= \exp\{A(\theta)\}\\
        A(\theta) &= \sum_{m=1}^d i \theta_m P_m\\
        P_m &\in \{I, X, Y, Z\}^{\otimes n} \setminus \{I^{\otimes n}\}

    Then the partial derivatives of the gate can be shown to be given by

    .. math::

        \frac{\partial}{\partial \theta_\ell} U(\theta) = U(\theta)
        \frac{\mathrm{d}}{\mathrm{d}t}\exp\left(t\Omega_\ell(\theta)\right)\large|_{t=0}

    where :math:`\Omega_\ell(\theta)` is the one-parameter generator belonging to the partial
    derivative :math:`\partial_\ell U(\theta)` at the parameters :math:`\theta`.
    It can be computed via

    .. math::

        \Omega_\ell(\theta) = U(\theta)^\dagger
        \left(\frac{\partial}{\partial \theta_\ell}\mathfrak{Re}[U(\theta)]
        +i\frac{\partial}{\partial \theta_\ell}\mathfrak{Im}[U(\theta)]\right)

    where we may compute the derivatives :math:`\frac{\partial}{\partial \theta_\ell} U(\theta)` using auto-differentiation.

    .. warning::

        This function requires JAX to be installed.

    """
    #theta = self.data[0]
    #if len(qml.math.shape(theta)) > 1:
        #raise ValueError("Broadcasting is not supported.")

    num_wires = len(op.wires)

    def _compute_matrix(*args, **kwargs):
        return op(*args, **kwargs).matrix()

    data = [qml.math.cast_like(d, 1j) for d in op.data]
    # These lines compute the Jacobian of compute_matrix every time -> to be optimized
    jac = jax.jacobian(_compute_matrix, argnums=0, holomorphic=True)(data, t=op.t)

    # Compute the Omegas from the Jacobian.
    U_dagger = qml.math.conj(_compute_matrix([qml.math.detach(d) for d in data], t=op.t))
    # After contracting, move the parameter derivative axis to the first position
    return [qml.math.moveaxis(qml.math.tensordot(U_dagger, j, axes=[[0], [0]]), (0, 1), (-2, -1)) for j in jac]

def _get_one_parameter_coeffs(generators, num_wires):
    """To be improved via fast Hadamard transform"""
    basis = pauli_basis_matrices(num_wires)
    return [
        qml.math.tensordot(basis, g, axes=[[1, 2], [-1, -2]]) / 2**num_wires 
        for g in generators
    ]

def _get_pauli_rots_to_insert(coefficients, wires, atol=1e-8, rtol=0.):
    #pauli_rot_parameters = [2j * coeffs for coeffs in coefficients]
    zero_ids = []
    words = pauli_basis_strings(len(wires))
    pauli_rots = []
    new_coefficients = [[] for _ in coefficients]
    for i, (word, *coeffs) in enumerate(zip(words, *coefficients)):
        if all(qml.math.allclose(c, 0., atol=atol, rtol=rtol) for c in coeffs):
            #print(f"{word} : all zero")
            zero_ids.append(i)
            continue
        #print(coeffs)
        #print(f"nonzero: {word}")
        pauli_rots.extend([qml.PauliRot(sign * np.pi/2, word, wires=wires) for sign in [1, -1]])
        for new_coeffs, c in zip(new_coefficients, coeffs):
            new_coeffs.append(c)
    return pauli_rots, new_coefficients

def _insert_op(tape, insert_ops, op_idx):
    """Replace a marked ``ParametrizedEvolution`` in a given tape by provided operations, creating
    one tape per group of operations.

    Args:
        tape (QuantumTape): original tape
        split_evolve_ops (tuple[list[qml.Operation]]): The time-split evolution operations as
            created by ``_split_evol_ops``. For each group of operations, a new tape
            is created.
        op_idx (int): index of the operation to replace within the tape

    Returns:
        list[QuantumTape]: new tapes with replaced operation, one tape per group of operations in
        ``split_evolve_ops``.
    """
    ops_pre = tape.operations[:op_idx]
    ops_post = tape.operations[op_idx:]
    return [
        qml.tape.QuantumScript(ops_pre + insert + ops_post, tape.measurements)
        for insert in insert_ops
    ]

def _hybrid_pulse_grad(tape, argnum=None, shots=None, tolerances=None):
    transform_name = "hybrid pulse parameter-shift"
    _assert_has_jax(transform_name)
    assert_active_return(transform_name)
    assert_no_state_returns(tape.measurements)
    assert_no_variance(tape.measurements, transform_name)

    if tolerances is None:
        tolerances = (1e-7, 0.)

    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad_new(tape, shots)

    #if use_broadcasting and tape.batch_size is not None:
        #raise ValueError("Broadcasting is not supported for tapes that already are broadcasted.")

    diff_methods = gradient_analysis_and_validation(tape, "analytic", grad_fn=hybrid_pulse_grad)

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad_new(tape, shots)

    method_map = choose_grad_methods(diff_methods, argnum)

    argnum = [i for i, dm in method_map.items() if dm == "A"]

    return _expval_hybrid_pulse_grad(tape, argnum, shots, tolerances)

def _generate_tapes_and_coeffs(tape, idx, atol, rtol):
    op, op_idx, term_idx = tape.get_operation(idx)
    if not isinstance(op, ParametrizedEvolution):
        raise ValueError(
            "hybrid_pulse_grad does not support differentiating parameters of "
            "other operations than pulses."
        )

    num_wires = len(op.wires)
    generators = _get_one_parameter_generators(op)
    coeffs = _get_one_parameter_coeffs(generators, num_wires)
    pauli_rots = _get_pauli_rots_to_insert(coeffs, op.wires, atol, rtol)
    tapes = _insert_op(tape, pauli_rots, op_idx)
    return tapes, coeffs

def _contract_with_coeffs(results, coeffs, single_measure, shot_vector):
    if single_measure and not shot_vector:
        return tuple([qml.math.tensordot(results, c, axes=[[0], [0]]) for c in coeffs])
    raise NotImplementedError


    

def _expval_hybrid_pulse_grad(tape, argnum, shots, tolerances):
    atol, rtol = tolerances
    gradient_data = []
    tapes = []
    for idx, trainable_idx in enumerate(tape.trainable_params):
        if trainable_idx not in argnum:
            # Only the number of tapes is needed to indicate a zero gradient entry
            gradient_data.append((0, None))
            continue

        _tapes, _coeffs = _generate_tapes_and_coeffs(tape, idx, atol, rtol)
        gradient_data.append((len(_tapes), _coeffs))
        tapes.extend(_tapes)

    num_measurements = len(tape.measurements)
    single_measure = num_measurements == 1
    num_params = len(tape.trainable_params)
    shot_vector = isinstance(shots, Sequence)
    tape_specs = (single_measure, num_params, num_measurements, shot_vector, shots)

    def processing_fn(results):
        start = 0
        grads = []
        for num_tapes, coeffs in gradient_data:
            if num_tapes == 0:
                grads.append(None)
                continue
            res = results[start : start + num_tapes]
            start += num_tapes
            # Apply the postprocessing of the parameter-shift rule and contract
            # with classical Jacobian, effectively computing the integral approximation
            g = _contract_with_coeffs(res, coeffs, single_measure, shot_vector)
            grads.append(g)

        # g will have been defined at least once (because otherwise all gradients would have
        # been zero), providing a representative for a zero gradient to emulate its type/shape.
        zero_rep = _make_zero_rep(g, single_measure, shot_vector)

        # Fill in zero-valued gradients
        grads = [zero_rep if g is None else g for g in grads]

        return _reorder_grads(grads, tape_specs)

    return tapes, processing_fn

def expand_invalid_trainable_hybrid_pulse_grad(x, *args, **kwargs):
    r"""Do not expand any operation. We expect the ``hybrid_pulse_grad`` to be used
    on pulse programs and we do not expect decomposition pipelines between pulses
    and gate-based circuits yet.
    """
    # pylint:disable=unused-argument
    return x


hybrid_pulse_grad = gradient_transform(
    _hybrid_pulse_grad, expand_fn=expand_invalid_trainable_hybrid_pulse_grad
)
