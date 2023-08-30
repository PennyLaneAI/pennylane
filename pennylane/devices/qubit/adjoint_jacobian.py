# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to apply adjoint jacobian differentiation"""
from numbers import Number
from typing import Tuple
import numpy as np

import pennylane as qml

from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape

from .apply_operation import apply_operation
from .simulate import get_final_state

# pylint: disable=protected-access, too-many-branches


def _dot_product_real(bra, ket, num_wires):
    """Helper for calculating the inner product for adjoint differentiation."""
    # broadcasted inner product not summing over first dimension of the bra tensor
    sum_axes = tuple(range(1, num_wires + 1))
    return qml.math.real(qml.math.sum(qml.math.conj(bra) * ket, axis=sum_axes))


def adjoint_jacobian(tape: QuantumTape, state=None):
    """Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::

        The adjoint differentiation method has the following restrictions:

        * Only expectation values are supported as measurements.

        * Cannot differentiate with respect to observables.

        * Observable being measured must have a matrix.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        array or tuple[array]: the derivative of the tape with respect to trainable parameters.
        Dimensions are ``(len(observables), len(trainable_params))``.
    """
    # Map wires if custom wire labels used
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tape = qml.map_wires(tape, wire_map)

    ket = state if state is not None else get_final_state(tape)[0]

    n_obs = len(tape.observables)
    bras = np.empty([n_obs] + [2] * len(tape.wires), dtype=np.complex128)
    for kk, obs in enumerate(tape.observables):
        bras[kk, ...] = apply_operation(obs, ket)

    jac = np.zeros((len(tape.observables), len(tape.trainable_params)))

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1
    for op in reversed(tape.operations[tape.num_preps :]):
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.grad_method is not None:
            if param_number in tape.trainable_params:
                d_op_matrix = operation_derivative(op)
                ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)
                jac[:, trainable_param_number] = 2 * _dot_product_real(
                    bras, ket_temp, len(tape.wires)
                )

                trainable_param_number -= 1
            param_number -= 1

        for kk in range(n_obs):
            bras[kk, ...] = apply_operation(adj_op, bras[kk, ...])

    # Post-process the Jacobian matrix for the new return
    jac = np.squeeze(jac)

    if jac.ndim == 0:
        return np.array(jac)

    if jac.ndim == 1:
        return tuple(np.array(j) for j in jac)

    # must be 2-dimensional
    return tuple(tuple(np.array(j_) for j_ in j) for j in jac)


def adjoint_jvp(tape: QuantumTape, tangents: Tuple[Number], state=None):
    """The jacobian vector product used in forward mode calculation of derivatives.

    Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::

        The adjoint differentiation method has the following restrictions:

        * Only expectation values are supported as measurements.

        * Cannot differentiate with respect to observables.

        * Observable being measured must have a matrix.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        tangents (Tuple[Number]): gradient vector for input parameters.
        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        Tuple[Number]: gradient vector for output parameters
    """
    # Map wires if custom wire labels used
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tape = qml.map_wires(tape, wire_map)

    ket = state if state is not None else get_final_state(tape)[0]

    n_obs = len(tape.observables)
    bras = np.empty([n_obs] + [2] * len(tape.wires), dtype=np.complex128)
    for i, obs in enumerate(tape.observables):
        bras[i] = apply_operation(obs, ket)

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1

    tangents_out = np.zeros(n_obs)

    for op in reversed(tape.operations[tape.num_preps :]):
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.grad_method is not None:
            if param_number in tape.trainable_params:
                # don't do anything if the tangent is 0
                if not np.allclose(tangents[trainable_param_number], 0):
                    d_op_matrix = operation_derivative(op)
                    ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)

                    tangents_out += (
                        2
                        * _dot_product_real(bras, ket_temp, len(tape.wires))
                        * tangents[trainable_param_number]
                    )

                trainable_param_number -= 1
            param_number -= 1

        for i in range(n_obs):
            bras[i] = apply_operation(adj_op, bras[i])

    if n_obs == 1:
        return np.array(tangents_out[0])

    return tuple(np.array(t) for t in tangents_out)


def adjoint_vjp(tape: QuantumTape, cotangents: Tuple[Number], state=None):
    """The vector jacobian product used in reverse-mode differentiation.

    Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::

        The adjoint differentiation method has the following restrictions:

        * Only expectation values are supported as measurements.

        * Cannot differentiate with respect to observables.

        * Observable being measured must have a matrix.

    Args:
        tape (QuantumTape): circuit that the function takes the gradient of
        cotangents (Tuple[Number]): gradient vector for output parameters
        state (TensorLike): the final state of the circuit; if not provided,
            the final state will be computed by executing the tape

    Returns:
        Tuple[Number]: gradient vector for input parameters
    """
    # Map wires if custom wire labels used
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tape = qml.map_wires(tape, wire_map)

    ket = state if state is not None else get_final_state(tape)[0]

    obs = qml.dot(cotangents, tape.observables)
    bra = apply_operation(obs, ket)

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1

    cotangents_in = np.empty(len(tape.trainable_params))

    for op in reversed(tape.operations[tape.num_preps :]):
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.grad_method is not None:
            if param_number in tape.trainable_params:
                d_op_matrix = operation_derivative(op)
                ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)

                cotangents_in[trainable_param_number] = 2 * np.real(np.sum(np.conj(bra) * ket_temp))

                trainable_param_number -= 1
            param_number -= 1

        bra = apply_operation(adj_op, bra)

    if len(tape.trainable_params) == 1:
        return np.array(cotangents_in[0])

    return tuple(np.array(t) for t in cotangents_in)
