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

import copy
import numpy as np

import pennylane as qml

from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape

from .apply_operation import apply_operation
from .initialize_state import create_initial_state

# pylint: disable=protected-access, too-many-branches


def _dot_product_real(bra, ket, num_wires):
    """Helper for calculating the inner product for adjoint differentiation."""
    # broadcasted inner product not summing over first dimension of the bra tensor
    sum_axes = tuple(range(1, num_wires + 1))
    return qml.math.real(qml.math.sum(qml.math.conj(bra) * ket, axis=sum_axes))


def adjoint_jacobian(tape: QuantumTape):  # pylint: disable=too-many-statements
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
        tape (.QuantumTape): circuit that the function takes the gradient of

    Returns:
        array or tuple[array]: the derivative of the tape with respect to trainable parameters.
        Dimensions are ``(len(observables), len(trainable_params))``.
    """

    # Map wires if custom wire labels used
    if set(tape.wires) != set(range(tape.num_wires)):
        wire_map = {w: i for i, w in enumerate(tape.wires)}
        tape = qml.map_wires(tape, wire_map)

    # Initialization of state
    prep_operation = None if len(tape._prep) == 0 else tape._prep[0]
    ket = create_initial_state(wires=tape.wires, prep_operation=prep_operation)

    for op in tape._ops:
        apply_operation(op, ket)

    n_obs = len(tape.observables)
    bras = []
    for obs in tape.observables:
        bras.append(copy.deepcopy(ket))
        apply_operation(obs, bras[-1])

    jac = np.zeros((len(tape.observables), len(tape.trainable_params)))

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(tape.trainable_params) - 1
    for op in reversed(tape._ops):
        adj_op = qml.adjoint(op)
        apply_operation(adj_op, ket)

        if op.grad_method is not None:
            if param_number in tape.trainable_params:
                d_op_matrix = operation_derivative(op)
                ket_temp = copy.deepcopy(ket)
                apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket_temp)
                for k, bra in enumerate(bras):
                    jac[k, trainable_param_number] = 2 * np.real(
                        (bra.psi.H & ket_temp.psi).contract()
                    )

                trainable_param_number -= 1
            param_number -= 1

        for kk in range(n_obs):
            apply_operation(adj_op, bras[kk])

    # Post-process the Jacobian matrix for the new return
    jac = np.squeeze(jac)

    if jac.ndim == 0:
        return np.array(jac)

    if jac.ndim == 1:
        return tuple(np.array(j) for j in jac)

    # must be 2-dimensional
    return tuple(tuple(np.array(j_) for j_ in j) for j in jac)
