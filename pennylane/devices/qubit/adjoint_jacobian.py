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

import numpy as np

import pennylane as qml
from pennylane.math import sum as qmlsum

from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape

from .apply_operation import apply_operation

# from .initialize_state import create_initial_state

# pylint: disable=protected-access, too-many-branches


def adjoint_jacobian(
    dev, tape: QuantumTape, starting_state=None, use_device_state=False
):  # pylint: disable=too-many-statements
    """Implements the adjoint method outlined in
    `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

    After a forward pass, the circuit is reversed by iteratively applying adjoint
    gates to scan backwards through the circuit.

    .. note::
        The adjoint differentiation method has the following restrictions:

        * As it requires knowledge of the statevector, only statevector simulator devices can be
          used.

        * Only expectation values are supported as measurements.

        * Does not work for parametrized observables like
          :class:`~.Hamiltonian` or :class:`~.Hermitian`.

    Args:
        tape (.QuantumTape): circuit that the function takes the gradient of

    Keyword Args:
        starting_state (tensor_like): post-forward pass state to start execution with. It should be
            complex-valued. Takes precedence over ``use_device_state``.
        use_device_state (bool): use current device state to initialize. A forward pass of the same
            circuit should be the last thing the device has executed. If a ``starting_state`` is
            provided, that takes precedence.

    Returns:
        array or tuple[array]: the derivative of the tape with respect to trainable parameters.
        Dimensions are ``(len(observables), len(trainable_params))``.

    Raises:
        QuantumFunctionError: if the input tape has measurements that are not expectation values
            or contains a multi-parameter operation aside from :class:`~.Rot`
    """
    # broadcasted inner product not summing over first dimension of b
    sum_axes = tuple(range(1, dev.num_wires + 1))
    # pylint: disable=unnecessary-lambda-assignment
    dot_product_real = lambda b, k: dev._real(qmlsum(dev._conj(b) * k, axis=sum_axes))

    # Initialization of state
    if starting_state is not None:
        ket = dev._reshape(starting_state, [2] * dev.num_wires)
    else:
        if not use_device_state:
            dev.reset()
            dev.execute(tape)
        ket = dev._pre_rotated_state

    # currently assumes only one observable
    bra = apply_operation(tape.observables[0], ket)

    expanded_ops = []
    for op in reversed(tape.operations):
        if op.num_params > 1:
            ops = op.decomposition()
            expanded_ops.extend(reversed(ops))
        elif op.name not in ("QubitStateVector", "BasisState", "Snapshot"):
            expanded_ops.append(op)

    trainable_params = []
    for k in tape.trainable_params:
        trainable_params.append(k)

    jac = np.zeros((len(tape.observables), len(trainable_params)))

    param_number = len(tape.get_parameters(trainable_only=False, operations_only=True)) - 1
    trainable_param_number = len(trainable_params) - 1
    for op in expanded_ops:
        adj_op = qml.adjoint(op)
        ket = apply_operation(adj_op, ket)

        if op.grad_method is not None:
            if param_number in trainable_params:
                d_op_matrix = operation_derivative(op)
                ket_temp = apply_operation(qml.QubitUnitary(d_op_matrix, wires=op.wires), ket)
                jac[:, trainable_param_number] = 2 * dot_product_real([bra], ket_temp)

                trainable_param_number -= 1
            param_number -= 1

        jac = np.squeeze(jac)

        if jac.ndim == 0:
            return np.array(jac)

        if jac.ndim == 1:
            return tuple(np.array(j) for j in jac)

        # must be 2-dimensional - I think this is only for batching
        return tuple(tuple(np.array(j_) for j_ in j) for j in jac)
