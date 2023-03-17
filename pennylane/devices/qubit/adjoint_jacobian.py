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

from pennylane.operation import operation_derivative
from pennylane.tape import QuantumTape
from pennylane.measurements import ExpectationMP

from .apply_operation import apply_operation

from .initialize_state import create_initial_state

# pylint: disable=protected-access, too-many-branches


def adjoint_jacobian(
    tape: QuantumTape, prep_operation: qml.operation.StatePrep = None
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

        * Does not work for parametrized observables like :class:`~.Hamiltonian`,
          :class:`~.Sum` or :class:`~.Hermitian`.

    Args:
        tape (.QuantumTape): circuit that the function takes the gradient of
        prep_operation (.StatePrep): state preparation operation applied for state initialization

    Returns:
        array or tuple[array]: the derivative of the tape with respect to trainable parameters.
        Dimensions are ``(len(observables), len(trainable_params))``.

    Raises:
        QuantumFunctionError: if the input tape has measurements that are not expectation values
            or contains a multi-parameter operation aside from :class:`~.Rot`
    """
    # broadcasted inner product not summing over first dimension of the bra tensor
    sum_axes = tuple(range(1, len(tape.wires) + 1))
    # pylint: disable=unnecessary-lambda-assignment
    dot_product_real = lambda b, k: qml.math.real(qml.math.sum(qml.math.conj(b) * k, axis=sum_axes))

    # Check validity of measurements
    for m in tape.measurements:
        if not isinstance(m, ExpectationMP):
            raise qml.QuantumFunctionError(
                "Adjoint differentiation method does not support"
                f" measurement {m.__class__.__name__}"
            )

        if m.obs.name in {"Hamiltonian", "Sum"}:
            raise qml.QuantumFunctionError(
                f"Adjoint differentiation method does not support observable {m.obs.name}."
            )

        if not hasattr(m.obs, "base_name"):
            m.obs.base_name = None  # This is needed for when the observable is a tensor product

    # Initialization of state
    ket = create_initial_state(
        wires=tape.wires, prep_operation=prep_operation
    )  #  ket(0) if prep_operation is None, else
    for op in tape.operations:
        ket = apply_operation(op, ket)

    n_obs = len(tape.observables)
    bras = np.empty([n_obs] + [2] * tape.wires, dtype=np.complex128)
    for kk in range(n_obs):
        bras[kk, ...] = apply_operation(tape.observables[kk], ket)

    expanded_ops = []
    for op in reversed(tape.operations):
        if op.num_params > 1:
            if not isinstance(op, qml.Rot):
                raise qml.QuantumFunctionError(
                    f"The {op.name} operation is not supported using "
                    'the "adjoint" differentiation method'
                )
            ops = op.decomposition()
            expanded_ops.extend(reversed(ops))
        elif op.name not in ("QubitStateVector", "BasisState", "Snapshot"):
            expanded_ops.append(op)

    trainable_params = []
    for k in tape.trainable_params:
        # TODO: Add check for trainable params on observable
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
                jac[:, trainable_param_number] = 2 * dot_product_real(bras, ket_temp)

                trainable_param_number -= 1
            param_number -= 1

        for kk in range(n_obs):
            bras[kk, ...] = apply_operation(adj_op, bras[kk, ...])

    if not qml.active_return():
        return jac

    # Post-process the Jacobian matrix for the new return types
    jac = np.squeeze(jac)

    if jac.ndim == 0:
        return np.array(jac)

    if jac.ndim == 1:
        return tuple(np.array(j) for j in jac)

    # must be 2-dimensional
    return tuple(tuple(np.array(j_) for j_ in j) for j in jac)
