# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Quantum tape for implementing the gradient method outlined in https://arxiv.org/abs/2009.02823,
referred to here as the "rewind" method.
"""
import numpy as np

# pylint: disable=protected-access
import pennylane as qml

from .jacobian_tape import JacobianTape


class RewindTape(JacobianTape):
    r"""Quantum tape for computing gradients using the `rewind <https://arxiv.org/abs/2009.02823>`__
    method.

    After a forward pass, the circuit is "rewound" by iteratively applying the inverse gate to scan
    backwards through the circuit. This method is similar to the reversible method, but has a lower
    time overhead and a similar memory overhead.

    .. note::

        The rewind analytic differentation method has the following restrictions:

        * As it requires knowledge of the statevector, only statevector simulator devices can be
          used.

        * Only expectation values are supported as measurements.

    This class extends the :class:`~.jacobian` method of the quantum tape to support analytic
    gradients of qubit operations using the rewind method of analytic differentiation.
    This gradient method returns *exact* gradients, however requires use of a statevector simulator.
    Simply create the tape, and then call the Jacobian method:

    >>> tape.jacobian(dev)

    For more details on the quantum tape, please see :class:`~.JacobianTape`.
    """

    def jacobian(self, device, params=None, **options):
        # The rewind tape only support differentiating expectation values of observables for now.
        for m in self.measurements:
            if m.return_type is not qml.operation.Expectation:
                raise qml.QuantumFunctionError(
                    f"The {m.return_type.value} return type is not supported with the rewind "
                    f"gradient method"
                )

        method = options.get("method", "analytic")

        if method == "device":
            # Using device mode; simply query the device for the Jacobian
            return self.device_pd(device, params=params, **options)
        if method == "numeric":
            return super().jacobian(device, params=params, **options)

        supported_device = hasattr(device, "_apply_operation")
        supported_device = supported_device and hasattr(device, "_apply_unitary")
        supported_device = supported_device and device.capabilities().get("returns_state")

        if not supported_device:
            raise qml.QuantumFunctionError(
                "The rewind gradient method is only supported on statevector-based devices"
            )

        return self._rewind_jacobian(device, params=params)

    def _rewind_jacobian(self, device, params=None):
        """Implements the method outlined in https://arxiv.org/abs/2009.02823 to calculate the
        Jacobian."""
        # Perform the forward pass
        self.execute(device, params=params)

        if params is not None:
            self.set_parameters(params)

        phi = device._reshape(device.state, [2] * device.num_wires)

        for obs in self.observables:  # This is needed for when the observable is a tensor product
            if not hasattr(obs, "base_name"):
                obs.base_name = None
        lambdas = [device._apply_operation(phi, obs) for obs in self.observables]

        jac = np.zeros((len(self.observables), len(self.trainable_params)))

        expanded_ops = []
        for op in reversed(self.operations):
            if op.num_params > 1:
                if isinstance(op, qml.Rot) and not op.inverse:
                    ops = op.decomposition(*op.parameters, wires=op.wires)
                    expanded_ops.extend(reversed(ops))
                else:
                    raise qml.QuantumFunctionError(f"The {op.name} operation is not supported using "
                                                   'the "rewind" differentiation method')
            else:
                expanded_ops.append(op)

        expanded_ops = [o for o in expanded_ops if not isinstance(o, (qml.QubitStateVector, qml.BasisState))]
        dot_product_real = lambda a, b: device._real(qml.math.sum(device._conj(a) * b))

        param_number = len(self._par_info) - 1
        trainable_param_number = len(self.trainable_params) - 1
        for op in expanded_ops:

            if op.grad_method and param_number in self.trainable_params:
                d_op_matrix = operation_derivative(op)

            op.inv()
            phi = device._apply_operation(phi, op)

            if op.grad_method:
                if param_number in self.trainable_params:
                    mu = device._apply_unitary(phi, d_op_matrix, op.wires)

                    jac_column = np.array(
                        [2 * dot_product_real(lambda_, mu) for lambda_ in lambdas]
                    )
                    jac[:, trainable_param_number] = jac_column
                    trainable_param_number -= 1
                param_number -= 1

            lambdas = [device._apply_operation(lambda_, op) for lambda_ in lambdas]
            op.inv()

        return jac


def operation_derivative(operation: qml.operation.Operation) -> np.ndarray:
    r"""Calculate the derivative of an operation.

    For an operation :math:`e^{i \hat{H} \phi t}`, this function returns the matrix representation
    in the standard basis of its derivative with respect to :math:`t`, i.e.,

    .. math:: \frac{d \, e^{i \hat{H} phi t}}{dt} = i \phi \hat{H} e^{i \hat{H} phi t}.

    Args:
        operation (qml.Operation): The operation to be differentiated.

    Returns:
        np.ndarray: the derivative of the operation as a matrix in the standard basis

    Raises:
        ValueError: if the operation does not have a generator or is not composed of a single
        trainable parameter
    """
    generator, prefactor = operation.generator

    if generator is None:
        raise ValueError(f"Operation {operation.name} does not have a generator")
    if operation.num_params != 1:
        # Note, this case should already be caught by the previous raise since we haven't worked out
        # how to have an operator for multiple parameters. It is added here in case of a future
        # change
        raise ValueError(
            f"Operation {operation.name} is not written in terms of a single parameter"
        )

    if not isinstance(generator, np.ndarray):
        generator = generator.matrix

    if operation.inverse:
        prefactor *= -1
        generator = generator.conj().T

    return 1j * prefactor * generator @ operation.matrix
