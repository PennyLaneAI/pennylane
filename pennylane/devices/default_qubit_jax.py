# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains an jax implementation of the :class:`~.DefaultQubit`
reference plugin.
"""


from pennylane.operation import DiagonalOperation

from pennylane.devices import DefaultQubit
from pennylane.devices import jax_ops

try:
    import jax.numpy as jnp

except ImportError as e:
    raise ImportError("default.qubit.jax device requires installing jax>0.2.0") from e


class DefaultQubitJax(DefaultQubit):
    """Simulator plugin based on ``"default.qubit"``, written using jax.

    **Short name:** ``default.qubit.jax``

    This device provides a pure-state qubit simulator written using jax. As a result, it
    supports classical backpropagation as a means to compute the gradient. This can be faster than
    the parameter-shift rule for analytic quantum gradients when the number of parameters to be
    optimized is large.

    To use this device, you will need to install jax:

    .. code-block:: console

        pip install jax jaxlib

    **Example**

    The ``default.qubit.jax`` is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the jax interface. This is the default method of
    differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your jax computation:

    >>> dev = qml.device("default.qubit.jax", wires=1)
    >>> @qml.qnode(dev, interface="jax", diff_method="backprop")
    ... def circuit(x):
    ...     qml.RX(x[1], wires=0)
    ...     qml.Rot(x[0], x[1], x[2], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> weights = jnp.array([0.2, 0.5, 0.1])
    >>> grad_fn = jax.grad(circuit)
    >>> print(grad_fn(weights))
    array([-2.2526717e-01 -1.0086454e+00  1.3877788e-17])

    There are a couple of things to keep in mind when using the ``"backprop"``
    differentiation method for QNodes:

    * You must use the ``"jax"`` interface for classical backpropagation, as jax is
      used as the device backend.

    * Only exact expectation values, variances, and probabilities are differentiable.
      When instantiating the device with ``analytic=False``, differentiating QNode
      outputs will result in an error.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to 1000 if not specified.
            If ``analytic == True``, then the number of shots is ignored
            in the calculation of expectation values and variances, and only controls the number
            of samples returned by ``sample``.
        analytic (bool): Indicates if the device should calculate expectations
            and variances analytically. In non-analytic mode, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
    """

    name = "Default qubit (jax) PennyLane plugin"
    short_name = "default.qubit.jax"

    parametric_ops = {
        "PhaseShift": jax_ops.PhaseShift,
        "RX": jax_ops.RX,
        "RY": jax_ops.RY,
        "RZ": jax_ops.RZ,
        "Rot": jax_ops.Rot,
        "CRX": jax_ops.CRX,
        "CRY": jax_ops.CRY,
        "CRZ": jax_ops.CRZ,
        "MultiRZ": jax_ops.MultiRZ,
    }

    C_DTYPE = jnp.complex128
    R_DTYPE = jnp.float64
    _asarray = staticmethod(jnp.array)
    _dot = staticmethod(jnp.dot)
    _abs = staticmethod(jnp.abs)
    _reduce_sum = staticmethod(lambda array, axes: jnp.sum(array, axis=tuple(axes)))
    _reshape = staticmethod(jnp.reshape)
    _flatten = staticmethod(lambda array: array.ravel())
    _gather = staticmethod(lambda array, indices: array[indices])
    _einsum = staticmethod(jnp.einsum)
    _cast = staticmethod(jnp.array)
    _transpose = staticmethod(jnp.transpose)
    _tensordot = staticmethod(jnp.tensordot)
    _conj = staticmethod(jnp.conj)
    _imag = staticmethod(jnp.imag)
    _roll = staticmethod(jnp.roll)
    _stack = staticmethod(jnp.stack)

    def __init__(self, wires, *, shots=1000, analytic=True):
        super().__init__(wires, shots=shots, analytic=analytic, cache=0)

        # prevent using special apply methods for these gates due to slowdown in jax
        # implementation
        del self._apply_ops["PauliY"]
        del self._apply_ops["Hadamard"]
        del self._apply_ops["CZ"]

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            passthru_interface="jax",
            supports_reversible_diff=False,
        )
        return capabilities

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        new_array = jnp.zeros(new_dimensions, dtype=array.dtype.type)
        new_array = new_array.at[indices].set(array)
        return new_array

    def _get_unitary_matrix(self, unitary):
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis, or, in the case of a diagonal unitary,
            a 1D array representing the matrix diagonal.
        """
        op_name = unitary.name
        if op_name in self.parametric_ops:
            if op_name == "MultiRZ":
                return self.parametric_ops[unitary.name](*unitary.parameters, len(unitary.wires))
            return self.parametric_ops[unitary.name](*unitary.parameters)

        if isinstance(unitary, DiagonalOperation):
            return unitary.eigvals

        return unitary.matrix
