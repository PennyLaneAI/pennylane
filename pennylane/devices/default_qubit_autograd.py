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
"""This module contains an autograd implementation of the :class:`~.DefaultQubitLegacy`
reference plugin.
"""
from pennylane import numpy as pnp

from pennylane.devices import DefaultQubitLegacy


class DefaultQubitAutograd(DefaultQubitLegacy):
    """Simulator plugin based on ``"default.qubit.legacy"``, written using Autograd.

    **Short name:** ``default.qubit.autograd``

    This device provides a pure-state qubit simulator written using Autograd. As a result, it
    supports classical backpropagation as a means to compute the gradient. This can be faster than
    the parameter-shift rule for analytic quantum gradients when the number of parameters to be
    optimized is large.

    To use this device, you will need to install Autograd:

    .. code-block:: console

        pip install autograd

    **Example**

    The ``default.qubit.autograd`` is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the Autograd interface. This is the default method of
    differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your Autograd computation:

    >>> dev = qml.device("default.qubit.autograd", wires=1)
    >>> @qml.qnode(dev, interface="autograd", diff_method="backprop")
    ... def circuit(x):
    ...     qml.RX(x[1], wires=0)
    ...     qml.Rot(x[0], x[1], x[2], wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> weights = np.array([0.2, 0.5, 0.1], requires_grad=True)
    >>> grad_fn = qml.grad(circuit)
    >>> print(grad_fn(weights))
    array([-2.2526717e-01 -1.0086454e+00  1.3877788e-17])

    There are a couple of things to keep in mind when using the ``"backprop"``
    differentiation method for QNodes:

    * You must use the ``"autograd"`` interface for classical backpropagation, as Autograd is
      used as the device backend.

    * Only exact expectation values, variances, and probabilities are differentiable.
      When instantiating the device with ``analytic=False``, differentiating QNode
      outputs will result in an error.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
        analytic (bool): Indicates if the device should calculate expectations
            and variances analytically. In non-analytic mode, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
    """

    name = "Default qubit (Autograd) PennyLane plugin"
    short_name = "default.qubit.autograd"

    _dot = staticmethod(pnp.dot)
    _abs = staticmethod(pnp.abs)
    _reduce_sum = staticmethod(lambda array, axes: pnp.sum(array, axis=tuple(axes)))
    _reshape = staticmethod(pnp.reshape)
    _flatten = staticmethod(lambda array: array.flatten())
    _einsum = staticmethod(pnp.einsum)
    _cast = staticmethod(pnp.asarray)
    _transpose = staticmethod(pnp.transpose)
    _tensordot = staticmethod(pnp.tensordot)
    _conj = staticmethod(pnp.conj)
    _real = staticmethod(pnp.real)
    _imag = staticmethod(pnp.imag)
    _roll = staticmethod(pnp.roll)
    _stack = staticmethod(pnp.stack)
    _size = staticmethod(pnp.size)
    _ndim = staticmethod(pnp.ndim)

    @staticmethod
    def _asarray(array, dtype=None):
        return pnp.asarray(array, dtype=dtype)

    @staticmethod
    def _const_mul(constant, array):
        return constant * array

    def __init__(self, wires, *, shots=None, analytic=None):
        r_dtype = pnp.float64
        c_dtype = pnp.complex128
        super().__init__(wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)

        # prevent using special apply methods for these gates due to slowdown in Autograd
        # implementation
        del self._apply_ops["PauliY"]
        del self._apply_ops["Hadamard"]
        del self._apply_ops["CZ"]

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(passthru_interface="autograd")
        return capabilities

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        new_array = pnp.zeros(new_dimensions, dtype=array.dtype.type)
        new_array[indices] = array
        return new_array
