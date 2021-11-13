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
"""This module contains an jax implementation of the :class:`~.DefaultQubit`
reference plugin.
"""
import pennylane as qml
from pennylane.operation import DiagonalOperation, Channel
from pennylane.devices import DefaultQubit, DefaultMixed

import numpy as np

try:
    import jax.numpy as jnp
    import jax
    from jax.config import config as jax_config

except ImportError as e:  # pragma: no cover
    raise ImportError("default.mixed.jax device requires installing jax>0.2.0") from e


class DefaultMixedJax(DefaultMixed):
    """Simulator plugin based on ``"default.mixed"``, written using jax.

    **Short name:** ``default.mixed.jax``

    This device provides a mixed-state qubit simulator written using jax. As a result, it
    supports classical backpropagation as a means to compute the gradient. This can be faster than
    the parameter-shift rule for analytic quantum gradients when the number of parameters to be
    optimized is large.

    To use this device, you will need to install jax:

    .. code-block:: console

        pip install jax jaxlib

    **Example**

    The ``default.mixed.jax`` device is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the JAX interface. This is the default method of
    differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your JAX computation:

    >>> dev = qml.device("default.mixed.jax", wires=1)
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

    * You must use the ``"jax"`` interface for classical backpropagation, as JAX is
      used as the device backend.

    .. UsageDetails::

        JAX does randomness in a special way when compared to NumPy, in that all randomness needs to
        be seeded. While we handle this for you automatically in op-by-op mode, when using ``jax.jit``,
        the automatically generated seed gets constantant compiled.

        Example:

        .. code-block:: python

            dev = qml.device("default.mixed.jax", wires=1, shots=10)

            @jax.jit
            @qml.qnode(dev, interface="jax", diff_method="backprop")
            def circuit():
                qml.Hadamard(0)
                return qml.sample(qml.PauliZ(wires=0))

            a = circuit()
            b = circuit() # Bad! b will be the exact same samples as a.


        To fix this, you should wrap your qnode in another function that takes a PRNGKey, and pass
        that in during your device construction.

        .. code-block:: python

            @jax.jit
            def keyed_circuit(key):
                dev = qml.device("default.mixed.jax", prng_key=key, wires=1, shots=10)
                @qml.qnode(dev, interface="jax", diff_method="backprop")
                def circuit():
                    qml.Hadamard(0)
                    return qml.sample(qml.PauliZ(wires=0))
                return circuit()

            key1 = jax.random.PRNGKey(0)
            key2 = jax.random.PRNGKey(1)
            a = keyed_circuit(key1)
            b = keyed_circuit(key2) # b will be different samples now.

        Check out out the `JAX random documentation <https://jax.readthedocs.io/en/latest/jax.random.html>`__
        for more information.

    Args:
        wires (int): The number of wires to initialize the device with.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
        analytic (bool): Indicates if the device should calculate expectations
            and variances analytically. In non-analytic mode, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.mixed`` and using ``diff_method="parameter-shift"``.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is the key to the
            pseudo random number generator. If None, a random key will be generated.

    """

    name = "Default mixed (jax) PennyLane plugin"
    short_name = "default.mixed.jax"

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
    _tensordot = staticmethod(
        lambda a, b, axes: jnp.tensordot(
            a, b, axes if isinstance(axes, int) else list(map(tuple, axes))
        )
    )
    _conj = staticmethod(jnp.conj)
    _imag = staticmethod(jnp.imag)
    _roll = staticmethod(jnp.roll)
    _stack = staticmethod(jnp.stack)
    _expand_dims = staticmethod(jnp.expand_dims)

    def __init__(self, wires, *, shots=None, prng_key=None, analytic=None):
        if jax_config.read("jax_enable_x64"):
            self.C_DTYPE = jnp.complex128
            self.R_DTYPE = jnp.float64
        else:
            self.C_DTYPE = jnp.complex64
            self.R_DTYPE = jnp.float32
        super().__init__(wires, shots=shots, cache=0, analytic=analytic)

        self._prng_key = prng_key

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            passthru_interface="jax",
            supports_reversible_diff=False,
        )
        return capabilities

    def _get_kraus(self, operation):
        """Return the Kraus operators representing the operation.

        Args:
           operation (.Operation): a PennyLane operation

        Returns:
           list[array[complex]]: Returns a list of 2D matrices representing the Kraus operators. If
           the operation is unitary, returns a single Kraus operator. In the case of a diagonal
           unitary, returns a 1D array representing the matrix diagonal.
        """
        op_name = operation.name.split(".inv")[0]

        if isinstance(operation, Channel):
            return self._asarray(operation.kraus_matrices)
        if isinstance(operation, DiagonalOperation):
            if operation.inverse:
                return self._conj(operation.eigvals)
            else:
                return self._asarray(operation.eigvals)
        return self._expand_dims(operation.matrix, 0)