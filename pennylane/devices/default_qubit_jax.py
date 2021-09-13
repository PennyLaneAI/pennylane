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
from pennylane.operation import DiagonalOperation
from pennylane.devices import DefaultQubit
from pennylane.devices import jax_ops

import numpy as np

try:
    import jax.numpy as jnp
    import jax
    from jax.config import config as jax_config

except ImportError as e:  # pragma: no cover
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

    The ``default.qubit.jax`` device is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the JAX interface. This is the default method of
    differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your JAX computation:

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

    * You must use the ``"jax"`` interface for classical backpropagation, as JAX is
      used as the device backend.

    .. UsageDetails::

        JAX does randomness in a special way when compared to NumPy, in that all randomness needs to
        be seeded. While we handle this for you automatically in op-by-op mode, when using ``jax.jit``,
        the automatically generated seed gets constantant compiled.

        Example:

        .. code-block:: python

            dev = qml.device("default.qubit.jax", wires=1, shots=10)

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
                dev = qml.device("default.qubit.jax", prng_key=key, wires=1, shots=10)
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
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
        prng_key (Optional[jax.random.PRNGKey]): An optional ``jax.random.PRNGKey``. This is the key to the
            pseudo random number generator. If None, a random key will be generated.

    """

    name = "Default qubit (jax) PennyLane plugin"
    short_name = "default.qubit.jax"

    parametric_ops = {
        "PhaseShift": jax_ops.PhaseShift,
        "ControlledPhaseShift": jax_ops.ControlledPhaseShift,
        "CPhase": jax_ops.ControlledPhaseShift,
        "RX": jax_ops.RX,
        "RY": jax_ops.RY,
        "RZ": jax_ops.RZ,
        "Rot": jax_ops.Rot,
        "CRX": jax_ops.CRX,
        "CRY": jax_ops.CRY,
        "CRZ": jax_ops.CRZ,
        "CRot": jax_ops.CRot,
        "MultiRZ": jax_ops.MultiRZ,
        "IsingXX": jax_ops.IsingXX,
        "IsingYY": jax_ops.IsingYY,
        "IsingZZ": jax_ops.IsingZZ,
        "SingleExcitation": jax_ops.SingleExcitation,
        "SingleExcitationPlus": jax_ops.SingleExcitationPlus,
        "SingleExcitationMinus": jax_ops.SingleExcitationMinus,
        "DoubleExcitation": jax_ops.DoubleExcitation,
        "DoubleExcitationPlus": jax_ops.DoubleExcitationPlus,
        "DoubleExcitationMinus": jax_ops.DoubleExcitationMinus,
    }

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

    def __init__(self, wires, *, shots=None, prng_key=None, analytic=None):
        if jax_config.read("jax_enable_x64"):
            self.C_DTYPE = jnp.complex128
            self.R_DTYPE = jnp.float64
        else:
            self.C_DTYPE = jnp.complex64
            self.R_DTYPE = jnp.float32
        super().__init__(wires, shots=shots, cache=0, analytic=analytic)

        # prevent using special apply methods for these gates due to slowdown in jax
        # implementation
        del self._apply_ops["PauliY"]
        del self._apply_ops["Hadamard"]
        del self._apply_ops["CZ"]
        self._prng_key = prng_key

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
        op_name = unitary.name.split(".inv")[0]

        if op_name in self.parametric_ops:
            if op_name == "MultiRZ":
                mat = self.parametric_ops[op_name](*unitary.parameters, len(unitary.wires))
            else:
                mat = self.parametric_ops[op_name](*unitary.parameters)

            if unitary.inverse:
                mat = self._transpose(self._conj(mat))

            return mat

        if isinstance(unitary, DiagonalOperation):
            return unitary.eigvals

        return unitary.matrix

    def sample_basis_states(self, number_of_states, state_probability):
        """Sample from the computational basis states based on the state
        probability.

        This is an auxiliary method to the generate_samples method.

        Args:
            number_of_states (int): the number of basis states to sample from

        Returns:
            List[int]: the sampled basis states
        """
        if self.shots is None:

            raise qml.QuantumFunctionError(
                "The number of shots has to be explicitly set on the device "
                "when using sample-based measurements."
            )

        shots = self.shots

        if self._prng_key is None:
            # Assuming op-by-op, so we'll just make one.
            key = jax.random.PRNGKey(np.random.randint(0, 2 ** 31))
        else:
            key = self._prng_key
        return jax.random.choice(key, number_of_states, shape=(shots,), p=state_probability)

    @staticmethod
    def states_to_binary(samples, num_wires, dtype=jnp.int32):
        """Convert basis states from base 10 to binary representation.

        This is an auxiliary method to the generate_samples method.

        Args:
            samples (List[int]): samples of basis states in base 10 representation
            num_wires (int): the number of qubits
            dtype (type): Type of the internal integer array to be used. Can be
                important to specify for large systems for memory allocation
                purposes.

        Returns:
            List[int]: basis states in binary representation
        """
        powers_of_two = 1 << jnp.arange(num_wires, dtype=dtype)
        states_sampled_base_ten = samples[:, None] & powers_of_two
        return (states_sampled_base_ten > 0).astype(dtype)[:, ::-1]
