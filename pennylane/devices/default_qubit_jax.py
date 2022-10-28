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
import numpy as np

import pennylane as qml
from pennylane.devices import DefaultQubit

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

    .. details::
        :title: Usage Details

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

    _asarray = staticmethod(jnp.array)
    _dot = staticmethod(jnp.dot)
    _abs = staticmethod(jnp.abs)
    _reduce_sum = staticmethod(lambda array, axes: jnp.sum(array, axis=tuple(axes)))
    _reshape = staticmethod(jnp.reshape)
    _flatten = staticmethod(lambda array: array.ravel())
    _einsum = staticmethod(jnp.einsum)
    _cast = staticmethod(jnp.array)
    _transpose = staticmethod(jnp.transpose)
    _tensordot = staticmethod(
        lambda a, b, axes: jnp.tensordot(
            a, b, axes if isinstance(axes, int) else list(map(tuple, axes))
        )
    )
    _conj = staticmethod(jnp.conj)
    _real = staticmethod(jnp.real)
    _imag = staticmethod(jnp.imag)
    _roll = staticmethod(jnp.roll)
    _stack = staticmethod(jnp.stack)
    _const_mul = staticmethod(jnp.multiply)
    _size = staticmethod(jnp.size)
    _ndim = staticmethod(jnp.ndim)

    def __init__(self, wires, *, shots=None, prng_key=None, analytic=None):
        if jax_config.read("jax_enable_x64"):
            c_dtype = jnp.complex128
            r_dtype = jnp.float64
        else:
            c_dtype = jnp.complex64
            r_dtype = jnp.float32
        super().__init__(wires, r_dtype=r_dtype, c_dtype=c_dtype, shots=shots, analytic=analytic)

        # prevent using special apply methods for these gates due to slowdown in jax
        # implementation
        del self._apply_ops["PauliY"]
        del self._apply_ops["Hadamard"]
        del self._apply_ops["CZ"]
        self._prng_key = prng_key

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(passthru_interface="jax")
        return capabilities

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        new_array = jnp.zeros(new_dimensions, dtype=array.dtype.type)
        new_array = new_array.at[indices].set(array)
        return new_array

    def sample_basis_states(self, number_of_states, state_probability, prng_key=None):
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

        if prng_key is not None:
            print('here, prng_key fed in: ', prng_key)
            key = prng_key
        elif self._prng_key is not None:
            key = self._prng_key
        else:
            # Assuming op-by-op, so we'll just make one.
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        if jnp.ndim(state_probability) == 2:
            # Produce separate keys for each of the probabilities along the broadcasted axis
            keys = []
            for _ in state_probability:
                key, subkey = jax.random.split(key)
                keys.append(subkey)
            return jnp.array(
                [
                    jax.random.choice(_key, number_of_states, shape=(shots,), p=prob)
                    for _key, prob in zip(keys, state_probability)
                ]
            )
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
        states_sampled_base_ten = samples[..., None] & powers_of_two
        return (states_sampled_base_ten > 0).astype(dtype)[..., ::-1]

    @staticmethod
    def _count_unbinned_samples(indices, batch_size, dim):
        """Count the occurences of sampled indices and convert them to relative
        counts in order to estimate their occurence probability."""

        shape = (dim + 1,) if batch_size is None else (batch_size, dim + 1)
        prob = qml.math.convert_like(jnp.zeros(shape, dtype=jnp.float64), indices)
        if batch_size is None:
            basis_states, counts = jnp.unique(indices, return_counts=True, size=dim, fill_value=-1)
            for state, count in zip(basis_states, counts):
                prob = prob.at[state].set(count / len(indices))
            # resize prob which discards the 'filled values'
            return prob[:-1]

        for i, idx in enumerate(indices):
            basis_states, counts = jnp.unique(idx, return_counts=True, size=dim, fill_value=-1)
            for state, count in zip(basis_states, counts):
                prob = prob.at[i, state].set(count / len(idx))

        # resize prob which discards the 'filled values'
        return prob[:, :-1]

    @staticmethod
    def _count_binned_samples(indices, batch_size, dim, bin_size, num_bins):
        """Count the occurences of bins of sampled indices and convert them to relative
        counts in order to estimate their occurence probability per bin."""

        # extend the probability vectors to store 'filled values'
        shape = (dim + 1, num_bins) if batch_size is None else (batch_size, dim + 1, num_bins)
        prob = qml.math.convert_like(jnp.zeros(shape, dtype=jnp.float64), indices)
        if batch_size is None:
            indices = indices.reshape((num_bins, bin_size))

            # count the basis state occurrences, and construct the probability vector for each bin
            for b, idx in enumerate(indices):
                idx = qml.math.convert_like(idx, indices)
                basis_states, counts = jnp.unique(idx, return_counts=True, size=dim, fill_value=-1)
                for state, count in zip(basis_states, counts):
                    prob = prob.at[state, b].set(count / bin_size)

            # resize prob which discards the 'filled values'
            return prob[:-1]

        indices = indices.reshape((batch_size, num_bins, bin_size))
        # count the basis state occurrences, and construct the probability vector
        # for each bin and broadcasting index
        for i, _indices in enumerate(indices):  # First iterate over broadcasting dimension
            for b, idx in enumerate(_indices):  # Then iterate over bins dimension
                idx = qml.math.convert_like(idx, indices)
                basis_states, counts = jnp.unique(idx, return_counts=True, size=dim, fill_value=-1)
                for state, count in zip(basis_states, counts):
                    prob = prob.at[i, state, b].set(count / bin_size)
        # resize prob which discards the 'filled values'
        return prob[:, :-1]

    def execute(self, circuit, **kwargs):
        """Execute a queue of quantum operations on the device and then
        measure the given observables.

        For plugin developers: instead of overwriting this, consider
        implementing a suitable subset of

        * :meth:`apply`

        * :meth:`~.generate_samples`

        * :meth:`~.probability`

        Additional keyword arguments may be passed to the this method
        that can be utilised by :meth:`apply`. An example would be passing
        the ``QNode`` hash that can be used later for parametric compilation.

        Args:
            circuit (~.tapes.QuantumTape): circuit to execute on the device

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            array[float]: measured value(s)
        """
        if qml.active_return():
            return self._execute_new(circuit, **kwargs)

        self.check_validity(circuit.operations, circuit.observables)

        # apply all circuit operations
        self.apply(circuit.operations, rotations=circuit.diagonalizing_gates, **kwargs)

        # generate computational basis samples
        if self.shots is not None or circuit.is_sampled:
            prng_key = kwargs.get("prng_key", None)
            self._samples = self.generate_samples(prng_key)

        ret_types = [m.return_type for m in circuit.measurements]
        counts_exist = any(
            ret in (qml.measurements.Counts, qml.measurements.AllCounts) for ret in ret_types
        )

        # compute the required statistics
        if not self.analytic and self._shot_vector is not None:
            results = self._collect_shotvector_results(circuit, counts_exist)
        else:
            results = self.statistics(circuit.observables, circuit=circuit)

        if not circuit.is_sampled:

            if len(circuit.measurements) == 1:
                if ret_types[0] is qml.measurements.State:
                    # State: assumed to only be allowed if it's the only measurement
                    results = self._asarray(results, dtype=self.C_DTYPE)
                else:
                    # Measurements with expval, var or probs
                    try:
                        # Feature for returning custom objects: if the type cannot be cast to float then we can still allow it as an output
                        results = self._asarray(results, dtype=self.R_DTYPE)
                    except TypeError:
                        pass

            elif all(
                ret in (qml.measurements.Expectation, qml.measurements.Variance)
                for ret in ret_types
            ):
                # Measurements with expval or var
                results = self._asarray(results, dtype=self.R_DTYPE)
            elif not counts_exist:
                # all the other cases except any counts
                results = self._asarray(results)

        elif circuit.all_sampled and not self._has_partitioned_shots() and not counts_exist:
            results = self._asarray(results)
        else:
            results = tuple(
                qml.math.squeeze(self._asarray(r)) if not isinstance(r, dict) else r
                for r in results
            )

        # increment counter for number of executions of qubit device
        self._num_executions += 1

        if self.tracker.active:
            self.tracker.update(executions=1, shots=self._shots)
            self.tracker.record()

        return results

    def generate_samples(self, prng_key=None):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant bit.

        .. warning::

            This method should be overwritten on devices that
            generate their own computational basis samples, with the resulting
            computational basis samples stored as ``self._samples``.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        number_of_states = 2**self.num_wires

        rotated_prob = self.analytic_probability()

        samples = self.sample_basis_states(number_of_states, rotated_prob, prng_key)
        return self.states_to_binary(samples, self.num_wires)

    def batch_execute(self, circuits, **kwargs):
        """Execute a batch of quantum circuits on the device.

        The circuits are represented by tapes, and they are executed one-by-one using the
        device's ``execute`` method. The results are collected in a list.

        For plugin developers: This function should be overwritten if the device can efficiently run multiple
        circuits on a backend, for example using parallel and/or asynchronous executions.

        Args:
            circuits (list[.tapes.QuantumTape]): circuits to execute on the device

        Returns:
            list[array[float]]: list of measured value(s)
        """
        # TODO: This method and the tests can be globally implemented by Device
        # once it has the same signature in the execute() method
        if qml.active_return():
            return self._batch_execute_new(circuits=circuits)

        results = []
        for circuit in circuits:
            # we need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()

            # TODO: Insert control on value here
            res = self.execute(circuit, **kwargs)
            results.append(res)

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

        return results
