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
"""
This module contains the :class:`QubitDevice` abstract base class.
"""

# For now, arguments may be different from the signatures provided in Device
# e.g. instead of expval(self, observable, wires, par) have expval(self, observable)
# pylint: disable=arguments-differ, abstract-method, no-value-for-parameter,too-many-instance-attributes
import abc
from collections import OrderedDict
import itertools

import numpy as np

import pennylane as qml
from pennylane.operation import (
    Sample,
    Variance,
    Expectation,
    Probability,
    State,
    operation_derivative,
)
from pennylane.qnodes import QuantumFunctionError
from pennylane import Device
from pennylane.math import sum as qmlsum
from pennylane.wires import Wires


class QubitDevice(Device):
    """Abstract base class for PennyLane qubit devices.

    The following abstract method **must** be defined:

    * :meth:`~.apply`: append circuit operations, compile the circuit (if applicable),
      and perform the quantum computation.

    Devices that generate their own samples (such as hardware) may optionally
    overwrite :meth:`~.probabilty`. This method otherwise automatically
    computes the probabilities from the generated samples, and **must**
    overwrite the following method:

    * :meth:`~.generate_samples`: Generate samples from the device from the
      exact or approximate probability distribution.

    Analytic devices **must** overwrite the following method:

    * :meth:`~.analytic_probability`: returns the probability or marginal probability from the
      device after circuit execution. :meth:`~.marginal_prob` may be used here.

    This device contains common utility methods for qubit-based devices. These
    do not need to be overwritten. Utility methods include:

    * :meth:`~.expval`, :meth:`~.var`, :meth:`~.sample`: return expectation values,
      variances, and samples of observables after the circuit has been rotated
      into the observable eigenbasis.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (int): number of circuit evaluations/random samples used to estimate
            expectation values of observables
        analytic (bool): If ``True``, the device calculates probability, expectation values,
            and variances analytically. If ``False``, a finite number of samples set by
            the argument ``shots`` are used to estimate these quantities.
        cache (int): Number of device executions to store in a cache to speed up subsequent
            executions. A value of ``0`` indicates that no caching will take place. Once filled,
            older elements of the cache are removed and replaced with the most recent device
            executions to keep the cache up to date.
    """

    # pylint: disable=too-many-public-methods
    C_DTYPE = np.complex128
    R_DTYPE = np.float64
    _asarray = staticmethod(np.asarray)
    _dot = staticmethod(np.dot)
    _abs = staticmethod(np.abs)
    _reduce_sum = staticmethod(lambda array, axes: np.sum(array, axis=tuple(axes)))
    _reshape = staticmethod(np.reshape)
    _flatten = staticmethod(lambda array: array.flatten())
    _gather = staticmethod(lambda array, indices: array[indices])
    _einsum = staticmethod(np.einsum)
    _cast = staticmethod(np.asarray)
    _transpose = staticmethod(np.transpose)
    _tensordot = staticmethod(np.tensordot)
    _conj = staticmethod(np.conj)
    _imag = staticmethod(np.imag)
    _roll = staticmethod(np.roll)
    _stack = staticmethod(np.stack)
    _outer = staticmethod(np.outer)
    _diag = staticmethod(np.diag)
    _real = staticmethod(np.real)

    @staticmethod
    def _scatter(indices, array, new_dimensions):
        new_array = np.zeros(new_dimensions, dtype=array.dtype.type)
        new_array[indices] = array
        return new_array

    observables = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian", "Identity"}

    def __init__(self, wires=1, shots=1000, analytic=True, cache=0):
        super().__init__(wires=wires, shots=shots)

        self.analytic = analytic
        """bool: If ``True``, the device supports exact calculation of expectation
        values, variances, and probabilities. If ``False``, samples are used
        to estimate the statistical quantities above."""

        self._samples = None
        """None or array[int]: stores the samples generated by the device
        *after* rotation to diagonalize the observables."""

        self._circuit_hash = None
        """None or int: stores the hash of the circuit from the last execution which
        can be used by devices in :meth:`apply` for parametric compilation."""

        self._cache = cache
        """int: Number of device executions to store in a cache to speed up subsequent
        executions. If set to zero, no caching occurs."""

        self._cache_execute = OrderedDict()
        """OrderedDict[int: Any]: Mapping from hashes of the circuit to results of executing the
        device."""

    @classmethod
    def capabilities(cls):

        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_finite_shots=True,
            supports_tensor_observables=True,
            returns_probs=True,
        )
        return capabilities

    def reset(self):
        """Reset the backend state.

        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        self._samples = None
        self._circuit_hash = None

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
            circuit (~.CircuitGraph): circuit to execute on the device

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            array[float]: measured value(s)
        """
        # TODO: Remove try/except when circuit is always QuantumTape and
        # consider merging with caching case
        try:
            self._circuit_hash = circuit.graph.hash
        except AttributeError as e:
            self._circuit_hash = circuit.hash

        if self._cache:
            try:  # TODO: Remove try/except when circuit is always QuantumTape
                circuit_hash = circuit.graph.hash
            except AttributeError as e:
                raise ValueError("Caching is only available when using tape mode") from e
            if circuit_hash in self._cache_execute:
                return self._cache_execute[circuit_hash]

        self.check_validity(circuit.operations, circuit.observables)

        # apply all circuit operations
        self.apply(circuit.operations, rotations=circuit.diagonalizing_gates, **kwargs)

        # generate computational basis samples
        if (not self.analytic) or circuit.is_sampled:
            self._samples = self.generate_samples()

        # compute the required statistics
        results = self.statistics(circuit.observables)

        if circuit.all_sampled or not circuit.is_sampled:
            results = self._asarray(results)
        else:
            results = tuple(self._asarray(r) for r in results)

        if self._cache and circuit_hash not in self._cache_execute:
            self._cache_execute[circuit_hash] = results
            if len(self._cache_execute) > self._cache:
                self._cache_execute.popitem(last=False)

        # increment counter for number of executions of qubit device
        self._num_executions += 1

        return results

    @property
    def cache(self):
        """int: Number of device executions to store in a cache to speed up subsequent
        executions. If set to zero, no caching occurs."""
        return self._cache

    def batch_execute(self, circuits):
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

        results = []
        for circuit in circuits:
            # we need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()

            res = self.execute(circuit)
            results.append(res)

        return results

    @abc.abstractmethod
    def apply(self, operations, **kwargs):
        """Apply quantum operations, rotate the circuit into the measurement
        basis, and compile and execute the quantum circuit.

        This method receives a list of quantum operations queued by the QNode,
        and should be responsible for:

        * Constructing the quantum program
        * (Optional) Rotating the quantum circuit using the rotation
          operations provided. This diagonalizes the circuit so that arbitrary
          observables can be measured in the computational basis.
        * Compile the circuit
        * Execute the quantum circuit

        Both arguments are provided as lists of PennyLane :class:`~.Operation`
        instances. Useful properties include :attr:`~.Operation.name`,
        :attr:`~.Operation.wires`, and :attr:`~.Operation.parameters`,
        and :attr:`~.Operation.inverse`:

        >>> op = qml.RX(0.2, wires=[0])
        >>> op.name # returns the operation name
        "RX"
        >>> op.wires # returns a Wires object representing the wires that the operation acts on
        <Wires = [0]>
        >>> op.parameters # returns a list of parameters
        [0.2]
        >>> op.inverse # check if the operation should be inverted
        False
        >>> op = qml.RX(0.2, wires=[0]).inv
        >>> op.inverse
        True

        Args:
            operations (list[~.Operation]): operations to apply to the device

        Keyword args:
            rotations (list[~.Operation]): operations that rotate the circuit
                pre-measurement into the eigenbasis of the observables.
            hash (int): the hash value of the circuit constructed by `CircuitGraph.hash`
        """

    @staticmethod
    def active_wires(operators):
        """Returns the wires acted on by a set of operators.

        Args:
            operators (list[~.Operation]): operators for which
                we are gathering the active wires

        Returns:
            Wires: wires activated by the specified operators
        """
        list_of_wires = [op.wires for op in operators]

        return Wires.all_wires(list_of_wires)

    def statistics(self, observables):
        """Process measurement results from circuit execution and return statistics.

        This includes returning expectation values, variance, samples, probabilities, states, and
        density matrices.

        Args:
            observables (List[.Observable]): the observables to be measured

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            Union[float, List[float]]: the corresponding statistics
        """
        results = []

        for obs in observables:
            # Pass instances directly
            if obs.return_type is Expectation:
                results.append(self.expval(obs))

            elif obs.return_type is Variance:
                results.append(self.var(obs))

            elif obs.return_type is Sample:
                results.append(self.sample(obs))

            elif obs.return_type is Probability:
                results.append(self.probability(wires=obs.wires))

            elif obs.return_type is State:
                if len(observables) > 1:
                    raise QuantumFunctionError(
                        "The state or density matrix cannot be returned in combination"
                        " with other return types"
                    )
                if self.wires.labels != tuple(range(self.num_wires)):
                    raise QuantumFunctionError(
                        "Returning the state is not supported when using custom wire labels"
                    )
                # Check if the state is accessible and decide to return the state or the density
                # matrix.
                results.append(self.access_state(wires=obs.wires))

            elif obs.return_type is not None:
                raise QuantumFunctionError(
                    "Unsupported return type specified for observable {}".format(obs.name)
                )

        return results

    def access_state(self, wires=None):
        """Check that the device has access to an internal state and return it if available.

        Args:
            wires (Wires): wires of the reduced system

        Raises:
            QuantumFunctionError: if the device is not capable of returning the state

        Returns:
            array or tensor: the state or the density matrix of the device
        """
        if not self.capabilities().get("returns_state"):
            raise QuantumFunctionError("The current device is not capable of returning the state")

        state = getattr(self, "state", None)

        if state is None:
            raise QuantumFunctionError("The state is not available in the current device")

        if wires:
            density_matrix = self.density_matrix(wires)
            return density_matrix

        return state

    def generate_samples(self):
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
        number_of_states = 2 ** self.num_wires

        rotated_prob = self.analytic_probability()

        samples = self.sample_basis_states(number_of_states, rotated_prob)
        return QubitDevice.states_to_binary(samples, self.num_wires)

    def sample_basis_states(self, number_of_states, state_probability):
        """Sample from the computational basis states based on the state
        probability.

        This is an auxiliary method to the generate_samples method.

        Args:
            number_of_states (int): the number of basis states to sample from

        Returns:
            List[int]: the sampled basis states
        """
        basis_states = np.arange(number_of_states)
        return np.random.choice(basis_states, self.shots, p=state_probability)

    @staticmethod
    def generate_basis_states(num_wires, dtype=np.uint32):
        """
        Generates basis states in binary representation according to the number
        of wires specified.

        The states_to_binary method creates basis states faster (for larger
        systems at times over x25 times faster) than the approach using
        ``itertools.product``, at the expense of using slightly more memory.

        Due to the large size of the integer arrays for more than 32 bits,
        memory allocation errors may arise in the states_to_binary method.
        Hence we constraint the dtype of the array to represent unsigned
        integers on 32 bits. Due to this constraint, an overflow occurs for 32
        or more wires, therefore this approach is used only for fewer wires.

        For smaller number of wires speed is comparable to the next approach
        (using ``itertools.product``), hence we resort to that one for testing
        purposes.

        Args:
            num_wires (int): the number wires
            dtype=np.uint32 (type): the data type of the arrays to use

        Returns:
            np.ndarray: the sampled basis states
        """
        if 2 < num_wires < 32:
            states_base_ten = np.arange(2 ** num_wires, dtype=dtype)
            return QubitDevice.states_to_binary(states_base_ten, num_wires, dtype=dtype)

        # A slower, but less memory intensive method
        basis_states_generator = itertools.product((0, 1), repeat=num_wires)
        return np.fromiter(itertools.chain(*basis_states_generator), dtype=int).reshape(
            -1, num_wires
        )

    @staticmethod
    def states_to_binary(samples, num_wires, dtype=np.int64):
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
        powers_of_two = 1 << np.arange(num_wires, dtype=dtype)
        states_sampled_base_ten = samples[:, None] & powers_of_two
        return (states_sampled_base_ten > 0).astype(dtype)[:, ::-1]

    @property
    def circuit_hash(self):
        """The hash of the circuit upon the last execution.

        This can be used by devices in :meth:`~.apply` for parametric compilation.
        """
        return self._circuit_hash

    @property
    def state(self):
        """Returns the state vector of the circuit prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        raise NotImplementedError

    def density_matrix(self, wires):
        """Returns the reduced density matrix prior to measurement.

        .. note::

            Only state vector simulators support this property. Please see the
            plugin documentation for more details.
        """
        raise NotImplementedError

    def analytic_probability(self, wires=None):
        r"""Return the (marginal) probability of each computational basis
        state from the last run of the device.

        PennyLane uses the convention
        :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where :math:`q_0` is the most
        significant bit.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            :meth:`marginal_prob` may be used as a utility method
            to calculate the marginal probability distribution.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            List[float]: list of the probabilities
        """
        raise NotImplementedError

    def estimate_probability(self, wires=None):
        """Return the estimated probability of each computational basis state
        using the generated samples.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to calculate
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            List[float]: list of the probabilities
        """

        wires = wires or self.wires
        # convert to a wires object
        wires = Wires(wires)
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        samples = self._samples[:, device_wires]

        # convert samples from a list of 0, 1 integers, to base 10 representation
        unraveled_indices = [2] * len(device_wires)
        indices = np.ravel_multi_index(samples.T, unraveled_indices)

        # count the basis state occurrences, and construct the probability vector
        basis_states, counts = np.unique(indices, return_counts=True)
        prob = np.zeros([2 ** len(device_wires)], dtype=np.float64)
        prob[basis_states] = counts / len(samples)
        return self._asarray(prob, dtype=self.R_DTYPE)

    def probability(self, wires=None):
        """Return either the analytic probability or estimated probability of
        each computational basis state.

        If no :attr:`~.analytic` attributes exists for the device, then return the
        estimated probability.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            List[float]: list of the probabilities
        """

        if hasattr(self, "analytic") and self.analytic:
            return self.analytic_probability(wires=wires)

        return self.estimate_probability(wires=wires)

    def marginal_prob(self, prob, wires=None):
        r"""Return the marginal probability of the computational basis
        states by summing the probabiliites on the non-specified wires.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            If the provided wires are not in the order as they appear on the device,
            the returned marginal probabilities take this permutation into account.

            For example, if the addressable wires on this device are ``Wires([0, 1, 2])`` and
            this function gets passed ``wires=[2, 0]``, then the returned marginal
            probability vector will take this 'reversal' of the two wires
            into account:

            .. math::

                \mathbb{P}^{(2, 0)}
                            = \left[
                               |00\rangle, |10\rangle, |01\rangle, |11\rangle
                              \right]

        Args:
            prob: The probabilities to return the marginal probabilities
                for
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            array[float]: array of the resulting marginal probabilities.
        """

        if wires is None:
            # no need to marginalize
            return prob

        wires = Wires(wires)
        # determine which subsystems are to be summed over
        inactive_wires = Wires.unique_wires([self.wires, wires])

        # translate to wire labels used by device
        device_wires = self.map_wires(wires)
        inactive_device_wires = self.map_wires(inactive_wires)

        # reshape the probability so that each axis corresponds to a wire
        prob = self._reshape(prob, [2] * self.num_wires)

        # sum over all inactive wires
        # hotfix to catch when default.qubit uses this method
        # since then device_wires is a list
        if isinstance(inactive_device_wires, Wires):
            prob = self._flatten(self._reduce_sum(prob, inactive_device_wires.labels))
        else:
            prob = self._flatten(self._reduce_sum(prob, inactive_device_wires))

        # The wires provided might not be in consecutive order (i.e., wires might be [2, 0]).
        # If this is the case, we must permute the marginalized probability so that
        # it corresponds to the orders of the wires passed.
        num_wires = len(device_wires)
        basis_states = self.generate_basis_states(num_wires)
        perm = np.ravel_multi_index(
            basis_states[:, np.argsort(np.argsort(device_wires))].T, [2] * len(device_wires)
        )
        return self._gather(prob, perm)

    def expval(self, observable):

        if self.analytic:
            # exact expectation value
            eigvals = self._asarray(observable.eigvals, dtype=self.R_DTYPE)
            prob = self.probability(wires=observable.wires)
            return self._dot(eigvals, prob)

        # estimate the ev
        return np.mean(self.sample(observable))

    def var(self, observable):

        if self.analytic:
            # exact variance value
            eigvals = self._asarray(observable.eigvals, dtype=self.R_DTYPE)
            prob = self.probability(wires=observable.wires)
            return self._dot((eigvals ** 2), prob) - self._dot(eigvals, prob) ** 2

        # estimate the variance
        return np.var(self.sample(observable))

    def sample(self, observable):

        # translate to wire labels used by device
        device_wires = self.map_wires(observable.wires)
        name = observable.name

        if isinstance(name, str) and name in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            # Process samples for observables with eigenvalues {1, -1}
            return 1 - 2 * self._samples[:, device_wires[0]]

        # Replace the basis state in the computational basis with the correct eigenvalue.
        # Extract only the columns of the basis samples required based on ``wires``.
        samples = self._samples[:, np.array(device_wires)]  # Add np.array here for Jax support.
        unraveled_indices = [2] * len(device_wires)
        indices = np.ravel_multi_index(samples.T, unraveled_indices)
        return observable.eigvals[indices]

    def adjoint_jacobian(self, tape):
        """Implements the adjoint method outlined in
        `Jones and Gacon <https://arxiv.org/abs/2009.02823>`__ to differentiate an input tape.

        After a forward pass, the circuit is reversed by iteratively applying inverse (adjoint)
        gates to scan backwards through the circuit. This method is similar to the reversible
        method, but has a lower time overhead and a similar memory overhead.

        .. note::
            The adjoint differentation method has the following restrictions:

            * As it requires knowledge of the statevector, only statevector simulator devices can be
              used.

            * Only expectation values are supported as measurements.

        Args:
            tape (.QuantumTape): circuit that the function takes the gradient of

        Returns:
            array: the derivative of the tape with respect to trainable parameters.
            Dimensions are ``(len(observables), len(trainable_params))``.

        Raises:
            QuantumFunctionError: if the input tape has measurements that are not expectation values
                or contains a multi-parameter operation aside from :class:`~.Rot`
        """

        for m in tape.measurements:
            if m.return_type is not qml.operation.Expectation:
                raise qml.QuantumFunctionError(
                    "Adjoint differentiation method does not support"
                    f" measurement {m.return_type.value}"
                )

            if not hasattr(m.obs, "base_name"):
                m.obs.base_name = None  # This is needed for when the observable is a tensor product

        # Perform the forward pass.
        # Consider using caching and calling lower-level functionality. We just need the device to
        # be in the post-forward pass state.
        # https://github.com/PennyLaneAI/pennylane/pull/1032/files#r563441040
        self.reset()
        self.execute(tape)

        phi = self._reshape(self.state, [2] * self.num_wires)

        lambdas = [self._apply_operation(phi, obs) for obs in tape.observables]

        expanded_ops = []
        for op in reversed(tape.operations):
            if op.num_params > 1:
                if isinstance(op, qml.Rot) and not op.inverse:
                    ops = op.decomposition(*op.parameters, wires=op.wires)
                    expanded_ops.extend(reversed(ops))
                else:
                    raise QuantumFunctionError(
                        f"The {op.name} operation is not supported using "
                        'the "adjoint" differentiation method'
                    )
            else:
                if op.name not in ("QubitStateVector", "BasisState"):
                    expanded_ops.append(op)

        jac = np.zeros((len(tape.observables), len(tape.trainable_params)))
        dot_product_real = lambda a, b: self._real(qmlsum(self._conj(a) * b))

        param_number = len(tape._par_info) - 1  # pylint: disable=protected-access
        trainable_param_number = len(tape.trainable_params) - 1
        for op in expanded_ops:

            if (op.grad_method is not None) and (param_number in tape.trainable_params):
                d_op_matrix = operation_derivative(op)

            op.inv()
            phi = self._apply_operation(phi, op)

            if op.grad_method is not None:
                if param_number in tape.trainable_params:
                    mu = self._apply_unitary(phi, d_op_matrix, op.wires)

                    jac_column = np.array(
                        [2 * dot_product_real(lambda_, mu) for lambda_ in lambdas]
                    )
                    jac[:, trainable_param_number] = jac_column
                    trainable_param_number -= 1
                param_number -= 1

            lambdas = [self._apply_operation(lambda_, op) for lambda_ in lambdas]
            op.inv()

        return jac
