import abc
import itertools
import warnings

import numpy as np

import pennylane as qml
from pennylane import DeviceError
from pennylane.operation import operation_derivative
from pennylane.measurements import Sample, Variance, Expectation, Probability, State
from pennylane import Device
from pennylane.math import sum as qmlsum
from pennylane.math import multiply as qmlmul
from pennylane.wires import Wires

from pennylane.measurements import MeasurementProcess


class QutritDevice(Device):
    """Abstract base class for Pennylane qutrit devices.

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

    This device contains common utility methods for qutrit-based devices. These
    do not need to be overwritten. Utility methods include:

    * :meth:`~.expval`, :meth:`~.var`, :meth:`~.sample`: return expectation values,
      variances, and samples of observables after the circuit has been rotated
      into the observable eigenbasis.

    Args:
        wires (int, Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int, list[int]): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. If ``None``, the device calculates probability, expectation values,
            and variances analytically. If an integer, it specifies the number of samples to estimate these quantities.
            If a list of integers is passed, the circuit evaluations are batched over the list of shots.
        r_dtype: Real floating point precision type.
        c_dtype: Complex floating point precision type.
    """

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

    @staticmethod
    def _const_mul(constant, array):
        """Data type preserving multiply operation"""
        return qmlmul(constant, array, dtype=array.dtype)

    def _permute_wires(self, observable):
        r"""Given an observable which acts on multiple wires, permute the wires to
        be consistent with the device wire order.

        This function uses the observable wires and the global device wire ordering in order to determine the
        permutation of the wires in the observable required such that if our quantum state vector is
        permuted accordingly then the amplitudes of the state will match the matrix representation of the observable.

        Args:
            observable (Observable): the observable whose wires are to be permuted.

        Returns:
            permuted_wires (Wires): permuted wires object
        """
        ordered_obs_wire_lst = self.order_wires(
            observable.wires
        ).tolist()  # order according to device wire order

        mapped_wires = self.map_wires(observable.wires)
        if isinstance(mapped_wires, Wires):
            # by default this should be a Wires obj, but it is overwritten to list object in default.qubit
            mapped_wires = mapped_wires.tolist()

        permutation = np.argsort(mapped_wires)  # extract permutation via argsort

        permuted_wires = Wires([ordered_obs_wire_lst[index] for index in permutation])
        return permuted_wires

    # TODO: Add list of observables
    observables = {
        "Identity",
    }

    def __init__(
        self, wires=1, shots=None, *, r_dtype=np.float64, c_dtype=np.complex128, analytic=None
    ):
        super().__init__(wires=wires, shots=shots, analytic=analytic)

        if "float" not in str(r_dtype):
            raise DeviceError("Real datatype must be a floating point type.")
        if "complex" not in str(c_dtype):
            raise DeviceError("Complex datatype must be a complex floating point type.")

        self.C_DTYPE = c_dtype
        self.R_DTYPE = r_dtype

        self._samples = None
        """None or array[int]: stores the samples generated by the device
        *after* rotation to diagonalize the observables."""

    @classmethod
    def capabilities(cls):

        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qutrit",
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

        self.check_validity(circuit.operations, circuit.observables)

        # apply all circuit operations
        self.apply(circuit.operations, rotations=circuit.diagonalizing_gates, **kwargs)

        # generate computational basis samples
        if self.shots is not None or circuit.is_sampled:
            self._samples = self.generate_samples()

        multiple_sampled_jobs = circuit.is_sampled and self._has_partitioned_shots()

        # compute the required statistics
        if not self.analytic and self._shot_vector is not None:

            results = []
            s1 = 0

            for shot_tuple in self._shot_vector:
                s2 = s1 + np.prod(shot_tuple)
                r = self.statistics(
                    circuit.observables, shot_range=[s1, s2], bin_size=shot_tuple.shots
                )

                if qml.math._multi_dispatch(r) == "jax":  # pylint: disable=protected-access
                    r = r[0]
                else:
                    r = qml.math.squeeze(r)

                if shot_tuple.copies > 1:
                    results.extend(r.T)
                else:
                    results.append(r.T)

                s1 = s2

            if not multiple_sampled_jobs:
                # Can only stack single element outputs
                results = qml.math.stack(results)

        else:
            results = self.statistics(circuit.observables)

        if not circuit.is_sampled:

            ret_types = [m.return_type for m in circuit.measurements]

            if len(circuit.measurements) == 1:
                if circuit.measurements[0].return_type is qml.measurements.State:
                    # State: assumed to only be allowed if it's the only measurement
                    results = self._asarray(results, dtype=self.C_DTYPE)
                else:
                    # Measurements with expval, var or probs
                    results = self._asarray(results, dtype=self.R_DTYPE)

            elif all(
                ret in (qml.measurements.Expectation, qml.measurements.Variance)
                for ret in ret_types
            ):
                # Measurements with expval or var
                results = self._asarray(results, dtype=self.R_DTYPE)
            else:
                results = self._asarray(results)

        elif circuit.all_sampled and not self._has_partitioned_shots():

            results = self._asarray(results)
        else:
            results = tuple(self._asarray(r) for r in results)

        # increment counter for number of executions of qubit device
        self._num_executions += 1

        if self.tracker.active:
            self.tracker.update(executions=1, shots=self._shots)
            self.tracker.record()

        return results

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

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(circuits))
            self.tracker.record()

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

    def statistics(self, observables, shot_range=None, bin_size=None):
        """Process measurement results from circuit execution and return statistics.

        This includes returning expectation values, variance, samples, probabilities, states, and
        density matrices.

        Args:
            observables (List[.Observable]): the observables to be measured
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            Union[float, List[float]]: the corresponding statistics

        .. details::
            :title: Usage Details

            The ``shot_range`` and ``bin_size`` arguments allow for the statistics
            to be performed on only a subset of device samples. This finer level
            of control is accessible from the main UI by instantiating a device
            with a batch of shots.

            For example, consider the following device:

            >>> dev = qml.device("my_device", shots=[5, (10, 3), 100])

            This device will execute QNodes using 135 shots, however
            measurement statistics will be **course grained** across these 135
            shots:

            * All measurement statistics will first be computed using the
              first 5 shots --- that is, ``shots_range=[0, 5]``, ``bin_size=5``.

            * Next, the tuple ``(10, 3)`` indicates 10 shots, repeated 3 times. We will want to use
              ``shot_range=[5, 35]``, performing the expectation value in bins of size 10
              (``bin_size=10``).

            * Finally, we repeat the measurement statistics for the final 100 shots,
              ``shot_range=[35, 135]``, ``bin_size=100``.
        """
        results = []

        for obs in observables:
            # Pass instances directly
            if obs.return_type is Expectation:
                results.append(self.expval(obs, shot_range=shot_range, bin_size=bin_size))

            elif obs.return_type is Variance:
                results.append(self.var(obs, shot_range=shot_range, bin_size=bin_size))

            elif obs.return_type is Sample:
                results.append(self.sample(obs, shot_range=shot_range, bin_size=bin_size))

            elif obs.return_type is Probability:
                results.append(
                    self.probability(wires=obs.wires, shot_range=shot_range, bin_size=bin_size)
                )

            elif obs.return_type is State:
                if len(observables) > 1:
                    raise qml.QuantumFunctionError(
                        "The state or density matrix cannot be returned in combination"
                        " with other return types"
                    )
                if self.wires.labels != tuple(range(self.num_wires)):
                    raise qml.QuantumFunctionError(
                        "Returning the state is not supported when using custom wire labels"
                    )
                # Check if the state is accessible and decide to return the state or the density
                # matrix.
                results.append(self.access_state(wires=obs.wires))

            elif obs.return_type is not None:
                raise qml.QuantumFunctionError(
                    f"Unsupported return type specified for observable {obs.name}"
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
            raise qml.QuantumFunctionError(
                "The current device is not capable of returning the state"
            )

        state = getattr(self, "state", None)

        if state is None:
            raise qml.QuantumFunctionError("The state is not available in the current device")

        if wires:
            density_matrix = self.density_matrix(wires)
            return density_matrix

        return state

    def generate_samples(self):
        r"""Returns the computational basis samples generated for all wires.

        Note that PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
        :math:`q_0` is the most significant trit.

        .. warning::

            This method should be overwritten on devices that
            generate their own computational basis samples, with the resulting
            computational basis samples stored as ``self._samples``.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        number_of_states = 3**self.num_wires

        rotated_prob = self.analytic_probability()

        samples = self.sample_basis_states(number_of_states, rotated_prob)
        return QutritDevice.states_to_ternary(samples, self.num_wires)

    def sample_basis_states(self, number_of_states, state_probability):
        """Sample from the computational basis states based on the state
        probability.

        This is an auxiliary method to the generate_samples method.

        Args:
            number_of_states (int): the number of basis states to sample from
            state_probability (array[float]): the computational basis probability vector

        Returns:
            array[int]: the sampled basis states
        """
        if self.shots is None:

            raise qml.QuantumFunctionError(
                "The number of shots has to be explicitly set on the device "
                "when using sample-based measurements."
            )

        shots = self.shots

        basis_states = np.arange(number_of_states)
        return np.random.choice(basis_states, shots, p=state_probability)

    @staticmethod
    def generate_basis_states(num_wires, dtype=np.uint32):
        """
        Generates basis states in ternary representation according to the number
        of wires specified.

        Args:
            num_wires (int): the number wires
            dtype=np.uint32 (type): the data type of the arrays to use

        Returns:
            array[int]: the sampled basis states
        """
        # A slower, but less memory intensive method
        basis_states_generator = itertools.product((0, 1, 2), repeat=num_wires)
        return np.fromiter(itertools.chain(*basis_states_generator), dtype=dtype).reshape(
            -1, num_wires
        )

    @staticmethod
    def states_to_ternary(samples, num_wires, dtype=np.int64):
        """Convert basis states from base 10 to ternary representation.

        This is an auxiliary method to the generate_samples method.

        Args:
            samples (array[int]): samples of basis states in base 10 representation
            num_wires (int): the number of qutrits
            dtype (type): Type of the internal integer array to be used. Can be
                important to specify for large systems for memory allocation
                purposes.

        Returns:
            array[int]: basis states in ternary representation
        """
        ternary_arr = []
        for sample in samples:
            num = []
            for _ in range(num_wires):
                sample, r = divmod(sample, 3)
                num.append(r)

            ternary_arr.append(num[::-1])

        return np.array(ternary_arr, dtype=dtype)


    @property
    def circuit_hash(self):
        """The hash of the circuit upon the last execution.

        This can be used by devices in :meth:`~.apply` for parametric compilation.
        """
        raise NotImplementedError

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
        significant trit.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        .. note::

            :meth:`marginal_prob` may be used as a utility method
            to calculate the marginal probability distribution.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """
        raise NotImplementedError

    def estimate_probability(self, wires=None, shot_range=None, bin_size=None):
        """Return the estimated probability of each computational basis state
        using the generated samples.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to calculate
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """

        wires = wires or self.wires
        # convert to a wires object
        wires = Wires(wires)
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        sample_slice = Ellipsis if shot_range is None else slice(*shot_range)
        samples = self._samples[sample_slice, device_wires]

        # convert samples from a list of 0, 1 integers, to base 10 representation
        powers_of_three = 3 ** np.arange(len(device_wires))[::-1]
        indices = samples @ powers_of_three

        # count the basis state occurrences, and construct the probability vector
        if bin_size is not None:
            bins = len(samples) // bin_size

            indices = indices.reshape((bins, -1))
            prob = np.zeros([3 ** len(device_wires), bins], dtype=np.float64)

            # count the basis state occurrences, and construct the probability vector
            for b, idx in enumerate(indices):
                basis_states, counts = np.unique(idx, return_counts=True)
                prob[basis_states, b] = counts / bin_size

        else:
            basis_states, counts = np.unique(indices, return_counts=True)
            prob = np.zeros([3 ** len(device_wires)], dtype=np.float64)
            prob[basis_states] = counts / len(samples)

        return self._asarray(prob, dtype=self.R_DTYPE)

    def probability(self, wires=None, shot_range=None, bin_size=None):
        """Return either the analytic probability or estimated probability of
        each computational basis state.

        Devices that require a finite number of shots always return the
        estimated probability.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to return
                marginal probabilities for. Wires not provided are traced out of the system.

        Returns:
            array[float]: list of the probabilities
        """

        if self.shots is None:
            return self.analytic_probability(wires=wires)

        return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

    # TODO: Implement function
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
        prob = self._reshape(prob, [3] * self.num_wires)

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
        basis_states = basis_states[:, np.argsort(np.argsort(device_wires))]

        powers_of_three = 3 ** np.arange(len(device_wires))[::-1]
        perm = basis_states @ powers_of_three
        return self._gather(prob, perm)

    def expval(self, observable, shot_range=None, bin_size=None):

        # TODO: Add Projector case
        # exact expectation value
        if self.shots is None:
            try:
                eigvals = self._asarray(observable.eigvals(), dtype=self.R_DTYPE)
            except qml.operation.EigvalsUndefinedError as e:
                raise qml.operation.EigvalsUndefinedError(
                    f"Cannot compute analytic expectations of {observable.name}."
                ) from e

            # the probability vector must be permuted to account for the permuted wire order of the observable
            permuted_wires = self._permute_wires(observable)

            prob = self.probability(wires=permuted_wires)
            return self._dot(eigvals, prob)

        # estimate the ev
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        return np.squeeze(np.mean(samples, axis=0))

    def var(self, observable, shot_range=None, bin_size=None):

        # TODO: Add Projector case
        # exact variance value
        if self.shots is None:
            try:
                eigvals = self._asarray(observable.eigvals(), dtype=self.R_DTYPE)
            except qml.operation.EigvalsUndefinedError as e:
                # if observable has no info on eigenvalues, we cannot return this measurement
                raise qml.operation.EigvalsUndefinedError(
                    f"Cannot compute analytic variance of {observable.name}."
                ) from e

            # the probability vector must be permuted to account for the permuted wire order of the observable
            permuted_wires = self._permute_wires(observable)

            prob = self.probability(wires=permuted_wires)
            return self._dot((eigvals**2), prob) - self._dot(eigvals, prob) ** 2

        # estimate the variance
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        return np.squeeze(np.var(samples, axis=0))

    def sample(self, observable, shot_range=None, bin_size=None):

        # TODO: Add special cases for any particular observables

        # translate to wire labels used by device
        device_wires = self.map_wires(observable.wires)
        name = observable.name
        sample_slice = Ellipsis if shot_range is None else slice(*shot_range)

        # Replace the basis state in the computational basis with the correct eigenvalue.
        # Extract only the columns of the basis samples required based on ``wires``.
        samples = self._samples[
            sample_slice, np.array(device_wires)
        ]  # Add np.array here for Jax support.
        powers_of_three = 3 ** np.arange(samples.shape[-1])[::-1]
        indices = samples @ powers_of_three
        indices = np.array(indices)  # Add np.array here for Jax support.
        try:
            samples = observable.eigvals()[indices]
        except qml.operation.EigvalsUndefinedError as e:
            # if observable has no info on eigenvalues, we cannot return this measurement
            raise qml.operation.EigvalsUndefinedError(
                f"Cannot compute samples of {observable.name}."
            ) from e

        if bin_size is None:
            return samples

        return samples.reshape((bin_size, -1))

    # TODO: Implement function
    def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        pass