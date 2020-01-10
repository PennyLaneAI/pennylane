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

import itertools
from collections import OrderedDict
import numpy as np

from pennylane.operation import Sample, Variance, Expectation, Probability
from pennylane.qnodes import QuantumFunctionError
from pennylane import Device


class QubitDevice(Device):
    """Abstract base class for PennyLane qubit devices.

    Args:
        wires (int): number of subsystems in the quantum state represented by the device.
            Default 1 if not specified.
        shots (int): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. Defaults to 1000 if not specified.
    """
    #pylint: disable=too-many-public-methods
    def __init__(self, wires=1, shots=1000, analytic=True):
        super().__init__(wires=wires, shots=shots)
        self.analytic = analytic

        self.reset()


    def reset(self):
        """Reset the backend state.

        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        self._state = None
        self._prob = None
        self._rotated_prob = None
        self._wires_used = None
        self._memory = None
        self._samples = None

    def execute(self, queue, observables, parameters=None):
        """Execute a queue of quantum operations on the device and then measure the given observables.

        For plugin developers: Instead of overwriting this, consider implementing a suitable subset of
        :meth:`pre_apply`, :meth:`apply`, :meth:`post_apply`, :meth:`pre_measure`,
        :meth:`expval`, :meth:`var`, :meth:`sample`, :meth:`post_measure`, and :meth:`execution_context`.

        Args:
            queue (Iterable[~.operation.Operation]): operations to execute on the device
            observables (Iterable[~.operation.Observable]): observables to measure and return
            parameters (dict[int->list[ParameterDependency]]): Mapping from free parameter index to the list of
                :class:`Operations <pennylane.operation.Operation>` (in the queue) that depend on it.

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            array[float]: measured value(s)
        """
        if parameters is None:
            parameters = {}

        self.check_validity(queue, observables)
        self._op_queue = queue
        self._obs_queue = observables
        self._parameters = {}
        self._parameters.update(parameters)

        results = []

        with self.execution_context():
            self.pre_apply()

            for operation in queue:
                # Pass instances directly
                self.apply(operation)

            self.rotate_basis(observables)

            self.post_apply()

            self.pre_measure()

            for obs in observables:
                # Pass instances directly
                if obs.return_type is Expectation:
                    results.append(self.expval(obs))

                elif obs.return_type is Variance:
                    results.append(self.var(obs))

                elif obs.return_type is Sample:
                    results.append(np.array(self.sample(obs)))

                elif obs.return_type is Probability:
                    results.append(list(self.probability(wires=obs.wires)))

                elif obs.return_type is not None:
                    raise QuantumFunctionError("Unsupported return type specified for observable {}".format(obs.name))

            self.post_measure()

            self._op_queue = None
            self._obs_queue = None
            self._parameters = None

            # Ensures that a combination with sample does not put
            # expvals and vars in superfluous arrays
            if all(obs.return_type is Sample for obs in observables):
                return self._asarray(results)
            if any(obs.return_type is Sample for obs in observables):
                return self._asarray(results, dtype="object")

            return self._asarray(results)

    def rotate_basis(self, obs_queue):
        """Rotates the specified wires such that they
        are in the eigenbasis of the provided observable.
        """
        self._prob = self.probability()
        self._memory = False

        wires = []

        for observable in obs_queue:
            if hasattr(observable, "return_type") and observable.return_type == Sample:
                self._memory = True  # make sure to return samples

            for diag_gate in observable.diagonalizing_gates():
                self.apply(diag_gate)

            for wire in observable.wires:
                if isinstance(wire, int):
                    wires.append(wire)
                else:
                    wires.extend(wire)

        self._wires_used = wires
        self._rotated_prob = self.probability(wires)

    def generate_samples(self):
        """If the device contains a sample return type, or the
        device is running in non-analytic mode, ``dev.shots`` number of
        computational basis samples are generated and stored within
        the :attr:`~._samples` attribute.

        .. note::

            This method should only be called by devices that do not
            generate their own computational basis samples.
        """
        if self._memory or (not self.analytic):
            # sample from the computational basis states based on the state probability
            number_of_states = 2**len(self._wires_used)
            basis_states = np.arange(number_of_states)
            samples = np.random.choice(basis_states, self.shots, p=self._rotated_prob)

            # convert the basis states from base 10 to binary representation
            powers_of_two = (1 << np.arange(number_of_states))
            states_sampled_base_ten = samples[:, None] & powers_of_two
            self._samples = (states_sampled_base_ten > 0).astype(int)

    def expval(self, observable):
        wires = observable.wires

        if self._rotated_prob is None:
            self.rotate_basis([observable])

        if self.analytic:
            # exact expectation value
            eigvals = observable.eigvals
            prob = self.probability(wires=wires)
            return (eigvals @ prob).real

        # estimate the ev
        return np.mean(self.sample(observable))

    def var(self, observable):
        wires = observable.wires

        if self._rotated_prob is None:
            self.rotate_basis([observable])

        if self.analytic:
            # exact variance value
            eigvals = observable.eigvals
            prob = self.probability(wires=wires)
            return (eigvals ** 2) @ prob - (eigvals @ prob).real ** 2

        return np.var(self.sample(observable))

    def sample(self, observable):
        wires = observable.wires
        name = observable.name

        if self._rotated_prob is None:
            observable.return_type = Sample
            self.rotate_basis([observable])
            self.generate_samples()

        if isinstance(name, str) and name in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            # observables all have eigenvalues {1, -1}, so post-processing step is known
            return 1 - 2 * self._samples[:, wires[0]]

        # Need to post-process the samples using the observables.
        # Extract only the columns of the basis samples required based on `wires`.
        wires = np.hstack(wires)
        samples = self._samples[:, np.array(wires)]

        # replace the basis state in the computational basis with the correct eigenvalue
        indices = np.ravel_multi_index(samples.T, [2] * len(wires))
        converted_measurements = observable.eigvals[indices]
        return converted_measurements

    @property
    def probabilitiy_with_basis_states(self):
        basis_states = itertools.product(range(2), repeat=len(wires))
        return dict(zip(basis_states, self.probability()))

    def probability(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.
            values_only (bool): If ``True``, the return type is modified to
                instead be a flattened array of the computational
                basis state probabilities.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)
        prob = self.marginal_prob(np.abs(self._state)**2, wires)
        return prob

    def marginal_prob(self, prob, wires=None):
        """Return the marginal probability of the computational basis
        states by summing the probabiliites on the non-specified wires.

        If no wires are specified, then all the basis states representable by
        the device are considered and no marginalization takes place.

        Args:
            prob: The probabilities to return the marginal probabilities
                for
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            list[float]: List of the resulting marginal probabilities.
        """
        wires = wires or range(self.num_wires)
        wires = np.hstack(wires)
        inactive_wires = list(set(range(self.num_wires)) - set(wires))
        prob = prob.reshape([2] * self.num_wires)
        return np.apply_over_axes(np.sum, prob, inactive_wires).flatten()

