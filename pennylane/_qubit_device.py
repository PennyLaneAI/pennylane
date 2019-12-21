# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
# pylint: disable=too-many-format-args
import abc

import numpy as np

from pennylane.operation import Operation, Observable, Sample, Variance, Expectation, Tensor
from pennylane.qnodes import QuantumFunctionError
from pennylane import Device, DeviceError


class QubitDevice(Device):
    """Abstract base class for PennyLane devices.

    Args:
        wires (int): number of subsystems in the quantum state represented by the device.
            Default 1 if not specified.
        shots (int): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. Defaults to 1000 if not specified.
    """
    #pylint: disable=too-many-public-methods
    def __init__(self, wires=1, shots=1000):
        super().__init__(wires=wires, shots=shots)

    def pre_measure(self, observables):
        """Called during :meth:`execute` before the individual observables are measured."""


    def execute(self, queue, observables, parameters={}):
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

            self.post_apply()

            # Pass queue of observables to pre_measure
            self.pre_measure()

            for obs in observables:
                # Pass instances directly
                if obs.return_type is Expectation:
                    results.append(self.expval(obs))

                elif obs.return_type is Variance:
                    results.append(self.var(obs))

                elif obs.return_type is Sample:
                    results.append(np.array(self.sample(obs)))

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

    def expval(self, observable, wires, par):
        if self.backend_name in self._state_backends and self.analytic:
            # exact expectation value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals @ prob).real

        if self.analytic:
            # Raise a warning if backend is a hardware simulator
            warnings.warn(self.hw_analytic_warning_message.
                          format(self.backend),
                          UserWarning)

        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def var(self, observable, wires, par):
        if self.backend_name in self._state_backends and self.analytic:
            # exact variance value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals ** 2) @ prob - (eigvals @ prob).real ** 2

        if self.analytic:
            # Raise a warning if backend is a hardware simulator
            warnings.warn(self.hw_analytic_warning_message.
                          format(self.backend),
                          UserWarning)

        return np.var(self.sample(observable, wires, par))

    def sample(self, observable, wires, par):
        if observable == "Identity":
            return np.ones([self.shots])

        # branch out depending on the type of backend
        if self.backend_name in self._state_backends:
            # software simulator. Need to sample from probabilities.
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return np.random.choice(eigvals, self.shots, p=prob)

        # a hardware simulator
        if self.memory:
            # get the samples
            samples = self._current_job.result().get_memory()

            # reverse qubit order to match PennyLane convention
            samples = np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

        else:
            # Need to convert counts into samples
            samples = np.vstack(
                [np.vstack([s] * int(self.shots * p)) for s, p in self.probabilities().items()]
            )

        if isinstance(observable, str) and observable in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            return 1 - 2 * samples[:, wires[0]]

        eigvals = self.eigvals(observable, wires, par)
        wires = np.hstack(wires)
        res = samples[:, np.array(wires)]
        samples = np.zeros([self.shots])

        for w, b in zip(eigvals, itertools.product([0, 1], repeat=len(wires))):
            samples = np.where(np.all(res == b, axis=1), w, samples)

        return samples

class HardwareQubitDevice(QubitDevice):
    def __init__(self, wires=1, shots=1000):
        super().__init__(wires=wires, shots=shots)

    def expval(self, observable, wires, par):
        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def var(self, observable, wires, par):
        return np.var(self.sample(observable, wires, par))

    def probabilities(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        # hardware simulator
        result = self._current_job.result()

        # sort the counts and reverse qubit order to match PennyLane convention
        nonzero_prob = {
            tuple(int(i) for i in s[::-1]): c / self.shots
            for s, c in result.get_counts().items()
        }

        if wires is None:
            # marginal probabilities not required
            return OrderedDict(tuple(sorted(nonzero_prob.items())))

        prob = np.zeros([2] * self.num_wires)

        for s, p in tuple(sorted(nonzero_prob.items())):
            prob[s] = p

        wires = wires or range(self.num_wires)
        wires = np.hstack(wires)

        basis_states = itertools.product(range(2), repeat=len(wires))
        inactive_wires = list(set(range(self.num_wires)) - set(wires))
        prob = np.apply_over_axes(np.sum, prob, inactive_wires).flatten()
        return OrderedDict(zip(basis_states, prob))

    @abc.abstractmethod
    def obtain_samples(self):
        """
        The specific device can implement obtaining samples
        """

    def sample(self, observable, wires, par):
        samples = self.obtain_samples()

        if isinstance(observable, str) and observable in {"PauliX", "PauliY", "PauliZ", "Hadamard"}:
            return 1 - 2 * samples[:, wires[0]]

        eigvals = self.eigvals(observable, wires, par)
        wires = np.hstack(wires)
        res = samples[:, np.array(wires)]
        samples = np.zeros([self.shots])

        for w, b in zip(eigvals, itertools.product([0, 1], repeat=len(wires))):
            samples = np.where(np.all(res == b, axis=1), w, samples)

        return samples


class StateVectorQubitDevice(QubitDevice):
    def __init__(self, wires=1, shots=1000, analytic=True):
        super().__init__(wires=wires, shots=shots)
        self.analytic = analytic

    def expval(self, observable, wires, par):
        if self.analytic:
            # exact expectation value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals @ prob).real

        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def var(self, observable, wires, par):
        if self.analytic:
            # exact variance value
            eigvals = self.eigvals(observable, wires, par)
            prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
            return (eigvals ** 2) @ prob - (eigvals @ prob).real ** 2

        # estimate the ev
        return np.mean(self.sample(observable, wires, par))

    def sample(self, observable, wires, par):
        if observable == "Identity":
            return np.ones([self.shots])

        # software simulator. Need to sample from probabilities.
        eigvals = self.eigvals(observable, wires, par)
        prob = np.fromiter(self.probabilities(wires=wires).values(), dtype=np.float64)
        return np.random.choice(eigvals, self.shots, p=prob)

    def probabilities(self, wires=None):
        """Return the (marginal) probability of each computational basis
        state from the last run of the device.

        Args:
            wires (Sequence[int]): Sequence of wires to return
                marginal probabilities for. Wires not provided
                are traced out of the system.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        prob = np.abs(self.state.reshape([2] * self.num_wires)) ** 2
        wires = wires or range(self.num_wires)
        wires = np.hstack(wires)

        basis_states = itertools.product(range(2), repeat=len(wires))
        inactive_wires = list(set(range(self.num_wires)) - set(wires))
        prob = np.apply_over_axes(np.sum, prob, inactive_wires).flatten()
        return OrderedDict(zip(basis_states, prob))

