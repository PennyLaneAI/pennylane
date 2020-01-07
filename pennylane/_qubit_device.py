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
# pylint: disable=too-many-format-args, arguments-differ, abstract-method
import itertools
from collections import OrderedDict
import numpy as np

import pennylane as qml
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

        self._state = None
        self._memory = None
        self._samples = None

    @property
    def _probability(self):
        return np.abs(self._state)**2

    @property
    def _basis_states(self):
        return np.arange(2**self.num_wires)


    def apply(self, operation):
        """Called during :meth:`execute` before the individual observables are measured."""


    def pre_measure(self):
        """Called during :meth:`execute` before the individual observables are measured."""

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
                    results.append(list(self.probability(wires=obs.wires).values()))

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

    def rotate_basis(self, observable):
        """Rotates the specified wires such that they
        are in the eigenbasis of the provided observable.

        Args:
            observable (Observable): the observable that is to be
                measured
        """
        for diag_gate in observable.diagonalizing_gates():
            self.apply(diag_gate)

    def expval(self, observable):
        wires = observable.wires
        self.rotate_basis(observable)

        if self.analytic:
            # exact expectation value
            eigvals = observable.eigvals
            prob = np.abs(self._state ** 2)
            prob = self.marginal_prob(prob, wires=wires)
            return (eigvals @ prob).real

        # estimate the ev
        return np.mean(self.sample(observable))

    def var(self, observable):
        wires = observable.wires
        self.rotate_basis(observable)

        if self.analytic:
            # exact variance value
            eigvals = observable.eigvals
            prob = np.abs(self._state ** 2)
            prob = self.marginal_prob(prob, wires=wires)
            return (eigvals ** 2) @ prob - (eigvals @ prob).real ** 2

        return np.var(self.sample(observable))


    def probability(self, wires=None):
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
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)

        prob = self.marginal_prob(self._probability, wires)
        basis_states = itertools.product(range(2), repeat=len(wires))
        return OrderedDict(zip(basis_states, prob))

    def marginal_prob(self, prob, wires=None):
        """Return the marginal probability of each computational basis
        state from the last run of the device.
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
