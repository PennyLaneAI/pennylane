# Copyright 2018 Xanadu Quantum Technologies Inc.

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
This module contains the :class:`Device` abstract base class.
"""
# pylint: disable=too-many-format-args
import abc

import autograd.numpy as np
from pennylane.operation import Operation, Observable, Sample, Variance, Expectation, Tensor
from .qnode import QuantumFunctionError


class DeviceError(Exception):
    """Exception raised by a :class:`~.pennylane._device.Device` when it encounters an illegal
    operation in the quantum circuit.
    """
    pass


class Device(abc.ABC):
    """Abstract base class for PennyLane devices.

    Args:
        wires (int): number of subsystems in the quantum state represented by the device.
            Default 1 if not specified.
        shots (int): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. Defaults to 1000 if not specified.
    """
    #pylint: disable=too-many-public-methods
    _capabilities = {} #: dict[str->*]: plugin capabilities
    _circuits = {}     #: dict[str->Circuit]: circuit templates associated with this API class

    def __init__(self, wires=1, shots=1000):
        self.num_wires = wires
        self._shots = 1000
        self.shots = shots

        self._op_queue = None
        self._obs_queue = None
        self._parameters = None

    def __repr__(self):
        """String representation."""
        return "{}.\nInstance: ".format(self.__module__, self.__class__.__name__, self.name)

    def __str__(self):
        """Verbose string representation."""
        return "{}\nName: \nAPI version: \nPlugin version: \nAuthor: ".format(self.name, self.pennylane_requires, self.version, self.author)

    @abc.abstractproperty
    def name(self):
        """The full name of the device."""
        raise NotImplementedError

    @abc.abstractproperty
    def short_name(self):
        """Returns the string used to load the device."""
        raise NotImplementedError

    @abc.abstractproperty
    def pennylane_requires(self):
        """The current API version that the device plugin was made for."""
        raise NotImplementedError

    @abc.abstractproperty
    def version(self):
        """The current version of the plugin."""
        raise NotImplementedError

    @abc.abstractproperty
    def author(self):
        """The author(s) of the plugin."""
        raise NotImplementedError

    @abc.abstractproperty
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        raise NotImplementedError

    @abc.abstractproperty
    def observables(self):
        """Get the supported set of observables.

        Returns:
            set[str]: the set of PennyLane observable names the device supports
        """
        raise NotImplementedError

    @property
    def shots(self):
        """Number of circuit evaluations/random samples used to estimate
        expectation values of observables"""

        return self._shots

    @shots.setter
    def shots(self, shots):
        """Changes the number of shots.

        Args:
            shots (int): number of circuit evaluations/random samples used to estimate
                expectation values of observables
        """
        if shots < 1:
            raise DeviceError("The specified number of shots needs to be at least 1. Got {}.".format(shots))

        self._shots = int(shots)

    @classmethod
    def capabilities(cls):
        """Get the other capabilities of the plugin.

        Measurements, batching etc.

        Returns:
            dict[str->*]: results
        """
        return cls._capabilities

    def execute(self, queue, observables, parameters={}):
        """Execute a queue of quantum operations on the device and then measure the given observables.

        For plugin developers: Instead of overwriting this, consider implementing a suitable subset of
        :meth:`pre_apply`, :meth:`apply`, :meth:`post_apply`, :meth:`pre_measure`,
        :meth:`expval`, :meth:`var`, :meth:`sample`, :meth:`post_measure`, and :meth:`execution_context`.

        Args:
            queue (Iterable[~.operation.Operation]): operations to execute on the device
            observables (Iterable[~.operation.Observable]): observables to measure and return
            parameters (dict[int->list[(int, int)]]): Mapping from free parameter index to the list of
                :class:`Operations <pennylane.operation.Operation>` (in the queue) that depend on it.
                The first element of the tuple is the index of the Operation in the program queue,
                the second the index of the parameter within the Operation.

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
                self.apply(operation.name, operation.wires, operation.parameters)

            self.post_apply()

            self.pre_measure()

            for obs in observables:
                if obs.return_type is Expectation:
                    results.append(self.expval(obs.name, obs.wires, obs.parameters))

                elif obs.return_type is Variance:
                    results.append(self.var(obs.name, obs.wires, obs.parameters))

                elif obs.return_type is Sample:
                    results.append(np.array(self.sample(obs.name, obs.wires, obs.parameters)))

                elif obs.return_type is not None:
                    raise QuantumFunctionError("Unsupported return type specified for observable {}".format(obs.name))

            self.post_measure()

            self._op_queue = None
            self._obs_queue = None
            self._parameters = None

            # Ensures that a combination with sample does not put
            # expvals and vars in superfluous arrays
            if all(obs.return_type is Sample for obs in observables):
                return np.asarray(results)
            if any(obs.return_type is Sample for obs in observables):
                return np.asarray(results, dtype="object")

            return np.asarray(results)

    @property
    def op_queue(self):
        """The operation queue to be applied.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Returns:
            list[~.operation.Operation]
        """
        if self._op_queue is None:
            raise ValueError("Cannot access the operation queue outside of the execution context!")

        return self._op_queue

    @property
    def obs_queue(self):
        """The observables to be measured and returned.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Returns:
            list[~.operation.Observable]
        """
        if self._obs_queue is None:
            raise ValueError("Cannot access the observable value queue outside of the execution context!")

        return self._obs_queue

    @property
    def parameters(self):
        """Mapping from free parameter index to the list of
        :class:`Operations <~.Operation>` in the device queue that depend on it.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Returns:
            dict[int->list[(int, int)]]: the first element of the tuple is the index
            of the Operation in the program queue, the second the index of the parameter
            within the Operation.
        """
        if self._parameters is None:
            raise ValueError("Cannot access the free parameter mapping outside of the execution context!")

        return self._parameters

    def pre_apply(self):
        """Called during :meth:`execute` before the individual operations are executed."""
        pass

    def post_apply(self):
        """Called during :meth:`execute` after the individual operations have been executed."""
        pass

    def pre_measure(self):
        """Called during :meth:`execute` before the individual observables are measured."""
        pass

    def post_measure(self):
        """Called during :meth:`execute` after the individual observables have been measured."""
        pass

    def execution_context(self):
        """The device execution context used during calls to :meth:`execute`.

        You can overwrite this function to return a context manager in case your
        quantum library requires that;
        all operations and method calls (including :meth:`apply` and :meth:`expval`)
        are then evaluated within the context of this context manager (see the
        source of :meth:`.Device.execute` for more details).
        """
        # pylint: disable=no-self-use
        class MockContext: # pylint: disable=too-few-public-methods
            """Mock class as a default for the with statement in execute()."""
            def __enter__(self):
                pass
            def __exit__(self, type, value, traceback):
                pass

        return MockContext()

    def supports_operation(self, operation):
        """Checks if an operation is supported by this device.

        Args:
            operation (Operation,str): operation to be checked

        Returns:
            bool: ``True`` iff supplied operation is supported
        """
        if isinstance(operation, type) and issubclass(operation, Operation):
            return operation.__name__ in self.operations
        if isinstance(operation, str):
            return operation in self.operations

        raise ValueError("The given operation must either be a pennylane.Operation class or a string.")

    def supports_observable(self, observable):
        """Checks if an observable is supported by this device.

        Args:
            operation (Observable,str): observable to be checked

        Returns:
            bool: ``True`` iff supplied observable is supported
        """
        if isinstance(observable, type) and issubclass(observable, Observable):
            return observable.__name__ in self.observables
        if isinstance(observable, str):
            return observable in self.observables

        raise ValueError("The given operation must either be a pennylane.Observable class or a string.")


    def check_validity(self, queue, observables):
        """Checks whether the operations and observables in queue are all supported by the device.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
                to be applied on the device
            observables (Iterable[~.operation.Observable]): observables which are intended
                to be evaluated on the device
        """
        for o in queue:
            if o.name not in self.operations:
                raise DeviceError("Gate {} not supported on device {}".format(o.name, self.short_name))

        for o in observables:

            if isinstance(o, Tensor):
                if "tensor_observables" not in self.capabilities() or not self.capabilities()["tensor_observables"]:
                    raise DeviceError("Tensor observables not supported on device {}".format(self.short_name))

                for i in o.obs:
                    if i.name not in self.observables:
                        raise DeviceError("Observable {} not supported on device {}".format(i.name, self.short_name))
            else:
                if o.name not in self.observables:
                    raise DeviceError("Observable {} not supported on device {}".format(o.name, self.short_name))

    @abc.abstractmethod
    def apply(self, operation, wires, par):
        """Apply a quantum operation.

        For plugin developers: this function should apply the operation on the device.

        Args:
            operation (str): name of the operation
            wires (Sequence[int]): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def expval(self, observable, wires, par):
        r"""Returns the expectation value of observable on specified wires.

        Note: all arguments accept _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (List[int] or List[List[int]]): subsystems the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Returns:
            float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """
        raise NotImplementedError

    def var(self, observable, wires, par):
        r"""Returns the variance of observable on specified wires.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (List[int] or List[List[int]]): subsystems the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Returns:
            float: variance :math:`\mathrm{var}(A) = \bra{\psi}A^2\ket{\psi} - \bra{\psi}A\ket{\psi}^2`
        """
        raise NotImplementedError("Returning variances from QNodes not currently supported by {}".format(self.short_name))

    def sample(self, observable, wires, par):
        """Return a sample of an observable.

        The number of samples is determined by the value of ``Device.shots``,
        which can be directly modified.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (List[int] or List[List[int]]): subsystems the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Returns:
            array[float]: samples in an array of dimension ``(n, num_wires)``
        """
        raise NotImplementedError("Returning samples from QNodes not currently supported by {}".format(self.short_name))

    def probability(self):
        """Return the full state probability of each computational basis state from the last run of the device.

        Returns:
            OrderedDict[tuple, float]: Dictionary mapping a tuple representing the state
            to the resulting probability. The dictionary should be sorted such that the
            state tuples are in lexicographical order.
        """
        raise NotImplementedError("Returning probability not currently supported by {}".format(self.short_name))

    @abc.abstractmethod
    def reset(self):
        """Reset the backend state.

        After the reset the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        raise NotImplementedError
