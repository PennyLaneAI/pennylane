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
This module contains the :class:`Device` abstract base class.
"""
# pylint: disable=too-many-format-args
import abc
from collections.abc import Iterable
from collections import OrderedDict

import numpy as np

import pennylane as qml
from pennylane.operation import (
    Operation,
    Observable,
    Sample,
    State,
    Variance,
    Expectation,
    Probability,
    Tensor,
)
from pennylane.qnodes import QuantumFunctionError
from pennylane.wires import Wires, WireError


class DeviceError(Exception):
    """Exception raised by a :class:`~.pennylane._device.Device` when it encounters an illegal
    operation in the quantum circuit.
    """


class Device(abc.ABC):
    """Abstract base class for PennyLane devices.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (int): Number of circuit evaluations/random samples used to estimate
            expectation values of observables. Defaults to 1000 if not specified.
    """

    # pylint: disable=too-many-public-methods
    _capabilities = {
        "model": None,
    }
    """The capabilities dictionary stores the properties of a device. Devices can add their
    own custom properties and overwrite existing ones by overriding the ``capabilities()`` method."""

    _circuits = {}  #: dict[str->Circuit]: circuit templates associated with this API class
    _asarray = staticmethod(np.asarray)

    def __init__(self, wires=1, shots=1000):

        self.shots = shots

        if not isinstance(wires, Iterable):
            # interpret wires as the number of consecutive wires
            wires = range(wires)

        self._wires = Wires(wires)
        self.num_wires = len(self._wires)
        self._wire_map = self.define_wire_map(self._wires)
        self._num_executions = 0
        self._op_queue = None
        self._obs_queue = None
        self._parameters = None

    def __repr__(self):
        """String representation."""
        return "<{} device (wires={}, shots={}) at {}>".format(
            self.__class__.__name__, self.num_wires, self.shots, hex(id(self))
        )

    def __str__(self):
        """Verbose string representation."""
        return "{}\nShort name: {}\nPackage: {}\nPlugin version: {}\nAuthor: {}\nWires: {}\nShots: {}".format(
            self.name,
            self.short_name,
            self.__module__.split(".")[0],
            self.version,
            self.author,
            self.num_wires,
            self.shots,
        )

    @property
    @abc.abstractmethod
    def name(self):
        """The full name of the device."""

    @property
    @abc.abstractmethod
    def short_name(self):
        """Returns the string used to load the device."""

    @property
    @abc.abstractmethod
    def pennylane_requires(self):
        """The current API version that the device plugin was made for."""

    @property
    @abc.abstractmethod
    def version(self):
        """The current version of the plugin."""

    @property
    @abc.abstractmethod
    def author(self):
        """The author(s) of the plugin."""

    @property
    @abc.abstractmethod
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """

    @property
    @abc.abstractmethod
    def observables(self):
        """Get the supported set of observables.

        Returns:
            set[str]: the set of PennyLane observable names the device supports
        """

    @property
    def shots(self):
        """Number of circuit evaluations/random samples used to estimate
        expectation values of observables"""
        return self._shots

    @property
    def wires(self):
        """All wires that can be addressed on this device"""
        return self._wires

    @property
    def wire_map(self):
        """Ordered dictionary that defines the map from user-provided wire labels to
        the wire labels used on this device"""
        return self._wire_map

    @property
    def num_executions(self):
        """Number of times this device is executed by the evaluation of QNodes
        running on this device

        Returns:
            int: number of executions
        """
        return self._num_executions

    @shots.setter
    def shots(self, shots):
        """Changes the number of shots.

        Args:
            shots (int): number of circuit evaluations/random samples used to estimate
                expectation values of observables

        Raises:
            DeviceError: if number of shots is less than 1
        """
        if shots < 1:
            raise DeviceError(
                "The specified number of shots needs to be at least 1. Got {}.".format(shots)
            )

        self._shots = int(shots)

    def define_wire_map(self, wires):
        """Create the map from user-provided wire labels to the wire labels used by the device.

        The default wire map maps the user wire labels to wire labels that are consecutive integers.

        However, by overwriting this function, devices can specify their preferred, non-consecutive and/or non-integer
        wire labels.

        Args:
            wires (Wires): user-provided wires for this device

        Returns:
            OrderedDict: dictionary specifying the wire map

        **Example**

        >>> dev = device('my.device', wires=['b', 'a'])
        >>> dev.wire_map()
        OrderedDict( [(<Wires = ['a']>, <Wires = [0]>), (<Wires = ['b']>, <Wires = [1]>)])
        """
        consecutive_wires = Wires(range(self.num_wires))

        wire_map = zip(wires, consecutive_wires)
        return OrderedDict(wire_map)

    def map_wires(self, wires):
        """Map the wire labels of wires using this device's wire map.

        Args:
            wires (Wires): wires whose labels we want to map to the device's internal labelling scheme

        Returns:
            Wires: wires with new labels
        """
        try:
            mapped_wires = wires.map(self.wire_map)
        except WireError as e:
            raise WireError(
                "Did not find some of the wires {} on device with wires {}.".format(
                    wires, self.wires
                )
            ) from e

        return mapped_wires

    @classmethod
    def capabilities(cls):
        """Get the capabilities of this device class.

        Inheriting classes that change or add capabilities must override this method, for example via

        .. code-block:: python

            @classmethod
            def capabilities(cls):
                capabilities = super().capabilities().copy()
                capabilities.update(
                    supports_inverse_operations=False,
                    supports_a_new_capability=True,
                )
                return capabilities

        Returns:
            dict[str->*]: results
        """
        return cls._capabilities

    def execute(self, queue, observables, parameters={}, **kwargs):
        """Execute a queue of quantum operations on the device and then measure the given observables.

        For plugin developers: Instead of overwriting this, consider implementing a suitable subset of
        :meth:`pre_apply`, :meth:`apply`, :meth:`post_apply`, :meth:`pre_measure`,
        :meth:`expval`, :meth:`var`, :meth:`sample`, :meth:`post_measure`, and :meth:`execution_context`.

        Args:
            queue (Iterable[~.operation.Operation]): operations to execute on the device
            observables (Iterable[~.operation.Observable]): observables to measure and return
            parameters (dict[int, list[ParameterDependency]]): Mapping from free parameter index to the list of
                :class:`Operations <pennylane.operation.Operation>` (in the queue) that depend on it.

        Keyword Args:
            return_native_type (bool): If True, return the result in whatever type the device uses
                internally, otherwise convert it into array[float]. Default: False.

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
                self.apply(operation.name, operation.wires, operation.parameters)

            self.post_apply()

            self.pre_measure()

            for obs in observables:

                if isinstance(obs, Tensor):
                    wires = [ob.wires for ob in obs.obs]
                else:
                    wires = obs.wires

                if obs.return_type is Expectation:
                    results.append(self.expval(obs.name, wires, obs.parameters))

                elif obs.return_type is Variance:
                    results.append(self.var(obs.name, wires, obs.parameters))

                elif obs.return_type is Sample:
                    results.append(np.array(self.sample(obs.name, wires, obs.parameters)))

                elif obs.return_type is Probability:
                    results.append(list(self.probability(wires=wires).values()))

                elif obs.return_type is State:
                    raise QuantumFunctionError("Returning the state is not supported")

                elif obs.return_type is not None:
                    raise QuantumFunctionError(
                        "Unsupported return type specified for observable {}".format(obs.name)
                    )

            self.post_measure()

            self._op_queue = None
            self._obs_queue = None
            self._parameters = None

            # increment counter for number of executions of device
            self._num_executions += 1

            # Ensures that a combination with sample does not put
            # expvals and vars in superfluous arrays
            if all(obs.return_type is Sample for obs in observables):
                return self._asarray(results)
            if any(obs.return_type is Sample for obs in observables):
                return self._asarray(results, dtype="object")

            return self._asarray(results)

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
        results = []
        for circuit in circuits:
            # we need to reset the device here, else it will
            # not start the next computation in the zero state
            self.reset()

            res = self.execute(circuit.operations, circuit.observables)
            results.append(res)

        return results

    @property
    def op_queue(self):
        """The operation queue to be applied.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Raises:
            ValueError: if outside of the execution context

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

        Raises:
            ValueError: if outside of the execution context

        Returns:
            list[~.operation.Observable]
        """
        if self._obs_queue is None:
            raise ValueError(
                "Cannot access the observable value queue outside of the execution context!"
            )

        return self._obs_queue

    @property
    def parameters(self):
        """Mapping from free parameter index to the list of
        :class:`Operations <~.Operation>` in the device queue that depend on it.

        Note that this property can only be accessed within the execution context
        of :meth:`~.execute`.

        Raises:
            ValueError: if outside of the execution context

        Returns:
            dict[int->list[ParameterDependency]]: the mapping
        """
        if self._parameters is None:
            raise ValueError(
                "Cannot access the free parameter mapping outside of the execution context!"
            )

        return self._parameters

    def pre_apply(self):
        """Called during :meth:`execute` before the individual operations are executed."""

    def post_apply(self):
        """Called during :meth:`execute` after the individual operations have been executed."""

    def pre_measure(self):
        """Called during :meth:`execute` before the individual observables are measured."""

    def post_measure(self):
        """Called during :meth:`execute` after the individual observables have been measured."""

    def execution_context(self):
        """The device execution context used during calls to :meth:`execute`.

        You can overwrite this function to return a context manager in case your
        quantum library requires that;
        all operations and method calls (including :meth:`apply` and :meth:`expval`)
        are then evaluated within the context of this context manager (see the
        source of :meth:`.Device.execute` for more details).
        """
        # pylint: disable=no-self-use
        class MockContext:  # pylint: disable=too-few-public-methods
            """Mock class as a default for the with statement in execute()."""

            def __enter__(self):
                pass

            def __exit__(self, type, value, traceback):
                pass

        return MockContext()

    def supports_operation(self, operation):
        """Checks if an operation is supported by this device.

        Args:
            operation (type or str): operation to be checked

        Raises:
            ValueError: if `operation` is not a :class:`~.Operation` class or string

        Returns:
            bool: ``True`` iff supplied operation is supported
        """
        if isinstance(operation, type) and issubclass(operation, Operation):
            return operation.__name__ in self.operations
        if isinstance(operation, str):

            if operation.endswith(Operation.string_for_inverse):
                in_ops = operation[: -len(Operation.string_for_inverse)] in self.operations
                # TODO: update when all capabilities keys changed to "supports_inverse_operations"
                supports_inv = self.capabilities().get(
                    "supports_inverse_operations", False
                ) or self.capabilities().get("inverse_operations", False)
                return in_ops and supports_inv

            return operation in self.operations

        raise ValueError(
            "The given operation must either be a pennylane.Operation class or a string."
        )

    def supports_observable(self, observable):
        """Checks if an observable is supported by this device. Raises a ValueError,
         if not a subclass or string of an Observable was passed.

        Args:
            observable (type or str): observable to be checked

        Raises:
            ValueError: if `observable` is not a :class:`~.Observable` class or string

        Returns:
            bool: ``True`` iff supplied observable is supported
        """
        if isinstance(observable, type) and issubclass(observable, Observable):
            return observable.__name__ in self.observables
        if isinstance(observable, str):

            # This check regards observables that are also operations
            if observable.endswith(Operation.string_for_inverse):
                return self.supports_operation(observable[: -len(Operation.string_for_inverse)])

            return observable in self.observables

        raise ValueError(
            "The given observable must either be a pennylane.Observable class or a string."
        )

    def check_validity(self, queue, observables):
        """Checks whether the operations and observables in queue are all supported by the device.
        Includes checks for inverse operations.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
                to be applied on the device
            observables (Iterable[~.operation.Observable]): observables which are intended
                to be evaluated on the device

        Raises:
            DeviceError: if there are operations in the queue or observables that the device does
                not support
        """

        for o in queue:

            operation_name = o.name

            if o.inverse:
                # TODO: update when all capabilities keys changed to "supports_inverse_operations"
                supports_inv = self.capabilities().get(
                    "supports_inverse_operations", False
                ) or self.capabilities().get("inverse_operations", False)
                if not supports_inv:
                    raise DeviceError(
                        "The inverse of gates are not supported on device {}".format(
                            self.short_name
                        )
                    )
                operation_name = o.base_name

            if not self.supports_operation(operation_name):
                raise DeviceError(
                    "Gate {} not supported on device {}".format(operation_name, self.short_name)
                )

        for o in observables:
            if isinstance(o, qml.tape.MeasurementProcess) and o.obs is not None:
                o = o.obs

            if isinstance(o, Tensor):
                # TODO: update when all capabilities keys changed to "supports_tensor_observables"
                supports_tensor = self.capabilities().get(
                    "supports_tensor_observables", False
                ) or self.capabilities().get("tensor_observables", False)
                if not supports_tensor:
                    raise DeviceError(
                        "Tensor observables not supported on device {}".format(self.short_name)
                    )

                for i in o.obs:
                    if not self.supports_observable(i.name):
                        raise DeviceError(
                            "Observable {} not supported on device {}".format(
                                i.name, self.short_name
                            )
                        )
            else:
                observable_name = o.name

                if issubclass(o.__class__, Operation) and o.inverse:
                    # TODO: update when all capabilities keys changed to "supports_inverse_operations"
                    supports_inv = self.capabilities().get(
                        "supports_inverse_operations", False
                    ) or self.capabilities().get("inverse_operations", False)
                    if not supports_inv:
                        raise DeviceError(
                            "The inverse of gates are not supported on device {}".format(
                                self.short_name
                            )
                        )
                    observable_name = o.base_name

                if not self.supports_observable(observable_name):
                    raise DeviceError(
                        "Observable {} not supported on device {}".format(
                            observable_name, self.short_name
                        )
                    )

    @abc.abstractmethod
    def apply(self, operation, wires, par):
        """Apply a quantum operation.

        For plugin developers: this function should apply the operation on the device.

        Args:
            operation (str): name of the operation
            wires (Wires): wires that the operation is applied to
            par (tuple): parameters for the operation
        """

    @abc.abstractmethod
    def expval(self, observable, wires, par):
        r"""Returns the expectation value of observable on specified wires.

        Note: all arguments accept _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) are to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Returns:
            float: expectation value :math:`\expect{A} = \bra{\psi}A\ket{\psi}`
        """

    def var(self, observable, wires, par):
        r"""Returns the variance of observable on specified wires.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Raises:
            NotImplementedError: if the device does not support variance computation

        Returns:
            float: variance :math:`\mathrm{var}(A) = \bra{\psi}A^2\ket{\psi} - \bra{\psi}A\ket{\psi}^2`
        """
        raise NotImplementedError(
            "Returning variances from QNodes not currently supported by {}".format(self.short_name)
        )

    def sample(self, observable, wires, par):
        """Return a sample of an observable.

        The number of samples is determined by the value of ``Device.shots``,
        which can be directly modified.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Raises:
            NotImplementedError: if the device does not support sampling

        Returns:
            array[float]: samples in an array of dimension ``(shots,)``
        """
        raise NotImplementedError(
            "Returning samples from QNodes not currently supported by {}".format(self.short_name)
        )

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
        raise NotImplementedError(
            "Returning probability not currently supported by {}".format(self.short_name)
        )

    @abc.abstractmethod
    def reset(self):
        """Reset the backend state.

        After the reset, the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
