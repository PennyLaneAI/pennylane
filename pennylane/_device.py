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
Device base class
=================

**Module name:** :mod:`pennylane._device`

.. currentmodule:: pennylane._device

This module contains the :class:`Device` abstract base class. To write a plugin containing a PennyLane-compatible device, :class:`Device`
must be subclassed, and the appropriate class attributes and methods
implemented. For examples of subclasses of :class:`Device`, see :class:`~.DefaultQubit`,
:class:`~.DefaultGaussian`, or the `StrawberryFields <https://pennylane-sf.readthedocs.io/>`_
and `ProjectQ <https://pennylane-pq.readthedocs.io/>`_ plugins.

.. autosummary::
    Device

Device attributes and methods
-----------------------------

.. currentmodule:: pennylane._device.Device

The following methods and attributes are accessible from the PennyLane
user interface:

.. autosummary::
    short_name
    capabilities
    supported
    execute
    reset

Abstract methods and attributes
-------------------------------

The following methods and attributes must be defined for all devices:

.. autosummary::
    name
    short_name
    pennylane_requires
    version
    author
    operations
    observables
    apply
    expval
    var

In addition, the following may also be optionally defined:

.. autosummary::
    pre_apply
    post_apply
    pre_measure
    post_measure
    execution_context


Internal attributes and methods
-------------------------------

The following methods and attributes are used internally by the :class:`Device` class,
to ensure correct operation and internal consistency.

.. autosummary::
    check_validity

.. currentmodule:: pennylane._device


Code details
~~~~~~~~~~~~
"""
# pylint: disable=too-many-format-args
import abc

import autograd.numpy as np


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
        shots (int): number of circuit evaluations/random samples used to estimate
            expectation values of observables. For simulator devices, a value of 0 results
            in the exact expectation value being returned. Defaults to 0 if not specified.
    """
    #pylint: disable=too-many-public-methods
    _capabilities = {} #: dict[str->*]: plugin capabilities
    _circuits = {}     #: dict[str->Circuit]: circuit templates associated with this API class

    def __init__(self, wires=1, shots=0):
        self.num_wires = wires
        self.shots = shots

        self._op_queue = None
        self._obs_queue = None

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

    @classmethod
    def capabilities(cls):
        """Get the other capabilities of the plugin.

        Measurements, batching etc.

        Returns:
            dict[str->*]: results
        """
        return cls._capabilities

    def execute(self, queue, observables):
        """Execute a queue of quantum operations on the device and then measure the given expectation values.

        For plugin developers: Instead of overwriting this, consider implementing a suitable subset of
        :meth:`pre_apply`, :meth:`apply`, :meth:`post_apply`, :meth:`pre_measure`,
        :meth:`expval`, :meth:`var`, :meth:`sample`, :meth:`post_measure`, and :meth:`execution_context`.

        Args:
            queue (Iterable[~.operation.Operation]): operations to execute on the device
            observables (Iterable[~.operation.Observable]): observables to measure and return

        Returns:
            array[float]: measured value(s)
        """
        self.check_validity(queue, observables)
        self._op_queue = queue
        self._obs_queue = observables

        results = []

        with self.execution_context():
            self.pre_apply()

            for operation in queue:
                self.apply(operation.name, operation.wires, operation.parameters)

            self.post_apply()

            self.pre_measure()

            for obs in observables:
                if obs.return_type == "expectation":
                    results.append(self.expval(obs.name, obs.wires, obs.parameters))
                elif obs.return_type == "variance":
                    results.append(self.var(obs.name, obs.wires, obs.parameters))

            self.post_measure()

            self._op_queue = None
            self._obs_queue = None

            return np.array(results)

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

    def supported(self, name):
        """Checks if an operation or observable is supported by this device.

        Args:
            name (str): name of the operation or observable

        Returns:
            bool: True iff it is supported
        """
        return name in self.operations.union(self.observables)

    def check_validity(self, queue, observables):
        """Checks whether the operations and observables in queue are all supported by the device.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
                to be applied on the device
            expectations (Iterable[~.operation.Observable]): observables which are intended
                to be evaluated on the device
        """
        for o in queue:
            if o.name not in self.operations:
                raise DeviceError("Gate {} not supported on device {}".format(o.name, self.short_name))

        for o in observables:
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
        """Return the expectation value of an observable.

        For plugin developers: this function should return the expectation value of the
        given observable on the device.

        Args:
            observable (str): name of the observable
            wires (Sequence[int]): subsystems the observable is to be measured on
            par (tuple): parameters for the observable

        Returns:
            float: expectation value
        """
        raise NotImplementedError

    def var(self, observable, wires, par):
        """Return the variance value of an observable.

        For plugin developers: this function should return the variance value of the
        given observable on the device.

        Args:
            observable (str): name of the observable
            wires (Sequence[int]): subsystems the observable is to be measured on
            par (tuple): parameters for the observable

        Returns:
            float: expectation value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset the backend state.

        After the reset the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        raise NotImplementedError
