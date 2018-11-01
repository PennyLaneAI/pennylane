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

This module contains the :class:`Device` class. This is an abstract base class,
which should be subclassed, and the appropriate class attributes and methods
defined. As such, the :class:`Device` class should never be accessed or instantiated
directly. For examples of subclasses of :class:`Device`, see :class:`~.DefaultQubit`
or :class:`~.DefaultGaussian`.

.. autosummary::
    Device

Device attributes and methods
-----------------------------

.. currentmodule:: pennylane._device.Device

The following methods and attributes are accessible from the PennyLane
user interface:

.. autosummary::
    short_name
    operations
    expectations
    capabilities
    supported
    execute
    reset

Abstract methods and attributes
-------------------------------

The following methods and attributes must be defined for all devices:

.. autosummary::
    _operation_map
    _expectation_map
    apply
    expval

In addition, the following may also be optionally defined:

.. autosummary::
    pre_apply
    post_apply
    pre_expval
    post_expval
    execution_context
    var


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
import logging

import autograd.numpy as np

logging.getLogger()


class DeviceError(Exception):
    """Exception raised by a :class:`~.pennylane._device.Device` when it encounters an illegal
    operation in the quantum circuit.
    """
    pass


class Device(abc.ABC):
    """Abstract base class for PennyLane devices.

    Args:
        name (str): name of the device.
        wires (int): number of subsystems in the quantum state represented by the device.
            Default 1 if not specified.
        shots (int): number of circuit evaluations/random samples used to estimate
            expectation values of observables. For simulator devices, 0 results
            in the exact expectation value being is returned. Default 0 if not specified.
    """
    name = ''          #: str: official device plugin name
    api_version = ''   #: str: version of PennyLane for which the plugin was made
    version = ''       #: str: version of the device plugin itself
    author = ''        #: str: plugin author(s)
    _capabilities = {} #: dict[str->*]: plugin capabilities
    _circuits = {}     #: dict[str->Circuit]: circuit templates associated with this API class

    def __init__(self, name, wires=1, shots=0):
        self.name = name
        self.num_wires = wires
        self.shots = shots

    def __repr__(self):
        """String representation."""
        return "{}.\nInstance: ".format(self.__module__, self.__class__.__name__, self.name)

    def __str__(self):
        """Verbose string representation."""
        return "{}\nName: \nAPI version: \nPlugin version: \nAuthor: ".format(self.name, self.api_version, self.version, self.author)

    @abc.abstractproperty
    def short_name(self):
        """Returns the string used to load the device."""
        raise NotImplementedError

    @abc.abstractproperty
    def _operation_map(self):
        """A dictionary {str: val} that maps PennyLane operation names to
        the corresponding operation in the device."""
        raise NotImplementedError

    @abc.abstractproperty
    def _expectation_map(self):
        """A dictionary {str: val} that maps PennyLane expectation names to
        the corresponding expectation in the device."""
        raise NotImplementedError

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    @property
    def expectations(self):
        """Get the supported set of expectations.

        Returns:
            set[str]: the set of PennyLane expectation names the device supports
        """
        return set(self._expectation_map.keys())

    @classmethod
    def capabilities(cls):
        """Get the other capabilities of the plugin.

        Measurements, batching etc.

        Returns:
            dict[str->*]: results
        """
        return cls._capabilities

    def execute(self, queue, expectation, variance=False):
        """Apply a queue of quantum operations to the device, and then measure the given expectation values.

        Instead of overwriting this, consider implementing a suitable subset of
        :meth:`pre_apply`, :meth:`apply`, :meth:`post_apply`, :meth:`pre_expval`,
        :meth:`expval`, :meth:`post_expval`, and :meth:`execution_context`.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects to apply on the device
            expectation (Iterable[~.operation.Expectation]): expectations to evaluate and return
            variance (bool): if True, the variances are returned instead of the expectation values.

        Returns:
            array[float]: expectation value(s)
        """
        self.check_validity(queue, expectation)
        with self.execution_context():
            self.pre_apply()
            for operation in queue:
                self.apply(operation.name, operation.wires, operation.parameters)
            self.post_apply()

            self.pre_expval()
            if variance:
                try:
                    result = [self.var(e.name, e.wires, e.parameters) for e in expectation]
                except NotImplementedError as e:
                    raise DeviceError("Device does not support expectation value variances.") from None
            else:
                result = [self.expval(e.name, e.wires, e.parameters) for e in expectation]
            self.post_expval()

            return np.array(result)

    def pre_apply(self):
        """Called during :meth:`execute` before the individual operations are executed."""
        pass

    def post_apply(self):
        """Called during :meth:`execute` after the individual operations have been executed."""
        pass

    def pre_expval(self):
        """Called during :meth:`execute` before the individual expectations are executed."""
        pass

    def post_expval(self):
        """Called during :meth:`execute` after the individual expectations have been executed."""
        pass

    def execution_context(self):
        """The device execution context used during calls to :meth:`execute`.

        You can overwrite this function to return a suitable context manager;
        all operations and method calls (including :meth:`apply` and :meth:`expval`)
        are then evaluated within the context of this context manager.
        """
        # pylint: disable=no-self-use
        class MockContext(object): # pylint: disable=too-few-public-methods
            """Mock class as a default for the with statement in execute()."""
            def __enter__(self):
                pass
            def __exit__(self, type, value, traceback):
                pass

        return MockContext()

    def supported(self, name):
        """Checks if an operation or expectation is supported by this device.

        Args:
            name (str): name of the operation or expectation

        Returns:
            bool: True iff it is supported
        """
        return name in self.operations.union(self.expectations)

    def check_validity(self, queue, expectations):
        """Check whether the operations and expectations are supported by the device.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended to be applied in device
            expectations (Iterable[~.operation.Expectation]): expectations which are intended to be evaluated in device
        """
        for o in queue:
            if not self.supported(o.name):
                raise DeviceError("Gate {} not supported on device {}".format(o.name, self.short_name))

        for e in expectations:
            if not self.supported(e.name):
                raise DeviceError("Expectation {} not supported on device {}".format(e.name, self.short_name))

    @abc.abstractmethod
    def apply(self, op_name, wires, par):
        """Apply a quantum operation.

        Args:
            op_name (str): name of the operation
            wires (Sequence[int]): subsystems the operation is applied on
            par (tuple): parameters for the operation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def expval(self, expectation, wires, par):
        """Expectation value of an observable.

        Args:
            expectation (str): name of the expectation to evaluate
            wires (Sequence[int]): subsystems the expectation is measured on
            par (tuple): parameters for the expectation

        Returns:
            float: expectation value
        """
        raise NotImplementedError

    def var(self, expectation, wires, par):
        """Variance value of an observable.

        Args:
            expectation (str): name of the expectation to evaluate
            wires (Sequence[int]): subsystems the expectation is measured on
            par (tuple): parameters for the expectation

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
