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
Devices
=======

**Module name:** :mod:`openqml.device`

.. currentmodule:: openqml.device

This module contains the :class:`Device` class.


Classes
-------

.. autosummary::
   Device

Device methods
--------------

.. currentmodule:: openqml.device.Device

.. autosummary::
   short_name
   gates
   observables
   templates
   capabilities
   execute
   reset

Internal Device methods
-----------------------

.. autosummary::
   _operator_map
   _observable_map
   pre_apply
   post_apply
   pre_expectations
   post_expectations
   execution_context
   supported
   check_validity
   apply
   expectation

.. currentmodule:: openqml.device

----
"""

import abc
import logging

import autograd.numpy as np

logging.getLogger()


class MethodFactory(type):
    """Metaclass that allows derived classes to dynamically instantiate
    new objects based on undefined methods. The dynamic methods pass their arguments
    directly to __init__ of the inheriting class."""
    def __getattr__(cls, name):
        """Get the attribute call via name"""
        def new_object(*args, **kwargs):
            """Return a new object of the same class, passing the attribute name
            as the first parameter, along with any additional parameters."""
            return cls(name, *args, **kwargs)
        return new_object


class DeviceError(Exception):
    """Exception raised by a :class:`Device` when it encounters an illegal
    operation in the quantum circuit.
    """
    pass


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""
    pass


class Device(abc.ABC):
    """Abstract base class for OpenQML devices.

    Args:
      name  (str): name of the device
      wires (int): number of subsystems in the quantum state represented by the device
      shots (int): number of circuit evaluations/random samples used to estimate expectation values of observables.
        For simulator devices, 0 means the exact EV is returned.
    """
    name = ''          #: str: official device plugin name
    api_version = ''   #: str: version of OpenQML for which the plugin was made
    version = ''       #: str: version of the device plugin itself
    author = ''        #: str: plugin author(s)
    _capabilities = {} #: dict[str->*]: plugin capabilities
    _circuits = {}     #: dict[str->Circuit]: circuit templates associated with this API class

    def __init__(self, name, wires, shots):
        self.name = name    #: str: name of the device
        self.wires = wires  #: int: number of subsystems
        self.shots = shots  #: int: number of circuit evaluations used to estimate expectation values, 0 means the exact ev is returned

    def __repr__(self):
        """String representation."""
        return self.__module__ +'.' +self.__class__.__name__ +'\nInstance: ' +self.name

    def __str__(self):
        """Verbose string representation."""
        return self.__repr__() +'\nName: ' +self.name +'\nAPI version: ' +self.api_version\
            +'\nPlugin version: ' +self.version +'\nAuthor: ' +self.author +'\n'

    @abc.abstractproperty
    def short_name(self):
        """Returns the string used to load the device."""
        raise NotImplementedError

    @abc.abstractproperty
    def _operator_map(self):
        """A dictionary {str: val} that maps OpenQML operator names to
        the corresponding operator in the device."""
        raise NotImplementedError

    @abc.abstractproperty
    def _observable_map(self):
        """A dictionary {str: val} that maps OpenQML observable names to
        the corresponding observable in the device."""
        raise NotImplementedError

    @property
    def gates(self):
        """Get the supported gate set.

        Returns:
            set[str]: the set of OpenQML operator names the device supports.
        """
        return set(self._operator_map.keys())

    @property
    def observables(self):
        """Get the supported observables.

        Returns:
            set[str]: the set of OpenQML observable names the device supports.
        """
        return set(self._observable_map.keys())

    @property
    def templates(self):
        """Get the predefined circuit templates.

        .. todo:: rename to circuits?

        Returns:
          dict[str->Circuit]: circuit templates
        """
        return self._circuits

    @classmethod
    def capabilities(cls):
        """Get the other capabilities of the plugin.

        Measurements, batching etc.

        Returns:
          dict[str->*]: results
        """
        return cls._capabilities

    def execute(self, queue, observe):
        """Apply a queue of quantum operations to the device, and then measure the given expectation values.

        Instead of overwriting this, consider implementing a suitable subset of
        :meth:`pre_apply`, :meth:`post_apply`, :meth:`execution_context`,
        :meth:`apply`, and :meth:`expectation`.

        Args:
          queue (Iterable[~.operation.Operation]): quantum operation objects to apply to the device
          observe (Iterable[~.operation.Expectation]): expectation values to measure and return
        Returns:
          array[float]: expectation value(s)

        """
        self.check_validity(queue, observe)
        with self.execution_context():
            self.pre_apply()
            for operation in queue:
                self.apply(operation.name, operation.wires, operation.parameters)
            self.post_apply()

            self.pre_expectations()
            expectations = [self.expectation(observable.name, observable.wires, observable.parameters) for observable in observe]
            self.post_expectations()

            return np.array(expectations)

    def pre_apply(self):
        """Called during :meth:`execute` before the individual gates are executed."""
        pass

    def post_apply(self):
        """Called during :meth:`execute` after the individual gates have been executed."""
        pass

    def pre_expectations(self):
        """Called during :meth:`execute` before the individual observables are executed."""
        pass

    def post_expectations(self):
        """Called during :meth:`execute` after the individual observables have been executed."""
        pass

    def execution_context(self):
        """The device execution context used during calls to :meth:`execute`.

        You can overwrite this function to return a suitable context manager;
        all operations and method calls (including :meth:`apply` and :meth:`expectation`)
        are then evaluated within the context of this context manager.
        """
        class MockContext(object): # pylint: disable=too-few-public-methods
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
        return name in self.gates.union(self.observables)

    def check_validity(self, queue, observe):
        """Check whether the operations and observables are supported by the device.

        Args:
          queue (Iterable[~.operation.Operation]): quantum operation objects to apply to the device
          observe (Iterable[~.operation.Expectation]): expectation values to measure and return
        """
        for operation in queue:
            if not self.supported(operation.name):
                raise DeviceError("Gate {} not supported on device {}".format(operation.name, self.short_name))

        for observable in observe:
            if not self.supported(observable.name):
                raise DeviceError("Observable {} not supported on device {}".format(observable.name, self.short_name))

    @abc.abstractmethod
    def apply(self, gate_name, wires, par):
        """Apply a quantum operation.

        Args:
          gate_name (str): name of the operation
          wires (Sequence[int]): subsystems the operation is applied on
          par (tuple): parameters for the operation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def expectation(self, observable, wires, par):
        """Expectation value of an observable.

        Args:
          observable (str): name of the observable
          wires (Sequence[int]): subsystems the observable is measured on
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
