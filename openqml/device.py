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
   gates
   observables
   templates
   capabilities
   execute
   reset

Internal Device methods
-----------------------

.. autosummary::
   execute_queued
   pre_execute_queued
   post_execute_queued
   pre_execute_operations
   post_execute_operations
   pre_execute_expectations
   post_execute_expectations
   execute_queued_with
   supported
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
    _gates = {}        #: dict[str->GateSpec]: specifications for supported gates
    _observables = {}  #: dict[str->GateSpec]: specifications for supported observables
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

    @property
    def gates(self):
        """Get the supported gate set.

        Returns:
          dict[str->GateSpec]:
        """
        return self._gates

    @property
    def observables(self):
        """Get the supported observables.

        Returns:
          dict[str->GateSpec]:
        """
        return self._observables

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

        Args:
          queue (Iterable[~.operation.Operation]): quantum operations to apply
          observe (Iterable[~.operation.Expectation]): expectation values to measure
        Returns:
          array[float]: expectation value(s)
        """
        self.pre_execute_queued()
        self._out = self.execute_queued(queue, observe)
        self.post_execute_queued()
        return self._out

    def execute_queued(self, queue, observe):
        """Executes the given quantum program.

        Should not be called directly by the user, use :meth:`execute` instead.
        Instead of overwriting this, consider implementing a suitable subset of pre_execute_queued(), post_execute_queued, execute_queued_with(), apply(), and expectation().

        Args:
          queue (Iterable[~.operation.Operation]): quantum operations to apply
          observe (Iterable[~.operation.Expectation]): expectation values to measure
        Returns:
          array[float]: expectation value(s)
        """
        with self.execute_queued_with():
            self.pre_execute_operations()
            for operation in queue:
                if not self.supported(operation.name):
                    raise DeviceError("Gate {} not supported on device {}".format(operation.name, self.name))

                par = operation.parameters
                self.apply(operation.name, operation.wires, par)

            self.post_execute_operations()

            self.pre_execute_expectations()
            expectations = np.array([self.expectation(observable.name, observable.wires, observable.parameters) for observable in observe], dtype=np.float64)
            self.post_execute_expectations()
            return expectations


    def pre_execute_queued(self):
        """Called during :meth:`execute` before the individual gates and observables are executed."""
        pass

    def post_execute_queued(self):
        """Called during :meth:`execute` after the individual gates and observables have been executed."""
        pass

    def pre_execute_operations(self):
        """Called during :meth:`execute` before the individual gates are executed."""
        pass

    def post_execute_operations(self):
        """Called during :meth:`execute` after the individual gates have been executed."""
        pass

    def pre_execute_expectations(self):
        """Called during :meth:`execute` before the individual observables are executed."""
        pass

    def post_execute_expectations(self):
        """Called during :meth:`execute` after the individual observables have been executed."""
        pass

    def execute_queued_with(self):
        """Called during :meth:`execute`.

        You can overwrite this function to return an object, the individual :meth:`apply` and :meth:`expectation`
        calls are then executed in the context of that object.
        See the implementation of :meth:`execute_queued` for more details.
        """
        class MockClassForWithStatment(object): # pylint: disable=too-few-public-methods
            """Mock class as a default for the with statement in execute_queued()."""
            def __enter__(self):
                pass
            def __exit__(self, type, value, traceback):
                pass

        return MockClassForWithStatment()

    def supported(self, gate_name):
        """Checks if the given gate is supported by this device.

        Args:
          gate_name (str): name of the operation
        Returns:
          bool: True iff it is supported
        """
        raise NotImplementedError

    def apply(self, gate_name, wires, par):
        """Apply a quantum operation.

        Args:
          gate_name (str): name of the operation
          wires (Sequence[int]): subsystems the operation is applied on
          par (tuple): parameters for the operation
        """
        raise NotImplementedError

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
