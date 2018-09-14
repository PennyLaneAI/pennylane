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
"""This module contains the device class and context manager"""

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
    """Abstract base class for devices."""
    name = ''          #: str: official device plugin name
    api_version = ''   #: str: version of OpenQML for which the plugin was made
    version = ''       #: str: version of the device plugin itself
    author = ''        #: str: plugin author(s)
    _capabilities = {} #: dict[str->*]: plugin capabilities
    _gates = {}        #: dict[str->GateSpec]: specifications for supported gates
    _observables = {}  #: dict[str->GateSpec]: specifications for supported observables
    _circuits = {}     #: dict[str->Circuit]: circuit templates associated with this API class

    def __init__(self, name, shots):
        self.name = name # the name of the device

        # number of circuit evaluations used to estimate
        # expectation values of observables. 0 means the exact ev is returned.
        self.shots = shots

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
        """Apply the queued operations to the device, and measure the expectation."""
        self.pre_execute_queued()
        self._out = self.execute_queued(queue, observe)
        self.post_execute_queued()
        return self._out

    def execute_queued(self, queue, observe):
        """Called during execute().

        Instead of overwriting this, consider implementing a suitable subset of pre_execute_queued(), post_execute_queued, execute_queued_with(), apply(), and expectation().

        Returns:
          float: expectation value(s) #todo: This is now a numpy array
        """
        with self.execute_queued_with():
            self.pre_execute_operations()
            for operation in queue:
                if not self.supported(operation.name):
                    raise DeviceError("Gate {} not supported on device {}".format(operation.name, self.name))

                par = operation.parameters()
                self.apply(operation.name, operation.wires, *par)

            self.post_execute_operations()

            self.pre_execute_expectations()
            expectations = np.array([self.expectation(observable.name, observable.wires, observable.params) for observable in observe], dtype=np.float64)
            self.post_execute_expectations()
            return expectations


    def pre_execute_queued(self):
        """Called during execute() before the individual gates and observables are executed."""
        pass

    def post_execute_queued(self):
        """Called during execute() after the individual gates and observables have been executed."""
        pass

    def pre_execute_operations(self):
        """Called during execute() before the individual gates are executed."""
        pass

    def post_execute_operations(self):
        """Called during execute() after the individual gates have been executed."""
        pass


    def pre_execute_expectations(self):
        """Called during execute() before the individual observables are executed."""
        pass

    def post_execute_expectations(self):
        """Called during execute() after the individual observables have been executed."""
        pass

    def execute_queued_with(self):
        """Called during execute(). You can overwrite this function to return an object, the individual apply() and expectation() calls are then executed in the context of that object. See the implementation of execute_queued() for mote details."""
        class MockClassForWithStatment(object): # pylint: disable=too-few-public-methods
            """Mock class as a default for the with statement in execute_queued()."""
            def __enter__(self):
                pass
            def __exit__(self, type, value, traceback):
                pass

        return MockClassForWithStatment()

    def supported(self, gate_name):
        """Return whether a gate with the give gate_name is supported by this plugin. """
        raise NotImplementedError

    def apply(self, gate_name, wires, *par):
        """Apply the gate with name gate_name to wires with parameters *par."""
        raise NotImplementedError

    def expectation(self, observable, wires, *par):
        """Return the expectation value of observable on wires with paramters *par."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Reset the backend state.

        After the reset the backend should be as if it was just constructed.
        Most importantly the quantum state is reset to its initial value.
        """
        raise NotImplementedError
