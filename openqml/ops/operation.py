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
"""This module contains a symbolic quantum operation"""
import logging as log
log.getLogger()

from pkg_resources import iter_entry_points
from openqml.device import Device, DeviceError


# get a list of installed operations
plugin_operations = {
    entry.name: entry.load() for entry in iter_entry_points('openqml.ops')
}


class OperationFactory(type):
    """Metaclass that allows derived classes to dynamically instantiate
    new operations as loaded from plugins."""
    def __getattr__(cls, name):
        """Get the attribute call via name"""
        if name not in plugin_operations:
            raise DeviceError("Operation {} not installed. Please install "
                              "the plugin that provides it.".format(name))

        return plugin_operations[name]


class Operation(metaclass=OperationFactory):
    """A type of quantum operation supported by a plugin, and its properties.

    Operation is used to describe unitary quantum gates.

    Args:
        name  (str): name of the operation
        wires (seq): subsystems it acts on
        params (seq): operation parameters
        grad_method (str): gradient computation method: 'A': angular, 'F': finite differences.
        par_domain  (str): domain of the gate parameters: 'N': natural numbers (incl. zero), 'R': floats.
            Parameters outside the domain are truncated into it.
    """
    def __init__(self, name, params, wires, *, grad_method='A', par_domain='R'):
        self.name  = name   #: str: name of the gate
        self.params = params  #: seq: operation parameters
        self.grad_method = grad_method  #: str: gradient computation method
        self.par_domain  = par_domain   #: str: domain of the gate parameters

        if isinstance(wires, int):
            self.wires = [wires]
        else:
            self.wires = wires

        self.queue()

    def __str__(self):
        """Print the operation name and information"""
        return self.name +': {} params, {} wires'.format(len(self.params), len(self.wires))

    def queue(self):
        """Append the operation to a device queue"""
        if Device._current_context is None:
            raise DeviceError("Quantum operations can only be used inside a qfunc or a device context manager.")
        else:
            Device._current_context._queue.append(self)
