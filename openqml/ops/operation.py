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
    r"""A type of quantum operation supported by a plugin, and its properties.

    Operation is used to describe unitary quantum gates.

    Args:
        name  (str): name of the operation
        wires (seq): subsystems it acts on
        params (seq): operation parameters
        par_domain  (str): domain of the gate parameters: 'N': natural numbers (incl. zero), 'R': floats.
            Parameters outside the domain are truncated into it.
        grad_method (str): gradient computation method: 'A': angular, 'F': finite differences.
        grad_recipe (list[tuple[float]]): gradient recipe for the 'A' method. One tuple for each parameter:
            (multiplier c_k, parameter shift s_k). None means (0.5, \pi/2) (the most common case).

            .. math:: \frac{\partial Q(\ldots, \theta_k, \ldots)}{\partial \theta_k}} = c_k (Q(\ldots, \theta_k+s_k, \ldots) -Q(\ldots, \theta_k-s_k, \ldots))

            To find out in detail how the circuit gradients are computed, see :ref:`circuit_gradients`.
    """
    def __init__(self, name, params, wires, *, par_domain='R', grad_method='A', grad_recipe=None):
        self.name  = name   #: str: name of the gate
        self.params = params  #: seq: operation parameters
        self.par_domain  = par_domain   #: str: domain of the gate parameters

        self.grad_method = grad_method  #: str: gradient computation method

        # default recipe for every parameter
        self.grad_recipe = [None] * len(self.params)

        if grad_recipe is not None:

            if len(grad_recipe) != len(self.params):
                raise ValueError('Gradient recipe must have one entry for each parameter!')

            self.grad_recipe = grad_recipe

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
