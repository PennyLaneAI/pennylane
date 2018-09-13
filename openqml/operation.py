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
import openqml.qnode as oq

from .variable import Variable


# get a list of installed operations
plugin_operations  = {entry.name: entry.load() for entry in iter_entry_points('openqml.ops')}
plugin_expectation = {entry.name: entry.load() for entry in iter_entry_points('openqml.expectation')}


class OperationFactory(type):
    """Metaclass that allows derived classes to dynamically instantiate
    new operations as loaded from plugins.

    .. note:: Currently unused.
    """
    def __getattr__(cls, name):
        """Get the attribute call via name"""
        if name not in plugin_operations:
            raise DeviceError("Operation {} not installed. Please install "
                              "the plugin that provides it.".format(name))

        return plugin_operations[name]


class ExpectationFactory(type):
    """Metaclass that allows derived classes to dynamically instantiate
    new expectations as loaded from plugins.

    .. note:: Currently unused.
    """
    def __getattr__(cls, name):
        """Get the attribute call via name"""
        if name not in plugin_expectation:
            raise DeviceError("Expectation {} not installed. Please install "
                              "the plugin that provides it.".format(name))

        return plugin_expectation[name]


class Operation:
    r"""A type of quantum operation supported by a plugin, and its properties.

    * Each Operation subclass represents a type of quantum operation.
    * Each instance of these subclasses represents an application of the
      operation with given parameter values to a given sequence of subsystems.

    Operation is used to describe unitary quantum gates.

    Args:
        *params (tuple[float, int, Variable]): operation parameters
        wires (Sequence[int]): subsystems it acts on

    The gradient recipe (multiplier :math:`c_k`, parameter shift :math:`s_k`)
    works as follows:

    .. math::

        \frac{\partial Q(\ldots, \theta_k, \ldots)}{\partial \theta_k}}
        = c_k (Q(\ldots, \theta_k+s_k, \ldots) -Q(\ldots, \theta_k-s_k, \ldots))

    To find out in detail how the circuit gradients are computed, see :ref:`circuit_gradients`.
    """
    n_params = 1        #: int: number of parameters the operation takes
    n_wires  = 1        #: int: number of subsystems the operation acts on. The value 0 means any number of subsystems is OK.
    par_domain  = 'R'   #: str: Domain of the gate parameters: 'N': natural numbers (incl. zero), 'R': floats. Parameters outside the domain are truncated into it.
    grad_method = 'A'   #: str: gradient computation method; 'A': angular, 'F': finite differences, None: may not be differentiated.
    grad_recipe = None  #: list[tuple[float]]: Gradient recipe for the 'A' method. One tuple for each parameter, (multiplier c_k, parameter shift s_k). None means (0.5, \pi/2) (the most common case).

    def __init__(self, *args, **kwargs):
        self.name  = self.__class__.__name__   #: str: name of the operation

        # extract the arguments
        if 'wires' in kwargs:
            params = args
            wires = kwargs['wires']
        else:
            params = args[:-1]
            wires = args[-1]

        if len(params) != self.n_params:
            raise ValueError("{}: wrong number of parameters. "
                             "{} parameters passed, {} expected.".format(self.name, params, self.n_params))

        self.params = list(params)  #: list[float, int, Variable]: operation parameters, both fixed and free

        # check the grad_recipe validity
        if self.grad_method == 'A':
            if self.grad_recipe is None:
                # default recipe for every parameter
                self.grad_recipe = [None] * self.n_params
            else:
                assert len(self.grad_recipe) == self.n_params, 'Gradient recipe must have one entry for each parameter!'
        else:
            assert self.grad_recipe is None, 'Gradient recipe is only used by the A method!'

        # apply the operation on the given wires
        if isinstance(wires, int):
            wires = [wires]

        if self.n_wires != 0 and len(wires) != self.n_wires:
            raise ValueError("{}: wrong number of wires. "
                             "{} wires requested, {} expected.".format(self.name, len(wires), self.n_wires))

        self.wires = wires  #: Sequence[int]: subsystems the operation acts on
        self.queue()

    def __str__(self):
        """Print the operation name and some information."""
        return self.name +': {} params, wires {}'.format(len(self.params), self.wires)

    def parameters(self):
        """Current parameter values.

        Fixed parameters are returned as is, free parameters represented by :class:`Variable` instances are replaced by their current numerical value.

        Returns:
          list[float]: parameter values
        """
        return [x.val if isinstance(x, Variable) else x for x in self.params]

    def queue(self):
        """Append the operation to a QNode queue."""
        if oq.QNode._current_context is None:
            raise oq.QuantumFunctionError("Quantum operations can only be used inside a qfunc.")
        else:
            oq.QNode._current_context._queue.append(self)



class Expectation(Operation):
    """A type of expectation value measurement supported by a device, and its properties.

    Expectation is used to describe Hermitian quantum observables.
    """
    n_params = 0
    grad_method = None
    grad_recipe = None

    def queue(self):
        """Append the expectation to a QNode queue."""
        if oq.QNode._current_context is None:
            raise oq.QuantumFunctionError("Quantum expectations can only be used inside a qfunc.")
        else:
            oq.QNode._current_context._observe.append(self)
