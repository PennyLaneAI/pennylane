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
Quantum circuit variables
=========================

**Module name:** :mod:`pennylane.variable`

.. currentmodule:: pennylane.variable

This module contains the :class:`Variable` class, which is used to track
and identify :class:`~pennylane.qnode.QNode` parameters.

The first time a QNode is evaluated (either by calling :meth:`~.QNode.evaluate`,
:meth:`~.QNode.__call__`, or :meth:`~.QNode.jacobian`), the :meth:`~.QNode.construct`
method is called, which performs a 'just-in-time' circuit construction
on the :mod:`~pennylane._device.Device`. As part of this construction, all arguments
and keyword arguments are wrapped in a `Variable` as follows:

* All positional arguments in ``*args``, including those with multiple dimensions, are
  flattened to a single list, and each element wrapped as a Variable instance,
  indexed by its position in the list.

  This allows PennyLane to inspect the shape and type of arguments
  the user wishes to pass. The list can then be unflattened back to the original
  shape of ``*args``.


* The same is done for each keyword argument in ``**kwargs``, the only
  difference being that the name of each contained Variable corresponds
  with the keyword name.

As a result, the device stores a list of operations and expectations, with all
free parameters stored as Variable instances.

.. note::
    The QNode can be differentiated with respect to positional arguments,
    but *not* with respect to keyword arguments. This makes keyword arguments
    a natural location for data placeholders.

.. important::
    If the user defines a keyword argument, then they always have to pass the
    corresponding variable as a keyword argument, otherwise it won't register.

For each successive QNode execution, the user-provided values for arguments and keyword
arguments are stored in the :attr:`Variable.free_param_values` list and the
:attr:`Variable.kwarg_values` dictionary respectively; these are
then returned by :meth:`Variable.val`, using its ``idx`` value, and, for
keyword arguments, its ``name``, to return the correct value to the operation.

.. note::
    The :meth:`Operation.parameters() <pennylane.operation.Operation.parameters>`
    property automates the process of unpacking the Variable value.
    The attribute :meth:`Variable.val` should not need to be accessed outside of advanced usage.


.. raw:: html

    <h3>Code details</h3>
"""
import logging
import copy

logging.getLogger()


class Variable:
    """A reference class to dynamically track and update circuit parameters.

    Represents a placeholder variable. This can either be a free quantum
    circuit parameter (with a non-fixed value) times an optional scalar multiplier,
    or a placeholder for data/other hard-coded data.

    Each time the circuit is executed, it is given a vector of
    parameter values, and a dictionary of keyword variable values.

    Variable is essentially an index into that vector.

    .. note:: Variables currently do not implement any arithmetic
        operations other than scalar multiplication.

    Args:
        idx  (int): parameter index >= 0
        name (str): name of the variable (optional)
    """
    # pylint: disable=too-few-public-methods
    free_param_values = None  #: array[float]: current free parameter values, set in :meth:`QNode.evaluate`
    kwarg_values = None #: dict: dictionary containing the keyword argument values, set in :meth:`QNode.evaluate`

    def __init__(self, idx=None, name=None):
        self.idx = idx    #: int: parameter index
        self.name = name  #: str: parameter name
        self.mult = 1     #: int, float: parameter scalar multiplier

    def __str__(self):
        temp = ' * {}'.format(self.mult) if self.mult != 1.0 else ''
        return 'Variable {}: name = {}, {}'.format(self.idx, self.name, temp)

    def __neg__(self):
        """Unary negation."""
        temp = copy.copy(self)
        temp.mult = -temp.mult
        return temp

    def __mul__(self, scalar):
        """Right multiplication by scalars."""
        temp = copy.copy(self)
        temp.mult *= scalar
        return temp

    __rmul__ = __mul__ # """Left multiplication by scalars."""

    @property
    def val(self):
        """Current numerical value of the Variable.

        Returns:
            float: current value of the Variable
        """
        # pylint: disable=unsubscriptable-object
        if self.name is None:
            # The variable is a placeholder for a positional argument
            value = self.free_param_values[self.idx] * self.mult
            return value

        # The variable is a placeholder for a keyword argument
        value = self.kwarg_values[self.name][self.idx] * self.mult
        return value
