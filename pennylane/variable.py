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
This module contains the :class:`Variable` class, which is used to track
and identify :class:`~pennylane.qnode.QNode` parameters.

Description
-----------

The first time a QNode is evaluated (either by calling :meth:`~.QNode.evaluate`,
:meth:`~.QNode.__call__`, or :meth:`~.QNode.jacobian`), the :meth:`~.QNode._construct`
method is called, which performs a 'just-in-time' circuit construction.
As part of this construction, all primary arguments
(and possibly also the auxiliary arguments) are wrapped in `Variable` instances as follows:

* The nested sequence of primary arguments is
  flattened to a single list, and each element wrapped in a `Variable` instance,
  indexed by its position in the list.
  The list can then be unflattened back to the original shape.

* The same is done for each auxiliary argument, the only
  difference being that the dict_key of each contained Variable corresponds
  with the argument name.

As a result, the circuit is described as a graph of operations and expectations, with all
the parameters stored as Variable instances.

.. note::
    The QNode can be differentiated with respect to primary arguments,
    but *not* with respect to auxiliary arguments. This makes auxiliary arguments
    a natural location for data placeholders.

For each successive QNode execution, the user-provided values for the primary and auxiliary
arguments are stored in :attr:`Variable.primary_arg_values` and
:attr:`Variable.auxiliary_arg_values` respectively; the values are
then returned by :meth:`Variable.val`, using the Variable's :attr:`idx` attribute, and, for
auxiliary arguments, its :attr:`dict_key`, to return the correct value to the operation.

.. note::
    The :meth:`Operation.parameters() <pennylane.operation.Operation.parameters>`
    property automates the process of unpacking the Variable value.
    The attribute :meth:`Variable.val` should not need to be accessed outside of advanced usage.
"""
import copy


class Variable:
    """A reference to dynamically track and update circuit parameters.

    Represents an atomic ("scalar") quantum circuit parameter (with a non-fixed value),
    or data placeholder.

    Each time the circuit is executed, it is given a vector of flattened primary argument values,
    and a dictionary mapping auxiliary argument names to vectors of their flattened values.
    Each element of these vectors corresponds to a Variable instance.
    :attr:`Variable.idx` is an index into the argument value vector.

    The Variable has an optional scalar multiplier for the argument it represents.

    .. note:: Variables currently do not implement any arithmetic
        operations other than scalar multiplication.

    Args:
        idx  (int): index into the value vector, >= 0
        name (str): structured name of the parameter
        dict_key (None, str): for auxiliary parameters the name of the base parameter, otherwise None
    """

    # pylint: disable=too-few-public-methods

    #: array[float]: current flattened primary parameter values, set in :meth:`.BaseQNode._set_variables`
    primary_arg_values = None

    #: dict[str->array[float]]: current flattened auxiliary parameter values, set in :meth:`.BaseQNode._set_variables`
    auxiliary_arg_values = None

    def __init__(self, idx, name=None, dict_key=None):
        self.idx = idx  #: int: parameter index
        self.name = name  #: str: parameter structured name
        self.dict_key = dict_key
        """str, None: for auxiliary parameters the key for the auxiliary_arg_values dict"""
        self.mult = 1  #: int, float: parameter scalar multiplier

    def __repr__(self):
        temp = " * {}".format(self.mult) if self.mult != 1.0 else ""
        return "<Variable({}:{}{})>".format(self.name, self.idx, temp)

    def __str__(self):
        temp = ", * {}".format(self.mult) if self.mult != 1.0 else ""
        return "Variable: name = {}, idx = {}{}".format(self.name, self.idx, temp)

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False

        return (
            self.name == other.name
            and self.idx == other.idx
            and self.is_auxiliary == other.is_auxiliary
            and self.mult == other.mult
        )

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

    def __truediv__(self, scalar):
        """Right division by scalars. Left division is not allowed."""
        temp = copy.copy(self)
        temp.mult /= scalar
        return temp

    __rmul__ = __mul__  # Left multiplication by scalars.

    @property
    def is_auxiliary(self):
        """Type of the parameter the VariableRef represents.

        Returns:
            bool: True iff the VariableRef represents (an element of) an auxiliary parameter
        """
        return self.dict_key is not None

    @property
    def val(self):
        """Current numerical value of the Variable.

        Returns:
            float: current value of the Variable
        """
        # pylint: disable=unsubscriptable-object
        if not self.is_auxiliary:
            # The variable is a placeholder for a primary argument
            return Variable.primary_arg_values[self.idx] * self.mult

        # The variable is a placeholder for an auxiliary argument
        values = Variable.auxiliary_arg_values[self.dict_key]
        return values[self.idx] * self.mult

    def render(self, show_name_only=False):
        """String representation of the Variable for CircuitDrawer.

        Args:
            show_name_only (bool, optional): Render the name instead of the value.

        Returns:
            str: string representation of the VariableRef
        """
        if not show_name_only:
            return str(round(self.val, 3))

        if self.mult != 1:
            return "{}*{}".format(str(round(self.mult, 3)), self.name)

        return self.name
