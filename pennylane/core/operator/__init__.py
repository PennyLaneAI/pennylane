# Copyright 2018-2026 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access

r"""
This module contains the abstract base classes for defining PennyLane
operations and observables.

.. warning::

    Unless you are a PennyLane or plugin developer, you likely do not need
    to use these classes directly.

    See the :doc:`main operations page <../introduction/operations>` for
    details on available operations and observables.

Description
-----------

Qubit Operations
~~~~~~~~~~~~~~~~
The :class:`Operator` class serves as a base class for operators,
and is inherited by the
:class:`Operation` class. These classes are subclassed to implement quantum operations
and measure observables in PennyLane.

* Each :class:`~.Operator` subclass represents a general type of
  map between physical states. Each instance of these subclasses
  represents either

  - an application of the operator or
  - an instruction to measure and return the respective result.

  Operators act on a sequence of wires (subsystems) using given parameter values.

* Each :class:`~.Operation` subclass represents a type of quantum operation,
  for example a unitary quantum gate. Each instance of these subclasses
  represents an application of the operation with given parameter values to
  a given sequence of wires (subsystems).


Differentiation
^^^^^^^^^^^^^^^

In general, an :class:`Operation` is differentiable (at least using the finite-difference
method) with respect to a parameter iff

* the domain of that parameter is continuous.

For an :class:`Operation` to be differentiable with respect to a parameter using the
analytic method of differentiation, it must satisfy an additional constraint:

* the parameter domain must be real.

.. note::

    These conditions are *not* sufficient for analytic differentiation. For example,
    CV gates must also define a matrix representing their Heisenberg linear
    transformation on the quadrature operators.

CV Operation base classes
~~~~~~~~~~~~~~~~~~~~~~~~~

Due to additional requirements, continuous-variable (CV) operations must subclass the
:class:`~.CVOperation` or :class:`~.CVObservable` classes instead of :class:`~.Operation`.

Differentiation
^^^^^^^^^^^^^^^

To enable gradient computation using the analytic method for Gaussian CV operations, in addition, you need to
provide the static class method :meth:`~.CV._heisenberg_rep` that returns the Heisenberg representation of
the operation given its list of parameters, namely:

* For Gaussian CV Operations this method should return the matrix of the linear transformation carried out by the
  operation on the vector of quadrature operators :math:`\mathbf{r}` for the given parameter
  values.

* For Gaussian CV Observables this method should return a real vector (first-order observables)
  or symmetric matrix (second-order observables) of coefficients of the quadrature
  operators :math:`\x` and :math:`\p`.

PennyLane uses the convention :math:`\mathbf{r} = (\I, \x, \p)` for single-mode operations and observables
and :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)` for multi-mode operations and observables.

.. note::
    Non-Gaussian CV operations and observables are currently only supported via
    the finite-difference method of gradient computation.

Contents
--------

.. currentmodule:: pennylane.core.operator

Operator Types
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.core.operator

.. autosummary::
    :toctree: api

    ~Operator
    ~Operation
    ~CV
    ~CVObservable
    ~CVOperation
    ~Channel
    ~StatePrepBase

.. currentmodule:: pennylane.core.operator

.. inheritance-diagram:: Operator Operation Channel CV CVObservable CVOperation StatePrepBase
    :parts: 1


Boolean Functions
~~~~~~~~~~~~~~~~~

:class:`~.BooleanFn`'s are functions of a single object that return ``True`` or ``False``.
The ``operation`` module provides the following:

.. currentmodule:: pennylane.operation

.. autosummary::
    :toctree: api

    ~is_trainable

Other
~~~~~

.. currentmodule:: pennylane.operation

.. autosummary::
    :toctree: api

    ~operation_derivative

.. currentmodule:: pennylane

PennyLane also provides a function for checking the consistency and correctness of an operator instance.

.. autosummary::
    :toctree: api

    ~ops.functions.assert_valid

Operation attributes
~~~~~~~~~~~~~~~~~~~~

PennyLane contains a mechanism for storing lists of operations with similar
attributes and behaviour (for example, those that are their own inverses).
The attributes below are already included, and are used primarily for the
purpose of compilation transforms. New attributes can be added by instantiating
new :class:`~pennylane.ops.qubit.attributes.Attribute` objects. Please note that
these objects are located in ``pennylane.ops.qubit.attributes``, not ``pennylane.operation``.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~ops.qubit.attributes.Attribute
    ~ops.qubit.attributes.composable_rotations
    ~ops.qubit.attributes.diagonal_in_z_basis
    ~ops.qubit.attributes.has_unitary_generator
    ~ops.qubit.attributes.self_inverses
    ~ops.qubit.attributes.supports_broadcasting
    ~ops.qubit.attributes.symmetric_over_all_wires
    ~ops.qubit.attributes.symmetric_over_control_wires

"""

from .base import Operator, Operation
from .channel import Channel
from .cv import CV, CVObservable, CVOperation
from .state_prep import StatePrepBase

__all__ = ["Operator", "Operation", "Channel", "CV", "CVObservable", "CVOperation", "StatePrepBase"]
