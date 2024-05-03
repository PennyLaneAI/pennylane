# Copyright 2018-2021 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access, no-member
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
and is inherited by both the :class:`Observable` class and the
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

* Each  :class:`~.Observable` subclass represents a type of physical observable.
  Each instance of these subclasses represents an instruction to measure and
  return the respective result for the given parameter values on a
  sequence of wires (subsystems).

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
:class:`~.CVOperation` or :class:`~.CVObservable` classes instead of :class:`~.Operation`
and :class:`~.Observable`.

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

.. currentmodule:: pennylane.operation

Operator Types
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.operation

.. autosummary::
    :toctree: api

    ~Operator
    ~Operation
    ~Observable
    ~CV
    ~CVObservable
    ~CVOperation
    ~Channel
    ~Tensor
    ~StatePrepBase

.. currentmodule:: pennylane.operation

.. inheritance-diagram:: Operator Operation Observable Channel CV CVObservable CVOperation Tensor StatePrepBase
    :parts: 1

Errors
~~~~~~

When an :class:`~.Operator` method is undefined, it raises a error type that depends
on the method that is undefined.

.. currentmodule:: pennylane.operation

.. autosummary::
    :toctree: api

    ~OperatorPropertyUndefined
    ~AdjointUndefinedError
    ~DecompositionUndefinedError
    ~DiagGatesUndefinedError
    ~EigvalsUndefinedError
    ~GeneratorUndefinedError
    ~MatrixUndefinedError
    ~ParameterFrequenciesUndefinedError
    ~PowUndefinedError
    ~SparseMatrixUndefinedError
    ~TermsUndefinedError

Boolean Functions
~~~~~~~~~~~~~~~~~

:class:`~.BooleanFn`'s are functions of a single object that return ``True`` or ``False``.
The ``operation`` module provides the following:

.. currentmodule:: pennylane.operation

.. autosummary::
    :toctree: api

    ~defines_diagonalizing_gates
    ~gen_is_multi_term_hamiltonian
    ~has_gen
    ~has_grad_method
    ~has_multipar
    ~has_nopar
    ~has_unitary_gen
    ~is_measurement
    ~is_trainable
    ~not_tape

Enabling New Arithmetic Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is in the process of replacing :class:`~pennylane.Hamiltonian` and :class:`~.Tensor`
with newer, more general arithmetic operators. These consist of :class:`~pennylane.ops.op_math.Prod`,
:class:`~pennylane.ops.op_math.Sum` and :class:`~pennylane.ops.op_math.SProd`. By default, using dunder
methods (eg. ``+``, ``-``, ``@``, ``*``) to combine operators with scalars or other operators will
create the aforementioned newer operators. To toggle the dunders to return the older arithmetic operators,
the ``operation`` module provides the following helper functions:

.. currentmodule:: pennylane.operation

.. autosummary::
    :toctree: api

    ~enable_new_opmath
    ~disable_new_opmath
    ~active_new_opmath
    ~convert_to_opmath
    ~convert_to_legacy_H

Other
~~~~~

.. currentmodule:: pennylane.operation

.. autosummary::
    :toctree: api

    ~operation_derivative
    ~WiresEnum
    ~AllWires
    ~AnyWires

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
# pylint:disable=access-member-before-definition,global-statement
import abc
import copy
import functools
import itertools
import warnings
from enum import IntEnum
from typing import List, Tuple
from contextlib import contextmanager

import numpy as np
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix, csr_matrix, eye, kron

import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .utils import pauli_eigs
from .pytrees import register_pytree

# =============================================================================
# Errors
# =============================================================================

SUPPORTED_INTERFACES = {"numpy", "scipy", "autograd", "torch", "tensorflow", "jax"}
__use_new_opmath = True
_UNSET_BATCH_SIZE = -1  # indicates that the (lazy) batch size has not yet been accessed/computed


class OperatorPropertyUndefined(Exception):
    """Generic exception to be used for undefined
    Operator properties or methods."""


class DecompositionUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's representation as a decomposition is undefined."""


class TermsUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's representation as a linear combination is undefined."""


class MatrixUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's matrix representation is undefined."""


class SparseMatrixUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's sparse matrix representation is undefined."""


class EigvalsUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's eigenvalues are undefined."""


class DiagGatesUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's diagonalizing gates are undefined."""


class AdjointUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's adjoint version is undefined."""


class PowUndefinedError(OperatorPropertyUndefined):
    """Raised when an Operator's power is undefined."""


class GeneratorUndefinedError(OperatorPropertyUndefined):
    """Exception used to indicate that an operator
    does not have a generator"""


class ParameterFrequenciesUndefinedError(OperatorPropertyUndefined):
    """Exception used to indicate that an operator
    does not have parameter_frequencies"""


# =============================================================================
# Wire types
# =============================================================================


class WiresEnum(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on"""

    AnyWires = -1
    AllWires = -2


AllWires = WiresEnum.AllWires
"""IntEnum: An enumeration which represents all wires in the
subsystem. It is equivalent to an integer with value 0."""

AnyWires = WiresEnum.AnyWires
"""IntEnum: An enumeration which represents any wires in the
subsystem. It is equivalent to an integer with value -1."""


# =============================================================================
# Class property
# =============================================================================


class ClassPropertyDescriptor:  # pragma: no cover
    """Allows a class property to be defined"""

    # pylint: disable=too-few-public-methods,too-many-public-methods
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        """Set the function as a class method, and store as an attribute."""
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """The class property decorator"""
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


# =============================================================================
# Base Operator class
# =============================================================================


def _process_data(op):
    def _mod_and_round(x, mod_val):
        if mod_val is None:
            return x
        return qml.math.round(qml.math.real(x) % mod_val, 10)

    # Use qml.math.real to take the real part. We may get complex inputs for
    # example when differentiating holomorphic functions with JAX: a complex
    # valued QNode (one that returns qml.state) requires complex typed inputs.
    if op.name in ("RX", "RY", "RZ", "PhaseShift", "Rot"):
        mod_val = 2 * np.pi
    else:
        mod_val = None

    return str([id(d) if qml.math.is_abstract(d) else _mod_and_round(d, mod_val) for d in op.data])


class Operator(abc.ABC):
    r"""Base class representing quantum operators.

    Operators are uniquely defined by their name, the wires they act on, their (trainable) parameters,
    and their (non-trainable) hyperparameters. The trainable parameters
    can be tensors of any supported auto-differentiation framework.

    An operator can define any of the following representations:

    * Representation as a **matrix** (:meth:`.Operator.matrix`), as specified by a
      global wire order that tells us where the wires are found on a register.

    * Representation as a **sparse matrix** (:meth:`.Operator.sparse_matrix`). Currently, this
      is a SciPy CSR matrix format.

    * Representation via the **eigenvalue decomposition** specified by eigenvalues
      (:meth:`.Operator.eigvals`) and diagonalizing gates (:meth:`.Operator.diagonalizing_gates`).

    * Representation as a **product of operators** (:meth:`.Operator.decomposition`).

    * Representation as a **linear combination of operators** (:meth:`.Operator.terms`).

    * Representation by a **generator** via :math:`e^{G}` (:meth:`.Operator.generator`).

    Each representation method comes with a static method prefixed by ``compute_``, which
    takes the signature ``(*parameters, **hyperparameters)`` (for numerical representations that do not need
    to know about wire labels) or ``(*parameters, wires, **hyperparameters)``, where ``parameters``, ``wires``, and
    ``hyperparameters`` are the respective attributes of the operator class.

    Args:
        *params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified

    **Example**

    A custom operator can be created by inheriting from :class:`~.Operator` or one of its subclasses.

    The following is an example for a custom gate that inherits from the :class:`~.Operation` subclass.
    It acts by potentially flipping a qubit and rotating another qubit.
    The custom operator defines a decomposition, which the devices can use (since it is unlikely that a device
    knows a native implementation for ``FlipAndRotate``). It also defines an adjoint operator.

    .. code-block:: python

        import pennylane as qml


        class FlipAndRotate(qml.operation.Operation):

            # Define how many wires the operator acts on in total.
            # In our case this may be one or two, which is why we
            # use the AnyWires Enumeration to indicate a variable number.
            num_wires = qml.operation.AnyWires

            # This attribute tells PennyLane what differentiation method to use. Here
            # we request parameter-shift (or "analytic") differentiation.
            grad_method = "A"

            def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False, id=None):

                # checking the inputs --------------

                if do_flip and wire_flip is None:
                    raise ValueError("Expected a wire to flip; got None.")

                #------------------------------------

                # do_flip is not trainable but influences the action of the operator,
                # which is why we define it to be a hyperparameter
                self._hyperparameters = {
                    "do_flip": do_flip
                }

                # we extract all wires that the operator acts on,
                # relying on the Wire class arithmetic
                all_wires = qml.wires.Wires(wire_rot) + qml.wires.Wires(wire_flip)

                # The parent class expects all trainable parameters to be fed as positional
                # arguments, and all wires acted on fed as a keyword argument.
                # The id keyword argument allows users to give their instance a custom name.
                super().__init__(angle, wires=all_wires, id=id)

            @property
            def num_params(self):
                # if it is known before creation, define the number of parameters to expect here,
                # which makes sure an error is raised if the wrong number was passed. The angle
                # parameter is the only trainable parameter of the operation
                return 1

            @property
            def ndim_params(self):
                # if it is known before creation, define the number of dimensions each parameter
                # is expected to have. This makes sure to raise an error if a wrongly-shaped
                # parameter was passed. The angle parameter is expected to be a scalar
                return (0,)

            @staticmethod
            def compute_decomposition(angle, wires, do_flip):  # pylint: disable=arguments-differ
                # Overwriting this method defines the decomposition of the new gate, as it is
                # called by Operator.decomposition().
                # The general signature of this function is (*parameters, wires, **hyperparameters).
                op_list = []
                if do_flip:
                    op_list.append(qml.X(wires[1]))
                op_list.append(qml.RX(angle, wires=wires[0]))
                return op_list

            def adjoint(self):
                # the adjoint operator of this gate simply negates the angle
                return FlipAndRotate(-self.parameters[0], self.wires[0], self.wires[1], do_flip=self.hyperparameters["do_flip"])

    We can use the operation as follows:

    .. code-block:: python

        from pennylane import numpy as np

        dev = qml.device("default.qubit", wires=["q1", "q2", "q3"])

        @qml.qnode(dev)
        def circuit(angle):
            FlipAndRotate(angle, wire_rot="q1", wire_flip="q1")
            return qml.expval(qml.Z("q1"))

    >>> a = np.array(3.14)
    >>> circuit(a)
    tensor(-0.99999873, requires_grad=True)

    .. details::
        :title: Serialization and Pytree format
        :href: serialization

        PennyLane operations are automatically registered as `Pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_ .

        For most operators, this process will happen automatically without need for custom implementations.

        Customization of this process must occur if:

        * The data and hyperparameters are insufficient to reproduce the original operation via its initialization
        * The hyperparameters contain a non-hashable component, such as a list or dictionary.

        Some examples include arithmetic operators, like :class:`~.Adjoint` or :class:`~.Sum`, or templates that
        perform preprocessing during initialization.

        See the ``Operator._flatten`` and ``Operator._unflatten`` methods for more information.

        >>> op = qml.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten()
        ((1.2,), (<Wires = [0, 1]>, (('pauli_word', 'XY'),)))
        >>> qml.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])


    .. details::
        :title: Parameter broadcasting
        :href: parameter-broadcasting

        Many quantum functions are executed repeatedly at different parameters, which
        can be done with parameter broadcasting. For usage details and examples see the
        :class:`~.pennylane.QNode` documentation.

        In order to support parameter broadcasting with an operator class,
        the following steps are necessary:

        #. Define the class attribute ``ndim_params``, a tuple that indicates
           the expected number of dimensions for each operator argument
           *without broadcasting*. For example, ``FlipAndRotate``
           above has ``ndim_params = (0,)`` for a single scalar argument.
           An operator taking a matrix argument and a scalar would have ``ndim_params = (2, 0)``.
           Note that ``ndim_params`` does *not require the size* of the axes.
        #. Make the representations of the operator broadcasting-compatible. Typically, one or
           multiple of the methods ``compute_matrix``, ``compute_eigvals`` and
           ``compute_decomposition`` are defined by an operator, and these need to work with
           the original input and output as well as with broadcasted inputs and outputs
           that have an additional, leading axis. See below for an example.
        #. Make sure that validation within the above representation methods and
           ``__init__``---if it is overwritten by the operator class---allow
           for broadcasted inputs. For custom operators this usually is a minor
           step or not necessary at all.
        #. For proper registration, add the name of the operator to
           :obj:`~.pennylane.ops.qubit.attributes.supports_broadcasting` in the file
           ``pennylane/ops/qubit/attributes.py``.
        #. Make sure that the operator's ``_check_batching`` method is called in all
           places required. This is typically done automatically but needs to be assured.
           See further below for details.

        **Examples**

        Consider an operator with the same matrix as ``qml.RX``. A basic variant of
        ``compute_matrix`` (which will not be compatible with all autodifferentiation
        frameworks or backpropagation) is

        .. code-block:: python

            @staticmethod
            def compute_matrix(theta):
                '''Broadcasting axis ends up in the wrong position.'''
                c = qml.math.cos(theta / 2)
                s = qml.math.sin(theta / 2)
                return qml.math.array([[c, -1j * s], [-1j * s, c]])

        If we passed a broadcasted argument ``theta`` of shape ``(batch_size,)`` to this method,
        which would have one instead of zero dimensions, ``cos`` and ``sin`` would correctly
        be applied elementwise.
        We would also obtain the correct matrix with shape ``(2, 2, batch_size)``.
        However, the broadcasting axis needs to be the *first* axis by convention, so that we need
        to move the broadcasting axis--if it exists--to the front before returning the matrix:

        .. code-block:: python

            @staticmethod
            def compute_matrix(theta):
                '''Broadcasting axis ends up in the correct leading position.'''
                c = qml.math.cos(theta / 2)
                s = qml.math.sin(theta / 2)
                mat = qml.math.array([[c, -1j * s], [-1j * s, c]])
                # Check whether the input has a broadcasting axis
                if qml.math.ndim(theta)==1:
                    # Move the broadcasting axis to the first position
                    return qml.math.moveaxis(mat, 2, 0)
                return mat

        Adapting ``compute_eigvals`` to broadcasting looks similar.

        Usually no major changes are required for ``compute_decomposition``, but we need
        to take care of the correct mapping of input arguments to the operators in the
        decomposition. As an example, consider the operator that represents a layer of
        ``RX`` rotations with individual angles for each rotation. Without broadcasting,
        it takes one onedimensional array, i.e. ``ndim_params=(1,)``.
        Its decomposition, which is a convenient way to support this custom operation
        on all devices that implement ``RX``, might look like this:

        .. code-block:: python

            @staticmethod
            def compute_decomposition(theta, wires):
                '''Iterate over the first axis of theta.'''
                decomp_ops = [qml.RX(x, wires=w) for x, w in zip(theta, wires)]
                return decomp_ops

        If ``theta`` is a broadcasted argument, its first axis is the broadcasting
        axis and we would like to iterate over the *second* axis within the ``for``
        loop instead. This is easily achieved by adding a transposition of ``theta``
        that switches the axes in this case. Conveniently this does not have any
        effect in the non-broadcasted case, so that we do not need to handle two
        cases separately.

        .. code-block:: python

            @staticmethod
            def compute_decomposition(theta, wires):
                '''Iterate over the last axis of theta, which is also the first axis
                or the second axis without and with broadcasting, respectively.'''
                decomp_ops = [qml.RX(x, wires=w) for x, w in zip(qml.math.T(theta), wires)]
                return decomp_ops

        **The ``_check_batching`` method**

        Each operator determines whether it is used with a batch of parameters within
        the ``_check_batching`` method, by comparing the shape of the input data to
        the expected shape. Therefore, it is necessary to call ``_check_batching`` on
        any new input parameters passed to the operator. By default, any class inheriting
        from :class:`~.operation.Operator` will do so the first time its
        ``batch_size`` property is accessed.

        ``_check_batching`` modifies the following instance attributes:

        - ``_ndim_params``: The number of dimensions of the parameters passed to
          ``_check_batching``. For an ``Operator`` that does _not_ set the ``ndim_params``
          attribute, ``_ndim_params`` is used as a surrogate, interpreting any parameters
          as "not broadcasted". This attribute should be understood as temporary and likely
          should not be used in other contexts.

        - ``_batch_size``: If the ``Operator`` is broadcasted: The batch size/size of the
          broadcasting axis. If it is not broadcasted: ``None``. An ``Operator`` that does
          not support broadcasting will report to not be broadcasted independently of the
          input.

        These two properties are defined lazily, and accessing the public version of either
        one of them (in other words, without the leading underscore) for the first time will
        trigger a call to ``_check_batching``, which validates and sets these properties.
    """

    # pylint: disable=too-many-public-methods, too-many-instance-attributes

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * ps1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    def __init_subclass__(cls, **_):
        register_pytree(cls, cls._flatten, cls._unflatten)

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.data = copy.copy(self.data)
        for attr, value in vars(self).items():
            if attr != "data":
                setattr(copied_op, attr, value)

        return copied_op

    def __deepcopy__(self, memo):
        copied_op = object.__new__(type(self))

        # The memo dict maps object ID to object, and is required by
        # the deepcopy function to keep track of objects it has already
        # deep copied.
        memo[id(self)] = copied_op

        for attribute, value in self.__dict__.items():
            if attribute == "data":
                # Shallow copy the list of parameters. We avoid a deep copy
                # here, since PyTorch does not support deep copying of tensors
                # within a differentiable computation.
                copied_op.data = copy.copy(value)
            else:
                # Deep copy everything else.
                setattr(copied_op, attribute, copy.deepcopy(value, memo))
        return copied_op

    @property
    def hash(self):
        """int: Integer hash that uniquely represents the operator."""
        return hash(
            (
                str(self.name),
                tuple(self.wires.tolist()),
                str(self.hyperparameters.values()),
                _process_data(self),
            )
        )

    def __eq__(self, other):
        return qml.equal(self, other)

    def __hash__(self):
        return self.hash

    @staticmethod
    def compute_matrix(*params, **hyperparams) -> TensorLike:  # pylint:disable=unused-argument
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`.Operator.matrix` and :func:`qml.matrix() <pennylane.matrix>`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            tensor_like: matrix representation
        """
        raise MatrixUndefinedError

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_matrix(cls):
        r"""Bool: Whether or not the Operator returns a defined matrix.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.compute_matrix != Operator.compute_matrix or cls.matrix != Operator.matrix

    def matrix(self, wire_order=None) -> TensorLike:
        r"""Representation of the operator as a matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        If the matrix depends on trainable parameters, the result
        will be cast in the same autodifferentiation framework as the parameters.

        A ``MatrixUndefinedError`` is raised if the matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator.compute_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires

        Returns:
            tensor_like: matrix representation
        """
        canonical_matrix = self.compute_matrix(*self.parameters, **self.hyperparameters)

        if (
            wire_order is None
            or self.wires == Wires(wire_order)
            or (
                self.name in qml.ops.qubit.attributes.symmetric_over_all_wires
                and set(self.wires) == set(wire_order)
            )
        ):
            return canonical_matrix

        return expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    @staticmethod
    def compute_sparse_matrix(
        *params, **hyperparams
    ) -> csr_matrix:  # pylint:disable=unused-argument
        r"""Representation of the operator as a sparse matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator.sparse_matrix`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters``
                attribute

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation
        """
        raise SparseMatrixUndefinedError

    def sparse_matrix(self, wire_order=None) -> csr_matrix:
        r"""Representation of the operator as a sparse matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        A ``SparseMatrixUndefinedError`` is raised if the sparse matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator.compute_sparse_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation

        """
        canonical_sparse_matrix = self.compute_sparse_matrix(
            *self.parameters, **self.hyperparameters
        )

        return expand_matrix(canonical_sparse_matrix, wires=self.wires, wire_order=wire_order)

    @staticmethod
    def compute_eigvals(*params, **hyperparams) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`Operator.eigvals() <.eigvals>` and :func:`qml.eigvals() <pennylane.eigvals>`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            tensor_like: eigenvalues
        """
        raise EigvalsUndefinedError

    def eigvals(self):
        r"""Eigenvalues of the operator in the computational basis.

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`, the operator
        can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. note::
            When eigenvalues are not explicitly defined, they are computed automatically from the matrix representation.
            Currently, this computation is *not* differentiable.

        A ``EigvalsUndefinedError`` is raised if the eigenvalues have not been defined and cannot be
        inferred from the matrix representation.

        .. seealso:: :meth:`~.Operator.compute_eigvals`

        Returns:
            tensor_like: eigenvalues
        """

        try:
            return self.compute_eigvals(*self.parameters, **self.hyperparameters)
        except EigvalsUndefinedError as e:
            # By default, compute the eigenvalues from the matrix representation if one is defined.
            if self.has_matrix:  # pylint: disable=using-constant-test
                return qml.math.linalg.eigvals(self.matrix())
            raise EigvalsUndefinedError from e

    def terms(self):  # pylint: disable=no-self-use
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`
        """
        raise TermsUndefinedError

    num_wires = AnyWires
    """Number of wires the operator acts on."""

    @property
    def name(self):
        """String for the name of the operator."""
        return self._name

    @property
    def id(self):
        """Custom string to label a specific operator instance."""
        return self._id

    @name.setter
    def name(self, value):
        self._name = value

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> op = qml.RX(1.23456, wires=0)
        >>> op.label()
        "RX"
        >>> op.label(base_label="my_label")
        "my_label"
        >>> op = qml.RX(1.23456, wires=0, id="test_data")
        >>> op.label()
        "RX("test_data")"
        >>> op.label(decimals=2)
        "RX\n(1.23,"test_data")"
        >>> op.label(base_label="my_label")
        "my_label("test_data")"
        >>> op.label(decimals=2, base_label="my_label")
        "my_label\n(1.23,"test_data")"

        If the operation has a matrix-valued parameter and a cache dictionary is provided,
        unique matrices will be cached in the ``'matrices'`` key list. The label will contain
        the index of the matrix in the ``'matrices'`` list.

        >>> op2 = qml.QubitUnitary(np.eye(2), wires=0)
        >>> cache = {'matrices': []}
        >>> op2.label(cache=cache)
        'U(M0)'
        >>> cache['matrices']
        [tensor([[1., 0.],
         [0., 1.]], requires_grad=True)]
        >>> op3 = qml.QubitUnitary(np.eye(4), wires=(0,1))
        >>> op3.label(cache=cache)
        'U(M1)'
        >>> cache['matrices']
        [tensor([[1., 0.],
                [0., 1.]], requires_grad=True),
        tensor([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.]], requires_grad=True)]

        """
        op_label = base_label or self.__class__.__name__

        if self.num_params == 0:
            return op_label if self._id is None else f'{op_label}("{self._id}")'

        params = self.parameters

        if len(qml.math.shape(params[0])) != 0:
            # assume that if the first parameter is matrix-valued, there is only a single parameter
            # this holds true for all current operations and templates unless parameter broadcasting
            # is used
            # TODO[dwierichs]: Implement a proper label for broadcasted operators
            if (
                cache is None
                or not isinstance(cache.get("matrices", None), list)
                or len(params) != 1
            ):
                return op_label if self._id is None else f'{op_label}("{self._id}")'

            for i, mat in enumerate(cache["matrices"]):
                if qml.math.shape(params[0]) == qml.math.shape(mat) and qml.math.allclose(
                    params[0], mat
                ):
                    return (
                        f"{op_label}(M{i})"
                        if self._id is None
                        else f'{op_label}(M{i},"{self._id}")'
                    )

            # matrix not in cache
            mat_num = len(cache["matrices"])
            cache["matrices"].append(params[0])
            return (
                f"{op_label}(M{mat_num})"
                if self._id is None
                else f'{op_label}(M{mat_num},"{self._id}")'
            )

        if decimals is None:
            return op_label if self._id is None else f'{op_label}("{self._id}")'

        def _format(x):
            try:
                return format(qml.math.toarray(x), f".{decimals}f")
            except ValueError:
                # If the parameter can't be displayed as a float
                return format(x)

        param_string = ",\n".join(_format(p) for p in params)

        return (
            f"{op_label}\n({param_string})"
            if self._id is None
            else f'{op_label}\n({param_string},"{self._id}")'
        )

    def __init__(self, *params, wires=None, id=None):
        # pylint: disable=too-many-branches

        self._name = self.__class__.__name__  #: str: name of the operator
        self._id = id
        self.queue_idx = None  #: int, None: index of the Operator in the circuit queue, or None if not in a queue
        self._pauli_rep = None  # Union[PauliSentence, None]: Representation of the operator as a pauli sentence, if applicable

        wires_from_args = False
        if wires is None:
            try:
                wires = params[-1]
                params = params[:-1]
                wires_from_args = True
            except IndexError as err:
                raise ValueError(
                    f"Must specify the wires that {type(self).__name__} acts on"
                ) from err

        self._num_params = len(params)

        # Check if the expected number of parameters coincides with the one received.
        # This is always true for the default `Operator.num_params` property, but
        # subclasses may overwrite it to define a fixed expected value.
        if len(params) != self.num_params:
            if wires_from_args and len(params) == (self.num_params - 1):
                raise ValueError(f"Must specify the wires that {type(self).__name__} acts on")
            raise ValueError(
                f"{self.name}: wrong number of parameters. "
                f"{len(params)} parameters passed, {self.num_params} expected."
            )

        self._wires = wires if isinstance(wires, Wires) else Wires(wires)

        # check that the number of wires given corresponds to required number
        if self.num_wires in {AllWires, AnyWires}:
            if (
                not isinstance(
                    self,
                    (
                        qml.Barrier,
                        qml.Snapshot,
                        qml.ops.Hamiltonian,
                        qml.ops.LinearCombination,
                        qml.GlobalPhase,
                        qml.Identity,
                    ),
                )
                and len(qml.wires.Wires(wires)) == 0
            ):
                raise ValueError(
                    f"{self.name}: wrong number of wires. " f"At least one wire has to be given."
                )

        elif len(self._wires) != self.num_wires:
            raise ValueError(
                f"{self.name}: wrong number of wires. "
                f"{len(self._wires)} wires given, {self.num_wires} expected."
            )

        self._batch_size = _UNSET_BATCH_SIZE
        self._ndim_params = _UNSET_BATCH_SIZE

        self.data = tuple(np.array(p) if isinstance(p, (list, tuple)) else p for p in params)

        self.queue()

    def _check_batching(self):
        """Check if the expected numbers of dimensions of parameters coincides with the
        ones received and sets the ``_batch_size`` attribute.

        The check always passes and sets the ``_batch_size`` to ``None`` for the default
        ``Operator.ndim_params`` property but subclasses may overwrite it to define fixed
        expected numbers of dimensions, allowing to infer a batch size.
        """
        self._batch_size = None
        params = self.data

        try:
            ndims = tuple(qml.math.ndim(p) for p in params)
        except ValueError as e:
            # TODO:[dwierichs] When using tf.function with an input_signature that contains
            # an unknown-shaped input, ndim() will not be able to determine the number of
            # dimensions because they are not specified yet. Failing example: Let `fun` be
            # a single-parameter QNode.
            # `tf.function(fun, input_signature=(tf.TensorSpec(shape=None, dtype=tf.float32),))`
            # There might be a way to support batching nonetheless, which remains to be
            # investigated. For now, the batch_size is left to be `None` when instantiating
            # an operation with abstract parameters that make `qml.math.ndim` fail.
            if any(qml.math.is_abstract(p) for p in params):
                self._batch_size = None
                self._ndim_params = (0,) * len(params)
                return
            raise e  # pragma: no cover

        if any(len(qml.math.shape(p)) >= 1 and qml.math.shape(p)[0] is None for p in params):
            # if the batch dimension is unknown, then skip the validation
            # this happens when a tensor with a partially known shape is passed, e.g. (None, 12),
            # typically during compilation of a function decorated with jax.jit or tf.function
            return

        self._ndim_params = ndims
        if ndims != self.ndim_params:
            ndims_matches = [
                (ndim == exp_ndim, ndim == exp_ndim + 1)
                for ndim, exp_ndim in zip(ndims, self.ndim_params)
            ]
            if not all(correct or batched for correct, batched in ndims_matches):
                raise ValueError(
                    f"{self.name}: wrong number(s) of dimensions in parameters. "
                    f"Parameters with ndims {ndims} passed, {self.ndim_params} expected."
                )

            first_dims = [
                qml.math.shape(p)[0] for (_, batched), p in zip(ndims_matches, params) if batched
            ]
            if not qml.math.allclose(first_dims, first_dims[0]):
                raise ValueError(
                    "Broadcasting was attempted but the broadcasted dimensions "
                    f"do not match: {first_dims}."
                )
            self._batch_size = first_dims[0]

    def __repr__(self):
        """Constructor-call-like representation."""
        if self.parameters:
            params = ", ".join([repr(p) for p in self.parameters])
            return f"{self.name}({params}, wires={self.wires.tolist()})"
        return f"{self.name}(wires={self.wires.tolist()})"

    @property
    def num_params(self):
        """Number of trainable parameters that the operator depends on.

        By default, this property returns as many parameters as were used for the
        operator creation. If the number of parameters for an operator subclass is fixed,
        this property can be overwritten to return the fixed value.

        Returns:
            int: number of parameters
        """
        return self._num_params

    @property
    def ndim_params(self):
        """Number of dimensions per trainable parameter of the operator.

        By default, this property returns the numbers of dimensions of the parameters used
        for the operator creation. If the parameter sizes for an operator subclass are fixed,
        this property can be overwritten to return the fixed value.

        Returns:
            tuple: Number of dimensions for each trainable parameter.
        """
        if self._batch_size is _UNSET_BATCH_SIZE:
            self._check_batching()
        return self._ndim_params

    @property
    def batch_size(self):
        r"""Batch size of the operator if it is used with broadcasted parameters.

        The ``batch_size`` is determined based on ``ndim_params`` and the provided parameters
        for the operator. If (some of) the latter have an additional dimension, and this
        dimension has the same size for all parameters, its size is the batch size of the
        operator. If no parameter has an additional dimension, the batch size is ``None``.

        Returns:
            int or None: Size of the parameter broadcasting dimension if present, else ``None``.
        """
        if self._batch_size is _UNSET_BATCH_SIZE:
            self._check_batching()
        return self._batch_size

    @property
    def wires(self):
        """Wires that the operator acts on.

        Returns:
            Wires: wires
        """
        return self._wires

    @property
    def parameters(self):
        """Trainable parameters that the operator depends on."""
        return list(self.data)

    @property
    def hyperparameters(self):
        """dict: Dictionary of non-trainable variables that this operation depends on."""
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "_hyperparameters"):
            return self._hyperparameters
        self._hyperparameters = {}
        return self._hyperparameters

    @property
    def pauli_rep(self):
        """A :class:`~.PauliSentence` representation of the Operator, or ``None`` if it doesn't have one."""
        return self._pauli_rep

    @property
    def is_hermitian(self):
        """This property determines if an operator is hermitian."""
        return False

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_decomposition(cls):
        r"""Bool: Whether or not the Operator returns a defined decomposition.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        # Some operators will overwrite `decomposition` instead of `compute_decomposition`
        # Currently, those are mostly classes from the operator arithmetic module.
        return (
            cls.compute_decomposition != Operator.compute_decomposition
            or cls.decomposition != Operator.decomposition
        )

    def decomposition(self) -> List["Operator"]:
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_decomposition`.

        Returns:
            list[Operator]: decomposition of the operator
        """
        return self.compute_decomposition(
            *self.parameters, wires=self.wires, **self.hyperparameters
        )

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters) -> List["Operator"]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """
        raise DecompositionUndefinedError

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_diagonalizing_gates(cls):
        r"""Bool: Whether or not the Operator returns defined diagonalizing gates.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        # Operators may overwrite `diagonalizing_gates` instead of `compute_diagonalizing_gates`
        # Currently, those are mostly classes from the operator arithmetic module.
        return (
            cls.compute_diagonalizing_gates != Operator.compute_diagonalizing_gates
            or cls.diagonalizing_gates != Operator.diagonalizing_gates
        )

    @staticmethod
    def compute_diagonalizing_gates(
        *params, wires, **hyperparams
    ) -> List["Operator"]:  # pylint: disable=unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Operator.diagonalizing_gates`.

        Args:
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[.Operator]: list of diagonalizing gates
        """
        raise DiagGatesUndefinedError

    def diagonalizing_gates(self):  # pylint:disable=no-self-use
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Returns:
            list[.Operator] or None: a list of operators
        """
        return self.compute_diagonalizing_gates(
            *self.parameters, wires=self.wires, **self.hyperparameters
        )

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_generator(cls):
        r"""Bool: Whether or not the Operator returns a defined generator.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.generator != Operator.generator

    def generator(self):  # pylint: disable=no-self-use
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator()
          0.5 * Y(0) + Z(0) @ X(1)

        The generator may also be provided in the form of a dense or sparse Hamiltonian
        (using :class:`.Hermitian` and :class:`.SparseHamiltonian` respectively).

        The default value to return is ``None``, indicating that the operation has
        no defined generator.
        """
        raise GeneratorUndefinedError(f"Operation {self.name} does not have a generator")

    def pow(self, z) -> List["Operator"]:
        """A list of new operators equal to this one raised to the given power.

        Args:
            z (float): exponent for the operator

        Returns:
            list[:class:`~.operation.Operator`]

        """
        # Child methods may call super().pow(z%period) where op**period = I
        # For example, PauliX**2 = I, SX**4 = I
        # Hence we define 0 and 1 special cases here.
        if z == 0:
            return []
        if z == 1:
            if QueuingManager.recording():
                return [qml.apply(self)]
            return [copy.copy(self)]
        raise PowUndefinedError

    def queue(self, context=QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self  # so pre-constructed Observable instances can be queued and returned in a single statement

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
            * `"_ops"`
            * `"_measurements"`
            * `None`
        """
        return "_ops"

    # pylint: disable=no-self-argument, comparison-with-callable
    @classproperty
    def has_adjoint(cls):
        r"""Bool: Whether or not the Operator can compute its own adjoint.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.adjoint != Operator.adjoint

    def adjoint(self) -> "Operator":  # pylint:disable=no-self-use
        """Create an operation that is the adjoint of this one.

        Adjointed operations are the conjugated and transposed version of the
        original operation. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        Returns:
            The adjointed operation.
        """
        raise AdjointUndefinedError

    def expand(self):
        """Returns a tape that contains the decomposition of the operator.

        Returns:
            .QuantumTape: quantum tape
        """
        if not self.has_decomposition:
            raise DecompositionUndefinedError

        qscript = qml.tape.QuantumScript(self.decomposition())

        if not self.data:
            # original operation has no trainable parameters
            qscript.trainable_params = {}

        return qscript

    @property
    def arithmetic_depth(self) -> int:
        """Arithmetic depth of the operator."""
        return 0

    def map_wires(self, wire_map: dict):
        """Returns a copy of the current operator with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .Operator: new operator
        """
        new_op = copy.copy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        if (p_rep := new_op.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)
        return new_op

    def simplify(self) -> "Operator":  # pylint: disable=unused-argument
        """Reduce the depth of nested operators to the minimum.

        Returns:
            .Operator: simplified operator
        """
        return self

    def __add__(self, other):
        """The addition operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator):
            return qml.sum(self, other, lazy=False)
        if isinstance(other, TensorLike):
            if qml.math.allequal(other, 0):
                return self
            return qml.sum(
                self,
                qml.s_prod(scalar=other, operator=qml.Identity(self.wires), lazy=False),
                lazy=False,
            )
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other):
        """The scalar multiplication between scalars and Operators."""
        if callable(other):
            return qml.pulse.ParametrizedHamiltonian([other], [self])
        if isinstance(other, TensorLike):
            return qml.s_prod(scalar=other, operator=self, lazy=False)
        return NotImplemented

    def __truediv__(self, other):
        """The division between an Operator and a number."""
        if isinstance(other, TensorLike):
            return self.__mul__(1 / other)
        return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other):
        """The product operation between Operator objects."""
        return qml.prod(self, other, lazy=False) if isinstance(other, Operator) else NotImplemented

    def __sub__(self, other):
        """The subtraction operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator):
            return self + qml.s_prod(-1, other, lazy=False)
        if isinstance(other, TensorLike):
            return self + (qml.math.multiply(-1, other))
        return NotImplemented

    def __rsub__(self, other):
        """The reverse subtraction operation of Operator-Operator objects and Operator-scalar."""
        return -self + other

    def __neg__(self):
        """The negation operation of an Operator object."""
        return qml.s_prod(scalar=-1, operator=self, lazy=False)

    def __pow__(self, other):
        r"""The power operation of an Operator object."""
        if isinstance(other, TensorLike):
            return qml.pow(self, z=other)
        return NotImplemented

    def _flatten(self):
        """Serialize the operation into trainable and non-trainable components.

        Returns:
            data, metadata: The trainable and non-trainable components.

        See ``Operator._unflatten``.

        The data component can be recursive and include other operations. For example, the trainable component of ``Adjoint(RX(1, wires=0))``
        will be the operator ``RX(1, wires=0)``.

        The metadata **must** be hashable.  If the hyperparameters contain a non-hashable component, then this
        method and ``Operator._unflatten`` should be overridden to provide a hashable version of the hyperparameters.

        **Example:**

        >>> op = qml.Rot(1.2, 2.3, 3.4, wires=0)
        >>> qml.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qml.PauliRot(1.2, "XY", wires=(0,1))
        >>> qml.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])

        Operators that have trainable components that differ from their ``Operator.data`` must implement their own
        ``_flatten`` methods.

        >>> op = qml.ctrl(qml.U2(3.4, 4.5, wires="a"), ("b", "c") )
        >>> op._flatten()
        ((U2(3.4, 4.5, wires=['a']),),
        (<Wires = ['b', 'c']>, (True, True), <Wires = []>))

        """
        hashable_hyperparameters = tuple(
            (key, value) for key, value in self.hyperparameters.items()
        )
        return self.data, (self.wires, hashable_hyperparameters)

    @classmethod
    def _unflatten(cls, data, metadata):
        """Recreate an operation from its serialized format.

        Args:
            data: the trainable component of the operation
            metadata: the non-trainable component of the operation.

        The output of ``Operator._flatten`` and the class type must be sufficient to reconstruct the original
        operation with ``Operator._unflatten``.

        **Example:**

        >>> op = qml.Rot(1.2, 2.3, 3.4, wires=0)
        >>> op._flatten()
        ((1.2, 2.3, 3.4), (<Wires = [0]>, ()))
        >>> qml.Rot._unflatten(*op._flatten())
        >>> op = qml.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten()
        ((1.2,), (<Wires = [0, 1]>, (('pauli_word', 'XY'),)))
        >>> op = qml.ctrl(qml.U2(3.4, 4.5, wires="a"), ("b", "c") )
        >>> type(op)._unflatten(*op._flatten())
        Controlled(U2(3.4, 4.5, wires=['a']), control_wires=['b', 'c'])

        """
        hyperparameters_dict = dict(metadata[1])
        return cls(*data, wires=metadata[0], **hyperparameters_dict)


# =============================================================================
# Base Operation class
# =============================================================================


class Operation(Operator):
    r"""Base class representing quantum gates or channels applied to quantum states.

    Operations define some additional properties, that are used for external
    transformations such as gradient transforms.

    The following three class attributes are optional, but in most cases
    at least one should be clearly defined to avoid unexpected behaviour during
    differentiation.

    * :attr:`~.Operation.grad_recipe`
    * :attr:`~.Operation.parameter_frequencies`
    * :attr:`~.Operation.generator`

    Note that ``grad_recipe`` takes precedence when computing parameter-shift
    derivatives. Finally, these optional class attributes are used by certain
    transforms, quantum optimizers, and gradient methods.
    For details on how they are used during differentiation and other transforms,
    please see the documentation for :class:`~.gradients.param_shift`,
    :class:`~.metric_tensor`, :func:`~.reconstruct`.

    Args:
        *params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    @property
    def grad_method(self):
        """Gradient computation method.

        * ``'A'``: analytic differentiation using the parameter-shift method.
        * ``'F'``: finite difference numerical differentiation.
        * ``None``: the operation may not be differentiated.

        Default is ``'F'``, or ``None`` if the Operation has zero parameters.
        """
        if self.num_params == 0:
            return None
        if self.grad_recipe != [None] * self.num_params:
            return "A"
        try:
            self.parameter_frequencies  # pylint:disable=pointless-statement
            return "A"
        except ParameterFrequenciesUndefinedError:
            return "F"

    grad_recipe = None
    r"""tuple(Union(list[list[float]], None)) or None: Gradient recipe for the
        parameter-shift method.

        This is a tuple with one nested list per operation parameter. For
        parameter :math:`\phi_k`, the nested list contains elements of the form
        :math:`[c_i, a_i, s_i]` where :math:`i` is the index of the
        term, resulting in a gradient recipe of

        .. math:: \frac{\partial}{\partial\phi_k}f = \sum_{i} c_i f(a_i \phi_k + s_i).

        If ``None``, the default gradient recipe containing the two terms
        :math:`[c_0, a_0, s_0]=[1/2, 1, \pi/2]` and :math:`[c_1, a_1,
        s_1]=[-1/2, 1, -\pi/2]` is assumed for every parameter.
    """

    # Attributes for compilation transforms
    @property
    def basis(self):
        """str or None: The basis of an operation, or for controlled gates, of the
        target operation. If not ``None``, should take a value of ``"X"``, ``"Y"``,
        or ``"Z"``.

        For example, ``X`` and ``CNOT`` have ``basis = "X"``, whereas
        ``ControlledPhaseShift`` and ``RZ`` have ``basis = "Z"``.
        """
        return None

    @property
    def control_wires(self):  # pragma: no cover
        r"""Control wires of the operator.

        For operations that are not controlled,
        this is an empty ``Wires`` object of length ``0``.

        Returns:
            Wires: The control wires of the operation.
        """
        return Wires([])

    def single_qubit_rot_angles(self) -> Tuple[float, float, float]:
        r"""The parameters required to implement a single-qubit gate as an
        equivalent ``Rot`` gate, up to a global phase.

        Returns:
            tuple[float, float, float]: A list of values :math:`[\phi, \theta, \omega]`
            such that :math:`RZ(\omega) RY(\theta) RZ(\phi)` is equivalent to the
            original operation.
        """
        raise NotImplementedError

    @property
    def parameter_frequencies(self):
        r"""Returns the frequencies for each operator parameter with respect
        to an expectation value of the form
        :math:`\langle \psi | U(\mathbf{p})^\dagger \hat{O} U(\mathbf{p})|\psi\rangle`.

        These frequencies encode the behaviour of the operator :math:`U(\mathbf{p})`
        on the value of the expectation value as the parameters are modified.
        For more details, please see the :mod:`.pennylane.fourier` module.

        Returns:
            list[tuple[int or float]]: Tuple of frequencies for each parameter.
            Note that only non-negative frequency values are returned.

        **Example**

        >>> op = qml.CRot(0.4, 0.1, 0.3, wires=[0, 1])
        >>> op.parameter_frequencies
        [(0.5, 1), (0.5, 1), (0.5, 1)]

        For operators that define a generator, the parameter frequencies are directly
        related to the eigenvalues of the generator:

        >>> op = qml.ControlledPhaseShift(0.1, wires=[0, 1])
        >>> op.parameter_frequencies
        [(1,)]
        >>> gen = qml.generator(op, format="observable")
        >>> gen_eigvals = qml.eigvals(gen)
        >>> qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals))
        (1.0,)

        For more details on this relationship, see :func:`.eigvals_to_frequencies`.
        """
        if self.num_params == 1:
            # if the operator has a single parameter, we can query the
            # generator, and if defined, use its eigenvalues.
            try:
                gen = qml.generator(self, format="observable")
            except GeneratorUndefinedError as e:
                raise ParameterFrequenciesUndefinedError(
                    f"Operation {self.name} does not have parameter frequencies defined."
                ) from e

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message=r".+ eigenvalues will be computed numerically\."
                )
                eigvals = qml.eigvals(gen, k=2**self.num_wires)

            eigvals = tuple(np.round(eigvals, 8))
            return [qml.gradients.eigvals_to_frequencies(eigvals)]

        raise ParameterFrequenciesUndefinedError(
            f"Operation {self.name} does not have parameter frequencies defined, "
            "and parameter frequencies can not be computed as no generator is defined."
        )

    def __init__(self, *params, wires=None, id=None):
        super().__init__(*params, wires=wires, id=id)

        # check the grad_recipe validity
        if self.grad_recipe is None:
            # Make sure grad_recipe is an iterable of correct length instead of None
            self.grad_recipe = [None] * self.num_params


class Channel(Operation, abc.ABC):
    r"""Base class for quantum channels.

    Quantum channels have to define an additional numerical representation
    as Kraus matrices.

    Args:
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    # pylint: disable=abstract-method

    @staticmethod
    @abc.abstractmethod
    def compute_kraus_matrices(
        *params, **hyperparams
    ) -> List[np.ndarray]:  # pylint:disable=unused-argument
        """Kraus matrices representing a quantum channel, specified in
        the computational basis.

        This is a static method that should be defined for all
        new channels, and which allows matrices to be computed
        directly without instantiating the channel first.

        To return the Kraus matrices of an *instantiated* channel,
        please use the :meth:`~.Operator.kraus_matrices()` method instead.

        .. note::
            This method gets overwritten by subclasses to define the kraus matrix representation
            of a particular operator.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator,
                as stored in the ``hyperparameters`` attribute

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qml.AmplitudeDamping.compute_kraus_matrices(0.1)
        [array([[1., 0.], [0., 0.9486833]]),
         array([[0., 0.31622777], [0., 0.]])]
        """
        raise NotImplementedError

    def kraus_matrices(self):
        r"""Kraus matrices of an instantiated channel
        in the computational basis.

        Returns:
            list (array): list of Kraus matrices

        ** Example**

        >>> U = qml.AmplitudeDamping(0.1, wires=1)
        >>> U.kraus_matrices()
        [array([[1., 0.], [0., 0.9486833]]),
         array([[0., 0.31622777], [0., 0.]])]
        """
        return self.compute_kraus_matrices(*self.parameters, **self.hyperparameters)


# =============================================================================
# Base Observable class
# =============================================================================


class Observable(Operator):
    """Base class representing observables.

    Observables define a return type

    Args:
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
            * `"_ops"`
            * `"_measurements"`
            * None

        Non-pauli observables, like Tensor, Hermitian, and Hamiltonian, should not be processed into any queue.
        The Pauli observables double as Operations, and should therefore be processed into `_ops` if unowned.
        """
        return "_ops" if isinstance(self, Operation) else None

    @property
    def is_hermitian(self):
        """All observables must be hermitian"""
        return True

    def __matmul__(self, other):
        if active_new_opmath():
            return super().__matmul__(other=other)

        if isinstance(other, (Tensor, qml.ops.Hamiltonian, qml.ops.LinearCombination)):
            return other.__rmatmul__(self)

        if isinstance(other, Observable):
            return Tensor(self, other)

        return super().__matmul__(other=other)

    def _obs_data(self):
        r"""Extracts the data from a Observable or Tensor and serializes it in an order-independent fashion.

        This allows for comparison between observables that are equivalent, but are expressed
        in different orders. For example, `qml.X(0) @ qml.Z(1)` and
        `qml.Z(1) @ qml.X(0)` are equivalent observables with different orderings.

        **Example**

        >>> tensor = qml.X(0) @ qml.Z(1)
        >>> print(tensor._obs_data())
        {("PauliZ", <Wires = [1]>, ()), ("PauliX", <Wires = [0]>, ())}
        """
        obs = Tensor(self).non_identity_obs
        tensor = set()

        for ob in obs:
            parameters = tuple(param.tobytes() for param in ob.parameters)
            if isinstance(ob, qml.GellMann):
                parameters += (ob.hyperparameters["index"],)
            tensor.add((ob.name, ob.wires, parameters))

        return tensor

    def compare(self, other):
        r"""Compares with another :class:`~.Hamiltonian`, :class:`~Tensor`, or :class:`~Observable`,
        to determine if they are equivalent.

        Observables/Hamiltonians are equivalent if they represent the same operator
        (their matrix representations are equal), and they are defined on the same wires.

        .. Warning::

            The compare method does **not** check if the matrix representation
            of a :class:`~.Hermitian` observable is equal to an equivalent
            observable expressed in terms of Pauli matrices.
            To do so would require the matrix form of Hamiltonians and Tensors
            be calculated, which would drastically increase runtime.

        Returns:
            (bool): True if equivalent.

        **Examples**

        >>> ob1 = qml.X(0) @ qml.Identity(1)
        >>> ob2 = qml.Hamiltonian([1], [qml.X(0)])
        >>> ob1.compare(ob2)
        True
        >>> ob1 = qml.X(0)
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """
        if isinstance(other, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
            return other.compare(self)
        if isinstance(other, (Tensor, Observable)):
            return other._obs_data() == self._obs_data()

        raise ValueError(
            "Can only compare an Observable/Tensor, and a Hamiltonian/Observable/Tensor."
        )

    def __add__(self, other):
        r"""The addition operation between Observables/Tensors/qml.Hamiltonian objects."""
        if active_new_opmath():
            return super().__add__(other=other)

        if isinstance(other, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
            return other + self
        if isinstance(other, (Observable, Tensor)):
            return qml.Hamiltonian([1, 1], [self, other], simplify=True)

        return super().__add__(other=other)

    __radd__ = __add__

    def __mul__(self, a):
        r"""The scalar multiplication operation between a scalar and an Observable/Tensor."""
        if active_new_opmath():
            return super().__mul__(other=a)

        if isinstance(a, (int, float)):
            return qml.Hamiltonian([a], [self], simplify=True)

        return super().__mul__(other=a)

    __rmul__ = __mul__

    def __sub__(self, other):
        r"""The subtraction operation between Observables/Tensors/qml.Hamiltonian objects."""
        if active_new_opmath():
            return super().__sub__(other=other)

        if isinstance(other, (Observable, Tensor, qml.ops.Hamiltonian, qml.ops.LinearCombination)):
            return self + (-1 * other)
        return super().__sub__(other=other)


class Tensor(Observable):
    """Container class representing tensor products of observables.

    To create a tensor, simply initiate it like so:

    >>> T = Tensor(qml.X(0), qml.Hermitian(A, [1, 2]))

    You can also create a tensor from other Tensors:

    >>> T = Tensor(T, qml.Z(4))

    The ``@`` symbol can be used as a tensor product operation:

    >>> T = qml.X(0) @ qml.Hadamard(2)

    .. note:

        This class is marked for deletion or overhaul.
    """

    # pylint: disable=abstract-method
    tensor = True
    has_matrix = True

    def _flatten(self):
        return tuple(self.obs), tuple()

    @classmethod
    def _unflatten(cls, data, _):
        return cls(*data)

    def __init__(self, *args):  # pylint: disable=super-init-not-called
        self._eigvals_cache = None
        self.obs: List[Observable] = []
        self._args = args
        self._batch_size = None
        self._pauli_rep = None
        self.queue(init=True)

        wires = [op.wires for op in self.obs]
        if len(wires) != len(set(wires)):
            warnings.warn(
                "Tensor object acts on overlapping wires; in some PennyLane functions this will "
                "lead to undefined behaviour",
                UserWarning,
            )

        # Queue before updating pauli_rep because self.queue updates self.obs
        if all(prs := [o.pauli_rep for o in self.obs]):
            self._pauli_rep = functools.reduce(lambda a, b: a @ b, prs)
        else:
            self._pauli_rep = None

    def label(self, decimals=None, base_label=None, cache=None):
        r"""How the operator is represented in diagrams and drawings.

        Args:
            decimals=None (Int): If ``None``, no parameters are included. Else,
                how to round the parameters.
            base_label=None (Iterable[str]): overwrite the non-parameter component of the label.
                Must be same length as ``obs`` attribute.
            cache=None (dict): dictionary that carries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        >>> T = qml.X(0) @ qml.Hadamard(2)
        >>> T.label()
        'X@H'
        >>> T.label(base_label=["X0", "H2"])
        'X0@H2'

        """
        if base_label is not None:
            if len(base_label) != len(self.obs):
                raise ValueError(
                    "Tensor label requires ``base_label`` keyword to be same length "
                    "as tensor components."
                )
            return "@".join(
                ob.label(decimals=decimals, base_label=lbl) for ob, lbl in zip(self.obs, base_label)
            )

        return "@".join(ob.label(decimals=decimals) for ob in self.obs)

    def queue(self, context=QueuingManager, init=False):  # pylint: disable=arguments-differ
        constituents = self._args if init else self.obs
        for o in constituents:
            if init:
                if isinstance(o, Tensor):
                    self.obs.extend(o.obs)
                elif isinstance(o, Observable):
                    self.obs.append(o)
                else:
                    raise ValueError("Can only perform tensor products between observables.")

            context.remove(o)

        context.append(self)
        return self

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
        copied_op.obs = self.obs.copy()
        copied_op._eigvals_cache = self._eigvals_cache
        copied_op._batch_size = self._batch_size
        copied_op._pauli_rep = self._pauli_rep
        return copied_op

    def __repr__(self):
        """Constructor-call-like representation."""
        return " @ ".join([repr(o) for o in self.obs])

    @property
    def name(self):
        """All constituent observable names making up the tensor product.

        Returns:
            list[str]: list containing all observable names
        """
        return [o.name for o in self.obs]

    @property
    def num_wires(self):
        """Number of wires the tensor product acts on.

        Returns:
            int: number of wires
        """
        return len(self.wires)

    @property
    def wires(self):
        """All wires in the system the tensor product acts on.

        Returns:
            Wires: wires addressed by the observables in the tensor product
        """
        return Wires.all_wires([o.wires for o in self.obs])

    @property
    def data(self):
        """Raw parameters of all constituent observables in the tensor product.

        Returns:
            tuple[Any]: flattened list containing all dependent parameters
        """
        return tuple(d for op in self.obs for d in op.data)

    @data.setter
    def data(self, new_data):
        """Setter used to set the parameters of all constituent observables in the tensor product.

        The ``new_data`` argument should contain a list of elements, where each element corresponds
        to a list containing the parameters of each observable (in order). If an observable doesn't
        have any parameter, an empty list must be used.

        **Example:**

        >>> op = qml.X(0) @ qml.Hermitian(np.eye(2), wires=1)
        >>> op.data
        [array([[1., 0.],
        [0., 1.]])]
        >>> op.data = [[], [np.eye(2) * 5]]
        >>> op.data
        [array([[5., 0.],
        [0., 5.]])]
        """
        if isinstance(new_data, tuple):
            start = 0
            for op in self.obs:
                op.data = new_data[start : start + len(op.data)]
                start += len(op.data)
        else:
            for new_entry, op in zip(new_data, self.obs):
                op.data = tuple(new_entry)

    @property
    def num_params(self):
        """Raw parameters of all constituent observables in the tensor product.

        Returns:
            list[Any]: flattened list containing all dependent parameters
        """
        return len(self.data)

    @property
    def parameters(self):
        """Evaluated parameter values of all constituent observables in the tensor product.

        Returns:
            list[list[Any]]: nested list containing the parameters per observable
            in the tensor product
        """
        return [o.parameters for o in self.obs]

    @property
    def non_identity_obs(self):
        """Returns the non-identity observables contained in the tensor product.

        Returns:
            list[:class:`~.Observable`]: list containing the non-identity observables
            in the tensor product
        """
        return [obs for obs in self.obs if not isinstance(obs, qml.Identity)]

    @property
    def arithmetic_depth(self) -> int:
        return 1 + max(o.arithmetic_depth for o in self.obs)

    def __matmul__(self, other):
        if isinstance(other, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
            return other.__rmatmul__(self)

        if isinstance(other, Observable):
            return Tensor(self, other)

        if isinstance(other, Operator):
            return qml.prod(*self.obs, other)

        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, Observable):
            return Tensor(other, self)

        return NotImplemented

    __imatmul__ = __matmul__

    def eigvals(self):
        """Return the eigenvalues of the specified tensor product observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible.

        Returns:
            array[float]: array containing the eigenvalues of the tensor product
            observable
        """
        if self._eigvals_cache is not None:
            return self._eigvals_cache

        standard_observables = {"PauliX", "PauliY", "PauliZ", "Hadamard"}

        # observable should be Z^{\otimes n}
        self._eigvals_cache = pauli_eigs(len(self.wires))

        # check if there are any non-standard observables (such as Identity)
        if set(self.name) - standard_observables:
            # Tensor product of observables contains a mixture
            # of standard and non-standard observables
            self._eigvals_cache = np.array([1])
            for k, g in itertools.groupby(self.obs, lambda x: x.name in standard_observables):
                if k:
                    # Subgroup g contains only standard observables.
                    self._eigvals_cache = np.kron(self._eigvals_cache, pauli_eigs(len(list(g))))
                else:
                    # Subgroup g contains only non-standard observables.
                    for ns_ob in g:
                        # loop through all non-standard observables
                        self._eigvals_cache = np.kron(self._eigvals_cache, ns_ob.eigvals())

        return self._eigvals_cache

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_diagonalizing_gates(self):
        r"""Bool: Whether or not the Tensor returns defined diagonalizing gates."""
        return all(o.has_diagonalizing_gates for o in self.obs)

    def diagonalizing_gates(self):
        """Return the gate set that diagonalizes a circuit according to the
        specified tensor observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            list: list containing the gates diagonalizing the tensor observable
        """
        diag_gates = []
        for o in self.obs:
            diag_gates.extend(o.diagonalizing_gates())

        return diag_gates

    def matrix(self, wire_order=None):
        r"""Matrix representation of the Tensor operator
        in the computational basis.

        .. note::

            The wire_order argument is added for compatibility, but currently not implemented.
            The Tensor class is planned to be removed soon.

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels in the operator's wires

        Returns:
            array: matrix representation

        **Example**

        >>> O = qml.Z(0) @ qml.Z(2)
        >>> O.matrix()
        array([[ 1,  0,  0,  0],
               [ 0, -1,  0,  0],
               [ 0,  0, -1,  0],
               [ 0,  0,  0,  1]])

        To get the full :math:`2^3\times 2^3` Hermitian matrix
        acting on the 3-qubit system, the identity on wire 1
        must be explicitly included:

        >>> O = qml.Z(0) @ qml.Identity(1) @ qml.Z(2)
        >>> O.matrix()
        array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0., -1.,  0., -0.,  0., -0.,  0., -0.],
               [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
               [ 0., -0.,  0., -1.,  0., -0.,  0., -0.],
               [ 0.,  0.,  0.,  0., -1., -0., -0., -0.],
               [ 0., -0.,  0., -0., -0.,  1., -0.,  0.],
               [ 0.,  0.,  0.,  0., -0., -0., -1., -0.],
               [ 0., -0.,  0., -0., -0.,  0., -0.,  1.]])
        """

        if wire_order is not None:
            raise NotImplementedError("The wire_order argument is currently not implemented.")

        # Check for partially (but not fully) overlapping wires in the observables
        partial_overlap = self.check_wires_partial_overlap()

        # group the observables based on what wires they act on
        U_list = []
        for _, g in itertools.groupby(self.obs, lambda x: x.wires.labels):
            # extract the matrices of each diagonalizing gate
            mats = [i.matrix() for i in g]

            if len(mats) > 1:
                # multiply all unitaries together before appending
                mats = [multi_dot(mats)]

            # append diagonalizing unitary for specific wire to U_list
            U_list.append(mats[0])

        mat_size = np.prod([np.shape(mat)[0] for mat in U_list])
        wire_size = 2 ** len(self.wires)
        if mat_size != wire_size:
            if partial_overlap:
                warnings.warn(
                    "The matrix for Tensors of Tensors/Observables with partially "
                    "overlapping wires might yield unexpected results. In particular "
                    "the matrix size might be larger than intended."
                )
            else:
                warnings.warn(
                    f"The size of the returned matrix ({mat_size}) will not be compatible "
                    f"with the subspace of the wires of the Tensor ({wire_size}). "
                    "This likely is due to wires being used in multiple tensor product "
                    "factors of the Tensor."
                )

        # Return the Hermitian matrix representing the observable
        # over the defined wires.
        return functools.reduce(np.kron, U_list)

    def check_wires_partial_overlap(self):
        r"""Tests whether any two observables in the Tensor have partially
        overlapping wires and raise a warning if they do.

        .. note::

            Fully overlapping wires, i.e., observables with
            same (sets of) wires are not reported, as the ``matrix`` method is
            well-defined and implemented for this scenario.
        """
        for o1, o2 in itertools.combinations(self.obs, r=2):
            shared = qml.wires.Wires.shared_wires([o1.wires, o2.wires])
            if shared and (shared != o1.wires or shared != o2.wires):
                return 1
        return 0

    def sparse_matrix(
        self, wire_order=None, wires=None, format="csr"
    ):  # pylint:disable=arguments-renamed, arguments-differ
        r"""Computes, by default, a `scipy.sparse.csr_matrix` representation of this Tensor.

        This is useful for larger qubit numbers, where the dense matrix becomes very large, while
        consisting mostly of zero entries.

        Args:
            wire_order (Iterable): Wire labels that indicate the order of wires according to which the matrix
                is constructed. If not provided, ``self.wires`` is used.
            wires (Iterable): Same as ``wire_order`` to ensure compatibility with all the classes. Must only
                provide one: either ``wire_order`` or ``wires``.
            format: the output format for the sparse representation. All scipy sparse formats are accepted.

        Raises:
            ValueError: if both ``wire_order`` and ``wires`` are provided at the same time.

        Returns:
            :class:`scipy.sparse._csr.csr_matrix`: sparse matrix representation

        **Example**

        Consider the following tensor:

        >>> t = qml.X(0) @ qml.Z(1)

        Without passing wires, the sparse representation is given by:

        >>> print(t.sparse_matrix())
        (0, 2)	1
        (1, 3)	-1
        (2, 0)	1
        (3, 1)	-1

        If we define a custom wire ordering, the matrix representation changes
        accordingly:

        >>> print(t.sparse_matrix(wire_order=[1, 0]))
        (0, 1)	1
        (1, 0)	1
        (2, 3)	-1
        (3, 2)	-1

        We can also enforce implicit identities by passing wire labels that
        are not present in the constituent operations:

        >>> res = t.sparse_matrix(wire_order=[0, 1, 2])
        >>> print(res.shape)
        (8, 8)
        """
        if wires is not None and wire_order is not None:
            raise ValueError(
                "Wire order has been specified twice. Provide only one of either "
                "``wire_order`` or ``wires``, but not both."
            )

        wires = wires or wire_order
        wires = self.wires if wires is None else Wires(wires)
        list_of_sparse_ops = [eye(2, format="coo")] * len(wires)

        for o in self.obs:
            if len(o.wires) > 1:
                # todo: deal with multi-qubit operations that do not act on consecutive qubits
                raise ValueError(
                    f"Can only compute sparse representation for tensors whose operations "
                    f"act on consecutive wires; got {o}."
                )
            # store the single-qubit ops according to the order of their wires
            idx = wires.index(o.wires)
            list_of_sparse_ops[idx] = coo_matrix(o.matrix())

        return functools.reduce(lambda i, j: kron(i, j, format=format), list_of_sparse_ops)

    def prune(self):
        """Returns a pruned tensor product of observables by removing :class:`~.Identity` instances from
        the observables building up the :class:`~.Tensor`.

        If the tensor product only contains one observable, then this observable instance is
        returned.

        Note that, as a result, this method can return observables that are not a :class:`~.Tensor`
        instance.

        **Example:**

        Pruning that returns a :class:`~.Tensor`:

        >>> O = qml.Z(0) @ qml.Identity(1) @ qml.Z(2)
        >>> O.prune()
        <pennylane.operation.Tensor at 0x7fc1642d1590
        >>> [(o.name, o.wires) for o in O.prune().obs]
        [('PauliZ', [0]), ('PauliZ', [2])]

        Pruning that returns a single observable:

        >>> O = qml.Z(0) @ qml.Identity(1)
        >>> O_pruned = O.prune()
        >>> (O_pruned.name, O_pruned.wires)
        ('PauliZ', [0])

        Returns:
            ~.Observable: the pruned tensor product of observables
        """
        if qml.QueuingManager.recording():
            qml.QueuingManager.remove(self)

        if len(self.non_identity_obs) == 0:
            # Return a single Identity as the tensor only contains Identities
            return qml.Identity(self.wires[0]) if self.wires else qml.Identity()
        return (
            self.non_identity_obs[0]
            if len(self.non_identity_obs) == 1
            else Tensor(*self.non_identity_obs)
        )

    def map_wires(self, wire_map: dict):
        """Returns a copy of the current tensor with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .Tensor: new tensor
        """
        cls = self.__class__
        new_op = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
        new_op.obs = [obs.map_wires(wire_map) for obs in self.obs]
        new_op._eigvals_cache = self._eigvals_cache
        new_op._batch_size = self._batch_size
        new_op._pauli_rep = (
            self._pauli_rep.map_wires(wire_map) if self.pauli_rep is not None else None
        )
        return new_op


# =============================================================================
# CV Operations and observables
# =============================================================================


class CV:
    """A mixin base class denoting a continuous-variable operation."""

    # pylint: disable=no-member

    def heisenberg_expand(self, U, wire_order):
        """Expand the given local Heisenberg-picture array into a full-system one.

        Args:
            U (array[float]): array to expand (expected to be of the dimension ``1+2*self.num_wires``)
            wire_order (Wires): global wire order defining which subspace the operator acts on

        Raises:
            ValueError: if the size of the input matrix is invalid or `num_wires` is incorrect

        Returns:
            array[float]: expanded array, dimension ``1+2*num_wires``
        """

        U_dim = len(U)
        nw = len(self.wires)

        if U.ndim > 2:
            raise ValueError("Only order-1 and order-2 arrays supported.")

        if U_dim != 1 + 2 * nw:
            raise ValueError(f"{self.name}: Heisenberg matrix is the wrong size {U_dim}.")

        if len(wire_order) == 0 or len(self.wires) == len(wire_order):
            # no expansion necessary (U is a full-system matrix in the correct order)
            return U

        if not wire_order.contains_wires(self.wires):
            raise ValueError(
                f"{self.name}: Some observable wires {self.wires} do not exist on this device with wires {wire_order}"
            )

        # get the indices that the operation's wires have on the device
        wire_indices = wire_order.indices(self.wires)

        # expand U into the I, x_0, p_0, x_1, p_1, ... basis
        dim = 1 + len(wire_order) * 2

        def loc(w):
            "Returns the slice denoting the location of (x_w, p_w) in the basis."
            ind = 2 * w + 1
            return slice(ind, ind + 2)

        if U.ndim == 1:
            W = np.zeros(dim)
            W[0] = U[0]
            for k, w in enumerate(wire_indices):
                W[loc(w)] = U[loc(k)]
        elif U.ndim == 2:
            W = np.zeros((dim, dim)) if isinstance(self, Observable) else np.eye(dim)
            W[0, 0] = U[0, 0]

            for k1, w1 in enumerate(wire_indices):
                s1 = loc(k1)
                d1 = loc(w1)

                # first column
                W[d1, 0] = U[s1, 0]
                # first row (for gates, the first row is always (1, 0, 0, ...), but not for observables!)
                W[0, d1] = U[0, s1]

                for k2, w2 in enumerate(wire_indices):
                    W[d1, loc(w2)] = U[s1, loc(k2)]  # block k1, k2 in U goes to w1, w2 in W.
        return W

    @staticmethod
    def _heisenberg_rep(p):
        r"""Heisenberg picture representation of the operation.

        * For Gaussian CV gates, this method returns the matrix of the linear
          transformation carried out by the gate for the given parameter values.
          The method is not defined for non-Gaussian gates.

          **The existence of this method is equivalent to setting** ``grad_method = 'A'``.

        * For observables, returns a real vector (first-order observables) or
          symmetric matrix (second-order observables) of expansion coefficients
          of the observable.

        For single-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x, \p)`.
        For multi-mode Operations we use the basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.

        .. note::

            For gates, we assume that the inverse transformation is obtained
            by negating the first parameter.

        Args:
            p (Sequence[float]): parameter values for the transformation

        Returns:
            array[float]: :math:`\tilde{U}` or :math:`q`
        """
        # pylint: disable=unused-argument
        return None

    @classproperty
    def supports_heisenberg(self):
        """Whether a CV operator defines a Heisenberg representation.

        This indicates that it is Gaussian and does not block the use
        of the parameter-shift differentiation method if found between the differentiated gate
        and an observable.

        Returns:
            boolean
        """
        return CV._heisenberg_rep != self._heisenberg_rep


class CVOperation(CV, Operation):
    """Base class representing continuous-variable quantum gates.

    CV operations provide a special Heisenberg representation, as well as custom methods
    for differentiation.

    Args:
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    # pylint: disable=abstract-method

    @classproperty
    def supports_parameter_shift(self):
        """Returns True iff the CV Operation supports the parameter-shift differentiation method.
        This means that it has ``grad_method='A'`` and
        has overridden the :meth:`~.CV._heisenberg_rep` static method.
        """
        return self.grad_method == "A" and self.supports_heisenberg

    def heisenberg_pd(self, idx):
        """Partial derivative of the Heisenberg picture transform matrix.

        Computed using grad_recipe.

        Args:
            idx (int): index of the parameter with respect to which the
                partial derivative is computed.
        Returns:
            array[float]: partial derivative
        """
        # get the gradient recipe for this parameter
        recipe = self.grad_recipe[idx]

        # Default values
        multiplier = 0.5
        a = 1
        shift = np.pi / 2

        # We set the default recipe to as follows:
        # ∂f(x) = c*f(x+s) - c*f(x-s)
        default_param_shift = [[multiplier, a, shift], [-multiplier, a, -shift]]
        param_shift = default_param_shift if recipe is None else recipe

        pd = None  # partial derivative of the transformation

        p = self.parameters

        original_p_idx = p[idx]
        for c, _a, s in param_shift:
            # evaluate the transform at the shifted parameter values
            p[idx] = _a * original_p_idx + s
            U = self._heisenberg_rep(p)  # pylint: disable=assignment-from-none

            if pd is None:
                pd = c * U
            else:
                pd += c * U

        return pd

    def heisenberg_tr(self, wire_order, inverse=False):
        r"""Heisenberg picture representation of the linear transformation carried
        out by the gate at current parameter values.

        Given a unitary quantum gate :math:`U`, we may consider its linear
        transformation in the Heisenberg picture, :math:`U^\dagger(\cdot) U`.

        If the gate is Gaussian, this linear transformation preserves the polynomial order
        of any observables that are polynomials in :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.
        This also means it maps :math:`\text{span}(\mathbf{r})` into itself:

        .. math:: U^\dagger \mathbf{r}_i U = \sum_j \tilde{U}_{ij} \mathbf{r}_j

        For Gaussian CV gates, this method returns the transformation matrix for
        the current parameter values of the Operation. The method is not defined
        for non-Gaussian (and non-CV) gates.

        Args:
            wire_order (Wires): global wire order defining which subspace the operator acts on
            inverse  (bool): if True, return the inverse transformation instead

        Raises:
            RuntimeError: if the specified operation is not Gaussian or is missing the `_heisenberg_rep` method

        Returns:
            array[float]: :math:`\tilde{U}`, the Heisenberg picture representation of the linear transformation
        """
        p = [qml.math.toarray(a) for a in self.parameters]
        if inverse:
            try:
                # TODO: expand this for the new par domain class, for non-unitary matrices.
                p[0] = np.linalg.inv(p[0])
            except np.linalg.LinAlgError:
                p[0] = -p[0]  # negate first parameter
        U = self._heisenberg_rep(p)  # pylint: disable=assignment-from-none

        # not defined?
        if U is None:
            raise RuntimeError(
                f"{self.name} is not a Gaussian operation, or is missing the _heisenberg_rep method."
            )

        return self.heisenberg_expand(U, wire_order)


class CVObservable(CV, Observable):
    r"""Base class representing continuous-variable observables.

    CV observables provide a special Heisenberg representation.

    The class attribute :attr:`~.ev_order` can be defined to indicate
    to PennyLane whether the corresponding CV observable is a polynomial in the
    quadrature operators. If so,

    * ``ev_order = 1`` indicates a first order polynomial in quadrature
      operators :math:`(\x, \p)`.

    * ``ev_order = 2`` indicates a second order polynomial in quadrature
      operators :math:`(\x, \p)`.

    If :attr:`~.ev_order` is not ``None``, then the Heisenberg representation
    of the observable should be defined in the static method :meth:`~.CV._heisenberg_rep`,
    returning an array of the correct dimension.

    Args:
       params (tuple[tensor_like]): trainable parameters
       wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
           If not given, args[-1] is interpreted as wires.
       id (str): custom label given to an operator instance,
           can be useful for some applications where the instance has to be identified
    """

    # pylint: disable=abstract-method
    ev_order = None  #: None, int: Order in `(x, p)` that a CV observable is a polynomial of.

    def heisenberg_obs(self, wire_order):
        r"""Representation of the observable in the position/momentum operator basis.

        Returns the expansion :math:`q` of the observable, :math:`Q`, in the
        basis :math:`\mathbf{r} = (\I, \x_0, \p_0, \x_1, \p_1, \ldots)`.

        * For first-order observables returns a real vector such
          that :math:`Q = \sum_i q_i \mathbf{r}_i`.

        * For second-order observables returns a real symmetric matrix
          such that :math:`Q = \sum_{ij} q_{ij} \mathbf{r}_i \mathbf{r}_j`.

        Args:
            wire_order (Wires): global wire order defining which subspace the operator acts on
        Returns:
            array[float]: :math:`q`
        """
        p = self.parameters
        U = self._heisenberg_rep(p)  # pylint: disable=assignment-from-none
        return self.heisenberg_expand(U, wire_order)


class StatePrepBase(Operation):
    """An interface for state-prep operations."""

    grad_method = None

    # pylint:disable=too-few-public-methods
    @abc.abstractmethod
    def state_vector(self, wire_order=None):
        """
        Returns the initial state vector for a circuit given a state preparation.

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels
                from the operator's wires

        Returns:
            array: A state vector for all wires in a circuit
        """

    def label(self, decimals=None, base_label=None, cache=None):
        return "|Ψ⟩"


def operation_derivative(operation) -> np.ndarray:
    r"""Calculate the derivative of an operation.

    For an operation :math:`e^{i \hat{H} \phi t}`, this function returns the matrix representation
    in the standard basis of its derivative with respect to :math:`t`, i.e.,

    .. math:: \frac{d \, e^{i \hat{H} \phi t}}{dt} = i \phi \hat{H} e^{i \hat{H} \phi t},

    where :math:`\phi` is a real constant.

    Args:
        operation (.Operation): The operation to be differentiated.

    Returns:
        array: the derivative of the operation as a matrix in the standard basis

    Raises:
        ValueError: if the operation does not have a generator or is not composed of a single
            trainable parameter
    """
    generator = qml.matrix(
        qml.generator(operation, format="observable"), wire_order=operation.wires
    )
    return 1j * generator @ operation.matrix()


@qml.BooleanFn
def not_tape(obj):
    """Returns ``True`` if the object is not a quantum tape"""
    return isinstance(obj, qml.tape.QuantumScript)


@qml.BooleanFn
def has_gen(obj):
    """Returns ``True`` if an operator has a generator defined."""
    if isinstance(obj, Operator):
        return obj.has_generator
    try:
        obj.generator()
    except (AttributeError, OperatorPropertyUndefined, GeneratorUndefinedError):
        return False
    return True


@qml.BooleanFn
def has_grad_method(obj):
    """Returns ``True`` if an operator has a grad_method defined."""
    return obj.grad_method is not None


@qml.BooleanFn
def has_multipar(obj):
    """Returns ``True`` if an operator has more than one parameter
    according to ``num_params``."""
    return obj.num_params > 1


@qml.BooleanFn
def has_nopar(obj):
    """Returns ``True`` if an operator has no parameters
    according to ``num_params``."""
    return obj.num_params == 0


@qml.BooleanFn
def has_unitary_gen(obj):
    """Returns ``True`` if an operator has a unitary_generator
    according to the ``has_unitary_generator`` flag."""
    # Linting check disabled as static analysis can misidentify qml.ops as the set instance qml.ops.qubit.ops
    return obj in qml.ops.qubit.attributes.has_unitary_generator  # pylint:disable=no-member


@qml.BooleanFn
def is_measurement(obj):
    """Returns ``True`` if an operator is a ``MeasurementProcess`` instance."""
    return isinstance(obj, qml.measurements.MeasurementProcess)


@qml.BooleanFn
def is_trainable(obj):
    """Returns ``True`` if any of the parameters of an operator is trainable
    according to ``qml.math.requires_grad``."""
    return any(qml.math.requires_grad(p) for p in obj.parameters)


@qml.BooleanFn
def defines_diagonalizing_gates(obj):
    """Returns ``True`` if an operator defines the diagonalizing gates.

    This helper function is useful if the property is to be checked in
    a queuing context, but the resulting gates must not be queued.
    """
    return obj.has_diagonalizing_gates


@qml.BooleanFn
def gen_is_multi_term_hamiltonian(obj):
    """Returns ``True`` if an operator has a generator defined and it is a Hamiltonian
    with more than one term."""

    try:
        o = obj.generator()
    except (AttributeError, OperatorPropertyUndefined, GeneratorUndefinedError):
        return False

    return isinstance(o, (qml.ops.Hamiltonian, qml.ops.LinearCombination)) and len(o.coeffs) > 1


def enable_new_opmath(warn=True):
    """
    Change dunder methods to return arithmetic operators instead of Hamiltonians and Tensors

    Args:
        warn (bool): Whether or not to emit a warning for re-enabling new opmath. Default is ``True``.

    **Example**

    >>> qml.operation.active_new_opmath()
    False
    >>> type(qml.X(0) @ qml.Z(1))
    <class 'pennylane.operation.Tensor'>
    >>> qml.operation.enable_new_opmath()
    >>> type(qml.X(0) @ qml.Z(1))
    <class 'pennylane.ops.op_math.prod.Prod'>
    """
    if warn:
        warnings.warn(
            "Re-enabling the new Operator arithmetic system after disabling it is not advised."
            "Please visit https://docs.pennylane.ai/en/stable/news/new_opmath.html for help troubleshooting.",
            UserWarning,
        )
    global __use_new_opmath
    __use_new_opmath = True


def disable_new_opmath(warn=True):
    """
    Change dunder methods to return Hamiltonians and Tensors instead of arithmetic operators

    Args:
        warn (bool): Whether or not to emit a warning for disabling new opmath. Default is ``True``.

    **Example**

    >>> qml.operation.active_new_opmath()
    True
    >>> type(qml.X(0) @ qml.Z(1))
    <class 'pennylane.ops.op_math.prod.Prod'>
    >>> qml.operation.disable_new_opmath()
    >>> type(qml.X(0) @ qml.Z(1))
    <class 'pennylane.operation.Tensor'>
    """
    if warn:
        warnings.warn(
            "Disabling the new Operator arithmetic system for legacy support."
            "If you need help troubleshooting your code, please visit"
            "https://docs.pennylane.ai/en/stable/news/new_opmath.html",
            UserWarning,
        )
    global __use_new_opmath
    __use_new_opmath = False


def active_new_opmath():
    """
    Function that checks if the new arithmetic operator dunders are active

    Returns:
        bool: Returns ``True`` if the new arithmetic operator dunders are active

    **Example**

    >>> qml.operation.active_new_opmath()
    False
    >>> qml.operation.enable_new_opmath()
    >>> qml.operation.active_new_opmath()
    True
    """
    return __use_new_opmath


def convert_to_opmath(op):
    """
    Converts :class:`~pennylane.Hamiltonian` and :class:`.Tensor` instances
    into arithmetic operators. Objects of any other type are returned directly.

    Arithmetic operators include :class:`~pennylane.ops.op_math.Prod`,
    :class:`~pennylane.ops.op_math.Sum` and :class:`~pennylane.ops.op_math.SProd`.

    Args:
        op (Operator): The operator instance to convert

    Returns:
        Operator: An operator using the new arithmetic operations, if relevant
    """
    if isinstance(op, (qml.ops.Hamiltonian, qml.ops.LinearCombination)):
        if qml.QueuingManager.recording():
            qml.QueuingManager.remove(op)
        c, ops = op.terms()
        ops = tuple(convert_to_opmath(o) for o in ops)
        return qml.dot(c, ops)
    if isinstance(op, Tensor):
        if qml.QueuingManager.recording():
            qml.QueuingManager.remove(op)
        return qml.prod(*op.obs)
    return op


@contextmanager
def disable_new_opmath_cm():
    r"""Allows to use the old operator arithmetic within a
    temporary context using the `with` statement."""

    was_active = qml.operation.active_new_opmath()
    try:
        if was_active:
            disable_new_opmath(warn=False)
        yield
    except Exception as e:
        raise e
    finally:
        if was_active:
            enable_new_opmath(warn=False)
        else:
            disable_new_opmath(warn=False)


@contextmanager
def enable_new_opmath_cm():
    r"""Allows to use the new operator arithmetic within a
    temporary context using the `with` statement."""

    was_active = qml.operation.active_new_opmath()
    if not was_active:
        enable_new_opmath(warn=False)
    yield
    if was_active:
        enable_new_opmath(warn=False)
    else:
        disable_new_opmath(warn=False)


# pylint: disable=too-many-branches
def convert_to_H(op):
    """
    Converts arithmetic operators into a :class:`~pennylane.ops.Hamiltonian` or
    :class:`~pennylane.ops.LinearCombination` instance, depending on whether
    new_opmath is enabled. Objects of any other type are returned directly.

    Arithmetic operators include :class:`~pennylane.ops.op_math.Prod`,
    :class:`~pennylane.ops.op_math.Sum` and :class:`~pennylane.ops.op_math.SProd`.

    Args:
        op (Operator): The operator instance to convert.

    Returns:
        Operator: The operator as a :class:`~pennylane.ops.LinearCombination` instance
            if `active_new_opmath()`, otherwise a :class:`~pennylane.ops.Hamiltonian`
    """
    if not isinstance(op, (qml.ops.op_math.Prod, qml.ops.op_math.SProd, qml.ops.op_math.Sum)):
        return op

    coeffs = []
    ops = []

    op = qml.simplify(op)
    product = qml.ops.op_math.Prod if active_new_opmath() else Tensor

    if isinstance(op, Observable):
        coeffs.append(1.0)
        ops.append(op)

    elif isinstance(op, qml.ops.SProd):
        coeffs.append(op.scalar)
        if isinstance(op.base, Observable):
            ops.append(op.base)
        elif isinstance(op.base, qml.ops.op_math.Prod):
            ops.append(product(*op.base))
        else:
            raise ValueError("The base of scalar product must be an observable or a product.")

    elif isinstance(op, qml.ops.Prod):
        coeffs.append(1.0)
        ops.append(product(*op))

    elif isinstance(op, qml.ops.Sum):
        for factor in op:
            if isinstance(factor, (qml.ops.SProd)):
                coeffs.append(factor.scalar)
                if isinstance(factor.base, Observable):
                    ops.append(factor.base)
                elif isinstance(factor.base, qml.ops.op_math.Prod):
                    ops.append(product(*factor.base))
                else:
                    raise ValueError(
                        "The base of scalar product must be an observable or a product."
                    )
            elif isinstance(factor, (qml.ops.Prod)):
                coeffs.append(1.0)
                ops.append(product(*factor))
            elif isinstance(factor, Observable):
                coeffs.append(1.0)
                ops.append(factor)
            else:
                raise ValueError(
                    "Could not convert to Hamiltonian. Some or all observables are not valid."
                )

    else:
        raise ValueError("Could not convert to Hamiltonian. Some or all observables are not valid.")

    return qml.Hamiltonian(coeffs, ops)


def convert_to_legacy_H(op):
    """
    Converts arithmetic operators into a legacy :class:`~pennylane.Hamiltonian` instance.
    Objects of any other type are returned directly.

    Arithmetic operators include :class:`~pennylane.ops.op_math.Prod`,
    :class:`~pennylane.ops.op_math.Sum` and :class:`~pennylane.ops.op_math.SProd`.

    Args:
        op (Operator): The operator instance to convert.

    Returns:
        Operator: The operator as a :class:`~pennylane.Hamiltonian` instance
    """
    with disable_new_opmath_cm():
        res = convert_to_H(op)
    return res


def __getattr__(name):
    """To facilitate StatePrep rename"""
    if name == "StatePrep":
        return StatePrepBase
    raise AttributeError(f"module 'pennylane.operation' has no attribute '{name}'")
