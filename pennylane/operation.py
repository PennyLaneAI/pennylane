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
# pylint: disable=protected-access
r"""
This module contains the abstract base classes for defining PennyLane
operations and observables.

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
"""
# pylint:disable=access-member-before-definition
import abc
import copy
import functools
import itertools
import numbers
import warnings
from enum import IntEnum
from typing import List

import numpy as np
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix, eye, kron

import pennylane as qml
from pennylane.wires import Wires
from pennylane.math import expand_matrix

from .utils import pauli_eigs


def __getattr__(name):
    # for more information on overwriting `__getattr__`, see https://peps.python.org/pep-0562/
    warning_names = {"Sample", "Variance", "Expectation", "Probability", "State", "MidMeasure"}
    if name in warning_names:
        obj = getattr(qml.measurements, name)
        warning_string = f"qml.operation.{name} is deprecated. Please import from qml.measurements.{name} instead"
        warnings.warn(warning_string, UserWarning)
        return obj
    try:
        return globals()[name]
    except KeyError as e:
        raise AttributeError from e


# =============================================================================
# Errors
# =============================================================================


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
    AllWires = 0


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

    # Use qml.math.real to take the real part. We may get complex inputs for
    # example when differentiating holomorphic functions with JAX: a complex
    # valued QNode (one that returns qml.state) requires complex typed inputs.
    if op.name in ("RX", "RY", "RZ", "PhaseShift", "Rot"):
        return str([qml.math.round(qml.math.real(d) % (2 * np.pi), 10) for d in op.data])

    if op.name in ("CRX", "CRY", "CRZ", "CRot"):
        return str([qml.math.round(qml.math.real(d) % (4 * np.pi), 10) for d in op.data])

    return str(op.data)


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
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
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

            def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False,
                               do_queue=True, id=None):

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
                # The do_queue keyword argument specifies whether or not
                # the operator is queued when created in a tape context.
                super().__init__(angle, wires=all_wires, do_queue=do_queue, id=id)

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
                    op_list.append(qml.PauliX(wires=wires[1]))
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
            return qml.expval(qml.PauliZ("q1"))

    >>> a = np.array(3.14)
    >>> circuit(a)
    -0.9999987318946099

    """
    # pylint: disable=too-many-public-methods, too-many-instance-attributes

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_op.data = self.data.copy()
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
                copied_op.data = value.copy()
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

    @staticmethod
    def compute_matrix(*params, **hyperparams):  # pylint:disable=unused-argument
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator.matrix` and :func:`~.matrix`

        Args:
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

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
        return cls.compute_matrix != Operator.compute_matrix

    def matrix(self, wire_order=None):
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

        if wire_order is None or self.wires == Wires(wire_order):
            return canonical_matrix

        return expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    @staticmethod
    def compute_sparse_matrix(*params, **hyperparams):  # pylint:disable=unused-argument
        r"""Representation of the operator as a sparse matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator.sparse_matrix`

        Args:
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters``
                attribute

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation
        """
        raise SparseMatrixUndefinedError

    def sparse_matrix(self, wire_order=None):
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
    def compute_eigvals(*params, **hyperparams):
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Operator.eigvals` and :func:`~.eigvals`

        Args:
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            tensor_like: eigenvalues
        """
        raise EigvalsUndefinedError

    def eigvals(self):
        r"""Eigenvalues of the operator in the computational basis.

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`, the operator
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
        except EigvalsUndefinedError:
            # By default, compute the eigenvalues from the matrix representation.
            # This will raise a NotImplementedError if the matrix is undefined.
            try:
                return qml.math.linalg.eigvals(self.matrix())
            except MatrixUndefinedError as e:
                raise EigvalsUndefinedError from e

    @staticmethod
    def compute_terms(*params, **hyperparams):  # pylint: disable=unused-argument
        r"""Representation of the operator as a linear combination of other operators (static method).

        .. math:: O = \sum_i c_i O_i

        .. seealso:: :meth:`~.Operator.terms`

        Args:
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the
                ``hyperparameters`` attribute

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients and list of operations
        """
        raise TermsUndefinedError

    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        .. seealso:: :meth:`~.Operator.compute_terms`

        Returns:
            tuple[list[tensor_like or float], list[.Operation]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`
        """
        return self.compute_terms(*self.parameters, **self.hyperparameters)

    @property
    @abc.abstractmethod
    def num_wires(self):
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

    @property
    def pauli_rep(self):
        return self._pauli_rep

    @pauli_rep.setter
    def pauli_rep(self, rep):
        self._pauli_rep = rep

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
        >>> op.label(decimals=2)
        "RX\n(1.23)"
        >>> op.label(base_label="my_label")
        "my_label"
        >>> op.label(decimals=2, base_label="my_label")
        "my_label\n(1.23)"
        >>> op.inv()
        >>> op.label()
        "RX⁻¹"

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
            return op_label

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
                return op_label

            for i, mat in enumerate(cache["matrices"]):
                if qml.math.shape(params[0]) == qml.math.shape(mat) and qml.math.allclose(
                    params[0], mat
                ):
                    return f"{op_label}(M{i})"

            # matrix not in cache
            mat_num = len(cache["matrices"])
            cache["matrices"].append(params[0])
            return f"{op_label}(M{mat_num})"

        if decimals is None:
            return op_label

        def _format(x):
            try:
                return format(qml.math.toarray(x), f".{decimals}f")
            except ValueError:
                # If the parameter can't be displayed as a float
                return format(x)

        param_string = ",\n".join(_format(p) for p in params)
        return op_label + f"\n({param_string})"

    def __init__(self, *params, wires=None, do_queue=True, id=None):
        # pylint: disable=too-many-branches
        self._name = self.__class__.__name__  #: str: name of the operator
        self._id = id
        self.queue_idx = None  #: int, None: index of the Operator in the circuit queue, or None if not in a queue
        self._pauli_rep = None

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
                not isinstance(self, (qml.Barrier, qml.Snapshot, qml.Hamiltonian))
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

        self._check_batching(params)

        self.data = list(params)  #: list[Any]: parameters of the operator

        if do_queue:
            self.queue()

    def _check_batching(self, params):
        """Check if the expected numbers of dimensions of parameters coincides with the
        ones received and sets the ``_batch_size`` attribute.

        Args:
            params (tuple): Parameters with which the operator is instantiated

        The check always passes and sets the ``_batch_size`` to ``None`` for the default
        ``Operator.ndim_params`` property but subclasses may overwrite it to define fixed
        expected numbers of dimensions, allowing to infer a batch size.
        """
        self._batch_size = None
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
                return
            raise e

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
        return self.data.copy()

    @property
    def hyperparameters(self):
        """dict: Dictionary of non-trainable variables that this operation depends on."""
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "_hyperparameters"):
            return self._hyperparameters
        self._hyperparameters = {}
        return self._hyperparameters

    @property
    def is_hermitian(self):
        """This property determines if an operator is hermitian."""
        return False

    def decomposition(self):
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
    def compute_decomposition(*params, wires=None, **hyperparameters):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """
        raise DecompositionUndefinedError

    @staticmethod
    def compute_diagonalizing_gates(
        *params, wires, **hyperparams
    ):  # pylint: disable=unused-argument
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

    def generator(self):  # pylint: disable=no-self-use
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator()
          (0.5) [Y0]
        + (1.0) [Z0 X1]

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
            return [copy.copy(self)]
        raise PowUndefinedError

    def queue(self, context=qml.QueuingContext):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self  # so pre-constructed Observable instances can be queued and returned in a single statement

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
            * `"_prep"`
            * `"_ops"`
            * `"_measurements"`
            * `None`
        """
        return "_ops"

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
        """Returns a tape that has recorded the decomposition of the operator.

        Returns:
            .QuantumTape: quantum tape
        """
        tape = qml.tape.QuantumTape(do_queue=False)

        with tape:
            self.decomposition()

        if not self.data:
            # original operation has no trainable parameters
            tape.trainable_params = {}

        # the inverse attribute can be defined by subclasses
        if getattr(self, "inverse", False):
            tape.inv()

        return tape

    @property
    def arithmetic_depth(self) -> int:
        """Arithmetic depth of the operator."""
        return 0

    def simplify(self) -> "Operator":  # pylint: disable=unused-argument
        """Reduce the depth of nested operators to the minimum.

        Returns:
            .Operator: simplified operator
        """
        return self

    def __add__(self, other):
        """The addition operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, numbers.Number):
            if other == 0:
                return self
            id_op = (
                qml.prod(*(qml.Identity(w) for w in self.wires))
                if len(self.wires) > 1
                else qml.Identity(self.wires[0])
            )
            return qml.op_sum(self, qml.s_prod(scalar=other, operator=id_op))
        if isinstance(other, Operator):
            return qml.op_sum(self, other)
        raise ValueError(f"Cannot add Operator and {type(other)}")

    __radd__ = __add__

    def __mul__(self, other):
        """The scalar multiplication between scalars and Operators."""
        if isinstance(other, numbers.Number):
            return qml.s_prod(scalar=other, operator=self)
        raise ValueError(f"Cannot multiply Operator and {type(other)}.")

    __rmul__ = __mul__

    def __matmul__(self, other):
        """The product operation between Operator objects."""
        if isinstance(other, Operator):
            return qml.prod(self, other)
        raise ValueError("Can only perform tensor products between operators.")

    def __sub__(self, other):
        """The substraction operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, (Operator, numbers.Number)):
            return self + (-other)
        raise ValueError(f"Cannot substract {type(other)} from Operator.")

    def __rsub__(self, other):
        """The reverse substraction operation of Operator-Operator objects and Operator-scalar."""
        return -self + other

    def __neg__(self):
        """The negation operation of an Operator object."""
        return qml.s_prod(scalar=-1, operator=self)

    def __pow__(self, other):
        r"""The power operation of an Operator object."""
        if isinstance(other, numbers.Number):
            return qml.ops.Pow(base=self, z=other)  # pylint: disable=no-member
        raise ValueError(f"Cannot raise an Operator with an exponent of type {type(other)}")


# =============================================================================
# Base Operation class
# =============================================================================


class Operation(Operator):
    r"""Base class representing quantum gates or channels applied to quantum states.

    Operations define some additional properties, that are used for external
    transformations such as gradient transforms.

    The following three class attributes are optional, but in most cases
    at least one should be clearly defined to avoid unexpected behavior during
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
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
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
    basis = None
    """str or None: The target operation for controlled gates.
    target operation. If not ``None``, should take a value of ``"X"``, ``"Y"``,
    or ``"Z"``.

    For example, ``X`` and ``CNOT`` have ``basis = "X"``, whereas
    ``ControlledPhaseShift`` and ``RZ`` have ``basis = "Z"``.
    """

    @property
    def control_wires(self):  # pragma: no cover
        r"""Control wires of the operator.

        For operations that are not controlled,
        this is an empty ``Wires`` object of length ``0``.

        Returns:
            Wires: The control wires of the operation.
        """
        return Wires([])

    def single_qubit_rot_angles(self):
        r"""The parameters required to implement a single-qubit gate as an
        equivalent ``Rot`` gate, up to a global phase.

        Returns:
            tuple[float, float, float]: A list of values :math:`[\phi, \theta, \omega]`
            such that :math:`RZ(\omega) RY(\theta) RZ(\phi)` is equivalent to the
            original operation.
        """
        raise NotImplementedError

    def get_parameter_shift(self, idx):
        r"""Multiplier and shift for the given parameter, based on its gradient recipe.

        Args:
            idx (int): parameter index within the operation

        Returns:
            list[[float, float, float]]: list of multiplier, coefficient, shift for each term in the gradient recipe

        Note that the default value for ``shift`` is None, which is replaced by the
        default shift :math:`\pi/2`.
        """
        warnings.warn(
            "The method get_parameter_shift is deprecated. Use the methods of "
            "the gradients module for general parameter-shift rules instead.",
            UserWarning,
        )
        # get the gradient recipe for this parameter
        recipe = self.grad_recipe[idx]
        if recipe is not None:
            return recipe

        # We no longer assume any default parameter-shift rule to apply.
        raise OperatorPropertyUndefined(
            f"The operation {self.name} does not have a parameter-shift recipe defined."
            " This error might occur if previously the two-term shift rule was assumed"
            " silently. In this case, consider adding it explicitly to the operation."
        )

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
                eigvals = qml.eigvals(gen)

            eigvals = tuple(np.round(eigvals, 8))
            return qml.gradients.eigvals_to_frequencies(eigvals)

        raise ParameterFrequenciesUndefinedError(
            f"Operation {self.name} does not have parameter frequencies defined, "
            "and parameter frequencies can not be computed as no generator is defined."
        )

    @property
    def inverse(self):
        """Boolean determining if the inverse of the operation was requested."""
        return self._inverse

    @inverse.setter
    def inverse(self, boolean):
        self._inverse = boolean

    def inv(self):
        """Inverts the operator.

        This method concatenates a string to the name of the operation,
        to indicate that the inverse will be used for computations.

        Any subsequent call of this method will toggle between the original
        operation and the inverse of the operation.

        Returns:
            :class:`Operator`: operation to be inverted
        """
        self.inverse = not self._inverse
        return self

    def matrix(self, wire_order=None):
        canonical_matrix = self.compute_matrix(*self.parameters, **self.hyperparameters)

        if self.inverse:
            canonical_matrix = qml.math.conj(qml.math.moveaxis(canonical_matrix, -2, -1))

        return expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    def eigvals(self):
        op_eigvals = super().eigvals()

        if self.inverse:
            return qml.math.conj(op_eigvals)

        return op_eigvals

    @property
    def base_name(self):
        """If inverse is requested, this is the name of the original
        operator to be inverted."""
        return self.__class__.__name__

    @property
    def name(self):
        """Name of the operator."""
        return self._name + ".inv" if self.inverse else self._name

    def label(self, decimals=None, base_label=None, cache=None):
        if self.inverse:
            base_label = base_label or self.__class__.__name__
            base_label += "⁻¹"
        return super().label(decimals=decimals, base_label=base_label, cache=cache)

    def __init__(self, *params, wires=None, do_queue=True, id=None):

        self._inverse = False
        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)

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
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """
    # pylint: disable=abstract-method

    @staticmethod
    @abc.abstractmethod
    def compute_kraus_matrices(*params, **hyperparams):  # pylint:disable=unused-argument
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
            params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            hyperparams (dict): non-trainable hyperparameters of the operator,
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
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
            * `"_prep"`
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

    # pylint: disable=abstract-method
    return_type = None
    """None or ObservableReturnTypes: Measurement type that this observable is called with."""

    def __repr__(self):
        """Constructor-call-like representation."""
        temp = super().__repr__()

        if self.return_type is None:
            return temp

        if self.return_type is qml.measurements.Probability:
            return repr(self.return_type) + f"(wires={self.wires.tolist()})"

        return repr(self.return_type) + "(" + temp + ")"

    def __matmul__(self, other):
        if isinstance(other, (Tensor, qml.Hamiltonian)):
            return other.__rmatmul__(self)

        if isinstance(other, Observable):
            return Tensor(self, other)

        try:
            return super().__matmul__(other=other)
        except ValueError as e:
            raise ValueError("Can only perform tensor products between operators.") from e

    def _obs_data(self):
        r"""Extracts the data from a Observable or Tensor and serializes it in an order-independent fashion.

        This allows for comparison between observables that are equivalent, but are expressed
        in different orders. For example, `qml.PauliX(0) @ qml.PauliZ(1)` and
        `qml.PauliZ(1) @ qml.PauliX(0)` are equivalent observables with different orderings.

        **Example**

        >>> tensor = qml.PauliX(0) @ qml.PauliZ(1)
        >>> print(tensor._obs_data())
        {("PauliZ", <Wires = [1]>, ()), ("PauliX", <Wires = [0]>, ())}
        """
        obs = Tensor(self).non_identity_obs
        tensor = set()

        for ob in obs:
            parameters = tuple(param.tobytes() for param in ob.parameters)
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

        >>> ob1 = qml.PauliX(0) @ qml.Identity(1)
        >>> ob2 = qml.Hamiltonian([1], [qml.PauliX(0)])
        >>> ob1.compare(ob2)
        True
        >>> ob1 = qml.PauliX(0)
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """
        if isinstance(other, qml.Hamiltonian):
            return other.compare(self)
        if isinstance(other, (Tensor, Observable)):
            return other._obs_data() == self._obs_data()

        raise ValueError(
            "Can only compare an Observable/Tensor, and a Hamiltonian/Observable/Tensor."
        )

    def __add__(self, other):
        r"""The addition operation between Observables/Tensors/qml.Hamiltonian objects."""
        if isinstance(other, qml.Hamiltonian):
            return other + self
        if isinstance(other, (Observable, Tensor)):
            return qml.Hamiltonian([1, 1], [self, other], simplify=True)
        try:
            return super().__add__(other=other)
        except ValueError as e:
            raise ValueError(f"Cannot add Observable and {type(other)}") from e

    __radd__ = __add__

    def __mul__(self, a):
        r"""The scalar multiplication operation between a scalar and an Observable/Tensor."""
        if isinstance(a, (int, float)):
            return qml.Hamiltonian([a], [self], simplify=True)
        try:
            return super().__mul__(other=a)
        except ValueError as e:
            raise ValueError(f"Cannot multiply Observable by {type(a)}") from e

    __rmul__ = __mul__

    def __sub__(self, other):
        r"""The subtraction operation between Observables/Tensors/qml.Hamiltonian objects."""
        if isinstance(other, (Observable, Tensor, qml.Hamiltonian)):
            return self.__add__(other.__mul__(-1))
        try:
            return super().__sub__(other=other)
        except ValueError as e:
            raise ValueError(f"Cannot subtract {type(other)} from Observable") from e


class Tensor(Observable):
    """Container class representing tensor products of observables.

    To create a tensor, simply initiate it like so:

    >>> T = Tensor(qml.PauliX(0), qml.Hermitian(A, [1, 2]))

    You can also create a tensor from other Tensors:

    >>> T = Tensor(T, qml.PauliZ(4))

    The ``@`` symbol can be used as a tensor product operation:

    >>> T = qml.PauliX(0) @ qml.Hadamard(2)

    .. note:

        This class is marked for deletion or overhaul.
    """

    # pylint: disable=abstract-method
    return_type = None
    tensor = True

    def __init__(self, *args):  # pylint: disable=super-init-not-called
        self._eigvals_cache = None
        self.obs: List[Observable] = []
        self._args = args
        self.queue(init=True)

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

        >>> T = qml.PauliX(0) @ qml.Hadamard(2)
        >>> T.label()
        'X@H'
        >>> T.label(base_label=["X0", "H2"])
        'X0@H2'

        """
        if base_label is not None:
            if len(base_label) != len(self.obs):
                raise ValueError(
                    "Tensor label requires ``base_label`` keyword to be same length"
                    " as tensor components."
                )
            return "@".join(
                ob.label(decimals=decimals, base_label=lbl) for ob, lbl in zip(self.obs, base_label)
            )

        return "@".join(ob.label(decimals=decimals) for ob in self.obs)

    def queue(self, context=qml.QueuingContext, init=False):  # pylint: disable=arguments-differ
        constituents = self.obs

        if init:
            constituents = self._args

        for o in constituents:

            if init:
                if isinstance(o, Tensor):
                    self.obs.extend(o.obs)
                elif isinstance(o, Observable):
                    self.obs.append(o)
                else:
                    raise ValueError("Can only perform tensor products between observables.")

            context.safe_update_info(o, owner=self)

        context.append(self, owns=tuple(constituents))
        return self

    def __copy__(self):
        cls = self.__class__
        copied_op = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
        copied_op.obs = self.obs.copy()
        copied_op._eigvals_cache = self._eigvals_cache
        return copied_op

    def __repr__(self):
        """Constructor-call-like representation."""

        s = " @ ".join([repr(o) for o in self.obs])

        if self.return_type is None:
            return s

        if self.return_type is qml.measurements.Probability:
            return repr(self.return_type) + f"(wires={self.wires.tolist()})"

        return repr(self.return_type) + "(" + s + ")"

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
            list[Any]: flattened list containing all dependent parameters
        """
        return sum((o.data for o in self.obs), [])

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
        if isinstance(other, Tensor):
            self.obs.extend(other.obs)

        elif isinstance(other, Observable):
            self.obs.append(other)

        else:
            raise ValueError("Can only perform tensor products between observables.")

        if (
            qml.QueuingContext.recording()
            and self not in qml.QueuingContext.active_context()._queue
        ):
            qml.QueuingContext.append(self)

        qml.QueuingContext.safe_update_info(self, owns=tuple(self.obs))
        qml.QueuingContext.safe_update_info(other, owner=self)

        return self

    def __rmatmul__(self, other):
        if isinstance(other, Observable):
            self.obs[:0] = [other]
            qml.QueuingContext.safe_update_info(self, owns=tuple(self.obs))
            qml.QueuingContext.safe_update_info(other, owner=self)
            return self

        raise ValueError("Can only perform tensor products between observables.")

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

        >>> O = qml.PauliZ(0) @ qml.PauliZ(2)
        >>> O.matrix()
        array([[ 1,  0,  0,  0],
               [ 0, -1,  0,  0],
               [ 0,  0, -1,  0],
               [ 0,  0,  0,  1]])

        To get the full :math:`2^3\times 2^3` Hermitian matrix
        acting on the 3-qubit system, the identity on wire 1
        must be explicitly included:

        >>> O = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)
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
        self, wires=None, format="csr"
    ):  # pylint:disable=arguments-renamed, arguments-differ
        r"""Computes, by default, a `scipy.sparse.csr_matrix` representation of this Tensor.

        This is useful for larger qubit numbers, where the dense matrix becomes very large, while
        consisting mostly of zero entries.

        Args:
            wires (Iterable): Wire labels that indicate the order of wires according to which the matrix
                is constructed. If not provided, ``self.wires`` is used.
            format: the output format for the sparse representation. All scipy sparse formats are accepted.

        Returns:
            :class:`scipy.sparse._csr.csr_matrix`: sparse matrix representation

        **Example**

        Consider the following tensor:

        >>> t = qml.PauliX(0) @ qml.PauliZ(1)

        Without passing wires, the sparse representation is given by:

        >>> print(t.sparse_matrix())
        (0, 2)	1
        (1, 3)	-1
        (2, 0)	1
        (3, 1)	-1

        If we define a custom wire ordering, the matrix representation changes
        accordingly:
        >>> print(t.sparse_matrix(wires=[1, 0]))
        (0, 1)	1
        (1, 0)	1
        (2, 3)	-1
        (3, 2)	-1

        We can also enforce implicit identities by passing wire labels that
        are not present in the constituent operations:

        >>> res = t.sparse_matrix(wires=[0, 1, 2])
        >>> print(res.shape)
        (8, 8)
        """

        if wires is None:
            wires = self.wires
        else:
            wires = Wires(wires)

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

        The ``return_type`` attribute is preserved while pruning.

        If the tensor product only contains one observable, then this observable instance is
        returned.

        Note that, as a result, this method can return observables that are not a :class:`~.Tensor`
        instance.

        **Example:**

        Pruning that returns a :class:`~.Tensor`:

        >>> O = qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2)
        >>> O.prune()
        <pennylane.operation.Tensor at 0x7fc1642d1590
        >>> [(o.name, o.wires) for o in O.prune().obs]
        [('PauliZ', [0]), ('PauliZ', [2])]

        Pruning that returns a single observable:

        >>> O = qml.PauliZ(0) @ qml.Identity(1)
        >>> O_pruned = O.prune()
        >>> (O_pruned.name, O_pruned.wires)
        ('PauliZ', [0])

        Returns:
            ~.Observable: the pruned tensor product of observables
        """
        if len(self.non_identity_obs) == 0:
            # Return a single Identity as the tensor only contains Identities
            obs = qml.Identity(self.wires[0])
        elif len(self.non_identity_obs) == 1:
            obs = self.non_identity_obs[0]
        else:
            obs = Tensor(*self.non_identity_obs)

        obs.return_type = self.return_type
        return obs


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
            if isinstance(self, Observable):
                W = np.zeros((dim, dim))
            else:
                W = np.eye(dim)

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
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
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
       do_queue (bool): indicates whether the operator should be
           recorded when created in a tape context
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
    return isinstance(obj, qml.tape.QuantumTape)


@qml.BooleanFn
def has_gen(obj):
    """Returns ``True`` if an operator has a generator defined."""
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
    """Returns ``True`` if an operator defines the diagonalizing
    gates are defined.

    This helper function is useful if the property is to be checked in
    a queuing context, but the resulting gates must not be queued.
    """

    with qml.tape.stop_recording():
        try:
            obj.diagonalizing_gates()
        except DiagGatesUndefinedError:
            return False
        return True


@qml.BooleanFn
def gen_is_multi_term_hamiltonian(obj):
    """Returns ``True`` if an operator has a generator defined and it is a Hamiltonian
    with more than one term."""

    try:
        o = obj.generator()
    except (AttributeError, OperatorPropertyUndefined, GeneratorUndefinedError):
        return False

    if isinstance(o, qml.Hamiltonian):
        if len(o.coeffs) > 1:
            return True
    return False
