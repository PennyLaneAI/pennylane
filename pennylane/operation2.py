# Copyright 2026 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
The :class:`Operator2` class serves as a base class for operators,
and is inherited by the
:class:`Operation2` class. These classes are subclassed to implement quantum operations
and measure observables in PennyLane.

* Each :class:`~.Operator2` subclass represents a general type of
  map between physical states. Each instance of these subclasses
  represents either

  - an application of the operator or
  - an instruction to measure and return the respective result.

  Operators act on a sequence of wires (subsystems) using given parameter values.

* Each :class:`~.Operation2` subclass represents a type of quantum operation,
  for example a unitary quantum gate. Each instance of these subclasses
  represents an application of the operation with given parameter values to
  a given sequence of wires (subsystems).


Differentiation
^^^^^^^^^^^^^^^

In general, an :class:`Operation2` is differentiable (at least using the finite-difference
method) with respect to a parameter iff

* the domain of that parameter is continuous.

For an :class:`Operation2` to be differentiable with respect to a parameter using the
analytic method of differentiation, it must satisfy an additional constraint:

* the parameter domain must be real.

.. note::

    These conditions are *not* sufficient for analytic differentiation. For example,
    CV gates must also define a matrix representing their Heisenberg linear
    transformation on the quadrature operators.

Contents
--------

.. currentmodule:: pennylane.operation2

Operator Types
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.operation2

.. autosummary::
    :toctree: api

    ~Operator2
    ~Operation2
    ~StatePrepBase2

.. currentmodule:: pennylane.operation2

.. inheritance-diagram:: Operator2 Operation2 StatePrepBase2
    :parts: 1
"""

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from functools import partial
from inspect import BoundArguments, Signature, signature
from typing import Any, ClassVar, Hashable

from scipy.sparse import spmatrix

import pennylane as qp
from pennylane import math
from pennylane.capture import ABCCaptureMeta
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    GeneratorUndefinedError,
    MatrixUndefinedError,
    ParameterFrequenciesUndefinedError,
    PennyLaneDeprecationWarning,
    PowUndefinedError,
    SparseMatrixUndefinedError,
    TermsUndefinedError,
)
from pennylane.operation import classproperty
from pennylane.typing import TensorLike, WiresLike
from pennylane.wires import Wires


class Operator2(ABC, metaclass=ABCCaptureMeta):
    r"""Base class representing quantum operators.

    Operators are uniquely defined by their name, the wires they act on, their dynamic
    inputs, and their static inputs. The dynamic inputs can be tensors of any supported
    auto-differentiation framework.

    An operator can define any of the following representations:

    * Representation as a **matrix** (:meth:`.Operator2.matrix`), as specified by a
      global wire order that tells us where the wires are found on a register.

    * Representation as a **sparse matrix** (:meth:`.Operator2.sparse_matrix`). Currently, this
      is a SciPy CSR matrix format.

    * Representation via the **eigenvalue decomposition** specified by eigenvalues
      (:meth:`.Operator2.eigvals`) and diagonalizing gates (:meth:`.Operator2.diagonalizing_gates`).

    * Representation as a **product of operators** (:meth:`.Operator2.decomposition`).

    * Representation as a **linear combination of operators** (:meth:`.Operator2.terms`).

    * Representation by a **generator** via :math:`e^{G}` (:meth:`.Operator2.generator`).

    Each representation method comes with a static method prefixed by ``compute_``, which takes
    the same signature as the operator itself.

    Args:
        *args (tuple[Any]): Positional arguments
        *kwargs (dict[str, Any]): Key-word arguments

    **Example**

    A custom operator can be created by inheriting from :class:`~.Operator2` or one of its
    subclasses.

    The following is an example for a custom gate that inherits from the :class:`~.Operation2`
    subclass. It acts by potentially flipping a qubit and rotating another qubit. The custom
    operator defines a decomposition, which the devices can use (since it is unlikely that a
    device knows a native implementation for ``FlipAndRotate``). It also defines an adjoint
    operator.

    # FIXME: Update example if necessary
    .. code-block:: python

        import pennylane as qp

        class FlipAndRotate(qp.operation2.Operation2):

            # This attribute tells PennyLane what differentiation method to use. Here
            # we request parameter-shift (or "analytic") differentiation.
            grad_method = "A"

            # This attribute tells PennyLane which arguments correspond to wires. If
            # not specified, it will be assumed that the operator only has one argument
            # corresponding to wires, aptly called "wires".
            wire_argnames = ("wire_rot", "wire_flip")

            # This attrubute tells PennyLane which arguments correspond to static inputs.
            # An input is static if its value is guaranteed to be known at compile-time.
            static_argnames = ("do_flip",)

            # This attribute tells PennyLane which arguments correspond to dynamic inputs.
            # An input is dynamic if its value may not be known at compile-time. This attribute
            # can be inferred automatically if ``static_argnames`` and ``wire_argnames`` are
            # defined, but can also be specified explicitly.
            dynamic_argnames = ("angle",)

            # The number of dynamic arguments. This must match ``len(dynamic_argnames)``.
            num_params = 1

            # The number of allowed wires. If an arbitrary number of wires is allowed, use ``None``.
            # This must be specified for each wire argument.
            num_wires = (1, 1)

            # The number of dimensions expected for each dynamic argument. In this example,
            # ``angle`` is expected to be a scalar.
            ndim_params = (0,)

            def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False):

                # checking the inputs --------------

                if do_flip and wire_flip is None:
                    raise ValueError("Expected a wire to flip; got None.")

                if wire_flip is None:
                    wire_flip = ()

                #------------------------------------

                # do_flip is not dynamic but influences the action of the operator,
                # which is why we define it to be a static input

                # The parent class expects to receive the canonicalized arguments.
                super().__init__(angle, wire_rot, wire_flip, do_flip)

            @staticmethod
            def compute_decomposition(angle, wire_rot, wire_flip=None, do_flip=False):
                # Overwriting this method defines the decomposition of the new gate, as it is
                # called by Operator2.decomposition().
                # The signature of this function must match the operator's signature.
                op_list = []
                if do_flip:
                    op_list.append(qp.X(wire_flip))
                op_list.append(qp.RX(angle, wires=wire_rot))
                return op_list

            def adjoint(self):
                # The adjoint operator of this gate simply negates the angle.
                # Additionally, note that for convenience, all arguments of the operator are
                # automatically available as read-only properties.
                return FlipAndRotate(-self.angle, self.wire_rot, self.wire_flip, do_flip=self.do_flip)

    We can use the operation as follows:

    # FIXME: Update example if necessary
    .. code-block:: python

        from pennylane import numpy as np

        dev = qp.device("default.qubit", wires=["q1", "q2", "q3"])

        @qp.qnode(dev)
        def circuit(angle):
            FlipAndRotate(angle, wire_rot="q1", wire_flip="q1")
            return qp.expval(qp.Z("q1"))

    >>> a = np.array(3.14)
    >>> circuit(a)
    tensor(-0.99999873, requires_grad=True)

    .. details::
        :title: Serialization and Pytree format
        :href: serialization

        PennyLane operations are automatically registered as `Pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_ .
        See the ``Operator2._flatten`` and ``Operator2._unflatten`` methods for more information.

        # FIXME: Update pytree example if necessary
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten()
        ((1.2,), (Wires([0, 1]), (('pauli_word', 'XY'),)))
        >>> qp.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])

    # TODO: ##################### FIX BELOW THIS POINT #####################
    .. details::
        :title: Parameter broadcasting
        :href: parameter-broadcasting

        Many quantum functions are executed repeatedly at different parameters, which
        can be done with parameter broadcasting. For usage details and examples see the
        :class:`~.pennylane.QNode` documentation.

        In order to support parameter broadcasting with an operator class,
        the following steps are necessary:

        #. Define the class attribute ``ndim_params``, a tuple that indicates
           the expected number of dimensions for each dynamic argument
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

        Consider an operator with the same matrix as ``qp.RX``. A basic variant of
        ``compute_matrix`` (which will not be compatible with all autodifferentiation
        frameworks or backpropagation) is

        .. code-block:: python

            @staticmethod
            def compute_matrix(theta):
                '''Broadcasting axis ends up in the wrong position.'''
                c = qp.math.cos(theta / 2)
                s = qp.math.sin(theta / 2)
                return qp.math.array([[c, -1j * s], [-1j * s, c]])

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
                c = qp.math.cos(theta / 2)
                s = qp.math.sin(theta / 2)
                mat = qp.math.array([[c, -1j * s], [-1j * s, c]])
                # Check whether the input has a broadcasting axis
                if qp.math.ndim(theta)==1:
                    # Move the broadcasting axis to the first position
                    return qp.math.moveaxis(mat, 2, 0)
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
                decomp_ops = [qp.RX(x, wires=w) for x, w in zip(theta, wires)]
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
                decomp_ops = [qp.RX(x, wires=w) for x, w in zip(qp.math.T(theta), wires)]
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

    # ------------ Class variables set manually --------------------

    wire_argnames: ClassVar[tuple[str, ...]] = ("wires",)
    """The names of arguments corresponding to wires."""

    dyn_argnames: ClassVar[tuple[str, ...]]
    """The names of arguments corresponding to dynamic arguments. Dynamic arguments
    are those whose concrete values may not be known at compile-time."""

    static_argnames: ClassVar[tuple[str, ...]]
    """The names of arguments corresponding to static arguments. Static arguments
    are those whose concrete values must be known at compile-time."""

    ndim_params: ClassVar[tuple[int, ...]]
    """The number of dimensions expected for each dynamic argument. This must be specified
    for parameter broadcasting support."""

    # ------------ Class variables set automatically ---------------

    _sig: ClassVar[Signature]
    """The signature of the operator."""

    # ------------ Instance variables set automatically ------------

    _bound_args: BoundArguments
    """BoundArguments mapping arguments names to their values."""

    # TODO: Add more class/instance variables as needed.

    def __init__(self, *args, **kwargs):
        self._bound_args = self._sig.bind(*args, **kwargs)
        self._bound_args.apply_defaults()

    def __init_subclass__(cls: type["Operator2"]) -> None:
        cls._sig = signature(cls)
        _add_dynamic_properties(cls)

    @classmethod
    def _primitive_bind_call(cls: type["Operator2"], *args, **kwargs) -> None:
        return

    # ------------------ General properties ------------------

    @property
    def arguments(self) -> dict[str, Any]:
        """Dictionary mapping argument names to their values."""
        return self._bound_args.arguments

    @property
    def dyn_args(self) -> dict[str, Any]:
        """Dictionary mapping dynamic argument names to their values."""
        return {name: self.arguments[name] for name in self.dyn_argnames}

    @property
    def static_args(self) -> dict[str, Any]:
        """Dictionary mapping static argument names to their values."""
        return {name: self.arguments[name] for name in self.static_argnames}

    @property
    def wire_args(self) -> dict[str, Any]:
        """Dictionary mapping wire argument names to their values."""
        return {name: self.arguments[name] for name in self.wire_argnames}

    # ------------------ Operator representations ------------------
    # pylint: disable=unused-argument,no-self-argument,comparison-with-callable

    @staticmethod
    def compute_matrix(*args, **kwargs) -> TensorLike:
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`.Operator.matrix` and :func:`qp.matrix() <pennylane.matrix>`

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            tensor_like: matrix representation
        """
        raise MatrixUndefinedError

    @classproperty
    def has_matrix(cls) -> bool:
        r"""Bool: Whether or not the Operator returns a defined matrix.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.compute_matrix != Operator2.compute_matrix or cls.matrix != Operator2.matrix

    def matrix(self, wire_order: WiresLike | None = None) -> TensorLike:
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
        canonical_matrix = self.compute_matrix(**self.arguments)

        if (
            wire_order is None
            or self.wires == Wires(wire_order)
            or (
                self.name in qp.ops.qubit.attributes.symmetric_over_all_wires
                and set(self.wires) == set(wire_order)
            )
        ):
            return canonical_matrix

        return math.expand_matrix(canonical_matrix, wires=self.wires, wire_order=wire_order)

    @staticmethod
    def compute_sparse_matrix(*args, format: str = "csr", **kwargs) -> spmatrix:
        r"""Representation of the operator as a sparse matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator.sparse_matrix`

        Args:
            *args (tuple): Positional arguments of the operator
            format (str): format of the returned scipy sparse matrix, for example 'csr'
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation
        """
        raise SparseMatrixUndefinedError

    @classproperty
    def has_sparse_matrix(cls) -> bool:
        r"""Bool: Whether the Operator returns a defined sparse matrix.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return (
            cls.compute_sparse_matrix != Operator2.compute_sparse_matrix
            or cls.sparse_matrix != Operator2.sparse_matrix
        )

    def sparse_matrix(self, wire_order: WiresLike | None = None, format="csr") -> spmatrix:
        r"""Representation of the operator as a sparse matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        A ``SparseMatrixUndefinedError`` is raised if the sparse matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator.compute_sparse_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires
            format (str): format of the returned scipy sparse matrix, for example 'csr'

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation

        """
        canonical_sparse_matrix = self.compute_sparse_matrix(**self.arguments, format="csr")
        return math.expand_matrix(
            canonical_sparse_matrix, wires=self.wires, wire_order=wire_order
        ).asformat(format)

    # Decomposition

    # Eigenvalues

    # Diagonalizing gates

    # Terms

    # Generator

    # ------------------ General dunder methods ------------------

    def __hash__(self) -> int:
        serialized_dyn = tuple(
            (n, str(id(d) if math.is_abstract(d) else d)) for n, d in self.dyn_args.items()
        )
        serialized_wires = tuple((n, tuple(w)) for n, w in self.wire_args.items())
        serialized_static = tuple((n, str(s)) for n, s in self.static_args.items())
        return hash((self.name, serialized_dyn, serialized_wires, serialized_static))

    def __eq__(self, other) -> bool:
        return qp.equal(self, other)

    @property
    def hash(self):
        """Hash."""
        # TODO: This is an artifact and should be removed if possible
        return hash(self)

    def __copy__(self) -> "Operator2":
        cls = type(self)
        copied_op = cls.__new__(cls)
        for attr, value in vars(self).items():
            setattr(copied_op, attr, value)

        return copied_op

    def __deepcopy__(self, memo) -> "Operator2":
        copied_op = object.__new__(type(self))

        # The memo dict maps object ID to object, and is required by
        # the deepcopy function to keep track of objects it has already
        # deep copied.
        memo[id(self)] = copied_op

        for attr, value in vars(self).items():
            setattr(copied_op, attr, deepcopy(value, memo))
        return copied_op

    # ------------------ Arithmetic dunder methods ------------------


def _add_dynamic_properties(cls: type[Operator2]) -> None:
    """Create dynamic properties for an operator using its signature."""

    for name in cls._sig.parameters.keys():
        if name not in vars(cls):
            dyn_property = partial(_dynamic_property, name=name)
            cls.__dict__[name] = property(dyn_property)


def _dynamic_property(self: Operator2, name: str) -> Any:
    """Dynamic property for an argument called ``name``."""
    if "_bound_args" in vars(self) and name in self._bound_args.arguments:
        return self._bound_args.arguments[name]

    return object.__getattribute__(self, name)
