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
TODO: [sc-120453] Fill docstring
"""

from abc import abstractmethod
from collections.abc import Callable, Hashable, Iterable, Sequence
from copy import copy, deepcopy
from enum import Enum, auto
from functools import partial
from importlib.util import find_spec
from inspect import BoundArguments, Signature, signature
from numbers import Number
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import numpy as np
from scipy.sparse import spmatrix

import pennylane as qp
from pennylane import math
from pennylane._class_property import classproperty
from pennylane.capture import enabled, pause
from pennylane.core.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.exceptions import (
    AdjointUndefinedError,
    DecompositionUndefinedError,
    DiagGatesUndefinedError,
    EigvalsUndefinedError,
    GeneratorUndefinedError,
    MatrixUndefinedError,
    PowUndefinedError,
    SparseMatrixUndefinedError,
    TermsUndefinedError,
)
from pennylane.pytrees import flatten, register_pytree, unflatten
from pennylane.typing import AbstractArray, AbstractWires, FlatPytree, TensorLike
from pennylane.wires import Wires, WiresLike

from .base import _UNSET_BATCH_SIZE, Operator, _get_abstract_operator
from .meta import OperatorMeta
from .utils import abstractify

if TYPE_CHECKING:
    from pennylane.pauli import PauliSentence

has_jax = find_spec("jax") is not None

ArgSpecType: TypeAlias = type[Number] | AbstractArray | AbstractWires


class Operator2(metaclass=OperatorMeta):
    r"""Base class representing quantum operators.
    TODO: [sc-120453] Fill docstring
    """

    # pylint: disable=too-many-public-methods, too-many-instance-attributes

    _operator_version = 2

    # ----------------- Class variables set manually -------------------------

    wire_argnames: ClassVar[tuple[str, ...]] = ("wires",)
    """The names of arguments corresponding to wires. Values for these arguments are
    automatically wrapped in :class:`~.Wires` objects by the ``Operator2`` constructor.
    If an argument is expected to be a structure that wraps wires (known as pytrees), such
    as a list of wire registers, make sure to include its name in ``hybrid_argnames`` as well.
    Additionally, ``Operator2`` does not descend into the pytree structure of such arguments,
    so subclasses must ensure every wire leaf inside a hybrid wire argument is already a
    :class:`~.Wires` object before forwarding it to ``super().__init__``.

    The order in which names appear in ``wire_argnames`` determines the order in which
    their wires appear in ``op.wires`` (see :attr:`Operator2.wires`). For hybrid wire
    arguments, the contained :class:`~.Wires` leaves are ordered by pytree traversal
    order. Wires contributed by arguments which are themselves :class:`~.Operator2`
    objects are appended *after* all ``wire_argnames`` wires. The special
    names ``"work_wires"`` and ``"work_wire"`` may be included in ``wire_argnames``
    but their values are excluded from ``op.wires``.
    """

    dynamic_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that are treated as dynamic. Dynamic arguments are numerical
    arguments whose concrete values may not be known at compile-time, such as rotation angles,
    matrices, etc., and may include trainable data. Additionally, dynamic arguments are the only
    ones used to infer :attr:`~.Operator2.batch_size`, necessary for broadcasting support.

    Inputs for these arguments must be scalars, arrays, or castable to arrays (e.g. multi-
    dimensional lists with homogenous shapes), and operator constructors will be responsible
    for "canonicalizing" them (such as casting a list input to an array). At compile-time,
    the concrete values of these arguments are assumed to be unknown—only their shapes and
    data types are known.
    """

    static_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that are treated as static. Static arguments are those
    whose concrete values are known when compiling the program. Arguments specified
    here are not lowered to a compiler intermediate representation (IR). Alternatively,
    if an argument is of a type that can be lowered, it can be moved to ``compilable_argnames``.

    .. note::

        An operator can only specify ``static_argnames`` and ``hybrid_argnames``, or
        ``compilable_argnames``, but not both; if **any** static or hybrid arguments are not
        or cannot be lowered to the IR, then all static and hybrid arguments are assumed to
        not be lowerable.
    """

    compilable_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that are treated as **compilable** static arguments.
    Compilable static arguments are a subset of static arguments—arguments whose
    concrete values are known when compiling the program. But, unlike ``static_argnames``,
    they are lowered to the compiler intermediate representation, making their values
    visible to the compiler, which may be useful if the compiler needs to interact with
    such values. Such values may include numbers, strings, and lists, tuples, or dictionaries
    thereof. This feature is opt-in; if any static arguments are not guaranteed to be
    compilable, it is safer to place them in ``static_argnames``.

    .. note::

        An operator can only specify ``static_argnames`` and ``hybrid_argnames``, or
        ``compilable_argnames``, but not both; if **any** static or hybrid arguments are not
        or cannot be lowered to the IR, then all static and hybrid arguments are assumed to
        not be lowerable.
    """

    hybrid_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that represent dynamic data wrapped in static
    structures (known as Pytrees). Names in this category must be disjoint from
    ``dynamic_argnames``, ``static_argnames``, and ``compilable_argnames``, but may
    overlap with ``wire_argnames`` when those arguments contain nested structures of
    wires. Examples of hybrid arguments include collections of wires or dynamic arrays,
    operators, etc.

    .. note::

        An operator can only specify ``static_argnames`` and ``hybrid_argnames``, or
        ``compilable_argnames``, but not both; if **any** static or hybrid arguments are not
        or cannot be lowered to the IR, then all static and hybrid arguments are assumed to
        not be lowerable.
    """

    wire_sizes: ClassVar[tuple[int | None, ...] | None] = None
    """The expected number of wire labels for each wire argument. If any wire arguments
    support an arbitrary size, ``None`` must be used. By default, all wire arguments are
    assumed to support arbitrary sizes. For hybrid wire arguments, the wire size must
    always be ``None``. Specifying wire sizes allows for additional automatic validation
    to be implemented, but, specifying it is optional if such validation is not needed.
    """

    arg_specs: ClassVar[dict[str, ArgSpecType] | None] = None
    """The expected types for the arguments of an operator. This attribute is optional—not
    setting it has no loss of functionality. If set, it can be used to perform automatic
    validation of an operators inputs during construction. Additionally, when defining
    decomposition rules for an operator, operator types with ``arg_specs`` that spans
    all the arguments with static types can be placed in the rules' resources without needing
    to fully construct abstract operators.
    """

    # ----------------- Class variables set automatically --------------------

    _sig: ClassVar[Signature]
    """The signature of the operator. Internal use only."""

    has_fixed_sig: ClassVar[bool]
    """Whether the expected signature of an operator is fixed. If ``True``, then the operator's
    signature will always be fully known. When defining decomposition rules for an operator,
    operator types with fixed signatures can be placed in the rules' resources without needing
    to fully construct abstract operators. This is set automatically when ``arg_specs`` covers
    every dynamic and wire argument, there are no hybrid, static, or compilable arguments, and
    every declared type is fully fixed (no unknown array shapes or wire counts).
    """

    # ----------------- Instance variables set automatically -----------------

    _bound_args: BoundArguments
    """``BoundArguments`` mapping arguments names to their values."""

    # ------------------------------------------------------------------------
    # ----------------------------- Initialization ---------------------------
    # ------------------------------------------------------------------------

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * ps1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    def __init__(self, *args, **kwargs):
        # Union[PauliSentence, None]: Representation of the operator as a
        # pauli sentence, if applicable
        self._pauli_rep: PauliSentence | None = None

        self._is_abstract = False

        self._bound_args = self._sig.bind(*args, **kwargs)
        self._bound_args.apply_defaults()

        self._wires = Wires([])
        _init_wires(self)
        _init_arg_types(self)

        # Broadcasting-related initialization
        self._batch_size: int | None = _UNSET_BATCH_SIZE
        self._ndim_params: tuple[int] = _UNSET_BATCH_SIZE

        self.tracer = None

    def __abstract_init__(self, *args, **kwargs):
        """Constructor for canonicalization of abstract inputs."""
        bound_args = self._sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arguments = bound_args.arguments

        target_args = self.dynamic_argnames + self.hybrid_argnames + self.wire_argnames
        for name in target_args:
            kind = _resolve_arg_kind(type(self), name)
            arguments[name] = _canonicalize_abstract_type(arguments[name], kind)

        Operator2.__init__(self, *bound_args.args, **bound_args.kwargs)
        self._is_abstract = True

    # ------------------------------------------------------------------------
    # -------------------------- Public properties ---------------------------
    # ------------------------------------------------------------------------

    @property
    def is_abstract(self) -> bool:
        """Whether the operator has abstract args."""
        return self._is_abstract

    @property
    def arguments(self) -> dict[str, Any]:
        """Dictionary mapping argument names to their values."""
        return self._bound_args.arguments

    @property
    def dynamic_args(self) -> dict[str, Any]:
        """Dictionary mapping dynamic argument names to their values."""
        return {name: self.arguments[name] for name in self.dynamic_argnames}

    @property
    def static_args(self) -> dict[str, Any]:
        """Dictionary mapping static argument names to their values."""
        return {name: self.arguments[name] for name in self.static_argnames}

    @property
    def wire_args(self) -> dict[str, Any]:
        """Dictionary mapping wire argument names to their values."""
        return {name: self.arguments[name] for name in self.wire_argnames}

    @property
    def hybrid_args(self) -> dict[str, Any]:
        """Dictionary mapping hybrid argument names to their values."""
        return {name: self.arguments[name] for name in self.hybrid_argnames}

    @property
    def compilable_args(self) -> dict[str, Any]:
        """Dictionary mapping compilable argument names to their values."""
        return {name: self.arguments[name] for name in self.compilable_argnames}

    @property
    def name(self) -> str:
        """Operator name."""
        return self.__class__.__name__

    @property
    def wires(self) -> Wires:
        """Wires that the operator acts on.

        The returned :class:`~.Wires` are collected from the operator's arguments in
        the following order:

        1. For each name in ``wire_argnames`` (in declaration order):

           * If the name is **not** in ``hybrid_argnames``, the canonical
             value of that argument is added.
           * If the name **is** in ``hybrid_argnames``, every :class:`~.Wires` leaf of
             the argument's pytree is added in pytree traversal order.

        2. After all ``wire_argnames`` have been processed, for each name in
           ``hybrid_argnames`` that is **not** in ``wire_argnames``, the wires of any
           :class:`~.Operator2` leaves inside that argument are appended to the end
           in pytree traversal order.

        Duplicate wires are removed while preserving the first occurrence, so the
        final result is the ordered union of all wires above.

        .. note::

            Work wires are **not included** in ``op.wires``. In particular, wire arguments
            named ``work_wires`` or ``work_wire`` are excluded.

        Returns:
            Wires: wires
        """
        return self._wires

    @property
    def batch_size(self) -> int | None:
        """Batch size of the operator if it is used with broadcasted parameters.

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
    def ndim_params(self) -> tuple[int]:
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
    def num_params(self):
        """Number of trainable parameters."""
        return len(self.ndim_params)

    @property
    def arithmetic_depth(self) -> int:
        """Arithmetic depth of the operator."""
        return 0

    @property
    def is_verified_hermitian(self) -> bool:
        """This property determines if an operator is verified to be Hermitian.

        .. note::

            This property provides a fast, non-exhaustive check used for internal
            optimizations. It relies on quick, provable shortcuts (e.g., operator
            properties) rather than a full, computationally expensive check.

            For a definitive check, use the :func:`pennylane.is_hermitian` function.
            Please note that this comes with increased computational cost.

        Returns:
            bool: The property will return ``True`` if the operator is guaranteed to be Hermitian and
            ``False`` if the check is inconclusive and the operator may or may not be Hermitian.

        Consider this operator,

        >>> op = (qp.X(0) @ qp.Y(0) - qp.X(0) @ qp.Z(0)) * 1j

        In this case, Hermicity cannot be verified and leads to an inconclusive result:

        >>> op.is_verified_hermitian # inconclusive
        False

        However, using :func:`pennylane.is_hermitian` will give the correct answer:

        >>> qp.is_hermitian(op) # definitive
        True

        """
        return False

    @property
    def pauli_rep(self) -> "PauliSentence | None":
        """A :class:`~.PauliSentence` representation of the Operator, or ``None``
        if it doesn't have one."""
        return self._pauli_rep

    # ------------------------------------------------------------------------
    # -------------- Legacy Operator compatibility views ----------------------
    # ------------------------------------------------------------------------
    # The following properties provide backwards-compatible read-only views
    # matching the legacy ``Operator`` API (data, parameters, hyperparameters,
    # control_wires).
    # They are *not* the canonical Operator2 API — prefer ``arguments``,
    # ``dynamic_args``, ``static_args``, etc. for new code.

    @property
    def data(self) -> tuple:
        """Legacy Operator compatibility view of dynamic numerical arguments."""
        return tuple(self.arguments[name] for name in self.dynamic_argnames)

    @property
    def parameters(self) -> list:
        """Legacy Operator compatibility view of dynamic numerical arguments as a list."""
        return list(self.data)

    @property
    def hyperparameters(self) -> dict:
        """Legacy Operator compatibility view of non-dynamic, non-wire constructor arguments."""
        return {
            name: value
            for name, value in self.arguments.items()
            if name not in self.dynamic_argnames and name not in self.wire_argnames
        }

    # ------------------------------------------------------------------------
    # --------------------------- Operator actions ---------------------------
    # ------------------------------------------------------------------------

    def label(
        self, decimals: int | None = None, base_label: str | None = None, cache: dict | None = None
    ) -> str:
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

        >>> op = qp.RX(1.23456, wires=0)
        >>> op.label()
        'RX'
        >>> op.label(base_label="my_label")
        'my_label'
        >>> op = qp.RX(1.23456, wires=0)
        >>> op.label()
        'RX'
        >>> op.label(decimals=2)
        'RX\n(1.23)'
        >>> op.label(base_label="my_label")
        'my_label'
        >>> op.label(decimals=2, base_label="my_label")
        'my_label\n(1.23)'

        Note that by default, operator wires and static inputs are not included in the label.
        To override this behaviour, ``Operator2`` subclasses must override the ``label`` method.

        If the operation has a matrix-valued parameter and a cache dictionary is provided,
        unique matrices will be cached in the ``'matrices'`` key list. The label will contain
        the index of the matrix in the ``'matrices'`` list.

        >>> op2 = qp.QubitUnitary(np.eye(2), wires=0)
        >>> cache = {'matrices': []}
        >>> op2.label(cache=cache)
        'U\n(M0)'
        >>> cache['matrices']
        [array([[1., 0.],
               [0., 1.]])]
        >>> op3 = qp.QubitUnitary(np.eye(4), wires=(0,1))
        >>> op3.label(cache=cache)
        'U\n(M1)'
        >>> cache['matrices']
        [array([[1., 0.],
               [0., 1.]]), array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])]

        """
        op_label = base_label or self.__class__.__name__

        if len(self.dynamic_argnames) == 0:
            return op_label

        # Format each parameter individually, excluding those that lead to empty strings
        param_strings = [
            out
            for p in self.dynamic_args.values()
            if (out := _format_label_arg(p, decimals, cache)) != ""
        ]
        inner_string = ",\n".join(param_strings)

        if inner_string == "":
            return f"{op_label}"
        return f"{op_label}\n({inner_string})"

    def pow(self, z: float) -> list["Operator2"]:
        """A list of new operators equal to this one raised to the given power. This method is used to simplify
        :class:`~.Pow` instances created by :func:`~.pow` or ``op ** power``.
        TODO: [sc-120843] Fix docstring after Pow is added

        ``Operator2.pow`` can be optionally defined by Operator developers, while :func:`~.pow` or ``op ** power``
        are the entry point for constructing generic powers to exponents.

        Args:
            z (float): exponent for the operator

        Returns:
            list[:class:`~.core.operator.Operator2`]

        >>> from pennylane.core.operator import Operator2
        >>> class MyClass(Operator2):
        ...     dynamic_argnames = ("phi",)
        ...
        ...     def __init__(self, phi, wires):
        ...         super().__init__(phi, wires)
        ...
        ...     def pow(self, z):
        ...         return [MyClass(self.phi*z, self.wires)]
        ...
        >>> MyClass(0.5, 0).pow(2)
        [MyClass(phi=1.0, wires=[0])]
        """
        # Child methods may call super().pow(z%period) where op**period = I
        # For example, PauliX**2 = I, SX**4 = I, TShift**3 = I (for qutrit)
        # Hence we define the non-negative integer cases here as a repeated list
        if z == 0:
            return []
        if isinstance(z, int) and z > 0:
            if QueuingManager.recording():
                return [apply(self) for _ in range(z)]
            return [copy(self) for _ in range(z)]
        raise PowUndefinedError

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        # return self so pre-constructed Observables can be queued and returned in
        # a single statement
        return self

    @classproperty
    @classmethod
    def has_adjoint(cls) -> bool:
        """Bool: Whether or not the Operator can compute its own adjoint.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.adjoint != Operator2.adjoint

    def adjoint(self) -> "Operator2":  # pylint:disable=no-self-use
        """Create an operation that is the adjoint of this one. Used to simplify
        :class:`~.Adjoint` operators constructed by :func:`~.adjoint`.
        TODO: [sc-120844] Fix docstring after Adjoint is added

        Adjointed operations are the conjugated and transposed version of the
        original operation. Adjointed ops are equivalent to the inverted operation for unitary
        gates.

        ``Operator2.adjoint`` can be optionally defined by Operator developers, while :func:`~.adjoint`
        is the entry point for constructing generic adjoint representations.

        Returns:
            The adjointed operation.

        >>> from pennylane.core.operator import Operator2
        >>> class MyClass(Operator2):
        ...     dynamic_argnames = ("phi",)
        ...
        ...     def __init__(self, phi, wires):
        ...         super().__init__(phi, wires)
        ...
        ...     def adjoint(self):
        ...         return self
        ...
        >>> op = MyClass(0.5, wires=0).adjoint()
        >>> op
        MyClass(phi=0.5, wires=[0])
        """
        raise AdjointUndefinedError

    def map_wires(self, wire_map: dict[Hashable, Hashable]) -> "Operator2":
        """Returns a new operator with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .Operator2: new operator
        """
        new_args = dict(self.arguments)

        for n, wires in self.wire_args.items():
            # Flattening/unflattening allows mapping hybrid wire arguments
            leaves, tree = flatten(wires, is_leaf=lambda w: isinstance(w, Wires))
            mapped_leaves = [Wires([wire_map.get(w, w) for w in leaf]) for leaf in leaves]
            new_args[n] = unflatten(mapped_leaves, tree)

        for n, arg in self.hybrid_args.items():
            if n in self.wire_argnames:
                continue
            leaves, tree = flatten(arg, is_leaf=_is_op)
            leaves = [
                leaf.map_wires(wire_map) if isinstance(leaf, Operator2) else leaf for leaf in leaves
            ]
            new_args[n] = unflatten(leaves, tree)

        return type(self)(**new_args)

    def simplify(self) -> "Operator2":
        """Reduce the depth of nested operators to the minimum.

        Returns:
            .Operator2: simplified operator
        """
        return self

    # ------------------------------------------------------------------------
    # ----------------------- Operator representations -----------------------
    # ------------------------------------------------------------------------
    # pylint: disable=unused-argument,comparison-with-callable,no-self-use

    @staticmethod
    def compute_matrix(*args, **kwargs) -> TensorLike:
        """Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`.Operator2.matrix` and :func:`qp.matrix() <pennylane.matrix>`

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            tensor_like: matrix representation
        """
        raise MatrixUndefinedError

    @classproperty
    @classmethod
    def has_matrix(cls) -> bool:
        """Bool: Whether or not the Operator returns a defined matrix.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.compute_matrix != Operator2.compute_matrix or cls.matrix != Operator2.matrix

    def matrix(self, wire_order: WiresLike | None = None) -> TensorLike:
        """Representation of the operator as a matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        If the matrix depends on trainable parameters, the result
        will be cast in the same autodifferentiation framework as the parameters.

        A ``MatrixUndefinedError`` is raised if the matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator2.compute_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires

        Returns:
            tensor_like: matrix representation
        """
        canonical_matrix = self.compute_matrix(**self.arguments)
        return self._expand_canonical_matrix(canonical_matrix, wire_order)

    def _expand_canonical_matrix(self, canonical_matrix, wire_order) -> TensorLike:
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
        """Representation of the operator as a sparse matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator2.sparse_matrix`

        Args:
            *args (tuple): Positional arguments of the operator
            format (str): format of the returned scipy sparse matrix, for example 'csr'
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation
        """
        raise SparseMatrixUndefinedError

    @classproperty
    @classmethod
    def has_sparse_matrix(cls) -> bool:
        """Bool: Whether the Operator returns a defined sparse matrix.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return (
            cls.compute_sparse_matrix != Operator2.compute_sparse_matrix
            or cls.sparse_matrix != Operator2.sparse_matrix
        )

    def sparse_matrix(self, wire_order: WiresLike | None = None, format="csr") -> spmatrix:
        """Representation of the operator as a sparse matrix in the computational basis.

        If ``wire_order`` is provided, the numerical representation considers the position of the
        operator's wires in the global wire order. Otherwise, the wire order defaults to the
        operator's wires.

        A ``SparseMatrixUndefinedError`` is raised if the sparse matrix representation has not been defined.

        .. seealso:: :meth:`~.Operator2.compute_sparse_matrix`

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires
            format (str): format of the returned scipy sparse matrix, for example 'csr'

        Returns:
            scipy.sparse._csr.csr_matrix: sparse matrix representation

        """
        canonical_sparse_matrix = self.compute_sparse_matrix(**self.arguments, format=format)
        return self._expand_canonical_matrix(canonical_sparse_matrix, wire_order).asformat(format)

    @staticmethod
    def compute_decomposition(*args, **kwargs) -> list["Operator2"]:
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator2.decomposition`.

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            list[Operator2]: decomposition of the operator
        """
        raise DecompositionUndefinedError

    @classproperty
    @classmethod
    def has_decomposition(cls) -> bool:
        """Bool: Whether or not the Operator returns a defined decomposition.

        This is a class-level check (no per-instance dispatch): ``True`` if
        ``compute_decomposition`` or ``decomposition`` is overridden, or if graph decomposition
        rules are registered for the operator type. Per-instance rule applicability is resolved
        in :meth:`~.Operator2.decomposition`, not here.
        """
        return (
            cls.compute_decomposition != Operator2.compute_decomposition
            or cls.decomposition != Operator2.decomposition
            or qp.decomposition.has_decomp(cls)
        )

    def decomposition(self) -> list["Operator2"]:
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator2.compute_decomposition`.

        Returns:
            list[Operator2]: decomposition of the operator
        """
        if type(self).compute_decomposition != Operator2.compute_decomposition:
            return self.compute_decomposition(**self.arguments)

        for decomp in qp.list_decomps(self):
            if decomp.is_applicable(**self.arguments):
                with AnnotatedQueue() as q:
                    decomp(**self.arguments)
                if QueuingManager.recording():
                    # no need for copies if we just use queue method
                    _ = [op.queue() for op in q.queue]
                return q.queue

        raise DecompositionUndefinedError

    @staticmethod
    def compute_eigvals(*args, **kwargs) -> TensorLike:
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`Operator2.eigvals() <.eigvals>` and :func:`qp.eigvals() <pennylane.eigvals>`

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            tensor_like: eigenvalues
        """
        raise EigvalsUndefinedError

    def eigvals(self) -> TensorLike:
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

        .. seealso:: :meth:`~.Operator2.compute_eigvals` and :func:`qp.eigvals() <pennylane.eigvals>`

        Returns:
            tensor_like: eigenvalues
        """

        try:
            return self.compute_eigvals(**self.arguments)
        except EigvalsUndefinedError as e:
            # By default, compute the eigenvalues from the matrix representation if one is defined.
            if self.has_matrix:  # pylint: disable=using-constant-test
                return math.linalg.eigvals(self.matrix())
            raise EigvalsUndefinedError from e

    @staticmethod
    def compute_diagonalizing_gates(*args, **kwargs) -> list["Operator2"]:
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Operator2.diagonalizing_gates`.

        Args:
            *args (tuple): Positional arguments of the operator
            **kwargs (dict): Keyword arguments of the operator

        Returns:
            list[.Operator2]: list of diagonalizing gates
        """
        raise DiagGatesUndefinedError

    @classproperty
    @classmethod
    def has_diagonalizing_gates(cls) -> bool:
        """Bool: Whether or not the Operator returns defined diagonalizing gates.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        # Operators may overwrite `diagonalizing_gates` instead of `compute_diagonalizing_gates`
        # Currently, those are mostly classes from the operator arithmetic module.
        return (
            cls.compute_diagonalizing_gates != Operator2.compute_diagonalizing_gates
            or cls.diagonalizing_gates != Operator2.diagonalizing_gates
        )

    def diagonalizing_gates(self) -> list["Operator2"]:
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator2.compute_diagonalizing_gates`.

        Returns:
            list[.Operator2] or None: a list of operators
        """
        return self.compute_diagonalizing_gates(**self.arguments)

    def terms(self) -> tuple[list[TensorLike], list["Operator2"]]:
        r"""Representation of the operator as a linear combination of other operators.

        .. math:: O = \sum_i c_i O_i

        A ``TermsUndefinedError`` is raised if no representation by terms is defined.

        Returns:
            tuple[list[tensor_like or float], list[.Operator2]]: list of coefficients :math:`c_i`
            and list of operations :math:`O_i`
        """
        raise TermsUndefinedError

    @classproperty
    @classmethod
    def has_generator(cls) -> bool:
        """Bool: Whether or not the Operator returns a defined generator.

        Note: Child classes may have this as an instance property instead of as a class property.
        """
        return cls.generator != Operator2.generator

    def generator(self) -> "Operator2":
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U.generator() # doctest: +SKIP
        0.5 * Y(0) + Z(0) @ X(1)

        The generator may also be provided in the form of a dense or sparse Hamiltonian
        (using :class:`.LinearCombination` and :class:`.SparseHamiltonian` respectively).

        """
        raise GeneratorUndefinedError(f"Operation {self.name} does not have a generator")

    # ------------------------------------------------------------------------
    # ------------------------ General dunder methods ------------------------
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        # NOTE: Handle special case for single wire non-parameteric
        # operators like 'repr(qp.X(wires=0)) = X(0)'
        non_wire_args = (
            self.dynamic_argnames
            + self.static_argnames
            + self.compilable_argnames
            + self.hybrid_argnames
        )
        if not non_wire_args and len(self.wire_argnames) == 1:
            wire_arg = self.arguments[self.wire_argnames[0]]
            if isinstance(wire_arg, Wires) and len(wire_arg) == 1:
                return f"{self.name}({wire_arg.tolist()[0]!r})"

        inputs = []

        remove_dyn_keywords = not (
            self.static_argnames or self.compilable_argnames or self.hybrid_argnames
        )

        for key, value in self.arguments.items():
            # Hybrid wire arguments.
            if key in self.wire_argnames and key in self.hybrid_argnames:
                leaves, tree = flatten(value, is_leaf=_is_wires)
                leaves = [w.tolist() if isinstance(w, Wires) else w for w in leaves]
                value = unflatten(leaves, tree)

            if key in self.dynamic_argnames and remove_dyn_keywords:
                inputs.append(f"{value}")
            else:
                inputs.append(f"{key}={value}")

        inputs = ", ".join(inputs)
        return f"{self.name}({inputs})"

    def __str__(self) -> str:
        if self.is_abstract and self.has_fixed_sig:
            return self.name
        return repr(self)

    def __hash__(self) -> int:
        serialized_dynamic = tuple(
            _canonicalize_dynamic(self.arguments[d], self.name) for d in self.dynamic_argnames
        )
        serialized_wires = tuple(
            self.arguments[w] for w in self.wire_argnames if w not in self.hybrid_argnames
        )
        serialized_static = tuple(str(self.arguments[s]) for s in self.static_argnames)
        serialized_compilable = tuple(str(self.arguments[c]) for c in self.compilable_argnames)

        serialized_hybrid = []
        for h in self.hybrid_argnames:
            leaves, tree = flatten(self.arguments[h], is_leaf=_is_hash_leaf)
            ser_leaves = tuple(
                l if isinstance(l, (Operator2, Wires)) else _canonicalize_dynamic(l) for l in leaves
            )
            serialized_hybrid.append((ser_leaves, tree))

        return hash(
            (
                self.name,
                serialized_dynamic,
                serialized_wires,
                serialized_static,
                serialized_compilable,
                tuple(serialized_hybrid),
            )
        )

    def __eq__(self, other) -> bool:
        return qp.equal(self, other)

    def __copy__(self) -> "Operator2":
        cls = type(self)
        copied_op = cls.__new__(cls)
        for attr, value in vars(self).items():
            setattr(copied_op, attr, value)
        return copied_op

    def __deepcopy__(self, memo) -> "Operator2":
        cls = type(self)
        copied_op = cls.__new__(cls)

        # The memo dict maps object ID to object, and is required by
        # the deepcopy function to keep track of objects it has already
        # deep copied.
        memo[id(self)] = copied_op

        for attr, value in vars(self).items():
            setattr(copied_op, attr, deepcopy(value, memo))
        return copied_op

    # ------------------------------------------------------------------------
    # ------------------ Operator arithmetic dunder methods ------------------
    # ------------------------------------------------------------------------

    def __add__(self, other: Operator | TensorLike) -> Operator:
        """The addition operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator):
            return qp.sum(self, other, lazy=False)
        if isinstance(other, TensorLike):
            if not qp.math.is_abstract(other) and qp.math.allequal(other, 0):
                return self
            return qp.sum(
                self,
                qp.s_prod(scalar=other, operator=qp.Identity(self.wires), lazy=False),
                lazy=False,
            )
        return NotImplemented

    __radd__ = __add__

    def __mul__(self, other: Callable | TensorLike) -> Operator:
        """The scalar multiplication between scalars and Operators."""
        if callable(other):
            return qp.pulse.ParametrizedHamiltonian([other], [self])
        if isinstance(other, TensorLike):
            return qp.s_prod(scalar=other, operator=self, lazy=False)
        return NotImplemented

    def __truediv__(self, other: TensorLike):
        """The division between an Operator and a number."""
        if isinstance(other, TensorLike):
            return self.__mul__(1 / other)
        return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other: Operator) -> Operator:
        """The product operation between Operator objects."""
        return qp.prod(self, other, lazy=False) if isinstance(other, Operator) else NotImplemented

    def __sub__(self, other: Operator | TensorLike) -> Operator:
        """The subtraction operation of Operator-Operator objects and Operator-scalar."""
        if isinstance(other, Operator):
            return self + qp.s_prod(-1, other, lazy=False)
        if isinstance(other, TensorLike):
            return self + (qp.math.multiply(-1, other))
        return NotImplemented

    def __rsub__(self, other: Operator | TensorLike) -> Operator:
        """The reverse subtraction operation of Operator-Operator objects and Operator-scalar."""
        return -self + other

    def __neg__(self) -> Operator:
        """The negation operation of an Operator object."""
        return qp.s_prod(scalar=-1, operator=self, lazy=False)

    def __pow__(self, other: TensorLike) -> Operator:
        r"""The power operation of an Operator object."""
        if isinstance(other, TensorLike):
            return qp.pow(self, z=other)
        return NotImplemented

    # ----------------------------------------------------------------------------
    # ------------------ Private utililities for initialization ------------------
    # ----------------------------------------------------------------------------

    def _flatten(self) -> FlatPytree:
        """Serialize the operation into dynamic and static components.

        Returns:
            data, metadata: The dynamic and static components.

        See ``Operator2._unflatten``.

        The dynamic component can be recursive and include other operators.

        The metadata **must** be hashable. If the static data contains a non-hashable component, then this
        method and ``Operator2._unflatten`` should be overridden to provide a hashable version of the static data.

        **Example:**

        # TODO: [sc-120453] Update code examples after migration as __repr__ has changed
        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> op._flatten() # doctest: +SKIP
        (([1.2, 2.3, 3.4], [Wires([0])], []), ())
        >>> qp.Rot._unflatten(*op._flatten()) # doctest: +SKIP
        Rot(phi=1.2, theta=2.3, omega=3.4, wires=[0])
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten() # doctest: +SKIP
        (([1.2], [Wires([0, 1])], []), ('XY',))
        """
        # Sort dynamic data as dynamic_args, wire_args, hybrid_args
        dyn_args = [self._bound_args.arguments[d] for d in self.dynamic_argnames]
        wires = [self._bound_args.arguments[w] for w in self.wire_argnames]
        hybrid_args = [
            self._bound_args.arguments[h]
            for h in self.hybrid_argnames
            if h not in self.wire_argnames
        ]
        leaves = (dyn_args, wires, hybrid_args)

        # Put static/compilable args in hashable_data
        hashable_argnames = self.static_argnames or self.compilable_argnames
        hashable_data = tuple(self._bound_args.arguments[name] for name in hashable_argnames)
        return leaves, hashable_data

    @classmethod
    def _unflatten(cls, data: Iterable[Any], metadata: Hashable):
        """Recreate an operation from its serialized format.

        Args:
            data: the dynamic component of the operation
            metadata: the static component of the operation.

        The output of ``Operator2._flatten`` and the class type must be sufficient to reconstruct the original
        operation with ``Operator2._unflatten``.

        **Example:**

        # TODO: [sc-120453] Update code examples after migration as __repr__ has changed
        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> op._flatten() # doctest: +SKIP
        (([1.2, 2.3, 3.4], [Wires([0])], []), ())
        >>> qp.Rot._unflatten(*op._flatten()) # doctest: +SKIP
        Rot(phi=1.2, theta=2.3, omega=3.4, wires=[0])
        """
        args = {}

        # Process dynamic data
        for name, value in zip(cls.dynamic_argnames, data[0], strict=True):
            args[name] = value
        for name, value in zip(cls.wire_argnames, data[1], strict=True):
            args[name] = value

        non_wire_hybrid_argnames = (
            name for name in cls.hybrid_argnames if name not in cls.wire_argnames
        )
        for i, name in enumerate(non_wire_hybrid_argnames):
            args[name] = data[2][i]

        # Process static data
        hashable_argnames = cls.static_argnames or cls.compilable_argnames
        for name, value in zip(hashable_argnames, metadata, strict=True):
            args[name] = value

        with QueuingManager.stop_recording():
            with pause():
                return cls(**args)

    def _check_batching(self):
        """Check if the expected numbers of dimensions of parameters coincides with the
        ones received and sets the ``_batch_size`` attribute.

        The check always passes and sets the ``_batch_size`` to ``None`` for the default
        ``Operator.ndim_params`` property but subclasses may overwrite it to define fixed
        expected numbers of dimensions, allowing to infer a batch size.
        """
        self._batch_size = None
        dynamic_args = tuple(self.dynamic_args.values())

        ndims = tuple(math.ndim(arg) for arg in dynamic_args)
        if any(len(math.shape(arg)) >= 1 and math.shape(arg)[0] is None for arg in dynamic_args):
            # if the batch dimension is unknown, then skip the validation
            # this happens when a tensor with a partially known shape is passed, e.g. (None, 12),
            # typically during compilation of a function decorated with jax.jit or tf.function
            return  # pragma: no cover

        self._ndim_params = ndims
        if ndims != self.ndim_params:
            ndims_matches = [
                (ndim == exp_ndim, ndim == exp_ndim + 1)
                for ndim, exp_ndim in zip(ndims, self.ndim_params, strict=True)
            ]
            if not all(correct or batched for correct, batched in ndims_matches):
                raise ValueError(
                    f"{self.name}: wrong number(s) of dimensions in parameters. "
                    f"Parameters with ndims {ndims} passed, {self.ndim_params} expected."
                )

            first_dims = [
                math.shape(arg)[0]
                for (_, batched), arg in zip(ndims_matches, dynamic_args, strict=True)
                if batched
            ]
            if not math.allclose(first_dims, first_dims[0]):
                raise ValueError(
                    "Broadcasting was attempted but the broadcasted dimensions "
                    f"do not match: {first_dims}."
                )

            self._batch_size = first_dims[0]

    def _bind_primitive(self):
        """Bind the operator plxpr primitive."""
        # Skip if program capture is disabled
        if not enabled():
            return

        pos_args = [self.arguments[d] for d in self.dynamic_argnames]

        wire_lens = []
        for name, value in self.wire_args.items():
            if name not in self.hybrid_argnames:
                pos_args.extend(value)
                wire_lens.append(len(value))

        hybrid_lens, hybrid_trees = [], []
        forward_mask = []
        for name in self.hybrid_argnames:
            leaves, tree, mask = _process_bind_hybrid_arg(
                self.arguments[name], is_wire_arg=name in self.wire_argnames
            )
            forward_mask.extend(mask)
            pos_args.extend(leaves)
            hybrid_lens.append(len(leaves))
            hybrid_trees.append(tree)

        static_args = {}
        for name in self.static_argnames + self.compilable_argnames:
            # Pytree flattening is a simple way to make static arguments hashable
            value = self.arguments[name]
            leaves, tree = flatten(value)
            static_args[name] = (tuple(leaves), tree)

        res = operator_p.bind(
            *pos_args,
            op_cls=type(self),
            wire_lens=wire_lens,
            hybrid_lens=hybrid_lens,
            hybrid_trees=hybrid_trees,
            forward_mask=forward_mask,
            n_ctrls=0,
            adjoint=False,
            **static_args,
        )
        # If we bind the primitive outside a tracing context but with program capture enabled,
        # `res`` will be a concrete operator, not an abstract tracer, so we don't save it.
        if math.is_abstract(res):
            self.tracer = res

    def __init_subclass__(cls: type["Operator2"], is_baseclass=False) -> None:
        cls._sig = signature(cls)
        if is_baseclass:
            return

        # Argnames setup
        for attr in (
            "dynamic_argnames",
            "wire_argnames",
            "static_argnames",
            "hybrid_argnames",
            "compilable_argnames",
        ):
            if isinstance(v := getattr(cls, attr), str):
                setattr(cls, attr, (v,))

        _init_subclass_validate_argnames(cls)
        _init_subclass_arg_specs_setup(cls)
        _init_subclass_wire_sizes_setup(cls)
        _init_subclass_add_dynamic_properties(cls)
        register_pytree(cls, cls._flatten, cls._unflatten)


# ---------------------------------------------------------------------------------
# ------------------------- Instance construction helpers -------------------------
# ---------------------------------------------------------------------------------


def _init_wires(op: Operator2):
    """Initialize operator wires.

    * Union of all wire_argnames into _wires
    * Flatten pytree arguments and look for operators
    * Append operator argument wires to _wires
    """
    # pylint: disable=protected-access
    all_algorithmic_wires = []

    for wname, wsize in zip(op.wire_argnames, op.wire_sizes, strict=True):
        if wname not in op.hybrid_argnames:
            warg = op._bound_args.arguments[wname]
            canonical_wires = warg if isinstance(warg, AbstractWires) else Wires(warg)
            op._bound_args.arguments[wname] = canonical_wires

            if wsize is not None and len(canonical_wires) != wsize:
                raise ValueError(
                    f"Incorrect number of wires for '{op.name}.{wname}'. Expected {wsize} "
                    f"wires but got {len(canonical_wires)}."
                )

            # Work wires are NOT included in the full wires list.
            if wname not in ("work_wires", "work_wire"):
                all_algorithmic_wires.append(canonical_wires)

        # Pytree wires handling
        else:
            leaves, _ = flatten(op._bound_args.arguments[wname], is_leaf=_is_wires)
            if not all(isinstance(l, (Wires, AbstractWires)) for l in leaves):
                raise ValueError(
                    f"Hybrid wires argument '{wname}' is invalid. All leaf values must be "
                    "cast to 'qp.wires.Wires'."
                )

            # Work wires are NOT included in the full wires list.
            if wname not in ("work_wires", "work_wire"):
                all_algorithmic_wires.extend(leaves)

    for hname in op.hybrid_argnames:
        if hname in op.wire_argnames:
            continue
        leaves, _ = flatten(op._bound_args.arguments[hname], is_leaf=_is_op)
        ops = filter(_is_op, leaves)
        all_algorithmic_wires.extend(op.wires for op in ops)

    if all_algorithmic_wires and isinstance(all_algorithmic_wires[0], AbstractWires):
        total_wires = sum(w.num_wires for w in all_algorithmic_wires)
        op._wires = AbstractWires(total_wires)
    else:
        op._wires = Wires.all_wires(all_algorithmic_wires)


def _init_arg_types(op: Operator2) -> None:
    """Validate the provided arguments against their expected type. This method
    only performs validation on operators if ``op.arg_specs`` is defined.
    """
    # arg_specs not present or there are no arguments
    if not op.arg_specs:
        return

    for name, exp_type in op.arg_specs.items():
        argval = op.arguments[name]
        if name in op.wire_argnames:  # pragma: no cover
            # This branch is effectively unreachable since a mismatch between the actual
            # and expected length for a wire argument is validated in __init_wires. We will
            # only ever reach this branch if __validate_arg_types is called manually.
            msg = f"Expected '{name}' to have length {exp_type.num_wires}, but got {argval}."
            assert exp_type.num_wires == -1 or exp_type.num_wires == len(argval), msg
            continue

        # Dynamic argument
        if isinstance(argval, (Number, list, tuple)):
            argval = np.array(argval)
        # If the argument is batched, compare the shape other than that batch dimension
        arg_shape = argval.shape if isinstance(argval, AbstractArray) else math.shape(argval)
        either_is_ellipsis = exp_type.shape is Ellipsis or arg_shape is Ellipsis
        is_broadcasted = False if either_is_ellipsis else len(arg_shape) > exp_type.ndim
        unbatched_shape = arg_shape[1:] if is_broadcasted else arg_shape

        argval_dtype = (
            argval.dtype if isinstance(argval, AbstractArray) else math.get_dtype_name(argval)
        )
        comparison_abstract_type = AbstractArray(unbatched_shape, np.dtype(argval_dtype))

        # Check if either shape or dtype is not compatible
        if not exp_type.is_compatible_with(comparison_abstract_type):
            # Isolate if it's a pure dtype issue by comparing with a mock type that has the
            # expected shape but the actual dtype
            actual_dtype = argval_dtype
            if not exp_type.is_compatible_with(AbstractArray(exp_type.shape, actual_dtype)):
                raise ValueError(
                    f"Parameter '{name}' does not match the operator's expected 'arg_specs' dtype. "
                    f"Expected {exp_type.dtype} but received {actual_dtype}."
                )

            # If dtype is fine, must be a shape mismatch
            broadcast_msg = " (non-broadcasting dimensions)" if is_broadcasted else ""
            raise ValueError(
                f"Parameter '{name}' does not match the operator's expected 'arg_specs' shape. "
                f"Expected {exp_type.shape}{broadcast_msg} but received {arg_shape}."
            )

        # NOTE: If the argval is an abstract type, we wish to canonicalize it to the
        # spec in 'arg_specs' in order to have a single source of truth.
        if isinstance(argval, AbstractArray):
            new_argval = AbstractArray(arg_shape, exp_type.dtype)
            # pylint: disable=protected-access
            # FIX: Hacky way to set attribute of a frozen dataclass
            object.__setattr__(new_argval, "_weak_type", exp_type._weak_type)
            op.arguments[name] = new_argval


# -------------------------------------------------------------------------------
# ----------------------- Subclass initialization helpers -----------------------
# -------------------------------------------------------------------------------


def _init_subclass_validate_argnames(cls: type[Operator2]) -> None:
    """Validate the values inside all ``**_argnames`` for an operator class."""
    if (cls.hybrid_argnames or cls.static_argnames) and cls.compilable_argnames:
        raise TypeError(
            "Operators can only contain 'static_argnames' and 'hybrid_argnames', or "
            "'compilable_argnames', not both."
        )

    # dynamic/wire/static/compilable argnames must be disjoint.
    seen: dict[str, str] = {}
    for group_name in (
        "dynamic_argnames",
        "wire_argnames",
        "static_argnames",
        "compilable_argnames",
    ):
        for name in getattr(cls, group_name):
            if other := seen.get(name):
                raise TypeError(
                    f"Argument '{name}' appears in both '{other}' and '{group_name}'; "
                    "dynamic, wire, static, and compilable argnames must not overlap."
                )
            seen[name] = group_name

    # hybrid_argnames may overlap with wire_argnames, but not with the others.
    hybrid = set(cls.hybrid_argnames)
    non_wire = {n for n, g in seen.items() if g != "wire_argnames"}
    if bad := hybrid & non_wire:
        raise TypeError(
            f"hybrid_argnames {bad} overlap with dynamic, static, or "
            "compilable argnames; hybrid_argnames may only overlap with wire_argnames."
        )

    # Every named signature parameter must appear in at least one *_argnames.
    sig_params = set(cls._sig.parameters.keys())  # pylint: disable=protected-access
    if unclassified := sig_params - set(seen.keys()) - hybrid:
        raise TypeError(
            f"The following parameters of '{cls.__name__}' are not classified in "
            f"any argnames tuples: {unclassified}."
        )


def _init_subclass_arg_specs_setup(cls: type[Operator2]) -> None:
    """Set up ``arg_specs`` for ``Operator2`` subclasses."""
    arg_specs = cls.arg_specs or {}
    disallowed_argnames = cls.hybrid_argnames + cls.compilable_argnames + cls.static_argnames

    if names := (set(arg_specs.keys()) & set(disallowed_argnames)):
        raise TypeError(
            f"{cls.__name__}.arg_specs can only contain dynamic and wire arguments, but got {names}."
        )

    cls.has_fixed_sig = (
        set(arg_specs.keys()) == set(cls.dynamic_argnames + cls.wire_argnames)
        and len(disallowed_argnames) == 0
    )

    for name, exp_type in arg_specs.items():
        canonical_exp_type = exp_type
        if isinstance(exp_type, type) and issubclass(exp_type, Number):
            canonical_exp_type = AbstractArray((), exp_type)
            cls.arg_specs[name] = canonical_exp_type

        if not canonical_exp_type.shape_fixed:
            cls.has_fixed_sig = False


def _init_subclass_wire_sizes_setup(cls: type[Operator2]) -> None:
    """Set up ``wire_sizes`` for ``Operator2`` subclasses."""
    arg_specs = cls.arg_specs or {}

    if cls.wire_sizes is None:
        cls.wire_sizes = tuple(
            (
                None
                if name not in arg_specs or arg_specs[name].num_wires == -1
                else arg_specs[name].num_wires
            )
            for name in cls.wire_argnames
        )
        return

    if not isinstance(cls.wire_sizes, Sequence):
        cls.wire_sizes = (cls.wire_sizes,)

    if len(cls.wire_sizes) != len(cls.wire_argnames):
        raise TypeError("'wire_sizes' must have the same length as 'wire_argnames'.")

    for wname, wsize in zip(cls.wire_argnames, cls.wire_sizes, strict=True):
        # Hybrid wire arguments' entry in wire_sizes must always be ``None``. Hybrid arguments
        # can be arbitrary pytrees by design
        if wname in cls.hybrid_argnames and wsize is not None:
            raise TypeError(
                f"Expected wire_size == None for '{wname}' as it is a hybrid wire argument."
            )

        if not ((isinstance(wsize, int) and wsize > 0) or wsize is None):
            raise TypeError(
                f"'{cls.__name__}.wire_sizes' is invalid. 'wire_sizes' must be a sequence "
                f"of positive integers or 'None' values, but got {cls.wire_sizes}."
            )

        # If the wire argument is in arg_specs, the entries in arg_specs
        # and wire_sizes must match. Arbitrary number of wires is denoted by ``None`` and
        # ``-1`` in wire_sizes and arg_specs respectively.
        if (et := arg_specs.get(wname, None)) is not None:
            nwires = et.num_wires
            if (nwires == -1 and wsize is not None) or (nwires not in (-1, wsize)):
                cname = cls.__name__
                raise TypeError(
                    f"Number of wires specified for '{wname}' does not match the declared "
                    f"type in {cname}.arg_specs and {cname}.wire_sizes. Got "
                    f"{nwires} and {wsize} respectively."
                )


def _init_subclass_add_dynamic_properties(cls: type[Operator2]) -> None:
    """Create dynamic properties for an operator using its signature."""
    # pylint: disable=protected-access
    for name in cls._sig.parameters:
        if not hasattr(cls, name):
            dyn_property = partial(_init_subclass_dynamic_property, name=name)
            setattr(cls, name, property(dyn_property))


def _init_subclass_dynamic_property(self: Operator2, name: str) -> Any:
    """Dynamic property for an argument called ``name``."""
    # pylint: disable=protected-access
    if "_bound_args" in vars(self) and name in self._bound_args.arguments:
        return self._bound_args.arguments[name]

    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'."
    )  # pragma: no cover


# -------------------------------------------------------------------------------
# --------------------------- Program capture helpers ---------------------------
# -------------------------------------------------------------------------------


if has_jax:
    # pylint: disable=import-outside-toplevel,ungrouped-imports
    from pennylane.capture.custom_primitives import QpPrimitive

    operator_p = QpPrimitive("operator")
    operator_p.prim_type = "operator"

    # pylint: disable=too-many-arguments,unused-argument
    @operator_p.def_impl
    def _op_impl(
        *all_args,
        op_cls,
        wire_lens,
        hybrid_lens,
        hybrid_trees,
        forward_mask,
        n_ctrls=0,
        adjoint=False,
        **static_args,
    ):
        args = {name: unflatten(*value) for name, value in static_args.items()}
        i = 0

        for name in op_cls.dynamic_argnames:
            args[name] = all_args[i]
            i += 1

        wire_lens_iter = iter(wire_lens)
        for name in op_cls.wire_argnames:
            if name not in op_cls.hybrid_argnames:
                len_ = next(wire_lens_iter)
                # We can safely cast to `int` inside the concrete impl because there
                # there should not be any abstract values when calling the concrete impl.
                args[name] = Wires(tuple(int(w) for w in all_args[i : i + len_]))
                i += len_

        # Reorder hybrid args such that hybrid wire args are first
        for name, len_, tree in zip(op_cls.hybrid_argnames, hybrid_lens, hybrid_trees, strict=True):
            leaves = all_args[i : i + len_]
            args[name] = unflatten(leaves, tree)
            i += len_

        if n_ctrls:
            control_wires = all_args[i : i + n_ctrls]
            i += n_ctrls
            control_values = all_args[i:]
            assert len(control_wires) == len(control_values)
        else:
            control_wires = control_values = ()

        op = type.__call__(op_cls, **args)
        if adjoint:
            op = type.__call__(qp.ops.op_math.Adjoint2, op)
        if n_ctrls:
            op = type.__call__(
                qp.ops.op_math.ControlledOp2,
                op,
                control_wires=control_wires,
                control_values=control_values,
            )
        return op

    @operator_p.def_abstract_eval
    def _op_aval(*_, **__):
        AbstractOperator = _get_abstract_operator()
        return AbstractOperator()

else:  # pragma: no cover
    operator_p = None


def pop_op_eqns(ops: Iterable):
    """Delete the jaxpr equations for operators that have been used as data.

    These equations must be deleted because operators used as data are treated as
    pytrees wrapping dynamic data rather than instructions. Thus, the equation that
    corresponds to the operator as an instruction should be removed.
    """
    old_eqns = []

    for op in ops:
        if op.tracer is not None:
            # pylint: disable=protected-access
            frame = op.tracer._trace.frame
            assert frame.auto_dce is False  # eqns are stored differently if this is enabled

            # for some reason the frame now wraps equations in lambdas
            eqn = op.tracer.parent
            old_eqns.append(eqn)
            frame.tracing_eqns = [r for r in frame.tracing_eqns if r() is not eqn]

            # delete reference to tracer after its equation has been deleted
            op.tracer = None

    return old_eqns


def _op_arg_forward_mask(op: Operator2) -> list[bool]:
    """Build ``forward_mask`` entries for an operator argument."""
    op_leaves, _ = flatten(op, is_leaf=_is_wires)
    hybrid_mask = []
    for op_leaf in op_leaves:
        if isinstance(op_leaf, Wires):
            hybrid_mask.extend([False] * len(op_leaf))
        else:
            hybrid_mask.append(True)
    return hybrid_mask


def _process_bind_hybrid_arg(hybrid_val, is_wire_arg: bool) -> tuple[list, Any, list[bool]]:
    """Process a hybrid argument for binding an operator primitive."""
    partial_leaves, _ = flatten(hybrid_val, is_leaf=_is_op)
    _ = pop_op_eqns(filter(_is_op, partial_leaves))

    leaves, tree = flatten(hybrid_val)
    if is_wire_arg:
        return leaves, tree, [False] * len(leaves)

    hybrid_mask: list[bool] = []
    for partial_leaf in partial_leaves:
        if isinstance(partial_leaf, Operator2):
            hybrid_mask.extend(_op_arg_forward_mask(partial_leaf))
        else:
            hybrid_mask.append(False)

    return leaves, tree, hybrid_mask


# -----------------------------------------------------------------------------
# --------------------------- Miscelleneous helpers ---------------------------
# -----------------------------------------------------------------------------


def _format_label_arg(x, decimals, cache):
    """Format a scalar parameter or retrieve/store a matrix-valued parameter
    from/to cache, formatting its position in the cache as parameter string."""
    if len(math.shape(x)) == 0:
        # Scalar case
        if decimals is None:
            return ""
        try:
            return format(math.toarray(x), f".{decimals}f")
        except ValueError:  # pragma: no cover
            # If the parameter can't be displayed as a float
            return format(x)

    if cache is None or not isinstance(mat_cache := cache.get("matrices", None), list):
        # No caching; matrices are not printed out fully, so no printing of this parameter
        return ""

    # Retrieve matrix location in cache, or write the matrix to cache as new entry
    for i, mat in enumerate(mat_cache):
        if math.shape(x) == math.shape(mat) and math.allclose(x, mat):
            return f"M{i}"
    mat_num = len(mat_cache)
    mat_cache.append(x)
    return f"M{mat_num}"


def _is_wires(val: Any) -> bool:
    """Check whether a value is a Wires object."""
    return isinstance(val, Wires)


def _is_op(val: Any) -> bool:
    """Check whether a value is an Operator2 object."""
    return isinstance(val, Operator2)


def _canonicalize_dynamic(d, op_name=None) -> Hashable:
    """Canonicalize dynamic data for hashing."""

    def _mod_and_round(x, mod_val):
        if qp.math.asarray(x).dtype == bool:
            return x
        x = x if mod_val is None else qp.math.real(x) % mod_val
        return qp.math.round(x, 10)

    # Use qp.math.real to take the real part. We may get complex inputs for
    # example when differentiating holomorphic functions with JAX: a complex
    # valued QNode (one that returns qp.state) requires complex typed inputs.
    if op_name is not None and op_name in ("RX", "RY", "RZ", "PhaseShift", "Rot"):
        mod_val = 2 * np.pi
    else:
        mod_val = None

    if isinstance(d, AbstractArray):
        return hash(d)

    # We stringify the data because arrays are unhashable
    return str(id(d) if math.is_abstract(d) else _mod_and_round(d, mod_val))


def _is_hash_leaf(l) -> bool:
    """Check whether a value is a pytree leaf for hashing. For the purpose of
    hashing, wires and operators are considered leaves."""
    return _is_op(l) or _is_wires(l)


class _ArgType(Enum):
    """Enum to keep track of an arguments type."""

    WIRES = auto()
    DYN = auto()
    HYBRID = auto()


def _resolve_arg_kind(cls, name: str) -> _ArgType:
    """Resolves an arguments name to what kind of argument type it is."""
    # Check hybrid first: hybrid args can also appear in wire_argnames
    # and must be treated as hybrid.
    if name in cls.hybrid_argnames:
        return _ArgType.HYBRID
    if name in cls.wire_argnames:
        return _ArgType.WIRES
    return _ArgType.DYN


def _canonicalize_abstract_type(val, kind: _ArgType):
    """Canonicalizes the input into its abstract equivalent.

    Args:
        val (Any): The input value.
        kind (_ArgType): The argument's classification.
            - WIRES: Coerce the value to be an AbstractWires instance.
            - DYN: Flatten into a single, unified AbstractArray
            - HYBRID: Preserve the PyTree structure, mapping internal leaves
                to either AbstractWires or AbstractArray.
    """

    if isinstance(val, (AbstractArray, AbstractWires)):
        return val

    if isinstance(val, type) and issubclass(val, Number):
        return AbstractArray((), val)

    match kind:
        case _ArgType.WIRES:
            # abstractify expects a Wires object for wire-routing, so we sanitize it first
            return abstractify(Wires(val))

        case _ArgType.DYN:
            # A sequence of types is not supported (i.e., [float, float, float])
            # for dynamic args. Ambiguous how to canonicalize it generally.
            if isinstance(val, (list, tuple)) and any(_is_abstract_specifier(x) for x in val):
                raise NotImplementedError(
                    "A sequence of types for a dynamic argument is not "
                    "currently supported. Instead, please use the type "
                    "specifiers found in pennylane.typing."
                )
            # Ensure it behaves like a clean array/scalar leaf before abstractifying
            return abstractify(math.asarray(val))

        case _ArgType.HYBRID:
            # Since abstractify natively handles PyTree recursion and leaves,
            # we can pass the entire structure straight through
            return abstractify(val)

        case _:  # pragma: no cover
            raise ValueError(f"Unknown kind: '{kind}'")


def _is_abstract_specifier(val):
    return isinstance(val, AbstractArray) or (isinstance(val, type) and issubclass(val, Number))


@abstractify.register(OperatorMeta)
def _abstractify_operator_type(op_type: type[Operator2]) -> Operator2:
    """Abstractify a subclass of operator."""

    if op_type.has_fixed_sig:
        return op_type(**op_type.arg_specs)

    raise TypeError(
        f"'{op_type.__name__}' must set 'arg_specs' and cover all dynamic and wire "
        "arguments with fixed abstract types to be abstractified."
    )


@abstractify.register(Operator2)
def _abstractify_operator(op: Operator2) -> Operator2:
    """Abstractify an operator."""
    if op.is_abstract:
        return op

    op_cls = type(op)
    target_args = op_cls.dynamic_argnames + op_cls.hybrid_argnames + op_cls.wire_argnames
    new_args = dict(op.arguments)
    for name in target_args:
        kind = _resolve_arg_kind(op_cls, name)
        new_args[name] = _canonicalize_abstract_type(new_args[name], kind)
    return op_cls(**new_args)


class StatePrepBase2(Operator2, is_baseclass=True):
    """An interface for state-prep operations."""

    @abstractmethod
    def state_vector(self, wire_order: WiresLike | None = None) -> TensorLike:
        """
        Returns the initial state vector for a circuit given a state preparation.

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels
                from the operator's wires

        Returns:
            array: A state vector for all wires in a circuit
        """

    # pylint: disable=unused-argument
    def label(
        self, decimals: int | None = None, base_label: str | None = None, cache: dict | None = None
    ) -> str:
        """The default label for a state prep operation."""
        return base_label or "|Ψ⟩"
