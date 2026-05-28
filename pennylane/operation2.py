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

from abc import ABC
from collections.abc import Hashable, Iterable, Sequence
from copy import copy, deepcopy
from functools import partial
from inspect import BoundArguments, Signature, signature
from typing import Any, ClassVar

import numpy as np
from scipy.sparse import spmatrix

import pennylane as qp
from pennylane import math
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
from pennylane.operation import _UNSET_BATCH_SIZE, FlatPytree, classproperty
from pennylane.pytrees import flatten, register_pytree, unflatten
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike


class Operator2(ABC):
    r"""Base class representing quantum operators.
    TODO: [sc-120453] Fill docstring
    """

    # pylint: disable=too-many-public-methods, too-many-instance-attributes

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
    order. Wires contributed by :class:`~.Operator2` leaves found inside non-wire
    ``hybrid_argnames`` are appended *after* all ``wire_argnames`` wires. The special
    names ``"work_wires"`` and ``"work_wire"`` may be included in ``wire_argnames``
    but their values are excluded from ``op.wires``."""

    dynamic_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that are treated as dynamic. Dynamic arguments are those
    whose concrete values may not be known at compile-time."""

    static_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that are treated as static. Static arguments are those
    whose concrete values are known when capturing the program. Arguments specified
    here are not lowered to a compiler intermediate representation (IR). Alternatively,
    if an argument is of a type that can be lowered, it can be moved to ``compilable_argnames``.

    .. note::

        An operator can only specify ``static_argnames`` or ``compilable_argnames``, but not
        both; if **any** static arguments are not or cannot be lowered to the IR, then all
        static arguments are assumed to not be lowerable.
    """

    compilable_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that are treated as **compilable** static arguments.
    Compilable static arguments are a subset of static arguments—arguments whose
    concrete values are known when capturing the program. But, unlike ``static_argnames``,
    they are lowered to the compiler intermediate representation, making their values
    visible to the compiler, which may be useful if the compiler needs to interact with
    such values. Such values may include numbers, strings, and lists, tuples, or dictionaries
    thereof. This feature is opt-in; if any static arguments are not guaranteed to be
    compilable, it is safer to place them in ``static_argnames``.

    .. note::

        An operator can only specify ``static_argnames`` or ``compilable_argnames``, but not
        both; if **any** static arguments are not or cannot be lowered to the IR, then all
        static arguments are assumed to not be lowerable.
    """

    hybrid_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that represent dynamic data wrapped in static
    structures (known as Pytrees). Names in this category must be disjoint from
    ``dynamic_argnames``, ``static_argnames``, and ``compilable_argnames``, but may
    overlap with ``wire_argnames`` when wire arguments contain nested structures of
    wires. This feature is opt-in, but is required for cases where arrays,
    operators, and wires are supplied within a collection."""

    wire_sizes: ClassVar[tuple[int | None, ...]]
    """The expected number of wire labels for each wire argument. If any wire arguments
    support an arbitrary size, ``None`` must be used. By default, all wire arguments are
    assumed to support arbitrary sizes. For hybrid wire arguments, the wire size must
    always be ``None``."""

    # TODO: [sc-120517] Add proper fixed_sig support
    fixed_sig: ClassVar[tuple[type, ...]]
    """The expected signature of an operator. This must be set only if the shape and data
    type of all dynamic parameters is fixed, the number of wires is fixed, there are no
    static (compilable or non-compilable) arguments, and no hybrid arguments."""

    # ----------------- Class variables set automatically --------------------

    _sig: ClassVar[Signature]
    """The signature of the operator."""

    # ----------------- Instance variables set automatically -----------------

    _bound_args: BoundArguments
    """BoundArguments mapping arguments names to their values."""

    # ------------------------------------------------------------------------
    # ----------------------------- Initialization ---------------------------
    # ------------------------------------------------------------------------

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * ps1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    def __init__(self, *args, **kwargs):
        # Union[PauliSentence, None]: Representation of the operator as a
        # pauli sentence, if applicable
        self._pauli_rep: qp.pauli.PauliSentence | None = None

        self._bound_args = self._sig.bind(*args, **kwargs)
        self._bound_args.apply_defaults()

        self.__init_wires()

        # Broadcasting-related initialization
        self._batch_size: int | None = _UNSET_BATCH_SIZE
        self._ndim_params: tuple[int] = _UNSET_BATCH_SIZE

        self.queue()

    def __init_wires(self):
        """Initialize operator wires.

        * Union of all wire_argnames into _wires
        * Flatten pytree arguments and look for operators
        * Append operator argument wires to _wires
        """
        all_algorithmic_wires = []

        for wname, wsize in zip(self.wire_argnames, self.wire_sizes, strict=True):
            if wname not in self.hybrid_argnames:
                canonical_wires = Wires(self._bound_args.arguments[wname])
                self._bound_args.arguments[wname] = canonical_wires

                if wsize is not None and len(canonical_wires) != wsize:
                    raise ValueError(
                        f"Incorrect number of wires for '{self.name}.{wname}'. Expected {wsize} "
                        f"wires but got {len(canonical_wires)}."
                    )

                # Work wires are NOT included in the full wires list.
                if wname not in ("work_wires", "work_wire"):
                    all_algorithmic_wires.append(canonical_wires)

            # Pytree wires handling
            else:
                leaves, _ = flatten(self._bound_args.arguments[wname], is_leaf=_is_wires)
                if not all(isinstance(l, Wires) for l in leaves):
                    raise ValueError(
                        f"Hybrid wires argument '{wname}' is invalid. All leaf values must be "
                        "cast to 'qp.wires.Wires'."
                    )

                # Work wires are NOT included in the full wires list.
                if wname not in ("work_wires", "work_wire"):
                    all_algorithmic_wires.extend(leaves)

        for hname in self.hybrid_argnames:
            if hname in self.wire_argnames:
                continue
            leaves, _ = flatten(self._bound_args.arguments[hname], is_leaf=_is_op)
            ops = filter(_is_op, leaves)
            all_algorithmic_wires.extend(op.wires for op in ops)

        self._wires = Wires.all_wires(all_algorithmic_wires)

    def __init_subclass__(cls: type["Operator2"]) -> None:  # pylint: disable=too-many-branches
        # TODO: [sc-120429] Add processing for overriding has_decomposition

        cls._sig = signature(cls)
        _add_dynamic_properties(cls)
        register_pytree(cls, cls._flatten, cls._unflatten)

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

        if cls.static_argnames and cls.compilable_argnames:
            raise TypeError(
                "Operators can only contain 'static_argnames' or 'compilable_argnames', not both."
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
        sig_params = set(cls._sig.parameters.keys())
        if unclassified := sig_params - set(seen.keys()) - hybrid:
            raise TypeError(
                f"The following parameters of '{cls.__name__}' are not classified in "
                f"any argnames tuples: {unclassified}."
            )

        # Wire sizes setup
        if hasattr(cls, "wire_sizes"):
            if not isinstance(cls.wire_sizes, Sequence):
                cls.wire_sizes = (cls.wire_sizes,)

            if len(cls.wire_sizes) != len(cls.wire_argnames):
                raise TypeError("'wire_sizes' must have the same length as 'wire_argnames'.")

            for wn, ws in zip(cls.wire_argnames, cls.wire_sizes, strict=True):
                if wn in cls.hybrid_argnames and ws is not None:
                    raise TypeError(
                        f"Expected wire_size == None for '{wn}' as it is a hybrid wire argument."
                    )

                if not ((isinstance(ws, int) and ws > 0) or ws is None):
                    raise TypeError(
                        f"'{cls.__name__}.wire_sizes' is invalid. 'wire_sizes' must be a sequence "
                        f"of positive integers or 'None' values, but got {cls.wire_sizes}."
                    )

        else:
            cls.wire_sizes = tuple(None for _ in cls.wire_argnames)

    def _flatten(self) -> FlatPytree:
        """Serialize the operation into dynamic and static components.

        Returns:
            data, metadata: The dynamic and static components.

        See ``Operator2._unflatten``.

        The dynamic component can be recursive and include other operators.

        The metadata **must** be hashable. If the static data contains a non-hashable component, then this
        method and ``Operator2._unflatten`` should be overridden to provide a hashable version of the static data.

        **Example:**

        # TODO: [sc-120453] Remove doctest: +SKIP
        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> qp.Rot._unflatten(*op._flatten()) # doctest: +SKIP
        Rot(phi=1.2, theta=2.3, omega=3.4, wires=[0])
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> qp.PauliRot._unflatten(*op._flatten()) # doctest: +SKIP
        PauliRot(theta=1.2, pauli_word=XY, wires=[0, 1])
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
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> op._flatten() # doctest: +SKIP
        (([1.2], [Wires([0, 1])], []), ('XY',))
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

    # ------------------------------------------------------------------------
    # -------------------------- Public properties ---------------------------
    # ------------------------------------------------------------------------

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
    def pauli_rep(self) -> "qp.pauli.PauliSentence | None":
        """A :class:`~.PauliSentence` representation of the Operator, or ``None``
        if it doesn't have one."""
        return self._pauli_rep

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
            list[:class:`~.operation2.Operator2`]

        >>> class MyClass(qp.operation2.Operator2):
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
                return [qp.apply(self) for _ in range(z)]
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

        >>> class MyClass(qp.operation2.Operator2):
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
        """Returns a shallow copy of the current operator with its wires changed according
        to the given wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .Operator2: new operator
        """
        # pylint: disable=protected-access
        new_op = copy(self)

        for n, wires in self.wire_args.items():
            leaves, tree = flatten(wires, is_leaf=lambda w: isinstance(w, Wires))
            mapped_leaves = [Wires([wire_map.get(w, w) for w in leaf]) for leaf in leaves]
            new_wires = unflatten(mapped_leaves, tree)
            new_op._bound_args.arguments[n] = new_wires

        if (p_rep := self.pauli_rep) is not None:
            new_op._pauli_rep = p_rep.map_wires(wire_map)

        return new_op

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
        canonical_sparse_matrix = self.compute_sparse_matrix(**self.arguments, format="csr")
        return math.expand_matrix(
            canonical_sparse_matrix, wires=self.wires, wire_order=wire_order
        ).asformat(format)

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
        """Bool: Whether or not the Operator returns a defined decomposition."""
        # TODO: [sc-120519] Update when integrating with graph decompositions
        # if compute_decomposition or decomposition overwritten and property
        # not overwritten, set as class property during __init_subclass__
        # return any(rule.is_applicable(**self.resource_params) for rule in qp.list_decomps(self))
        return (
            cls.compute_decomposition != Operator2.compute_decomposition
            or cls.decomposition != Operator2.decomposition
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

        # TODO: [sc-120519] Update when integrating with graph decompositions
        # for decomp in qp.list_decomps(self):
        #     if decomp.is_applicable(**self.resource_params):
        #         with AnnotatedQueue() as q:
        #             decomp(**self.arguments)
        #         if QueuingManager.recording():
        #             # no need for copies if we just use queue method
        #             _ = [op.queue() for op in q.queue]
        #         return q.queue

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
                return qp.math.linalg.eigvals(self.matrix())
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
        inputs = []

        for key, value in self.arguments.items():
            # Non-wire arguments
            if key not in self.wire_argnames:
                res = value
            # Non-hybrid wire arguments
            elif key not in self.hybrid_argnames:
                res = value.tolist()
            # Hybrid wire arguments
            else:
                leaves, tree = flatten(value, is_leaf=_is_wires)
                leaves = [w.tolist() for w in leaves]
                res = unflatten(leaves, tree)

            inputs.append(f"{key}={res}")

        inputs = ", ".join(inputs)
        return f"{self.name}({inputs})"

    def __hash__(self) -> int:
        serialized_dynamic = tuple(
            _canonicalize_dynamic(self.arguments[d], self.name) for d in self.dynamic_argnames
        )
        serialized_wires = tuple(
            tuple(self.arguments[w]) for w in self.wire_argnames if w not in self.hybrid_argnames
        )
        serialized_static = tuple(str(self.arguments[s]) for s in self.static_argnames)
        serialized_compilable = tuple(str(self.arguments[c]) for c in self.compilable_argnames)

        serialized_hybrid = []
        for h in self.hybrid_argnames:
            leaves, tree = flatten(self.arguments[h], is_leaf=_is_hash_leaf)
            serialized_leaves = []
            for l in leaves:
                if isinstance(l, Wires):
                    serialized_leaves.append(tuple(l))
                elif isinstance(l, Operator2):
                    serialized_leaves.append(l)
                else:
                    serialized_leaves.append(_canonicalize_dynamic(l))

            entry = (tuple(serialized_leaves), tree)
            serialized_hybrid.append(entry)

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


# ------------------------------------------------------------------------------
# ------------------------------- Helper methods -------------------------------
# ------------------------------------------------------------------------------


def _add_dynamic_properties(cls: type[Operator2]) -> None:
    """Create dynamic properties for an operator using its signature."""
    # pylint: disable=protected-access
    for name in cls._sig.parameters:
        if not hasattr(cls, name):
            dyn_property = partial(_dynamic_property, name=name)
            setattr(cls, name, property(dyn_property))


def _dynamic_property(self: Operator2, name: str) -> Any:
    """Dynamic property for an argument called ``name``."""
    # pylint: disable=protected-access
    if "_bound_args" in vars(self) and name in self._bound_args.arguments:
        return self._bound_args.arguments[name]

    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'."
    )  # pragma: no cover


def _format_label_arg(x, decimals, cache):
    """Format a scalar parameter or retrieve/store a matrix-valued parameter
    from/to cache, formatting its position in the cache as parameter string."""
    if len(qp.math.shape(x)) == 0:
        # Scalar case
        if decimals is None:
            return ""
        try:
            return format(qp.math.toarray(x), f".{decimals}f")
        except ValueError:  # pragma: no cover
            # If the parameter can't be displayed as a float
            return format(x)

    if cache is None or not isinstance(mat_cache := cache.get("matrices", None), list):
        # No caching; matrices are not printed out fully, so no printing of this parameter
        return ""

    # Retrieve matrix location in cache, or write the matrix to cache as new entry
    for i, mat in enumerate(mat_cache):
        if qp.math.shape(x) == qp.math.shape(mat) and qp.math.allclose(x, mat):
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
        x = x if mod_val is None else qp.math.real(x) % mod_val
        return qp.math.round(x, 10)

    # Use qp.math.real to take the real part. We may get complex inputs for
    # example when differentiating holomorphic functions with JAX: a complex
    # valued QNode (one that returns qp.state) requires complex typed inputs.
    if op_name is not None and op_name in ("RX", "RY", "RZ", "PhaseShift", "Rot"):
        mod_val = 2 * np.pi
    else:
        mod_val = None

    return str(id(d) if math.is_abstract(d) else _mod_and_round(d, mod_val))


def _is_hash_leaf(l) -> bool:
    """Check whether a value is a pytree leaf for hashing. For the purpose of
    hashing, wires and operators are considered leaves."""
    return _is_op(l) or _is_wires(l)
