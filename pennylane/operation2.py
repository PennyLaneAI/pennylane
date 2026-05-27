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
from copy import deepcopy
from functools import partial
from inspect import BoundArguments, Signature, signature
from typing import Any, ClassVar, Literal

import pennylane as qp
from pennylane import math
from pennylane.operation import _UNSET_BATCH_SIZE, FlatPytree
from pennylane.pytrees import flatten, register_pytree
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


class Operator2(ABC):
    r"""Base class representing quantum operators.
    TODO: [sc-120453] Fill docstring
    """

    # pylint: disable=too-many-public-methods, too-many-instance-attributes

    # ------------ Class variables set manually --------------------

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
    whose concrete values are known when capturing the program. Arguments in this
    category are not lowered to a compiler intermediate representation (IR). Thus, if a
    static argument is not guaranteed to be lowerable, its name should be in ``static_argnames``.

    .. note::

        An operator can only specify ``static_argnames`` or ``compilable_argnames``, but not
        both; if **any** static arguments are not or cannot be lowered to the IR, then all
        static arguments are assumed to not be lowerable.
    """

    compilable_argnames: ClassVar[tuple[str, ...]] = ()
    """The names of arguments that are treated as **compilable** static arguments.
    Compilable static arguments are a subset of static arguments—these these arguments
    have concrete values that are known when capturing the program. But, unlike
    ``static_argnames``, they are lowered to the compiler intermediate representation.
    This feature is opt-in, but is useful for making static data visible to the compiler.

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

    # ------------ Class variables set automatically ---------------

    _sig: ClassVar[Signature]
    """The signature of the operator."""

    # ------------ Instance variables set automatically ------------

    _bound_args: BoundArguments
    """BoundArguments mapping arguments names to their values."""

    # ------------------ Initialization ------------------

    # this allows scalar multiplication from left with numpy arrays np.array(0.5) * ps1
    # taken from [stackexchange](https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634)
    __array_priority__ = 1000

    def __init__(self, *args, **kwargs):
        # Union[PauliSentence, None]: Representation of the operator as a
        # pauli sentence, if applicable
        self._pauli_rep: qp.pauli.PauliSentence | None = None

        self._bound_args = self._sig.bind(*args, **kwargs)
        self._bound_args.apply_defaults()

        # Initialize wires:
        #   1. Union of all wire_argnames into _wires
        #   2. Flatten pytree arguments and look for operators
        #   3. Append operator argument wires to _wires
        all_wires = []

        for wname, wsize in zip(self.wire_argnames, self.wire_sizes):
            if wname not in self.hybrid_argnames:
                canonical_wires = Wires(self._bound_args.arguments[wname])
                self._bound_args.arguments[wname] = canonical_wires

                if wsize is not None and len(canonical_wires) != wsize:
                    raise TypeError(
                        f"Incorrect number of wires for '{self.name}.{wname}'. Expected {wsize} "
                        f"wires but got {len(canonical_wires)}."
                    )

                # Work wires are NOT included in the full wires list.
                if wname not in ("work_wires", "work_wire"):
                    all_wires.append(canonical_wires)

            # Pytree wires handling
            else:
                leaves, _ = flatten(self._bound_args.arguments[wname], is_leaf=_is_wires)
                if not all(isinstance(l, Wires) for l in leaves):
                    raise TypeError(
                        f"Hybrid wires argument '{wname}' have not been cast to "
                        "'qp.wires.Wires' correctly."
                    )

                # Work wires are NOT included in the full wires list.
                if wname not in ("work_wires", "work_wire"):
                    all_wires.extend(leaves)

        for hname in self.hybrid_argnames:
            if hname in self.wire_argnames:
                continue
            leaves, _ = flatten(self._bound_args.arguments[hname], is_leaf=_is_op)
            ops = filter(_is_op, leaves)
            all_wires.extend(op.wires for op in ops)

        self._wires = Wires.all_wires(all_wires)

        # Broadcasting-related initialization
        self._batch_size: int | None = _UNSET_BATCH_SIZE
        self._ndim_params: tuple[int] = _UNSET_BATCH_SIZE

        self.queue()

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

        # Wire sizes setup
        if hasattr(cls, "wire_sizes"):
            if not isinstance(cls.wire_sizes, Sequence):
                cls.wire_sizes = (cls.wire_sizes,)

            if len(cls.wire_sizes) != len(cls.wire_argnames):
                raise TypeError("'wire_sizes' must have the same length as 'wire_argnames'.")

            for wn, ws in zip(cls.wire_argnames, cls.wire_sizes):
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

        # Every named signature parameter must appear in at least one *_argnames.
        sig_params = set(cls._sig.parameters.keys())
        if unclassified := sig_params - set(seen.keys()) - hybrid:
            raise TypeError(
                f"The following parameters of '{cls.__name__}' are not classified in "
                f"any argnames tuples: {unclassified}."
            )

    def _flatten(self) -> FlatPytree:
        """Serialize the operation into dynamic and static components.

        Returns:
            data, metadata: The dynamic and static components.

        See ``Operator2._unflatten``.

        The dynamic component can be recursive and include other operators.

        The metadata **must** be hashable. If the static data contains a non-hashable component, then this
        method and ``Operator2._unflatten`` should be overridden to provide a hashable version of the static data.

        **Example:**

        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> qp.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qp.PauliRot(1.2, "XY", wires=(0,1))
        >>> qp.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])
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

        >>> op = qp.Rot(1.2, 2.3, 3.4, wires=0)
        >>> op._flatten() # doctest: +SKIP
        (([1.2, 2.3, 3.4], [Wires([0])], []), ())
        >>> qp.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
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

        i = 0
        for name in cls.hybrid_argnames:
            if name in cls.wire_argnames:
                continue
            args[name] = data[2][i]
            i += 1

        # Process static data
        hashable_argnames = cls.static_argnames or cls.compilable_argnames
        for name, value in zip(hashable_argnames, metadata):
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

        try:
            ndims = tuple(qp.math.ndim(arg) for arg in dynamic_args)
        except (
            ValueError
        ) as e:  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            # When using tf.function with an input_signature that contains
            # an unknown-shaped input, ndim() will not be able to determine the number of
            # dimensions because they are not specified yet. Failing example: Let `fun` be
            # a single-parameter QNode.
            # `tf.function(fun, input_signature=(tf.TensorSpec(shape=None, dtype=tf.float32),))`
            # There might be a way to support batching nonetheless, which remains to be
            # investigated. For now, the batch_size is left to be `None` when instantiating
            # an operation with abstract parameters that make `qp.math.ndim` fail.
            if any(math.is_abstract(p) for p in dynamic_args):
                self._batch_size = None
                self._ndim_params = (0,) * len(dynamic_args)
                return
            raise e  # pragma: no cover

        if any(
            len(qp.math.shape(arg)) >= 1 and qp.math.shape(arg)[0] is None for arg in dynamic_args
        ):
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
                qp.math.shape(arg)[0]
                for (_, batched), arg in zip(ndims_matches, dynamic_args, strict=True)
                if batched
            ]
            if not qp.math.allclose(first_dims, first_dims[0]):
                raise ValueError(
                    "Broadcasting was attempted but the broadcasted dimensions "
                    f"do not match: {first_dims}."
                )

            self._batch_size = first_dims[0]

    # ------------------ Public properties ------------------

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
    def pauli_rep(self) -> qp.pauli.PauliSentence | None:
        """A :class:`~.PauliSentence` representation of the Operator, or ``None`` if it doesn't have one."""
        return self._pauli_rep

    def queue(self, context: QueuingManager = QueuingManager):
        """Append the operator to the Operator queue."""
        context.append(self)
        return self  # so pre-constructed Observable instances can be queued and returned in a single statement

    @property
    def _queue_category(self) -> Literal["_ops", "_measurements"]:
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Options are:
            * `"_ops"`
            * `"_measurements"`
        """
        return "_ops"

    # ------------------ General dunder methods ------------------

    def __repr__(self) -> str:
        """Constructor-call-like representation."""
        if self.dynamic_argnames:
            params = ", ".join([repr(self.arguments[d]) for d in self.dynamic_argnames])
            return f"{self.name}({params}, wires={self.wires.tolist()})"
        return f"{self.name}(wires={self.wires.tolist()})"

    # TODO: [sc-120431] Implement __hash__ and __eq__ after qp.equal supports `Operator2`
    # def __hash__(self) -> int:
    #     serialized_dyn = tuple(
    #         (n, str(id(d) if math.is_abstract(d) else d)) for n, d in self.dynamic_args.items()
    #     )
    #     serialized_wires = tuple((n, tuple(w)) for n, w in self.wire_args.items())
    #     serialized_static = tuple((n, str(s)) for n, s in self.static_args.items())
    #     return hash((self.name, serialized_dyn, serialized_wires, serialized_static))

    # def __eq__(self, other) -> bool:
    #     return qp.equal(self, other)

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


def _is_wires(val: Any) -> bool:
    """Check whether a value is a Wires object."""
    return isinstance(val, Wires)


def _is_op(val: Any) -> bool:
    """Check whether a value is an Operator2 object."""
    return isinstance(val, Operator2)
