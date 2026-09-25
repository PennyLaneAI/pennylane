# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This module implements utility functions for the decomposition module.
"""

import re
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import singledispatch
from types import MappingProxyType, NoneType
from typing import overload

from pennylane.core.operator import Operator, Operator1, Operator2, abstractify
from pennylane.pytrees import flatten
from pennylane.typing import AbstractArray, AbstractWires

OP_NAME_ALIASES = {
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
    "I": "Identity",
    "H": "Hadamard",
    "measure": "MidMeasureMP",
    "MidMeasure": "MidMeasureMP",
    "MidCircuitMeasure": "MidMeasureMP",
    "MidCircuitPauliMeasure": "PauliMeasure",
    "ppm": "PauliMeasure",
    "pauli_measure": "PauliMeasure",
    "Elbow": "TemporaryAND",
    "BasisStateProjector": "Projector",
    "StateVectorProjector": "Projector",
    "BasisEmbedding": "BasisState",
}


def translate_op_alias(op_alias):
    """Translates an operator alias to its proper name."""
    if op_alias in OP_NAME_ALIASES:
        return OP_NAME_ALIASES[op_alias]
    if match := re.match(r"(?:C|Controlled)\((\w+)\)", op_alias):
        base_op_name = match.group(1)
        return f"C({translate_op_alias(base_op_name)})"
    if match := re.match(r"Adjoint\((\w+)\)", op_alias):
        base_op_name = match.group(1)
        return f"Adjoint({translate_op_alias(base_op_name)})"
    if match := re.match(r"Pow\((\w+)\)", op_alias):
        base_op_name = match.group(1)
        return f"Pow({translate_op_alias(base_op_name)})"
    if match := re.match(r"Conditional\((\w+)\)", op_alias):
        base_op_name = match.group(1)
        return f"Conditional({translate_op_alias(base_op_name)})"
    if match := re.match(r"(\w+)\(\w+\)", op_alias):
        raise ValueError(
            f"'{match.group(1)}' is not a valid name for a symbolic operator. Supported "
            f'names include: "Adjoint", "C", "Controlled", "Pow".'
        )
    return op_alias


@singledispatch
def to_name(op) -> str:
    """Get the canocial name of an operation for the graph."""
    raise NotImplementedError(f"{type(op)} is not a valid type for to_name.")  # pragma: no cover


@to_name.register
def _type_to_name(op: type):
    return translate_op_alias(op.__name__)


@to_name.register
def _operator_to_name(op: Operator):
    return translate_op_alias(op.name)


@to_name.register
def _str_to_name(op: str):
    return translate_op_alias(op)


def _get_decomp_args(op: Operator):
    if isinstance(op, Operator1):
        return op.resource_params, op.data, {"wires": op.wires, **op.hyperparameters}
    return abstractify(op).arguments, (), op.arguments


def toggle_graph_decomposition():
    """A closure that toggles the experimental graph-based decomposition on and off."""

    _GRAPH_DECOMPOSITION = ContextVar("_GRAPH_DECOMPOSITION", default=False)

    def enable():
        """
        A global toggle for enabling the experimental graph-based decomposition system
        in PennyLane (introduced in v0.41). This new way of doing decompositions is
        generally more performant and allows for specifying custom decompositions.

        When this is enabled, :func:`~pennylane.transforms.decompose` will use the new decompositions system.
        """
        _GRAPH_DECOMPOSITION.set(True)

    def disable() -> None:
        """
        A global toggle for disabling the experimental graph-based decomposition
        system in PennyLane (introduced in v0.41). The experimental graph-based
        decomposition system is disabled by default in PennyLane.

        .. seealso:: :func:`~pennylane.decomposition.enable_graph`

        """
        _GRAPH_DECOMPOSITION.set(False)

    def status() -> bool:
        """
        A global toggle for checking the status of the experimental graph-based
        decomposition system in PennyLane (introduced in v0.41). The experimental
        graph-based decomposition system is disabled by default in PennyLane.

        .. seealso:: :func:`~pennylane.decomposition.enable_graph`

        """
        return _GRAPH_DECOMPOSITION.get()

    @contextmanager
    def toggle_ctx(new_state: bool):
        """A context manager in which graph is enabled or disabled temporarily."""

        token = _GRAPH_DECOMPOSITION.set(new_state)
        try:
            yield
        finally:
            _GRAPH_DECOMPOSITION.reset(token)

    return enable, disable, status, toggle_ctx


enable_graph, disable_graph, enabled_graph, toggle_graph_ctx = toggle_graph_decomposition()


def _init_signature_registration():
    # The signature registry is deliberately built in two stages: a *lazy* registry that records
    # pending registrations, and a *materialized* registry of the resulting abstract operators.
    #
    # This split exists because ``register`` is invoked from ``Operator2.__init_subclass__`` to
    # auto-register every fixed-signature operator as its class is defined, which happens *while
    # pennylane itself is still being imported*. At that point we cannot eagerly build the abstract
    # operator a signature ultimately needs, because both required steps assume a fully imported
    # pennylane:
    #
    #   * Constructing an instance (``op_cls(**specs)``) runs the operator's ``__init__``, which for
    #     many operators references other operators that do not exist yet
    #   * ``abstractify`` dispatches through ``functools.singledispatch``, whose MRO resolution walks
    #     the operator ABC hierarchy and triggers legacy ``__subclasshook__`` methods that read
    #     ``qp.ops.op_math.*`` - attributes that only exist once ``pennylane.ops`` has imported.
    #
    # So ``register`` performs only cheap, construction-free validation and stashes the raw
    # specs/instance in ``_lazy_registry``. The expensive "construct + abstractify" step is deferred
    # to the first access of ``signature_registry``, which only ever happens after import completes.

    # op class -> set of fully abstract operator instances (the public, materialized registry).
    _registry = defaultdict(set)
    # op class -> tuple of pending registrations, each a fully abstract instance or a specs dict.
    _lazy_registry = defaultdict(tuple)

    @overload
    def register(op: Operator2) -> None: ...
    @overload
    def register(op: type[Operator2], **kwargs) -> None: ...
    def register(op: Operator2 | type[Operator2], **kwargs) -> None:
        r"""Register a possible signature for an operator.

        A *signature* is a fully abstract instance of an operator, capturing the abstract type of
        every argument (its dynamic parameters and wires) along with the values of any static
        arguments. Registered signatures are collected in :func:`~.signature_registry`
        and are used to determine ahead of time which decomposition rules can be precompiled,
        improving the performance of decomposition passes in :func:`~.qjit`-compiled workflows.

        Args:
            op (~.Operator2 | type[~.Operator2]): the operator to register a signature for. If an
                operator *instance* is given, it must be fully abstract and is stored as-is. If an
                operator *type* is given, its ``arg_specs`` (optionally overridden by keyword
                arguments) are used to construct the abstract instance when the registry is
                materialized.

        Keyword Args:
            **kwargs: the type or value of each argument, overriding the corresponding entry in
                ``op.arg_specs``. Together with ``op.arg_specs`` these must cover *every* argument of
                the operator. These can only be provided when ``op`` is an operator *type*, not an
                instance.

        Raises:
            ValueError: if keyword arguments are provided together with an operator instance; if an
                operator instance is not fully abstract; if the provided specs do not cover every
                argument of the operator; or if a dynamic or wire argument is not fully abstract.

        .. seealso:: :func:`pennylane.decomposition.signature_registry`
        """
        if isinstance(op, Operator2):
            if kwargs:
                raise ValueError(
                    "Keyword arguments can only be provided when registering a signature for an "
                    "operator type, not an operator instance."
                )
            if not op.is_fully_abstract:
                raise ValueError(
                    "Signatures can only be registered for fully abstract operator instances. "
                    "All dynamic and wire arguments of the operator must be abstract."
                )

            _lazy_registry[type(op)] += (op,)
            return

        specs = dict(op.arg_specs or {})
        specs.update(**kwargs)

        # pylint: disable=protected-access
        if set(specs.keys()) != set(op._sig.parameters.keys()):
            raise ValueError(
                "Signatures being registered must cover all operator arguments. Expected "
                f"{tuple(op._sig.parameters.keys())} but got {tuple(specs.keys())}."
            )

        # Static/compilable arguments carry concrete values; only dynamic and wire arguments must
        # be abstract so that the operator can be constructed into an abstract instance later.
        # Iterate over `kwargs` instead of `specs` because we don't need to verify the correctness
        # of values in `op.arg_specs`
        for argname, argval in kwargs.items():
            if argname in op.static_argnames + op.compilable_argnames:
                continue

            leaves, _ = flatten(argval)
            if any(not isinstance(l, (AbstractArray, AbstractWires, NoneType)) for l in leaves):
                raise ValueError(
                    f"Cannot register a signature for {op.__name__!r}: the dynamic/wire argument "
                    f"'{argname}' must be fully abstract. Specify it using abstract types "
                    f"(e.g. Float or Wire[1]), not concrete values."
                )

        _lazy_registry[op] += (specs,)

    def registry() -> dict[type[Operator2], set[Operator2]]:
        r"""Return a read-only mapping of the registered operator signatures.

        Returns:
            MappingProxyType[type[~.Operator2], set[~.Operator2]]: a read-only mapping from each
            registered operator class to the set of its fully abstract signature instances.

        .. seealso:: :func:`pennylane.decomposition.register_signature`
        """
        # see the note in ``_init_signature_registration`` for why construction/abstractify
        # cannot happen at registration time
        for op_cls, sigs in _lazy_registry.items():
            for sig in sigs:
                op = sig if isinstance(sig, Operator2) else op_cls(**sig)
                _registry[op_cls].add(abstractify(op))

        _lazy_registry.clear()
        return MappingProxyType(_registry)

    return register, registry


register_signature, signature_registry = _init_signature_registration()
