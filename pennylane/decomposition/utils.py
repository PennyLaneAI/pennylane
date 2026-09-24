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
from types import MappingProxyType
from typing import overload

from pennylane.core.operator import Operator, Operator1, Operator2, abstractify

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

    _registry = defaultdict(set)
    _lazy_registry = defaultdict(tuple)

    def lazy_register(op: type[Operator2], **kwargs) -> None:
        """Lazily register the signature of an operator."""
        specs = dict(op.arg_specs or {})
        specs.update(**kwargs)
        _lazy_registry[op] += (specs,)

    @overload
    def register(op: Operator2) -> None: ...
    @overload
    def register(op: type[Operator2], **kwargs) -> None: ...
    def register(op: Operator2 | type[Operator2], **kwargs) -> None:
        r"""Register a possible signature for an operator.

        A *signature* is a fully abstract instance of an operator, capturing the abstract type of
        every argument (its dynamic parameters and wires) along with the values of any compilable
        static arguments. Registered signatures are collected in :func:`~.signature_registry` and
        are used to determine ahead of time which decomposition rules can be precompiled, improving
        the performance of decomposition passes in :func:`~.qjit`-compiled workflows.

        The signatures of all built-in operators with a fixed signature (i.e.,
        ``op.has_fixed_sig`` is ``True``) are registered by ``initialize_signature_registry``. This
        function can be called directly to register additional signatures, for example the same
        operator with different fixed wire counts or static argument values.

        Args:
            op (~.Operator2 | type[~.Operator2]): the operator to register a signature for. If an
                operator *instance* is given, it is abstractified before being stored. If an
                operator *type* is given, an instance is constructed from its ``arg_specs``
                (optionally overridden by keyword arguments) and then abstractified.

        Keyword Args:
            **kwargs: the type or value of each argument, overriding the corresponding entry in
                ``op.arg_specs``. These can only be provided when ``op`` is an operator *type*, not
                an instance.

        Raises:
            ValueError: if keyword arguments are provided together with an operator instance.

        .. seealso:: :func:`pennylane.decomposition.signature_registry`
        """
        if isinstance(op, Operator2):
            if kwargs:
                raise ValueError(
                    "Keyword arguments can only be provided when registering a signature for an "
                    "operator type, not an operator instance."
                )
            op_cls = type(op)
            inst = op

        else:
            specs = dict(op.arg_specs or {})
            specs.update(**kwargs)
            inst = op(**specs)
            op_cls = op

        _registry[op_cls].add(abstractify(inst))

    def registry() -> dict[type[Operator2], set[Operator2]]:
        r"""Return the operator signatures registered with :func:`~.register_signature`.

        Returns:
            MappingProxyType[type[~.Operator2], set[~.Operator2]]: a mapping from each registered
            operator class to the set of registered signatures, where each signature is a fully
            abstract instance of the operator.

        .. seealso:: :func:`pennylane.decomposition.register_signature`
        """
        return MappingProxyType(_registry)

    def initialize_registry():
        """Initialize the registry by migrating signatures from the lazy registry."""
        for op_cls, sigs in _lazy_registry.items():
            for sig in sigs:
                _registry[op_cls].add(abstractify(op_cls(**sig)))
        _lazy_registry.clear()

    return register, lazy_register, registry, initialize_registry


register_signature, lazy_register_signature, signature_registry, initialize_signature_registry = (
    _init_signature_registration()
)
