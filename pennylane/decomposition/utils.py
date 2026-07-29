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
from numbers import Number
from typing import Any

from pennylane.core import Operator2
from pennylane.core.operator import Operator, Operator1, abstractify
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

    _registry = defaultdict(tuple)

    def register(op: type[Operator2], **kwargs) -> None:
        r"""Register a possible signature for an operator.

        A *signature* records the abstract type of every argument of an operator (its
        dynamic parameters and wires), along with the values of any compilable static
        arguments. Registered signatures are collected in :func:`~.signature_registry`
        and are used to determine ahead of time which decomposition rules can be
        precompiled, improving the performance of decomposition passes in
        :func:`~.qjit`-compiled workflows.

        Operators with a fixed signature (i.e., ``op.has_fixed_sig`` is ``True``) are
        registered automatically when the class is defined. This function can be called
        directly to register additional signatures, for example the same operator with
        different fixed wire counts or static argument values.

        Args:
            op (type[~.Operator2]): the operator class to register a signature for.

        Keyword Args:
            **kwargs: the type or value of each argument, overriding the corresponding
                entry in ``op.arg_specs``. Together, ``op.arg_specs`` and these keyword
                arguments must specify every argument of ``op``. Dynamic arguments must be
                given an abstract numeric type (a subclass of ``numbers.Number`` or an
                :class:`~.AbstractArray`) and wire arguments an :class:`~.AbstractWires`,
                each with a fixed shape.

        Raises:
            ValueError: if ``op`` has hybrid or non-compilable static arguments, if the
                resulting signature does not cover every argument of ``op``, or if a
                dynamic or wire argument is not given a fixed-shape abstract type.

        .. seealso:: :func:`pennylane.decomposition.signature_registry`
        """
        spec = dict(op.arg_specs)
        spec.update(**kwargs)

        if any(op.hybrid_argnames or op.static_argnames):
            # Precompiling decomposition rules will require UID generation for operators
            # with hybrid/non-compilable static arguments. But, the UID is Python session
            # dependent, and precompilation happens in a different Python session.
            raise ValueError(
                "Signatures cannot be registered for operators that contain hybrid or "
                "non-compilable static arguments."
            )

        # pylint: disable=protected-access
        if set(spec.keys()) != set(op._sig.parameters.keys()):
            raise ValueError(
                "Signatures being registered must cover all operator arguments. Expected "
                f"{tuple(op._sig.parameters.keys())} but got {tuple(spec.keys())}."
            )

        for dname in op.dynamic_argnames:
            aval = spec[dname]
            if not (
                (isinstance(aval, type) and issubclass(aval, Number))
                or isinstance(aval, AbstractArray)
            ):
                raise ValueError(
                    f"Expected an abstract type for '{dname}' when registering a signature "
                    f"for {op.__name__}."
                )
            aval = abstractify(aval)
            if not aval.shape_fixed:
                raise ValueError(
                    "Signatures can only be registered if dynamic data has fixed shapes, "
                    f"but got shape {aval.shape} for '{op.__name__}.{dname}'."
                )

            spec[dname] = aval

        for wname in op.wire_argnames:
            aval = spec[wname]
            if not isinstance(aval, AbstractWires):
                raise ValueError(
                    f"Expected an abstract type for '{wname}' when registering a signature "
                    f"for {op.__name__}."
                )
            if not aval.shape_fixed:
                raise ValueError(
                    "Signatures can only be registered if all wire arguments have fixed shapes, "
                    f"but got shape {aval.shape} for '{op.__name__}.{wname}'."
                )

        _registry[op] += (spec,)

    def registry() -> dict[type[Operator2], tuple[dict[str, Any]], ...]:
        r"""Return the operator signatures registered with :func:`~.register_signature`.

        Returns:
            dict[type[~.Operator2], list[dict[str, Any]]]: a mapping from each registered
            operator class to the list of signatures registered for it. Each signature is
            a dictionary mapping argument names to their abstract type (for dynamic and
            wire arguments) or value (for compilable static arguments).

        .. seealso:: :func:`pennylane.decomposition.register_signature`
        """
        # Create a copy so mutation doesn't affect the registry
        return dict(_registry)

    return register, registry


register_signature, signature_registry = _init_signature_registration()
