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
from contextlib import contextmanager
from contextvars import ContextVar
from functools import singledispatch

from pennylane.operation import Operator

OP_NAME_ALIASES = {
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
    "I": "Identity",
    "H": "Hadamard",
    "measure": "MidMeasureMP",
    "MidMeasure": "MidMeasureMP",
    "MidCircuitMeasure": "MidMeasureMP",
    "ppm": "PauliMeasure",
    "pauli_measure": "PauliMeasure",
    "Elbow": "TemporaryAND",
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
    raise NotImplementedError(f"{type(op)} is not a valid type for to_name.")


@to_name.register
def _type_to_name(op: type):
    return translate_op_alias(op.__name__)


@to_name.register
def _operator_to_name(op: Operator):
    return translate_op_alias(op.name)


@to_name.register
def _str_to_name(op: str):
    return translate_op_alias(op)


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
