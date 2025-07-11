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

OP_NAME_ALIASES = {
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
    "I": "Identity",
    "H": "Hadamard",
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
    if match := re.match(r"(\w+)\(\w+\)", op_alias):
        raise ValueError(
            f"'{match.group(1)}' is not a valid name for a symbolic operator. Supported "
            f'names include: "Adjoint", "C", "Controlled", "Pow".'
        )
    return op_alias


def toggle_graph_decomposition():
    """A closure that toggles the experimental graph-based decomposition on and off."""

    _GRAPH_DECOMPOSITION = False

    def enable():
        """
        A global toggle for enabling the experimental graph-based decomposition system
        in PennyLane (introduced in v0.41). This new way of doing decompositions is
        generally more performant and allows for specifying custom decompositions.

        When this is enabled, :func:`~pennylane.transforms.decompose` will use the new decompositions system.
        """

        nonlocal _GRAPH_DECOMPOSITION
        _GRAPH_DECOMPOSITION = True

    def disable() -> None:
        """
        A global toggle for disabling the experimental graph-based decomposition
        system in PennyLane (introduced in v0.41). The experimental graph-based
        decomposition system is disabled by default in PennyLane.

        .. seealso:: :func:`~pennylane.decomposition.enable_graph`
        """

        nonlocal _GRAPH_DECOMPOSITION
        _GRAPH_DECOMPOSITION = False

    def status() -> bool:
        """
        A global toggle for checking the status of the experimental graph-based
        decomposition system in PennyLane (introduced in v0.41). The experimental
        graph-based decomposition system is disabled by default in PennyLane.

        .. seealso:: :func:`~pennylane.decomposition.enable_graph`
        """

        nonlocal _GRAPH_DECOMPOSITION
        return _GRAPH_DECOMPOSITION

    return enable, disable, status


enable_graph, disable_graph, enabled_graph = toggle_graph_decomposition()
