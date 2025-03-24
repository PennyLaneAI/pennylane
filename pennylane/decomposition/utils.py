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


class DecompositionError(Exception):
    """Base class for decomposition errors."""


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

        See also: :func:`~pennylane.decomposition.enable_graph`
        """

        nonlocal _GRAPH_DECOMPOSITION
        _GRAPH_DECOMPOSITION = False

    def status() -> bool:
        """
        A global toggle for checking the status of the experimental graph-based
        decomposition system in PennyLane (introduced in v0.41). The experimental
        graph-based decomposition system is disabled by default in PennyLane.

        See also: :func:`~pennylane.decomposition.enable_graph`
        """

        nonlocal _GRAPH_DECOMPOSITION
        return _GRAPH_DECOMPOSITION

    return enable, disable, status


enable_graph, disable_graph, enabled_graph = toggle_graph_decomposition()
