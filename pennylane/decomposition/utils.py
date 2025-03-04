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


def toggle_graph_decomposition():
    """A closure that toggles the graph-based decomposition on and off."""

    _GRAPH_DECOMPOSITION = False

    def enable():
        nonlocal _GRAPH_DECOMPOSITION
        _GRAPH_DECOMPOSITION = True

    def disable() -> None:
        nonlocal _GRAPH_DECOMPOSITION
        _GRAPH_DECOMPOSITION = False

    def status() -> bool:
        nonlocal _GRAPH_DECOMPOSITION
        return _GRAPH_DECOMPOSITION

    return enable, disable, status


"""
The following functions are used to enable and disable the graph-based decomposition algorithm:
- enable_graph: Enables the graph-based decomposition algorithm.
- disable_graph: Disables the graph-based decomposition algorithm.
- enabled_graph: Returns the current status of the graph-based decomposition algorithm.
"""
enable_graph, disable_graph, enabled_graph = toggle_graph_decomposition()


# FIXME(remove)
def toggle_graph_decomposition_debug():
    """A closure that toggles the graph-based decomposition debug info on and off."""

    _GRAPH_DECOMPOSITION_DEBUG = False

    def enable():
        nonlocal _GRAPH_DECOMPOSITION_DEBUG
        _GRAPH_DECOMPOSITION_DEBUG = True

    def disable() -> None:
        nonlocal _GRAPH_DECOMPOSITION_DEBUG
        _GRAPH_DECOMPOSITION_DEBUG = False

    def status() -> bool:
        nonlocal _GRAPH_DECOMPOSITION_DEBUG
        return _GRAPH_DECOMPOSITION_DEBUG

    return enable, disable, status


enable_graph_debug, disable_graph_debug, enabled_graph_debug = toggle_graph_decomposition_debug()


class DecompositionError(Exception):
    """Base class for decomposition errors."""
