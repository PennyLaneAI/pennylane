# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Utility functions provided for the QAOA submodule. These are useful for standard input checks,
for example to make sure that arguments have the right shape, range or type.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections.abc import Iterable

def check_iterable_graph(graph):
    """ Checks if a graph supplied in 'list format' is valid

        Args:
            graph (list): The graph that is being checked
    """

    for i in graph:

        if not isinstance(i, Iterable):
            raise ValueError("Elements of graph must be Iterable objects, got {}".format(type(i).__name__))
        if len(i) != 2:
            raise ValueError("Elements of graph must be Iterable objects of length 2, got length {}".format(len(i)))
        if i[0] == i[1]:
            raise ValueError("Edges must end in distinct nodes, got {}".format(i))

    if len(set([tuple(i) for i in graph])) != len(graph):
        raise ValueError("Nodes cannot be connected by more than one edge")

def get_nodes(graph):
    """Gets the nodes of an iterable graph

    Args:
            graph (list): The graph that is being checked
    Returns:
            List of nodes contained in the graph
    """

    node_set = {}
    for i in graph:
        node_set.update([i[0], i[1]])

    return list(node_set)
