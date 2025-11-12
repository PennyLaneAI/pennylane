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
"""Unit tests for the PyDotDAGBuilder subclass."""

import pydot
import pytest

from pennylane.compiler.python_compiler.visualization.pydot_dag_builder import PyDotDAGBuilder


@pytest.mark.unit
def test_initialization_defaults():
    """Tests the initiliazation of the PyDotDAGBuilder"""

    dag_builder = PyDotDAGBuilder()

    assert isinstance(dag_builder.graph, pydot.Graph)
    assert dag_builder.graph.get_graph_type() == "digraph"
    assert dag_builder.graph.get_rankdir() == "TB"
    assert dag_builder.graph.get_compound() == "true"


@pytest.mark.unit
def test_add_node():
    """Unit test the `add_node` method."""

    dag_builder = PyDotDAGBuilder()

    dag_builder.add_node("0", "node0")
    node_list = dag_builder.graph.get_node_list()
    assert len(node_list) == 1
    assert node_list[0].get_label() == "node0"


@pytest.mark.unit
def test_add_edge():
    """Unit test the `add_edge` method."""

    dag_builder = PyDotDAGBuilder()
    dag_builder.add_node("0", "node0")
    dag_builder.add_node("1", "node1")
    dag_builder.add_edge("0", "1")

    assert len(dag_builder.graph.get_edges()) == 1
    edge = dag_builder.graph.get_edges()[0]
    assert edge.get_source() == "0"
    assert edge.get_destination() == "1"


@pytest.mark.unit
def test_add_cluster():
    """Unit test the 'add_cluster' method."""

    dag_builder = PyDotDAGBuilder()
    dag_builder.add_cluster("0", "my_cluster")
    print(dag_builder.graph.get_subgraphs()[0])

    assert len(dag_builder.graph.get_subgraphs()) == 1
    assert dag_builder.graph.get_subgraphs()[0].get_name() == "cluster_0"
