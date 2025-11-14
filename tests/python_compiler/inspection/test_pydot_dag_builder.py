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

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.external

pydot = pytest.importorskip("pydot")
pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from pennylane.compiler.python_compiler.inspection.pydot_dag_builder import PyDotDAGBuilder


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

    assert len(dag_builder.graph.get_subgraphs()) == 1
    assert dag_builder.graph.get_subgraphs()[0].get_name() == "cluster_0"


@pytest.mark.unit
def test_add_node_with_attrs():
    """Tests that default attributes are applied and can be overridden."""
    dag_builder = PyDotDAGBuilder()

    # Defaults
    dag_builder.add_node("0", "node0")
    node0 = dag_builder.graph.get_node("0")[0]
    assert node0.get("fillcolor") == "lightblue"
    assert node0.get("penwidth") == 3

    # Make sure we can override
    dag_builder.add_node("1", "node1", fillcolor="red", penwidth=4)
    node1 = dag_builder.graph.get_node("1")[0]
    assert node1.get("fillcolor") == "red"
    assert node1.get("penwidth") == 4


@pytest.mark.unit
def test_add_edge_with_attrs():
    """Tests that default attributes are applied and can be overridden."""
    dag_builder = PyDotDAGBuilder()

    # Defaults
    dag_builder.add_node("0", "node0")
    dag_builder.add_node("1", "node1")
    dag_builder.add_edge("0", "1")
    edge = dag_builder.graph.get_edges()[0]
    assert edge.get("color") == "lightblue4"
    assert edge.get("penwidth") == 3

    # Make sure we can override
    dag_builder.add_edge("0", "1", color="red", penwidth=4)
    edge = dag_builder.graph.get_edges()[1]
    assert edge.get("color") == "red"
    assert edge.get("penwidth") == 4


@pytest.mark.unit
def test_add_cluster_with_attrs():
    """Tests that default cluster attributes are applied and can be overridden."""
    dag_builder = PyDotDAGBuilder()

    dag_builder.add_cluster("0", "cluster0")
    cluster1 = dag_builder.graph.get_subgraph("cluster_0")[0]

    # Defaults
    assert cluster1.get("style") == "solid"
    assert cluster1.get("fillcolor") is None
    assert cluster1.get("penwidth") == 2
    assert cluster1.get("fontname") == "Helvetica"

    dag_builder.add_cluster("1", "cluster1", style="filled", penwidth=10, fillcolor="red")
    cluster2 = dag_builder.graph.get_subgraph("cluster_1")[0]

    # Make sure we can override
    assert cluster2.get("style") == "filled"
    assert cluster2.get("penwidth") == 10
    assert cluster2.get("fillcolor") == "red"

    # Check that other defaults are still present
    assert cluster2.get("fontname") == "Helvetica"


@pytest.mark.unit
def test_render(monkeypatch, capsys):
    """Tests that the `render` method works correctly."""
    dag_builder = PyDotDAGBuilder()

    # mock out the graph writing functionality
    mock_write = MagicMock()
    monkeypatch.setattr(dag_builder.graph, "write", mock_write)
    dag_builder.render("my_graph.png")

    # make sure the function handles extensions correctly
    mock_write.assert_called_once_with("my_graph.png", format="png")

    # make sure we get information back to the user
    captured = capsys.readouterr()
    assert "Successfully rendered graph to: my_graph.png" in captured.out


@pytest.mark.unit
def test_to_string():
    """Tests that the `to_string` method works correclty."""

    dag_builder = PyDotDAGBuilder()
    dag_builder.add_node("n0", "node0")
    dag_builder.add_node("n1", "node1")
    dag_builder.add_edge("n0", "n1")

    string = dag_builder.to_string()
    assert isinstance(string, str)

    # make sure important things show up in the string
    assert "digraph" in string
    assert "n0" in string
    assert "n1" in string
    assert "n0 -> n1" in string
