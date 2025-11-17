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
    """Tests the default graph attributes are as expected."""

    dag_builder = PyDotDAGBuilder()

    assert isinstance(dag_builder.graph, pydot.Dot)
    # Ensure it's a directed graph
    assert dag_builder.graph.get_graph_type() == "digraph"
    # Ensure that it flows top to bottom
    assert dag_builder.graph.get_rankdir() == "TB"
    # Ensure edges can be connected directly to clusters / subgraphs
    assert dag_builder.graph.get_compound() == "true"
    # Ensure duplicated edges cannot be added
    assert dag_builder.graph.obj_dict["strict"] is True


class TestAddMethods:
    """Test that elements can be added to the graph."""

    @pytest.mark.unit
    def test_add_node(self):
        """Unit test the `add_node` method."""

        dag_builder = PyDotDAGBuilder()

        dag_builder.add_node("0", "node0")
        node_list = dag_builder.graph.get_node_list()
        assert len(node_list) == 1
        assert node_list[0].get_label() == "node0"

    @pytest.mark.unit
    def test_add_edge(self):
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
    def test_add_cluster(self):
        """Unit test the 'add_cluster' method."""

        dag_builder = PyDotDAGBuilder()
        dag_builder.add_cluster("0", "my_cluster")

        assert len(dag_builder.graph.get_subgraphs()) == 1
        assert dag_builder.graph.get_subgraphs()[0].get_name() == "cluster_0"

    @pytest.mark.unit
    def test_add_node_to_parent_graph(self):
        """Tests that you can add a node to a parent graph."""
        dag_builder = PyDotDAGBuilder()

        # Create node
        dag_builder.add_node("0", "node0")

        # Create cluster
        dag_builder.add_cluster("c0", "cluster0")

        # Create node inside cluster
        dag_builder.add_node("1", "node1", parent_graph_id="c0")

        # Verify graph structure
        root_graph = dag_builder.graph

        # Make sure the base graph has node0
        assert root_graph.get_node("0"), "Node 0 not found in root graph"

        # Get the cluster and verify it has node1 and not node0
        cluster_list = root_graph.get_subgraph("cluster_c0")
        assert cluster_list, "Subgraph 'cluster_c0' not found"
        cluster_graph = cluster_list[0]  # Get the actual subgraph object

        assert cluster_graph.get_node("1"), "Node 1 not found in cluster 'c0'"
        assert not cluster_graph.get_node("0"), "Node 0 was incorrectly added to cluster"

        assert not root_graph.get_node("1"), "Node 1 was incorrectly added to root"

    @pytest.mark.unit
    def test_add_cluster_to_parent_graph(self):
        """Test that you can add a cluster to a parent graph."""
        dag_builder = PyDotDAGBuilder()

        # Level 0 (Root): Adds cluster on top of base graph
        dag_builder.add_node("n_root", "node_root")
        dag_builder.add_cluster("c0", "cluster_outer")

        # Level 1 (Inside c0): Add node on outer cluster and create new cluster on top
        dag_builder.add_node("n_outer", "node_outer", parent_graph_id="c0")
        dag_builder.add_cluster("c1", "cluster_inner", parent_graph_id="c0")

        # Level 2 (Inside c1): Add node on second cluster
        dag_builder.add_node("n_inner", "node_inner", parent_graph_id="c1")

        root_graph = dag_builder.graph

        outer_cluster_list = root_graph.get_subgraph("cluster_c0")
        assert outer_cluster_list, "Outer cluster 'c0' not found in root"
        c0 = outer_cluster_list[0]

        inner_cluster_list = c0.get_subgraph("cluster_c1")
        assert inner_cluster_list, "Inner cluster 'c1' not found in 'c0'"
        c1 = inner_cluster_list[0]

        # Check Level 0 (Root)
        assert root_graph.get_node("n_root"), "n_root not found in root"
        assert root_graph.get_subgraph("cluster_c0"), "c0 not found in root"
        assert not root_graph.get_node("n_outer"), "n_outer incorrectly found in root"
        assert not root_graph.get_node("n_inner"), "n_inner incorrectly found in root"
        assert not root_graph.get_subgraph("cluster_c1"), "c1 incorrectly found in root"

        # Check Level 1 (c0)
        assert c0.get_node("n_outer"), "n_outer not found in c0"
        assert c0.get_subgraph("cluster_c1"), "c1 not found in c0"
        assert not c0.get_node("n_root"), "n_root incorrectly found in c0"
        assert not c0.get_node("n_inner"), "n_inner incorrectly found in c0"

        # Check Level 2 (c1)
        assert c1.get_node("n_inner"), "n_inner not found in c1"
        assert not c1.get_node("n_root"), "n_root incorrectly found in c1"
        assert not c1.get_node("n_outer"), "n_outer incorrectly found in c1"


class TestAttributes:
    """Tests that the attributes for elements in the graph are overridden correctly."""

    @pytest.mark.unit
    def test_default_graph_attrs(self):
        """Test that default graph attributes can be set."""

        dag_builder = PyDotDAGBuilder(attrs={"fontname": "Times"})

        dag_builder.add_node("0", "node0")
        node0 = dag_builder.graph.get_node("0")[0]
        assert node0.get("fontname") == "Times"

        dag_builder.add_cluster("1", "cluster0")
        cluster = dag_builder.graph.get_subgraphs()[0]
        assert cluster.get("fontname") == "Times"

    @pytest.mark.unit
    def test_add_node_with_attrs(self):
        """Tests that default attributes are applied and can be overridden."""
        dag_builder = PyDotDAGBuilder(node_attrs={"fillcolor": "lightblue", "penwidth": 3})

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
    def test_add_edge_with_attrs(self):
        """Tests that default attributes are applied and can be overridden."""
        dag_builder = PyDotDAGBuilder(edge_attrs={"color": "lightblue4", "penwidth": 3})

        dag_builder.add_node("0", "node0")
        dag_builder.add_node("1", "node1")
        dag_builder.add_edge("0", "1")
        edge = dag_builder.graph.get_edges()[0]
        # Defaults defined earlier
        assert edge.get("color") == "lightblue4"
        assert edge.get("penwidth") == 3

        # Make sure we can override
        dag_builder.add_edge("0", "1", color="red", penwidth=4)
        edge = dag_builder.graph.get_edges()[1]
        assert edge.get("color") == "red"
        assert edge.get("penwidth") == 4

    @pytest.mark.unit
    def test_add_cluster_with_attrs(self):
        """Tests that default cluster attributes are applied and can be overridden."""
        dag_builder = PyDotDAGBuilder(
            cluster_attrs={
                "style": "solid",
                "fillcolor": None,
                "penwidth": 2,
                "fontname": "Helvetica",
            }
        )

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


class TestOutput:
    """Test that the graph can be outputted correctly."""

    @pytest.mark.unit
    def test_render(self, monkeypatch):
        """Tests that the `render` method works correctly."""
        dag_builder = PyDotDAGBuilder()

        # mock out the graph writing functionality
        mock_write = MagicMock()
        monkeypatch.setattr(dag_builder.graph, "write", mock_write)
        dag_builder.render("my_graph.png")

        # make sure the function handles extensions correctly
        mock_write.assert_called_once_with("my_graph.png", format="png")

    @pytest.mark.unit
    def test_to_string(self):
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
