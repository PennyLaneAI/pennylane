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
"""Unit test module for the DAGBuilder abstract base class."""

from typing import Any

import pytest

from pennylane.compiler.python_compiler.visualization.dag_builder import DAGBuilder


class ConcreteDAGBuilder(DAGBuilder):
    """Concrete subclass of an ABC for testing purposes."""

    def add_node(
        self, node_id: str, node_label: str, parent_graph_id: str | None = None, **node_attrs: Any
    ) -> None:
        return

    def add_edge(self, from_node_id: str, to_node_id: str, **edge_attrs: Any) -> None:
        return

    def add_cluster(
        self,
        cluster_id: str,
        cluster_label: str,
        parent_graph_id: str | None = None,
        **cluster_attrs: Any,
    ) -> None:
        return

    def render(self, output_filename: str) -> None:
        return

    def to_string(self) -> str:
        return "test"


def test_concrete_implementation_works():
    """Unit test for concrete implementation of abc."""

    dag_builder = ConcreteDAGBuilder()
    # pylint: disable = assignment-from-none
    node = dag_builder.add_node("0", "node0")
    edge = dag_builder.add_edge("0", "1")
    cluster = dag_builder.add_cluster("0", "cluster0")
    render = dag_builder.render("test.png")
    string = dag_builder.to_string()

    assert node is None
    assert edge is None
    assert cluster is None
    assert render is None
    assert string == "test"


def test_abc_cannot_be_instantiated():
    """Tests that the DAGBuilder ABC cannot be instantiated."""

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # pylint: disable=abstract-class-instantiated
        DAGBuilder()


def test_incomplete_subclass():
    """Tests that an incomplete subclass will fail"""

    class IncompleteDAGBuilder(DAGBuilder):

        def add_node(self, *args, **kwargs):
            pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # pylint: disable=abstract-class-instantiated
        IncompleteDAGBuilder()
