# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test the profiler functionality.
"""

from collections import defaultdict

import pytest

import pennylane as qp
import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resources_base import Resources
from pennylane.labs.profiler import ProfileNode, export_flame_graph_data, profile

# pylint: disable=no-self-use, too-few-public-methods


def _basic_circuit():
    """A simple quantum function which queues operators for profiling."""
    for _ in range(5):
        qre.Hadamard()
        qre.CRZ(1e-9)

    qre.CRZ(1e-3)
    qre.CNOT()


def _manually_generate_profile():
    h_node = ProfileNode(
        cmpr_op=qre.Hadamard.resource_rep(),
        scalar=1,
        gate_data={qre.Hadamard.resource_rep(): 1},
        children=None,
    )
    t_node_1 = ProfileNode(
        cmpr_op=qre.T.resource_rep(), scalar=44, gate_data={qre.T.resource_rep(): 88}, children=None
    )
    t_node_2 = ProfileNode(
        cmpr_op=qre.T.resource_rep(), scalar=21, gate_data={qre.T.resource_rep(): 42}, children=None
    )
    cnot_node_1 = ProfileNode(
        cmpr_op=qre.CNOT.resource_rep(),
        scalar=1,
        gate_data={qre.CNOT.resource_rep(): 1},
        children=None,
    )
    cnot_node_2 = ProfileNode(
        cmpr_op=qre.CNOT.resource_rep(),
        scalar=2,
        gate_data={qre.CNOT.resource_rep(): 2},
        children=None,
    )

    rz_node_1 = ProfileNode(
        cmpr_op=qre.RZ.resource_rep(1e-9),
        scalar=2,
        gate_data={qre.T.resource_rep(): 88},
        children=[t_node_1],
    )
    rz_node_2 = ProfileNode(
        cmpr_op=qre.RZ.resource_rep(1e-3),
        scalar=2,
        gate_data={qre.T.resource_rep(): 42},
        children=[t_node_2],
    )

    crz_node_1 = ProfileNode(
        cmpr_op=qre.CRZ.resource_rep(1e-9),
        scalar=1,
        gate_data={
            qre.T.resource_rep(): 88,
            qre.CNOT.resource_rep(): 2,
        },
        children=[cnot_node_2, rz_node_1],
    )
    crz_node_2 = ProfileNode(
        cmpr_op=qre.CRZ.resource_rep(1e-3),
        scalar=1,
        gate_data={
            qre.T.resource_rep(): 42,
            qre.CNOT.resource_rep(): 2,
        },
        children=[cnot_node_2, rz_node_2],
    )

    root_node = ProfileNode(
        gate_data={
            qre.T.resource_rep(): 482,
            qre.CNOT.resource_rep(): 13,
            qre.Hadamard.resource_rep(): 5,
        },
        children=[
            h_node,
            crz_node_1,
            h_node,
            crz_node_1,
            h_node,
            crz_node_1,
            h_node,
            crz_node_1,
            h_node,
            crz_node_1,
            crz_node_2,
            cnot_node_1,
        ],
    )
    return root_node


def _manually_generated_resources():
    gate_data = {
        qre.T.resource_rep(): 482,
        qre.CNOT.resource_rep(): 13,
        qre.Hadamard.resource_rep(): 5,
    }
    return Resources(0, 0, 2, gate_data)


def equal_resource_profile(root: ProfileNode, other_root: ProfileNode):
    """Check equality of resource profiles"""



    equality_checks = (
        root.cmpr_op == other_root.cmpr_op,
        root.scalar == other_root.scalar,
        root.gate_data == other_root.gate_data,
        len(root.children) == len(other_root.children),
    )
    if all(equality_checks) and all(
        equal_resource_profile(c1, c2) for c1, c2 in zip(root.children, other_root.children)
    ):
        return True
    return False


class TestProfileNode:
    """Tests for the ``ProfileNode`` class."""

    def test_default_initialization(self):
        """Test that a ProfileNode is created with the expected default attributes."""
        node = ProfileNode()

        assert node.cmpr_op is None
        assert node.scalar == 1
        assert node.children == []
        assert isinstance(node.gate_data, defaultdict)
        assert len(node.gate_data) == 0
        # gate_data defaults to a defaultdict(int)
        assert node.gate_data["missing"] == 0

    def test_custom_initialization(self):
        """Test that the supplied arguments are stored on the node."""
        cmpr_op = qre.Hadamard().resource_rep_from_op()
        gate_data = defaultdict(int, {cmpr_op: 4})
        child = ProfileNode()

        node = ProfileNode(
            cmpr_op=cmpr_op,
            scalar=4,
            gate_data=gate_data,
            children=[child],
        )

        assert node.cmpr_op is cmpr_op
        assert node.scalar == 4
        assert node.gate_data is gate_data
        assert node.children == [child]

    def test_group_by_name_empty(self):
        """Test that grouping an empty or None list of nodes returns None."""
        assert ProfileNode.group_by_name([]) is None
        assert ProfileNode.group_by_name(None) is None

    def test_group_by_type_empty(self):
        """Test that grouping an empty or None list of nodes returns None."""
        assert ProfileNode.group_by_type([]) is None
        assert ProfileNode.group_by_type(None) is None

    def test_group_by_name_merges_duplicates(self):
        """Test that nodes sharing a name are merged, regardless of resource params."""
        gate_set = {"T", "Hadamard", "CNOT"}

        def circuit():
            for _ in range(5):
                qre.Hadamard()
                qre.CRZ(1e-9)
            qre.CRZ(1e-3)
            qre.Hadamard()
            qre.QFT(4)

        res_profile, _ = profile(circuit, gate_set)()
        grouped = ProfileNode.group_by_name(res_profile.children)

        grouped_child_nodes = {name: data[0] for name, data in grouped.items()}
        assert grouped_child_nodes == {"Hadamard": 6, "CRZ": 6, "QFT(4)": 1}

    def test_group_by_type_merges_only_identical(self):
        """Test that grouping by type keeps operators with distinct params separate."""
        gate_set = {"T", "Hadamard", "CNOT"}

        def circuit():
            for _ in range(5):
                qre.Hadamard()
                qre.CRZ(1e-9)
            qre.CRZ(1e-3)
            qre.Hadamard()
            qre.QFT(4)

        res_profile, _ = profile(circuit, gate_set)()
        grouped = ProfileNode.group_by_type(res_profile.children)

        names_and_scalars = sorted((op.name, data[0]) for op, data in grouped.items())
        # The two CRZ operators differ in precision so they remain separate entries.
        assert names_and_scalars == [
            ("CRZ", 1),
            ("CRZ", 5),
            ("Hadamard", 6),
            ("QFT(4)", 1),
        ]

    def test_default_cost_func(self):
        """Test that the default cost function counts the number of T gates."""
        gate_data = {
            qre.X.resource_rep(): 1,
            qre.T.resource_rep(): 5,
            qre.Hadamard.resource_rep(): 2,
        }
        assert ProfileNode.default_cost_func(gate_data) == 5

    def test_default_cost_func_no_t_gates(self):
        """Test that the default cost function returns 0 when there are no T gates."""
        gate_data = {qre.X.resource_rep(): 1, qre.Hadamard.resource_rep(): 2}
        assert ProfileNode.default_cost_func(gate_data) == 0

    def test_set_cost_func_with_list(self):
        """Test that a cost function built from a list sums the named gate counts."""
        gate_data = {
            qre.X.resource_rep(): 1,
            qre.T.resource_rep(): 5,
            qre.Hadamard.resource_rep(): 2,
        }
        cost_func = ProfileNode.set_cost_func(["T", "Hadamard"])
        assert cost_func(gate_data) == 7

    def test_set_cost_func_with_str(self):
        """Test that a single gate name produces a cost function over that gate."""
        gate_data = {
            qre.X.resource_rep(): 1,
            qre.T.resource_rep(): 5,
            qre.Hadamard.resource_rep(): 2,
        }
        cost_func = ProfileNode.set_cost_func("Hadamard")
        assert cost_func(gate_data) == 2


class TestProfile:
    """Tests for the ``profile`` function and its dispatch across input types."""

    expected_profile = _manually_generate_profile()
    expected_resources = _manually_generated_resources()

    def test_profile_qfunc(self):
        """Test profiling a quantum function returns a ProfileNode and Resources."""
        gate_set = {"T", "Hadamard", "CNOT"}
        res_profile, resources = profile(_basic_circuit, gate_set)()

        assert isinstance(res_profile, ProfileNode)
        assert equal_resource_profile(res_profile, self.expected_profile)
        assert isinstance(resources, Resources)
        assert resources == self.expected_resources

    def test_profile_returns_callable_for_qfunc(self):
        """Test that profiling a quantum function returns a callable wrapper."""
        wrapped = profile(_basic_circuit, {"T", "Hadamard", "CNOT"})
        assert callable(wrapped)

    def test_profile_qnode(self):
        """Test profiling a QNode profiles the underlying quantum function."""
        dev = qp.device("default.qubit", wires=4)

        @qp.qnode(dev)
        def circuit():
            _basic_circuit()
            return qp.state()

        res_profile, resources = profile(circuit, {"T", "Hadamard", "CNOT"})()
        assert isinstance(res_profile, ProfileNode)
        assert equal_resource_profile(res_profile, self.expected_profile)
        assert isinstance(resources, Resources)
        assert resources == self.expected_resources

    def test_profile_resource_operator(self):
        """Test profiling a single ResourceOperator returns a tuple directly."""
        res_profile, resources = profile(qre.CRZ(1e-9), {"T", "Hadamard", "CNOT"})

        # Expected Profile and Resources
        cnot_node = ProfileNode(
            cmpr_op=qre.CNOT.resource_rep(),
            scalar=2,
            gate_data={qre.CNOT.resource_rep(): 2},
            children=None,
        )
        t_node = ProfileNode(
            cmpr_op=qre.T.resource_rep(),
            scalar=44,
            gate_data={qre.T.resource_rep(): 88},
            children=None,
        )
        rz_node = ProfileNode(
            cmpr_op=qre.RZ.resource_rep(1e-9),
            scalar=2,
            gate_data={qre.T.resource_rep(): 88},
            children=[t_node],
        )
        crz_node = ProfileNode(
            cmpr_op=qre.CRZ.resource_rep(1e-9),
            scalar=1,
            gate_data={
                qre.T.resource_rep(): 88,
                qre.CNOT.resource_rep(): 2,
            },
            children=[cnot_node, rz_node],
        )
        root_node = ProfileNode(
            cmpr_op=None,
            scalar=1,
            gate_data={
                qre.T.resource_rep(): 88,
                qre.CNOT.resource_rep(): 2,
            },
            children=[crz_node],
        )

        expected_resources = Resources(
            0,
            0,
            2,
            {qre.T.resource_rep(): 88, qre.CNOT.resource_rep(): 2},
        )

        assert isinstance(res_profile, ProfileNode)
        assert equal_resource_profile(root_node, res_profile)
        assert isinstance(resources, Resources)
        assert resources == expected_resources

    def test_profile_resources(self):
        """Test profiling a precomputed Resources object."""
        _, base_resources = profile(_basic_circuit, {"T", "Hadamard", "CNOT"})()
        res_profile, resources = profile(base_resources, {"T", "Hadamard", "CNOT"})

        ## Expected Profile:
        h_node = ProfileNode(
            cmpr_op=qre.Hadamard.resource_rep(),
            scalar=5,
            gate_data={qre.Hadamard.resource_rep(): 5},
            children=None,
        )
        t_node = ProfileNode(
            cmpr_op=qre.T.resource_rep(),
            scalar=482,
            gate_data={qre.T.resource_rep(): 482},
            children=None,
        )
        cnot_node = ProfileNode(
            cmpr_op=qre.CNOT.resource_rep(),
            scalar=13,
            gate_data={qre.CNOT.resource_rep(): 13},
            children=None,
        )
        root_node = ProfileNode(
            gate_data={
                qre.T.resource_rep(): 482,
                qre.CNOT.resource_rep(): 13,
                qre.Hadamard.resource_rep(): 5,
            },
            children=[h_node, cnot_node, t_node],
        )

        assert isinstance(res_profile, ProfileNode)
        assert equal_resource_profile(res_profile, root_node)
        assert isinstance(resources, Resources)
        assert resources == base_resources

    def test_profile_pl_operation(self):
        """Test profiling a PennyLane operation maps it to a resource operator."""
        res_profile, resources = profile(qp.CRZ(1.23, wires=[0, 1]), {"T", "Hadamard", "CNOT"})

        # Expected Profile and Resources
        cnot_node = ProfileNode(
            cmpr_op=qre.CNOT.resource_rep(),
            scalar=2,
            gate_data={qre.CNOT.resource_rep(): 2},
            children=None,
        )
        t_node = ProfileNode(
            cmpr_op=qre.T.resource_rep(),
            scalar=44,
            gate_data={qre.T.resource_rep(): 88},
            children=None,
        )
        rz_node = ProfileNode(
            cmpr_op=qre.RZ.resource_rep(None),
            scalar=2,
            gate_data={qre.T.resource_rep(): 88},
            children=[t_node],
        )
        crz_node = ProfileNode(
            cmpr_op=qre.CRZ.resource_rep(None),
            scalar=1,
            gate_data={
                qre.T.resource_rep(): 88,
                qre.CNOT.resource_rep(): 2,
            },
            children=[cnot_node, rz_node],
        )
        root_node = ProfileNode(
            cmpr_op=None,
            scalar=1,
            gate_data={
                qre.T.resource_rep(): 88,
                qre.CNOT.resource_rep(): 2,
            },
            children=[crz_node],
        )

        expected_resources = Resources(
            0,
            0,
            2,
            {qre.T.resource_rep(): 88, qre.CNOT.resource_rep(): 2},
        )

        assert isinstance(res_profile, ProfileNode)
        assert equal_resource_profile(root_node, res_profile)
        assert isinstance(resources, Resources)
        assert resources == expected_resources

    def test_profile_invalid_type_raises(self):
        """Test that an unsupported workflow type raises a TypeError."""
        with pytest.raises(TypeError, match="Could not obtain resources for workflow"):
            profile(42)


class TestExportFlameGraphData:
    """Tests for the ``export_flame_graph_data`` function."""

    def test_export_by_name(self):
        """Test the column data produced when grouping by operator name."""
        gate_set = {"T", "Hadamard", "CNOT"}
        res_profile, _ = profile(_basic_circuit, gate_set)()

        ids, names, values, parents = export_flame_graph_data(res_profile)

        # All four parallel lists must have the same length.
        assert len(ids) == len(names) == len(values) == len(parents)

        # The root block is labelled "circuit" and has no parent.
        assert ids[0] == "circuit"
        assert names[0] == "circuit"
        assert parents[0] == ""

        # The first few entries match the documented output.
        assert names[:5] == [
            "circuit",
            "Hadamard [x5]",
            "CRZ [x6]",
            "CNOT [x12]",
            "RZ [x12]",
        ]

    def test_export_root_value_is_total_cost(self):
        """Test that the root value equals the total cost (T count by default)."""
        gate_set = {"T", "Hadamard", "CNOT"}
        res_profile, resources = profile(_basic_circuit, gate_set)()

        _, _, values, _ = export_flame_graph_data(res_profile)

        total_t = {op.name: count for op, count in resources.gate_types.items()}["T"]
        assert values[0] == total_t

    def test_export_parents_reference_valid_ids(self):
        """Test that every non-root parent refers to an id that exists in the data."""
        gate_set = {"T", "Hadamard", "CNOT"}
        res_profile, _ = profile(_basic_circuit, gate_set)()

        ids, _, _, parents = export_flame_graph_data(res_profile)
        id_set = set(ids)
        for parent in parents[1:]:
            assert parent in id_set

    def test_export_by_type(self):
        """Test the column data produced when grouping by operator type."""
        gate_set = {"T", "Hadamard", "CNOT"}
        res_profile, _ = profile(_basic_circuit, gate_set)()

        ids, names, values, parents = export_flame_graph_data(res_profile, group_by="type")

        # All four parallel lists must have the same length.
        assert len(ids) == len(names) == len(values) == len(parents)

        # The root block is labelled "circuit" and has no parent.
        assert ids[0] == "circuit"
        assert names[0] == "circuit"
        assert parents[0] == ""

        # The first few entries match the documented output.
        assert names[:10] == [
            "circuit",
            "Hadamard [x5]",
            "CRZ [x5]",
            "CNOT [x10]",
            "RZ [x10]",
            "T [x440]",
            "CRZ",
            "CNOT [x2]",
            "RZ [x2]",
            "T [x42]",
        ]

    def test_export_invalid_group_by_raises(self):
        """Test that an unknown group_by value raises a ValueError."""
        res_profile, _ = profile(qre.QFT(4), {"T", "Hadamard", "CNOT"})
        with pytest.raises(ValueError, match="Unknown group_by value"):
            export_flame_graph_data(res_profile, group_by="bogus")

    def test_export_custom_cost_func(self):
        """Test that a custom cost function is used to compute the block values."""
        gate_set = {"T", "Hadamard", "CNOT"}
        res_profile, resources = profile(_basic_circuit, gate_set)()

        cost_func = ProfileNode.set_cost_func("Hadamard")
        _, _, values, _ = export_flame_graph_data(res_profile, cost_func=cost_func)

        total_hadamard = {op.name: count for op, count in resources.gate_types.items()}["Hadamard"]
        assert values[0] == total_hadamard

    def test_export_single_operator(self):
        """Test exporting flame graph data for a single resource operator profile."""
        res_profile, _ = profile(qre.QFT(4), {"T", "Hadamard", "CNOT"})
        ids, names, _, parents = export_flame_graph_data(res_profile)

        assert ids[0] == "circuit"
        assert names[0] == "circuit"
        assert parents[0] == ""
        assert "QFT(4)" in names
