# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for symbolic resource operators.
"""

import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=protected-access,no-self-use


class TestResourceAdjoint:
    """Tests for ResourceAdjoint"""

    adjoint_ops = [
        re.ResourceAdjoint(re.ResourceQFT([0, 1])),
        re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1]))),
        re.ResourceAdjoint(re.ResourcePow(re.ResourceX(0), 5)),
    ]

    expected_params = [
        {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
        {
            "base_class": re.ResourceAdjoint,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
        },
        {
            "base_class": re.ResourcePow,
            "base_params": {"base_class": re.ResourceX, "base_params": {}, "z": 5},
        },
    ]

    @pytest.mark.parametrize("op, expected", zip(adjoint_ops, expected_params))
    def test_resource_params(self, op, expected):
        """Test that the resources are correct"""
        assert op.resource_params() == expected

    expected_names = [
        "Adjoint(QFT)",
        "Adjoint(Adjoint(QFT))",
        "Adjoint(Pow(X, 5))",
    ]

    @pytest.mark.parametrize("op, expected", zip(adjoint_ops, expected_names))
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        name = op.tracking_name_from_op()
        assert name == expected

    expected_resources = [
        re.Resources(gate_types={"Adjoint(QFT)": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Adjoint(Adjoint(QFT))": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Adjoint(Pow(X, 5))": 1}, num_gates=1, num_wires=1),
    ]

    @pytest.mark.parametrize("op, expected", zip(adjoint_ops, expected_resources))
    def test_tracking(self, op, expected):
        """Test that adjoints can be tracked."""
        tracking_name = op.tracking_name_from_op()
        gate_set = {tracking_name}

        assert re.get_resources(op, gate_set=gate_set) == expected


class TestResourceControlled:
    """Tests for ResourceControlled"""

    controlled_ops = [
        re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2]),
        re.ResourceControlled(
            re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2]), control_wires=[3]
        ),
        re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2, 3], control_values=[0, 1]),
        re.ResourceControlled(
            re.ResourceAdjoint(re.ResourceQFT([0, 1])),
            control_wires=[2, 3],
            control_values=[0, 1],
            work_wires=[4],
        ),
    ]

    expected_params = [
        {
            "base_class": re.ResourceQFT,
            "base_params": {"num_wires": 2},
            "num_ctrl_wires": 1,
            "num_ctrl_values": 0,
            "num_work_wires": 0,
        },
        {
            "base_class": re.ResourceControlled,
            "base_params": {
                "base_class": re.ResourceQFT,
                "base_params": {"num_wires": 2},
                "num_ctrl_wires": 1,
                "num_ctrl_values": 0,
                "num_work_wires": 0,
            },
            "num_ctrl_wires": 1,
            "num_ctrl_values": 0,
            "num_work_wires": 0,
        },
        {
            "base_class": re.ResourceQFT,
            "base_params": {"num_wires": 2},
            "num_ctrl_wires": 2,
            "num_ctrl_values": 1,
            "num_work_wires": 0,
        },
        {
            "base_class": re.ResourceAdjoint,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
            "num_ctrl_wires": 2,
            "num_ctrl_values": 1,
            "num_work_wires": 1,
        },
    ]

    @pytest.mark.parametrize("op, expected", zip(controlled_ops, expected_params))
    def test_resource_params(self, op, expected):
        """Test that the resources are correct"""
        assert op.resource_params() == expected

    expected_names = [
        "C(QFT,1,0,0)",
        "C(C(QFT,1,0,0),1,0,0)",
        "C(QFT,2,1,0)",
        "C(Adjoint(QFT),2,1,1)",
    ]

    @pytest.mark.parametrize("op, expected", zip(controlled_ops, expected_names))
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        name = op.tracking_name_from_op()
        assert name == expected

    expected_resources = [
        re.Resources(gate_types={"C(QFT,1,0,0)": 1}, num_gates=1, num_wires=3),
        re.Resources(gate_types={"C(C(QFT,1,0,0),1,0,0)": 1}, num_gates=1, num_wires=4),
        re.Resources(gate_types={"C(QFT,2,1,0)": 1}, num_gates=1, num_wires=4),
        re.Resources(
            gate_types={"C(Adjoint(QFT),2,1,1)": 1}, num_gates=1, num_wires=4
        ),  # PL does not count work wires for controlled operators
    ]

    @pytest.mark.parametrize("op, expected", zip(controlled_ops, expected_resources))
    def test_tracking(self, op, expected):
        """Test that adjoints can be tracked."""
        tracking_name = op.tracking_name_from_op()
        gate_set = {tracking_name}

        assert re.get_resources(op, gate_set=gate_set) == expected


class TestResourcePow:
    """Tests for ResourcePow"""

    pow_ops = [
        re.ResourcePow(re.ResourceQFT([0, 1]), 2),
        re.ResourcePow(re.ResourceAdjoint(re.ResourceQFT([0, 1])), 2),
        re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 3),
    ]

    expected_params = [
        {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}, "z": 2},
        {
            "base_class": re.ResourceAdjoint,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}},
            "z": 2,
        },
        {
            "base_class": re.ResourcePow,
            "base_params": {"base_class": re.ResourceQFT, "base_params": {"num_wires": 2}, "z": 2},
            "z": 3,
        },
    ]

    @pytest.mark.parametrize("op, expected", zip(pow_ops, expected_params))
    def test_resource_params(self, op, expected):
        """Test that the resources are correct"""
        assert op.resource_params() == expected

    expected_names = [
        "Pow(QFT, 2)",
        "Pow(Adjoint(QFT), 2)",
        "Pow(Pow(QFT, 2), 3)",
    ]

    @pytest.mark.parametrize("op, expected", zip(pow_ops, expected_names))
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        rep = op.resource_rep_from_op()
        name = rep.op_type.tracking_name(**rep.params)
        assert name == expected

    expected_resources = [
        re.Resources(gate_types={"Pow(QFT, 2)": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Pow(Adjoint(QFT), 2)": 1}, num_gates=1, num_wires=2),
        re.Resources(gate_types={"Pow(Pow(QFT, 2), 3)": 1}, num_gates=1, num_wires=2),
    ]

    @pytest.mark.parametrize("op, expected", zip(pow_ops, expected_resources))
    def test_tracking(self, op, expected):
        """Test that adjoints can be tracked."""
        tracking_name = op.tracking_name_from_op()
        gate_set = {tracking_name}

        assert re.get_resources(op, gate_set=gate_set) == expected

    @pytest.mark.parametrize(
        "nested_op, expected_op",
        [
            (
                re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2),
                re.ResourcePow(re.ResourceQFT([0, 1]), 4),
            ),
            (
                re.ResourcePow(re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2), 2),
                re.ResourcePow(re.ResourceQFT([0, 1]), 8),
            ),
            (
                re.ResourcePow(
                    re.ResourcePow(re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2), 2),
                    2,
                ),
                re.ResourcePow(re.ResourceQFT([0, 1]), 16),
            ),
        ],
    )
    def test_nested_pow(self, nested_op, expected_op):
        """Test the resources for nested Pow operators."""
        assert re.get_resources(nested_op) == re.get_resources(expected_op)
