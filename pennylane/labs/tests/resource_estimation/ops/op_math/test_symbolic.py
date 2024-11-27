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
from pennylane.labs.resource_estimation.resource_container import _scale_dict

# pylint: disable=protected-access,no-self-use


class TestResourceAdjoint:
    """Tests for ResourceAdjoint"""

    def test_resource_params(self):
        """Test that the resources are correct"""

        base = re.ResourceQFT(wires=[0, 1, 2])
        op = re.ResourceAdjoint(base=base)
        assert op.resource_params() == {
            "base_class": re.ResourceQFT,
            "base_params": base.resource_params(),
        }

    @pytest.mark.parametrize(
        "op, expected",
        [
            (re.ResourceAdjoint(re.ResourceQFT([0, 1])), "Adjoint(QFT(2))"),
            (
                re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1]))),
                "Adjoint(Adjoint(QFT(2)))",
            ),
        ],
    )
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        rep = op.resource_rep_from_op()
        name = rep.op_type.tracking_name(**rep.params)
        assert name == expected

    @pytest.mark.parametrize(
        "nested_op, base_op",
        [
            (
                re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2]))),
                re.ResourceQFT([0, 1, 2]),
            ),
            (
                re.ResourceAdjoint(
                    re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])))
                ),
                re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])),
            ),
            (
                re.ResourceAdjoint(
                    re.ResourceAdjoint(
                        re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])))
                    )
                ),
                re.ResourceQFT([0, 1, 2]),
            ),
            (
                re.ResourceAdjoint(
                    re.ResourceAdjoint(
                        re.ResourceAdjoint(
                            re.ResourceAdjoint(re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])))
                        )
                    )
                ),
                re.ResourceAdjoint(re.ResourceQFT([0, 1, 2])),
            ),
        ],
    )
    def test_nested_adjoints(self, nested_op, base_op):
        """Test the resources of nested Adjoints."""

        nested_rep = nested_op.resource_rep_from_op()
        nested_params = nested_rep.params
        nested_type = nested_rep.op_type
        nested_resources = nested_type.resources(**nested_params)

        base_op = base_op.resource_rep_from_op()
        base_params = base_op.params
        base_type = base_op.op_type
        base_resources = base_type.resources(**base_params)

        assert nested_resources == base_resources


class TestResourceControlled:
    """Tests for ResourceControlled"""

    def test_resource_params(self):
        """Test that the resources are correct"""

        base = re.ResourceQFT(wires=[0, 1, 2])
        op = re.ResourceControlled(base=base, control_wires=[3])
        assert op.resource_params() == {
            "base_class": re.ResourceQFT,
            "base_params": base.resource_params(),
            "num_ctrl_wires": 1,
            "num_ctrl_values": 1,
            "num_work_wires": 0,
        }

    @pytest.mark.parametrize(
        "op, expected",
        [
            (re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2]), "C(QFT(2),1,1,0)"),
            (
                re.ResourceControlled(
                    re.ResourceControlled(re.ResourceQFT([0, 1]), control_wires=[2]),
                    control_wires=[3],
                ),
                "C(C(QFT(2),1,1,0),1,1,0)",
            ),
            (
                re.ResourceControlled(
                    re.ResourceQFT([0, 1]), control_wires=[2, 3], control_values=[0, 1]
                ),
                "C(QFT(2),2,1,0)",
            ),
            (
                re.ResourceControlled(
                    re.ResourceQFT([0, 1]),
                    control_wires=[2, 3],
                    control_values=[0, 1],
                    work_wires=[4],
                ),
                "C(QFT(2),2,1,1)",
            ),
        ],
    )
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        rep = op.resource_rep_from_op()
        name = rep.op_type.tracking_name(**rep.params)
        assert name == expected

    @pytest.mark.parametrize(
        "nested_op, expected_op",
        [
            (
                re.ResourceControlled(
                    re.ResourceControlled(re.ResourceX(0), control_wires=[1]), control_wires=[2]
                ),
                re.ResourceToffoli([0, 1, 2]),
            ),
            (
                re.ResourceControlled(
                    re.ResourceControlled(re.ResourceX(0), control_wires=[1]), control_wires=[2]
                ),
                re.ResourceControlled(re.ResourceX(0), control_wires=[1, 2]),
            ),
        ],
    )
    def test_nested_controls(self, nested_op, expected_op):
        """Test the resources for nested Controlled operators."""

        nested_rep = nested_op.resource_rep_from_op()
        nested_params = nested_rep.params
        nested_type = nested_rep.op_type
        nested_resources = nested_type.resources(**nested_params)

        expected_rep = expected_op.resource_rep_from_op()
        expected_params = expected_rep.params
        expected_type = expected_rep.op_type
        expected_resources = expected_type.resources(**expected_params)

        assert nested_resources == expected_resources


class TestResourcePow:
    """Tests for ResourcePow"""

    def test_resource_params(self):
        """Test that the resources are correct"""

        base = re.ResourceQFT(wires=[0, 1, 2])
        op = re.ResourcePow(base=base, z=5)
        assert op.resource_params() == {
            "base_class": re.ResourceQFT,
            "z": 5,
            "base_params": base.resource_params(),
        }

    @pytest.mark.parametrize(
        "op, expected",
        [
            (re.ResourcePow(re.ResourceQFT([0, 1]), 2), "(QFT(2))**2"),
            (re.ResourcePow(re.ResourceAdjoint(re.ResourceQFT([0, 1])), 2), "(Adjoint(QFT(2)))**2"),
            (re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 3), "((QFT(2))**2)**3"),
        ],
    )
    def test_tracking_name(self, op, expected):
        """Test that the tracking name is correct"""
        rep = op.resource_rep_from_op()
        name = rep.op_type.tracking_name(**rep.params)
        assert name == expected

    @pytest.mark.parametrize(
        "nested_op, base_op, z",
        [
            (
                re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2),
                re.ResourceQFT([0, 1]),
                4,
            ),
            (
                re.ResourcePow(re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2), 2),
                re.ResourceQFT([0, 1]),
                8,
            ),
            (
                re.ResourcePow(
                    re.ResourcePow(re.ResourcePow(re.ResourcePow(re.ResourceQFT([0, 1]), 2), 2), 2),
                    2,
                ),
                re.ResourceQFT([0, 1]),
                16,
            ),
        ],
    )
    def test_nested_pow(self, nested_op, base_op, z):
        """Test the resources for nested Pow operators."""

        nested_rep = nested_op.resource_rep_from_op()
        nested_params = nested_rep.params
        nested_type = nested_rep.op_type
        nested_resources = nested_type.resources(**nested_params)

        base_rep = base_op.resource_rep_from_op()
        base_params = base_rep.params
        base_type = base_rep.op_type
        base_resources = base_type.resources(**base_params)

        assert nested_resources == _scale_dict(base_resources, z)
