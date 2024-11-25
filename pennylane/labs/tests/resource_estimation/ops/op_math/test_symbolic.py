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

    def test_name(self):
        """Test that the name of the compressed representation is correct"""

        base = re.ResourceQFT(wires=[0, 1, 2])
        op = re.ResourceAdjoint(base=base)
        assert op.resource_rep_from_op()._name == "Adjoint(QFT)"

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
            "num_ctrl_vals": 1,
        }

    def test_name(self):
        """Test that the name of the compressed representation is correct"""

        base = re.ResourceQFT(wires=[0, 1, 2])
        op = re.ResourceControlled(base=base, control_wires=[3])
        assert op.resource_rep_from_op()._name == "Controlled(QFT, wires=1)"


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

    def test_name(self):
        """Test that the name of the compressed representation is correct"""

        base = re.ResourceQFT(wires=[0, 1, 2])
        op = re.ResourcePow(base=base, z=5)
        assert op.resource_rep_from_op()._name == "QFT**5"

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

        base_op = base_op.resource_rep_from_op()
        base_params = base_op.params
        base_type = base_op.op_type
        base_resources = base_type.resources(**base_params)

        assert nested_resources == _scale_dict(base_resources, z)
