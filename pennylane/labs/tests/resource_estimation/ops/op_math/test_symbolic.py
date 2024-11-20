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

import pennylane.labs.resource_estimation as re

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
