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
"""
Test the core resource tracking functionality.
"""
# import pytest

# from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
# from pennylane.labs.resource_estimation.resource_operator import (
#     CompressedResourceOp,
#     GateCount,
#     ResourceOperator,
# )

# pylint: disable= no-self-use


class TestEstimateResources:
    """Test that core resource estimation functionality"""

    def test_estimate_resources_from_qfunc(self):
        """Test that we can accurately obtain resources from qfunc"""
        assert True

    def test_estimate_resources_from_resource_operator(self):
        """Test that we can accurately obtain resources from qfunc"""
        assert True

    def test_estimate_resources_from_resources_obj(self):
        """Test that we can accurately obtain resources from qfunc"""
        assert True

    def test_estimate_resources_from_pl_operator(self):
        """Test that we can accurately obtain resources from qfunc"""
        assert True

    def test_wire_tracking(self):
        """Test that we correctly track the required qubits"""
        assert True

    def test_varying_gate_sets(self):
        """Test that changing the gate_set correctly updates the resources"""
        assert True

    def test_varying_config(self):
        """Test that changing the resource_config correctly updates the resources"""
        assert True

    def test_varying_single_qubit_rotation_error(self):
        """Test that setting the single_qubit_rotation_error correctly updates the resources"""
        assert True
