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
"""Tests for qchem ops resource operators."""

import pennylane.labs.resource_estimation as re

# pylint: disable=use-implicit-booleaness-not-comparison,no-self-use


class TestSingleExcitation:
    """Tests for the ResourceSingleExcitation class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceAdjoint.make_resource_rep(re.ResourceT, {}): 2,
            re.ResourceHadamard.make_resource_rep(): 4,
            re.ResourceS.make_resource_rep(): 2,
            re.ResourceAdjoint.make_resource_rep(re.ResourceS, {}): 2,
            re.ResourceCNOT.make_resource_rep(): 2,
            re.ResourceRZ.make_resource_rep(): 1,
            re.ResourceRY.make_resource_rep(): 1,
            re.ResourceT.make_resource_rep(): 2,
        }
        assert re.ResourceSingleExcitation.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceSingleExcitation(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceSingleExcitation, {})
        assert re.ResourceSingleExcitation.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceSingleExcitation(0.5, wires=[0, 1])
        expected = {
            re.ResourceAdjoint.make_resource_rep(re.ResourceT, {}): 2,
            re.ResourceHadamard.make_resource_rep(): 4,
            re.ResourceS.make_resource_rep(): 2,
            re.ResourceAdjoint.make_resource_rep(re.ResourceS, {}): 2,
            re.ResourceCNOT.make_resource_rep(): 2,
            re.ResourceRZ.make_resource_rep(): 1,
            re.ResourceRY.make_resource_rep(): 1,
            re.ResourceT.make_resource_rep(): 2,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestSingleExcitationMinus:
    """Tests for the ResourceSingleExcitationMinus class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceX.make_resource_rep(): 4,
            re.ResourceControlledPhaseShift.make_resource_rep(): 2,
            re.ResourceCNOT.make_resource_rep(): 2,
            re.ResourceCRY.make_resource_rep(): 1,
        }
        assert re.ResourceSingleExcitationMinus.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceSingleExcitationMinus(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceSingleExcitationMinus, {})
        assert re.ResourceSingleExcitationMinus.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceSingleExcitationMinus(0.5, wires=[0, 1])
        expected = {
            re.ResourceX.make_resource_rep(): 4,
            re.ResourceControlledPhaseShift.make_resource_rep(): 2,
            re.ResourceCNOT.make_resource_rep(): 2,
            re.ResourceCRY.make_resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestSingleExcitationPlus:
    """Tests for the ResourceSingleExcitationPlus class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceX.make_resource_rep(): 4,
            re.ResourceControlledPhaseShift.make_resource_rep(): 2,
            re.ResourceCNOT.make_resource_rep(): 2,
            re.ResourceCRY.make_resource_rep(): 1,
        }
        assert re.ResourceSingleExcitationPlus.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceSingleExcitationPlus(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceSingleExcitationPlus, {})
        assert re.ResourceSingleExcitationPlus.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceSingleExcitationPlus(0.5, wires=[0, 1])
        expected = {
            re.ResourceX.make_resource_rep(): 4,
            re.ResourceControlledPhaseShift.make_resource_rep(): 2,
            re.ResourceCNOT.make_resource_rep(): 2,
            re.ResourceCRY.make_resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestDoubleExcitation:
    """Tests for the ResourceDoubleExcitation class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceHadamard.make_resource_rep(): 6,
            re.ResourceRY.make_resource_rep(): 8,
            re.ResourceCNOT.make_resource_rep(): 14,
        }
        assert re.ResourceDoubleExcitation.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceDoubleExcitation(0.5, wires=[0, 1, 2, 3])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceDoubleExcitation, {})
        assert re.ResourceDoubleExcitation.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceDoubleExcitation(0.5, wires=[0, 1, 2, 3])
        expected = {
            re.ResourceHadamard.make_resource_rep(): 6,
            re.ResourceRY.make_resource_rep(): 8,
            re.ResourceCNOT.make_resource_rep(): 14,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestDoubleExcitationMinus:
    """Tests for the ResourceDoubleExcitationMinus class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceGlobalPhase.make_resource_rep(): 1,
            re.ResourceDoubleExcitation.make_resource_rep(): 1,
            re.ResourceControlled.make_resource_rep(re.ResourceZ, {}, 3, 1, 0): 2,
            re.ResourceControlled.make_resource_rep(re.ResourcePhaseShift, {}, 3, 1, 0): 2,
        }
        assert re.ResourceDoubleExcitationMinus.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceDoubleExcitationMinus(0.5, wires=[0, 1, 2, 3])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceDoubleExcitationMinus, {})
        assert re.ResourceDoubleExcitationMinus.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceDoubleExcitationMinus(0.5, wires=[0, 1, 2, 3])
        expected = {
            re.ResourceGlobalPhase.make_resource_rep(): 1,
            re.ResourceDoubleExcitation.make_resource_rep(): 1,
            re.ResourceControlled.make_resource_rep(re.ResourceZ, {}, 3, 1, 0): 2,
            re.ResourceControlled.make_resource_rep(re.ResourcePhaseShift, {}, 3, 1, 0): 2,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestDoubleExcitationPlus:
    """Tests for the ResourceDoubleExcitationPlus class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceGlobalPhase.make_resource_rep(): 1,
            re.ResourceDoubleExcitation.make_resource_rep(): 1,
            re.ResourceControlled.make_resource_rep(re.ResourceZ, {}, 3, 1, 0): 2,
            re.ResourceControlled.make_resource_rep(re.ResourcePhaseShift, {}, 3, 1, 0): 2,
        }
        assert re.ResourceDoubleExcitationPlus.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceDoubleExcitationPlus(0.5, wires=[0, 1, 3, 4])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceDoubleExcitationPlus, {})
        assert re.ResourceDoubleExcitationPlus.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceDoubleExcitationPlus(0.5, wires=[0, 1, 3, 4])
        expected = {
            re.ResourceGlobalPhase.make_resource_rep(): 1,
            re.ResourceDoubleExcitation.make_resource_rep(): 1,
            re.ResourceControlled.make_resource_rep(re.ResourceZ, {}, 3, 1, 0): 2,
            re.ResourceControlled.make_resource_rep(re.ResourcePhaseShift, {}, 3, 1, 0): 2,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestOrbitalRotation:
    """Tests for the ResourceOrbitalRotation class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceFermionicSWAP.make_resource_rep(): 2,
            re.ResourceSingleExcitation.make_resource_rep(): 2,
        }
        assert re.ResourceOrbitalRotation.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceOrbitalRotation(0.5, wires=[0, 1, 3, 4])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceOrbitalRotation, {})
        assert re.ResourceOrbitalRotation.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceOrbitalRotation(0.5, wires=[0, 1, 3, 4])
        expected = {
            re.ResourceFermionicSWAP.make_resource_rep(): 2,
            re.ResourceSingleExcitation.make_resource_rep(): 2,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestFermionicSWAP:
    """Tests for the ResourceFermionicSWAP class."""

    def test_resources(self):
        """Test that the resources are correct."""
        expected = {
            re.ResourceHadamard.make_resource_rep(): 4,
            re.ResourceMultiRZ.make_resource_rep(num_wires=2): 2,
            re.ResourceRX.make_resource_rep(): 4,
            re.ResourceRZ.make_resource_rep(): 2,
            re.ResourceGlobalPhase.make_resource_rep(): 1,
        }
        assert re.ResourceFermionicSWAP.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = re.ResourceFermionicSWAP(0.5, wires=[0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = re.CompressedResourceOp(re.ResourceFermionicSWAP, {})
        assert re.ResourceFermionicSWAP.make_resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compressed representation."""
        op = re.ResourceFermionicSWAP(0.5, wires=[0, 1])
        expected = {
            re.ResourceHadamard.make_resource_rep(): 4,
            re.ResourceMultiRZ.make_resource_rep(num_wires=2): 2,
            re.ResourceRX.make_resource_rep(): 4,
            re.ResourceRZ.make_resource_rep(): 2,
            re.ResourceGlobalPhase.make_resource_rep(): 1,
        }
        op_compressed_rep = op.resource_rep()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected
