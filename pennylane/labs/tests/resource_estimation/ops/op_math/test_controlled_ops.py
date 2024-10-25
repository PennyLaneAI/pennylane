import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceCNOT, ResourceControlledPhaseShift, ResourcesNotDefined

class TestControlledPhaseShift:
    """Test ResourceControlledPhaseShift"""

    @pytest.mark.parametrize("phi, wires",
        [
            (1.2, [0, 1]),
            (2.4, [2, 3]),
        ])
    def test_resources(self, phi, wires):
        """Test the resources method"""
        op = ResourceControlledPhaseShift(phi, wires)

        expected = {
                CompressedResourceOp(qml.CNOT, {}): 2,
                CompressedResourceOp(qml.RZ, {}): 3,
        }

        assert op.resources() == expected

    @pytest.mark.parametrize("phi, wires",
        [
            (1.2, [0, 1]),
            (2.4, [2, 3]),
        ])
    def test_resource_rep(self, phi, wires):
        """Test the compressed representation"""
        op = ResourceControlledPhaseShift(phi, wires)
        expected = CompressedResourceOp(qml.ControlledPhaseShift, {})

        assert op.resource_rep() == expected


    @pytest.mark.parametrize("phi, wires",
        [
            (1.2, [0, 1]),
            (2.4, [2, 3]),
        ])
    def test_resources_from_rep(self, phi, wires):
        """Compute the resources from the compressed representation"""
        op = ResourceControlledPhaseShift(phi, wires)

        expected = {
                CompressedResourceOp(qml.CNOT, {}): 2,
                CompressedResourceOp(qml.RZ, {}): 3,
        }

        assert op.resources(*op.resource_rep().params) == expected

class TestCNOT:
    """Test ResourceCNOT"""

    def test_resources(self):
        """Test that the resources method is not implemented"""
        op = ResourceCNOT([0, 1])
        with pytest.raises(ResourcesNotDefined):
            op.resources()

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = ResourceCNOT([0, 1])
        expected = CompressedResourceOp(qml.CNOT, {})
        assert op.resource_rep() == expected
