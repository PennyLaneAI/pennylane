import numpy as np
import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourcesNotDefined, ResourceRZ
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import _rotation_resources

@pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
def test_rotation_resources(epsilon):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = CompressedResourceOp(qml.T, {})
    gate_types[t] = num_gates
    assert gate_types == _rotation_resources(epsilon=epsilon)

class TestRZ:
    """Test ResourceRZ"""

    @pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
    def test_resources(self, epsilon):
        """Test the resources method"""
        op = ResourceRZ(1.24, wires=0)
        assert op.resources(epsilon=epsilon) == _rotation_resources(epsilon=epsilon)

    @pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
    def test_resource_rep(self, epsilon):
        """Test the compact representation"""
        op = ResourceRZ(1.24, wires=0)
        expected = CompressedResourceOp(qml.RZ, {"epsilon": epsilon})

        assert op.resource_rep(epsilon=epsilon) == expected

    @pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
    def test_resources_from_rep(self, epsilon):
        """Test the resources can be obtained from the compact representation"""
        op = ResourceRZ(1.24, wires=0)
        expected = _rotation_resources(epsilon=epsilon)

        assert ResourceRZ.resources(**op.resource_rep(epsilon=epsilon).params) == expected
