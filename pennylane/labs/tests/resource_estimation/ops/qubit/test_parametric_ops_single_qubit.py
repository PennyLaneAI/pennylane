import numpy as np
import pytest

import pennylane as qml
from pennylane import RX, RY, RZ #pylint: disable=unused-import
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceRX, ResourceRY, ResourceRZ
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import _rotation_resources

@pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
def test_rotation_resources(epsilon):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = CompressedResourceOp(qml.T, {})
    gate_types[t] = num_gates
    assert gate_types == _rotation_resources(epsilon=epsilon)

class TestPauliRotation:
    """Test ResourceRX, ResourceRY, and ResourceRZ"""

    params = list(zip([ResourceRX, ResourceRY, ResourceRZ], [10e-3, 10e-4, 10e-5]))

    @pytest.mark.parametrize("resource_class, epsilon", params)
    def test_resources(self, epsilon, resource_class):
        """Test the resources method"""

        op = resource_class(1.24, wires=0)
        assert op.resources(epsilon=epsilon) == _rotation_resources(epsilon=epsilon)

    @pytest.mark.parametrize("resource_class, epsilon", params)
    def test_resource_rep(self, epsilon, resource_class):
        """Test the compact representation"""

        op = resource_class(1.24, wires=0)
        pl_class = globals()[resource_class.__name__[8:]]
        expected = CompressedResourceOp(pl_class, {"epsilon": epsilon})

        assert op.resource_rep(epsilon=epsilon) == expected

    @pytest.mark.parametrize("resource_class, epsilon", params)
    def test_resources_from_rep(self, epsilon, resource_class):
        """Test the resources can be obtained from the compact representation"""

        op = resource_class(1.24, wires=0)
        expected = _rotation_resources(epsilon=epsilon)

        assert resource_class.resources(**op.resource_rep(epsilon=epsilon).params) == expected
