import numpy as np
import pytest

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    _rotation_resources,
)


@pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
def test_rotation_resources(epsilon):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = {}

    num_gates = round(1.149 * np.log2(1 / epsilon) + 9.2)
    t = re.CompressedResourceOp(re.ResourceT, {})
    gate_types[t] = num_gates
    assert gate_types == _rotation_resources(epsilon=epsilon)


class TestRZ:
    """Test ResourceRZ"""

    @pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
    def test_resources(self, epsilon):
        """Test the resources method"""
        op = re.ResourceRZ(1.24, wires=0)
        config = {"error_rz": epsilon}
        assert op.resources(config) == _rotation_resources(epsilon=epsilon)

    @pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
    def test_resource_rep(self, epsilon):
        """Test the compact representation"""
        op = re.ResourceRZ(1.24, wires=0)
        expected = re.CompressedResourceOp(re.ResourceRZ, {})

        assert op.resource_rep() == expected

    @pytest.mark.parametrize("epsilon", [10e-3, 10e-4, 10e-5])
    def test_resources_from_rep(self, epsilon):
        """Test the resources can be obtained from the compact representation"""
        config = {"error_rz": epsilon}
        expected = _rotation_resources(epsilon=epsilon)
        assert (
            re.ResourceRZ.resources(config, **re.ResourceRZ.resource_rep().params)
            == expected
        )
