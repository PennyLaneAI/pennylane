import pytest

import pennylane as qml
import pennylane.labs.resource_estimation.ops as ops
from pennylane.labs.resource_estimation import CompressedResourceOp

class TestQFT:
    """Test the ResourceQFT class"""

    @pytest.mark.parametrize("num_wires, num_hadamard, num_swap, num_ctrl_phase_shift",
        [
            (1, 1, 0, 0),
            (2, 2, 1, 1),
            (3, 3, 1, 3),
            (4, 4, 2, 6),
        ]
    )
    def test_compute_resources(self, num_wires, num_hadamard, num_swap, num_ctrl_phase_shift):
        """Test the compute_resources method returns the correct dictionary"""
        hadamard = CompressedResourceOp(qml.Hadamard, {})
        swap = CompressedResourceOp(qml.SWAP, {})
        ctrl_phase_shift = CompressedResourceOp(qml.ControlledPhaseShift, {})

        expected = {
            hadamard: num_hadamard,
            swap: num_swap,
            ctrl_phase_shift: num_ctrl_phase_shift
        }

        assert ops.ResourceQFT.compute_resources(num_wires) == expected

    @pytest.mark.parametrize("num_wires", [1, 2, 3, 4])
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = CompressedResourceOp(qml.QFT, {"num_wires": num_wires})
        op = ops.ResourceQFT(wires=range(num_wires))

        assert op.resource_rep() == expected

    @pytest.mark.parametrize("num_wires, num_hadamard, num_swap, num_ctrl_phase_shift",
        [
            (1, 1, 0, 0),
            (2, 2, 1, 1),
            (3, 3, 1, 3),
            (4, 4, 2, 6),
        ]
    )
    def test_resources_from_rep(self, num_wires, num_hadamard, num_swap, num_ctrl_phase_shift):
        """Test that computing the resources from a compressed representation works"""

        hadamard = CompressedResourceOp(qml.Hadamard, {})
        swap = CompressedResourceOp(qml.SWAP, {})
        ctrl_phase_shift = CompressedResourceOp(qml.ControlledPhaseShift, {})

        expected = {
            hadamard: num_hadamard,
            swap: num_swap,
            ctrl_phase_shift: num_ctrl_phase_shift
        }

        rep = ops.ResourceQFT(wires=range(num_wires)).resource_rep()
        actual = ops.ResourceQFT.compute_resources(**rep.params)

        assert actual == expected

    @pytest.mark.parametrize("num_wires", [2.5, -0.5])
    def test_type_error(self, num_wires):
        """Test that compute_resources correctly raises a TypeError"""
        with pytest.raises(TypeError, match="num_wires must be an int."):
            ops.ResourceQFT.compute_resources(num_wires)

    @pytest.mark.parametrize("num_wires", [0, -1])
    def test_value_error(self, num_wires):
        """Test that compute_resources correctly raises a ValueError"""
        with pytest.raises(ValueError, match="num_wires must be greater than 0."):
            ops.ResourceQFT.compute_resources(num_wires)
