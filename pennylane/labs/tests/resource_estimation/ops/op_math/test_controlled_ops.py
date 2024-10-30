import pytest

import pennylane as qml

<<<<<<< HEAD
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceCNOT,
    ResourceControlledPhaseShift,
    ResourcesNotDefined,
)

=======
import pennylane.labs.resource_estimation as re

>>>>>>> resource_qft

class TestControlledPhaseShift:
    """Test ResourceControlledPhaseShift"""

<<<<<<< HEAD
    @pytest.mark.parametrize(
        "phi, wires",
        [
            (1.2, [0, 1]),
            (2.4, [2, 3]),
        ],
    )
=======
    params = [(1.2, [0, 1]), (2.4, [2, 3])]

    @pytest.mark.parametrize("phi, wires", params)
>>>>>>> resource_qft
    def test_resources(self, phi, wires):
        """Test the resources method"""

        op = re.ResourceControlledPhaseShift(phi, wires)

        expected = {
<<<<<<< HEAD
            CompressedResourceOp(qml.CNOT, {}): 2,
            CompressedResourceOp(qml.RZ, {}): 3,
=======
                re.CompressedResourceOp(qml.CNOT, {}): 2,
                re.CompressedResourceOp(qml.RZ, {"epsilon": 10e-3}): 3,
>>>>>>> resource_qft
        }

        assert op.resources() == expected

<<<<<<< HEAD
    @pytest.mark.parametrize(
        "phi, wires",
        [
            (1.2, [0, 1]),
            (2.4, [2, 3]),
        ],
    )
=======
    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_params(self, phi, wires):
        """Test the resource parameters"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        assert op.resource_params() == {} #pylint: disable=use-implicit-booleaness-not-comparison

    @pytest.mark.parametrize("phi, wires", params)
>>>>>>> resource_qft
    def test_resource_rep(self, phi, wires):
        """Test the compressed representation"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        expected = re.CompressedResourceOp(qml.ControlledPhaseShift, {})

        assert op.resource_rep() == expected

<<<<<<< HEAD
    @pytest.mark.parametrize(
        "phi, wires",
        [
            (1.2, [0, 1]),
            (2.4, [2, 3]),
        ],
    )
=======
    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_rep_from_op(self, phi, wires):
        """Test resource_rep_from_op method"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        assert op.resource_rep_from_op() == re.ResourceControlledPhaseShift.resource_rep(**op.resource_params())

    @pytest.mark.parametrize("phi, wires", params)
>>>>>>> resource_qft
    def test_resources_from_rep(self, phi, wires):
        """Compute the resources from the compressed representation"""

        op = re.ResourceControlledPhaseShift(phi, wires)

        expected = {
<<<<<<< HEAD
            CompressedResourceOp(qml.CNOT, {}): 2,
            CompressedResourceOp(qml.RZ, {}): 3,
=======
                re.CompressedResourceOp(qml.CNOT, {}): 2,
                re.CompressedResourceOp(qml.RZ, {"epsilon": 10e-3}): 3,
>>>>>>> resource_qft
        }

        assert op.resources(**re.ResourceControlledPhaseShift.resource_rep(**op.resource_params()).params) == expected


class TestCNOT:
    """Test ResourceCNOT"""

    def test_resources(self):
        """Test that the resources method is not implemented"""
        op = re.ResourceCNOT([0, 1])
        with pytest.raises(re.ResourcesNotDefined):
            op.resources()

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = re.ResourceCNOT([0, 1])
        expected = re.CompressedResourceOp(qml.CNOT, {})
        assert op.resource_rep() == expected
