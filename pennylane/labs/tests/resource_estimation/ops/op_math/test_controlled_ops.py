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
Tests for controlled resource operators.
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison


class TestResourceCH:
    """Test the ResourceCH operation"""

    op = re.ResourceCH(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceRY.resource_rep(): 2,
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 1,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCH, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCY:
    """Test the ResourceCY operation"""

    op = re.ResourceCY(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceS.resource_rep(): 4,
            re.ResourceCNOT.resource_rep(): 1,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCY, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCZ:
    """Test the ResourceCZ operation"""

    op = re.ResourceCZ(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 1,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCZ, {})
        assert self.op.resource_rep() == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCSWAP:
    """Test the ResourceCSWAP operation"""

    op = re.ResourceCSWAP(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = {
            re.ResourceToffoli.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCSWAP, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCCZ:
    """Test the ResourceCZZ operation"""

    op = re.ResourceCCZ(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceToffoli.resource_rep(): 1,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCCZ, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCNOT:
    """Test ResourceCNOT operation"""

    op = re.ResourceCNOT([0, 1])

    def test_resources(self):
        """Test that the resources method is not implemented"""
        with pytest.raises(re.ResourcesNotDefined):
            self.op.resources()

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected = re.CompressedResourceOp(re.ResourceCNOT, {})
        assert self.op.resource_rep() == expected

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceToffoli:
    """Test the ResourceToffoli operation"""

    op = re.ResourceToffoli(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceS.resource_rep(): 1,
            re.ResourceT.resource_rep(): 16,
            re.ResourceCZ.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 9,
            re.ResourceHadamard.resource_rep(): 3,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceToffoli, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceMultiControlledX:
    """Test the ResourceMultiControlledX operation"""

    res_ops = (
        re.ResourceMultiControlledX(control_wires=[0], wires=["t"], control_values=[1]),
        re.ResourceMultiControlledX(control_wires=[0, 1], wires=["t"], control_values=[1, 1]),
        re.ResourceMultiControlledX(control_wires=[0, 1, 2], wires=["t"], control_values=[1, 1, 1]),
        re.ResourceMultiControlledX(
            control_wires=[0, 1, 2, 3, 4], wires=["t"], control_values=[1, 1, 1, 1, 1]
        ),
        re.ResourceMultiControlledX(
            control_wires=[0], wires=["t"], control_values=[0], work_wires=["w1"]
        ),
        re.ResourceMultiControlledX(
            control_wires=[0, 1], wires=["t"], control_values=[1, 0], work_wires=["w1", "w2"]
        ),
        re.ResourceMultiControlledX(control_wires=[0, 1, 2], wires=["t"], control_values=[0, 0, 1]),
        re.ResourceMultiControlledX(
            control_wires=[0, 1, 2, 3, 4],
            wires=["t"],
            control_values=[1, 0, 0, 1, 0],
            work_wires=["w1"],
        ),
    )

    res_params = (
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (5, 0, 0),
        (1, 1, 1),
        (2, 1, 2),
        (3, 2, 0),
        (5, 3, 1),
    )

    expected_resources = (
        {re.ResourceCNOT.resource_rep(): 1},
        {re.ResourceToffoli.resource_rep(): 1},
        {
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceToffoli.resource_rep(): 1,
        },
        {re.ResourceCNOT.resource_rep(): 69},
        {
            re.ResourceX.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 1,
        },
        {
            re.ResourceX.resource_rep(): 2,
            re.ResourceToffoli.resource_rep(): 1,
        },
        {
            re.ResourceX.resource_rep(): 4,
            re.ResourceCNOT.resource_rep(): 2,
            re.ResourceToffoli.resource_rep(): 1,
        },
        {
            re.ResourceX.resource_rep(): 6,
            re.ResourceCNOT.resource_rep(): 69,
        },
    )

    @staticmethod
    def _prep_params(num_control, num_control_values, num_work_wires):
        return {
            "num_ctrl_wires": num_control,
            "num_ctrl_values": num_control_values,
            "num_work_wires": num_work_wires,
        }

    @pytest.mark.parametrize("params, expected_res", zip(res_params, expected_resources))
    def test_resources(self, params, expected_res):
        """Test that the resources method produces the expected resources."""
        op_resource_params = self._prep_params(*params)
        assert re.ResourceMultiControlledX.resources(**op_resource_params) == expected_res

    @pytest.mark.parametrize("op, params", zip(res_ops, res_params))
    def test_resource_rep(self, op, params):
        """Test the resource_rep produces the correct compressed representation."""
        op_resource_params = self._prep_params(*params)
        expected_rep = re.CompressedResourceOp(re.ResourceMultiControlledX, op_resource_params)
        assert op.resource_rep(**op.resource_params()) == expected_rep

    @pytest.mark.parametrize("op, params", zip(res_ops, res_params))
    def test_resource_params(self, op, params):
        """Test that the resource_params are produced as expected."""
        expected_params = self._prep_params(*params)
        assert op.resource_params() == expected_params


class TestResourceCRX:
    """Test the ResourceCRX operation"""

    op = re.ResourceCRX(phi=1.23, wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceRZ.resource_rep(): 2,
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRX, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCRY:
    """Test the ResourceCRY operation"""

    op = re.ResourceCRY(phi=1.23, wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceRY.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRY, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCRZ:
    """Test the ResourceCRZ operation"""

    op = re.ResourceCRZ(phi=1.23, wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceRZ.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRZ, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceCRot:
    """Test the ResourceCRot operation"""

    op = re.ResourceCRot(0.1, 0.2, 0.3, wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = {
            re.ResourceRY.resource_rep(): 2,
            re.ResourceRZ.resource_rep(): 3,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params()) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRot, {})
        assert self.op.resource_rep(**self.op.resource_params()) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params() == expected_params


class TestResourceControlledPhaseShift:
    """Test ResourceControlledPhaseShift"""

    params = [(1.2, [0, 1]), (2.4, [2, 3])]

    @pytest.mark.parametrize("phi, wires", params)
    def test_resources(self, phi, wires):
        """Test the resources method"""

        op = re.ResourceControlledPhaseShift(phi, wires)

        expected = {
            re.CompressedResourceOp(re.ResourceCNOT, {}): 2,
            re.CompressedResourceOp(re.ResourceRZ, {}): 3,
        }

        assert op.resources(**op.resource_params()) == expected

    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_params(self, phi, wires):
        """Test the resource parameters"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        assert op.resource_params() == {}  # pylint: disable=use-implicit-booleaness-not-comparison

    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_rep(self, phi, wires):
        """Test the compressed representation"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        expected = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        assert op.resource_rep() == expected

    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_rep_from_op(self, phi, wires):
        """Test resource_rep_from_op method"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        assert op.resource_rep_from_op() == re.ResourceControlledPhaseShift.resource_rep(
            **op.resource_params()
        )

    @pytest.mark.parametrize("phi, wires", params)
    def test_resources_from_rep(self, phi, wires):
        """Compute the resources from the compressed representation"""

        op = re.ResourceControlledPhaseShift(phi, wires)

        expected = {
            re.CompressedResourceOp(re.ResourceCNOT, {}): 2,
            re.CompressedResourceOp(re.ResourceRZ, {}): 3,
        }

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_params = op_compressed_rep.params
        op_compressed_rep_type = op_compressed_rep.op_type

        assert op_compressed_rep_type.resources(**op_resource_params) == expected
