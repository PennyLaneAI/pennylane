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
"""Tests for controlled resource operators."""
import pytest

from pennylane.estimator.ops import CH, CNOT, RY, Hadamard, Identity
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourcesNotDefined,
)

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments,too-many-positional-arguments


class TestCH:
    """Test the CH operation"""

    op = CH(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(Hadamard.resource_rep(), 2),
            GateCount(RY.resource_rep(), 2),
            GateCount(CNOT.resource_rep(), 1),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CH, 2, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    def test_resource_controlled(self):
        """Test that the controlled resources are as expected"""
        with pytest.raises(ResourcesNotDefined):
            self.op.controlled_resource_decomp(1, 0)

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""

        assert self.op.pow_resource_decomp(z) == expected_res


class TestCY:
    """Test the CY operation"""

    op = CY(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(CNOT.resource_rep(), 1),
            GateCount(S.resource_rep(), 2),
            GateCount(Z.resource_rep(), 1),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CY, 2, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [re.GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (1, 0, 0)

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values):
        """Test that the controlled resources are as expected"""

        with pytest.raises(ResourcesNotDefined):
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""

        assert self.op.pow_resource_decomp(z) == expected_res


class TestCZ:
    """Test the CZ operation"""

    op = CZ(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(CNOT.resource_rep(), 1),
            GateCount(Hadamard.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CZ, 2, {})
        assert self.op.resource_rep() == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]

        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (1, 0)

    @pytest.mark.parametrize("num_ctrl_wires, num_ctrl_values", ctrl_data)
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values):
        """Test that the controlled resources are as expected"""
        with pytest.raises(ResourcesNotDefined):
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = [
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    ]

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestCSWAP:
    """Test the CSWAP operation"""

    op = CSWAP(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = [
            GateCount(Toffoli.resource_rep(), 1),
            GateCount(CNOT.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CSWAP, 3, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (1, 0)

    @pytest.mark.parametrize("num_ctrl_wires, num_ctrl_values", ctrl_data)
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values):
        """Test that the controlled resources are as expected"""
        with pytest.raises(ResourcesNotDefined):
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestCCZ:
    """Test the CCZ operation"""

    op = CCZ(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = [
            GateCount(Toffoli.resource_rep(), 1),
            GateCount(Hadamard.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CCZ, 3, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]

        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (1, 0)

    @pytest.mark.parametrize("num_ctrl_wires, num_ctrl_values", ctrl_data)
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values):
        """Test that the controlled resources are as expected"""
        with pytest.raises(ResourcesNotDefined):
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestCNOT:
    """Test Resource CNOT operation"""

    op = CNOT([0, 1])

    def test_resources(self):
        """Test that the resources method is not implemented"""
        with pytest.raises(ResourcesNotDefined):
            self.op.resource_decomp()

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected = CompressedResourceOp(CNOT, 2, {})
        assert self.op.resource_rep() == expected

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (["c1"], [1])

    @pytest.mark.parametrize("ctrl_wires, ctrl_values", ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        with pytest.raises(ResourcesNotDefined):
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
        (8, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestToffoli:
    """Test the Resource Toffoli operation"""

    op = Toffoli(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            Allocate(2),
            GateCount(CNOT.resource_rep(), 9),
            GateCount(Hadamard.resource_rep(), 3),
            GateCount(S.resource_rep(), 3),
            GateCount(CZ.resource_rep(), 1),
            GateCount(T.resource_rep(), 4),
            GateCount(Z.resource_rep(), 2),
            Deallocate(2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceToffoli, 3, {"elbow": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"elbow": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]

        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (["c1"], [1])

    @pytest.mark.parametrize("ctrl_wires, ctrl_values", ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        with pytest.raises(ResourcesNotDefined):
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [re.GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
        (8, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestResourceMultiControlledX:
    """Test the ResourceMultiControlledX operation"""

    res_ops = (
        re.ResourceMultiControlledX(1, 0),
        re.ResourceMultiControlledX(2, 0),
        re.ResourceMultiControlledX(3, 0),
        re.ResourceMultiControlledX(5, 0),
        re.ResourceMultiControlledX(1, 1),
        re.ResourceMultiControlledX(2, 1),
        re.ResourceMultiControlledX(3, 2),
        re.ResourceMultiControlledX(5, 3),
    )

    res_params = (
        (1, 0),
        (2, 0),
        (3, 0),
        (5, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 3),
    )

    expected_resources = (
        [re.GateCount(re.ResourceCNOT.resource_rep(), 1)],
        [re.GateCount(re.ResourceToffoli.resource_rep(), 1)],
        [
            AllocWires(1),
            re.GateCount(re.ResourceTempAND.resource_rep(), 1),
            re.GateCount(re.ResourceAdjoint.resource_rep(re.ResourceTempAND.resource_rep()), 1),
            re.GateCount(re.ResourceToffoli.resource_rep(), 1),
            FreeWires(1),
        ],
        [
            AllocWires(3),
            re.GateCount(re.ResourceTempAND.resource_rep(), 3),
            re.GateCount(re.ResourceAdjoint.resource_rep(re.ResourceTempAND.resource_rep()), 3),
            re.GateCount(re.ResourceToffoli.resource_rep(), 1),
            FreeWires(3),
        ],
        [
            re.GateCount(re.ResourceX.resource_rep(), 2),
            re.GateCount(re.ResourceCNOT.resource_rep(), 1),
        ],
        [
            re.GateCount(re.ResourceX.resource_rep(), 2),
            re.GateCount(re.ResourceToffoli.resource_rep(), 1),
        ],
        [
            re.GateCount(re.resource_rep(re.ResourceX), 4),
            AllocWires(1),
            re.GateCount(re.ResourceTempAND.resource_rep(), 1),
            re.GateCount(re.ResourceAdjoint.resource_rep(re.ResourceTempAND.resource_rep()), 1),
            re.GateCount(re.ResourceToffoli.resource_rep(), 1),
            FreeWires(1),
        ],
        [
            re.GateCount(re.resource_rep(re.ResourceX), 6),
            AllocWires(3),
            re.GateCount(re.ResourceTempAND.resource_rep(), 3),
            re.GateCount(re.ResourceAdjoint.resource_rep(re.ResourceTempAND.resource_rep()), 3),
            re.GateCount(re.ResourceToffoli.resource_rep(), 1),
            FreeWires(3),
        ],
    )

    @staticmethod
    def _prep_params(num_control, num_control_values):
        return {
            "num_ctrl_wires": num_control,
            "num_ctrl_values": num_control_values,
        }

    @pytest.mark.parametrize("params, expected_res", zip(res_params, expected_resources))
    def test_resources(self, params, expected_res):
        """Test that the resources method produces the expected resources."""
        op_resource_params = self._prep_params(*params)
        assert re.ResourceMultiControlledX.resource_decomp(**op_resource_params) == expected_res

    @pytest.mark.parametrize("op, params", zip(res_ops, res_params))
    def test_resource_rep(self, op, params):
        """Test the resource_rep produces the correct compressed representation."""
        op_resource_params = self._prep_params(*params)
        num_wires = op_resource_params["num_ctrl_wires"] + 1
        expected_rep = re.CompressedResourceOp(
            re.ResourceMultiControlledX, num_wires, op_resource_params
        )
        assert op.resource_rep(**op.resource_params) == expected_rep

    @pytest.mark.parametrize("op, params", zip(res_ops, res_params))
    def test_resource_params(self, op, params):
        """Test that the resource_params are produced as expected."""
        expected_params = self._prep_params(*params)
        assert op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = re.ResourceMultiControlledX(5, 3)
        expected_res = [re.GateCount(op.resource_rep(**op.resource_params), 1)]

        assert op.adjoint_resource_decomp(**op.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [re.GateCount(re.ResourceMultiControlledX.resource_rep(4, 2))],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                re.GateCount(re.ResourceMultiControlledX.resource_rep(5, 2)),
            ],
        ),
        (
            ["c1", "c2", "c3", "c4"],
            [1, 0, 0, 1],
            [
                re.GateCount(re.ResourceMultiControlledX.resource_rep(7, 4)),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        op = re.ResourceMultiControlledX(3, 2)
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, **op.resource_params)
            == expected_res
        )

    pow_data = (
        (1, [re.GateCount(re.ResourceMultiControlledX.resource_rep(5, 3), 1)]),
        (2, [re.GateCount(re.ResourceIdentity.resource_rep())]),
        (5, [re.GateCount(re.ResourceMultiControlledX.resource_rep(5, 3), 1)]),
        (6, [re.GateCount(re.ResourceIdentity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourceMultiControlledX(5, 3)
        assert op.pow_resource_decomp(z, **op.resource_params) == expected_res


class TestResourceCRX:
    """Test the ResourceCRX operation"""

    op = re.ResourceCRX(wires=[0, 1])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"eps"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(re.ResourceRZ.resource_rep(), 2),
            re.GateCount(re.ResourceHadamard.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRX, 2, {"eps": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"eps": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [re.GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(re.ResourceControlled(re.ResourceRX(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(re.ResourceControlled(re.ResourceRX(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(re.ResourceControlled(re.ResourceRX(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == [expected_res]

    pow_data = (
        (1, [re.GateCount(op.resource_rep(), 1)]),
        (2, [re.GateCount(op.resource_rep(), 1)]),
        (5, [re.GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""

        assert self.op.pow_resource_decomp(z) == expected_res


class TestResourceCRY:
    """Test the ResourceCRY operation"""

    op = re.ResourceCRY(wires=[0, 1])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"eps"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(re.ResourceRY.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRY, 2, {"eps": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"eps": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [re.GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(re.ResourceControlled(re.ResourceRY(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(re.ResourceControlled(re.ResourceRY(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(re.ResourceControlled(re.ResourceRY(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == [expected_res]

    pow_data = (
        (1, [re.GateCount(op.resource_rep(), 1)]),
        (2, [re.GateCount(op.resource_rep(), 1)]),
        (5, [re.GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestResourceCRZ:
    """Test the ResourceCRZ operation"""

    op = re.ResourceCRZ(wires=[0, 1])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"eps"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(re.ResourceRZ.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRZ, 2, {"eps": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"eps": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [re.GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(re.ResourceControlled(re.ResourceRZ(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(re.ResourceControlled(re.ResourceRZ(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(re.ResourceControlled(re.ResourceRZ(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == [expected_res]

    pow_data = (
        (1, [re.GateCount(op.resource_rep(), 1)]),
        (2, [re.GateCount(op.resource_rep(), 1)]),
        (5, [re.GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestResourceCRot:
    """Test the ResourceCRot operation"""

    op = re.ResourceCRot(wires=[0, 1])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"eps"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = [
            re.GateCount(re.ResourceCNOT.resource_rep(), 2),
            re.GateCount(re.ResourceRZ.resource_rep(), 3),
            re.GateCount(re.ResourceRY.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRot, 2, {"eps": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"eps": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [re.GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(re.ResourceControlled(re.ResourceRot(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(re.ResourceControlled(re.ResourceRot(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(re.ResourceControlled(re.ResourceRot(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == [expected_res]

    pow_data = (
        (1, [re.GateCount(op.resource_rep(), 1)]),
        (2, [re.GateCount(op.resource_rep(), 1)]),
        (5, [re.GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z) == expected_res


class TestResourceControlledPhaseShift:
    """Test ResourceControlledPhaseShift"""

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        op = re.ResourceControlledPhaseShift()
        assert op.resource_keys == {"eps"}

    def test_resources(self):
        """Test the resources method"""

        op = re.ResourceControlledPhaseShift()

        expected = [
            re.GateCount(re.CompressedResourceOp(re.ResourceCNOT, 2, {}), 2),
            re.GateCount(re.CompressedResourceOp(re.ResourceRZ, 1, {"eps": None}), 3),
        ]

        assert op.resource_decomp(**op.resource_params) == expected

    def test_resource_params(self):
        """Test the resource parameters"""

        op = re.ResourceControlledPhaseShift()
        assert op.resource_params == {
            "eps": None
        }  # pylint: disable=use-implicit-booleaness-not-comparison

    def test_resource_rep(self):
        """Test the compressed representation"""

        op = re.ResourceControlledPhaseShift()
        expected = re.CompressedResourceOp(re.ResourceControlledPhaseShift, 2, {"eps": None})

        assert op.resource_rep() == expected

    def test_resource_rep_from_op(self):
        """Test resource_rep_from_op method"""

        op = re.ResourceControlledPhaseShift()
        assert op.resource_rep_from_op() == re.ResourceControlledPhaseShift.resource_rep(
            **op.resource_params
        )

    def test_resources_from_rep(self):
        """Compute the resources from the compressed representation"""

        op = re.ResourceControlledPhaseShift()

        expected = [
            re.GateCount(re.CompressedResourceOp(re.ResourceCNOT, 2, {}), 2),
            re.GateCount(re.CompressedResourceOp(re.ResourceRZ, 1, {"eps": None}), 3),
        ]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_params = op_compressed_rep.params
        op_compressed_rep_type = op_compressed_rep.op_type

        assert op_compressed_rep_type.resource_decomp(**op_resource_params) == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint resources are correct."""

        op = re.ResourceControlledPhaseShift()

        assert op.default_adjoint_resource_decomp() == [
            re.GateCount(re.ResourceControlledPhaseShift.resource_rep(), 1)
        ]

    pow_data = ((1, [re.GateCount(re.ResourceControlledPhaseShift.resource_rep(), 1)]),)

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the adjoint resources are correct."""

        op = re.ResourceControlledPhaseShift
        assert op.default_pow_resource_decomp(z) == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(re.ResourceControlled(re.ResourcePhaseShift(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(re.ResourceControlled(re.ResourcePhaseShift(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(re.ResourceControlled(re.ResourcePhaseShift(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        op = re.ResourceControlledPhaseShift()

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == [expected_res]
