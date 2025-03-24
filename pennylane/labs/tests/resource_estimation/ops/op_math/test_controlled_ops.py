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
Tests for controlled resource operators.
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments,too-many-positional-arguments


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
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCH, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceHadamard, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceHadamard, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceHadamard, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {re.ResourceIdentity.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceCY:
    """Test the ResourceCY operation"""

    op = re.ResourceCY(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceS.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 1,
            re.ResourceAdjoint.resource_rep(re.ResourceS, {}): 1,
        }
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCY, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceY, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceY, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceY, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {re.ResourceIdentity.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceCZ:
    """Test the ResourceCZ operation"""

    op = re.ResourceCZ(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 1,
        }
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCZ, {})
        assert self.op.resource_rep() == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceCCZ.resource_rep(): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceZ, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceZ, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {re.ResourceIdentity.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceCSWAP:
    """Test the ResourceCSWAP operation"""

    op = re.ResourceCSWAP(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = {
            re.ResourceToffoli.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCSWAP, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceSWAP, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceSWAP, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceSWAP, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {re.ResourceIdentity.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceCCZ:
    """Test the ResourceCZZ operation"""

    op = re.ResourceCCZ(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceToffoli.resource_rep(): 1,
        }
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCCZ, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceZ, {}, 3, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceZ, {}, 4, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceZ, {}, 5, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {re.ResourceIdentity.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


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
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceToffoli.resource_rep(): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceMultiControlledX.resource_rep(3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceMultiControlledX.resource_rep(4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {re.ResourceIdentity.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
        (8, {re.ResourceIdentity.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceToffoli:
    """Test the ResourceToffoli operation"""

    op = re.ResourceToffoli(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceS.resource_rep(): 1,
            re.ResourceT.resource_rep(): 2,
            re.ResourceAdjoint.resource_rep(re.ResourceT, {}): 2,
            re.ResourceCZ.resource_rep(): 1,
            re.ResourceCNOT.resource_rep(): 9,
            re.ResourceHadamard.resource_rep(): 3,
        }
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceToffoli, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceMultiControlledX.resource_rep(3, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceMultiControlledX.resource_rep(4, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceMultiControlledX.resource_rep(5, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {re.ResourceIdentity.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
        (8, {re.ResourceIdentity.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceMultiControlledX:
    """Test the ResourceMultiControlledX operation"""

    res_ops = (
        re.ResourceMultiControlledX(wires=[0, "t"], control_values=[1]),
        re.ResourceMultiControlledX(wires=[0, 1, "t"], control_values=[1, 1]),
        re.ResourceMultiControlledX(wires=[0, 1, 2, "t"], control_values=[1, 1, 1]),
        re.ResourceMultiControlledX(wires=[0, 1, 2, 3, 4, "t"], control_values=[1, 1, 1, 1, 1]),
        re.ResourceMultiControlledX(wires=[0, "t"], control_values=[0], work_wires=["w1"]),
        re.ResourceMultiControlledX(
            wires=[0, 1, "t"], control_values=[1, 0], work_wires=["w1", "w2"]
        ),
        re.ResourceMultiControlledX(wires=[0, 1, 2, "t"], control_values=[0, 0, 1]),
        re.ResourceMultiControlledX(
            wires=[0, 1, 2, 3, 4, "t"],
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
        assert op.resource_rep(**op.resource_params) == expected_rep

    @pytest.mark.parametrize("op, params", zip(res_ops, res_params))
    def test_resource_params(self, op, params):
        """Test that the resource_params are produced as expected."""
        expected_params = self._prep_params(*params)
        assert op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = re.ResourceMultiControlledX(
            wires=[0, 1, 2, 3, 4, "t"],
            control_values=[1, 0, 0, 1, 0],
            work_wires=["w1"],
        )

        expected_res = {op.resource_rep(**op.resource_params): 1}
        op2 = re.ResourceAdjoint(op)

        assert op.adjoint_resource_decomp(**op.resource_params) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceToffoli.resource_rep(): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["work1"],
            {
                re.ResourceCNOT.resource_rep(): 2,
                re.ResourceToffoli.resource_rep(): 1,
            },
        ),
        (
            ["c1", "c2", "c3", "c4"],
            [1, 0, 0, 1],
            ["work1", "work2"],
            {
                re.ResourceX.resource_rep(): 4,
                re.ResourceCNOT.resource_rep(): 69,
            },
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        op = re.ResourceMultiControlledX(wires=[0, "t"], control_values=[1])

        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            op.controlled_resource_decomp(
                num_ctrl_wires, num_ctrl_values, num_work_wires, **op.resource_params
            )
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {re.ResourceMultiControlledX.resource_rep(5, 3, 1): 1}),
        (2, {}),
        (5, {re.ResourceMultiControlledX.resource_rep(5, 3, 1): 1}),
        (6, {}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourceMultiControlledX(
            wires=[0, 1, 2, 3, 4, "t"],
            control_values=[1, 0, 0, 1, 0],
            work_wires=["w1"],
        )

        op2 = re.ResourcePow(op, z)

        assert op.pow_resource_decomp(z, **op.resource_params) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


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
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRX, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceRX, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceRX, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceRX, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {op.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceCRY:
    """Test the ResourceCRY operation"""

    op = re.ResourceCRY(phi=1.23, wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceRY.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRY, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceRY, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceRY, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceRY, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {op.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


class TestResourceCRZ:
    """Test the ResourceCRZ operation"""

    op = re.ResourceCRZ(phi=1.23, wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = {
            re.ResourceRZ.resource_rep(): 2,
            re.ResourceCNOT.resource_rep(): 2,
        }
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRZ, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceRZ, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {op.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


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
        assert self.op.resources(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = re.CompressedResourceOp(re.ResourceCRot, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = {self.op.resource_rep(): 1}
        op2 = re.ResourceAdjoint(self.op)

        assert self.op.adjoint_resource_decomp() == expected_res
        assert op2.resources(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourceRot, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourceRot, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourceRot, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, work_wires, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op2 = re.ResourceControlled(
            self.op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            self.op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res

    pow_data = (
        (1, {op.resource_rep(): 1}),
        (2, {op.resource_rep(): 1}),
        (5, {op.resource_rep(): 1}),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op2 = re.ResourcePow(self.op, z)

        assert self.op.pow_resource_decomp(z) == expected_res
        assert op2.resources(**op2.resource_params) == expected_res


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

        assert op.resources(**op.resource_params) == expected

    @pytest.mark.parametrize("phi, wires", params)
    def test_resource_params(self, phi, wires):
        """Test the resource parameters"""

        op = re.ResourceControlledPhaseShift(phi, wires)
        assert op.resource_params == {}  # pylint: disable=use-implicit-booleaness-not-comparison

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
            **op.resource_params
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

    @pytest.mark.parametrize("phi, wires", params)
    def test_adjoint_decomp(self, phi, wires):
        """Test that the adjoint resources are correct."""

        op = re.ResourceControlledPhaseShift(phi, wires)
        adjoint = re.ResourceAdjoint(op)

        assert re.get_resources(op) == re.get_resources(adjoint)

    @pytest.mark.parametrize("phi, wires", params)
    def test_pow_decomp(self, phi, wires):
        """Test that the adjoint resources are correct."""

        op = re.ResourceControlledPhaseShift(phi, wires)
        pow = re.ResourcePow(op, 2)

        assert re.get_resources(op) == re.get_resources(pow)

    ctrl_data = (
        (
            ["c1"],
            [1],
            [],
            {re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 2, 0, 0): 1},
        ),
        (
            ["c1", "c2"],
            [1, 1],
            ["w1"],
            {re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 3, 0, 1): 1},
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            ["w1", "w2"],
            {re.ResourceControlled.resource_rep(re.ResourcePhaseShift, {}, 4, 2, 2): 1},
        ),
    )

    @pytest.mark.parametrize("phi, wires", params)
    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, work_wires, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(
        self, phi, wires, ctrl_wires, ctrl_values, work_wires, expected_res
    ):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])
        num_work_wires = len(work_wires)

        op = re.ResourceControlledPhaseShift(phi, wires)
        op2 = re.ResourceControlled(
            op, control_wires=ctrl_wires, control_values=ctrl_values, work_wires=work_wires
        )

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values, num_work_wires)
            == expected_res
        )
        assert op2.resources(**op2.resource_params) == expected_res
