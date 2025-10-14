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

import pennylane.estimator as qre
from pennylane.estimator.ops import RY, RZ, Hadamard, Identity, S, T, X
from pennylane.estimator.ops.op_math.controlled_ops import (
    CCZ,
    CH,
    CNOT,
    CRX,
    CRY,
    CRZ,
    CSWAP,
    CY,
    CZ,
    ControlledPhaseShift,
    CRot,
    MultiControlledX,
    TemporaryAND,
    Toffoli,
)
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.exceptions import ResourcesUndefinedError

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments,too-many-positional-arguments


class TestCH:
    """Test the Resource CH operation"""

    op = CH(wires=[0, 1])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CH(wires=[0, 1, 2])

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
        num_ctrl_wires = 3
        num_zero_ctrl = 1

        expected_op = qre.Controlled(
            qre.Hadamard(),
            num_ctrl_wires=4,
            num_zero_ctrl=1,
        )
        expected_res = [GateCount(expected_op.resource_rep_from_op())]

        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl) == expected_res

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
    """Test the Resource CY operation"""

    op = CY(wires=[0, 1])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CY(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(CNOT.resource_rep(), 1),
            GateCount(S.resource_rep()),
            GateCount(qre.Adjoint.resource_rep(S.resource_rep())),
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
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.Y(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.Y(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.Y(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""

        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl) == [expected_res]

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
    """Test the Resource CZ operation"""

    op = CZ(wires=[0, 1])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CZ(wires=[0, 1, 2])

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

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.CCZ().resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.Z(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.Z(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl) == [expected_res]

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
    """Test the Resource CSWAP operation"""

    op = CSWAP(wires=[0, 1, 2])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 3 wires, got 2"):
            CSWAP(wires=[0, 1])

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

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.SWAP(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.SWAP(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.SWAP(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl) == [expected_res]

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
    """Test the Resource CCZ operation"""

    op = CCZ(wires=[0, 1, 2])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 3 wires, got 2"):
            CCZ(wires=[0, 1])

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

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.Z(), 3, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.Z(), 4, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.Z(), 5, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl) == [expected_res]

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

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CNOT(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method is not implemented"""
        with pytest.raises(ResourcesUndefinedError):
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

    ctrl_data = (
        (
            ["c1"],
            [1],
            [GateCount(Toffoli.resource_rep(), 1)],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [GateCount(MultiControlledX.resource_rep(3, 0), 1)],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [GateCount(MultiControlledX.resource_rep(4, 2), 1)],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_zero_ctrl = len([v for v in ctrl_values if not v])

        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl) == expected_res

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


class TestTemporaryAND:
    """Test the Resource TemporaryAND operation"""

    op = TemporaryAND(wires=[0, 1, 2])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 3 wires, got 2"):
            TemporaryAND(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = [GateCount(Toffoli.resource_rep(elbow="left"), 1)]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(TemporaryAND, 3, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(Hadamard.resource_rep()), GateCount(CZ.resource_rep())]
        assert self.op.adjoint_resource_decomp() == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [qre.GateCount(qre.MultiControlledX.resource_rep(3, 0), 1)],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [qre.GateCount(qre.MultiControlledX.resource_rep(4, 0), 1)],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [qre.GateCount(qre.MultiControlledX.resource_rep(5, 2), 1)],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_zero_ctrl = len([v for v in ctrl_values if not v])

        assert (
            self.op.controlled_resource_decomp(
                num_ctrl_wires, num_zero_ctrl, self.op.resource_params
            )
            == expected_res
        )


class TestToffoli:
    """Test the Resource Toffoli operation"""

    op = Toffoli(wires=[0, 1, 2])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 3 wires, got 2"):
            Toffoli(wires=[0, 1])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            Allocate(2),
            GateCount(CNOT.resource_rep(), 9),
            GateCount(Hadamard.resource_rep(), 3),
            GateCount(S.resource_rep()),
            GateCount(CZ.resource_rep()),
            GateCount(T.resource_rep(), 2),
            GateCount(qre.Adjoint.resource_rep(T.resource_rep()), 2),
            Deallocate(2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

        textbook_expected_resources = [
            GateCount(CNOT.resource_rep(), 6),
            GateCount(Hadamard.resource_rep(), 2),
            GateCount(T.resource_rep(), 4),
            GateCount(qre.Adjoint.resource_rep(T.resource_rep()), 3),
        ]
        assert (
            self.op.textbook_resource_decomp(**self.op.resource_params)
            == textbook_expected_resources
        )

    def test_resource_elbows(self):
        """Test that the resource_rep produces the correct compressed representation."""
        expected_rep = [
            GateCount(T.resource_rep(), 2),
            GateCount(qre.Adjoint.resource_rep(T.resource_rep()), 2),
            GateCount(CNOT.resource_rep(), 3),
            GateCount(qre.Adjoint.resource_rep(S.resource_rep())),
        ]
        assert self.op.resource_decomp(elbow="left") == expected_rep
        assert self.op.textbook_resource_decomp(elbow="left") == expected_rep

        expected_rep = [
            GateCount(Hadamard.resource_rep(), 1),
            GateCount(CZ.resource_rep(), 1),
        ]
        assert self.op.resource_decomp(elbow="right") == expected_rep
        assert self.op.textbook_resource_decomp(elbow="right") == expected_rep

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(Toffoli, 3, {"elbow": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"elbow": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]

        assert self.op.adjoint_resource_decomp(self.op.resource_params) == expected_res

        expected_elbows_res = [
            [GateCount(self.op.resource_rep(elbow="left"), 1)],
            [GateCount(self.op.resource_rep(elbow="right"), 1)],
        ]
        assert self.op.adjoint_resource_decomp({"elbow": "right"}) == expected_elbows_res[0]
        assert self.op.adjoint_resource_decomp({"elbow": "left"}) == expected_elbows_res[1]

    ctrl_data = (
        (
            ["c1"],
            [1],
            [qre.GateCount(qre.MultiControlledX.resource_rep(3, 0), 1)],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [qre.GateCount(qre.MultiControlledX.resource_rep(4, 0), 1)],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [qre.GateCount(qre.MultiControlledX.resource_rep(5, 2), 1)],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_zero_ctrl = len([v for v in ctrl_values if not v])

        assert (
            self.op.controlled_resource_decomp(
                num_ctrl_wires, num_zero_ctrl, self.op.resource_params
            )
            == expected_res
        )

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
        (8, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z, self.op.resource_params) == expected_res


class TestMultiControlledX:
    """Test the Resource MultiControlledX operation"""

    res_ops = (
        MultiControlledX(1, 0),
        MultiControlledX(2, 0),
        MultiControlledX(1, 1),
        MultiControlledX(2, 1),
    )

    res_params = (
        (1, 0),
        (2, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 3),
    )

    expected_resources = (
        [GateCount(CNOT.resource_rep(), 1)],
        [GateCount(Toffoli.resource_rep(), 1)],
        [
            GateCount(X.resource_rep(), 2),
            GateCount(CNOT.resource_rep(), 1),
        ],
        [
            GateCount(X.resource_rep(), 2),
            GateCount(Toffoli.resource_rep(), 1),
        ],
    )

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 2"):
            MultiControlledX(num_ctrl_wires=3, num_zero_ctrl=0, wires=[0, 1])

    def test_init_no_num_ctrl_wires(self):
        """Test that we can instantiate the operator without providing num_ctrl_wires"""
        op = qre.MultiControlledX(num_zero_ctrl=1, wires=[0, 1, 2, 3, 4])
        assert op.num_wires == 5
        assert op.resource_params == {"num_ctrl_wires": 4, "num_zero_ctrl": 1}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and num_ctrl_wires are both not provided"""
        with pytest.raises(ValueError, match="Must provide atleast one of"):
            qre.MultiControlledX(num_zero_ctrl=1)

    @staticmethod
    def _prep_params(num_control, num_zero_ctrl):
        return {
            "num_ctrl_wires": num_control,
            "num_zero_ctrl": num_zero_ctrl,
        }

    @pytest.mark.parametrize("params, expected_res", zip(res_params, expected_resources))
    def test_resources(self, params, expected_res):
        """Test that the resources method produces the expected resources."""
        op_resource_params = self._prep_params(*params)
        assert MultiControlledX.resource_decomp(**op_resource_params) == expected_res

    def test_resource_decomp_min_wires(self):
        """Test that the resource_decomp raises an error"""
        assert MultiControlledX.resource_decomp(0, 1) == []
        assert MultiControlledX.resource_decomp(0, 0) == [GateCount(X.resource_rep())]

    def test_resource_decomp_max_wires(self):
        """Test that the controlled resources raise an error"""
        assert MultiControlledX.resource_decomp(5, 2) == [
            GateCount(X.resource_rep(), 4),
            Allocate(3),
            GateCount(TemporaryAND.resource_rep(), 3),
            GateCount(qre.Adjoint.resource_rep(TemporaryAND.resource_rep()), 3),
            GateCount(Toffoli.resource_rep(), 1),
            Deallocate(3),
        ]

    @pytest.mark.parametrize("op, params", zip(res_ops, res_params))
    def test_resource_rep(self, op, params):
        """Test the resource_rep produces the correct compressed representation."""
        op_resource_params = self._prep_params(*params)
        num_wires = op_resource_params["num_ctrl_wires"] + 1
        expected_rep = CompressedResourceOp(MultiControlledX, num_wires, op_resource_params)
        assert op.resource_rep(**op.resource_params) == expected_rep

    @pytest.mark.parametrize("op, params", zip(res_ops, res_params))
    def test_resource_params(self, op, params):
        """Test that the resource_params are produced as expected."""
        expected_params = self._prep_params(*params)
        assert op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = MultiControlledX(5, 3)
        expected_res = [GateCount(op.resource_rep(**op.resource_params), 1)]

        assert op.adjoint_resource_decomp(op.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [GateCount(MultiControlledX.resource_rep(4, 2))],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                GateCount(MultiControlledX.resource_rep(5, 2)),
            ],
        ),
        (
            ["c1", "c2", "c3", "c4"],
            [1, 0, 0, 1],
            [
                GateCount(MultiControlledX.resource_rep(7, 4)),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        op = MultiControlledX(3, 2)
        num_ctrl_wires = len(ctrl_wires)
        num_zero_ctrl = len([v for v in ctrl_values if not v])

        assert (
            op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, op.resource_params)
            == expected_res
        )

    pow_data = (
        (1, [GateCount(MultiControlledX.resource_rep(5, 3), 1)]),
        (2, [GateCount(Identity.resource_rep())]),
        (5, [GateCount(MultiControlledX.resource_rep(5, 3), 1)]),
        (6, [GateCount(Identity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = MultiControlledX(5, 3)
        assert op.pow_resource_decomp(z, op.resource_params) == expected_res


class TestCRX:
    """Test the Resource CRX operation"""

    op = CRX(wires=[0, 1])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CRX(wires=[0, 1, 2])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"precision"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(CNOT.resource_rep(), 2),
            GateCount(RZ.resource_rep(), 2),
            GateCount(Hadamard.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CRX, 2, {"precision": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"precision": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp(self.op.resource_params) == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.RX(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.RX(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.RX(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(
            num_ctrl_wires, num_zero_ctrl, self.op.resource_params
        ) == [expected_res]

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(op.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""

        assert self.op.pow_resource_decomp(z, self.op.resource_params) == expected_res


class TestCRY:
    """Test the Resource CRY operation"""

    op = CRY(wires=[0, 1])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CRY(wires=[0, 1, 2])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"precision"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(CNOT.resource_rep(), 2),
            GateCount(RY.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CRY, 2, {"precision": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"precision": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp(self.op.resource_params) == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.RY(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.RY(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.RY(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(
            num_ctrl_wires, num_zero_ctrl, self.op.resource_params
        ) == [expected_res]

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(op.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z, self.op.resource_params) == expected_res


class TestCRZ:
    """Test the Resource CRZ operation"""

    op = CRZ(wires=[0, 1])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"precision"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(CNOT.resource_rep(), 2),
            GateCount(RZ.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CRZ, 2, {"precision": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"precision": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp(self.op.resource_params) == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.RZ(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.RZ(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.RZ(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(
            num_ctrl_wires, num_zero_ctrl, self.op.resource_params
        ) == [expected_res]

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(op.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z, self.op.resource_params) == expected_res


class TestCRot:
    """Test the Resource CRot operation"""

    op = CRot(wires=[0, 1])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CRZ(wires=[0, 1, 2])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        assert self.op.resource_keys == {"precision"}

    def test_resources(self):
        """Test that the resources method produces the expected resources."""
        expected_resources = [
            GateCount(CNOT.resource_rep(), 2),
            GateCount(RZ.resource_rep(), 3),
            GateCount(RY.resource_rep(), 2),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CRot, 2, {"precision": None})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {"precision": None}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp(self.op.resource_params) == expected_res

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(op.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        assert self.op.pow_resource_decomp(z, self.op.resource_params) == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.Rot(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.Rot(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.Rot(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        assert self.op.controlled_resource_decomp(
            num_ctrl_wires, num_zero_ctrl, self.op.resource_params
        ) == [expected_res]


class TestControlledPhaseShift:
    """Test Resource ControlledPhaseShift"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            ControlledPhaseShift(wires=[0, 1, 2])

    def test_resource_keys(self):
        """test that the resource keys are correct"""
        op = ControlledPhaseShift()
        assert op.resource_keys == {"precision"}

    def test_resources(self):
        """Test the resources method"""

        op = ControlledPhaseShift()

        expected = [
            GateCount(CompressedResourceOp(CNOT, 2, {}), 2),
            GateCount(CompressedResourceOp(RZ, 1, {"precision": None}), 3),
        ]

        assert op.resource_decomp(**op.resource_params) == expected

    def test_resource_params(self):
        """Test the resource parameters"""

        op = ControlledPhaseShift()
        assert op.resource_params == {
            "precision": None
        }  # pylint: disable=use-implicit-booleaness-not-comparison

    def test_resource_rep(self):
        """Test the compressed representation"""

        op = ControlledPhaseShift()
        expected = CompressedResourceOp(ControlledPhaseShift, 2, {"precision": None})

        assert op.resource_rep() == expected

    def test_resource_rep_from_op(self):
        """Test resource_rep_from_op method"""

        op = ControlledPhaseShift()
        assert op.resource_rep_from_op() == ControlledPhaseShift.resource_rep(**op.resource_params)

    def test_resources_from_rep(self):
        """Compute the resources from the compressed representation"""

        op = ControlledPhaseShift()

        expected = [
            GateCount(CompressedResourceOp(CNOT, 2, {}), 2),
            GateCount(CompressedResourceOp(RZ, 1, {"precision": None}), 3),
        ]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_params = op_compressed_rep.params
        op_compressed_rep_type = op_compressed_rep.op_type

        assert op_compressed_rep_type.resource_decomp(**op_resource_params) == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint resources are correct."""

        op = ControlledPhaseShift()

        assert op.adjoint_resource_decomp({"precision": None}) == [
            GateCount(ControlledPhaseShift.resource_rep(), 1)
        ]

    pow_data = ((1, [GateCount(ControlledPhaseShift.resource_rep(), 1)]),)

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the adjoint resources are correct."""

        op = ControlledPhaseShift
        assert op.pow_resource_decomp(z, {"precision": None}) == expected_res

    ctrl_data = (
        (
            1,
            0,
            GateCount(qre.Controlled(qre.PhaseShift(), 2, 0).resource_rep_from_op()),
        ),
        (
            2,
            0,
            GateCount(qre.Controlled(qre.PhaseShift(), 3, 0).resource_rep_from_op()),
        ),
        (
            3,
            2,
            GateCount(qre.Controlled(qre.PhaseShift(), 4, 2).resource_rep_from_op()),
        ),
    )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resources are as expected"""
        op = qre.ControlledPhaseShift()

        assert op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, op.resource_params) == [
            expected_res
        ]
