# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Unittests for is_commuting
"""
# pylint: disable=too-many-public-methods
import pytest

import pennylane as qp
import pennylane.numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.ops.functions.is_commuting import _check_mat_commutation, _get_target_name

control_base_map_data = [
    (qp.CNOT((0, 1)), "PauliX"),
    (qp.CZ((0, 1)), "PauliZ"),
    (qp.CY((0, 1)), "PauliY"),
    (qp.CSWAP(range(3)), "SWAP"),
    (qp.Toffoli(range(3)), "PauliX"),
    (qp.ControlledPhaseShift(1.234, (0, 1)), "PhaseShift"),
    (qp.CRX(1.23, range(2)), "RX"),
    (qp.CRY(1.34, range(2)), "RY"),
    (qp.CRZ(1.234, range(2)), "RZ"),
    (qp.CRot(1.2, 2.3, 3.4, range(2)), "Rot"),
    (qp.MultiControlledX(wires=range(4)), "PauliX"),
]


class TestGetTargetName:
    """Tests the _get_target_name helper function."""

    @pytest.mark.parametrize("op, target_name", control_base_map_data)
    def test_explicitly_specified_control_op(self, op, target_name):
        """Test getting the target name for operations explicitly specified in the map."""
        assert _get_target_name(op) == target_name

    @pytest.mark.parametrize("op", (qp.PauliX(0), qp.RX(1.2, 0), qp.IsingXX(0, range(2))))
    def test_Controlled_op(self, op):
        """Test it gets the base's name for a controlled op."""
        c_op = qp.ops.op_math.Controlled(op, control_wires=("a", "b"))
        assert _get_target_name(c_op) == op.name

    @pytest.mark.parametrize("op", (qp.PauliX(0), qp.RX(1.2, 0), qp.IsingXX(0, range(2))))
    def test_basic_op(self, op):
        """Test that for non-controlled gates, the helper simply returns the name"""
        assert _get_target_name(op) == op.name


class TestCheckMatCommutation:
    """Tests the _check_mat_commutation helper method."""

    def test_matrices_commute(self):
        """Test that if the operations commute, then the helper function returns True"""
        s0 = qp.S(0)
        t0 = qp.T(0)

        assert _check_mat_commutation(s0, t0)
        assert _check_mat_commutation(t0, s0)

    def test_matrices_dont_commute(self):
        """Check matrices don't commute for two simple ops."""
        x0 = qp.PauliX(0)
        z0 = qp.PauliZ(0)

        assert not _check_mat_commutation(x0, z0)
        assert not _check_mat_commutation(z0, x0)


class TestControlledOps:
    """Test how is_commuting integrates with Controlled operators."""

    def test_commuting_overlapping_targets(self):
        """Test commuting when targets commute and overlap wires."""
        op1 = qp.ops.op_math.Controlled(qp.PauliX(3), control_wires=(0, 1, 2))
        op2 = qp.ops.op_math.Controlled(qp.RX(1.2, 3), control_wires=(0, 1))
        assert qp.is_commuting(op1, op2)
        assert qp.is_commuting(op2, op1)

    def test_non_commuting_overlapping_targets(self):
        """Test not commuting when targets don't commute and overlap wires."""
        op1 = qp.ops.op_math.Controlled(qp.PauliZ(3), control_wires=(0, 1, 2))
        op2 = qp.ops.op_math.Controlled(qp.RX(1.2, 3), control_wires=(0, 1))
        assert not qp.is_commuting(op1, op2)
        assert not qp.is_commuting(op2, op1)

    def test_commuting_one_target_commutes_with_ctrl(self):
        """Test it is commuting if one target overlaps with the others control wires, and target
        commutes with control wires."""

        op1 = qp.ops.op_math.Controlled(qp.PauliZ(3), control_wires=0)
        op2 = qp.ops.op_math.Controlled(qp.PauliX(2), control_wires=3)
        assert qp.is_commuting(op1, op2)
        assert qp.is_commuting(op2, op1)

    def test_not_commuting_one_target_not_commute_with_ctrl(self):
        """Test it is not commuting if a target overlaps with control wires, and target
        does not commute with ctrl."""
        op1 = qp.ops.op_math.Controlled(qp.PauliX(3), control_wires=0)
        op2 = qp.ops.op_math.Controlled(qp.PauliZ(2), control_wires=3)
        assert not qp.is_commuting(op1, op2)
        assert not qp.is_commuting(op2, op1)


class TestCommutingFunction:
    """Commutation function tests."""

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1, 0]], False),
            ([[1, 0], [1, 0]], True),
            ([[0, 1], [2, 3]], True),
            ([[0, 1], [3, 1]], True),
        ],
    )
    def test_cnot(self, wires, res):
        """Commutation between two CNOTs."""
        commutation = qp.is_commuting(qp.CNOT(wires=wires[0]), qp.CNOT(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1, 2], [1, 0, 2]], True),
            ([[1, 2], [0, 1, 2]], True),
            ([[3, 2], [0, 1, 2]], True),
            ([[0, 1], [0, 1, 2]], False),
        ],
    )
    def test_cnot_toffoli(self, wires, res):
        """Commutation between CNOT and Toffoli"""
        commutation = qp.is_commuting(qp.CNOT(wires=wires[0]), qp.Toffoli(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1, 2], [1, 0]], True),
            ([[0, 1], [0, 1]], False),
            ([[0, 1], [2, 0]], True),
            ([[0, 1], [0, 2]], True),
        ],
    )
    def test_cnot_cz(self, wires, res):
        """Commutation between CNOT and CZ"""
        commutation = qp.is_commuting(qp.CNOT(wires=wires[0]), qp.CZ(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1, 2]], True),
            ([[0, 2], [0, 1, 2]], True),
            ([[0, 2], [0, 2, 1]], True),
        ],
    )
    def test_cz_mcz(self, wires, res):
        """Commutation between CZ and MCZ."""

        op1 = qp.ctrl(qp.PauliZ(wires=wires[1][-1]), control=wires[1][:-1])
        op2 = qp.CZ(wires=wires[0])

        assert qp.is_commuting(op1, op2) == res
        assert qp.is_commuting(op2, op1) == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0, 1, 2]], False),
            ([[1], [0, 1, 2]], False),
            ([[2], [0, 1, 2]], False),
        ],
    )
    def test_rx_mcz(self, wires, res):
        """Commutation between RX and MCZ"""

        op1 = qp.RX(0.1, wires=wires[0][0])
        op2 = qp.ctrl(qp.PauliZ(wires=wires[1][2]), control=wires[1][:-1])

        assert qp.is_commuting(op1, op2) == res
        assert qp.is_commuting(op2, op1) == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1, 2], [0, 1, 2]], True),
            ([[0, 2, 1], [0, 1, 2]], True),
            ([[1, 2, 0], [0, 2, 1]], True),
        ],
    )
    def test_mcz_mcz(self, wires, res):
        """Commutation between MCZ and MCZ."""

        op1 = qp.ctrl(qp.PauliZ(wires=wires[0][2]), control=wires[0][:-1])
        op2 = qp.ctrl(qp.PauliZ(wires=wires[1][2]), control=wires[1][:-1])

        assert qp.is_commuting(op1, op2) == res
        assert qp.is_commuting(op2, op1) == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1, 2]], False),
            ([[0, 2], [0, 1, 2]], False),
            ([[0, 2], [0, 2, 1]], False),
            ([[0, 3], [0, 2, 1]], True),
            ([[0, 3], [1, 2, 0]], True),
        ],
    )
    def test_cnot_mcz(self, wires, res):
        """Commutation between CNOT and MCZ."""

        op1 = qp.CNOT(wires=wires[0])
        op2 = qp.ctrl(qp.PauliZ(wires=wires[1][2]), control=wires[1][:-1])
        assert qp.is_commuting(op1, op2) == res
        assert qp.is_commuting(op2, op1) == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1], [0, 1]], True),
            ([[0], [0, 1]], False),
            ([[2], [0, 1]], True),
        ],
    )
    def test_x_cnot(self, wires, res):
        """Commutation between PauliX and CNOT."""
        op1 = qp.PauliX(wires=wires[0])
        op2 = qp.CNOT(wires=wires[1])
        assert qp.is_commuting(op1, op2) == res
        assert qp.is_commuting(op2, op1) == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1], [0, 1]], False),
            ([[0], [0, 1]], False),
            ([[2], [0, 1]], True),
        ],
    )
    def test_x_cy(self, wires, res):
        """Commutation between PauliX and CY."""
        commutation = qp.is_commuting(qp.PauliX(wires=wires[0]), qp.CY(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 2], [0, 1, 2]], False),
            ([[0, 1], [0, 1, 2]], False),
            ([[0, 3], [0, 1, 2]], True),
            ([[1, 2], [0, 1, 2]], False),
        ],
    )
    def test_cnot_cswap(self, wires, res):
        """Commutation between CNOT and CSWAP."""
        commutation = qp.is_commuting(qp.CNOT(wires=wires[0]), qp.CSWAP(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1, 2], [1, 2]], False),
        ],
    )
    def test_cswap_cnot(self, wires, res):
        """Commutation between CSWAP and CNOT."""
        commutation = qp.is_commuting(qp.CSWAP(wires=wires[0]), qp.CNOT(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1, 2], [2, 1, 0]], False),
        ],
    )
    def test_cswap_cswap(self, wires, res):
        """Commutation between CSWAP and CSWAP."""
        commutation = qp.is_commuting(qp.CSWAP(wires=wires[0]), qp.CSWAP(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1]], False),
        ],
    )
    def test_cnot_swap(self, wires, res):
        """Commutation between CNOT and SWAP."""
        commutation = qp.is_commuting(qp.CNOT(wires=wires[0]), qp.SWAP(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1]], False),
        ],
    )
    def test_swap_cnot(self, wires, res):
        """Commutation between SWAP and CNOT."""
        commutation = qp.is_commuting(qp.SWAP(wires=wires[0]), qp.CNOT(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 2], [0, 1, 2]], False),
            ([[0, 1], [0, 1, 2]], False),
            ([[0, 3], [0, 1, 2]], True),
        ],
    )
    def test_cz_cswap(self, wires, res):
        """Commutation between CZ and CSWAP."""
        commutation = qp.is_commuting(qp.CZ(wires=wires[0]), qp.CSWAP(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 2], [0, 1, 2, 3]], False),
            ([[0, 1], [0, 1, 2, 3]], False),
            ([[0, 3], [0, 1, 2, 3]], True),
        ],
    )
    def test_cnot_multicx(self, wires, res):
        """Commutation between CNOT and MultiControlledX."""
        commutation = qp.is_commuting(
            qp.CNOT(wires=wires[0]),
            qp.MultiControlledX(wires=wires[1], control_values=[1, 1, 1]),
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], True),
            ([[0, 1], [1]], True),
        ],
    )
    def test_cphase_z(self, wires, res):
        """Commutation between CPhase and PauliZ."""
        commutation = qp.is_commuting(qp.CPhase(0.2, wires=wires[0]), qp.PauliZ(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], True),
            ([[0, 1], [1]], True),
        ],
    )
    def test_cphase_phase(self, wires, res):
        """Commutation between CPhase and Phase."""
        commutation = qp.is_commuting(
            qp.CPhase(0.2, wires=wires[0]), qp.PhaseShift(0.1, wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], False),
            ([[0, 1], [1]], False),
        ],
    )
    def test_cphase_paulix(self, wires, res):
        """Commutation between CPhase and PauliX."""
        commutation = qp.is_commuting(qp.CPhase(0.2, wires=wires[0]), qp.PauliX(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], True),
            ([[0, 1], [1]], True),
        ],
    )
    def test_cphase_zero_paulix(self, wires, res):
        """Commutation between CPhase(0.0) and PauliX."""
        commutation = qp.is_commuting(qp.CPhase(0.0, wires=wires[0]), qp.PauliX(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], True),
            ([[0, 1], [1]], False),
        ],
    )
    def test_crx_pauliz(self, wires, res):
        """Commutation between CRX(0.1) and PauliZ."""
        commutation = qp.is_commuting(qp.CRX(0.1, wires=wires[0]), qp.PauliZ(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], True),
            ([[0, 1], [1]], True),
        ],
    )
    def test_crx_zero_pauliz(self, wires, res):
        """Commutation between CRX(0.0) and PauliZ."""
        commutation = qp.is_commuting(qp.CRX(0.0, wires=wires[0]), qp.PauliZ(wires=wires[1]))
        assert commutation == res
        commutation = qp.is_commuting(qp.PauliZ(wires=wires[1]), qp.CRX(0.0, wires=wires[0]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], False),
            ([[0, 1], [1]], False),
        ],
    )
    def test_crz_paulix(self, wires, res):
        """Commutation between CRZ(0.1) and PauliX."""
        commutation = qp.is_commuting(qp.CRZ(0.1, wires=wires[0]), qp.PauliX(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], True),
            ([[0, 1], [1]], True),
        ],
    )
    def test_crz_zero_paulix(self, wires, res):
        """Commutation between CRZ(0.0) and PauliX."""
        commutation = qp.is_commuting(qp.CRZ(0.0, wires=wires[0]), qp.PauliX(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], False),
            ([[0, 1], [1]], False),
        ],
    )
    def test_cry_hadamard(self, wires, res):
        """Commutation between CRY(0.1) and Hadamard."""
        commutation = qp.is_commuting(qp.CRY(0.1, wires=wires[0]), qp.Hadamard(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], True),
            ([[0, 1], [1]], True),
        ],
    )
    def test_cry_zero_hadamard(self, wires, res):
        """Commutation between CRY(0.0) and Hadamard."""
        commutation = qp.is_commuting(qp.CRY(0.0, wires=wires[0]), qp.Hadamard(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_rot_x_simplified(self, wires, res):
        """Commutation between Rot(np.pi / 2, 0.1, -np.pi / 2) and PauliX."""
        commutation = qp.is_commuting(
            qp.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=wires[0]), qp.PauliX(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_rot_y_simplified(self, wires, res):
        """Commutation between Rot(0, 0.1, 0) and PauliY."""
        commutation = qp.is_commuting(
            qp.Rot(0, 0.1, 0, wires=wires[0]), qp.PauliY(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_rot_z_simplified(self, wires, res):
        """Commutation between Rot(0.1, 0.0, 0.2) and PauliZ."""
        commutation = qp.is_commuting(
            qp.Rot(0.1, 0, 0.2, wires=wires[0]), qp.PauliZ(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_rot_hadamard_simplified(self, wires, res):
        """Commutation between Rot(np.pi, np.pi / 2, 0) and Hadamard."""
        commutation = qp.is_commuting(
            qp.Rot(np.pi, np.pi / 2, 0, wires=wires[0]), qp.Hadamard(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], False),
            ([[0], [1]], True),
        ],
    )
    def test_rot_z(self, wires, res):
        """Commutation between Rot(0.1, 0.2, 0.3) and PauliZ."""
        commutation = qp.is_commuting(
            qp.Rot(0.1, 0.2, 0.3, wires=wires[0]), qp.PauliZ(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1]], True),
            ([[0, 1], [0]], False),
        ],
    )
    def test_crot_x_simplified(self, wires, res):
        """Commutation between CRot(np.pi / 2, 0.1, -np.pi / 2) and PauliX."""
        commutation = qp.is_commuting(
            qp.CRot(np.pi / 2, 0.1, -np.pi / 2, wires=wires[0]), qp.PauliX(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1]], True),
            ([[0, 1], [0]], False),
        ],
    )
    def test_crot_y_simplified(self, wires, res):
        """Commutation between CRot(0, 0.1, 0) and PauliY."""
        commutation = qp.is_commuting(
            qp.CRot(0, 0.1, 0, wires=wires[0]), qp.PauliY(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1]], True),
            ([[0, 1], [0]], True),
        ],
    )
    def test_crot_z_simplified(self, wires, res):
        """Commutation between CRot(0.1, 0, 0.2) and PauliZ."""
        commutation = qp.is_commuting(
            qp.CRot(0.1, 0, 0.2, wires=wires[0]), qp.PauliZ(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1]], True),
            ([[0, 1], [0]], False),
        ],
    )
    def test_crot_hadamard_simplified(self, wires, res):
        """Commutation between CRot(np.pi, np.pi / 2, 0) and Hadamard."""
        op1 = qp.CRot(np.pi, np.pi / 2, 0, wires=wires[0])
        op2 = qp.Hadamard(wires=wires[1])
        assert qp.is_commuting(op1, op2) == res
        assert qp.is_commuting(op2, op1) == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1]], False),
            ([[0, 1], [0]], True),
        ],
    )
    def test_crot_z(self, wires, res):
        """Commutation between CRot(0.1, 0.2, 0.3) and PauliZ."""
        commutation = qp.is_commuting(
            qp.CRot(0.1, 0.2, 0.3, wires=wires[0]), qp.PauliZ(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_u2_y_simplified(self, wires, res):
        """Commutation between U2(2*np.pi, -2*np.pi) and PauliY."""
        commutation = qp.is_commuting(
            qp.U2(2 * np.pi, -2 * np.pi, wires=wires[0]), qp.PauliY(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_u2_x_simplified(self, wires, res):
        """Commutation between U2(np.pi/2, -np.pi/2) and PauliX."""
        commutation = qp.is_commuting(
            qp.U2(np.pi / 2, -np.pi / 2, wires=wires[0]), qp.PauliX(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], False),
            ([[0], [1]], True),
        ],
    )
    def test_u2_u2(self, wires, res):
        """Commutation between U2(0.1, 0.2) and U2(0.3, 0.1)."""
        commutation = qp.is_commuting(
            qp.U2(0.1, 0.2, wires=wires[0]), qp.U2(0.3, 0.1, wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], False),
            ([[0, 1], [1]], False),
        ],
    )
    def test_crot_u2(self, wires, res):
        """Commutation between CRot(0.1, 0.2, 0.3) and U2(0.4, 0.5)."""
        commutation = qp.is_commuting(
            qp.CRot(0.1, 0.2, 0.3, wires=wires[0]), qp.U2(0.4, 0.5, wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0]], False),
            ([[0, 1], [1]], False),
        ],
    )
    def test_u2_crot(self, wires, res):
        """Commutation between U2(0.1, 0.2) and CRot(0.3, 0.4, 0.5)."""
        commutation = qp.is_commuting(
            qp.U2(0.1, 0.2, wires=wires[1]), qp.CRot(0.3, 0.4, 0.5, wires=wires[0])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1]], False),
            ([[0, 1], [1, 0]], False),
            ([[0, 2], [0, 1]], True),
            ([[0, 2], [1, 2]], False),
        ],
    )
    def test_crot_crot(self, wires, res):
        """Commutation between CRot(0.1, 0.2, 0.3) and CRot(0.3, 0.4, 0.5)."""
        commutation = qp.is_commuting(
            qp.CRot(0.1, 0.2, 0.3, wires=wires[1]), qp.CRot(0.3, 0.4, 0.5, wires=wires[0])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_u3_simplified_z(self, wires, res):
        """Commutation between U3(0.0, 0.1, 0.0) and PauliZ."""
        commutation = qp.is_commuting(
            qp.U3(0.0, 0.1, 0.0, wires=wires[1]), qp.PauliZ(wires=wires[0])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_u3_simplified_y(self, wires, res):
        """Commutation between U3(0.1, 0.0, 0.0) and PauliY."""
        commutation = qp.is_commuting(
            qp.U3(0.1, 0.0, 0.0, wires=wires[1]), qp.PauliY(wires=wires[0])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], True),
            ([[0], [1]], True),
        ],
    )
    def test_u3_simplified_x(self, wires, res):
        """Commutation between U3(0.1, -np.pi/2, np.pi/2) and PauliX."""
        commutation = qp.is_commuting(
            qp.U3(0.1, -np.pi / 2, np.pi / 2, wires=wires[0]), qp.PauliX(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], False),
            ([[0], [1]], True),
        ],
    )
    def test_u3_rot(self, wires, res):
        """Commutation between U3(0.1, 0.2, 0.3) and Rot(0.3, 0.2, 0.1)."""
        commutation = qp.is_commuting(
            qp.U3(0.1, 0.2, 0.3, wires=wires[0]), qp.Rot(0.3, 0.2, 0.1, wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], False),
            ([[0], [1]], True),
        ],
    )
    def test_u3_identity_barrier(self, wires, res):
        """Commutation between U3(0.0, 0.0, 0.0) and Barrier."""
        commutation = qp.is_commuting(
            qp.U3(0.0, 0.0, 0.0, wires=wires[0]), qp.Barrier(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0]], False),
            ([[0], [1]], True),
        ],
    )
    def test_barrier_u3_identity(self, wires, res):
        """Commutation between Barrier and U3(0.0, 0.0, 0.0)."""
        commutation = qp.is_commuting(
            qp.Barrier(wires=wires[1]), qp.U3(0.0, 0.0, 0.0, wires=wires[0])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "pauli_word_1,pauli_word_2,commute_status",
        [
            (qp.Identity(0), qp.PauliZ(0), True),
            (qp.PauliY(0), qp.PauliZ(0), False),
            (qp.PauliX(0), qp.PauliX(1), True),
            (qp.PauliY("x"), qp.PauliX("y"), True),
            (
                qp.prod(qp.PauliZ("a"), qp.PauliY("b"), qp.PauliZ("d")),
                qp.prod(qp.PauliX("a"), qp.PauliZ("c"), qp.PauliY("d")),
                True,
            ),
            (
                qp.prod(qp.PauliX("a"), qp.PauliY("b"), qp.PauliZ("d")),
                qp.prod(qp.PauliX("a"), qp.PauliZ("c"), qp.PauliY("d")),
                False,
            ),
            (
                qp.sum(qp.PauliZ("a"), qp.PauliY("b"), qp.PauliZ("d")),
                qp.sum(qp.PauliX("a"), qp.PauliZ("c"), qp.PauliY("d")),
                False,
            ),
            (
                qp.sum(qp.PauliZ("a"), qp.PauliY("a"), qp.PauliZ("b")),
                qp.sum(qp.PauliX("c"), qp.PauliZ("c"), qp.PauliY("d")),
                True,
            ),
            (
                qp.sum(qp.PauliZ("a"), qp.PauliY("b"), qp.PauliZ("d")),
                qp.sum(qp.PauliZ("a"), qp.PauliY("c"), qp.PauliZ("d")),
                True,
            ),
        ],
    )
    def test_pauli_words(self, pauli_word_1, pauli_word_2, commute_status):
        """Test that (non)-commuting Pauli words are correctly identified."""
        do_they_commute = qp.is_commuting(pauli_word_1, pauli_word_2)
        assert do_they_commute == commute_status

    @pytest.mark.parametrize(
        "pauli_word_1,pauli_word_2",
        [
            (
                qp.prod(qp.PauliX(0), qp.Hadamard(1), qp.Identity(2)),
                qp.sum(qp.PauliX(0), qp.PauliY(2)),
            ),
            (qp.PauliX(2), qp.sum(qp.Hadamard(1), qp.prod(qp.PauliX(1), qp.Identity(2)))),
            (qp.prod(qp.PauliX(1), qp.PauliY(2)), qp.s_prod(0.5, qp.Hadamard(1))),
        ],
    )
    def test_non_pauli_word_ops_not_supported(self, pauli_word_1, pauli_word_2):
        """Ensure invalid inputs are handled properly when determining commutativity."""
        with pytest.raises(QuantumFunctionError):
            qp.is_commuting(pauli_word_1, pauli_word_2)

    def test_operation_1_not_supported(self):
        """Test that giving a non supported operation raises an error."""
        rho = np.zeros((2**1, 2**1), dtype=np.complex128)
        rho[0, 0] = 1
        with pytest.raises(
            QuantumFunctionError,
            match="Operation QubitDensityMatrix not supported.",
        ):
            qp.is_commuting(qp.QubitDensityMatrix(rho, wires=[0]), qp.PauliX(wires=0))

    def test_operation_2_not_supported(self):
        """Test that giving a non supported operation raises an error."""

        with pytest.raises(QuantumFunctionError, match="Operation PauliRot not supported."):
            qp.is_commuting(qp.PauliX(wires=0), qp.PauliRot(1, "X", wires=0))

    @pytest.mark.parametrize(
        "op, name",
        [
            (qp.exp(qp.PauliX(0), 1.2), "Exp"),
        ],
    )
    def test_composite_arithmetic_ops_not_supported(self, op, name):
        """Test that giving a non supported operation raises an error."""

        with pytest.raises(QuantumFunctionError, match=f"Operation {name} not supported."):
            qp.is_commuting(qp.PauliX(wires=0), op)

    def test_non_commuting(self):
        """Test the function with an operator from the non-commuting list."""

        res = qp.is_commuting(qp.PauliX(wires=0), qp.QFT(wires=[1, 0]))
        assert res is False
