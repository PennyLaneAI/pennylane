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
import pytest
import pennylane.numpy as np
import pennylane as qml

from pennylane.ops.functions.is_commuting import (
    _get_target_name,
    _check_mat_commutation,
    is_commuting,
)

control_base_map_data = [
    (qml.CNOT((0, 1)), "PauliX"),
    (qml.CZ((0, 1)), "PauliZ"),
    (qml.CY((0, 1)), "PauliY"),
    (qml.CSWAP(range(3)), "SWAP"),
    (qml.Toffoli(range(3)), "PauliX"),
    (qml.ControlledPhaseShift(1.234, (0, 1)), "PhaseShift"),
    (qml.CRX(1.23, range(2)), "RX"),
    (qml.CRY(1.34, range(2)), "RY"),
    (qml.CRZ(1.234, range(2)), "RZ"),
    (qml.CRot(1.2, 2.3, 3.4, range(2)), "Rot"),
    (qml.MultiControlledX(wires=range(4)), "PauliX"),
]


class TestGetTargetName:
    """Tests the _get_target_name helper function."""

    @pytest.mark.parametrize("op, target_name", control_base_map_data)
    def test_explicitly_specified_control_op(self, op, target_name):
        """Test getting the target name for operations explicitly specified in the map."""
        assert _get_target_name(op) == target_name

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.RX(1.2, 0), qml.IsingXX(0, range(2))))
    def test_Controlled_op(self, op):
        """Test it gets the base's name for a controlled op."""
        c_op = qml.ops.op_math.Controlled(op, control_wires=("a", "b"))
        assert _get_target_name(c_op) == op.name

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.RX(1.2, 0), qml.IsingXX(0, range(2))))
    def test_basic_op(self, op):
        """Test that for non-controlled gates, the helper simply returns the name"""
        assert _get_target_name(op) == op.name


class TestCheckMatCommutation:
    """Tests the _check_mat_commutation helper method."""

    def test_matrices_commute(self):
        """Test that if the operations commute, then the helper function returns True"""
        op1 = qml.S(0)
        op2 = qml.T(0)

        assert _check_mat_commutation(op1, op2)

    def test_matrices_dont_commute(self):
        """Check matrices don't commute for two simple ops."""
        op1 = qml.PauliX(0)
        op2 = qml.PauliZ(0)

        assert not _check_mat_commutation(op1, op2)


class TestControlledOps:
    """Test how is_commuting integrates with Controlled operators."""

    def test_commuting_overlapping_targets(self):
        """Test commuting when targets commute and overlap wires."""
        op1 = qml.ops.op_math.Controlled(qml.PauliX(3), control_wires=(0, 1, 2))
        op2 = qml.ops.op_math.Controlled(qml.RX(1.2, 3), control_wires=(0, 1))
        assert qml.is_commuting(op1, op2)

    def test_non_commuting_overlapping_targets(self):
        """Test not commuting when targets don't commute and overlap wires."""
        op1 = qml.ops.op_math.Controlled(qml.PauliZ(3), control_wires=(0, 1, 2))
        op2 = qml.ops.op_math.Controlled(qml.RX(1.2, 3), control_wires=(0, 1))
        assert not qml.is_commuting(op1, op2)

    def test_commuting_one_target_commutes_with_ctrl(self):
        """Test it is commuting if one target overlaps with the others control wires, and target
        commutes with control wires."""

        op1 = qml.ops.op_math.Controlled(qml.PauliZ(3), control_wires=0)
        op2 = qml.ops.op_math.Controlled(qml.PauliX(2), control_wires=3)
        assert qml.is_commuting(op1, op2)
        assert qml.is_commuting(op2, op1)

    def test_not_commuting_one_target_not_commute_with_ctrl(self):
        """Test it is not commuting if a target overlaps with control wires, and target
        does not commute with ctrl."""
        op1 = qml.ops.op_math.Controlled(qml.PauliX(3), control_wires=0)
        op2 = qml.ops.op_math.Controlled(qml.PauliZ(2), control_wires=3)
        assert not qml.is_commuting(op1, op2)
        assert not qml.is_commuting(op2, op1)


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
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.CNOT(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.Toffoli(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.CZ(wires=wires[1]))
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

        def z():
            qml.PauliZ(wires=wires[1][2])

        commutation = qml.is_commuting(qml.CZ(wires=wires[0]), qml.ctrl(z, control=wires[1][:-1])())
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1, 2]], True),
            ([[0, 2], [0, 1, 2]], True),
            ([[0, 2], [0, 2, 1]], True),
        ],
    )
    def test_mcz_cz(self, wires, res):
        """Commutation between MCZ and CZ"""

        def z():
            qml.PauliZ(wires=wires[1][2])

        commutation = qml.is_commuting(qml.ctrl(z, control=wires[1][:-1])(), qml.CZ(wires=wires[0]))
        assert commutation == res

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

        def z():
            qml.PauliZ(wires=wires[1][2])

        commutation = qml.is_commuting(
            qml.RX(0.1, wires=wires[0][0]), qml.ctrl(z, control=wires[1][:-1])()
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0], [0, 1, 2]], True),
            ([[1], [0, 1, 2]], True),
            ([[2], [0, 1, 2]], True),
        ],
    )
    def test_mcz_rx(self, wires, res):
        """Commutation between MCZ and RZ"""

        def z():
            qml.PauliZ(wires=wires[1][2])

        commutation = qml.is_commuting(
            qml.ctrl(z, control=wires[1][:-1])(), qml.RZ(0.1, wires=wires[0][0])
        )
        assert commutation == res

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

        def z_1():
            qml.PauliZ(wires=wires[0][2])

        def z_2():
            qml.PauliZ(wires=wires[1][2])

        commutation = qml.is_commuting(
            qml.ctrl(z_1, control=wires[0][:-1])(),
            qml.ctrl(z_2, control=wires[1][:-1])(),
        )
        assert commutation == res

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

        def z():
            qml.PauliZ(wires=wires[1][2])

        commutation = qml.is_commuting(
            qml.CNOT(wires=wires[0]), qml.ctrl(z, control=wires[1][:-1])()
        )
        assert commutation == res

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
        commutation = qml.is_commuting(qml.PauliX(wires=wires[0]), qml.CNOT(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[1], [0, 1]], True),
            ([[0], [0, 1]], False),
            ([[2], [0, 1]], True),
        ],
    )
    def test_cnot_x(self, wires, res):
        """Commutation between CNOT and PauliX."""
        commutation = qml.is_commuting(qml.CNOT(wires=wires[1]), qml.PauliX(wires=wires[0]))
        assert commutation == res

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
        commutation = qml.is_commuting(qml.PauliX(wires=wires[0]), qml.CY(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.CSWAP(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1, 2], [1, 2]], False),
        ],
    )
    def test_cswap_cnot(self, wires, res):
        """Commutation between CSWAP and CNOT."""
        commutation = qml.is_commuting(qml.CSWAP(wires=wires[0]), qml.CNOT(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1, 2], [2, 1, 0]], False),
        ],
    )
    def test_cswap_cswap(self, wires, res):
        """Commutation between CSWAP and CSWAP."""
        commutation = qml.is_commuting(qml.CSWAP(wires=wires[0]), qml.CSWAP(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1]], False),
        ],
    )
    def test_cnot_swap(self, wires, res):
        """Commutation between CNOT and SWAP."""
        commutation = qml.is_commuting(qml.CNOT(wires=wires[0]), qml.SWAP(wires=wires[1]))
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [0, 1]], False),
        ],
    )
    def test_swap_cnot(self, wires, res):
        """Commutation between SWAP and CNOT."""
        commutation = qml.is_commuting(qml.SWAP(wires=wires[0]), qml.CNOT(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CZ(wires=wires[0]), qml.CSWAP(wires=wires[1]))
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
        commutation = qml.is_commuting(
            qml.CNOT(wires=wires[0]),
            qml.MultiControlledX(wires=wires[1], control_values="111"),
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
        commutation = qml.is_commuting(qml.CPhase(0.2, wires=wires[0]), qml.PauliZ(wires=wires[1]))
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
        commutation = qml.is_commuting(
            qml.CPhase(0.2, wires=wires[0]), qml.PhaseShift(0.1, wires=wires[1])
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
        commutation = qml.is_commuting(qml.CPhase(0.2, wires=wires[0]), qml.PauliX(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CPhase(0.0, wires=wires[0]), qml.PauliX(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CRX(0.1, wires=wires[0]), qml.PauliZ(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CRX(0.0, wires=wires[0]), qml.PauliZ(wires=wires[1]))
        assert commutation == res
        commutation = qml.is_commuting(qml.PauliZ(wires=wires[1]), qml.CRX(0.0, wires=wires[0]))
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
        commutation = qml.is_commuting(qml.CRZ(0.1, wires=wires[0]), qml.PauliX(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CRZ(0.0, wires=wires[0]), qml.PauliX(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CRY(0.1, wires=wires[0]), qml.Hadamard(wires=wires[1]))
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
        commutation = qml.is_commuting(qml.CRY(0.0, wires=wires[0]), qml.Hadamard(wires=wires[1]))
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
        commutation = qml.is_commuting(
            qml.Rot(np.pi / 2, 0.1, -np.pi / 2, wires=wires[0]), qml.PauliX(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.Rot(0, 0.1, 0, wires=wires[0]), qml.PauliY(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.Rot(0.1, 0, 0.2, wires=wires[0]), qml.PauliZ(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.Rot(np.pi, np.pi / 2, 0, wires=wires[0]), qml.Hadamard(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.Rot(0.1, 0.2, 0.3, wires=wires[0]), qml.PauliZ(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.CRot(np.pi / 2, 0.1, -np.pi / 2, wires=wires[0]), qml.PauliX(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.CRot(0, 0.1, 0, wires=wires[0]), qml.PauliY(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.CRot(0.1, 0, 0.2, wires=wires[0]), qml.PauliZ(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.CRot(np.pi, np.pi / 2, 0, wires=wires[0]), qml.Hadamard(wires=wires[1])
        )
        assert commutation == res

    @pytest.mark.parametrize(
        "wires,res",
        [
            ([[0, 1], [1]], False),
            ([[0, 1], [0]], True),
        ],
    )
    def test_crot_z(self, wires, res):
        """Commutation between CRot(0.1, 0.2, 0.3) and PauliZ."""
        commutation = qml.is_commuting(
            qml.CRot(0.1, 0.2, 0.3, wires=wires[0]), qml.PauliZ(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.U2(2 * np.pi, -2 * np.pi, wires=wires[0]), qml.PauliY(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.U2(np.pi / 2, -np.pi / 2, wires=wires[0]), qml.PauliX(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.U2(0.1, 0.2, wires=wires[0]), qml.U2(0.3, 0.1, wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.CRot(0.1, 0.2, 0.3, wires=wires[0]), qml.U2(0.4, 0.5, wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.U2(0.1, 0.2, wires=wires[1]), qml.CRot(0.3, 0.4, 0.5, wires=wires[0])
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
        commutation = qml.is_commuting(
            qml.CRot(0.1, 0.2, 0.3, wires=wires[1]), qml.CRot(0.3, 0.4, 0.5, wires=wires[0])
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
        commutation = qml.is_commuting(
            qml.U3(0.0, 0.1, 0.0, wires=wires[1]), qml.PauliZ(wires=wires[0])
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
        commutation = qml.is_commuting(
            qml.U3(0.1, 0.0, 0.0, wires=wires[1]), qml.PauliY(wires=wires[0])
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
        commutation = qml.is_commuting(
            qml.U3(0.1, -np.pi / 2, np.pi / 2, wires=wires[0]), qml.PauliX(wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.U3(0.1, 0.2, 0.3, wires=wires[0]), qml.Rot(0.3, 0.2, 0.1, wires=wires[1])
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
        commutation = qml.is_commuting(
            qml.U3(0.0, 0.0, 0.0, wires=wires[0]), qml.Barrier(wires=wires[1])
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
        """Commutation between Barrier and U3(0.0, 0.0, 0.0)."""
        commutation = qml.is_commuting(
            qml.Barrier(wires=wires[1]), qml.U3(0.0, 0.0, 0.0, wires=wires[0])
        )
        assert commutation == res

    def test_operation_1_not_supported(self):
        """Test that giving a non supported operation raises an error."""
        rho = np.zeros((2**1, 2**1), dtype=np.complex128)
        rho[0, 0] = 1
        with pytest.raises(
            qml.QuantumFunctionError, match="Operation QubitDensityMatrix not supported."
        ):
            qml.is_commuting(qml.QubitDensityMatrix(rho, wires=[0]), qml.PauliX(wires=0))

    def test_operation_2_not_supported(self):
        """Test that giving a non supported operation raises an error."""

        with pytest.raises(qml.QuantumFunctionError, match="Operation PauliRot not supported."):
            qml.is_commuting(qml.PauliX(wires=0), qml.PauliRot(1, "X", wires=0))

    def test_operation_1_multiple_targets(self):
        """Test that giving a multiple target controlled operation raises an error."""

        def op():
            qml.PauliZ(wires=2)
            qml.PauliY(wires=2)

        with pytest.raises(
            qml.QuantumFunctionError, match="MultipleTargets controlled is not supported."
        ):
            qml.is_commuting(qml.ctrl(op, control=[0, 1])(), qml.PauliX(wires=0))

    def test_operation_2_multiple_targets(self):
        """Test that giving a multiple target controlled operation raises an error."""

        def op():
            qml.PauliZ(wires=2)
            qml.PauliY(wires=2)

        with pytest.raises(
            qml.QuantumFunctionError, match="MultipleTargets controlled is not supported."
        ):
            qml.is_commuting(qml.PauliX(wires=0), qml.ctrl(op, control=[0, 1])())

    def test_non_commuting(self):
        """Test the function with an operator from the non-commuting list."""

        res = qml.is_commuting(qml.PauliX(wires=0), qml.QFT(wires=[1, 0]))
        assert res == False
