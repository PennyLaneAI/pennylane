# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for quantum algorithmic subroutines resource operators.
"""

import math
from collections import defaultdict

import pytest

import pennylane as qp
import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.labs.estimator_beta import Allocate, Deallocate
from pennylane.labs.estimator_beta.templates import LabsQROM
from pennylane.labs.tests.estimator_beta.utils import assert_decomp_equal
from pennylane.math import ceil_log2

# pylint: disable=too-few-public-methods, too-many-arguments, no-self-use, protected-access


class TestLabsSelectPauliRot:
    """Test the custom controlled decomposition for ResourceSelectPauliRot template"""

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res",
        (
            (
                1,
                0,
                1,
                "X",
                None,
                [
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.RX, {"precision": 1e-9}),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        ),
                        2,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2),
                ],
            ),
            (
                2,
                0,
                2,
                "Y",
                1e-3,
                [
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.RY, {"precision": 1e-3}),
                                "num_ctrl_wires": 2,
                                "num_zero_ctrl": 0,
                            },
                        ),
                        2**2,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2**2),
                ],
            ),
            (
                2,
                2,
                5,
                "Z",
                1e-5,
                [
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.RZ, {"precision": 1e-5}),
                                "num_ctrl_wires": 2,
                                "num_zero_ctrl": 2,
                            },
                        ),
                        2**5,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2**5),
                ],
            ),
        ),
    )
    def test_controlled_resources(
        self, num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res
    ):
        """Test that the controlled resources are correct."""
        if precision is None:
            config = qre.LabsResourceConfig()
            kwargs = config.resource_op_precisions[qp.estimator.SelectPauliRot]
            assert (
                qre.selectpaulirot_controlled_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    num_zero_ctrl=num_zero_ctrl,
                    target_resource_params={
                        "num_ctrl_wires": num_ctrl_wires_base,
                        "rot_axis": rot_axis,
                        "precision": kwargs["precision"],
                    },
                )
                == expected_res
            )
        else:
            assert (
                qre.selectpaulirot_controlled_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    num_zero_ctrl=num_zero_ctrl,
                    target_resource_params={
                        "num_ctrl_wires": num_ctrl_wires_base,
                        "rot_axis": rot_axis,
                        "precision": precision,
                    },
                )
                == expected_res
            )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res",
        (
            (
                2,
                0,
                2,
                "Y",
                1e-3,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=5,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 4,
                            resource_rep(qre.Toffoli, {"elbow": None}): 8,
                            resource_rep(qre.T): 168,
                        },
                    ),
                ),
            ),
            (
                2,
                2,
                5,
                "Z",
                1e-5,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=8,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 32,
                            resource_rep(qre.Toffoli, {"elbow": None}): 64,
                            resource_rep(qre.T): 1792,
                            resource_rep(qre.X): 256,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_controlled_resources_estimate(
        self, num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res
    ):
        """Test that the controlled resources are correct when estimate is used."""
        op = qre.Controlled(
            qre.SelectPauliRot(
                rot_axis=rot_axis, num_ctrl_wires=num_ctrl_wires_base, precision=precision
            ),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        assert qre.estimate(op) == expected_res


class TestLabsQFT:
    """Test the resource decompositions for QFT"""

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(qre.Hadamard))],
            ),
            (
                2,
                [
                    phase_grad_reg := qre.Allocate(1, "any", True),
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(max_register_size=1),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                    qre.Deallocate(allocated_register=phase_grad_reg),
                ],
            ),
            (
                3,
                [
                    phase_grad_reg := qre.Allocate(2, "any", True),
                    GateCount(resource_rep(qre.Hadamard), 3),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(max_register_size=1),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(max_register_size=2),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                    qre.Deallocate(allocated_register=phase_grad_reg),
                ],
            ),
        ),
    )
    def test_resources_phasegrad(self, num_wires, expected_res):
        """Test that the resources are correct for phase gradient method."""
        actual_decomp = qre.qft_phase_grad_resource_decomp(num_wires)
        assert assert_decomp_equal(actual_decomp, expected_res)


class TestLabsAQFT:
    """Test the resource decompositions for AQFT"""

    @pytest.mark.parametrize(
        "num_wires, order, expected_res",
        (
            (
                5,
                1,
                [
                    GateCount(resource_rep(qre.Hadamard), 5),
                ],
            ),
            (
                5,
                3,
                [
                    GateCount(resource_rep(qre.Hadamard), 5),
                    phase_grad := Allocate(2, "any", True),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=resource_rep(qre.S),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        4,
                    ),
                    load_reg := Allocate(1, restored=True),
                    GateCount(resource_rep(qre.TemporaryAND), 1),
                    GateCount(qre.SemiAdder.resource_rep(1)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        1,
                    ),
                    Deallocate(allocated_register=load_reg),
                    load_reg := Allocate(2, restored=True),
                    GateCount(resource_rep(qre.TemporaryAND), 2 * 2),
                    GateCount(qre.SemiAdder.resource_rep(2), 2),
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        2 * 2,
                    ),
                    Deallocate(allocated_register=load_reg),
                    GateCount(resource_rep(qre.SWAP), 2),
                    Deallocate(allocated_register=phase_grad),
                ],
            ),
            (
                5,
                5,
                [
                    GateCount(resource_rep(qre.Hadamard), 5),
                    phase_grad := Allocate(3, "any", True),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=resource_rep(qre.S),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        4,
                    ),
                    data_reg := Allocate(1, restored=True),
                    GateCount(resource_rep(qre.TemporaryAND), 1),
                    GateCount(qre.SemiAdder.resource_rep(1)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        1,
                    ),
                    Deallocate(allocated_register=data_reg),
                    data_reg := Allocate(2, restored=True),
                    GateCount(resource_rep(qre.TemporaryAND), 2),
                    GateCount(qre.SemiAdder.resource_rep(2)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        2,
                    ),
                    Deallocate(allocated_register=data_reg),
                    data_reg := Allocate(3, restored=True),
                    GateCount(resource_rep(qre.TemporaryAND), 3),
                    GateCount(qre.SemiAdder.resource_rep(3)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TemporaryAND)),
                        3,
                    ),
                    Deallocate(allocated_register=data_reg),
                    GateCount(resource_rep(qre.SWAP), 2),
                    Deallocate(allocated_register=phase_grad),
                ],
            ),
        ),
    )
    def test_resources(self, order, num_wires, expected_res):
        """Test that the resources are correct."""
        assert assert_decomp_equal(qre.aqft_resource_decomp(order, num_wires), expected_res)


class TestLabsSelectTHC:
    """Test the resource decompositions for SelectTHC"""

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, num_batches, rotation_prec, selswap_depth, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                1,
                13,
                1,
                {"algo_wires": 138, "auxiliary_wires": 764, "toffoli_gates": 5671},
            ),
            (
                qre.THCHamiltonian(10, 50),
                1,
                15,
                None,
                {"algo_wires": 38, "auxiliary_wires": 162, "toffoli_gates": 1104},
            ),
            (
                qre.THCHamiltonian(4, 20),
                1,
                15,
                2,
                {"algo_wires": 24, "auxiliary_wires": 107, "toffoli_gates": 450},
            ),
            # These numbers were obtained manually for batched rotations based on the technique described in arXiv:2501.06165
            (
                qre.THCHamiltonian(58, 160),
                2,
                13,
                None,
                {"algo_wires": 138, "auxiliary_wires": 400, "toffoli_gates": 6044},
            ),
        ),
    )
    def test_resources(self, thc_ham, num_batches, rotation_prec, selswap_depth, expected_res):
        """Test that the resource decompostion for SelectTHC is correct."""

        select_cost = qre.estimate(
            qre.SelectTHC(
                thc_ham,
                num_batches=num_batches,
                rotation_precision=rotation_prec,
                select_swap_depth=selswap_depth,
            )
        )
        assert select_cost.algo_wires == expected_res["algo_wires"]
        assert (
            select_cost.zeroed_wires + select_cost.any_state_wires
            == expected_res["auxiliary_wires"]
        )
        assert select_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, num_batches, rotation_prec, selswap_depth, num_ctrl_wires, num_zero_ctrl, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                1,
                13,
                1,
                1,
                1,
                {"algo_wires": 139, "auxiliary_wires": 764, "toffoli_gates": 5672},
            ),
            (
                qre.THCHamiltonian(10, 50),
                1,
                15,
                None,
                2,
                0,
                {"algo_wires": 40, "auxiliary_wires": 163, "toffoli_gates": 1107},
            ),
            (
                qre.THCHamiltonian(4, 20),
                1,
                15,
                2,
                3,
                2,
                {"algo_wires": 27, "auxiliary_wires": 108, "toffoli_gates": 457},
            ),
            # These numbers were obtained manually for batched rotations based on the technique described in arXiv:2501.06165
            (
                qre.THCHamiltonian(58, 160),
                2,
                13,
                None,
                1,
                1,
                {"algo_wires": 139, "auxiliary_wires": 400, "toffoli_gates": 6045},
            ),
        ),
    )
    def test_controlled_resources(
        self,
        thc_ham,
        num_batches,
        rotation_prec,
        selswap_depth,
        num_ctrl_wires,
        num_zero_ctrl,
        expected_res,
    ):
        """Test that the controlled resource decompostion for SelectTHC is correct."""

        ctrl_select_cost = qre.estimate(
            qre.Controlled(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                base_op=qre.SelectTHC(
                    thc_ham,
                    num_batches=num_batches,
                    rotation_precision=rotation_prec,
                    select_swap_depth=selswap_depth,
                ),
            )
        )
        assert ctrl_select_cost.algo_wires == expected_res["algo_wires"]
        assert (
            ctrl_select_cost.zeroed_wires + ctrl_select_cost.any_state_wires
            == expected_res["auxiliary_wires"]
        )
        assert ctrl_select_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]


class TestLabsQROMStatePreparation:
    """Test the update resource decomposition functions for QROMStatePrep"""

    @pytest.mark.parametrize(
        "num_state_qubits, precision, positive_and_real, selswap_depths, expected_res",
        (
            (
                5,
                None,
                False,
                1,
                [
                    load := Allocate(32, restored=True),
                    phase_grad := Allocate(32, "any", True),
                    GateCount(qre.Hadamard.resource_rep(), 32),
                    GateCount(qre.S.resource_rep(), 32),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=32,
                                num_bit_flips=16,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(32),
                                1,
                                0,
                            ),
                            num_wires=64,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=32,
                                num_bit_flips=32,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(32),
                                1,
                                0,
                            ),
                            num_wires=65,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=32,
                                num_bit_flips=64,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(32),
                                1,
                                0,
                            ),
                            num_wires=66,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=32,
                                num_bit_flips=128,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(32),
                                1,
                                0,
                            ),
                            num_wires=67,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=32,
                                num_bit_flips=256,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(32),
                                1,
                                0,
                            ),
                            num_wires=68,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=32,
                                size_bitstring=32,
                                num_bit_flips=512,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(32),
                                1,
                                0,
                            ),
                            num_wires=69,
                        ),
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 32),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        32,
                    ),
                    Deallocate(allocated_register=load),
                    Deallocate(allocated_register=phase_grad),
                ],
            ),
            (
                4,
                1e-5,
                False,
                1,
                [
                    load := Allocate(19, restored=True),
                    phase_grad := Allocate(19, "any", True),
                    GateCount(qre.Hadamard.resource_rep(), 19),
                    GateCount(qre.S.resource_rep(), 19),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=19,
                                num_bit_flips=9,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(19),
                                1,
                                0,
                            ),
                            num_wires=38,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=19,
                                num_bit_flips=19,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(19),
                                1,
                                0,
                            ),
                            num_wires=39,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=19,
                                num_bit_flips=38,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(19),
                                1,
                                0,
                            ),
                            num_wires=40,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=19,
                                num_bit_flips=76,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(19),
                                1,
                                0,
                            ),
                            num_wires=41,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=19,
                                num_bit_flips=152,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(19),
                                1,
                                0,
                            ),
                            num_wires=42,
                        ),
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 19),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        19,
                    ),
                    Deallocate(allocated_register=load),
                    Deallocate(allocated_register=phase_grad),
                ],
            ),
            (
                3,
                1e-4,
                False,
                2,
                [
                    load := Allocate(15, restored=True),
                    phase_grad := Allocate(15, "any", True),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(qre.S.resource_rep(), 15),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=30,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=31,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=32,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=33,
                        ),
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        15,
                    ),
                    Deallocate(allocated_register=load),
                    Deallocate(allocated_register=phase_grad),
                ],
            ),
            (
                3,
                1e-4,
                True,
                [1, 2, 2],
                [
                    load := Allocate(15, restored=True),
                    phase_grad := Allocate(15, "any", True),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(qre.S.resource_rep(), 15),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=30,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=31,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=32,
                        ),
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        15,
                    ),
                    Deallocate(allocated_register=load),
                    Deallocate(allocated_register=phase_grad),
                ],
            ),
            (
                3,
                1e-4,
                False,
                [None, 1, None, 4],
                [
                    load := Allocate(15, restored=True),
                    phase_grad := Allocate(15, "any", True),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(qre.S.resource_rep(), 15),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                borrow_qubits=True,
                                select_swap_depth=None,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=30,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=31,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                borrow_qubits=True,
                                select_swap_depth=None,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=32,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                borrow_qubits=True,
                                select_swap_depth=4,
                            ),
                            cmpr_target_op=qre.Controlled.resource_rep(
                                qre.SemiAdder.resource_rep(15),
                                1,
                                0,
                            ),
                            num_wires=33,
                        ),
                    ),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.S.resource_rep()),
                        15,
                    ),
                    Deallocate(allocated_register=load),
                    Deallocate(allocated_register=phase_grad),
                ],
            ),
        ),
    )
    def test_default_resources(
        self, num_state_qubits, precision, positive_and_real, selswap_depths, expected_res
    ):
        """Test that the resources are as expected for the default decomposition"""

        if precision is None:
            config = qre.LabsResourceConfig()
            kwargs = config.resource_op_precisions[qre.QROMStatePreparation]
            actual_resources = qre.qrom_state_preparation_phase_grad_resource_decomp(
                num_state_qubits=num_state_qubits,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
                **kwargs,
            )
        else:
            actual_resources = qre.qrom_state_preparation_phase_grad_resource_decomp(
                num_state_qubits=num_state_qubits,
                precision=precision,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
            )

        assert assert_decomp_equal(actual_resources, expected_res)

    @pytest.mark.parametrize(
        "num_state_qubits, precision, positive_and_real, selswap_depths, expected_res",
        (
            (
                5,
                None,
                False,
                1,
                [
                    load := Allocate(32, restored=True),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=32,
                                num_bit_flips=16,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 32),),
                                num_wires=33,
                            ),
                            num_wires=33,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=32,
                                num_bit_flips=32,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 32),),
                                num_wires=33,
                            ),
                            num_wires=34,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=32,
                                num_bit_flips=64,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 32),),
                                num_wires=33,
                            ),
                            num_wires=35,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=32,
                                num_bit_flips=128,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 32),),
                                num_wires=33,
                            ),
                            num_wires=36,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=32,
                                num_bit_flips=256,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 32),),
                                num_wires=33,
                            ),
                            num_wires=37,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=32,
                                size_bitstring=32,
                                num_bit_flips=512,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 32),),
                                num_wires=32,
                            ),
                            num_wires=37,
                        ),
                    ),
                    Deallocate(allocated_register=load),
                ],
            ),
            (
                4,
                1e-5,
                False,
                1,
                [
                    load := Allocate(19, restored=True),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=19,
                                num_bit_flips=9,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 19),),
                                num_wires=20,
                            ),
                            num_wires=20,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=19,
                                num_bit_flips=19,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 19),),
                                num_wires=20,
                            ),
                            num_wires=21,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=19,
                                num_bit_flips=38,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 19),),
                                num_wires=20,
                            ),
                            num_wires=22,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=19,
                                num_bit_flips=76,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 19),),
                                num_wires=20,
                            ),
                            num_wires=23,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=16,
                                size_bitstring=19,
                                num_bit_flips=152,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 19),),
                                num_wires=19,
                            ),
                            num_wires=23,
                        ),
                    ),
                    Deallocate(allocated_register=load),
                ],
            ),
            (
                3,
                1e-4,
                False,
                2,
                [
                    load := Allocate(15, restored=True),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=16,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=17,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=18,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 15),),
                                num_wires=15,
                            ),
                            num_wires=18,
                        ),
                    ),
                    Deallocate(allocated_register=load),
                ],
            ),
            (
                3,
                1e-4,
                True,
                [1, 2, 2],
                [
                    load := Allocate(15, restored=True),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=16,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=17,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                borrow_qubits=True,
                                select_swap_depth=2,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=18,
                        ),
                    ),
                    Deallocate(allocated_register=load),
                ],
            ),
            (
                3,
                1e-4,
                False,
                [None, 1, None, 4],
                [
                    load := Allocate(15, restored=True),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=1,
                                size_bitstring=15,
                                num_bit_flips=7,
                                borrow_qubits=True,
                                select_swap_depth=None,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=16,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=2,
                                size_bitstring=15,
                                num_bit_flips=15,
                                borrow_qubits=True,
                                select_swap_depth=1,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=17,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=4,
                                size_bitstring=15,
                                num_bit_flips=30,
                                borrow_qubits=True,
                                select_swap_depth=None,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.CRY.resource_rep(), 15),),
                                num_wires=16,
                            ),
                            num_wires=18,
                        ),
                    ),
                    GateCount(
                        qre.ChangeOpBasis.resource_rep(
                            cmpr_compute_op=qre.QROM.resource_rep(
                                num_bitstrings=8,
                                size_bitstring=15,
                                num_bit_flips=60,
                                borrow_qubits=True,
                                select_swap_depth=4,
                            ),
                            cmpr_target_op=qre.Prod.resource_rep(
                                ((qre.PhaseShift.resource_rep(), 15),),
                                num_wires=15,
                            ),
                            num_wires=18,
                        ),
                    ),
                    Deallocate(allocated_register=load),
                ],
            ),
        ),
    )
    def test_control_ry_resources(
        self, num_state_qubits, precision, positive_and_real, selswap_depths, expected_res
    ):
        """Test that the resources are as expected for the controlled-RY decomposition"""
        if precision is None:
            config = qre.LabsResourceConfig()
            kwargs = config.resource_op_precisions[qre.QROMStatePreparation]
            actual_resources = qre.qrom_state_preparation_resource_decomp(
                num_state_qubits=num_state_qubits,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
                **kwargs,
            )
        else:
            actual_resources = qre.qrom_state_preparation_resource_decomp(
                num_state_qubits=num_state_qubits,
                precision=precision,
                positive_and_real=positive_and_real,
                selswap_depths=selswap_depths,
            )

        assert assert_decomp_equal(actual_resources, expected_res)


class TestLabsQROM:
    """Test the resource LabsQROM class."""

    def test_select_swap_depth_errors(self):
        """Test that the correct error is raised when invalid values of
        select_swap_depth are provided.
        """
        select_swap_depth = "Not A Valid Input"
        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            LabsQROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            LabsQROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

        select_swap_depth = 3
        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            LabsQROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            LabsQROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, borrow",
        (
            (10, 3, 15, None, True),
            (100, 5, 50, 2, False),
            (12, 2, 5, 1, True),
        ),
    )
    def test_resource_params(self, num_data_points, size_data_points, num_bit_flips, depth, borrow):
        """Test that the resource params are correct."""
        if depth is None:
            op = LabsQROM(num_data_points, size_data_points)
        else:
            op = LabsQROM(num_data_points, size_data_points, num_bit_flips, borrow, depth)

        assert op.resource_params == {
            "num_bitstrings": num_data_points,
            "size_bitstring": size_data_points,
            "num_bit_flips": num_bit_flips,
            "select_swap_depth": depth,
            "borrow_qubits": borrow,
        }

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, borrow",
        (
            (10, 3, 15, None, True),
            (100, 5, 50, 2, False),
            (12, 2, 5, 1, True),
        ),
    )
    def test_resource_rep(self, num_data_points, size_data_points, num_bit_flips, depth, borrow):
        """Test that the compressed representation is correct."""
        expected_num_wires = size_data_points + qp.math.ceil_log2(num_data_points)
        expected = qre.CompressedResourceOp(
            LabsQROM,
            expected_num_wires,
            {
                "num_bitstrings": num_data_points,
                "size_bitstring": size_data_points,
                "num_bit_flips": num_bit_flips,
                "select_swap_depth": depth,
                "borrow_qubits": borrow,
            },
        )
        assert (
            LabsQROM.resource_rep(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                borrow_qubits=borrow,
                select_swap_depth=depth,
            )
            == expected
        )

    def test_t_select_swap_width(self):
        """Test that the private function doesn't give negative or
        fractional values for the depth"""
        num_bitstrings = 8
        size_bitstring = 17

        opt_width = LabsQROM._t_optimized_select_swap_width(
            num_bitstrings,
            size_bitstring,
            borrow=False,
        )
        assert opt_width == 1

    @pytest.mark.parametrize(
        "num_data, num_flips, repeat, expected",  # computed by hand
        (
            (1, 5, 1, [GateCount(qre.X.resource_rep(), 5)]),
            (
                2,
                3,
                1,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.CNOT.resource_rep(), 3),
                ],
            ),
            (
                3,
                10,
                1,
                [
                    GateCount(qre.X.resource_rep(), 4),
                    GateCount(qre.CNOT.resource_rep(), 10 + 1),
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 1),
                ],
            ),
            (
                4,
                10,
                1,
                [
                    GateCount(qre.X.resource_rep(), 4),
                    GateCount(qre.CNOT.resource_rep(), 10 + 4),
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 1),
                ],
            ),
            (
                9,
                10,
                1,
                [
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 10 + 7),
                    GateCount(qre.TemporaryAND.resource_rep(), 7),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 7),
                ],
            ),
            (1, 5, 2, [GateCount(qre.X.resource_rep(), 10)]),
            (
                2,
                3,
                3,
                [
                    GateCount(qre.X.resource_rep(), 6),
                    GateCount(qre.CNOT.resource_rep(), 9),
                ],
            ),
            (
                3,
                10,
                4,
                [
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 40 + 4),
                    GateCount(qre.TemporaryAND.resource_rep(), 4),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 4),
                ],
            ),
            (
                4,
                10,
                3,
                [
                    GateCount(qre.X.resource_rep(), 12),
                    GateCount(qre.CNOT.resource_rep(), 30 + 12),
                    GateCount(qre.TemporaryAND.resource_rep(), 3),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 3),
                ],
            ),
            (
                9,
                10,
                2,
                [
                    GateCount(qre.X.resource_rep(), 32),
                    GateCount(qre.CNOT.resource_rep(), 20 + 14),
                    GateCount(qre.TemporaryAND.resource_rep(), 14),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 14),
                ],
            ),
        ),
    )
    def test_select_cost(self, num_data, num_flips, repeat, expected):
        """Test that the private _select_cost method works as expected"""
        assert LabsQROM._select_cost(num_data, num_flips, repeat) == expected

    @pytest.mark.parametrize(
        "num_data, num_flips, repeat, expected",  # computed by hand
        (
            (1, 5, 1, [GateCount(qre.CNOT.resource_rep(), 5)]),
            (
                2,
                3,
                1,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.CNOT.resource_rep(), 3 + 1),
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 1),
                ],
            ),
            (
                3,
                10,
                1,
                [
                    GateCount(qre.X.resource_rep(), 4),
                    GateCount(qre.CNOT.resource_rep(), 10 + 2),
                    GateCount(qre.TemporaryAND.resource_rep(), 2),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 2),
                ],
            ),
            (
                4,
                10,
                1,
                [
                    GateCount(qre.X.resource_rep(), 6),
                    GateCount(qre.CNOT.resource_rep(), 10 + 3),
                    GateCount(qre.TemporaryAND.resource_rep(), 3),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 3),
                ],
            ),
            (
                9,
                10,
                1,
                [
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 10 + 8),
                    GateCount(qre.TemporaryAND.resource_rep(), 8),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 8),
                ],
            ),
            (1, 5, 2, [GateCount(qre.CNOT.resource_rep(), 10)]),
            (
                2,
                3,
                3,
                [
                    GateCount(qre.X.resource_rep(), 6),
                    GateCount(qre.CNOT.resource_rep(), 9 + 3),
                    GateCount(qre.TemporaryAND.resource_rep(), 3),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 3),
                ],
            ),
            (
                3,
                10,
                4,
                [
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 40 + 8),
                    GateCount(qre.TemporaryAND.resource_rep(), 8),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 8),
                ],
            ),
            (
                4,
                10,
                3,
                [
                    GateCount(qre.X.resource_rep(), 18),
                    GateCount(qre.CNOT.resource_rep(), 30 + 9),
                    GateCount(qre.TemporaryAND.resource_rep(), 9),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 9),
                ],
            ),
            (
                9,
                10,
                2,
                [
                    GateCount(qre.X.resource_rep(), 32),
                    GateCount(qre.CNOT.resource_rep(), 20 + 16),
                    GateCount(qre.TemporaryAND.resource_rep(), 16),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 16),
                ],
            ),
        ),
    )
    def test_control_select_cost(self, num_data, num_flips, repeat, expected):
        """Test that the private _single_ctrl_select_cost method works as expected"""
        assert LabsQROM._single_ctrl_select_cost(num_data, num_flips, repeat) == expected

    @pytest.mark.parametrize(
        "reg_size, num_swap_ctrls, repeat, expected",  # computed by hand
        (
            (1, 1, 1, [GateCount(qre.CSWAP.resource_rep(), 1)]),
            (2, 1, 1, [GateCount(qre.CSWAP.resource_rep(), 2)]),
            (3, 2, 1, [GateCount(qre.CSWAP.resource_rep(), 3 * 3)]),
            (5, 3, 1, [GateCount(qre.CSWAP.resource_rep(), 5 * 7)]),
            (3, 4, 1, [GateCount(qre.CSWAP.resource_rep(), 3 * 15)]),
            (1, 1, 6, [GateCount(qre.CSWAP.resource_rep(), 6)]),
            (2, 1, 7, [GateCount(qre.CSWAP.resource_rep(), 2 * 7)]),
            (3, 2, 8, [GateCount(qre.CSWAP.resource_rep(), 3 * 3 * 8)]),
            (5, 3, 7, [GateCount(qre.CSWAP.resource_rep(), 5 * 7 * 7)]),
            (3, 4, 6, [GateCount(qre.CSWAP.resource_rep(), 3 * 15 * 6)]),
        ),
    )
    def test_swap_cost(self, reg_size, num_swap_ctrls, repeat, expected):
        """Test that the private _swap_cost method works as expected"""
        assert LabsQROM._swap_cost(reg_size, num_swap_ctrls, repeat) == expected

    @pytest.mark.parametrize(
        "reg_size, num_swap_ctrls, repeat, base_decomp",  # computed by hand
        (
            (
                1,
                1,
                1,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.CSWAP.resource_rep(), 1),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 1),
                ],
            ),
            (
                2,
                1,
                1,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.CSWAP.resource_rep(), 2),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 1),
                ],
            ),
            (
                3,
                2,
                1,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 2),
                    GateCount(qre.CSWAP.resource_rep(), 3 * 3),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 2),
                ],
            ),
            (
                5,
                3,
                1,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 3),
                    GateCount(qre.CSWAP.resource_rep(), 5 * 7),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 3),
                ],
            ),
            (
                3,
                4,
                1,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 4),
                    GateCount(qre.CSWAP.resource_rep(), 3 * 15),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 4),
                ],
            ),
            (
                1,
                1,
                6,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 6),
                    GateCount(qre.CSWAP.resource_rep(), 6),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 6),
                ],
            ),
            (
                2,
                1,
                7,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 7),
                    GateCount(qre.CSWAP.resource_rep(), 2 * 7),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 7),
                ],
            ),
            (
                3,
                2,
                8,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 2 * 8),
                    GateCount(qre.CSWAP.resource_rep(), 3 * 3 * 8),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 2 * 8),
                ],
            ),
            (
                5,
                3,
                7,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 3 * 7),
                    GateCount(qre.CSWAP.resource_rep(), 5 * 7 * 7),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 3 * 7),
                ],
            ),
            (
                3,
                4,
                6,
                [
                    GateCount(qre.TemporaryAND.resource_rep(), 4 * 6),
                    GateCount(qre.CSWAP.resource_rep(), 3 * 15 * 6),
                    GateCount(qre.Adjoint.resource_rep(qre.TemporaryAND.resource_rep()), 4 * 6),
                ],
            ),
        ),
    )
    def test_control_swap_cost(self, reg_size, num_swap_ctrls, repeat, base_decomp):
        """Test that the private _single_ctrl_swap_cost method works as expected"""
        ## Add qubit allocation:
        alloc_reg = qre.Allocate(1, "zero", True)
        dealloc_reg = qre.Deallocate(allocated_register=alloc_reg)
        expected = [alloc_reg] + base_decomp + [dealloc_reg]

        computed = LabsQROM._single_ctrl_swap_cost(reg_size, num_swap_ctrls, repeat)
        assert assert_decomp_equal(computed, expected)

    @pytest.mark.parametrize(
        "reg_size, num_swap_ctrls, repeat, expected",  # computed by hand
        (
            (
                1,
                1,
                1,
                [
                    GateCount(qre.Hadamard.resource_rep(), 1),
                    GateCount(qre.CZ.resource_rep(), 1),
                    GateCount(qre.CNOT.resource_rep(), 1),
                ],
            ),
            (
                2,
                1,
                1,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CZ.resource_rep(), 2),
                    GateCount(qre.CNOT.resource_rep(), 2),
                ],
            ),
            (
                3,
                2,
                1,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3 * 3),
                    GateCount(qre.CZ.resource_rep(), 3 * 3),
                    GateCount(qre.CNOT.resource_rep(), 3 * 3),
                ],
            ),
            (
                5,
                3,
                1,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5 * 7),
                    GateCount(qre.CZ.resource_rep(), 5 * 7),
                    GateCount(qre.CNOT.resource_rep(), 5 * 7),
                ],
            ),
            (
                3,
                4,
                1,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3 * 15),
                    GateCount(qre.CZ.resource_rep(), 3 * 15),
                    GateCount(qre.CNOT.resource_rep(), 3 * 15),
                ],
            ),
            (
                1,
                1,
                6,
                [
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.CZ.resource_rep(), 6),
                    GateCount(qre.CNOT.resource_rep(), 6),
                ],
            ),
            (
                2,
                1,
                7,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2 * 7),
                    GateCount(qre.CZ.resource_rep(), 2 * 7),
                    GateCount(qre.CNOT.resource_rep(), 2 * 7),
                ],
            ),
            (
                3,
                2,
                8,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3 * 3 * 8),
                    GateCount(qre.CZ.resource_rep(), 3 * 3 * 8),
                    GateCount(qre.CNOT.resource_rep(), 3 * 3 * 8),
                ],
            ),
            (
                5,
                3,
                7,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5 * 7 * 7),
                    GateCount(qre.CZ.resource_rep(), 5 * 7 * 7),
                    GateCount(qre.CNOT.resource_rep(), 5 * 7 * 7),
                ],
            ),
            (
                3,
                4,
                6,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3 * 15 * 6),
                    GateCount(qre.CZ.resource_rep(), 3 * 15 * 6),
                    GateCount(qre.CNOT.resource_rep(), 3 * 15 * 6),
                ],
            ),
        ),
    )
    def test_swap_adj_cost(self, reg_size, num_swap_ctrls, repeat, expected):
        """Test that the private _swap_adj_cost method works as expected"""
        assert LabsQROM._swap_adj_cost(reg_size, num_swap_ctrls, repeat) == expected

    @pytest.mark.parametrize(
        "resource_params, alloc_reg, base_decomp",  # computed by hand,
        (
            (
                {
                    "num_bitstrings": 25,  # d
                    "size_bitstring": 8,  # M
                    "num_bit_flips": 50,
                },
                None,  # k ~ 4 < M = 8, don't need to allocate
                (
                    [GateCount(qre.Hadamard.resource_rep(), 8), GateCount(qre.X.resource_rep())]
                    + LabsQROM._swap_cost(register_size=1, num_swap_ctrls=2)
                    + [GateCount(qre.Hadamard.resource_rep(), 4)]
                    + LabsQROM._select_cost(math.ceil(25 / 4), 50)
                    + [GateCount(qre.Hadamard.resource_rep(), 4)]
                    + LabsQROM._swap_adj_cost(register_size=1, num_swap_ctrls=2)
                    + [GateCount(qre.X.resource_rep())]
                ),
            ),
            (
                {
                    "num_bitstrings": 49,  # d
                    "size_bitstring": 4,  # M
                    "num_bit_flips": 50,
                },
                qre.Allocate(8 - 4, state="zero", restored=True),  # k ~ 8 > M = 4,
                (
                    [GateCount(qre.Hadamard.resource_rep(), 4), GateCount(qre.X.resource_rep())]
                    + LabsQROM._swap_cost(register_size=1, num_swap_ctrls=3)
                    + [GateCount(qre.Hadamard.resource_rep(), 8)]
                    + LabsQROM._select_cost(math.ceil(49 / 8), 50)
                    + [GateCount(qre.Hadamard.resource_rep(), 8)]
                    + LabsQROM._swap_adj_cost(register_size=1, num_swap_ctrls=3)
                    + [GateCount(qre.X.resource_rep())]
                ),
            ),
        ),
    )
    def test_qrom_adjoint_clean(self, resource_params, alloc_reg, base_decomp):
        """Test that the qrom_clean_auxiliary_adjoint_resource_decomp method works as expected"""
        expected = base_decomp
        if alloc_reg:
            dealloc_reg = qre.Deallocate(allocated_register=alloc_reg)
            expected = expected[:1] + [alloc_reg] + expected[1:] + [dealloc_reg]

        computed = LabsQROM.qrom_clean_auxiliary_adjoint_resource_decomp(resource_params)
        assert assert_decomp_equal(computed, expected)

    @pytest.mark.parametrize(
        "resource_params, alloc_reg, base_decomp",  # computed by hand,
        (
            (
                {
                    "num_bitstrings": 25,  # d
                    "size_bitstring": 8,  # M
                    "num_bit_flips": 50,
                },
                None,  # k ~ 4 < M = 8, don't need to allocate
                (
                    [GateCount(qre.Hadamard.resource_rep(), 8), GateCount(qre.X.resource_rep())]
                    + LabsQROM._swap_cost(register_size=1, num_swap_ctrls=2)
                    + [GateCount(qre.Hadamard.resource_rep(), 4)]
                    + LabsQROM._select_cost(math.ceil(25 / 4), 50)
                    + [GateCount(qre.Hadamard.resource_rep(), 4)]
                    + LabsQROM._swap_adj_cost(register_size=1, num_swap_ctrls=2)
                    + [GateCount(qre.X.resource_rep())]
                ),
            ),
            (
                {
                    "num_bitstrings": 49,  # d
                    "size_bitstring": 4,  # M
                    "num_bit_flips": 50,
                },
                qre.Allocate(8 - 4, state="zero", restored=True),  # k ~ 8 > M = 4,
                (
                    [GateCount(qre.Hadamard.resource_rep(), 4), GateCount(qre.X.resource_rep())]
                    + LabsQROM._swap_cost(register_size=1, num_swap_ctrls=3)
                    + [GateCount(qre.Hadamard.resource_rep(), 8)]
                    + LabsQROM._select_cost(math.ceil(49 / 8), 50)
                    + [GateCount(qre.Hadamard.resource_rep(), 8)]
                    + LabsQROM._swap_adj_cost(register_size=1, num_swap_ctrls=3)
                    + [GateCount(qre.X.resource_rep())]
                ),
            ),
        ),
    )
    def test_adjoint_resources(self, resource_params, alloc_reg, base_decomp):
        """Test that the adjoint resources are as expected"""
        expected = base_decomp
        if alloc_reg:
            dealloc_reg = qre.Deallocate(allocated_register=alloc_reg)
            expected = expected[:1] + [alloc_reg] + expected[1:] + [dealloc_reg]

        computed = LabsQROM.adjoint_resource_decomp(resource_params)
        assert assert_decomp_equal(computed, expected)

    @pytest.mark.parametrize(
        "resource_params, alloc_reg, base_decomp",  # computed by hand,
        (
            (
                {
                    "num_bitstrings": 50,  # d
                    "size_bitstring": 8,  # M
                    "num_bit_flips": 50,
                },
                None,  # k ~ 4 < M = 8, don't need to allocate
                (
                    [
                        GateCount(qre.Hadamard.resource_rep(), 8),
                        GateCount(qre.Z.resource_rep(), 2),
                        GateCount(qre.Hadamard.resource_rep(), 2),
                    ]
                    + LabsQROM._swap_cost(register_size=1, num_swap_ctrls=2, repeat=4)
                    + LabsQROM._select_cost(math.ceil(50 / 4), 50, 2)
                ),
            ),
            (
                {
                    "num_bitstrings": 98,  # d
                    "size_bitstring": 4,  # M
                    "num_bit_flips": 50,
                },
                qre.Allocate(8 - 4, state="any", restored=True),  # k ~ 8 > M = 4,
                (
                    [
                        GateCount(qre.Hadamard.resource_rep(), 4),
                        GateCount(qre.Z.resource_rep(), 2),
                        GateCount(qre.Hadamard.resource_rep(), 2),
                    ]
                    + LabsQROM._swap_cost(register_size=1, num_swap_ctrls=3, repeat=4)
                    + LabsQROM._select_cost(math.ceil(98 / 8), 50, 2)
                ),
            ),
        ),
    )
    def test_qrom_adjoint_dirty(self, resource_params, alloc_reg, base_decomp):
        """Test that the qrom_dirty_auxiliary_adjoint_resource_decomp method works as expected"""
        expected = base_decomp
        if alloc_reg:
            dealloc_reg = qre.Deallocate(allocated_register=alloc_reg)
            expected = expected[:1] + [alloc_reg] + expected[1:] + [dealloc_reg]

        computed = LabsQROM.qrom_dirty_auxiliary_adjoint_resource_decomp(resource_params)
        assert assert_decomp_equal(computed, expected)

    @staticmethod
    def resources_data(index):
        """Store the expected resources used in the test_resources method"""
        resources = []
        if index == 0:  # 10, 3, 15, None, True
            # opt_depth = 1 because sqrt(10/(3*2)) ~ 1
            allocate_sel = qre.Allocate(ceil_log2(10) - 1, "zero", True)

            resources = (
                [allocate_sel]
                + LabsQROM._select_cost(10, 15)
                + [qre.Deallocate(allocated_register=allocate_sel)]
            )
        if index == 1:  # 100, 5, 50, 2, False
            allocate_sel = qre.Allocate(ceil_log2(50) - 1, "zero", True)
            allocate_swap = qre.Allocate(5, "zero", True)
            resources = (
                [allocate_sel, allocate_swap]
                + LabsQROM._select_cost(50, 50)
                + [qre.Deallocate(allocated_register=allocate_sel)]
                + LabsQROM._swap_cost(5, 1)
                + [
                    qre.GateCount(qre.Hadamard.resource_rep(), 5),
                    qre.Deallocate(allocated_register=allocate_swap),
                ]
            )
        if index == 2:  # 12, 2, 5, 1, True
            allocate_sel = qre.Allocate(ceil_log2(12) - 1, "zero", True)
            resources = (
                [allocate_sel]
                + LabsQROM._select_cost(12, 5)
                + [qre.Deallocate(allocated_register=allocate_sel)]
            )
        if index == 3:  # 12, 2, 5, 128, False
            max_depth = 16  # 128 depth is not possible, truncate to 16
            allocate_swap = qre.Allocate((max_depth - 1) * 2, "zero", True)
            resources = (
                [allocate_swap]
                + LabsQROM._select_cost(1, 5)
                + LabsQROM._swap_cost(2, 4)
                + [
                    qre.GateCount(qre.Hadamard.resource_rep(), (max_depth - 1) * 2),
                    qre.Deallocate(allocated_register=allocate_swap),
                ]
            )
        if index == 4:  # 12, 2, 5, 4, True
            allocate_sel = qre.Allocate(ceil_log2(12 / 4) - 1, "zero", True)
            allocate_swap = qre.Allocate((4 - 1) * 2, "any", True)
            h = qre.Hadamard.resource_rep()

            resources = (
                [allocate_sel, allocate_swap]
                + [qre.GateCount(h, 2 * 2)]
                + LabsQROM._select_cost(3, 5, repeat=2)
                + [qre.Deallocate(allocated_register=allocate_sel)]
                + LabsQROM._swap_cost(2, 2, repeat=4)
                + [qre.Deallocate(allocated_register=allocate_swap)]
            )
        return resources

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, borrow, expected_res_index",
        (
            (10, 3, 15, None, True, 0),
            (100, 5, 50, 2, False, 1),
            (12, 2, 5, 1, True, 2),
            (12, 2, 5, 128, False, 3),
            (12, 2, 5, 4, True, 4),
        ),
    )
    def test_resources(
        self, num_data_points, size_data_points, num_bit_flips, depth, borrow, expected_res_index
    ):
        """Test that the resources are correct."""
        expected_decomp = self.resources_data(expected_res_index)

        computed_decomp = LabsQROM.resource_decomp(
            num_bitstrings=num_data_points,
            size_bitstring=size_data_points,
            num_bit_flips=num_bit_flips,
            borrow_qubits=borrow,
            select_swap_depth=depth,
        )
        assert assert_decomp_equal(computed_decomp, expected_decomp)

    @staticmethod
    def single_ctrl_resources_data(index):
        """Store the expected resources used in the test_single_controlled_res_decomp method"""
        resources = []
        if index == 0:  # 10, 3, 15, None, True
            # opt_depth = 1 because sqrt(10/(3*2)) ~ 1
            allocate_sel = qre.Allocate(ceil_log2(10), "zero", True)

            resources = (
                [allocate_sel]
                + LabsQROM._single_ctrl_select_cost(10, 15)
                + [qre.Deallocate(allocated_register=allocate_sel)]
            )
        if index == 1:  # 10, 3, 15, 2, True
            allocate_sel = qre.Allocate(ceil_log2(5), "zero", True)
            allocate_swap = qre.Allocate(3, "any", True)
            h = qre.Hadamard.resource_rep()

            resources = (
                [allocate_sel, allocate_swap]
                + [qre.GateCount(h, 3 * 2)]
                + LabsQROM._single_ctrl_select_cost(5, 15, repeat=2)
                + [qre.Deallocate(allocated_register=allocate_sel)]
                + LabsQROM._single_ctrl_swap_cost(3, 1, repeat=4)
                + [qre.Deallocate(allocated_register=allocate_swap)]
            )
        if index == 2:  # 12, 2, 5, 16, True
            allocate_swap = qre.Allocate(15 * 2, "any", True)
            h = qre.Hadamard.resource_rep()

            resources = (
                [allocate_swap]
                + [qre.GateCount(h, 2 * 2)]
                + LabsQROM._single_ctrl_select_cost(1, 5, repeat=2)
                + LabsQROM._single_ctrl_swap_cost(2, 4, repeat=4)
                + [qre.Deallocate(allocated_register=allocate_swap)]
            )
        return resources

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, borrow, expected_res_index",
        (
            (10, 3, 15, None, True, 0),
            (10, 3, 15, 2, True, 1),
            (12, 2, 5, 16, True, 2),
        ),
    )
    def test_single_controlled_res_decomp(
        self, num_data_points, size_data_points, num_bit_flips, depth, borrow, expected_res_index
    ):
        """Test that the resources computed by single_controlled_res_decomp are correct."""
        expected_decomp = self.single_ctrl_resources_data(expected_res_index)
        computed_decomp = LabsQROM.single_controlled_res_decomp(
            num_bitstrings=num_data_points,
            size_bitstring=size_data_points,
            num_bit_flips=num_bit_flips,
            borrow_qubits=borrow,
            select_swap_depth=depth,
        )
        assert assert_decomp_equal(computed_decomp, expected_decomp)

    @staticmethod
    def ctrl_resources_data(index):
        """Store the expected resources used in the test_single_controlled_res_decomp method"""
        resources = []
        if index == 0:  # 1, 0, 10, 3, 15, 2, True
            allocate_sel = qre.Allocate(ceil_log2(5), "zero", True)
            allocate_swap = qre.Allocate(3, "any", True)
            h = qre.Hadamard.resource_rep()

            resources = (
                [allocate_sel, allocate_swap]
                + [qre.GateCount(h, 3 * 2)]
                + LabsQROM._single_ctrl_select_cost(5, 15, repeat=2)
                + [qre.Deallocate(allocated_register=allocate_sel)]
                + LabsQROM._single_ctrl_swap_cost(3, 1, repeat=4)
                + [qre.Deallocate(allocated_register=allocate_swap)]
            )
        if index == 1:  # 2, 1, 10, 3, 15, 2, True
            allocate_sel = qre.Allocate(ceil_log2(5), "zero", True)
            allocate_swap = qre.Allocate(3, "any", True)
            allocate_mcx_aux = qre.Allocate(2 - 1, "zero", True)

            x = qre.X.resource_rep()
            h = qre.Hadamard.resource_rep()
            l_elbow = qre.TemporaryAND.resource_rep()
            r_elbow = qre.Adjoint.resource_rep(l_elbow)

            resources = (
                [
                    qre.GateCount(x, 2),
                    allocate_mcx_aux,
                    qre.GateCount(l_elbow, 2 - 1),
                ]
                + [allocate_sel, allocate_swap]
                + [qre.GateCount(h, 3 * 2)]
                + LabsQROM._single_ctrl_select_cost(5, 15, repeat=2)
                + [qre.Deallocate(allocated_register=allocate_sel)]
                + LabsQROM._single_ctrl_swap_cost(3, 1, repeat=4)
                + [qre.Deallocate(allocated_register=allocate_swap)]
                + [
                    qre.GateCount(r_elbow, 2 - 1),
                    qre.Deallocate(allocated_register=allocate_mcx_aux),
                ]
            )
        if index == 2:  # 5, 3, 10, 3, 15, 2, True
            allocate_sel = qre.Allocate(ceil_log2(5), "zero", True)
            allocate_swap = qre.Allocate(3, "any", True)
            allocate_mcx_aux = qre.Allocate(5 - 1, "zero", True)

            x = qre.X.resource_rep()
            h = qre.Hadamard.resource_rep()
            l_elbow = qre.TemporaryAND.resource_rep()
            r_elbow = qre.Adjoint.resource_rep(l_elbow)

            resources = (
                [
                    qre.GateCount(x, 2 * 3),
                    allocate_mcx_aux,
                    qre.GateCount(l_elbow, 5 - 1),
                ]
                + [allocate_sel, allocate_swap]
                + [qre.GateCount(h, 3 * 2)]
                + LabsQROM._single_ctrl_select_cost(5, 15, repeat=2)
                + [qre.Deallocate(allocated_register=allocate_sel)]
                + LabsQROM._single_ctrl_swap_cost(3, 1, repeat=4)
                + [qre.Deallocate(allocated_register=allocate_swap)]
                + [
                    qre.GateCount(r_elbow, 5 - 1),
                    qre.Deallocate(allocated_register=allocate_mcx_aux),
                ]
            )
        return resources

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_data_points, size_data_points, num_bit_flips, depth, borrow, expected_res_index",
        (
            (1, 0, 10, 3, 15, 2, True, 0),
            (2, 1, 10, 3, 15, 2, True, 1),
            (5, 3, 10, 3, 15, 2, True, 2),
        ),
    )
    def test_controlled_res_decomp(
        self,
        num_ctrl_wires,
        num_zero_ctrl,
        num_data_points,
        size_data_points,
        num_bit_flips,
        depth,
        borrow,
        expected_res_index,
    ):
        """Test that the resources computed by single_controlled_res_decomp are correct."""
        expected_decomp = self.ctrl_resources_data(expected_res_index)
        computed_decomp = LabsQROM.controlled_resource_decomp(
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
            target_resource_params={
                "num_bitstrings": num_data_points,
                "size_bitstring": size_data_points,
                "num_bit_flips": num_bit_flips,
                "borrow_qubits": borrow,
                "select_swap_depth": depth,
            },
        )
        assert assert_decomp_equal(computed_decomp, expected_decomp)
