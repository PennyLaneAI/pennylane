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
Tests for the state preparation subroutines resource operators.
"""

import re

import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.labs.tests.estimator_beta.utils import decomp_equal

# pylint: disable=no-self-use,too-many-arguments, undefined-variable, unused-variable


class TestMottonenStatePreparation:
    """Test the MottonenStatePreparation class"""

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources(self, num_wires):
        """Test that the resources are correct"""
        rz = resource_rep(qre.RZ)
        cnot = resource_rep(qre.CNOT)

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        expected = [GateCount(rz, r_count), GateCount(cnot, cnot_count)]

        assert decomp_equal(qre.MottonenStatePreparation.resource_decomp(num_wires), expected)

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct"""
        op = qre.MottonenStatePreparation(num_wires)

        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.MottonenStatePreparation,
            num_wires,
            {"num_wires": num_wires},
        )
        assert expected == qre.MottonenStatePreparation.resource_rep(num_wires)


class TestCosineWindow:
    """Test the CosineWindow class"""

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources(self, num_wires):
        """Test that the resources are correct"""
        hadamard = resource_rep(qre.Hadamard)
        rz = resource_rep(qre.RZ)
        iqft = resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": resource_rep(qre.QFT, {"num_wires": num_wires})},
        )
        phase_shift = resource_rep(qre.PhaseShift)

        expected = [
            GateCount(hadamard, 1),
            GateCount(rz, 1),
            GateCount(iqft, 1),
            GateCount(phase_shift, num_wires),
        ]

        assert decomp_equal(qre.CosineWindow.resource_decomp(num_wires), expected)

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct"""
        op = qre.CosineWindow(num_wires)

        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.CosineWindow,
            num_wires,
            {"num_wires": num_wires},
        )
        assert expected == qre.CosineWindow.resource_rep(num_wires)


class TestSumOfSlatersPrep:
    """Test the SumOfSlatersPrep class"""

    @pytest.mark.parametrize(
        "num_coeffs, num_wires, num_bits, stateprep_op, select_swap_depth, expected_resources",
        [
            (
                1,
                4,
                None,
                None,
                None,
                [GateCount(resource_rep(qre.BasisState, {"num_wires": 4}), 1)],
            ),
            (
                40,
                16,
                8,
                None,
                1,
                [
                    alloc_reg := qre.Allocate(6, state="zero", restored=True),
                    GateCount(resource_rep(qre.MottonenStatePreparation, {"num_wires": 6}), 1),
                    GateCount(
                        resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 40,
                                "size_bitstring": 16,
                                "borrow_qubits": False,
                                "select_swap_depth": 1,
                            },
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.X), 320),
                    GateCount(resource_rep(qre.CNOT), 100),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 8, "num_zero_ctrl": 0}
                        ),
                        39,
                    ),
                    qre.Deallocate(allocated_register=alloc_reg),
                ],
            ),
            (
                56,
                20,
                None,
                None,
                1,
                [
                    alloc_reg := qre.Allocate(6, state="zero", restored=True),
                    GateCount(resource_rep(qre.MottonenStatePreparation, {"num_wires": 6}), 1),
                    GateCount(
                        resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 56,
                                "size_bitstring": 20,
                                "borrow_qubits": False,
                                "select_swap_depth": 1,
                            },
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.X), 616),
                    GateCount(resource_rep(qre.CNOT), 156),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 11, "num_zero_ctrl": 0}
                        ),
                        55,
                    ),
                    qre.Deallocate(allocated_register=alloc_reg),
                ],
            ),
            (
                100,
                10,
                None,
                resource_rep(qre.QROMStatePreparation, {"num_state_qubits": 7}),
                2,
                [
                    alloc_reg := qre.Allocate(7, state="zero", restored=True),
                    GateCount(resource_rep(qre.QROMStatePreparation, {"num_state_qubits": 7}), 1),
                    GateCount(
                        resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 100,
                                "size_bitstring": 10,
                                "borrow_qubits": False,
                                "select_swap_depth": 2,
                            },
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.X), 1000),
                    GateCount(resource_rep(qre.CNOT), 316),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 10, "num_zero_ctrl": 0}
                        ),
                        99,
                    ),
                    qre.Deallocate(allocated_register=alloc_reg),
                ],
            ),
            (
                16,
                20,
                15,
                None,
                1,
                [
                    alloc_reg := qre.Allocate(4, state="zero", restored=True),
                    GateCount(resource_rep(qre.MottonenStatePreparation, {"num_wires": 4}), 1),
                    GateCount(
                        resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 16,
                                "size_bitstring": 20,
                                "borrow_qubits": False,
                                "select_swap_depth": 1,
                            },
                        ),
                        1,
                    ),
                    alloc_reg_2 := qre.Allocate(7, state="zero", restored=True),
                    GateCount(resource_rep(qre.CNOT), 280),
                    GateCount(resource_rep(qre.X), 112),
                    GateCount(resource_rep(qre.CNOT), 32),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 7, "num_zero_ctrl": 0}
                        ),
                        15,
                    ),
                    qre.Deallocate(allocated_register=alloc_reg_2),
                    qre.Deallocate(allocated_register=alloc_reg),
                ],
            ),
        ],
    )
    def test_resources(
        self, num_coeffs, num_wires, num_bits, stateprep_op, select_swap_depth, expected_resources
    ):
        """Test that the resources are correct"""
        res = qre.SumOfSlatersPrep.resource_decomp(
            num_coeffs, num_wires, num_bits, stateprep_op, select_swap_depth
        )
        assert decomp_equal(res, expected_resources)

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = qre.SumOfSlatersPrep(num_coeffs=100, num_wires=10, num_bits=5)

        assert op.resource_params == {
            "num_wires": 10,
            "num_coeffs": 100,
            "num_bits": 5,
            "stateprep_cmpr_op": None,
            "select_swap_depth": None,
        }

    def test_resource_rep(self):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.SumOfSlatersPrep,
            10,
            {
                "num_wires": 10,
                "num_coeffs": 100,
                "num_bits": 5,
                "stateprep_cmpr_op": None,
                "select_swap_depth": None,
            },
        )
        assert expected == qre.SumOfSlatersPrep.resource_rep(
            num_coeffs=100, num_wires=10, num_bits=5
        )

    def test_invalid_num_coeffs(self):
        """Test that an error is raised if num_coeffs is greater than 2^num_wires"""
        with pytest.raises(
            ValueError,
            match=re.escape("Number of coefficients 17 cannot be greater than 2^num_wires, 16."),
        ):
            qre.SumOfSlatersPrep.resource_rep(num_coeffs=17, num_wires=4)

        with pytest.raises(
            ValueError,
            match=re.escape("Number of coefficients 17 cannot be greater than 2^num_wires, 16."),
        ):
            qre.SumOfSlatersPrep.resource_rep(num_coeffs=17, num_wires=4)

    def test_invalid_num_bits(self):
        """Test that an error is raised if num_bits is greater than num_wires"""
        with pytest.raises(
            ValueError, match=re.escape("num_bits 8 cannot be greater than num_wires, 4.")
        ):
            qre.SumOfSlatersPrep.resource_rep(num_coeffs=10, num_wires=4, num_bits=8)

        with pytest.raises(
            ValueError, match=re.escape("num_bits 8 cannot be greater than num_wires, 4.")
        ):
            qre.SumOfSlatersPrep.resource_rep(num_coeffs=10, num_wires=4, num_bits=8)
