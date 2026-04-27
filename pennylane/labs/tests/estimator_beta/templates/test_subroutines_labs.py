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
from pennylane.estimator import GateCount, ResourceConfig, resource_rep
from pennylane.labs.estimator_beta.templates import LabsQROM
from pennylane.math import ceil_log2

# pylint: disable=too-few-public-methods, too-many-arguments, no-self-use, protected-access


def _test_decomp_equal(decomp1, decomp2):
    if len(decomp1) != len(decomp2):
        return False

    for op1, op2 in zip(decomp1, decomp2):
        if isinstance(op1, (qre.Allocate, qre.Deallocate)):
            ops_equal = op1.equal(op2)
        else:
            ops_equal = op1 == op2

        if not ops_equal:
            return False

    return True


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
            config = ResourceConfig()
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
        expected_num_wires = size_data_points + qml.math.ceil_log2(num_data_points)
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
        assert _test_decomp_equal(computed, expected)

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
        assert _test_decomp_equal(computed, expected)

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
        assert _test_decomp_equal(computed, expected)

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
        assert _test_decomp_equal(computed, expected)

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
        assert _test_decomp_equal(computed_decomp, expected_decomp)

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
        assert _test_decomp_equal(computed_decomp, expected_decomp)

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
        assert _test_decomp_equal(computed_decomp, expected_decomp)
