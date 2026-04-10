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

import pennylane as qml
import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, ResourceConfig, resource_rep


# pylint: disable=too-few-public-methods, too-many-arguments, no-self-use
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
            kwargs = config.resource_op_precisions[qml.estimator.SelectPauliRot]
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
    """Test the resource QROM class."""

    def test_select_swap_depth_errors(self):
        """Test that the correct error is raised when invalid values of
        select_swap_depth are provided.
        """
        select_swap_depth = "Not A Valid Input"
        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            qre.QROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            qre.QROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

        select_swap_depth = 3
        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            qre.QROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            qre.QROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, borrow",
        (
            (10, 3, 15, None, True),
            (100, 5, 50, 2, False),
            (12, 2, 5, 1, True),
        ),
    )
    def test_resource_params(
        self, num_data_points, size_data_points, num_bit_flips, depth, borrow
    ):
        """Test that the resource params are correct."""
        if depth is None:
            op = qre.QROM(num_data_points, size_data_points)
        else:
            op = qre.QROM(num_data_points, size_data_points, num_bit_flips, borrow, depth)

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
            qre.QROM,
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
            qre.QROM.resource_rep(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                borrow_qubits=borrow,
                select_swap_depth=depth,
            )
            == expected
        )

    # pylint: disable=protected-access
    def test_t_select_swap_width(self):
        """Test that the private function doesn't give negative or
        fractional values for the depth"""
        num_bitstrings = 8
        size_bitstring = 17

        opt_width = qre.QROM._t_optimized_select_swap_width(
            num_bitstrings,
            size_bitstring,
        )
        assert opt_width == 1

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                10,
                3,
                15,
                None,
                True,
                [
                    qre.Allocate(5),
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.X.resource_rep(), 14),
                    GateCount(qre.CNOT.resource_rep(), 36),
                    GateCount(qre.TemporaryAND.resource_rep(), 6),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        6,
                    ),
                    qre.Deallocate(2),
                    GateCount(qre.CSWAP.resource_rep(), 12),
                    qre.Deallocate(3),
                ],
            ),
            (
                100,
                5,
                50,
                2,
                False,
                [
                    qre.Allocate(10),
                    GateCount(qre.X.resource_rep(), 97),
                    GateCount(qre.CNOT.resource_rep(), 98),
                    GateCount(qre.TemporaryAND.resource_rep(), 48),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        48,
                    ),
                    qre.Deallocate(5),
                    GateCount(qre.CSWAP.resource_rep(), 5),
                    GateCount(qre.X.resource_rep(), 5),
                    qre.Deallocate(5),
                ],
            ),
            (
                12,
                2,
                5,
                1,
                True,
                [
                    qre.Allocate(3),
                    GateCount(qre.X.resource_rep(), 21),
                    GateCount(qre.CNOT.resource_rep(), 15),
                    GateCount(qre.TemporaryAND.resource_rep(), 10),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        10,
                    ),
                    qre.Deallocate(3),
                ],
            ),
            (
                12,
                2,
                5,
                128,  # This will get truncated to 16 as the max depth
                False,
                [
                    qre.Allocate(30),
                    GateCount(qre.X.resource_rep(), 5),
                    GateCount(qre.CSWAP.resource_rep(), 30),
                    GateCount(qre.X.resource_rep(), 30),
                    qre.Deallocate(30),
                ],
            ),
            (
                12,
                2,
                5,
                16,
                True,
                [
                    qre.Allocate(30),
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(qre.X.resource_rep(), 10),
                    GateCount(qre.CSWAP.resource_rep(), 120),
                    qre.Deallocate(30),
                ],
            ),
        ),
    )
    def test_resources(
        self, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res
    ):
        """Test that the resources are correct."""
        assert (
            qre.QROM.resource_decomp(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                restored=restored,
                select_swap_depth=depth,
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                10,
                3,
                15,
                None,
                True,
                [
                    qre.Allocate(6),
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 38),
                    GateCount(qre.TemporaryAND.resource_rep(), 8),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        8,
                    ),
                    qre.Deallocate(3),
                    qre.Allocate(1),
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.CSWAP.resource_rep(), 12),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                    qre.Deallocate(3),
                ],
            ),
            (
                10,
                3,
                15,
                1,
                True,
                [
                    qre.Allocate(4),
                    GateCount(qre.X.resource_rep(), 18),
                    GateCount(qre.CNOT.resource_rep(), 24),
                    GateCount(qre.TemporaryAND.resource_rep(), 9),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        9,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                12,
                2,
                5,
                16,
                True,
                [
                    qre.Allocate(30),
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(qre.X.resource_rep(), 10),
                    qre.Allocate(1),
                    GateCount(qre.TemporaryAND.resource_rep(), 4),
                    GateCount(qre.CSWAP.resource_rep(), 120),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        4,
                    ),
                    qre.Deallocate(1),
                    qre.Deallocate(30),
                ],
            ),
        ),
    )
    def test_single_controlled_res_decomp(
        self, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res
    ):
        """Test that the resources computed by single_controlled_res_decomp are correct."""
        assert (
            qre.QROM.single_controlled_res_decomp(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                restored=restored,
                select_swap_depth=depth,
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                1,
                0,
                10,
                3,
                15,
                None,
                True,
                [
                    qre.Allocate(6),
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.X.resource_rep(), 16),
                    GateCount(qre.CNOT.resource_rep(), 38),
                    GateCount(qre.TemporaryAND.resource_rep(), 8),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        8,
                    ),
                    qre.Deallocate(3),
                    qre.Allocate(1),
                    GateCount(qre.TemporaryAND.resource_rep(), 1),
                    GateCount(qre.CSWAP.resource_rep(), 12),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                    qre.Deallocate(3),
                ],
            ),
            (
                2,
                1,
                10,
                3,
                15,
                1,
                True,
                [
                    GateCount(qre.X.resource_rep(), 2),
                    qre.Allocate(1),
                    GateCount(qre.MultiControlledX.resource_rep(2, 0), 1),
                    qre.Allocate(4),
                    GateCount(qre.X.resource_rep(), 18),
                    GateCount(qre.CNOT.resource_rep(), 24),
                    GateCount(qre.TemporaryAND.resource_rep(), 9),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        9,
                    ),
                    qre.Deallocate(4),
                    GateCount(qre.MultiControlledX.resource_rep(2, 0), 1),
                    qre.Deallocate(1),
                ],
            ),
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
        restored,
        expected_res,
    ):
        """Test that the resources computed by single_controlled_res_decomp are correct."""
        assert (
            qre.QROM.controlled_resource_decomp(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                target_resource_params={
                    "num_bitstrings": num_data_points,
                    "size_bitstring": size_data_points,
                    "num_bit_flips": num_bit_flips,
                    "restored": restored,
                    "select_swap_depth": depth,
                },
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res",
        (
            (
                10,
                3,
                15,
                None,
                True,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    qre.Allocate(4),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CSWAP.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CZ.resource_rep(), 2),
                    GateCount(qre.CNOT.resource_rep(), 2),
                    GateCount(qre.X.resource_rep(), 14),
                    GateCount(qre.CNOT.resource_rep(), 16),
                    GateCount(qre.TemporaryAND.resource_rep(), 6),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        6,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                100,
                5,
                50,
                2,
                False,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5),
                    qre.Allocate(7),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(qre.CSWAP.resource_rep(), 1),
                    GateCount(qre.Hadamard.resource_rep(), 1),
                    GateCount(qre.CZ.resource_rep(), 1),
                    GateCount(qre.CNOT.resource_rep(), 1),
                    GateCount(qre.X.resource_rep(), 97),
                    GateCount(qre.CNOT.resource_rep(), 98),
                    GateCount(qre.TemporaryAND.resource_rep(), 48),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        48,
                    ),
                    qre.Deallocate(7),
                ],
            ),
            (
                12,
                2,
                5,
                1,
                True,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    qre.Allocate(4),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CSWAP.resource_rep(), 0),
                    GateCount(qre.Hadamard.resource_rep(), 0),
                    GateCount(qre.CZ.resource_rep(), 0),
                    GateCount(qre.CNOT.resource_rep(), 0),
                    GateCount(qre.X.resource_rep(), 21),
                    GateCount(qre.CNOT.resource_rep(), 16),
                    GateCount(qre.TemporaryAND.resource_rep(), 10),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TemporaryAND.resource_rep(),
                        ),
                        10,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                12,
                2,
                5,
                128,  # This will get truncated to 16 as the max depth
                False,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    qre.Allocate(16),
                    GateCount(qre.X.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 32),
                    GateCount(qre.CSWAP.resource_rep(), 15),
                    GateCount(qre.Hadamard.resource_rep(), 15),
                    GateCount(qre.CZ.resource_rep(), 15),
                    GateCount(qre.CNOT.resource_rep(), 15),
                    GateCount(qre.X.resource_rep(), 8),
                    qre.Deallocate(16),
                ],
            ),
            (
                12,
                2,
                5,
                16,
                True,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    qre.Allocate(16),
                    GateCount(qre.Z.resource_rep(), 2),
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.CSWAP.resource_rep(), 30),
                    GateCount(qre.Hadamard.resource_rep(), 30),
                    GateCount(qre.CZ.resource_rep(), 30),
                    GateCount(qre.CNOT.resource_rep(), 30),
                    GateCount(qre.X.resource_rep(), 16),
                    qre.Deallocate(16),
                ],
            ),
        ),
    )
    def test_adjoint_resources(
        self, num_data_points, size_data_points, num_bit_flips, depth, restored, expected_res
    ):
        """Test that the resources are correct."""

        assert (
            qre.QROM.adjoint_resource_decomp(
                {
                    "num_bitstrings": num_data_points,
                    "size_bitstring": size_data_points,
                    "num_bit_flips": num_bit_flips,
                    "restored": restored,
                    "select_swap_depth": depth,
                }
            )
            == expected_res
        )

    @pytest.mark.parametrize(
        "num_data_points, output_size, restored, depth",
        (
            (100, 10, False, 2),
            (100, 2, False, 4),
            (12, 1, False, 1),
            (12, 3, True, 1),
            (160, 8, True, 2),
        ),
    )
    def test_toffoli_counts(self, num_data_points, output_size, restored, depth):
        """Test that the Toffoli counts are correct compared to arXiv:1902.02134."""

        qrom = qre.Adjoint(
            qre.QROM(
                num_bitstrings=num_data_points,
                size_bitstring=output_size,
                restored=restored,
                select_swap_depth=depth,
            )
        )
        resources = qre.estimate(qrom)

        toffoli_count = int(math.ceil(num_data_points / depth)) + depth - 3
        if restored and depth > 1:
            toffoli_count *= 2

        assert resources.gate_counts["Toffoli"] == toffoli_count
