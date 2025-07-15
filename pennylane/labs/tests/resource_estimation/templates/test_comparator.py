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
Tests for quantum algorithmic subroutines resource operators.
"""
import pytest

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation import AllocWires, FreeWires, GateCount, resource_rep

# pylint: disable=no-self-use,too-many-arguments


class TestSingleQubitCompare:
    """Test the ResourceSingleQubitCompare class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = plre.ResourceSingleQubitCompare()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(plre.ResourceSingleQubitCompare, {})
        assert plre.ResourceSingleQubitCompare.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(plre.ResourceTempAND)),
            GateCount(resource_rep(plre.ResourceCNOT), 4),
            GateCount(resource_rep(plre.ResourceX), 3),
        ]
        assert plre.ResourceSingleQubitCompare.resource_decomp() == expected


class TestTwoQubitCompare:
    """Test the ResourceTwoQubitCompare class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = plre.ResourceTwoQubitCompare()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(plre.ResourceTwoQubitCompare, {})
        assert plre.ResourceTwoQubitCompare.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = [
            AllocWires(1),
            GateCount(resource_rep(plre.ResourceCSWAP), 2),
            GateCount(resource_rep(plre.ResourceCNOT), 3),
            FreeWires(1),
        ]
        assert plre.ResourceTwoQubitCompare.resource_decomp() == expected


class TestIntegerComparator:
    """Test the ResourceIntegerComparator class."""

    @pytest.mark.parametrize(
        "val, register_size, geq",
        (
            (10, 3, True),
            (6, 4, False),
            (2, 2, False),
        ),
    )
    def test_resource_params(self, val, register_size, geq):
        """Test that the resource params are correct."""
        op = plre.ResourceIntegerComparator(val, register_size, geq)
        assert op.resource_params == {"val": val, "register_size": register_size, "geq": geq}

    @pytest.mark.parametrize(
        "val, register_size, geq",
        (
            (10, 3, True),
            (6, 4, False),
            (2, 2, False),
        ),
    )
    def test_resource_rep(self, val, register_size, geq):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceIntegerComparator, {"val": val, "register_size": register_size, "geq": geq}
        )
        assert plre.ResourceIntegerComparator.resource_rep(val, register_size, geq) == expected

    @pytest.mark.parametrize(
        "val, register_size, geq, expected_res",
        (
            (10, 3, True, []),
            (10, 3, False, [GateCount(resource_rep(plre.ResourceX))]),
            (0, 3, True, [GateCount(resource_rep(plre.ResourceX))]),
            (
                10,
                4,
                True,
                [
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 4, "num_ctrl_values": 1},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 4, "num_ctrl_values": 0},
                        ),
                        1,
                    ),
                ],
            ),
            (0, 3, False, []),
            (
                6,
                4,
                False,
                [
                    GateCount(resource_rep(plre.ResourceX), 6),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 0},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 3, "num_ctrl_values": 0},
                        ),
                        1,
                    ),
                ],
            ),
            (
                2,
                2,
                False,
                [
                    GateCount(resource_rep(plre.ResourceX), 2),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 1, "num_ctrl_values": 0},
                        ),
                        1,
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, val, register_size, geq, expected_res):
        """Test that the resources are correct."""
        assert (
            plre.ResourceIntegerComparator.resource_decomp(val, register_size, geq) == expected_res
        )


class TestRegisterComparator:
    """Test the ResourceRegisterComparator class."""

    @pytest.mark.parametrize(
        "a_num_qubits, b_num_qubits, geq",
        (
            (10, 10, True),
            (6, 4, False),
            (4, 6, False),
        ),
    )
    def test_resource_params(self, a_num_qubits, b_num_qubits, geq):
        """Test that the resource params are correct."""
        op = plre.ResourceRegisterComparator(a_num_qubits, b_num_qubits, geq)
        assert op.resource_params == {
            "a_num_qubits": a_num_qubits,
            "b_num_qubits": b_num_qubits,
            "geq": geq,
        }

    @pytest.mark.parametrize(
        "a_num_qubits, b_num_qubits, geq",
        (
            (10, 10, True),
            (6, 4, False),
            (4, 6, False),
        ),
    )
    def test_resource_rep(self, a_num_qubits, b_num_qubits, geq):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceRegisterComparator,
            {"a_num_qubits": a_num_qubits, "b_num_qubits": b_num_qubits, "geq": geq},
        )
        assert (
            plre.ResourceRegisterComparator.resource_rep(a_num_qubits, b_num_qubits, geq)
            == expected
        )

    @pytest.mark.parametrize(
        "a_num_qubits, b_num_qubits, geq, expected_res",
        (
            (
                10,
                10,
                True,
                [
                    AllocWires(20),
                    GateCount(resource_rep(plre.ResourceTwoQubitCompare), 9),
                    GateCount(resource_rep(plre.ResourceSingleQubitCompare), 1),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceTwoQubitCompare)},
                        ),
                        9,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceSingleQubitCompare)},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceX), 1),
                    GateCount(resource_rep(plre.ResourceCNOT), 1),
                    FreeWires(20),
                ],
            ),
            (
                6,
                4,
                False,
                [
                    AllocWires(8),
                    GateCount(resource_rep(plre.ResourceTwoQubitCompare), 3),
                    GateCount(resource_rep(plre.ResourceSingleQubitCompare), 1),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceTwoQubitCompare)},
                        ),
                        3,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceSingleQubitCompare)},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 2},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceMultiControlledX,
                                    {"num_ctrl_wires": 2, "num_ctrl_values": 2},
                                )
                            },
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceMultiControlledX,
                                    {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                                )
                            },
                        ),
                        1,
                    ),
                    FreeWires(8),
                ],
            ),
            (
                4,
                6,
                True,
                [
                    AllocWires(8),
                    GateCount(resource_rep(plre.ResourceTwoQubitCompare), 3),
                    GateCount(resource_rep(plre.ResourceSingleQubitCompare), 1),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceTwoQubitCompare)},
                        ),
                        3,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceSingleQubitCompare)},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 2},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceMultiControlledX,
                                    {"num_ctrl_wires": 2, "num_ctrl_values": 2},
                                )
                            },
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": resource_rep(
                                    plre.ResourceMultiControlledX,
                                    {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                                )
                            },
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceX), 1),
                    FreeWires(8),
                ],
            ),
        ),
    )
    def test_resources(self, a_num_qubits, b_num_qubits, geq, expected_res):
        """Test that the resources are correct."""
        assert (
            plre.ResourceRegisterComparator.resource_decomp(a_num_qubits, b_num_qubits, geq)
            == expected_res
        )
