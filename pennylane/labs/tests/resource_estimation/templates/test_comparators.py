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

# pylint: disable=no-self-use,too-many-arguments,use-implicit-booleaness-not-comparison


class TestSingleQubitComparator:
    """Test the ResourceSingleQubitComparator class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = plre.ResourceSingleQubitComparator()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(plre.ResourceSingleQubitComparator, 4, {})
        assert plre.ResourceSingleQubitComparator.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(plre.ResourceTempAND)),
            GateCount(resource_rep(plre.ResourceCNOT), 4),
            GateCount(resource_rep(plre.ResourceX), 3),
        ]
        assert plre.ResourceSingleQubitComparator.resource_decomp() == expected


class TestTwoQubitComparator:
    """Test the ResourceTwoQubitComparator class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = plre.ResourceTwoQubitComparator()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(plre.ResourceTwoQubitComparator, 4, {})
        assert plre.ResourceTwoQubitComparator.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = [
            AllocWires(1),
            GateCount(resource_rep(plre.ResourceCSWAP), 2),
            GateCount(resource_rep(plre.ResourceCNOT), 3),
            GateCount(resource_rep(plre.ResourceX), 1),
            FreeWires(1),
        ]
        assert plre.ResourceTwoQubitComparator.resource_decomp() == expected


class TestIntegerComparator:
    """Test the ResourceIntegerComparator class."""

    @pytest.mark.parametrize(
        "value, register_size, geq",
        (
            (10, 3, True),
            (6, 4, False),
            (2, 2, False),
        ),
    )
    def test_resource_params(self, value, register_size, geq):
        """Test that the resource params are correct."""
        op = plre.ResourceIntegerComparator(value, register_size, geq)
        assert op.resource_params == {"value": value, "register_size": register_size, "geq": geq}

    @pytest.mark.parametrize(
        "value, register_size, geq",
        (
            (10, 3, True),
            (6, 4, False),
            (2, 2, False),
        ),
    )
    def test_resource_rep(self, value, register_size, geq):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceIntegerComparator,
            register_size + 1,
            {"value": value, "register_size": register_size, "geq": geq},
        )
        assert plre.ResourceIntegerComparator.resource_rep(value, register_size, geq) == expected

    @pytest.mark.parametrize(
        "value, register_size, geq, expected_res",
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
    def test_resources(self, value, register_size, geq, expected_res):
        """Test that the resources are correct."""
        assert (
            plre.ResourceIntegerComparator.resource_decomp(value, register_size, geq)
            == expected_res
        )


class TestRegisterComparator:
    """Test the ResourceRegisterComparator class."""

    @pytest.mark.parametrize(
        "first_register, second_register, geq",
        (
            (10, 10, True),
            (6, 4, False),
            (4, 6, False),
        ),
    )
    def test_resource_params(self, first_register, second_register, geq):
        """Test that the resource params are correct."""
        op = plre.ResourceRegisterComparator(first_register, second_register, geq)
        assert op.resource_params == {
            "first_register": first_register,
            "second_register": second_register,
            "geq": geq,
        }

    @pytest.mark.parametrize(
        "first_register, second_register, geq",
        (
            (10, 10, True),
            (6, 4, False),
            (4, 6, False),
        ),
    )
    def test_resource_rep(self, first_register, second_register, geq):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceRegisterComparator,
            first_register + second_register + 1,
            {"first_register": first_register, "second_register": second_register, "geq": geq},
        )
        assert (
            plre.ResourceRegisterComparator.resource_rep(first_register, second_register, geq)
            == expected
        )

    @pytest.mark.parametrize(
        "first_register, second_register, geq, expected_res",
        (
            (
                10,
                10,
                True,
                [
                    AllocWires(18),
                    GateCount(resource_rep(plre.ResourceTempAND), 18),
                    GateCount(resource_rep(plre.ResourceCNOT), 72),
                    GateCount(resource_rep(plre.ResourceX), 27),
                    GateCount(resource_rep(plre.ResourceSingleQubitComparator), 1),
                    FreeWires(18),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceTempAND)},
                        ),
                        18,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceCNOT)}
                        ),
                        72,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceX)}
                        ),
                        27,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceSingleQubitComparator)},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(plre.ResourceX), 1),
                    GateCount(resource_rep(plre.ResourceCNOT), 1),
                ],
            ),
            (
                6,
                4,
                False,
                [
                    AllocWires(6),
                    GateCount(resource_rep(plre.ResourceTempAND), 6),
                    GateCount(resource_rep(plre.ResourceCNOT), 24),
                    GateCount(resource_rep(plre.ResourceX), 9),
                    GateCount(resource_rep(plre.ResourceSingleQubitComparator), 1),
                    FreeWires(6),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceTempAND)},
                        ),
                        6,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceCNOT)}
                        ),
                        24,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceX)}
                        ),
                        9,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceSingleQubitComparator)},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 2},
                        ),
                        2,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                        ),
                        2,
                    ),
                ],
            ),
            (
                4,
                6,
                True,
                [
                    AllocWires(6),
                    GateCount(resource_rep(plre.ResourceTempAND), 6),
                    GateCount(resource_rep(plre.ResourceCNOT), 24),
                    GateCount(resource_rep(plre.ResourceX), 9),
                    GateCount(resource_rep(plre.ResourceSingleQubitComparator), 1),
                    FreeWires(6),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceTempAND)},
                        ),
                        6,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceCNOT)}
                        ),
                        24,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceX)}
                        ),
                        9,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceSingleQubitComparator)},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 2},
                        ),
                        2,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceMultiControlledX,
                            {"num_ctrl_wires": 2, "num_ctrl_values": 1},
                        ),
                        2,
                    ),
                    GateCount(resource_rep(plre.ResourceX), 1),
                ],
            ),
        ),
    )
    def test_resources(self, first_register, second_register, geq, expected_res):
        """Test that the resources are correct."""
        assert (
            plre.ResourceRegisterComparator.resource_decomp(first_register, second_register, geq)
            == expected_res
        )
