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

import pennylane.estimator as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.estimator.wires_manager import Allocate, Deallocate

# pylint: disable=no-self-use,too-many-arguments,use-implicit-booleaness-not-comparison


class TestSingleQubitComparator:
    """Test the ResourceSingleQubitComparator class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 3"):
            qre.SingleQubitComparator(wires=[0, 1, 2])

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = qre.SingleQubitComparator()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.SingleQubitComparator, 4, {})
        assert qre.SingleQubitComparator.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(qre.TemporaryAND)),
            GateCount(resource_rep(qre.CNOT), 4),
            GateCount(resource_rep(qre.X), 3),
        ]
        assert qre.SingleQubitComparator.resource_decomp() == expected


class TestTwoQubitComparator:
    """Test the ResourceTwoQubitComparator class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 3"):
            qre.TwoQubitComparator(wires=[0, 1, 2])

    def test_resource_params(self):
        """Test that the resource params are correct."""
        op = qre.TwoQubitComparator()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.TwoQubitComparator, 4, {})
        assert qre.TwoQubitComparator.resource_rep() == expected

    def test_resources(self):
        """Test that the resources are correct."""
        expected = [
            Allocate(1),
            GateCount(resource_rep(qre.CSWAP), 2),
            GateCount(resource_rep(qre.CNOT), 3),
            GateCount(resource_rep(qre.X), 1),
            Deallocate(1),
        ]
        assert qre.TwoQubitComparator.resource_decomp() == expected


class TestIntegerComparator:
    """Test the ResourceIntegerComparator class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 3"):
            qre.IntegerComparator(10, 3, wires=[0, 1, 2])

    def test_init_no_register_size(self):
        """Test that we can instantiate the operator without providing register_size"""
        op = qre.IntegerComparator(value=10, geq=True, wires=range(3))
        assert op.resource_params == {"value": 10, "register_size": 2, "geq": True}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and register_size are both not provided"""
        with pytest.raises(ValueError, match="Must provide atleast one of"):
            qre.IntegerComparator(value=3)

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
        op = qre.IntegerComparator(value, register_size, geq)
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
        expected = qre.CompressedResourceOp(
            qre.IntegerComparator,
            register_size + 1,
            {"value": value, "register_size": register_size, "geq": geq},
        )
        assert qre.IntegerComparator.resource_rep(value, register_size, geq) == expected

    @pytest.mark.parametrize(
        "value, register_size, geq, expected_res",
        (
            (10, 3, True, []),
            (10, 3, False, [GateCount(resource_rep(qre.X))]),
            (0, 3, True, [GateCount(resource_rep(qre.X))]),
            (
                3,
                2,
                True,
                [
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 2, "num_zero_ctrl": 0},
                        ),
                        1,
                    ),
                ],
            ),
            (
                10,
                4,
                True,
                [
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 2, "num_zero_ctrl": 1},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 4, "num_zero_ctrl": 1},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 4, "num_zero_ctrl": 0},
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
                    GateCount(resource_rep(qre.X), 6),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 2, "num_zero_ctrl": 0},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 3, "num_zero_ctrl": 0},
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
                    GateCount(resource_rep(qre.X), 2),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 1, "num_zero_ctrl": 0},
                        ),
                        1,
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, value, register_size, geq, expected_res):
        """Test that the resources are correct."""
        assert qre.IntegerComparator.resource_decomp(value, register_size, geq) == expected_res


class TestRegisterComparator:
    """Test the ResourceRegisterComparator class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 21 wires, got 3"):
            qre.RegisterComparator(10, 10, wires=[0, 1, 2])

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
        op = qre.RegisterComparator(first_register, second_register, geq)
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
        expected = qre.CompressedResourceOp(
            qre.RegisterComparator,
            first_register + second_register + 1,
            {"first_register": first_register, "second_register": second_register, "geq": geq},
        )
        assert qre.RegisterComparator.resource_rep(first_register, second_register, geq) == expected

    @pytest.mark.parametrize(
        "first_register, second_register, geq, expected_res",
        (
            (
                10,
                10,
                True,
                [
                    Allocate(18),
                    GateCount(resource_rep(qre.TemporaryAND), 18),
                    GateCount(resource_rep(qre.CNOT), 72),
                    GateCount(resource_rep(qre.X), 27),
                    GateCount(resource_rep(qre.SingleQubitComparator), 1),
                    Deallocate(18),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.TemporaryAND)},
                        ),
                        18,
                    ),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.CNOT)}),
                        72,
                    ),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.X)}),
                        27,
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.SingleQubitComparator)},
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.X), 1),
                    GateCount(resource_rep(qre.CNOT), 1),
                ],
            ),
            (
                6,
                4,
                False,
                [
                    Allocate(6),
                    GateCount(resource_rep(qre.TemporaryAND), 6),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.X), 9),
                    GateCount(resource_rep(qre.SingleQubitComparator), 1),
                    Deallocate(6),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.TemporaryAND)},
                        ),
                        6,
                    ),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.CNOT)}),
                        24,
                    ),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.X)}),
                        9,
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.SingleQubitComparator)},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 2, "num_zero_ctrl": 2},
                        ),
                        2,
                    ),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 2, "num_zero_ctrl": 1},
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
                    Allocate(6),
                    GateCount(resource_rep(qre.TemporaryAND), 6),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.X), 9),
                    GateCount(resource_rep(qre.SingleQubitComparator), 1),
                    Deallocate(6),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.TemporaryAND)},
                        ),
                        6,
                    ),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.CNOT)}),
                        24,
                    ),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.X)}),
                        9,
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.SingleQubitComparator)},
                        ),
                        1,
                    ),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 2, "num_zero_ctrl": 2},
                        ),
                        2,
                    ),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {"num_ctrl_wires": 2, "num_zero_ctrl": 1},
                        ),
                        2,
                    ),
                    GateCount(resource_rep(qre.X), 1),
                ],
            ),
        ),
    )
    def test_resources(self, first_register, second_register, geq, expected_res):
        """Test that the resources are correct."""
        assert (
            qre.RegisterComparator.resource_decomp(first_register, second_register, geq)
            == expected_res
        )
