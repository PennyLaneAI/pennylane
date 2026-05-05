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

import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.labs.estimator_beta.wires_manager import Allocate, Deallocate

# pylint: disable=no-self-use


class TestOutOfPlaceIntegerComparator:
    """Test the OutOfPlaceIntegerComparator class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 4 wires, got 3"):
            qre.OutOfPlaceIntegerComparator(10, 3, wires=[0, 1, 2])

    def test_init_no_register_size(self):
        """Test that we can instantiate the operator without providing register_size"""
        op = qre.OutOfPlaceIntegerComparator(value=10, geq=True, wires=range(3))
        assert op.resource_params == {"value": 10, "register_size": 2, "geq": True}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and register_size are both not provided"""
        with pytest.raises(ValueError, match="Must provide at least one of"):
            qre.OutOfPlaceIntegerComparator(value=3)

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
        op = qre.OutOfPlaceIntegerComparator(value, register_size, geq)
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
            qre.OutOfPlaceIntegerComparator,
            register_size + 1,
            {"value": value, "register_size": register_size, "geq": geq},
        )
        assert qre.OutOfPlaceIntegerComparator.resource_rep(value, register_size, geq) == expected

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
                    Allocate(1),
                    GateCount(resource_rep(qre.TemporaryAND), 1),
                    GateCount(resource_rep(qre.CNOT), 2),
                    GateCount(resource_rep(qre.X), 6),
                    GateCount(resource_rep(qre.X), 1),
                ],
            ),
            (
                10,
                4,
                True,
                [
                    Allocate(3),
                    GateCount(resource_rep(qre.TemporaryAND), 3),
                    GateCount(resource_rep(qre.CNOT), 4),
                    GateCount(resource_rep(qre.X), 6),
                    GateCount(resource_rep(qre.X), 1),
                ],
            ),
            (0, 3, False, []),
            (
                6,
                4,
                False,
                [
                    Allocate(3),
                    GateCount(resource_rep(qre.TemporaryAND), 3),
                    GateCount(resource_rep(qre.CNOT), 4),
                    GateCount(resource_rep(qre.X), 6),
                ],
            ),
            (
                2,
                2,
                False,
                [
                    Allocate(1),
                    GateCount(resource_rep(qre.TemporaryAND), 1),
                    GateCount(resource_rep(qre.CNOT), 2),
                    GateCount(resource_rep(qre.X), 2),
                ],
            ),
        ),
    )
    def test_resources(self, value, register_size, geq, expected_res):
        """Test that the resources are correct."""
        result = qre.OutOfPlaceIntegerComparator.resource_decomp(value, register_size, geq)
        for r, e in zip(result, expected_res, strict=True):
            if hasattr(r, "equal"):
                assert r.equal(e)
            else:
                assert r == e

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
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        1,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2),
                    GateCount(resource_rep(qre.X), 6),
                    GateCount(resource_rep(qre.X), 1),
                    Deallocate(1),
                ],
            ),
            (
                10,
                4,
                True,
                [
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        3,
                    ),
                    GateCount(resource_rep(qre.CNOT), 4),
                    GateCount(resource_rep(qre.X), 6),
                    GateCount(resource_rep(qre.X), 1),
                    Deallocate(3),
                ],
            ),
            (0, 3, False, []),
            (
                6,
                4,
                False,
                [
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        3,
                    ),
                    GateCount(resource_rep(qre.CNOT), 4),
                    GateCount(resource_rep(qre.X), 6),
                    Deallocate(3),
                ],
            ),
            (
                2,
                2,
                False,
                [
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        1,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2),
                    GateCount(resource_rep(qre.X), 2),
                    Deallocate(1),
                ],
            ),
        ),
    )
    def test_adjoint_resources(self, value, register_size, geq, expected_res):
        """Test that the resources are correct for the adjoint of the operator."""
        result = qre.OutOfPlaceIntegerComparator.adjoint_resource_decomp(
            {"value": value, "register_size": register_size, "geq": geq}
        )
        for r, e in zip(result, expected_res, strict=True):
            if hasattr(r, "equal"):
                assert r.equal(e)
            else:
                assert r == e


class TestRegisterEquality:
    """Test the RegisterEquality class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 21 wires, got 3"):
            qre.RegisterEquality(10, wires=[0, 1, 2])

    def test_init_no_register_size(self):
        """Test that we can instantiate the operator without providing register_size"""
        op = qre.RegisterEquality(wires=range(9))
        assert op.resource_params == {"register_size": 4}

    def test_init_raises_error(self):
        """Test that an error is raised when wires and register_size are both not provided"""
        with pytest.raises(ValueError, match="Must provide at least one of"):
            qre.RegisterEquality(register_size=None)

    @pytest.mark.parametrize(
        "register_size",
        (
            10,
            6,
            2,
        ),
    )
    def test_resource_params(self, register_size):
        """Test that the resource params are correct."""
        op = qre.RegisterEquality(register_size)
        assert op.resource_params == {"register_size": register_size}

    @pytest.mark.parametrize(
        "register_size",
        (
            10,
            6,
            2,
        ),
    )
    def test_resource_rep(self, register_size):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.RegisterEquality,
            2 * register_size + 1,
            {"register_size": register_size},
        )
        assert qre.RegisterEquality.resource_rep(register_size) == expected

    @pytest.mark.parametrize(
        "register_size, expected_res",
        (
            (0, []),
            (1, [GateCount(resource_rep(qre.X)), GateCount(resource_rep(qre.CNOT), 2)]),
            (
                3,
                [
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 3, "num_zero_ctrl": 0}
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.X), 3),
                    GateCount(resource_rep(qre.CNOT), 3),
                ],
            ),
            (
                6,
                [
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 6, "num_zero_ctrl": 0}
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.X), 6),
                    GateCount(resource_rep(qre.CNOT), 6),
                ],
            ),
        ),
    )
    def test_resources(self, register_size, expected_res):
        """Test that the resources are correct."""
        result = qre.RegisterEquality.resource_decomp(register_size)
        for r, e in zip(result, expected_res, strict=True):
            if hasattr(r, "equal"):
                assert r.equal(e)
            else:
                assert r == e
