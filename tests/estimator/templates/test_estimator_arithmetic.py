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

import numpy as np
import pytest

import pennylane as qml
import pennylane.estimator as qre
from pennylane.estimator import GateCount, resource_rep
from pennylane.estimator.resource_config import ResourceConfig
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.wires import Wires

# pylint: disable=no-self-use,too-many-arguments


class TestResourceOutOfPlaceSquare:
    """Test the OutOfPlaceSquare class."""

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resource_params(self, register_size):
        """Test that the resource params are correct."""
        op = qre.OutOfPlaceSquare(register_size)
        assert op.resource_params == {"register_size": register_size}

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resource_rep(self, register_size):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.OutOfPlaceSquare, 3 * register_size, {"register_size": register_size}
        )
        assert qre.OutOfPlaceSquare.resource_rep(register_size=register_size) == expected

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resources(self, register_size):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(qre.Toffoli), (register_size - 1) ** 2),
            GateCount(resource_rep(qre.CNOT), register_size),
        ]
        assert qre.OutOfPlaceSquare.resource_decomp(register_size=register_size) == expected


class TestResourceOutMultiplier:
    """Test the OutMultiplier class."""

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_params(self, a_register_size, b_register_size):
        """Test that the resource params are correct."""
        op = qre.OutMultiplier(a_register_size, b_register_size)
        assert op.resource_params == {
            "a_num_wires": a_register_size,
            "b_num_wires": b_register_size,
        }

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_rep(self, a_register_size, b_register_size):
        """Test that the compressed representation is correct."""
        expected_num_wires = a_register_size + 3 * b_register_size
        expected = qre.CompressedResourceOp(
            qre.OutMultiplier,
            expected_num_wires,
            {"a_num_wires": a_register_size, "b_num_wires": b_register_size},
        )
        assert qre.OutMultiplier.resource_rep(a_register_size, b_register_size) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        a_register_size = 5
        b_register_size = 3

        toff = resource_rep(qre.Toffoli)
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        num_elbows = 12
        num_toff = 1

        expected = [
            GateCount(l_elbow, num_elbows),
            GateCount(r_elbow, num_elbows),
            GateCount(toff, num_toff),
        ]
        assert qre.OutMultiplier.resource_decomp(a_register_size, b_register_size) == expected


class TestResourceSemiAdder:
    """Test the ResourceSemiAdder class."""

    @pytest.mark.parametrize("register_size", (1, 2, 3, 4))
    def test_resource_params(self, register_size):
        """Test that the resource params are correct."""
        op = qre.SemiAdder(register_size)
        assert op.resource_params == {"max_register_size": register_size}

    @pytest.mark.parametrize("register_size", (1, 2, 3, 4))
    def test_resource_rep(self, register_size):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.SemiAdder, 2 * register_size, {"max_register_size": register_size}
        )
        assert qre.SemiAdder.resource_rep(max_register_size=register_size) == expected

    @pytest.mark.parametrize(
        "register_size, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(qre.CNOT))],
            ),
            (
                2,
                [
                    GateCount(resource_rep(qre.CNOT), 2),
                    GateCount(resource_rep(qre.X), 2),
                    GateCount(resource_rep(qre.Toffoli)),
                ],
            ),
            (
                3,
                [
                    qre.Allocate(2),
                    GateCount(resource_rep(qre.CNOT), 9),
                    GateCount(resource_rep(qre.TemporaryAND), 2),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.TemporaryAND)},
                        ),
                        2,
                    ),
                    qre.Deallocate(2),
                ],
            ),
        ),
    )
    def test_resources(self, register_size, expected_res):
        """Test that the resources are correct."""
        assert qre.SemiAdder.resource_decomp(register_size) == expected_res

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, max_register_size, expected_res",
        (
            (
                1,
                1,
                1,
                [
                    GateCount(resource_rep(qre.X), 2),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.CNOT),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                ],
            ),
            (
                1,
                0,
                5,
                [
                    qre.Allocate(4),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.TemporaryAND), 8),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        8,
                    ),
                    qre.Deallocate(4),
                ],
            ),
            (
                2,
                1,
                5,
                [
                    qre.Allocate(1),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX,
                            {
                                "num_ctrl_wires": 2,
                                "num_zero_ctrl": 1,
                            },
                        ),
                        2,
                    ),
                    qre.Allocate(4),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.TemporaryAND), 8),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        8,
                    ),
                    qre.Deallocate(4),
                    qre.Deallocate(1),
                ],
            ),
            (
                1,
                1,
                5,
                [
                    qre.Allocate(4),
                    GateCount(resource_rep(qre.CNOT), 24),
                    GateCount(resource_rep(qre.TemporaryAND), 8),
                    GateCount(
                        resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TemporaryAND)}),
                        8,
                    ),
                    qre.Deallocate(4),
                    GateCount(resource_rep(qre.X), 2),
                ],
            ),
        ),
    )
    def test_resources_controlled(
        self, num_ctrl_wires, num_zero_ctrl, max_register_size, expected_res
    ):
        """Test that the special case controlled resources are correct."""
        op = qre.Controlled(
            qre.SemiAdder(max_register_size=max_register_size),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        assert op.resource_decomp(**op.resource_params) == expected_res
