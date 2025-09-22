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
import math

import pytest

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation import AllocWires, FreeWires, GateCount, resource_rep
from pennylane.labs.resource_estimation.resource_config import ResourceConfig

# pylint: disable=no-self-use,too-many-arguments


class TestResourceOutOfPlaceSquare:
    """Test the OutOfPlaceSquare class."""

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resource_params(self, register_size):
        """Test that the resource params are correct."""
        op = plre.ResourceOutOfPlaceSquare(register_size)
        assert op.resource_params == {"register_size": register_size}

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resource_rep(self, register_size):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceOutOfPlaceSquare, 3 * register_size, {"register_size": register_size}
        )
        assert plre.ResourceOutOfPlaceSquare.resource_rep(register_size=register_size) == expected

    @pytest.mark.parametrize("register_size", (1, 2, 3))
    def test_resources(self, register_size):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(plre.ResourceToffoli), (register_size - 1) ** 2),
            GateCount(resource_rep(plre.ResourceCNOT), register_size),
        ]
        assert (
            plre.ResourceOutOfPlaceSquare.resource_decomp(register_size=register_size) == expected
        )


class TestResourcePhaseGradient:
    """Test the PhaseGradient class."""

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = plre.ResourcePhaseGradient(num_wires)
        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourcePhaseGradient, num_wires, {"num_wires": num_wires}
        )
        assert plre.ResourcePhaseGradient.resource_rep(num_wires=num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [
                    GateCount(plre.ResourceHadamard.resource_rep()),
                    GateCount(plre.ResourceZ.resource_rep()),
                ],
            ),
            (
                2,
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 2),
                    GateCount(plre.ResourceZ.resource_rep()),
                    GateCount(plre.ResourceS.resource_rep()),
                ],
            ),
            (
                3,
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 3),
                    GateCount(plre.ResourceZ.resource_rep()),
                    GateCount(plre.ResourceS.resource_rep()),
                    GateCount(plre.ResourceT.resource_rep()),
                ],
            ),
            (
                5,
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 5),
                    GateCount(plre.ResourceZ.resource_rep()),
                    GateCount(plre.ResourceS.resource_rep()),
                    GateCount(plre.ResourceT.resource_rep()),
                    GateCount(plre.ResourceRZ.resource_rep(), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_wires, expected_res):
        """Test that the resources are correct."""
        assert plre.ResourcePhaseGradient.resource_decomp(num_wires=num_wires) == expected_res


class TestResourceOutMultiplier:
    """Test the OutMultiplier class."""

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_params(self, a_register_size, b_register_size):
        """Test that the resource params are correct."""
        op = plre.ResourceOutMultiplier(a_register_size, b_register_size)
        assert op.resource_params == {
            "a_num_qubits": a_register_size,
            "b_num_qubits": b_register_size,
        }

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_rep(self, a_register_size, b_register_size):
        """Test that the compressed representation is correct."""
        expected_num_wires = a_register_size + 3 * b_register_size
        expected = plre.CompressedResourceOp(
            plre.ResourceOutMultiplier,
            expected_num_wires,
            {"a_num_qubits": a_register_size, "b_num_qubits": b_register_size},
        )
        assert plre.ResourceOutMultiplier.resource_rep(a_register_size, b_register_size) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        a_register_size = 5
        b_register_size = 3

        toff = resource_rep(plre.ResourceToffoli)
        l_elbow = resource_rep(plre.ResourceTempAND)
        r_elbow = resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": l_elbow})

        num_elbows = 12
        num_toff = 1

        expected = [
            GateCount(l_elbow, num_elbows),
            GateCount(r_elbow, num_elbows),
            GateCount(toff, num_toff),
        ]
        assert (
            plre.ResourceOutMultiplier.resource_decomp(a_register_size, b_register_size) == expected
        )


class TestResourceSemiAdder:
    """Test the ResourceSemiAdder class."""

    @pytest.mark.parametrize("register_size", (1, 2, 3, 4))
    def test_resource_params(self, register_size):
        """Test that the resource params are correct."""
        op = plre.ResourceSemiAdder(register_size)
        assert op.resource_params == {"max_register_size": register_size}

    @pytest.mark.parametrize("register_size", (1, 2, 3, 4))
    def test_resource_rep(self, register_size):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceSemiAdder, 2 * register_size, {"max_register_size": register_size}
        )
        assert plre.ResourceSemiAdder.resource_rep(max_register_size=register_size) == expected

    @pytest.mark.parametrize(
        "register_size, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(plre.ResourceCNOT))],
            ),
            (
                2,
                [
                    GateCount(resource_rep(plre.ResourceCNOT), 2),
                    GateCount(resource_rep(plre.ResourceX), 2),
                    GateCount(resource_rep(plre.ResourceToffoli)),
                ],
            ),
            (
                3,
                [
                    plre.AllocWires(2),
                    GateCount(resource_rep(plre.ResourceCNOT), 9),
                    GateCount(resource_rep(plre.ResourceTempAND), 2),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": resource_rep(plre.ResourceTempAND)},
                        ),
                        2,
                    ),
                    plre.FreeWires(2),
                ],
            ),
        ),
    )
    def test_resources(self, register_size, expected_res):
        """Test that the resources are correct."""
        assert plre.ResourceSemiAdder.resource_decomp(register_size) == expected_res

    def test_resources_controlled(self):
        """Test that the special case controlled resources are correct."""
        op = plre.ResourceControlled(
            plre.ResourceSemiAdder(max_register_size=5),
            num_ctrl_wires=1,
            num_ctrl_values=0,
        )

        expected_res = [
            plre.AllocWires(4),
            GateCount(resource_rep(plre.ResourceCNOT), 24),
            GateCount(resource_rep(plre.ResourceTempAND), 8),
            GateCount(
                resource_rep(
                    plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceTempAND)}
                ),
                8,
            ),
            plre.FreeWires(4),
        ]
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceControlledSequence:
    """Test the ResourceControlledSequence class."""

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires",
        (
            (plre.ResourceQFT(5), 5),
            (plre.ResourceRZ(precision=1e-3), 10),
            (
                plre.ResourceMultiRZ(
                    3,
                    1e-5,
                ),
                3,
            ),
        ),
    )
    def test_resource_params(self, base_op, num_ctrl_wires):
        """Test the resource params"""
        op = plre.ResourceControlledSequence(base_op, num_ctrl_wires)
        expected_params = {
            "base_cmpr_op": base_op.resource_rep_from_op(),
            "num_ctrl_wires": num_ctrl_wires,
        }

        assert op.resource_params == expected_params

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires",
        (
            (plre.ResourceQFT(5), 5),
            (plre.ResourceRZ(precision=1e-3), 10),
            (
                plre.ResourceMultiRZ(
                    3,
                    1e-5,
                ),
                3,
            ),
        ),
    )
    def test_resource_rep(self, base_op, num_ctrl_wires):
        """Test the resource rep method"""
        base_cmpr_op = base_op.resource_rep_from_op()
        expected = plre.CompressedResourceOp(
            plre.ResourceControlledSequence,
            base_cmpr_op.num_wires + num_ctrl_wires,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_ctrl_wires": num_ctrl_wires,
            },
        )

        assert (
            plre.ResourceControlledSequence.resource_rep(base_cmpr_op, num_ctrl_wires) == expected
        )

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires, expected_res",
        (
            (
                plre.ResourceQFT(5),
                5,
                [
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQFT.resource_rep(5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQFT.resource_rep(5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQFT.resource_rep(5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQFT.resource_rep(5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQFT.resource_rep(5),
                                16,
                            ),
                            1,
                            0,
                        )
                    ),
                ],
            ),
            (
                plre.ResourceRZ(precision=1e-3),
                3,
                [
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(precision=1e-3),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(precision=1e-3),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(precision=1e-3),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                ],
            ),
            (
                plre.ResourceChangeBasisOp(
                    compute_op=plre.ResourceAQFT(3, 5),
                    base_op=plre.ResourceRZ(),
                ),
                3,
                [
                    GateCount(plre.ResourceAQFT.resource_rep(3, 5)),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(plre.ResourceAQFT.resource_rep(3, 5))
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_ctrl_wires, expected_res):
        """Test resources"""
        op = plre.ResourceControlledSequence(base_op, num_ctrl_wires)
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceQPE:
    """Test the ResourceQPE class."""

    @pytest.mark.parametrize(
        "base_op, num_est_wires, adj_qft_op",
        (
            (plre.ResourceRX(precision=1e-5), 5, None),
            (plre.ResourceX(), 3, plre.ResourceQFT(3)),
            (plre.ResourceRZ(), 4, plre.ResourceAdjoint(plre.ResourceAQFT(3, 4))),
        ),
    )
    def test_resource_params(self, base_op, num_est_wires, adj_qft_op):
        """Test the resource_params method"""
        base_cmpr_op = base_op.resource_rep_from_op()

        if adj_qft_op is None:
            op = plre.ResourceQPE(base_op, num_est_wires)
            adj_qft_cmpr_op = None
        else:
            op = plre.ResourceQPE(base_op, num_est_wires, adj_qft_op)
            adj_qft_cmpr_op = adj_qft_op.resource_rep_from_op()

        assert op.resource_params == {
            "base_cmpr_op": base_cmpr_op,
            "num_estimation_wires": num_est_wires,
            "adj_qft_cmpr_op": adj_qft_cmpr_op,
        }

    @pytest.mark.parametrize(
        "base_cmpr_op, num_est_wires, adj_qft_cmpr_op",
        (
            (plre.ResourceRX.resource_rep(precision=1e-5), 5, None),
            (plre.ResourceX.resource_rep(), 3, plre.ResourceQFT.resource_rep(3)),
            (
                plre.ResourceRZ.resource_rep(),
                4,
                plre.ResourceAdjoint.resource_rep(plre.ResourceAQFT.resource_rep(3, 4)),
            ),
        ),
    )
    def test_resource_rep(self, base_cmpr_op, num_est_wires, adj_qft_cmpr_op):
        """Test the resource_rep method"""
        if adj_qft_cmpr_op is None:
            adj_qft_cmpr_op = plre.ResourceAdjoint.resource_rep(
                plre.ResourceQFT.resource_rep(num_est_wires)
            )

        expected = plre.CompressedResourceOp(
            plre.ResourceQPE,
            base_cmpr_op.num_wires + num_est_wires,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_estimation_wires": num_est_wires,
                "adj_qft_cmpr_op": adj_qft_cmpr_op,
            },
        )

        assert (
            plre.ResourceQPE.resource_rep(base_cmpr_op, num_est_wires, adj_qft_cmpr_op) == expected
        )

    @pytest.mark.parametrize(
        "base_op, num_est_wires, adj_qft_op, expected_res",
        (
            (
                plre.ResourceRX(precision=1e-5),
                5,
                None,
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 5),
                    GateCount(
                        plre.ResourceControlledSequence.resource_rep(
                            plre.ResourceRX.resource_rep(precision=1e-5),
                            5,
                        ),
                    ),
                    GateCount(plre.ResourceAdjoint.resource_rep(plre.ResourceQFT.resource_rep(5))),
                ],
            ),
            (
                plre.ResourceX(),
                3,
                plre.ResourceQFT(3),
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 3),
                    GateCount(
                        plre.ResourceControlledSequence.resource_rep(
                            plre.ResourceX.resource_rep(),
                            3,
                        ),
                    ),
                    GateCount(plre.ResourceQFT.resource_rep(3)),
                ],
            ),
            (
                plre.ResourceRZ(),
                4,
                plre.ResourceAdjoint(plre.ResourceAQFT(3, 4)),
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 4),
                    GateCount(
                        plre.ResourceControlledSequence.resource_rep(
                            plre.ResourceRZ.resource_rep(),
                            4,
                        ),
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(plre.ResourceAQFT.resource_rep(3, 4)),
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_est_wires, adj_qft_op, expected_res):
        """Test that resources method is correct"""
        op = (
            plre.ResourceQPE(base_op, num_est_wires)
            if adj_qft_op is None
            else plre.ResourceQPE(base_op, num_est_wires, adj_qft_op)
        )
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceIterativeQPE:
    """Test the ResourceIterativeQPE class."""

    @pytest.mark.parametrize(
        "base_op, num_iter",
        (
            (plre.ResourceRX(precision=1e-5), 5),
            (plre.ResourceQubitUnitary(4, 1e-5), 7),
            (
                plre.ResourceChangeBasisOp(
                    plre.ResourceRY(precision=1e-3),
                    plre.ResourceRZ(precision=1e-5),
                ),
                3,
            ),
        ),
    )
    def test_resource_params(self, base_op, num_iter):
        """Test the resource_params method"""
        op = plre.ResourceIterativeQPE(base_op, num_iter)
        expected = {
            "base_cmpr_op": base_op.resource_rep_from_op(),
            "num_iter": num_iter,
        }
        assert op.resource_params == expected

    @pytest.mark.parametrize(
        "base_op, num_iter",
        (
            (plre.ResourceRX(precision=1e-5), 5),
            (plre.ResourceQubitUnitary(4, 1e-5), 7),
            (
                plre.ResourceChangeBasisOp(
                    plre.ResourceRY(precision=1e-3),
                    plre.ResourceRZ(precision=1e-5),
                ),
                3,
            ),
        ),
    )
    def test_resource_rep(self, base_op, num_iter):
        """Test the resource_rep method"""
        base_cmpr_op = base_op.resource_rep_from_op()
        expected = plre.CompressedResourceOp(
            plre.ResourceIterativeQPE,
            base_cmpr_op.num_wires,
            {"base_cmpr_op": base_cmpr_op, "num_iter": num_iter},
        )
        assert plre.ResourceIterativeQPE.resource_rep(base_cmpr_op, num_iter) == expected

    @pytest.mark.parametrize(
        "base_op, num_iter, expected_res",
        (
            (
                plre.ResourceRX(precision=1e-5),
                5,
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 10),
                    AllocWires(1),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRX.resource_rep(precision=1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRX.resource_rep(precision=1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRX.resource_rep(precision=1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRX.resource_rep(precision=1e-5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRX.resource_rep(precision=1e-5),
                                16,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(plre.ResourcePhaseShift.resource_rep(), 10),
                    FreeWires(1),
                ],
            ),
            (
                plre.ResourceQubitUnitary(7, 1e-5),
                4,
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 8),
                    AllocWires(1),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQubitUnitary.resource_rep(7, 1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQubitUnitary.resource_rep(7, 1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQubitUnitary.resource_rep(7, 1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceQubitUnitary.resource_rep(7, 1e-5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(plre.ResourcePhaseShift.resource_rep(), 6),
                    FreeWires(1),
                ],
            ),
            (
                plre.ResourceChangeBasisOp(
                    plre.ResourceRY(precision=1e-3),
                    plre.ResourceRZ(precision=1e-5),
                ),
                3,
                [
                    GateCount(plre.ResourceHadamard.resource_rep(), 6),
                    AllocWires(1),
                    GateCount(plre.ResourceRY.resource_rep(precision=1e-3)),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(precision=1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(precision=1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourcePow.resource_rep(
                                plre.ResourceRZ.resource_rep(precision=1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceRY.resource_rep(precision=1e-3)
                        ),
                    ),
                    GateCount(plre.ResourcePhaseShift.resource_rep(), 3),
                    FreeWires(1),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_iter, expected_res):
        """Test the resources method"""
        op = plre.ResourceIterativeQPE(base_op, num_iter)
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceQFT:
    """Test the ResourceQFT class."""

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = plre.ResourceQFT(num_wires)
        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(plre.ResourceQFT, num_wires, {"num_wires": num_wires})
        assert plre.ResourceQFT.resource_rep(num_wires=num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(plre.ResourceHadamard))],
            ),
            (
                2,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                    GateCount(resource_rep(plre.ResourceSWAP)),
                    GateCount(resource_rep(plre.ResourceControlledPhaseShift)),
                ],
            ),
            (
                3,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 3),
                    GateCount(resource_rep(plre.ResourceSWAP)),
                    GateCount(resource_rep(plre.ResourceControlledPhaseShift), 3),
                ],
            ),
        ),
    )
    def test_resources(self, num_wires, expected_res):
        """Test that the resources are correct."""
        assert plre.ResourceQFT.resource_decomp(num_wires) == expected_res

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [GateCount(resource_rep(plre.ResourceHadamard))],
            ),
            (
                2,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                    GateCount(resource_rep(plre.ResourceSWAP)),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(max_register_size=1),
                            num_ctrl_wires=1,
                            num_ctrl_values=0,
                        )
                    ),
                ],
            ),
            (
                3,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 3),
                    GateCount(resource_rep(plre.ResourceSWAP)),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(max_register_size=1),
                            num_ctrl_wires=1,
                            num_ctrl_values=0,
                        )
                    ),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            plre.ResourceSemiAdder.resource_rep(max_register_size=2),
                            num_ctrl_wires=1,
                            num_ctrl_values=0,
                        )
                    ),
                ],
            ),
        ),
    )
    def test_resources_phasegrad(self, num_wires, expected_res):
        """Test that the resources are correct for phase gradient method."""
        assert plre.ResourceQFT.phase_grad_resource_decomp(num_wires) == expected_res


class TestResourceAQFT:
    """Test the ResourceAQFT class."""

    @pytest.mark.parametrize(
        "num_wires, order",
        (
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (5, 5),
        ),
    )
    def test_resource_params(self, num_wires, order):
        """Test that the resource params are correct."""
        op = plre.ResourceAQFT(order, num_wires)
        assert op.resource_params == {"order": order, "num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires, order",
        (
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 3),
            (5, 5),
        ),
    )
    def test_resource_rep(self, order, num_wires):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceAQFT, num_wires, {"order": order, "num_wires": num_wires}
        )
        assert plre.ResourceAQFT.resource_rep(order=order, num_wires=num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, order, expected_res",
        (
            (
                5,
                1,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 5),
                ],
            ),
            (
                5,
                3,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 5),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            base_cmpr_op=resource_rep(plre.ResourceS),
                            num_ctrl_wires=1,
                            num_ctrl_values=0,
                        ),
                        4,
                    ),
                    AllocWires(1),
                    GateCount(resource_rep(plre.ResourceTempAND), 1),
                    GateCount(plre.ResourceSemiAdder.resource_rep(1)),
                    GateCount(resource_rep(plre.ResourceHadamard)),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(resource_rep(plre.ResourceTempAND)),
                        1,
                    ),
                    FreeWires(1),
                    AllocWires(2),
                    GateCount(resource_rep(plre.ResourceTempAND), 2 * 2),
                    GateCount(plre.ResourceSemiAdder.resource_rep(2), 2),
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(resource_rep(plre.ResourceTempAND)),
                        2 * 2,
                    ),
                    FreeWires(2),
                    GateCount(resource_rep(plre.ResourceSWAP), 2),
                ],
            ),
            (
                5,
                5,
                [
                    GateCount(resource_rep(plre.ResourceHadamard), 5),
                    GateCount(
                        plre.ResourceControlled.resource_rep(
                            base_cmpr_op=resource_rep(plre.ResourceS),
                            num_ctrl_wires=1,
                            num_ctrl_values=0,
                        ),
                        4,
                    ),
                    AllocWires(1),
                    GateCount(resource_rep(plre.ResourceTempAND), 1),
                    GateCount(plre.ResourceSemiAdder.resource_rep(1)),
                    GateCount(resource_rep(plre.ResourceHadamard)),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(resource_rep(plre.ResourceTempAND)),
                        1,
                    ),
                    FreeWires(1),
                    AllocWires(2),
                    GateCount(resource_rep(plre.ResourceTempAND), 2),
                    GateCount(plre.ResourceSemiAdder.resource_rep(2)),
                    GateCount(resource_rep(plre.ResourceHadamard)),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(resource_rep(plre.ResourceTempAND)),
                        2,
                    ),
                    FreeWires(2),
                    AllocWires(3),
                    GateCount(resource_rep(plre.ResourceTempAND), 3),
                    GateCount(plre.ResourceSemiAdder.resource_rep(3)),
                    GateCount(resource_rep(plre.ResourceHadamard)),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(resource_rep(plre.ResourceTempAND)),
                        3,
                    ),
                    FreeWires(3),
                    GateCount(resource_rep(plre.ResourceSWAP), 2),
                ],
            ),
        ),
    )
    def test_resources(self, order, num_wires, expected_res):
        """Test that the resources are correct."""
        assert plre.ResourceAQFT.resource_decomp(order, num_wires) == expected_res


class TestResourceBasisRotation:
    """Test the BasisRotation class."""

    @pytest.mark.parametrize("dim_n", (1, 2, 3))
    def test_resource_params(self, dim_n):
        """Test that the resource params are correct."""
        op = plre.ResourceBasisRotation(dim_n)
        assert op.resource_params == {"dim_N": dim_n}

    @pytest.mark.parametrize("dim_n", (1, 2, 3))
    def test_resource_rep(self, dim_n):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(plre.ResourceBasisRotation, dim_n, {"dim_N": dim_n})
        assert plre.ResourceBasisRotation.resource_rep(dim_N=dim_n) == expected

    @pytest.mark.parametrize("dim_n", (1, 2, 3))
    def test_resources(self, dim_n):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(plre.ResourcePhaseShift), dim_n + (dim_n * (dim_n - 1) // 2)),
            GateCount(resource_rep(plre.ResourceSingleExcitation), dim_n * (dim_n - 1) // 2),
        ]
        assert plre.ResourceBasisRotation.resource_decomp(dim_n) == expected


class TestResourceSelect:
    """Test the Select class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        ops = [plre.ResourceRX(), plre.ResourceZ(), plre.ResourceCNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        op = plre.ResourceSelect(ops)
        assert op.resource_params == {"cmpr_ops": cmpr_ops, "num_wires": 4}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        ops = [plre.ResourceRX(wires=0), plre.ResourceZ(wires=1), plre.ResourceCNOT(wires=[1, 2])]
        num_wires = 3 + 2  # 3 op wires + 2 control wires
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = plre.CompressedResourceOp(
            plre.ResourceSelect, num_wires, {"cmpr_ops": cmpr_ops, "num_wires": num_wires}
        )
        print(expected)
        print(plre.ResourceSelect.resource_rep(cmpr_ops, num_wires))
        assert plre.ResourceSelect.resource_rep(cmpr_ops, num_wires) == expected

        op = plre.ResourceSelect(ops)
        print(op.resource_rep(**op.resource_params))
        assert op.resource_rep(**op.resource_params) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        ops = [plre.ResourceRX(), plre.ResourceZ(), plre.ResourceCNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = [
            plre.AllocWires(1),
            GateCount(
                plre.ResourceControlled.resource_rep(
                    plre.ResourceRX.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(
                plre.ResourceControlled.resource_rep(
                    plre.ResourceZ.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(
                plre.ResourceControlled.resource_rep(
                    plre.ResourceCNOT.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(plre.ResourceX.resource_rep(), 4),
            GateCount(plre.ResourceCNOT.resource_rep(), 2),
            GateCount(plre.ResourceTempAND.resource_rep(), 2),
            GateCount(
                plre.ResourceAdjoint.resource_rep(
                    plre.ResourceTempAND.resource_rep(),
                ),
                2,
            ),
            plre.FreeWires(1),
        ]
        assert plre.ResourceSelect.resource_decomp(cmpr_ops, num_wires=4) == expected


class TestResourceQROM:
    """Test the ResourceQROM class."""

    def test_select_swap_depth_errors(self):
        """Test that the correct error is raised when invalid values of
        select_swap_depth are provided.
        """
        select_swap_depth = "Not A Valid Input"
        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            plre.ResourceQROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(ValueError, match="`select_swap_depth` must be None or an integer."):
            plre.ResourceQROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

        select_swap_depth = 3
        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            plre.ResourceQROM(100, 10, select_swap_depth=select_swap_depth)

        with pytest.raises(
            ValueError, match="`select_swap_depth` must be 1 or a positive integer power of 2."
        ):
            plre.ResourceQROM.resource_rep(100, 10, select_swap_depth=select_swap_depth)

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, clean",
        (
            (10, 3, 15, None, True),
            (100, 5, 50, 2, False),
            (12, 2, 5, 1, True),
        ),
    )
    def test_resource_params(self, num_data_points, size_data_points, num_bit_flips, depth, clean):
        """Test that the resource params are correct."""
        if depth is None:
            op = plre.ResourceQROM(num_data_points, size_data_points)
        else:
            op = plre.ResourceQROM(num_data_points, size_data_points, num_bit_flips, clean, depth)

        assert op.resource_params == {
            "num_bitstrings": num_data_points,
            "size_bitstring": size_data_points,
            "num_bit_flips": num_bit_flips,
            "select_swap_depth": depth,
            "clean": clean,
        }

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, clean",
        (
            (10, 3, 15, None, True),
            (100, 5, 50, 2, False),
            (12, 2, 5, 1, True),
        ),
    )
    def test_resource_rep(self, num_data_points, size_data_points, num_bit_flips, depth, clean):
        """Test that the compressed representation is correct."""
        expected_num_wires = size_data_points + math.ceil(math.log2(num_data_points))
        expected = plre.CompressedResourceOp(
            plre.ResourceQROM,
            expected_num_wires,
            {
                "num_bitstrings": num_data_points,
                "size_bitstring": size_data_points,
                "num_bit_flips": num_bit_flips,
                "select_swap_depth": depth,
                "clean": clean,
            },
        )
        assert (
            plre.ResourceQROM.resource_rep(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                clean=clean,
                select_swap_depth=depth,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "num_data_points, size_data_points, num_bit_flips, depth, clean, expected_res",
        (
            (
                10,
                3,
                15,
                None,
                True,
                [
                    plre.AllocWires(5),
                    GateCount(plre.ResourceHadamard.resource_rep(), 6),
                    GateCount(plre.ResourceX.resource_rep(), 14),
                    GateCount(plre.ResourceCNOT.resource_rep(), 36),
                    GateCount(plre.ResourceTempAND.resource_rep(), 6),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceTempAND.resource_rep(),
                        ),
                        6,
                    ),
                    plre.FreeWires(2),
                    GateCount(plre.ResourceCSWAP.resource_rep(), 12),
                    plre.FreeWires(3),
                ],
            ),
            (
                100,
                5,
                50,
                2,
                False,
                [
                    plre.AllocWires(10),
                    GateCount(plre.ResourceX.resource_rep(), 97),
                    GateCount(plre.ResourceCNOT.resource_rep(), 98),
                    GateCount(plre.ResourceTempAND.resource_rep(), 48),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceTempAND.resource_rep(),
                        ),
                        48,
                    ),
                    plre.FreeWires(5),
                    GateCount(plre.ResourceCSWAP.resource_rep(), 5),
                ],
            ),
            (
                12,
                2,
                5,
                1,
                True,
                [
                    plre.AllocWires(3),
                    GateCount(plre.ResourceHadamard.resource_rep(), 4),
                    GateCount(plre.ResourceX.resource_rep(), 42),
                    GateCount(plre.ResourceCNOT.resource_rep(), 30),
                    GateCount(plre.ResourceTempAND.resource_rep(), 20),
                    GateCount(
                        plre.ResourceAdjoint.resource_rep(
                            plre.ResourceTempAND.resource_rep(),
                        ),
                        20,
                    ),
                    plre.FreeWires(3),
                    GateCount(plre.ResourceCSWAP.resource_rep(), 0),
                    plre.FreeWires(0),
                ],
            ),
            (
                12,
                2,
                5,
                128,  # This will get turncated to 16 as the max depth
                False,
                [
                    plre.AllocWires(30),
                    GateCount(plre.ResourceX.resource_rep(), 5),
                    GateCount(plre.ResourceCSWAP.resource_rep(), 30),
                ],
            ),
            (
                12,
                2,
                5,
                16,
                True,
                [
                    plre.AllocWires(30),
                    GateCount(plre.ResourceHadamard.resource_rep(), 4),
                    GateCount(plre.ResourceX.resource_rep(), 10),
                    GateCount(plre.ResourceCSWAP.resource_rep(), 120),
                    plre.FreeWires(30),
                ],
            ),
        ),
    )
    def test_resources(
        self, num_data_points, size_data_points, num_bit_flips, depth, clean, expected_res
    ):
        """Test that the resources are correct."""
        assert (
            plre.ResourceQROM.resource_decomp(
                num_bitstrings=num_data_points,
                size_bitstring=size_data_points,
                num_bit_flips=num_bit_flips,
                clean=clean,
                select_swap_depth=depth,
            )
            == expected_res
        )

    # pylint: disable=protected-access
    def test_t_select_swap_width(self):
        """Test that the private function doesn't give negative or
        fractional values for the depth"""
        num_bitstrings = 8
        size_bitstring = 17

        opt_width = plre.ResourceQROM._t_optimized_select_swap_width(
            num_bitstrings,
            size_bitstring,
        )
        assert opt_width == 1


class TestResourceQubitUnitary:
    """Test the ResourceQubitUnitary template"""

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5, 6))
    def test_resource_params(self, num_wires, precision):
        """Test that the resource params are correct."""
        op = (
            plre.ResourceQubitUnitary(num_wires, precision)
            if precision
            else plre.ResourceQubitUnitary(num_wires)
        )
        assert op.resource_params == {"num_wires": num_wires, "precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5, 6))
    def test_resource_rep(self, num_wires, precision):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceQubitUnitary, num_wires, {"num_wires": num_wires, "precision": precision}
        )
        assert (
            plre.ResourceQubitUnitary.resource_rep(num_wires=num_wires, precision=precision)
            == expected
        )

    @pytest.mark.parametrize(
        "num_wires, precision, expected_res",
        (
            (
                1,
                None,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"precision": 1e-9})),
                ],
            ),
            (
                2,
                1e-3,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"precision": 1e-3}), 4),
                    GateCount(resource_rep(plre.ResourceCNOT), 3),
                ],
            ),
            (
                5,
                1e-5,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"precision": 1e-5}), (4**3) * 4),
                    GateCount(resource_rep(plre.ResourceCNOT), (4**3) * 3),
                    GateCount(
                        resource_rep(
                            plre.ResourceSelectPauliRot,
                            {
                                "rotation_axis": "Z",
                                "num_ctrl_wires": 2,
                                "precision": 1e-5,
                            },
                        ),
                        2 * 4**2,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceSelectPauliRot,
                            {
                                "rotation_axis": "Y",
                                "num_ctrl_wires": 2,
                                "precision": 1e-5,
                            },
                        ),
                        4**2,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceSelectPauliRot,
                            {
                                "rotation_axis": "Z",
                                "num_ctrl_wires": 3,
                                "precision": 1e-5,
                            },
                        ),
                        2 * 4,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceSelectPauliRot,
                            {
                                "rotation_axis": "Y",
                                "num_ctrl_wires": 3,
                                "precision": 1e-5,
                            },
                        ),
                        4,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceSelectPauliRot,
                            {
                                "rotation_axis": "Z",
                                "num_ctrl_wires": 4,
                                "precision": 1e-5,
                            },
                        ),
                        2,
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceSelectPauliRot,
                            {
                                "rotation_axis": "Y",
                                "num_ctrl_wires": 4,
                                "precision": 1e-5,
                            },
                        ),
                        1,
                    ),
                ],
            ),
        ),
    )
    def test_default_resources(self, num_wires, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[plre.ResourceQubitUnitary]
            assert (
                plre.ResourceQubitUnitary.resource_decomp(num_wires=num_wires, **kwargs)
                == expected_res
            )
        else:
            assert (
                plre.ResourceQubitUnitary.resource_decomp(num_wires=num_wires, precision=precision)
                == expected_res
            )


class TestResourceSelectPauliRot:
    """Test the ResourceSelectPauliRot template"""

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("rotation_axis", ("X", "Y", "Z"))
    @pytest.mark.parametrize("num_ctrl_wires", (1, 2, 3, 4, 5))
    def test_resource_params(self, num_ctrl_wires, rotation_axis, precision):
        """Test that the resource params are correct."""
        op = (
            plre.ResourceSelectPauliRot(rotation_axis, num_ctrl_wires, precision)
            if precision
            else plre.ResourceSelectPauliRot(rotation_axis, num_ctrl_wires)
        )
        assert op.resource_params == {
            "rotation_axis": rotation_axis,
            "num_ctrl_wires": num_ctrl_wires,
            "precision": precision,
        }

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("rotation_axis", ("X", "Y", "Z"))
    @pytest.mark.parametrize("num_ctrl_wires", (1, 2, 3, 4, 5))
    def test_resource_rep(self, num_ctrl_wires, rotation_axis, precision):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceSelectPauliRot,
            num_ctrl_wires + 1,
            {
                "rotation_axis": rotation_axis,
                "num_ctrl_wires": num_ctrl_wires,
                "precision": precision,
            },
        )
        assert (
            plre.ResourceSelectPauliRot.resource_rep(num_ctrl_wires, rotation_axis, precision)
            == expected
        )

    @pytest.mark.parametrize(
        "num_ctrl_wires, rotation_axis, precision, expected_res",
        (
            (
                1,
                "X",
                None,
                [
                    GateCount(resource_rep(plre.ResourceRX, {"precision": 1e-9}), 2),
                    GateCount(resource_rep(plre.ResourceCNOT), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    GateCount(resource_rep(plre.ResourceRY, {"precision": 1e-3}), 2**2),
                    GateCount(resource_rep(plre.ResourceCNOT), 2**2),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"precision": 1e-5}), 2**5),
                    GateCount(resource_rep(plre.ResourceCNOT), 2**5),
                ],
            ),
        ),
    )
    def test_default_resources(self, num_ctrl_wires, rotation_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[plre.ResourceSelectPauliRot]
            assert (
                plre.ResourceSelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rotation_axis=rotation_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                plre.ResourceSelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rotation_axis=rotation_axis,
                    precision=precision,
                )
                == expected_res
            )

    @pytest.mark.parametrize(
        "num_ctrl_wires, rotation_axis, precision, expected_res",
        (
            (
                1,
                "X",
                None,
                [
                    AllocWires(33),
                    GateCount(plre.ResourceQROM.resource_rep(2, 33, 33, False)),
                    GateCount(
                        resource_rep(
                            plre.ResourceControlled,
                            {
                                "base_cmpr_op": plre.ResourceSemiAdder.resource_rep(33),
                                "num_ctrl_wires": 1,
                                "num_ctrl_values": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": plre.ResourceQROM.resource_rep(2, 33, 33, False),
                            },
                        )
                    ),
                    FreeWires(33),
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    AllocWires(13),
                    GateCount(plre.ResourceQROM.resource_rep(4, 13, 26, False)),
                    GateCount(
                        resource_rep(
                            plre.ResourceControlled,
                            {
                                "base_cmpr_op": plre.ResourceSemiAdder.resource_rep(13),
                                "num_ctrl_wires": 1,
                                "num_ctrl_values": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": plre.ResourceQROM.resource_rep(4, 13, 26, False),
                            },
                        )
                    ),
                    FreeWires(13),
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                    GateCount(resource_rep(plre.ResourceS)),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint, {"base_cmpr_op": resource_rep(plre.ResourceS)}
                        )
                    ),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    AllocWires(20),
                    GateCount(plre.ResourceQROM.resource_rep(32, 20, 320, False)),
                    GateCount(
                        resource_rep(
                            plre.ResourceControlled,
                            {
                                "base_cmpr_op": plre.ResourceSemiAdder.resource_rep(20),
                                "num_ctrl_wires": 1,
                                "num_ctrl_values": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": plre.ResourceQROM.resource_rep(32, 20, 320, False),
                            },
                        )
                    ),
                    FreeWires(20),
                ],
            ),
        ),
    )
    def test_phase_gradient_resources(self, num_ctrl_wires, rotation_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[plre.ResourceSelectPauliRot]
            assert (
                plre.ResourceSelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rotation_axis=rotation_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                plre.ResourceSelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rotation_axis=rotation_axis,
                    precision=precision,
                )
                == expected_res
            )
