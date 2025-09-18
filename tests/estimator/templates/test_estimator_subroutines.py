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

import pennylane.estimator as qre
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.estimator import GateCount, resource_rep
from pennylane.estimator.resource_config import ResourceConfig

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


class TestResourcePhaseGradient:
    """Test the PhaseGradient class."""

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = qre.PhaseGradient(num_wires)
        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.PhaseGradient, num_wires, {"num_wires": num_wires})
        assert qre.PhaseGradient.resource_rep(num_wires=num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, expected_res",
        (
            (
                1,
                [
                    GateCount(qre.Hadamard.resource_rep()),
                    GateCount(qre.Z.resource_rep()),
                ],
            ),
            (
                2,
                [
                    GateCount(qre.Hadamard.resource_rep(), 2),
                    GateCount(qre.Z.resource_rep()),
                    GateCount(qre.S.resource_rep()),
                ],
            ),
            (
                3,
                [
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    GateCount(qre.Z.resource_rep()),
                    GateCount(qre.S.resource_rep()),
                    GateCount(qre.T.resource_rep()),
                ],
            ),
            (
                5,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5),
                    GateCount(qre.Z.resource_rep()),
                    GateCount(qre.S.resource_rep()),
                    GateCount(qre.T.resource_rep()),
                    GateCount(qre.RZ.resource_rep(), 2),
                ],
            ),
        ),
    )
    def test_resources(self, num_wires, expected_res):
        """Test that the resources are correct."""
        assert qre.PhaseGradient.resource_decomp(num_wires=num_wires) == expected_res


class TestResourceOutMultiplier:
    """Test the OutMultiplier class."""

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_params(self, a_register_size, b_register_size):
        """Test that the resource params are correct."""
        op = qre.OutMultiplier(a_register_size, b_register_size)
        assert op.resource_params == {
            "a_num_qubits": a_register_size,
            "b_num_qubits": b_register_size,
        }

    @pytest.mark.parametrize("a_register_size", (1, 2, 3))
    @pytest.mark.parametrize("b_register_size", (4, 5, 6))
    def test_resource_rep(self, a_register_size, b_register_size):
        """Test that the compressed representation is correct."""
        expected_num_wires = a_register_size + 3 * b_register_size
        expected = qre.CompressedResourceOp(
            qre.OutMultiplier,
            expected_num_wires,
            {"a_num_qubits": a_register_size, "b_num_qubits": b_register_size},
        )
        assert qre.OutMultiplier.resource_rep(a_register_size, b_register_size) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        a_register_size = 5
        b_register_size = 3

        toff = resource_rep(qre.Toffoli)
        l_elbow = resource_rep(qre.TempAND)
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
                    GateCount(resource_rep(qre.TempAND), 2),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": resource_rep(qre.TempAND)},
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

    def test_resources_controlled(self):
        """Test that the special case controlled resources are correct."""
        op = qre.Controlled(
            qre.SemiAdder(max_register_size=5),
            num_ctrl_wires=1,
            num_zero_ctrl=0,
        )

        expected_res = [
            qre.Allocate(4),
            GateCount(resource_rep(qre.CNOT), 24),
            GateCount(resource_rep(qre.TempAND), 8),
            GateCount(
                resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.TempAND)}),
                8,
            ),
            qre.Deallocate(4),
        ]
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceControlledSequence:
    """Test the ResourceControlledSequence class."""

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires",
        (
            (qre.QFT(5), 5),
            (qre.RZ(precision=1e-3), 10),
            (
                qre.MultiRZ(
                    3,
                    1e-5,
                ),
                3,
            ),
        ),
    )
    def test_resource_params(self, base_op, num_ctrl_wires):
        """Test the resource params"""
        op = qre.ControlledSequence(base_op, num_ctrl_wires)
        expected_params = {
            "base_cmpr_op": base_op.resource_rep_from_op(),
            "num_ctrl_wires": num_ctrl_wires,
        }

        assert op.resource_params == expected_params

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires",
        (
            (qre.QFT(5), 5),
            (qre.RZ(precision=1e-3), 10),
            (
                qre.MultiRZ(
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
        expected = qre.CompressedResourceOp(
            qre.ControlledSequence,
            base_cmpr_op.num_wires + num_ctrl_wires,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_ctrl_wires": num_ctrl_wires,
            },
        )

        assert qre.ControlledSequence.resource_rep(base_cmpr_op, num_ctrl_wires) == expected

    @pytest.mark.parametrize(
        "base_op, num_ctrl_wires, expected_res",
        (
            (
                qre.QFT(5),
                5,
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QFT.resource_rep(5),
                                16,
                            ),
                            1,
                            0,
                        )
                    ),
                ],
            ),
            (
                qre.RZ(precision=1e-3),
                3,
                [
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-3),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-3),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-3),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                ],
            ),
            (
                qre.ChangeOpBasis(
                    compute_op=qre.AQFT(3, 5),
                    target_op=qre.RZ(),
                ),
                3,
                [
                    GateCount(qre.AQFT.resource_rep(3, 5)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 5))),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_ctrl_wires, expected_res):
        """Test resources"""
        op = qre.ControlledSequence(base_op, num_ctrl_wires)
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceQPE:
    """Test the ResourceQPE class."""

    @pytest.mark.parametrize(
        "base_op, num_est_wires, adj_qft_op",
        (
            (qre.RX(precision=1e-5), 5, None),
            (qre.X(), 3, qre.QFT(3)),
            (qre.RZ(), 4, qre.Adjoint(qre.AQFT(3, 4))),
        ),
    )
    def test_resource_params(self, base_op, num_est_wires, adj_qft_op):
        """Test the resource_params method"""
        base_cmpr_op = base_op.resource_rep_from_op()

        if adj_qft_op is None:
            op = qre.QPE(base_op, num_est_wires)
            adj_qft_cmpr_op = None
        else:
            op = qre.QPE(base_op, num_est_wires, adj_qft_op)
            adj_qft_cmpr_op = adj_qft_op.resource_rep_from_op()

        assert op.resource_params == {
            "base_cmpr_op": base_cmpr_op,
            "num_estimation_wires": num_est_wires,
            "adj_qft_cmpr_op": adj_qft_cmpr_op,
        }

    @pytest.mark.parametrize(
        "base_cmpr_op, num_est_wires, adj_qft_cmpr_op",
        (
            (qre.RX.resource_rep(precision=1e-5), 5, None),
            (qre.X.resource_rep(), 3, qre.QFT.resource_rep(3)),
            (
                qre.RZ.resource_rep(),
                4,
                qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 4)),
            ),
        ),
    )
    def test_resource_rep(self, base_cmpr_op, num_est_wires, adj_qft_cmpr_op):
        """Test the resource_rep method"""
        if adj_qft_cmpr_op is None:
            adj_qft_cmpr_op = qre.Adjoint.resource_rep(qre.QFT.resource_rep(num_est_wires))

        expected = qre.CompressedResourceOp(
            qre.QPE,
            base_cmpr_op.num_wires + num_est_wires,
            {
                "base_cmpr_op": base_cmpr_op,
                "num_estimation_wires": num_est_wires,
                "adj_qft_cmpr_op": adj_qft_cmpr_op,
            },
        )

        assert qre.QPE.resource_rep(base_cmpr_op, num_est_wires, adj_qft_cmpr_op) == expected

    @pytest.mark.parametrize(
        "base_op, num_est_wires, adj_qft_op, expected_res",
        (
            (
                qre.RX(precision=1e-5),
                5,
                None,
                [
                    GateCount(qre.Hadamard.resource_rep(), 5),
                    GateCount(
                        qre.ControlledSequence.resource_rep(
                            qre.RX.resource_rep(precision=1e-5),
                            5,
                        ),
                    ),
                    GateCount(qre.Adjoint.resource_rep(qre.QFT.resource_rep(5))),
                ],
            ),
            (
                qre.X(),
                3,
                qre.QFT(3),
                [
                    GateCount(qre.Hadamard.resource_rep(), 3),
                    GateCount(
                        qre.ControlledSequence.resource_rep(
                            qre.X.resource_rep(),
                            3,
                        ),
                    ),
                    GateCount(qre.QFT.resource_rep(3)),
                ],
            ),
            (
                qre.RZ(),
                4,
                qre.Adjoint(qre.AQFT(3, 4)),
                [
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(
                        qre.ControlledSequence.resource_rep(
                            qre.RZ.resource_rep(),
                            4,
                        ),
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.AQFT.resource_rep(3, 4)),
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_est_wires, adj_qft_op, expected_res):
        """Test that resources method is correct"""
        op = (
            qre.QPE(base_op, num_est_wires)
            if adj_qft_op is None
            else qre.QPE(base_op, num_est_wires, adj_qft_op)
        )
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceIterativeQPE:
    """Test the ResourceIterativeQPE class."""

    @pytest.mark.parametrize(
        "base_op, num_iter",
        (
            (qre.RX(precision=1e-5), 5),
            (qre.QubitUnitary(4, 1e-5), 7),
            (
                qre.ChangeOpBasis(
                    qre.RY(precision=1e-3),
                    qre.RZ(precision=1e-5),
                ),
                3,
            ),
        ),
    )
    def test_resource_params(self, base_op, num_iter):
        """Test the resource_params method"""
        op = qre.IterativeQPE(base_op, num_iter)
        expected = {
            "base_cmpr_op": base_op.resource_rep_from_op(),
            "num_iter": num_iter,
        }
        assert op.resource_params == expected

    @pytest.mark.parametrize(
        "base_op, num_iter",
        (
            (qre.RX(precision=1e-5), 5),
            (qre.QubitUnitary(4, 1e-5), 7),
            (
                qre.ChangeOpBasis(
                    qre.RY(precision=1e-3),
                    qre.RZ(precision=1e-5),
                ),
                3,
            ),
        ),
    )
    def test_resource_rep(self, base_op, num_iter):
        """Test the resource_rep method"""
        base_cmpr_op = base_op.resource_rep_from_op()
        expected = qre.CompressedResourceOp(
            qre.IterativeQPE,
            base_cmpr_op.num_wires,
            {"base_cmpr_op": base_cmpr_op, "num_iter": num_iter},
        )
        assert qre.IterativeQPE.resource_rep(base_cmpr_op, num_iter) == expected

    @pytest.mark.parametrize(
        "base_op, num_iter, expected_res",
        (
            (
                qre.RX(precision=1e-5),
                5,
                [
                    GateCount(qre.Hadamard.resource_rep(), 10),
                    Allocate(1),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RX.resource_rep(precision=1e-5),
                                16,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(qre.PhaseShift.resource_rep(), 10),
                    Deallocate(1),
                ],
            ),
            (
                qre.QubitUnitary(7, 1e-5),
                4,
                [
                    GateCount(qre.Hadamard.resource_rep(), 8),
                    Allocate(1),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.QubitUnitary.resource_rep(7, 1e-5),
                                8,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(qre.PhaseShift.resource_rep(), 6),
                    Deallocate(1),
                ],
            ),
            (
                qre.ChangeOpBasis(
                    qre.RY(precision=1e-3),
                    qre.RZ(precision=1e-5),
                ),
                3,
                [
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    Allocate(1),
                    GateCount(qre.RY.resource_rep(precision=1e-3)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-5),
                                1,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-5),
                                2,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.Pow.resource_rep(
                                qre.RZ.resource_rep(precision=1e-5),
                                4,
                            ),
                            1,
                            0,
                        )
                    ),
                    GateCount(
                        qre.Adjoint.resource_rep(qre.RY.resource_rep(precision=1e-3)),
                    ),
                    GateCount(qre.PhaseShift.resource_rep(), 3),
                    Deallocate(1),
                ],
            ),
        ),
    )
    def test_resources(self, base_op, num_iter, expected_res):
        """Test the resources method"""
        op = qre.IterativeQPE(base_op, num_iter)
        assert op.resource_decomp(**op.resource_params) == expected_res


class TestResourceQFT:
    """Test the ResourceQFT class."""

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4))
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct."""
        op = qre.QFT(num_wires)
        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4))
    def test_resource_rep(self, num_wires):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.QFT, num_wires, {"num_wires": num_wires})
        assert qre.QFT.resource_rep(num_wires=num_wires) == expected

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
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(resource_rep(qre.ControlledPhaseShift)),
                ],
            ),
            (
                3,
                [
                    GateCount(resource_rep(qre.Hadamard), 3),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(resource_rep(qre.ControlledPhaseShift), 3),
                ],
            ),
        ),
    )
    def test_resources(self, num_wires, expected_res):
        """Test that the resources are correct."""
        assert qre.QFT.resource_decomp(num_wires) == expected_res

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
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.SWAP)),
                    GateCount(
                        qre.Controlled.resource_rep(
                            qre.SemiAdder.resource_rep(max_register_size=1),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        )
                    ),
                ],
            ),
            (
                3,
                [
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
                ],
            ),
        ),
    )
    def test_resources_phasegrad(self, num_wires, expected_res):
        """Test that the resources are correct for phase gradient method."""
        assert qre.QFT.phase_grad_resource_decomp(num_wires) == expected_res


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
        op = qre.AQFT(order, num_wires)
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
        expected = qre.CompressedResourceOp(
            qre.AQFT, num_wires, {"order": order, "num_wires": num_wires}
        )
        assert qre.AQFT.resource_rep(order=order, num_wires=num_wires) == expected

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
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=resource_rep(qre.S),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        4,
                    ),
                    Allocate(1),
                    GateCount(resource_rep(qre.TempAND), 1),
                    GateCount(qre.SemiAdder.resource_rep(1)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TempAND)),
                        1,
                    ),
                    Deallocate(1),
                    Allocate(2),
                    GateCount(resource_rep(qre.TempAND), 2 * 2),
                    GateCount(qre.SemiAdder.resource_rep(2), 2),
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TempAND)),
                        2 * 2,
                    ),
                    Deallocate(2),
                    GateCount(resource_rep(qre.SWAP), 2),
                ],
            ),
            (
                5,
                5,
                [
                    GateCount(resource_rep(qre.Hadamard), 5),
                    GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=resource_rep(qre.S),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        4,
                    ),
                    Allocate(1),
                    GateCount(resource_rep(qre.TempAND), 1),
                    GateCount(qre.SemiAdder.resource_rep(1)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TempAND)),
                        1,
                    ),
                    Deallocate(1),
                    Allocate(2),
                    GateCount(resource_rep(qre.TempAND), 2),
                    GateCount(qre.SemiAdder.resource_rep(2)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TempAND)),
                        2,
                    ),
                    Deallocate(2),
                    Allocate(3),
                    GateCount(resource_rep(qre.TempAND), 3),
                    GateCount(qre.SemiAdder.resource_rep(3)),
                    GateCount(resource_rep(qre.Hadamard)),
                    GateCount(
                        qre.Adjoint.resource_rep(resource_rep(qre.TempAND)),
                        3,
                    ),
                    Deallocate(3),
                    GateCount(resource_rep(qre.SWAP), 2),
                ],
            ),
        ),
    )
    def test_resources(self, order, num_wires, expected_res):
        """Test that the resources are correct."""
        assert qre.AQFT.resource_decomp(order, num_wires) == expected_res


class TestResourceBasisRotation:
    """Test the BasisRotation class."""

    @pytest.mark.parametrize("dim_n", (1, 2, 3))
    def test_resource_params(self, dim_n):
        """Test that the resource params are correct."""
        op = qre.BasisRotation(dim_n)
        assert op.resource_params == {"dim_N": dim_n}

    @pytest.mark.parametrize("dim_n", (1, 2, 3))
    def test_resource_rep(self, dim_n):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(qre.BasisRotation, dim_n, {"dim_N": dim_n})
        assert qre.BasisRotation.resource_rep(dim_N=dim_n) == expected

    @pytest.mark.parametrize("dim_n", (1, 2, 3))
    def test_resources(self, dim_n):
        """Test that the resources are correct."""
        expected = [
            GateCount(resource_rep(qre.PhaseShift), dim_n + (dim_n * (dim_n - 1) // 2)),
            GateCount(resource_rep(qre.SingleExcitation), dim_n * (dim_n - 1) // 2),
        ]
        assert qre.BasisRotation.resource_decomp(dim_n) == expected


class TestResourceSelect:
    """Test the Select class."""

    def test_resource_params(self):
        """Test that the resource params are correct."""
        ops = [qre.RX(), qre.Z(), qre.CNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        op = qre.Select(ops)
        assert op.resource_params == {"cmpr_ops": cmpr_ops, "num_wires": 4}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        ops = [qre.RX(wires=0), qre.Z(wires=1), qre.CNOT(wires=[1, 2])]
        num_wires = 3 + 2  # 3 op wires + 2 control wires
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = qre.CompressedResourceOp(
            qre.Select, num_wires, {"cmpr_ops": cmpr_ops, "num_wires": num_wires}
        )
        print(expected)
        print(qre.Select.resource_rep(cmpr_ops, num_wires))
        assert qre.Select.resource_rep(cmpr_ops, num_wires) == expected

        op = qre.Select(ops)
        print(op.resource_rep(**op.resource_params))
        assert op.resource_rep(**op.resource_params) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        ops = [qre.RX(), qre.Z(), qre.CNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = [
            qre.Allocate(1),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.RX.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.Z.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(
                qre.Controlled.resource_rep(
                    qre.CNOT.resource_rep(),
                    1,
                    0,
                )
            ),
            GateCount(qre.X.resource_rep(), 4),
            GateCount(qre.CNOT.resource_rep(), 2),
            GateCount(qre.TempAND.resource_rep(), 2),
            GateCount(
                qre.Adjoint.resource_rep(
                    qre.TempAND.resource_rep(),
                ),
                2,
            ),
            qre.Deallocate(1),
        ]
        assert qre.Select.resource_decomp(cmpr_ops, num_wires=4) == expected


class TestResourceQROM:
    """Test the ResourceQROM class."""

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
            op = qre.QROM(num_data_points, size_data_points)
        else:
            op = qre.QROM(num_data_points, size_data_points, num_bit_flips, clean, depth)

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
        expected = qre.CompressedResourceOp(
            qre.QROM,
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
            qre.QROM.resource_rep(
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
                    qre.Allocate(5),
                    GateCount(qre.Hadamard.resource_rep(), 6),
                    GateCount(qre.X.resource_rep(), 14),
                    GateCount(qre.CNOT.resource_rep(), 36),
                    GateCount(qre.TempAND.resource_rep(), 6),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TempAND.resource_rep(),
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
                    GateCount(qre.TempAND.resource_rep(), 48),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TempAND.resource_rep(),
                        ),
                        48,
                    ),
                    qre.Deallocate(5),
                    GateCount(qre.CSWAP.resource_rep(), 5),
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
                    GateCount(qre.Hadamard.resource_rep(), 4),
                    GateCount(qre.X.resource_rep(), 42),
                    GateCount(qre.CNOT.resource_rep(), 30),
                    GateCount(qre.TempAND.resource_rep(), 20),
                    GateCount(
                        qre.Adjoint.resource_rep(
                            qre.TempAND.resource_rep(),
                        ),
                        20,
                    ),
                    qre.Deallocate(3),
                    GateCount(qre.CSWAP.resource_rep(), 0),
                    qre.Deallocate(0),
                ],
            ),
            (
                12,
                2,
                5,
                128,  # This will get turncated to 16 as the max depth
                False,
                [
                    qre.Allocate(30),
                    GateCount(qre.X.resource_rep(), 5),
                    GateCount(qre.CSWAP.resource_rep(), 30),
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
        self, num_data_points, size_data_points, num_bit_flips, depth, clean, expected_res
    ):
        """Test that the resources are correct."""
        assert (
            qre.QROM.resource_decomp(
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

        opt_width = qre.QROM._t_optimized_select_swap_width(
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
        op = qre.QubitUnitary(num_wires, precision) if precision else qre.QubitUnitary(num_wires)
        assert op.resource_params == {"num_wires": num_wires, "precision": precision}

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5, 6))
    def test_resource_rep(self, num_wires, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.QubitUnitary, num_wires, {"num_wires": num_wires, "precision": precision}
        )
        assert qre.QubitUnitary.resource_rep(num_wires=num_wires, precision=precision) == expected

    @pytest.mark.parametrize(
        "num_wires, precision, expected_res",
        (
            (
                1,
                None,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-9})),
                ],
            ),
            (
                2,
                1e-3,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-3}), 4),
                    GateCount(resource_rep(qre.CNOT), 3),
                ],
            ),
            (
                5,
                1e-5,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-5}), (4**3) * 4),
                    GateCount(resource_rep(qre.CNOT), (4**3) * 3),
                    GateCount(
                        resource_rep(
                            qre.SelectPauliRot,
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
                            qre.SelectPauliRot,
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
                            qre.SelectPauliRot,
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
                            qre.SelectPauliRot,
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
                            qre.SelectPauliRot,
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
                            qre.SelectPauliRot,
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
            kwargs = config.resource_op_precisions[qre.QubitUnitary]
            assert qre.QubitUnitary.resource_decomp(num_wires=num_wires, **kwargs) == expected_res
        else:
            assert (
                qre.QubitUnitary.resource_decomp(num_wires=num_wires, precision=precision)
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
            qre.SelectPauliRot(rotation_axis, num_ctrl_wires, precision)
            if precision
            else qre.SelectPauliRot(rotation_axis, num_ctrl_wires)
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
        expected = qre.CompressedResourceOp(
            qre.SelectPauliRot,
            num_ctrl_wires + 1,
            {
                "rotation_axis": rotation_axis,
                "num_ctrl_wires": num_ctrl_wires,
                "precision": precision,
            },
        )
        assert qre.SelectPauliRot.resource_rep(num_ctrl_wires, rotation_axis, precision) == expected

    @pytest.mark.parametrize(
        "num_ctrl_wires, rotation_axis, precision, expected_res",
        (
            (
                1,
                "X",
                None,
                [
                    GateCount(resource_rep(qre.RX, {"precision": 1e-9}), 2),
                    GateCount(resource_rep(qre.CNOT), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    GateCount(resource_rep(qre.RY, {"precision": 1e-3}), 2**2),
                    GateCount(resource_rep(qre.CNOT), 2**2),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-5}), 2**5),
                    GateCount(resource_rep(qre.CNOT), 2**5),
                ],
            ),
        ),
    )
    def test_default_resources(self, num_ctrl_wires, rotation_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qre.SelectPauliRot]
            assert (
                qre.SelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rotation_axis=rotation_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                qre.SelectPauliRot.resource_decomp(
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
                    Allocate(33),
                    GateCount(qre.QROM.resource_rep(2, 33, 33, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(33),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(2, 33, 33, False),
                            },
                        )
                    ),
                    Deallocate(33),
                    GateCount(resource_rep(qre.Hadamard), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    Allocate(13),
                    GateCount(qre.QROM.resource_rep(4, 13, 26, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(13),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(4, 13, 26, False),
                            },
                        )
                    ),
                    Deallocate(13),
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.S)),
                    GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)})),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    Allocate(20),
                    GateCount(qre.QROM.resource_rep(32, 20, 320, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(20),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(32, 20, 320, False),
                            },
                        )
                    ),
                    Deallocate(20),
                ],
            ),
        ),
    )
    def test_phase_gradient_resources(self, num_ctrl_wires, rotation_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qre.SelectPauliRot]
            assert (
                qre.SelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rotation_axis=rotation_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                qre.SelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rotation_axis=rotation_axis,
                    precision=precision,
                )
                == expected_res
            )
