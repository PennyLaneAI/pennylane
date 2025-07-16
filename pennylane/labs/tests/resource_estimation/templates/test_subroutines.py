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
            plre.ResourceOutOfPlaceSquare, {"register_size": register_size}
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
        expected = plre.CompressedResourceOp(plre.ResourcePhaseGradient, {"num_wires": num_wires})
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
        expected = plre.CompressedResourceOp(
            plre.ResourceOutMultiplier,
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
            plre.ResourceSemiAdder, {"max_register_size": register_size}
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
        expected = plre.CompressedResourceOp(plre.ResourceBasisRotation, {"dim_N": dim_n})
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
        assert op.resource_params == {"cmpr_ops": cmpr_ops}

    def test_resource_rep(self):
        """Test that the compressed representation is correct."""
        ops = [plre.ResourceRX(), plre.ResourceZ(), plre.ResourceCNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = plre.CompressedResourceOp(plre.ResourceSelect, {"cmpr_ops": cmpr_ops})
        assert plre.ResourceSelect.resource_rep(cmpr_ops) == expected

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
        assert plre.ResourceSelect.resource_decomp(cmpr_ops) == expected


class TestResourceQROM:
    """Test the ResourceQROM class."""

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
        expected = plre.CompressedResourceOp(
            plre.ResourceQROM,
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
                True,  # AllocWires(3), (4 x Hadamard), (42 x X), (30 x CNOT), (20 x TempAND), (20 x Adjoint(TempAND)), FreeWires(3), (0 x CSWAP), FreeWires(0)
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


class TestResourceQubitUnitary:
    """Test the ResourceQubitUnitary template"""

    @pytest.mark.parametrize("eps", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5, 6))
    def test_resource_params(self, num_wires, eps):
        """Test that the resource params are correct."""
        op = (
            plre.ResourceQubitUnitary(num_wires, eps)
            if eps
            else plre.ResourceQubitUnitary(num_wires)
        )
        assert op.resource_params == {"num_wires": num_wires, "precision": eps}

    @pytest.mark.parametrize("eps", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("num_wires", (1, 2, 3, 4, 5, 6))
    def test_resource_rep(self, num_wires, eps):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceQubitUnitary, {"num_wires": num_wires, "precision": eps}
        )
        assert (
            plre.ResourceQubitUnitary.resource_rep(num_wires=num_wires, precision=eps) == expected
        )

    @pytest.mark.parametrize(
        "num_wires, eps, expected_res",
        (
            (
                1,
                None,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"eps": 1e-9})),
                ],
            ),
            (
                2,
                1e-3,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"eps": 1e-3}), 4),
                    GateCount(resource_rep(plre.ResourceCNOT), 3),
                ],
            ),
            (
                5,
                1e-5,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"eps": 1e-5}), (4**3) * 4),
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
    def test_default_resources(self, num_wires, eps, expected_res):
        """Test that the resources are correct."""
        if eps is None:
            config = {"precision_qubit_unitary": 1e-9}
            assert (
                plre.ResourceQubitUnitary.resource_decomp(
                    num_wires=num_wires, precision=eps, config=config
                )
                == expected_res
            )
        else:
            assert (
                plre.ResourceQubitUnitary.resource_decomp(num_wires=num_wires, precision=eps)
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
                    GateCount(resource_rep(plre.ResourceRX, {"eps": 1e-9}), 2),
                    GateCount(resource_rep(plre.ResourceCNOT), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    GateCount(resource_rep(plre.ResourceRY, {"eps": 1e-3}), 2**2),
                    GateCount(resource_rep(plre.ResourceCNOT), 2**2),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    GateCount(resource_rep(plre.ResourceRZ, {"eps": 1e-5}), 2**5),
                    GateCount(resource_rep(plre.ResourceCNOT), 2**5),
                ],
            ),
        ),
    )
    def test_default_resources(self, num_ctrl_wires, rotation_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = {"precision_select_pauli_rot": 1e-9}
            assert (
                plre.ResourceSelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rotation_axis=rotation_axis,
                    precision=precision,
                    config=config,
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
                    AllocWires(30),
                    GateCount(plre.ResourceQROM.resource_rep(2, 30, 30, False)),
                    GateCount(
                        resource_rep(
                            plre.ResourceControlled,
                            {
                                "base_cmpr_op": plre.ResourceSemiAdder.resource_rep(30),
                                "num_ctrl_wires": 1,
                                "num_ctrl_values": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": plre.ResourceQROM.resource_rep(2, 30, 30, False),
                            },
                        )
                    ),
                    FreeWires(30),
                    GateCount(resource_rep(plre.ResourceHadamard), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    AllocWires(10),
                    GateCount(plre.ResourceQROM.resource_rep(4, 10, 20, False)),
                    GateCount(
                        resource_rep(
                            plre.ResourceControlled,
                            {
                                "base_cmpr_op": plre.ResourceSemiAdder.resource_rep(10),
                                "num_ctrl_wires": 1,
                                "num_ctrl_values": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": plre.ResourceQROM.resource_rep(4, 10, 20, False),
                            },
                        )
                    ),
                    FreeWires(10),
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
                    AllocWires(17),
                    GateCount(plre.ResourceQROM.resource_rep(32, 17, 272, False)),
                    GateCount(
                        resource_rep(
                            plre.ResourceControlled,
                            {
                                "base_cmpr_op": plre.ResourceSemiAdder.resource_rep(17),
                                "num_ctrl_wires": 1,
                                "num_ctrl_values": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            plre.ResourceAdjoint,
                            {
                                "base_cmpr_op": plre.ResourceQROM.resource_rep(32, 17, 272, False),
                            },
                        )
                    ),
                    FreeWires(17),
                ],
            ),
        ),
    )
    def test_phase_gradient_resources(self, num_ctrl_wires, rotation_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = {"precision_select_pauli_rot": 1e-9}
            assert (
                plre.ResourceSelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rotation_axis=rotation_axis,
                    precision=precision,
                    config=config,
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
