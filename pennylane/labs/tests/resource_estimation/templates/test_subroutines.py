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
            plre.GateCount(plre.resource_rep(plre.ResourceToffoli), (register_size - 1) ** 2),
            plre.GateCount(plre.resource_rep(plre.ResourceCNOT), register_size),
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
                    plre.GateCount(plre.ResourceHadamard.resource_rep()),
                    plre.GateCount(plre.ResourceZ.resource_rep()),
                ],
            ),
            (
                2,
                [
                    plre.GateCount(plre.ResourceHadamard.resource_rep(), 2),
                    plre.GateCount(plre.ResourceZ.resource_rep()),
                    plre.GateCount(plre.ResourceS.resource_rep()),
                ],
            ),
            (
                3,
                [
                    plre.GateCount(plre.ResourceHadamard.resource_rep(), 3),
                    plre.GateCount(plre.ResourceZ.resource_rep()),
                    plre.GateCount(plre.ResourceS.resource_rep()),
                    plre.GateCount(plre.ResourceT.resource_rep()),
                ],
            ),
            (
                5,
                [
                    plre.GateCount(plre.ResourceHadamard.resource_rep(), 5),
                    plre.GateCount(plre.ResourceZ.resource_rep()),
                    plre.GateCount(plre.ResourceS.resource_rep()),
                    plre.GateCount(plre.ResourceT.resource_rep()),
                    plre.GateCount(plre.ResourceRZ.resource_rep(), 2),
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

        toff = plre.resource_rep(plre.ResourceToffoli)
        l_elbow = plre.resource_rep(plre.ResourceTempAND)
        r_elbow = plre.resource_rep(plre.ResourceAdjoint, {"base_cmpr_op": l_elbow})

        num_elbows = 12
        num_toff = 1

        expected = [
            plre.GateCount(l_elbow, num_elbows),
            plre.GateCount(r_elbow, num_elbows),
            plre.GateCount(toff, num_toff),
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
                [plre.GateCount(plre.resource_rep(plre.ResourceCNOT))],
            ),
            (
                2,
                [
                    plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 2),
                    plre.GateCount(plre.resource_rep(plre.ResourceX), 2),
                    plre.GateCount(plre.resource_rep(plre.ResourceToffoli)),
                ],
            ),
            (
                3,
                [
                    plre.AllocWires(2),
                    plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 9),
                    plre.GateCount(plre.resource_rep(plre.ResourceTempAND), 2),
                    plre.GateCount(
                        plre.resource_rep(
                            plre.ResourceAdjoint,
                            {"base_cmpr_op": plre.resource_rep(plre.ResourceTempAND)},
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
            plre.GateCount(plre.resource_rep(plre.ResourceCNOT), 24),
            plre.GateCount(plre.resource_rep(plre.ResourceTempAND), 8),
            plre.GateCount(
                plre.resource_rep(
                    plre.ResourceAdjoint, {"base_cmpr_op": plre.resource_rep(plre.ResourceTempAND)}
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
            plre.GateCount(
                plre.resource_rep(plre.ResourcePhaseShift), dim_n + (dim_n * (dim_n - 1) // 2)
            ),
            plre.GateCount(
                plre.resource_rep(plre.ResourceSingleExcitation), dim_n * (dim_n - 1) // 2
            ),
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
        assert plre.ResourceSelect.resource_rep(ops) == expected

    def test_resources(self):
        """Test that the resources are correct."""
        ops = [plre.ResourceRX(), plre.ResourceZ(), plre.ResourceCNOT()]
        cmpr_ops = tuple(op.resource_rep_from_op() for op in ops)

        expected = [plre.AllocWires(2)]
        assert plre.ResourceSelect.resource_decomp(ops) == expected


# class TestResourceQROM:
#     """Test the OutOfPlaceSquare class."""

#     @pytest.mark.parametrize("register_size", (1, 2, 3))
#     def test_resource_params(self, register_size):
#         """Test that the resource params are correct."""
#         op = plre.ResourceOutOfPlaceSquare(register_size)
#         assert op.resource_params == {"register_size": register_size}

#     @pytest.mark.parametrize("register_size", (1, 2, 3))
#     def test_resource_rep(self, register_size):
#         """Test that the compressed representation is correct."""
#         expected = plre.CompressedResourceOp(plre.ResourceOutOfPlaceSquare, {"register_size": register_size})
#         assert plre.ResourceOutOfPlaceSquare.resource_rep(register_size=register_size) == expected

#     @pytest.mark.parametrize("register_size", (1, 2, 3))
#     def test_resources(self, register_size):
#         """Test that the resources are correct."""
#         expected = [
#             plre.GateCount(plre.resource_rep(plre.ResourceToffoli), (register_size - 1)**2),
#             plre.GateCount(plre.resource_rep(plre.ResourceCNOT), register_size),

#         ]
#         assert plre.ResourceOutOfPlaceSquare.resource_decomp(register_size=register_size) == expected
