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
"""Tests for Quantum Signal Processing (QSP) resource operators."""

import pytest
import pennylane.estimator as qre
from pennylane.estimator.resource_operator import ResourceOperator, CompressedResourceOp


class DummyOp(ResourceOperator):
    def __init__(self, wires):
        self.num_wires = len(wires)
        super().__init__(wires=wires)

    @property
    def resource_params(self):
        return {}

    @classmethod
    def resource_rep(cls):
        return CompressedResourceOp(cls, 1, {})

    @classmethod
    def resource_decomp(cls, **kwargs):
        return []


class TestQSVT:
    """Test the QSVT class."""

    def test_init_error_encoding_dims(self):
        """Test that an error is raised when encoding_dims is invalid."""
        op = DummyOp(wires=[0])
        with pytest.raises(
            ValueError, match="Expected `encoding_dims` to be an int or tuple of int"
        ):
            qre.QSVT(op, encoding_dims="invalid", poly_deg=2)

        with pytest.raises(
            ValueError, match="Expected `encoding_dims` to be a tuple of two integers"
        ):
            qre.QSVT(op, encoding_dims=(1, 2, 3), poly_deg=2)

    def test_init_encoding_dims_tuple_len_1(self):
        """Test that encoding_dims is correctly handled when it is a tuple of length 1."""
        op = DummyOp(wires=[0])
        qsvt = qre.QSVT(op, encoding_dims=(2,), poly_deg=2)
        assert qsvt.resource_params["encoding_dims"] == (2, 2)

    def test_init_wires_mismatch(self):
        """Test that an error is raised when wires don't match block encoding."""
        op = DummyOp(wires=[0])
        with pytest.raises(ValueError, match="Expected 1 wires, got 2"):
            qre.QSVT(op, encoding_dims=2, poly_deg=2, wires=[0, 1])

    @pytest.mark.parametrize(
        "encoding_dims, poly_deg",
        [
            (2, 3),
            ((2, 4), 2),
        ],
    )
    def test_resource_params(self, encoding_dims, poly_deg):
        """Test that the resource params are correct."""
        op = DummyOp(wires=[0])
        qsvt = qre.QSVT(op, encoding_dims=encoding_dims, poly_deg=poly_deg)

        expected_dims = (
            (encoding_dims, encoding_dims) if isinstance(encoding_dims, int) else encoding_dims
        )

        assert qsvt.resource_params == {
            "block_encoding": op.resource_rep(),
            "encoding_dims": expected_dims,
            "poly_deg": poly_deg,
        }

    @pytest.mark.parametrize(
        "encoding_dims, poly_deg",
        [
            (2, 3),
            ((2, 4), 2),
        ],
    )
    def test_resource_rep(self, encoding_dims, poly_deg):
        """Test that the compressed representation is correct."""
        op = DummyOp(wires=[0])
        expected_dims = (
            (encoding_dims, encoding_dims) if isinstance(encoding_dims, int) else encoding_dims
        )

        expected = qre.CompressedResourceOp(
            qre.QSVT,
            1,
            {
                "block_encoding": op.resource_rep(),
                "encoding_dims": expected_dims,
                "poly_deg": poly_deg,
            },
        )
        assert qre.QSVT.resource_rep(op.resource_rep(), expected_dims, poly_deg) == expected

    @pytest.mark.parametrize(
        "encoding_dims, poly_deg, expected_res",
        [
            (
                2,
                3,
                [
                    qre.GateCount(DummyOp.resource_rep(), 2),
                    qre.GateCount(qre.Adjoint.resource_rep(DummyOp.resource_rep()), 1),
                    qre.GateCount(qre.PCPhase.resource_rep(1, 2, None), 1),
                    qre.GateCount(qre.PCPhase.resource_rep(1, 2, None), 2),
                ],
            ),
            (
                (2, 4),
                2,
                [
                    qre.GateCount(DummyOp.resource_rep(), 1),
                    qre.GateCount(qre.Adjoint.resource_rep(DummyOp.resource_rep()), 1),
                    qre.GateCount(qre.PCPhase.resource_rep(1, 4, None), 1),
                    qre.GateCount(qre.PCPhase.resource_rep(1, 2, None), 1),
                ],
            ),
        ],
    )
    def test_resources(self, encoding_dims, poly_deg, expected_res):
        """Test that the resources are correct."""
        op = DummyOp(wires=[0])
        dims = (encoding_dims, encoding_dims) if isinstance(encoding_dims, int) else encoding_dims

        res = qre.QSVT.resource_decomp(op.resource_rep(), dims, poly_deg)
        assert res == expected_res


class TestQSP:
    """Test the QSP class."""

    def test_init_error_block_encoding_wires(self):
        """Test that an error is raised when block encoding has > 1 wires."""

        class MultiWireOp(ResourceOperator):
            num_wires = 2

            def __init__(self, wires):
                super().__init__(wires=wires)

            @property
            def resource_params(self):
                return {}

            @classmethod
            def resource_rep(cls):
                return CompressedResourceOp(cls, 2, {})

            @classmethod
            def resource_decomp(cls, **kwargs):
                return []

        op = MultiWireOp(wires=[0, 1])
        with pytest.raises(
            ValueError, match="The block encoding operator should act on a single qubit"
        ):
            qre.QSP(op, poly_deg=2)

    def test_init_error_convention(self):
        """Test that an error is raised when convention is invalid."""
        op = DummyOp(wires=[0])
        with pytest.raises(ValueError, match="The valid conventions are 'Z' or 'X'"):
            qre.QSP(op, poly_deg=2, convention="Y")

    def test_init_wires_mismatch(self):
        """Test that an error is raised when wires don't match block encoding."""
        op = DummyOp(wires=[0])
        with pytest.raises(ValueError, match="Expected 1 wires, got 2"):
            qre.QSP(op, poly_deg=2, wires=[0, 1])

    @pytest.mark.parametrize(
        "poly_deg, convention, rot_precision",
        [
            (3, "Z", 1e-5),
            (2, "X", 1e-5),
        ],
    )
    def test_resource_params(self, poly_deg, convention, rot_precision):
        """Test that the resource params are correct."""
        op = DummyOp(wires=[0])
        qsp = qre.QSP(
            op, poly_deg=poly_deg, convention=convention, rotation_precision=rot_precision
        )

        assert qsp.resource_params == {
            "block_encoding": op.resource_rep(),
            "poly_deg": poly_deg,
            "convention": convention,
            "rotation_precision": rot_precision,
        }

    @pytest.mark.parametrize(
        "poly_deg, convention, rot_precision",
        [
            (3, "Z", 1e-5),
            (2, "X", 1e-5),
        ],
    )
    def test_resource_rep(self, poly_deg, convention, rot_precision):
        """Test that the compressed representation is correct."""
        op = DummyOp(wires=[0])
        expected = qre.CompressedResourceOp(
            qre.QSP,
            1,
            {
                "block_encoding": op.resource_rep(),
                "poly_deg": poly_deg,
                "convention": convention,
                "rotation_precision": rot_precision,
            },
        )
        assert (
            qre.QSP.resource_rep(op.resource_rep(), poly_deg, convention, rot_precision) == expected
        )

    @pytest.mark.parametrize(
        "poly_deg, convention, rot_precision, expected_res",
        [
            (
                3,
                "Z",
                1e-5,
                [
                    qre.GateCount(DummyOp.resource_rep(), 3),
                    qre.GateCount(qre.RZ.resource_rep(1e-5), 4),
                ],
            ),
            (
                2,
                "X",
                1e-5,
                [
                    qre.GateCount(DummyOp.resource_rep(), 2),
                    qre.GateCount(qre.RX.resource_rep(1e-5), 3),
                ],
            ),
        ],
    )
    def test_resources(self, poly_deg, convention, rot_precision, expected_res):
        """Test that the resources are correct."""
        op = DummyOp(wires=[0])
        res = qre.QSP.resource_decomp(op.resource_rep(), poly_deg, convention, rot_precision)
        assert res == expected_res
