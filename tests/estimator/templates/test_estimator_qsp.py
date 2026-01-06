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
from pennylane.estimator.resource_operator import CompressedResourceOp, ResourceOperator

# pylint: disable=no-self-use, too-many-arguments


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

    # pylint: disable=unused-argument
    @classmethod
    def resource_decomp(cls, **kwargs):
        return []


class TestQSVT:
    """Test the QSVT class."""

    @pytest.mark.parametrize(
        "dims, poly_deg, error_type, error_msg",
        [
            (0.5, 5, TypeError, "Expected `encoding_dims` to be an integer or tuple of integers."),
            ((1, 2, 3), 5, ValueError, "Expected `encoding_dims` to be a tuple of two integers"),
            ((0,), 5, ValueError, "Expected elements of `encoding_dims` to be positive integers."),
            (
                (-2, 2),
                5,
                ValueError,
                "Expected elements of `encoding_dims` to be positive integers.",
            ),
            (2, -1, ValueError, "'poly_deg' must be a positive integer greater than zero, got -1"),
            (
                2,
                4.5,
                ValueError,
                "'poly_deg' must be a positive integer greater than zero, got 4.5",
            ),
        ],
    )
    def test_init_failures(self, dims, poly_deg, error_type, error_msg):
        """Test all invalid inputs raise the correct exceptions."""
        op = DummyOp(wires=[0])
        with pytest.raises(error_type, match=error_msg):
            qre.QSVT(block_encoding=op, encoding_dims=dims, poly_deg=poly_deg)

        with pytest.raises(error_type, match=error_msg):
            qre.QSVT.resource_rep(
                block_encoding=op.resource_rep(), encoding_dims=dims, poly_deg=poly_deg
            )

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

            # pylint: disable=unused-argument
            @classmethod
            def resource_decomp(cls, **kwargs):
                return []

        op = MultiWireOp(wires=[0, 1])
        with pytest.raises(
            ValueError, match="The block encoding operator should act on a single qubit"
        ):
            qre.QSP(op, poly_deg=2)

        with pytest.raises(
            ValueError, match="The block encoding operator should act on a single qubit"
        ):
            qre.QSP.resource_rep(op, poly_deg=2)

    def test_init_error_convention(self):
        """Test that an error is raised when convention is invalid."""
        op = DummyOp(wires=[0])
        with pytest.raises(ValueError, match="The valid conventions are 'Z' or 'X'"):
            qre.QSP(op, poly_deg=2, convention="Y")

        with pytest.raises(ValueError, match="The valid conventions are 'Z' or 'X'"):
            qre.QSP.resource_rep(op, poly_deg=2, convention="Y")

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

    def test_resource_decomp_error_convention(self):
        """Test that an error is raised in resource_decomp when convention is invalid."""
        op = DummyOp(wires=[0])
        with pytest.raises(ValueError, match="The valid conventions are 'Z' or 'X'"):
            qre.QSP.resource_decomp(
                op.resource_rep(), poly_deg=2, convention="Y", rotation_precision=1e-5
            )


class TestGQSP:
    """Test the GQSP class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(ValueError, match="Expected 2 wires, got 1"):
            qre.GQSP(op, d_plus=2, wires=[0])

    @pytest.mark.parametrize("rot_prec", (0, -1, -2.5))
    def test_rotation_precision_error(self, rot_prec):
        """Test that an error is raised if the rotation_precision is negative"""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(
            ValueError,
            match="Expected 'rotation_precision' to be a positive real number greater than zero",
        ):
            _ = qre.GQSP(op, d_plus=2, rotation_precision=rot_prec)

        with pytest.raises(
            ValueError,
            match="Expected 'rotation_precision' to be a positive real number greater than zero",
        ):
            _ = qre.GQSP.resource_rep(op, d_plus=2, rotation_precision=rot_prec)

    @pytest.mark.parametrize(
        "d_plus, d_minus, error_msg",
        (
            (0.1, 2, "'d_plus' must be a positive integer greater than zero,"),
            (-3, 3, "'d_plus' must be a positive integer greater than zero,"),
            (0, 5, "'d_plus' must be a positive integer greater than zero,"),
            (1, 0.5, "'d_minus' must be a non-negative integer,"),
            (2, -3, "'d_minus' must be a non-negative integer,"),
        ),
    )
    def test_d_error(self, d_plus, d_minus, error_msg):
        """Test that an error is raised of incompatible values are
        passed for 'poly_deg' and 'neg_poly_deg'."""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(ValueError, match=error_msg):
            _ = qre.GQSP(op, d_plus, d_minus)

        with pytest.raises(ValueError, match=error_msg):
            _ = qre.GQSP.resource_rep(op, d_plus, d_minus)

    @pytest.mark.parametrize(
        "poly_deg, neg_poly_deg, rot_precision",
        (
            (5, 0, 1e-5),
            (10, 5, None),
        ),
    )
    def test_resource_params(self, poly_deg, neg_poly_deg, rot_precision):
        """Test that the resource params for GQSP are correct."""
        op = qre.RX(0.1, wires=0)
        gqsp = qre.GQSP(op, poly_deg, neg_poly_deg, rot_precision)

        assert gqsp.resource_params["d_plus"] == poly_deg
        assert gqsp.resource_params["d_minus"] == neg_poly_deg
        assert gqsp.resource_params["rotation_precision"] == rot_precision
        assert gqsp.resource_params["cmpr_signal_op"].op_type == qre.RX

    @pytest.mark.parametrize(
        "poly_deg, neg_poly_deg, rot_precision",
        (
            (5, 0, 1e-5),
            (10, 5, None),
        ),
    )
    def test_resource_rep(self, poly_deg, neg_poly_deg, rot_precision):
        """Test that the compressed representation for GQSP is correct."""
        op = qre.RX(0.1, wires=0)
        cmpr_op = op.resource_rep_from_op()

        expected = qre.CompressedResourceOp(
            qre.GQSP,
            2,  # 1 wire for RX + 1 control
            {
                "cmpr_signal_op": cmpr_op,
                "d_plus": poly_deg,
                "d_minus": neg_poly_deg,
                "rotation_precision": rot_precision,
            },
        )
        assert qre.GQSP.resource_rep(cmpr_op, poly_deg, neg_poly_deg, rot_precision) == expected

    @pytest.mark.parametrize(
        "poly_deg, neg_poly_deg, expected_res",
        (
            (
                5,
                0,
                [
                    qre.GateCount(qre.Rot.resource_rep(), 6),
                    qre.GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.RX.resource_rep(0.1),
                            num_ctrl_wires=1,
                            num_zero_ctrl=1,
                        ),
                        5,
                    ),
                ],
            ),
            (
                5,
                3,
                [
                    qre.GateCount(qre.Rot.resource_rep(), 9),
                    qre.GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.RX.resource_rep(0.1),
                            num_ctrl_wires=1,
                            num_zero_ctrl=1,
                        ),
                        5,
                    ),
                    qre.GateCount(
                        qre.Controlled.resource_rep(
                            base_cmpr_op=qre.Adjoint.resource_rep(qre.RX.resource_rep(0.1)),
                            num_ctrl_wires=1,
                            num_zero_ctrl=0,
                        ),
                        3,
                    ),
                ],
            ),
        ),
    )
    def test_resources(self, poly_deg, neg_poly_deg, expected_res):
        """Test the resource decomposition."""
        op = qre.RX(0.1, wires=0)
        cmpr_op = op.resource_rep_from_op()
        decomp = qre.GQSP.resource_decomp(cmpr_op, poly_deg, neg_poly_deg, None)
        assert decomp == expected_res


class TestGQSPTimeEvolution:
    """Test the GQSPTimeEvolution class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(ValueError, match="Expected 2 wires, got 1"):
            qre.GQSPTimeEvolution(op, time=1.0, one_norm=1.0, poly_approx_precision=0.1, wires=[0])

    @pytest.mark.parametrize(
        "time, one_norm, poly_approx_precision, error_msg",
        (
            (-1, 1.0, 0.1, "Expected 'time' to be a positive real number greater than zero"),
            (0, 1.0, 0.1, "Expected 'time' to be a positive real number greater than zero"),
            (10 + 1j, 1.0, 0.1, "Expected 'time' to be a positive real number greater than zero"),
            (10, -2, 0.1, "Expected 'one_norm' to be a positive real number greater than zero"),
            (10, 0, 0.1, "Expected 'one_norm' to be a positive real number greater than zero"),
            (
                10,
                1.0 + 1j,
                0.1,
                "Expected 'one_norm' to be a positive real number greater than zero",
            ),
            (
                10,
                1.0,
                -0.5,
                "Expected 'poly_approx_precision' to be a positive real number greater than zero",
            ),
            (
                10,
                1.0,
                0,
                "Expected 'poly_approx_precision' to be a positive real number greater than zero",
            ),
            (
                10,
                1.0,
                0.1 + 1j,
                "Expected 'poly_approx_precision' to be a positive real number greater than zero",
            ),
        ),
    )
    def test_argument_error(self, time, one_norm, poly_approx_precision, error_msg):
        """Test that an error is raised if any of the input arguments are not positive real
        numbers greater than zero."""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(ValueError, match=error_msg):
            qre.GQSPTimeEvolution(op, time, one_norm, poly_approx_precision)

        with pytest.raises(ValueError, match=error_msg):
            qre.GQSPTimeEvolution.resource_rep(op, time, one_norm, poly_approx_precision)

    def test_resource_params(self):
        """Test that the resource params for GQSPTimeEvolution are correct."""
        op = qre.RX(0.1, wires=0)
        hamsim = qre.GQSPTimeEvolution(op, time=1.0, one_norm=2.0, poly_approx_precision=0.01)

        assert hamsim.resource_params["time"] == 1.0
        assert hamsim.resource_params["one_norm"] == 2.0
        assert hamsim.resource_params["poly_approx_precision"] == 0.01
        assert hamsim.resource_params["walk_op"].op_type == qre.RX

    def test_resource_rep(self):
        """Test that the compressed representation for GQSPTimeEvolution is correct."""
        op = qre.RX(0.1, wires=0)
        cmpr_op = op.resource_rep_from_op()

        expected = qre.CompressedResourceOp(
            qre.GQSPTimeEvolution,
            2,
            {
                "walk_op": cmpr_op,
                "time": 1.0,
                "one_norm": 2.0,
                "poly_approx_precision": 0.01,
            },
        )
        assert qre.GQSPTimeEvolution.resource_rep(cmpr_op, 1.0, 2.0, 0.01) == expected

    def test_poly_approx(self):
        """Test poly_approx returns a positive integer."""
        deg = qre.GQSPTimeEvolution.poly_approx(time=1.0, one_norm=1.0, epsilon=0.1)
        assert isinstance(deg, int)
        assert deg > 0

    def test_resources(self):
        """Test the resource decomposition."""
        op = qre.RX(0.1, wires=0)
        cmpr_op = op.resource_rep_from_op()

        decomp = qre.GQSPTimeEvolution.resource_decomp(
            cmpr_op, time=1.0, one_norm=1.0, poly_approx_precision=0.1
        )
        assert len(decomp) == 1
        assert decomp[0].gate.op_type == qre.GQSP
