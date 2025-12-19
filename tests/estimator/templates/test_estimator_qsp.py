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
Tests for select resource operators.
"""
import pytest

import pennylane.estimator as qre

# pylint: disable=no-self-use, too-many-arguments


class TestGQSP:
    """Test the GQSP class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(ValueError, match="Expected 2 wires, got 1"):
            qre.GQSP(op, poly_deg=2, wires=[0])

    @pytest.mark.parametrize("rot_prec", (0, -1, -2.5))
    def test_rotation_precision_error(self, rot_prec):
        """Test that an error is raised if the rotation_precision is negative"""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(
            ValueError,
            match="Expected 'rotation_precision' to be a positive real number greater than zero",
        ):
            _ = qre.GQSP(op, poly_deg=2, rotation_precision=rot_prec)

    @pytest.mark.parametrize(
        "poly_deg, neg_poly_deg, error_msg",
        (
            (0.1, 2, "'poly_deg' must be a positive integer greater than zero,"),
            (-3, 3, "'poly_deg' must be a positive integer greater than zero,"),
            (0, 5, "'poly_deg' must be a positive integer greater than zero,"),
            (1, 0.5, "'neg_poly_deg' must be a positive integer,"),
            (2, -3, "'neg_poly_deg' must be a positive integer,"),
        ),
    )
    def test_poly_deg_error(self, poly_deg, neg_poly_deg, error_msg):
        """Test that an error is raised of incompatible values are
        passed for 'poly_deg' and 'neg_poly_deg'."""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(ValueError, match=error_msg):
            _ = qre.GQSP(op, poly_deg, neg_poly_deg)

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

        assert gqsp.resource_params["poly_deg"] == poly_deg
        assert gqsp.resource_params["neg_poly_deg"] == neg_poly_deg
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
                "poly_deg": poly_deg,
                "neg_poly_deg": neg_poly_deg,
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
