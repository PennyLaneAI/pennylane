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


class TestHamiltonianGQSP:
    """Test the HamiltonianGQSP class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        op = qre.RX(0.1, wires=0)
        with pytest.raises(ValueError, match="Expected 2 wires, got 1"):
            qre.HamiltonianGQSP(op, time=1.0, one_norm=1.0, approximation_error=0.1, wires=[0])

    def test_resource_params(self):
        """Test that the resource params for HamiltonianGQSP are correct."""
        op = qre.RX(0.1, wires=0)
        hamsim = qre.HamiltonianGQSP(op, time=1.0, one_norm=2.0, approximation_error=0.01)

        assert hamsim.resource_params["time"] == 1.0
        assert hamsim.resource_params["one_norm"] == 2.0
        assert hamsim.resource_params["approximation_error"] == 0.01
        assert hamsim.resource_params["walk_op"].op_type == qre.RX

    def test_resource_rep(self):
        """Test that the compressed representation for HamiltonianGQSP is correct."""
        op = qre.RX(0.1, wires=0)
        cmpr_op = op.resource_rep_from_op()

        expected = qre.CompressedResourceOp(
            qre.HamiltonianGQSP,
            2,
            {
                "walk_op": cmpr_op,
                "time": 1.0,
                "one_norm": 2.0,
                "approximation_error": 0.01,
            },
        )
        assert qre.HamiltonianGQSP.resource_rep(cmpr_op, 1.0, 2.0, 0.01) == expected

    def test_degree_of_poly_approx(self):
        """Test degree_of_poly_approx returns a positive integer."""
        deg = qre.HamiltonianGQSP.degree_of_poly_approx(time=1.0, one_norm=1.0, epsilon=0.1)
        assert isinstance(deg, int)
        assert deg > 0

    def test_resources(self):
        """Test the resource decomposition."""
        op = qre.RX(0.1, wires=0)
        cmpr_op = op.resource_rep_from_op()

        decomp = qre.HamiltonianGQSP.resource_decomp(
            cmpr_op, time=1.0, one_norm=1.0, approximation_error=0.1
        )
        assert len(decomp) == 1
        assert decomp[0].gate.op_type == qre.GQSP
