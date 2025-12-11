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
from collections import defaultdict
import pytest

import pennylane.estimator as qre

# pylint: disable=no-self-use, too-many-arguments


class TestSelectTHC:
    """Test the SelectTHC class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        ch = qre.THCHamiltonian(2, 4)
        with pytest.raises(ValueError, match="Expected 16 wires, got 3"):
            qre.ControlledSequence(base=qre.SelectTHC(ch, wires=[0, 1, 2]))

    @pytest.mark.parametrize(
        "thc_ham, rotation_prec, selswap_depth",
        (
            (qre.THCHamiltonian(58, 160), 13, 1),
            (qre.THCHamiltonian(10, 50), None, None),
            (qre.THCHamiltonian(4, 20), None, 2),
        ),
    )
    def test_resource_params(self, thc_ham, rotation_prec, selswap_depth):
        """Test that the resource params for SelectTHC are correct."""
        op = qre.SelectTHC(thc_ham, rotation_prec, selswap_depth)
        assert op.resource_params == {
            "thc_ham": thc_ham,
            "rotation_precision": rotation_prec,
            "select_swap_depth": selswap_depth,
        }

    @pytest.mark.parametrize(
        "thc_ham, rotation_prec, selswap_depth, num_wires",
        (
            (qre.THCHamiltonian(58, 160), 13, 1, 138),
            (qre.THCHamiltonian(10, 50), None, None, 38),
            (qre.THCHamiltonian(4, 20), None, 2, 24),
        ),
    )
    def test_resource_rep(self, thc_ham, rotation_prec, selswap_depth, num_wires):
        """Test that the compressed representation for SelectTHC is correct."""
        expected = qre.CompressedResourceOp(
            qre.SelectTHC,
            num_wires,
            {
                "thc_ham": thc_ham,
                "rotation_precision": rotation_prec,
                "select_swap_depth": selswap_depth,
            },
        )
        assert qre.SelectTHC.resource_rep(thc_ham, rotation_prec, selswap_depth) == expected

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, rotation_prec, selswap_depth, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                13,
                1,
                {"algo_wires": 138, "auxiliary_wires": 752, "toffoli_gates": 5997},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                None,
                {"algo_wires": 38, "auxiliary_wires": 163, "toffoli_gates": 1139},
            ),
            (
                qre.THCHamiltonian(4, 20),
                None,
                2,
                {"algo_wires": 24, "auxiliary_wires": 73, "toffoli_gates": 425},
            ),
        ),
    )
    def test_resources(self, thc_ham, rotation_prec, selswap_depth, expected_res):
        """Test that the resource decompostion for SelectTHC is correct."""

        select_cost = qre.estimate(
            qre.SelectTHC(
                thc_ham, rotation_precision=rotation_prec, select_swap_depth=selswap_depth
            )
        )
        assert select_cost.algo_wires == expected_res["algo_wires"]
        assert (
            select_cost.zeroed_wires + select_cost.any_state_wires
            == expected_res["auxiliary_wires"]
        )
        assert select_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, rotation_prec, selswap_depth, num_ctrl_wires, num_zero_ctrl, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                13,
                1,
                1,
                1,
                {"algo_wires": 139, "auxiliary_wires": 752, "toffoli_gates": 5998},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                None,
                2,
                0,
                {"algo_wires": 40, "auxiliary_wires": 164, "toffoli_gates": 1142},
            ),
            (
                qre.THCHamiltonian(4, 20),
                None,
                2,
                3,
                2,
                {"algo_wires": 27, "auxiliary_wires": 74, "toffoli_gates": 430},
            ),
        ),
    )
    def test_controlled_resources(
        self, thc_ham, rotation_prec, selswap_depth, num_ctrl_wires, num_zero_ctrl, expected_res
    ):
        """Test that the controlled resource decompostion for SelectTHC is correct."""

        ctrl_select_cost = qre.estimate(
            qre.Controlled(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                base_op=qre.SelectTHC(
                    thc_ham, rotation_precision=rotation_prec, select_swap_depth=selswap_depth
                ),
            )
        )
        assert ctrl_select_cost.algo_wires == expected_res["algo_wires"]
        assert (
            ctrl_select_cost.zeroed_wires + ctrl_select_cost.any_state_wires
            == expected_res["auxiliary_wires"]
        )
        assert ctrl_select_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    def test_incompatible_hamiltonian(self):
        """Test that an error is raised for incompatible Hamiltonians."""
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for SelectTHC."
        ):
            qre.SelectTHC(qre.CDFHamiltonian(58, 160))

        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for SelectTHC."
        ):
            qre.SelectTHC.resource_rep(qre.CDFHamiltonian(58, 160))

    def test_type_error_precision(self):
        "Test that an error is raised when wrong type is provided for precision."
        with pytest.raises(
            TypeError,
            match=f"`rotation_precision` must be an integer, but type {type(2.5)} was provided.",
        ):
            qre.SelectTHC(qre.THCHamiltonian(58, 160), rotation_precision=2.5)

        with pytest.raises(
            TypeError,
            match=f"`rotation_precision` must be an integer, but type {type(2.5)} was provided.",
        ):
            qre.SelectTHC.resource_rep(qre.THCHamiltonian(58, 160), rotation_precision=2.5)


class TestGQSP:
    """Test the GQSP class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        thc_ham = qre.THCHamiltonian(4, 20)
        op = qre.SelectTHC(thc_ham)

        with pytest.raises(ValueError, match="Expected"):
            qre.GQSP(op, poly_deg=10, wires=[0])

    @pytest.mark.parametrize(
        "poly_deg, neg_poly_deg, rot_precision",
        [
            (10, 5, 1e-5),
            (2, 0, None),
        ],
    )
    def test_resource_params(self, poly_deg, neg_poly_deg, rot_precision):
        """Test that the resource params for GQSP are correct."""
        thc_ham = qre.THCHamiltonian(4, 20)
        op = qre.SelectTHC(thc_ham)

        gqsp = qre.GQSP(op, poly_deg, neg_poly_deg, rot_precision)

        assert gqsp.resource_params["poly_deg"] == poly_deg
        assert gqsp.resource_params["neg_poly_deg"] == neg_poly_deg
        assert gqsp.resource_params["rot_precision"] == rot_precision
        assert gqsp.resource_params["cmpr_signal_op"] == op.resource_rep_from_op()

    def test_resource_decomp(self):
        """Test the resource decomposition of GQSP."""
        thc_ham = qre.THCHamiltonian(4, 20)
        op = qre.SelectTHC(thc_ham)
        poly_deg = 2
        neg_poly_deg = 1
        rot_precision = 1e-5

        gqsp = qre.GQSP(op, poly_deg, neg_poly_deg, rot_precision)
        decomp = gqsp.resource_decomp(
            gqsp.resource_params["cmpr_signal_op"],
            poly_deg,
            neg_poly_deg,
            rot_precision,
        )

        assert len(decomp) == 3
        # Check types
        assert decomp[0].gate.op_type == qre.Rot
        assert decomp[1].gate.op_type == qre.Controlled
        assert decomp[2].gate.op_type == qre.Controlled

        # Check counts
        assert decomp[0].count == poly_deg + neg_poly_deg + 1
        assert decomp[1].count == poly_deg
        assert decomp[2].count == neg_poly_deg

    def test_resource_decomp_no_neg(self):
        """Test the resource decomposition of GQSP with no negative degree."""
        thc_ham = qre.THCHamiltonian(4, 20)
        op = qre.SelectTHC(thc_ham)
        poly_deg = 2
        neg_poly_deg = 0
        rot_precision = 1e-5

        gqsp = qre.GQSP(op, poly_deg, neg_poly_deg, rot_precision)
        decomp = gqsp.resource_decomp(
            gqsp.resource_params["cmpr_signal_op"],
            poly_deg,
            neg_poly_deg,
            rot_precision,
        )

        assert len(decomp) == 2
        assert decomp[0].gate.op_type == qre.Rot
        assert decomp[1].gate.op_type == qre.Controlled

        assert decomp[0].count == poly_deg + 1
        assert decomp[1].count == poly_deg


class TestHamSimGQSP:
    """Test the HamSimGQSP class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        thc_ham = qre.THCHamiltonian(4, 20)
        op = qre.SelectTHC(thc_ham)

        with pytest.raises(ValueError, match="Expected"):
            qre.HamSimGQSP(op, time=1.0, one_norm=1.0, approximation_error=0.1, wires=[0])

    @pytest.mark.parametrize(
        "time, one_norm, approx_error",
        [
            (1.0, 10.0, 0.01),
            (0.5, 5.0, 1e-5),
        ],
    )
    def test_resource_params(self, time, one_norm, approx_error):
        """Test that the resource params for HamSimGQSP are correct."""
        thc_ham = qre.THCHamiltonian(4, 20)
        op = qre.SelectTHC(thc_ham)

        hamsim = qre.HamSimGQSP(op, time, one_norm, approx_error)

        assert hamsim.resource_params["time"] == time
        assert hamsim.resource_params["one_norm"] == one_norm
        assert hamsim.resource_params["approximation_error"] == approx_error
        assert hamsim.resource_params["walk_op"] == op.resource_rep_from_op()

    def test_degree_of_poly_approx(self):
        """Test the degree calculation."""
        deg = qre.HamSimGQSP.degree_of_poly_approx(1.0, 1.0, 0.1)
        assert isinstance(deg, int)
        assert deg > 0

    def test_resource_decomp(self):
        """Test the resource decomposition of HamSimGQSP."""
        thc_ham = qre.THCHamiltonian(4, 20)
        op = qre.SelectTHC(thc_ham)
        time = 1.0
        one_norm = 10.0
        approx_error = 0.01

        hamsim = qre.HamSimGQSP(op, time, one_norm, approx_error)
        decomp = hamsim.resource_decomp(
            hamsim.resource_params["walk_op"],
            time,
            one_norm,
            approx_error,
        )

        assert len(decomp) == 1
        assert decomp[0].gate.op_type == qre.GQSP


class TestSelectPauli:
    """Test the SelectPauli class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        ph = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        with pytest.raises(ValueError, match=r"Expected 3 wires \(1 control \+ 2 target\), got 2"):
            qre.SelectPauli(ph, wires=[0, 1])

    def test_resource_params(self):
        """Test that the resource params for SelectPauli are correct."""
        ph = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        op = qre.SelectPauli(ph)
        assert op.resource_params == {"pauli_ham": ph}

    def test_resource_rep(self):
        """Test that the compressed representation for SelectPauli is correct."""
        ph = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        expected = qre.CompressedResourceOp(qre.SelectPauli, 3, {"pauli_ham": ph})
        assert qre.SelectPauli.resource_rep(ph) == expected

    @pytest.mark.parametrize(
        "pauli_ham, expected_res",
        (
            (
                qre.PauliHamiltonian(2, {"X": 1, "Z": 1}),
                {
                    qre.CNOT: 2,
                    qre.CZ: 1,
                    qre.X: 2,
                    qre.resource_rep(qre.TemporaryAND): 1,
                    qre.resource_rep(qre.Adjoint, {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)}): 1
                }
            ),
             (
                qre.PauliHamiltonian(3, {"XY": 1, "Z": 2, "Y": 1}),
                {
                    qre.CNOT: 4,
                    qre.CY: 2,
                    qre.CZ: 2,
                    qre.X: 6,
                    qre.resource_rep(qre.TemporaryAND): 3,
                    qre.resource_rep(qre.Adjoint, {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)}): 3,
                    qre.Allocate: 1,
                    qre.Deallocate: 1
                }
            ),
        ),
    )
    def test_resources(self, pauli_ham, expected_res):
        """Test that the resource decompostion for SelectPauli is correct."""
        op = qre.SelectPauli(pauli_ham)
        res = op.resource_decomp(pauli_ham)
        
        res_dict = defaultdict(int)
        for item in res:
            if isinstance(item, qre.GateCount):
                res_dict[item.gate] += item.count
            elif isinstance(item, (qre.Allocate, qre.Deallocate)):
                res_dict[type(item)] += 1
        
        for key, val in expected_res.items():
            if isinstance(key, type) and issubclass(key, (qre.Allocate, qre.Deallocate)):
                 assert res_dict[key] == val
            elif isinstance(key, qre.CompressedResourceOp):
                 assert res_dict[key] == val
            elif isinstance(key, type) and issubclass(key, qre.ResourceOperator):
                 assert res_dict[key.resource_rep()] == val

    def test_adjoint_resources(self):
        """Test that the adjoint resource decomposition is correct."""
        ph = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        op = qre.SelectPauli(ph)
        res = op.adjoint_resource_decomp(op.resource_params)
        
        assert len(res) == 1
        assert res[0].gate == qre.SelectPauli.resource_rep(ph)
        assert res[0].count == 1

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl",
        ((1, 0), (1, 1), (2, 0))
    )
    def test_controlled_resources(self, num_ctrl_wires, num_zero_ctrl):
        """Test that the controlled resource decomposition is correct."""
        ph = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        op = qre.SelectPauli(ph)
        res = op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, op.resource_params)
        
        res_dict = defaultdict(int)
        for item in res:
            if isinstance(item, qre.GateCount):
                res_dict[item.gate] += item.count
            elif isinstance(item, (qre.Allocate, qre.Deallocate)):
                res_dict[type(item)] += 1
        
        # Common counts for ph with work_qubits=1
        assert res_dict[qre.CNOT.resource_rep()] == 2
        assert res_dict[qre.CZ.resource_rep()] == 1
        assert res_dict[qre.resource_rep(qre.TemporaryAND)] == 1
        assert res_dict[qre.resource_rep(qre.Adjoint, {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)})] == 1
        
        # X count
        expected_x = 2
        if num_zero_ctrl == 1 and num_ctrl_wires == 1:
            expected_x += 2
        assert res_dict[qre.X.resource_rep()] == expected_x
        
        # Allocate/Deallocate count
        expected_alloc = 1
        if num_ctrl_wires > 1:
            expected_alloc += 1
        assert res_dict[qre.Allocate] == expected_alloc
        assert res_dict[qre.Deallocate] == expected_alloc
        
        # MCX
        if num_ctrl_wires > 1:
            mcx_rep = qre.MultiControlledX.resource_rep(num_ctrl_wires, num_zero_ctrl)
            assert res_dict[mcx_rep] == 2
