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

# pylint: disable=no-self-use


class TestSelectTHC:
    """Test the SelectTHC class."""

    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth",
        (
            (qre.CompactHamiltonian.thc(58, 160), 13, 1),
            (qre.CompactHamiltonian.thc(10, 50), None, None),
            (qre.CompactHamiltonian.thc(4, 20), None, 2),
        ),
    )
    def test_resource_params(self, compact_ham, rotation_prec, selswap_depth):
        """Test that the resource params for SelectTHC are correct."""
        op = qre.SelectTHC(compact_ham, rotation_prec, selswap_depth)
        assert op.resource_params == {
            "compact_ham": compact_ham,
            "rotation_precision": rotation_prec,
            "select_swap_depth": selswap_depth,
        }

    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth, num_wires",
        (
            (qre.CompactHamiltonian.thc(58, 160), 13, 1, 138),
            (qre.CompactHamiltonian.thc(10, 50), None, None, 38),
            (qre.CompactHamiltonian.thc(4, 20), None, 2, 24),
        ),
    )
    def test_resource_rep(self, compact_ham, rotation_prec, selswap_depth, num_wires):
        """Test that the compressed representation for SelectTHC is correct."""
        expected = qre.CompressedResourceOp(
            qre.SelectTHC,
            num_wires,
            {
                "compact_ham": compact_ham,
                "rotation_precision": rotation_prec,
                "select_swap_depth": selswap_depth,
            },
        )
        assert qre.SelectTHC.resource_rep(compact_ham, rotation_prec, selswap_depth) == expected

    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth, expected_res",
        (
            (
                qre.CompactHamiltonian.thc(58, 160),
                13,
                1,
                {"algo_wires": 138, "auxiliary_wires": 752, "toffoli_gates": 5997},
            ),
            (
                qre.CompactHamiltonian.thc(10, 50),
                None,
                None,
                {"algo_wires": 38, "auxiliary_wires": 163, "toffoli_gates": 1139},
            ),
            (
                qre.CompactHamiltonian.thc(4, 20),
                None,
                2,
                {"algo_wires": 24, "auxiliary_wires": 73, "toffoli_gates": 425},
            ),
        ),
    )
    def test_resources(self, compact_ham, rotation_prec, selswap_depth, expected_res):
        """Test that the resource decompostion for SelectTHC is correct."""

        select_cost = qre.estimate(
            qre.SelectTHC(
                compact_ham, rotation_precision=rotation_prec, select_swap_depth=selswap_depth
            )
        )
        assert select_cost.algo_wires == expected_res["algo_wires"]
        assert select_cost.zeroed + select_cost.any_state == expected_res["auxiliary_wires"]
        assert select_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth, expected_res",
        (
            (
                qre.CompactHamiltonian.thc(58, 160),
                13,
                1,
                {"algo_wires": 141, "auxiliary_wires": 753, "toffoli_gates": 6002},
            ),
            (
                qre.CompactHamiltonian.thc(10, 50),
                None,
                None,
                {"algo_wires": 41, "auxiliary_wires": 164, "toffoli_gates": 1144},
            ),
            (
                qre.CompactHamiltonian.thc(4, 20),
                None,
                2,
                {"algo_wires": 27, "auxiliary_wires": 74, "toffoli_gates": 430},
            ),
        ),
    )
    def test_controlled_resources(self, compact_ham, rotation_prec, selswap_depth, expected_res):
        """Test that the controlled resource decompostion for SelectTHC is correct."""

        ctrl_select_cost = qre.estimate(
            qre.Controlled(
                num_ctrl_wires=3,
                num_zero_ctrl=2,
                base_op=qre.SelectTHC(
                    compact_ham, rotation_precision=rotation_prec, select_swap_depth=selswap_depth
                ),
            )
        )
        assert ctrl_select_cost.algo_wires == expected_res["algo_wires"]
        assert (
            ctrl_select_cost.zeroed + ctrl_select_cost.any_state == expected_res["auxiliary_wires"]
        )
        assert ctrl_select_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    def test_incompatible_hamiltonian(self):
        """Test that an error is raised for incompatible Hamiltonians."""
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for SelectTHC."
        ):
            qre.SelectTHC(qre.CompactHamiltonian.cdf(58, 160))

        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for SelectTHC."
        ):
            qre.SelectTHC.resource_rep(qre.CompactHamiltonian.cdf(58, 160))

    def test_type_error_precision(self):
        "Test that an error is raised when wrong type is provided for precision."
        with pytest.raises(
            TypeError,
            match=f"`rotation_precision` must be an integer, but type {type(2.5)} was provided.",
        ):
            qre.SelectTHC(qre.CompactHamiltonian.thc(58, 160), rotation_precision=2.5)

        with pytest.raises(
            TypeError,
            match=f"`rotation_precision` must be an integer, but type {type(2.5)} was provided.",
        ):
            qre.SelectTHC.resource_rep(qre.CompactHamiltonian.thc(58, 160), rotation_precision=2.5)
