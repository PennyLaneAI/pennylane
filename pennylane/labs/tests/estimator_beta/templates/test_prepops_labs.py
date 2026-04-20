# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for the state preparation subroutines resource operators.
"""
import pytest

import pennylane.labs.estimator_beta as qre

class TestPrepFirstQuantization:
    """Test the PrepFirstQuantization class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        fq_ham = qre.FirstQuantizedHamiltonian(4, 10, 1.0, 2)
        with pytest.raises(ValueError, match="Expected 51 wires, got 3"):
            qre.PrepFirstQuantization(fq_ham, wires=[0, 1, 2])

    @pytest.mark.parametrize(
        "fq_ham, selection_precision, coordinates_precision, select_swap_depth",
        (
            (qre.FirstQuantizedHamiltonian(200, 10, 1.0, 2), 13, 15, 1),
            (qre.FirstQuantizedHamiltonian(1000, 50, 3.5, 2), None, None, None),
            (qre.FirstQuantizedHamiltonian(1000, 50, 3.5, 2), None, 12, 2),
        ),
    )
    def test_resource_params(self, fq_ham, selection_precision, coordinates_precision, select_swap_depth):
        """Test that the resource params for PrepFirstQuantization are correct."""

        if selection_precision and coordinates_precision:
            op = qre.PrepFirstQuantization(fq_ham, selection_precision, coordinates_precision, select_swap_depth)
        elif selection_precision:
            op = qre.PrepFirstQuantization(fq_ham, selection_precision, select_swap_depth=select_swap_depth)
        elif coordinates_precision:
            op = qre.PrepFirstQuantization(fq_ham, coordinates_precision=coordinates_precision, select_swap_depth=select_swap_depth)
        else:
            op = qre.PrepFirstQuantization(fq_ham, select_swap_depth=select_swap_depth)

        assert op.resource_params == {
            "fq_ham": fq_ham,
            "selection_precision": selection_precision if selection_precision else 15,
            "coordinates_precision": coordinates_precision if coordinates_precision else 15,
            "select_swap_depth": select_swap_depth,
        }

    @pytest.mark.parametrize(
        "fq_ham, selection_precision, coordinates_precision, select_swap_depth, num_wires",
        (
            (qre.FirstQuantizedHamiltonian(1000, 50, 3.5, 2), 13, 15, 1, 73),
            (qre.FirstQuantizedHamiltonian(1000, 50, 3.5, 2), 15, 15, None, 73),
            (qre.FirstQuantizedHamiltonian(1000, 50, 3.5, 2), 15, 15, 2, 73),
        ),
    )
    def test_resource_rep(self, fq_ham, selection_precision, coordinates_precision, select_swap_depth, num_wires):
        """Test that the compressed representation of PrepFirstQuantization is correct."""
        expected = qre.CompressedResourceOp(
            qre.PrepFirstQuantization,
            num_wires,
            {
                "fq_ham": fq_ham,
                "selection_precision": selection_precision,
                "coordinates_precision": coordinates_precision,
                "select_swap_depth": select_swap_depth,
            },
        )
        print(qre.PrepFirstQuantization.resource_rep(fq_ham, selection_precision, coordinates_precision, select_swap_depth))
        assert qre.PrepFirstQuantization.resource_rep(fq_ham, selection_precision, coordinates_precision, select_swap_depth) == expected

    # # The Toffoli and qubit costs are compared here
    # # Expected number of Toffolis and wires were obtained from Eq. 33 in https://arxiv.org/abs/2011.03494.
    # # The numbers were adjusted slightly to account for a different QROM decomposition
    # @pytest.mark.parametrize(
    #     "thc_ham, coeff_prec, selswap_depth, expected_res",
    #     (
    #         (
    #             qre.THCHamiltonian(58, 160),
    #             13,
    #             1,
    #             {"algo_wires": 80, "auxiliary_wires": 24, "toffoli_gates": 13156},
    #         ),
    #         (
    #             qre.THCHamiltonian(10, 50),
    #             None,
    #             None,
    #             {"algo_wires": 73, "auxiliary_wires": 95, "toffoli_gates": 579},
    #         ),
    #         (
    #             qre.THCHamiltonian(4, 20),
    #             None,
    #             2,
    #             {"algo_wires": 66, "auxiliary_wires": 33, "toffoli_gates": 279},
    #         ),
    #     ),
    # )
    # def test_resources(self, thc_ham, coeff_prec, selswap_depth, expected_res):
    #     """Test that the resources for PrepTHC are correct."""

    #     if coeff_prec:
    #         prep_cost = qre.estimate(
    #             qre.PrepTHC(thc_ham, coeff_precision=coeff_prec, select_swap_depth=selswap_depth)
    #         )
    #     else:
    #         prep_cost = qre.estimate(qre.PrepTHC(thc_ham, select_swap_depth=selswap_depth))
    #     assert prep_cost.algo_wires == expected_res["algo_wires"]
    #     assert prep_cost.zeroed_wires + prep_cost.any_state_wires == expected_res["auxiliary_wires"]
    #     assert prep_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    # def test_incompatible_hamiltonian(self):
    #     """Test that an error is raised for incompatible Hamiltonians."""
    #     with pytest.raises(TypeError, match="Unsupported Hamiltonian representation for PrepTHC."):
    #         qre.PrepTHC(qre.CDFHamiltonian(58, 160))

    #     with pytest.raises(TypeError, match="Unsupported Hamiltonian representation for PrepTHC."):
    #         qre.PrepTHC.resource_rep(qre.CDFHamiltonian(58, 160))

    # def test_type_error_precision(self):
    #     "Test that an error is raised when wrong type is provided for precision."
    #     with pytest.raises(
    #         TypeError,
    #         match=f"`coeff_precision` must be an integer, but type {type(2.5)} was provided.",
    #     ):
    #         qre.PrepTHC(qre.THCHamiltonian(58, 160), coeff_precision=2.5)

    #     with pytest.raises(
    #         TypeError,
    #         match=f"`coeff_precision` must be an integer, but type {type(2.5)} was provided.",
    #     ):
    #         qre.PrepTHC.resource_rep(qre.THCHamiltonian(58, 160), coeff_precision=2.5)
