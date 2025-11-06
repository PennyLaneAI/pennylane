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

import pennylane.labs.resource_estimation as plre

# pylint: disable=no-self-use


class TestSelectTHC:
    """Test the ResourceSelectTHC class."""

    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth",
        (
            (plre.CompactHamiltonian.thc(58, 160), 13, 1),
            (plre.CompactHamiltonian.thc(10, 50), None, None),
            (plre.CompactHamiltonian.thc(4, 20), None, 2),
        ),
    )
    def test_resource_params(self, compact_ham, rotation_prec, selswap_depth):
        """Test that the resource params are correct."""
        op = plre.ResourceSelectTHC(compact_ham, rotation_prec, selswap_depth)
        assert op.resource_params == {
            "compact_ham": compact_ham,
            "rotation_precision": rotation_prec,
            "select_swap_depth": selswap_depth,
        }

    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth, num_wires",
        (
            (plre.CompactHamiltonian.thc(58, 160), 13, 1, 138),
            (plre.CompactHamiltonian.thc(10, 50), None, None, 38),
            (plre.CompactHamiltonian.thc(4, 20), None, 2, 24),
        ),
    )
    def test_resource_rep(self, compact_ham, rotation_prec, selswap_depth, num_wires):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceSelectTHC,
            num_wires,
            {
                "compact_ham": compact_ham,
                "rotation_precision": rotation_prec,
                "select_swap_depth": selswap_depth,
            },
        )
        assert (
            plre.ResourceSelectTHC.resource_rep(compact_ham, rotation_prec, selswap_depth)
            == expected
        )

    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and qubits were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth, expected_res",
        (
            (
                plre.CompactHamiltonian.thc(58, 160),
                13,
                1,
                {"algo_qubits": 138, "ancilla_qubits": 752, "toffoli_gates": 5997},
            ),
            (
                plre.CompactHamiltonian.thc(10, 50),
                None,
                None,
                {"algo_qubits": 38, "ancilla_qubits": 163, "toffoli_gates": 1139},
            ),
            (
                plre.CompactHamiltonian.thc(4, 20),
                None,
                2,
                {"algo_qubits": 24, "ancilla_qubits": 73, "toffoli_gates": 425},
            ),
        ),
    )
    def test_resources(self, compact_ham, rotation_prec, selswap_depth, expected_res):
        """Test that the resources are correct."""

        select_cost = plre.estimate(
            plre.ResourceSelectTHC(
                compact_ham, rotation_precision=rotation_prec, select_swap_depth=selswap_depth
            )
        )
        assert select_cost.qubit_manager.algo_qubits == expected_res["algo_qubits"]
        assert (
            select_cost.qubit_manager.clean_qubits + select_cost.qubit_manager.dirty_qubits
            == expected_res["ancilla_qubits"]
        )
        assert select_cost.clean_gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    def test_incompatible_hamiltonian(self):
        """Test that an error is raised for incompatible Hamiltonians."""
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceSelectTHC."
        ):
            plre.ResourceSelectTHC(plre.CompactHamiltonian.cdf(58, 160))

        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceSelectTHC."
        ):
            plre.ResourceSelectTHC.resource_rep(plre.CompactHamiltonian.cdf(58, 160))

    def test_typeerror_precision(self):
        "Test that an error is raised when wrong type is provided for precision."
        with pytest.raises(
            TypeError, match=f"`rotation_precision` must be an integer, provided {type(2.5)}."
        ):
            plre.ResourceSelectTHC(plre.CompactHamiltonian.thc(58, 160), rotation_precision=2.5)

        with pytest.raises(
            TypeError, match=f"`rotation_precision` must be an integer, provided {type(2.5)}."
        ):
            plre.ResourceSelectTHC.resource_rep(
                plre.CompactHamiltonian.thc(58, 160), rotation_precision=2.5
            )
