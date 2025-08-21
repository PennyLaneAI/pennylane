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
Test the Resource classes for Qubitization
"""
import pytest

import pennylane.labs.resource_estimation as plre

# pylint: disable=too-many-arguments, no-self-use


class TestQubitizeTHC:
    """Test the ResourceQubitizeTHC class."""

    @pytest.mark.parametrize(
        "compact_ham, coeff_prec, rotation_prec, selswap_depth",
        (
            (plre.CompactHamiltonian.thc(58, 160), 13, 13, 1),
            (plre.CompactHamiltonian.thc(10, 50), None, None, None),
            (plre.CompactHamiltonian.thc(4, 20), None, None, [2, 2]),
        ),
    )
    def test_resource_params(self, compact_ham, coeff_prec, rotation_prec, selswap_depth):
        """Test that the resource params are correct."""
        op = plre.ResourceQubitizeTHC(compact_ham, coeff_prec, rotation_prec, selswap_depth)
        assert op.resource_params == {
            "compact_ham": compact_ham,
            "coeff_precision_bits": coeff_prec,
            "rotation_precision_bits": rotation_prec,
            "select_swap_depths": selswap_depth,
        }

    @pytest.mark.parametrize(
        "compact_ham, coeff_prec, rotation_prec, selswap_depth",
        (
            (plre.CompactHamiltonian.thc(58, 160), 13, 13, 1),
            (plre.CompactHamiltonian.thc(10, 50), None, None, None),
            (plre.CompactHamiltonian.thc(4, 20), None, None, [2, 2]),
        ),
    )
    def test_resource_rep(self, compact_ham, coeff_prec, rotation_prec, selswap_depth):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceQubitizeTHC,
            {
                "compact_ham": compact_ham,
                "coeff_precision_bits": coeff_prec,
                "rotation_precision_bits": rotation_prec,
                "select_swap_depths": selswap_depth,
            },
        )
        assert (
            plre.ResourceQubitizeTHC.resource_rep(
                compact_ham, coeff_prec, rotation_prec, selswap_depth
            )
            == expected
        )

    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and qubits were obtained from https://arxiv.org/abs/2011.03494
    @pytest.mark.parametrize(
        "compact_ham, coeff_prec, rotation_prec, selswap_depth, expected_res",
        (
            (
                plre.CompactHamiltonian.thc(58, 160),
                13,
                13,
                1,
                {"algo_qubits": 138, "ancilla_qubits": 752, "toffoli_gates": 32317},
            ),
            (
                plre.CompactHamiltonian.thc(10, 50),
                None,
                None,
                None,
                {"algo_qubits": 38, "ancilla_qubits": 174, "toffoli_gates": 2299},
            ),
            (
                plre.CompactHamiltonian.thc(4, 20),
                None,
                None,
                [2, 2],
                {"algo_qubits": 24, "ancilla_qubits": 109, "toffoli_gates": 983},
            ),
        ),
    )
    def test_resources(self, compact_ham, coeff_prec, rotation_prec, selswap_depth, expected_res):
        """Test that the resources are correct."""

        wo_cost = plre.estimate_resources(
            plre.ResourceQubitizeTHC(
                compact_ham,
                coeff_precision_bits=coeff_prec,
                rotation_precision_bits=rotation_prec,
                select_swap_depths=selswap_depth,
            )
        )
        assert wo_cost.qubit_manager.algo_qubits == expected_res["algo_qubits"]
        assert (
            wo_cost.qubit_manager.clean_qubits + wo_cost.qubit_manager.dirty_qubits
            == expected_res["ancilla_qubits"]
        )
        assert wo_cost.clean_gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    def test_incompatible_hamiltonian(self):
        """Test that an error is raised for incompatible Hamiltonians."""
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceQubitizeTHC."
        ):
            plre.ResourceQubitizeTHC(plre.CompactHamiltonian.cdf(58, 160))

    def test_incompatible_select_swap_depths(self):
        """Test that an error is raised for incompatible select swap depths."""
        with pytest.raises(
            TypeError, match="`select_swap_depths` must be an integer, None or iterable"
        ):
            plre.ResourceQubitizeTHC(plre.CompactHamiltonian.thc(58, 160), select_swap_depths="a")
