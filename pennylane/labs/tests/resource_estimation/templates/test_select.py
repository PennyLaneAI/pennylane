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
from pennylane.labs.resource_estimation.resource_operator import resource_rep, GateCount

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
        assert op.resource_params == {"compact_ham": compact_ham, "rotation_precision_bits": rotation_prec, "select_swap_depth": selswap_depth}

    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth",
        (
            (plre.CompactHamiltonian.thc(58, 160), 13, 1),
            (plre.CompactHamiltonian.thc(10, 50), None, None),
            (plre.CompactHamiltonian.thc(4, 20), None, 2),
        ),
    )
    def test_resource_rep(self, compact_ham, rotation_prec, selswap_depth):
        """Test that the compressed representation is correct."""
        expected = plre.CompressedResourceOp(
            plre.ResourceSelectTHC, {"compact_ham": compact_ham, "rotation_precision_bits": rotation_prec, "select_swap_depth": selswap_depth}
        )
        assert plre.ResourceSelectTHC.resource_rep(compact_ham, rotation_prec, selswap_depth) == expected


    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and qubits were obtained from https://arxiv.org/abs/2011.03494
    @pytest.mark.parametrize(
        "compact_ham, rotation_prec, selswap_depth, expected_res",
        (
            (plre.CompactHamiltonian.thc(58, 160), 13, 1, {"algo_qubits": 16, "ancilla_qubits":86, "toffoli_gates": 13156}),
            (plre.CompactHamiltonian.thc(10, 50), None, None, {"algo_qubits": 12, "ancilla_qubits": 174, "toffoli_gates": 579}),
            (plre.CompactHamiltonian.thc(4, 20), None, 2, {"algo_qubits": 10, "ancilla_qubits": 109, "toffoli_gates": 279})
        ),
    )
    def test_resources(self, compact_ham, rotation_prec, selswap_depth, expected_res):
        """Test that the resources are correct."""

        select_cost = plre.estimate_resources(plre.ResourceSelectTHC(compact_ham, rotation_precision_bits=rotation_prec, select_swap_depth=selswap_depth))
        assert select_cost.qubit_manager.algo_qubits == expected_res["algo_qubits"]
        assert select_cost.qubit_manager.clean_qubits + prep_cost.qubit_manager.dirty_qubits == expected_res["ancilla_qubits"]
        assert select_cost.clean_gate_counts["Toffoli"] == expected_res["toffoli_gates"]
