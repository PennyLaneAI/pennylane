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
from pennylane.labs.resource_estimation import resource_rep

# pylint: disable=too-many-arguments, no-self-use


class TestQubitizeTHC:
    """Test the ResourceQubitizeTHC class."""

    @pytest.mark.parametrize(
        "compact_ham, prep_op, select_op",
        (
            (
                plre.CompactHamiltonian.thc(58, 160),
                plre.ResourcePrepTHC(plre.CompactHamiltonian.thc(58, 160), coeff_precision=13),
                plre.ResourceSelectTHC(plre.CompactHamiltonian.thc(58, 160), rotation_precision=13),
            ),
            (plre.CompactHamiltonian.thc(10, 50), None, None),
            (
                plre.CompactHamiltonian.thc(4, 20),
                plre.ResourcePrepTHC(plre.CompactHamiltonian.thc(4, 20), select_swap_depth=2),
                None,
            ),
        ),
    )
    def test_resource_params(self, compact_ham, prep_op, select_op):
        """Test that the resource params are correct."""
        op = plre.ResourceQubitizeTHC(compact_ham, prep_op=prep_op, select_op=select_op)

        if prep_op is not None:
            prep_op = prep_op.resource_rep_from_op()
        if select_op is not None:
            select_op = select_op.resource_rep_from_op()

        assert op.resource_params == {
            "compact_ham": compact_ham,
            "prep_op": prep_op,
            "select_op": select_op,
        }

    @pytest.mark.parametrize(
        "compact_ham, prep_op, select_op, num_wires",
        (
            (
                plre.CompactHamiltonian.thc(58, 160),
                resource_rep(
                    plre.ResourcePrepTHC,
                    {"compact_ham": plre.CompactHamiltonian.thc(58, 160), "coeff_precision": 13},
                ),
                resource_rep(
                    plre.ResourceSelectTHC,
                    {"compact_ham": plre.CompactHamiltonian.thc(58, 160), "rotation_precision": 13},
                ),
                152,
            ),
            (plre.CompactHamiltonian.thc(10, 50), None, None, 49),
            (
                plre.CompactHamiltonian.thc(4, 20),
                resource_rep(
                    plre.ResourcePrepTHC,
                    {"compact_ham": plre.CompactHamiltonian.thc(4, 20), "select_swap_depth": 2},
                ),
                None,
                32,
            ),
        ),
    )
    def test_resource_rep(self, compact_ham, prep_op, select_op, num_wires):
        """Test that the compressed representation is correct."""

        expected = plre.CompressedResourceOp(
            plre.ResourceQubitizeTHC,
            num_wires,
            {
                "compact_ham": compact_ham,
                "prep_op": prep_op,
                "select_op": select_op,
            },
        )
        assert (
            plre.ResourceQubitizeTHC.resource_rep(compact_ham, prep_op=prep_op, select_op=select_op)
            == expected
        )

    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and qubits were obtained from equations 44 and 46  in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "compact_ham, prep_op, select_op, expected_res",
        (
            (
                # This test was taken from arXiv:2501.06165, numbers are adjusted to only the walk operator cost without unary iteration
                plre.CompactHamiltonian.thc(58, 160),
                plre.ResourcePrepTHC(plre.CompactHamiltonian.thc(58, 160), coeff_precision=13),
                plre.ResourceSelectTHC(plre.CompactHamiltonian.thc(58, 160), rotation_precision=13),
                {"algo_qubits": 152, "ancilla_qubits": 791, "toffoli_gates": 8579},
            ),
            (
                plre.CompactHamiltonian.thc(10, 50),
                None,
                None,
                {"algo_qubits": 49, "ancilla_qubits": 174, "toffoli_gates": 2299},
            ),
            # (
            #     plre.CompactHamiltonian.thc(4, 20),
            #     plre.ResourcePrepTHC(plre.CompactHamiltonian.thc(4,20), select_swap_depth=2),
            #     None,
            #     {"algo_qubits": 32, "ancilla_qubits": 109, "toffoli_gates": 983},
            # ),
        ),
    )
    def test_resources(self, compact_ham, prep_op, select_op, expected_res):
        """Test that the resources are correct."""

        wo_cost = plre.estimate(
            plre.ResourceQubitizeTHC(
                compact_ham,
                prep_op=prep_op,
                select_op=select_op,
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

        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for ResourceQubitizeTHC."
        ):
            plre.ResourceQubitizeTHC.resource_rep(plre.CompactHamiltonian.cdf(58, 160))
