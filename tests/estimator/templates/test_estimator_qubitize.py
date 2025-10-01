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

import pennylane.estimator as qre
from pennylane.estimator import resource_rep

# pylint: disable=too-many-arguments, no-self-use


class TestQubitizeTHC:
    """Test the QubitizeTHC class."""

    @pytest.mark.parametrize(
        "thc_ham, prep_op, select_op",
        (
            (
                qre.THCHamiltonian(58, 160),
                qre.PrepTHC(qre.THCHamiltonian(58, 160), coeff_precision=13),
                qre.SelectTHC(qre.THCHamiltonian(58, 160), rotation_precision=13),
            ),
            (qre.THCHamiltonian(10, 50), None, None),
            (
                qre.THCHamiltonian(4, 20),
                qre.PrepTHC(qre.THCHamiltonian(4, 20), select_swap_depth=2),
                None,
            ),
        ),
    )
    def test_resource_params(self, thc_ham, prep_op, select_op):
        """Test that the resource params for QubitizeTHC are correct."""
        op = qre.QubitizeTHC(thc_ham, prep_op=prep_op, select_op=select_op)

        if prep_op is not None:
            prep_op = prep_op.resource_rep_from_op()
        if select_op is not None:
            select_op = select_op.resource_rep_from_op()

        assert op.resource_params == {
            "thc_ham": thc_ham,
            "prep_op": prep_op,
            "select_op": select_op,
            "coeff_precision": None,
            "rotation_precision": None,
        }

    @pytest.mark.parametrize(
        "thc_ham, prep_op, select_op, num_wires",
        (
            (
                qre.THCHamiltonian(58, 160),
                resource_rep(
                    qre.PrepTHC,
                    {"thc_ham": qre.THCHamiltonian(58, 160), "coeff_precision": 13},
                ),
                resource_rep(
                    qre.SelectTHC,
                    {"thc_ham": qre.THCHamiltonian(58, 160), "rotation_precision": 13},
                ),
                152,
            ),
            (qre.THCHamiltonian(10, 50), None, None, 49),
            (
                qre.THCHamiltonian(4, 20),
                resource_rep(
                    qre.PrepTHC,
                    {"thc_ham": qre.THCHamiltonian(4, 20), "select_swap_depth": 2},
                ),
                None,
                32,
            ),
        ),
    )
    def test_resource_rep(self, thc_ham, prep_op, select_op, num_wires):
        """Test that the compressed representation  for QubitizeTHC is correct."""

        expected = qre.CompressedResourceOp(
            qre.QubitizeTHC,
            num_wires,
            {
                "thc_ham": thc_ham,
                "prep_op": prep_op,
                "select_op": select_op,
                "coeff_precision": None,
                "rotation_precision": None,
            },
        )
        assert (
            qre.QubitizeTHC.resource_rep(thc_ham, prep_op=prep_op, select_op=select_op) == expected
        )

    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and wires were obtained from equations 44 and 46  in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, prep_op, select_op, expected_res",
        (
            (
                # This test was taken from arXiv:2501.06165, numbers are adjusted to only the walk operator cost without unary iteration
                qre.THCHamiltonian(58, 160),
                qre.PrepTHC(qre.THCHamiltonian(58, 160), coeff_precision=13),
                qre.SelectTHC(qre.THCHamiltonian(58, 160), rotation_precision=13),
                {"algo_wires": 152, "auxiliary_wires": 791, "toffoli_gates": 8579},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                None,
                {"algo_wires": 49, "auxiliary_wires": 174, "toffoli_gates": 2299},
            ),
            (
                qre.THCHamiltonian(4, 20),
                qre.PrepTHC(qre.THCHamiltonian(4, 20), select_swap_depth=2),
                None,
                {"algo_wires": 32, "auxiliary_wires": 109, "toffoli_gates": 967},
            ),
        ),
    )
    def test_resources(self, thc_ham, prep_op, select_op, expected_res):
        """Test that the resource decomposition for QubitizeTHC is correct."""

        wo_cost = qre.estimate(
            qre.QubitizeTHC(
                thc_ham,
                prep_op=prep_op,
                select_op=select_op,
            )
        )

        assert wo_cost.algo_wires == expected_res["algo_wires"]
        assert wo_cost.zeroed + wo_cost.any_state == expected_res["auxiliary_wires"]
        assert wo_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    # We are comparing the Toffoli and qubit cost here
    # Expected number of Toffolis and wires were obtained from equations 44 and 46  in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, prep_op, select_op, expected_res",
        (
            (
                # This test was taken from arXiv:2501.06165, numbers are adjusted to only the walk operator cost without unary iteration
                qre.THCHamiltonian(58, 160),
                qre.PrepTHC(qre.THCHamiltonian(58, 160), coeff_precision=13),
                qre.SelectTHC(qre.THCHamiltonian(58, 160), rotation_precision=13),
                {"algo_wires": 155, "auxiliary_wires": 792, "toffoli_gates": 8584},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                None,
                {"algo_wires": 52, "auxiliary_wires": 175, "toffoli_gates": 2304},
            ),
            (
                qre.THCHamiltonian(4, 20),
                qre.PrepTHC(qre.THCHamiltonian(4, 20), select_swap_depth=2),
                None,
                {"algo_wires": 35, "auxiliary_wires": 110, "toffoli_gates": 972},
            ),
        ),
    )
    def test_control_resources(self, thc_ham, prep_op, select_op, expected_res):
        """Test that the controlled resource decomposition for QubitizeTHC is correct."""

        wo_cost = qre.estimate(
            qre.Controlled(
                num_ctrl_wires=3,
                num_zero_ctrl=2,
                base_op=qre.QubitizeTHC(
                    thc_ham,
                    prep_op=prep_op,
                    select_op=select_op,
                ),
            )
        )

        assert wo_cost.algo_wires == expected_res["algo_wires"]
        assert wo_cost.zeroed + wo_cost.any_state == expected_res["auxiliary_wires"]
        assert wo_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    def test_incompatible_hamiltonian(self):
        """Test that an error is raised for incompatible Hamiltonians."""
        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for QubitizeTHC."
        ):
            qre.QubitizeTHC(qre.CDFHamiltonian(58, 160))

        with pytest.raises(
            TypeError, match="Unsupported Hamiltonian representation for QubitizeTHC."
        ):
            qre.QubitizeTHC.resource_rep(qre.CDFHamiltonian(58, 160))
