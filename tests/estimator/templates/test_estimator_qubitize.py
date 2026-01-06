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

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        ch = qre.THCHamiltonian(2, 4)
        with pytest.raises(ValueError, match="Expected 43 wires, got 3"):
            qre.ControlledSequence(base=qre.QubitizeTHC(ch, wires=[0, 1, 2]))

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
        coeff_precision = prep_op.coeff_precision if prep_op else 15
        rotation_precision = select_op.rotation_precision if select_op else 15

        if prep_op is None:
            prep_op = qre.PrepTHC(thc_ham, coeff_precision=coeff_precision)
        prep_op = prep_op.resource_rep_from_op()

        if select_op is None:
            select_op = qre.SelectTHC(thc_ham, rotation_precision=rotation_precision)
        select_op = select_op.resource_rep_from_op()

        assert op.resource_params == {
            "thc_ham": thc_ham,
            "prep_op": prep_op,
            "select_op": select_op,
            "coeff_precision": coeff_precision,
            "rotation_precision": rotation_precision,
        }

        assert len(qre.QubitizeTHC.resource_keys) == len(op.resource_params)
        assert all(tuple((k in qre.QubitizeTHC.resource_keys) for k in op.resource_params))

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
                183,
            ),
            (qre.THCHamiltonian(10, 50), None, None, 78),
            (
                qre.THCHamiltonian(4, 20),
                resource_rep(
                    qre.PrepTHC,
                    {"thc_ham": qre.THCHamiltonian(4, 20), "select_swap_depth": 2},
                ),
                None,
                59,
            ),
        ),
    )
    def test_resource_rep(self, thc_ham, prep_op, select_op, num_wires):
        """Test that the compressed representation  for QubitizeTHC is correct."""

        if prep_op:
            coeff_precision = prep_op.params["coeff_precision"]
        else:
            coeff_precision = 15

        if select_op:
            rotation_precision = select_op.params["rotation_precision"]
        else:
            rotation_precision = 15

        expected = qre.CompressedResourceOp(
            qre.QubitizeTHC,
            num_wires,
            {
                "thc_ham": thc_ham,
                "prep_op": prep_op,
                "select_op": select_op,
                "coeff_precision": coeff_precision,
                "rotation_precision": rotation_precision,
            },
        )
        assert (
            qre.QubitizeTHC.resource_rep(thc_ham, prep_op=prep_op, select_op=select_op) == expected
        )

    # The Toffoli and qubit costs are compared here
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
                {"algo_wires": 183, "auxiliary_wires": 752, "toffoli_gates": 8539},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                None,
                {"algo_wires": 78, "auxiliary_wires": 148, "toffoli_gates": 2265},
            ),
            (
                qre.THCHamiltonian(4, 20),
                qre.PrepTHC(qre.THCHamiltonian(4, 20), select_swap_depth=2),
                None,
                {"algo_wires": 59, "auxiliary_wires": 58, "toffoli_gates": 941},
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
        assert wo_cost.zeroed_wires + wo_cost.any_state_wires == expected_res["auxiliary_wires"]
        assert wo_cost.gate_counts["Toffoli"] == expected_res["toffoli_gates"]

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from equations 44 and 46  in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, prep_op, select_op, num_ctrl_wires, num_zero_ctrl, expected_res",
        (
            (
                # This test was taken from arXiv:2501.06165, numbers are adjusted to only the walk operator cost without unary iteration
                qre.THCHamiltonian(58, 160),
                qre.PrepTHC(qre.THCHamiltonian(58, 160), coeff_precision=13),
                qre.SelectTHC(qre.THCHamiltonian(58, 160), rotation_precision=13),
                1,
                1,
                {"algo_wires": 184, "auxiliary_wires": 752, "toffoli_gates": 8540},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                None,
                2,
                0,
                {"algo_wires": 80, "auxiliary_wires": 149, "toffoli_gates": 2268},
            ),
            (
                qre.THCHamiltonian(4, 20),
                qre.PrepTHC(qre.THCHamiltonian(4, 20), select_swap_depth=2),
                None,
                3,
                2,
                {"algo_wires": 62, "auxiliary_wires": 59, "toffoli_gates": 946},
            ),
        ),
    )
    def test_control_resources(
        self, thc_ham, prep_op, select_op, num_ctrl_wires, num_zero_ctrl, expected_res
    ):
        """Test that the controlled resource decomposition for QubitizeTHC is correct."""

        wo_cost = qre.estimate(
            qre.Controlled(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                base_op=qre.QubitizeTHC(
                    thc_ham,
                    prep_op=prep_op,
                    select_op=select_op,
                ),
            )
        )

        assert wo_cost.algo_wires == expected_res["algo_wires"]
        assert wo_cost.zeroed_wires + wo_cost.any_state_wires == expected_res["auxiliary_wires"]
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
