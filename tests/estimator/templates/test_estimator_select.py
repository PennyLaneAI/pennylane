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


class TestSelectTHC:
    """Test the SelectTHC class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        ch = qre.THCHamiltonian(2, 4)
        with pytest.raises(ValueError, match="Expected 16 wires, got 3"):
            qre.ControlledSequence(base=qre.SelectTHC(ch, wires=[0, 1, 2]))

    @pytest.mark.parametrize(
        "thc_ham, batched_rotations, rotation_prec, selswap_depth",
        (
            (qre.THCHamiltonian(58, 160), None, 13, 1),
            (qre.THCHamiltonian(10, 50), None, None, None),
            (qre.THCHamiltonian(4, 20), 2, None, 2),
        ),
    )
    def test_resource_params(self, thc_ham, batched_rotations, rotation_prec, selswap_depth):
        """Test that the resource params for SelectTHC are correct."""
        if rotation_prec:
            op = qre.SelectTHC(thc_ham, batched_rotations, rotation_prec, selswap_depth)
        else:
            op = qre.SelectTHC(
                thc_ham, batched_rotations=batched_rotations, select_swap_depth=selswap_depth
            )
            rotation_prec = 15

        assert op.resource_params == {
            "thc_ham": thc_ham,
            "batched_rotations": batched_rotations,
            "rotation_precision": rotation_prec,
            "select_swap_depth": selswap_depth,
        }

    @pytest.mark.parametrize(
        "thc_ham, batched_rotations, rotation_prec, selswap_depth, num_wires",
        (
            (qre.THCHamiltonian(58, 160), None, 13, 1, 138),
            (qre.THCHamiltonian(10, 50), None, None, None, 38),
            (qre.THCHamiltonian(4, 20), 2, None, 2, 24),
        ),
    )
    def test_resource_rep(
        self, thc_ham, batched_rotations, rotation_prec, selswap_depth, num_wires
    ):
        """Test that the compressed representation for SelectTHC is correct."""
        if rotation_prec:
            expected = qre.CompressedResourceOp(
                qre.SelectTHC,
                num_wires,
                {
                    "thc_ham": thc_ham,
                    "batched_rotations": batched_rotations,
                    "rotation_precision": rotation_prec,
                    "select_swap_depth": selswap_depth,
                },
            )
            assert (
                qre.SelectTHC.resource_rep(thc_ham, batched_rotations, rotation_prec, selswap_depth)
                == expected
            )
        else:
            expected = qre.CompressedResourceOp(
                qre.SelectTHC,
                num_wires,
                {
                    "thc_ham": thc_ham,
                    "batched_rotations": batched_rotations,
                    "rotation_precision": 15,
                    "select_swap_depth": selswap_depth,
                },
            )
            assert (
                qre.SelectTHC.resource_rep(
                    thc_ham, batched_rotations=batched_rotations, select_swap_depth=selswap_depth
                )
                == expected
            )

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, batched_rotations, rotation_prec, selswap_depth, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                None,
                13,
                1,
                {"algo_wires": 138, "auxiliary_wires": 752, "toffoli_gates": 5997},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                15,
                None,
                {"algo_wires": 38, "auxiliary_wires": 148, "toffoli_gates": 1189},
            ),
            (
                qre.THCHamiltonian(4, 20),
                None,
                15,
                2,
                {"algo_wires": 24, "auxiliary_wires": 103, "toffoli_gates": 545},
            ),
            # These numbers were obtained manually for batched rotations based on the technique described in arXiv:2501.06165
            (
                qre.THCHamiltonian(58, 160),
                29,
                13,
                None,
                {"algo_wires": 138, "auxiliary_wires": 388, "toffoli_gates": 7493},
            ),
        ),
    )
    def test_resources(
        self, thc_ham, batched_rotations, rotation_prec, selswap_depth, expected_res
    ):
        """Test that the resource decompostion for SelectTHC is correct."""

        select_cost = qre.estimate(
            qre.SelectTHC(
                thc_ham,
                batched_rotations=batched_rotations,
                rotation_precision=rotation_prec,
                select_swap_depth=selswap_depth,
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
        "thc_ham, batched_rotations, rotation_prec, selswap_depth, num_ctrl_wires, num_zero_ctrl, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                None,
                13,
                1,
                1,
                1,
                {"algo_wires": 139, "auxiliary_wires": 752, "toffoli_gates": 5998},
            ),
            (
                qre.THCHamiltonian(10, 50),
                None,
                15,
                None,
                2,
                0,
                {"algo_wires": 40, "auxiliary_wires": 149, "toffoli_gates": 1192},
            ),
            (
                qre.THCHamiltonian(4, 20),
                None,
                15,
                2,
                3,
                2,
                {"algo_wires": 27, "auxiliary_wires": 104, "toffoli_gates": 550},
            ),
            # These numbers were obtained manually for batched rotations based on the technique described in arXiv:2501.06165
            (
                qre.THCHamiltonian(58, 160),
                29,
                13,
                None,
                1,
                1,
                {"algo_wires": 139, "auxiliary_wires": 388, "toffoli_gates": 7494},
            ),
        ),
    )
    def test_controlled_resources(
        self,
        thc_ham,
        batched_rotations,
        rotation_prec,
        selswap_depth,
        num_ctrl_wires,
        num_zero_ctrl,
        expected_res,
    ):
        """Test that the controlled resource decompostion for SelectTHC is correct."""

        ctrl_select_cost = qre.estimate(
            qre.Controlled(
                num_ctrl_wires=num_ctrl_wires,
                num_zero_ctrl=num_zero_ctrl,
                base_op=qre.SelectTHC(
                    thc_ham,
                    batched_rotations=batched_rotations,
                    rotation_precision=rotation_prec,
                    select_swap_depth=selswap_depth,
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

    def test_value_error_batched_rotations(self):
        "Test that an error is raised when wrong value is provided for batched rotations."
        with pytest.raises(
            ValueError,
            match="`batched_rotations` must be a positive integer less than the number of orbitals 58, but got 60.",
        ):
            qre.SelectTHC(qre.THCHamiltonian(58, 160), batched_rotations=60)

        with pytest.raises(
            ValueError,
            match="`batched_rotations` must be a positive integer less than the number of orbitals 58, but got 0.5.",
        ):
            qre.SelectTHC.resource_rep(qre.THCHamiltonian(58, 160), batched_rotations=0.5)
