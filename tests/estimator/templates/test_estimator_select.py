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
import re

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
        "thc_ham, num_batches, rotation_prec, selswap_depth",
        (
            (qre.THCHamiltonian(58, 160), 1, 13, 1),
            (qre.THCHamiltonian(10, 50), 1, None, None),
            (qre.THCHamiltonian(4, 20), 2, None, 2),
        ),
    )
    def test_resource_params(self, thc_ham, num_batches, rotation_prec, selswap_depth):
        """Test that the resource params for SelectTHC are correct."""
        if rotation_prec:
            op = qre.SelectTHC(thc_ham, num_batches, rotation_prec, selswap_depth)
        else:
            op = qre.SelectTHC(thc_ham, num_batches=num_batches, select_swap_depth=selswap_depth)
            rotation_prec = 15

        assert op.resource_params == {
            "thc_ham": thc_ham,
            "num_batches": num_batches,
            "rotation_precision": rotation_prec,
            "select_swap_depth": selswap_depth,
        }

    @pytest.mark.parametrize(
        "thc_ham, num_batches, rotation_prec, selswap_depth, num_wires",
        (
            (qre.THCHamiltonian(58, 160), 1, 13, 1, 138),
            (qre.THCHamiltonian(10, 50), 1, None, None, 38),
            (qre.THCHamiltonian(4, 20), 2, None, 2, 24),
        ),
    )
    def test_resource_rep(self, thc_ham, num_batches, rotation_prec, selswap_depth, num_wires):
        """Test that the compressed representation for SelectTHC is correct."""
        if rotation_prec:
            expected = qre.SelectTHC(
                thc_ham,
                num_batches=num_batches,
                rotation_precision=rotation_prec,
                select_swap_depth=selswap_depth,
            )
            assert (
                qre.SelectTHC.resource_rep(thc_ham, num_batches, rotation_prec, selswap_depth)
                == expected
            )
        else:
            expected = qre.SelectTHC(
                thc_ham,
                num_batches=num_batches,
                rotation_precision=15,
                select_swap_depth=selswap_depth,
            )
            assert (
                qre.SelectTHC.resource_rep(
                    thc_ham, num_batches=num_batches, select_swap_depth=selswap_depth
                )
                == expected
            )

    # The Toffoli and qubit costs are compared here
    # Expected number of Toffolis and wires were obtained from Eq. 44 and 46 in https://arxiv.org/abs/2011.03494
    # The numbers were adjusted slightly to account for removal of phase gradient state and a different QROM decomposition
    @pytest.mark.parametrize(
        "thc_ham, num_batches, rotation_prec, selswap_depth, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                1,
                13,
                1,
                {"algo_wires": 138, "auxiliary_wires": 752, "toffoli_gates": 5997},
            ),
            (
                qre.THCHamiltonian(10, 50),
                1,
                15,
                None,
                {"algo_wires": 38, "auxiliary_wires": 148, "toffoli_gates": 1189},
            ),
            (
                qre.THCHamiltonian(4, 20),
                1,
                15,
                2,
                {"algo_wires": 24, "auxiliary_wires": 93, "toffoli_gates": 457},
            ),
            # These numbers were obtained manually for batched rotations based on the technique described in arXiv:2501.06165
            (
                qre.THCHamiltonian(58, 160),
                2,
                13,
                None,
                {"algo_wires": 138, "auxiliary_wires": 388, "toffoli_gates": 6371},
            ),
        ),
    )
    def test_resources(self, thc_ham, num_batches, rotation_prec, selswap_depth, expected_res):
        """Test that the resource decompostion for SelectTHC is correct."""

        select_cost = qre.estimate(
            qre.SelectTHC(
                thc_ham,
                num_batches=num_batches,
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
        "thc_ham, num_batches, rotation_prec, selswap_depth, num_ctrl_wires, num_zero_ctrl, expected_res",
        (
            (
                qre.THCHamiltonian(58, 160),
                1,
                13,
                1,
                1,
                1,
                {"algo_wires": 139, "auxiliary_wires": 752, "toffoli_gates": 5998},
            ),
            (
                qre.THCHamiltonian(10, 50),
                1,
                15,
                None,
                2,
                0,
                {"algo_wires": 40, "auxiliary_wires": 149, "toffoli_gates": 1192},
            ),
            (
                qre.THCHamiltonian(4, 20),
                1,
                15,
                2,
                3,
                2,
                {"algo_wires": 27, "auxiliary_wires": 94, "toffoli_gates": 462},
            ),
            # These numbers were obtained manually for batched rotations based on the technique described in arXiv:2501.06165
            (
                qre.THCHamiltonian(58, 160),
                2,
                13,
                None,
                1,
                1,
                {"algo_wires": 139, "auxiliary_wires": 388, "toffoli_gates": 6372},
            ),
        ),
    )
    def test_controlled_resources(
        self,
        thc_ham,
        num_batches,
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
                    num_batches=num_batches,
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
            match=f"`rotation_precision` must be a positive integer, but got {2.5}.",
        ):
            qre.SelectTHC(qre.THCHamiltonian(58, 160), rotation_precision=2.5)

        with pytest.raises(
            TypeError,
            match=f"`rotation_precision` must be a positive integer, but got {2.5}.",
        ):
            qre.SelectTHC.resource_rep(qre.THCHamiltonian(58, 160), rotation_precision=2.5)

    def test_value_error_num_batches(self):
        "Test that an error is raised when wrong value is provided for batched rotations."
        with pytest.raises(
            ValueError,
            match=re.escape(
                "`num_batches` must be a positive integer less than the number of orbitals (58), but got 60."
            ),
        ):
            qre.SelectTHC(qre.THCHamiltonian(58, 160), num_batches=60)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "`num_batches` must be a positive integer less than the number of orbitals (58), but got 0.5."
            ),
        ):
            qre.SelectTHC.resource_rep(qre.THCHamiltonian(58, 160), num_batches=0.5)


class TestSelectPauli:
    """Test the SelectPauli class."""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        pauli_ham = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        with pytest.raises(ValueError, match=r"Expected 3 wires \(1 control \+ 2 target\), got 2"):
            qre.SelectPauli(pauli_ham, wires=[0, 1])

    def test_ham_error(self):
        """Test that an error is raised if the input Hamiltonian
        is not the correct type."""
        ham = qre.THCHamiltonian(num_orbitals=5, tensor_rank=5)
        with pytest.raises(TypeError, match="'pauli_ham' must be an instance of PauliHamiltonian"):
            _ = qre.SelectPauli(ham)

    def test_resource_params(self):
        """Test that the resource params for SelectPauli are correct."""
        pauli_ham = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        op = qre.SelectPauli(pauli_ham)
        assert op.resource_params == {"pauli_ham": pauli_ham}

    def test_resource_rep(self):
        """Test that the compressed representation for SelectPauli is correct."""
        pauli_ham = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        expected = qre.SelectPauli(pauli_ham)
        assert qre.SelectPauli.resource_rep(pauli_ham) == expected

    @pytest.mark.parametrize(
        "pauli_ham, expected_res",
        (
            (
                qre.PauliHamiltonian(2, {"X": 1, "Z": 1}),
                [
                    qre.Allocate(0),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.CY), 0),
                    qre.GateCount(qre.resource_rep(qre.CZ), 1),
                    qre.GateCount(qre.resource_rep(qre.X), 2),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.TemporaryAND), 1),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)},
                        ),
                        1,
                    ),
                    qre.Deallocate(0),
                ],
            ),
            (
                qre.PauliHamiltonian(3, {"XY": 1, "Z": 2, "Y": 1}),
                [
                    qre.Allocate(1),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.CY), 2),
                    qre.GateCount(qre.resource_rep(qre.CZ), 2),
                    qre.GateCount(qre.resource_rep(qre.X), 6),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 3),
                    qre.GateCount(qre.resource_rep(qre.TemporaryAND), 3),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)},
                        ),
                        3,
                    ),
                    qre.Deallocate(1),
                ],
            ),
            (
                qre.PauliHamiltonian(2, [{"X": 1}, {"Z": 1}]),
                [
                    qre.Allocate(0),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.CY), 0),
                    qre.GateCount(qre.resource_rep(qre.CZ), 1),
                    qre.GateCount(qre.resource_rep(qre.X), 2),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.TemporaryAND), 1),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)},
                        ),
                        1,
                    ),
                    qre.Deallocate(0),
                ],
            ),
        ),
    )
    def test_resources(self, pauli_ham, expected_res):
        """Test that the resource decompostion for SelectPauli is correct."""
        op = qre.SelectPauli(pauli_ham)
        res = op.resource_decomp(pauli_ham)
        assert res == expected_res

    def test_adjoint_resources(self):
        """Test that the adjoint resource decomposition is correct."""
        pauli_ham = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        op = qre.SelectPauli(pauli_ham)
        res = op.adjoint_resource_decomp(op.resource_params)

        assert len(res) == 1
        assert res[0].gate == qre.SelectPauli.resource_rep(pauli_ham)
        assert res[0].count == 1

    @pytest.mark.parametrize(
        "pauli_ham, num_ctrl_wires, num_zero_ctrl, expected_res",
        (
            (
                qre.PauliHamiltonian(2, {"X": 1, "Z": 1}),
                1,
                0,
                [
                    qre.Allocate(1),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.CY), 0),
                    qre.GateCount(qre.resource_rep(qre.CZ), 1),
                    qre.GateCount(qre.resource_rep(qre.X), 2),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.TemporaryAND), 1),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)},
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                ],
            ),
            (
                qre.PauliHamiltonian(2, {"X": 1, "Z": 1}),
                1,
                1,
                [
                    qre.GateCount(qre.resource_rep(qre.X), 2),
                    qre.Allocate(1),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.CY), 0),
                    qre.GateCount(qre.resource_rep(qre.CZ), 1),
                    qre.GateCount(qre.resource_rep(qre.X), 2),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.TemporaryAND), 1),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)},
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                ],
            ),
            (
                qre.PauliHamiltonian(2, {"X": 1, "Z": 1}),
                2,
                0,
                [
                    qre.Allocate(1),
                    qre.GateCount(qre.MultiControlledX.resource_rep(2, 0), 1),
                    qre.Allocate(1),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.CY), 0),
                    qre.GateCount(qre.resource_rep(qre.CZ), 1),
                    qre.GateCount(qre.resource_rep(qre.X), 2),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.TemporaryAND), 1),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)},
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                    qre.GateCount(qre.MultiControlledX.resource_rep(2, 0), 1),
                    qre.Deallocate(1),
                ],
            ),
            (
                qre.PauliHamiltonian(2, [{"X": 1}, {"Z": 1}]),
                1,
                0,
                [
                    qre.Allocate(1),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.CY), 0),
                    qre.GateCount(qre.resource_rep(qre.CZ), 1),
                    qre.GateCount(qre.resource_rep(qre.X), 2),
                    qre.GateCount(qre.resource_rep(qre.CNOT), 1),
                    qre.GateCount(qre.resource_rep(qre.TemporaryAND), 1),
                    qre.GateCount(
                        qre.resource_rep(
                            qre.Adjoint,
                            {"base_cmpr_op": qre.resource_rep(qre.TemporaryAND)},
                        ),
                        1,
                    ),
                    qre.Deallocate(1),
                ],
            ),
        ),
    )
    def test_controlled_resources(self, pauli_ham, num_ctrl_wires, num_zero_ctrl, expected_res):
        """Test that the controlled resource decomposition is correct."""
        op = qre.SelectPauli(pauli_ham)
        res = op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl, op.resource_params)
        assert res == expected_res

    def test_init_with_wires(self):
        """Test initialization with wires."""
        pauli_ham = qre.PauliHamiltonian(2, {"X": 1, "Z": 1})
        wires = [0, 1, 2]
        op = qre.SelectPauli(pauli_ham, wires=wires)
        assert list(op.wires) == wires
