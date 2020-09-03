# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the circuit implementations required in measurement optimization found in
`grouping/transformations.py`.
"""
import pytest
import numpy as np
from pennylane import PauliX, PauliY, PauliZ, Identity, Hadamard, RX, RY
from pennylane.grouping.utils import are_identical_pauli_words
from pennylane.grouping.transformations import qwc_rotation, diagonalize_qwc_grouping


class TestMeasurementTransformations:
    """Tests for the functions involved in obtaining post-rotations necessary in the measurement
    optimization schemes implemented in :mod:`grouping`."""

    def are_identical_rotation_gates(self, gate_1, gate_2, param_tol=1e-6):
        """Checks whether the two input gates are identical up to a certain threshold in their
        parameters.

        Arguments:
            gate_1 (Union[RX, RY, RZ]): the first single-qubit rotation gate.
            gate_2 (Union[RX, RY, RZ]): the second single-qubit rotation gate.

        Keyword arguments:
            param_tol (float): the relative tolerance for considering whether two gates parameter
                values are the same.

        Returns:
            bool: whether the input rotation gates are identical up to the parameter tolerance.

        """

        return (
            gate_1.wires == gate_2.wires
            and np.isclose(gate_1.parameters, gate_2.parameters, param_tol).all()
            and gate_1.name == gate_2.name
        )

    qwc_rotation_io = [
        ([PauliX(0), PauliZ(1), PauliZ(3)], [RY(-np.pi / 2, wires=[0])]),
        ([Identity(0), PauliZ(1)], []),
        (
            [PauliX(2), PauliY(0), Identity(1)],
            [RY(-np.pi / 2, wires=[2]), RX(np.pi / 2, wires=[0])],
        ),
        (
            [PauliZ("a"), PauliX("b"), PauliY(0)],
            [RY(-np.pi / 2, wires=["b"]), RX(np.pi / 2, wires=[0])],
        ),
    ]

    @pytest.mark.parametrize("pauli_ops,qwc_rot_sol", qwc_rotation_io)
    def test_qwc_rotation(self, pauli_ops, qwc_rot_sol):
        """Tests that the correct single-qubit post-rotation gates are obtained for the input list
        of Pauli operators."""

        qwc_rot = qwc_rotation(pauli_ops)

        assert all(
            [
                self.are_identical_rotation_gates(qwc_rot[i], qwc_rot_sol[i])
                for i in range(len(qwc_rot))
            ]
        )

    invalid_qwc_rotation_inputs = [
        [PauliX(0), PauliY(1), Hadamard(2)],
        [PauliX(0) @ PauliY(1), PauliZ(1), Identity(2)],
        [RX(1, wires="a"), PauliX("b")],
    ]

    @pytest.mark.parametrize("bad_input", invalid_qwc_rotation_inputs)
    def test_invalid_qwc_rotation_input_catch(self, bad_input):
        """Verifies that a TypeError is raised when the input to qwc_rotations is not a list of
        single Pauli operators."""

        assert pytest.raises(TypeError, qwc_rotation, bad_input)

    qwc_diagonalization_io = [
        (
            [PauliX(0) @ PauliY(1), PauliX(0) @ PauliZ(2)],
            (
                [RY(-np.pi / 2, wires=[0]), RX(np.pi / 2, wires=[1])],
                [PauliZ(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[0]) @ PauliZ(wires=[2])],
            ),
        ),
        (
            [PauliX(2) @ Identity(0), PauliY(1), PauliZ(0) @ PauliY(1), PauliX(2) @ PauliY(1)],
            (
                [RY(-np.pi / 2, wires=[2]), RX(np.pi / 2, wires=[1])],
                [
                    PauliZ(wires=[2]),
                    PauliZ(wires=[1]),
                    PauliZ(wires=[0]) @ PauliZ(wires=[1]),
                    PauliZ(wires=[2]) @ PauliZ(wires=[1]),
                ],
            ),
        ),
        (
            [PauliZ("a") @ PauliY("b") @ PauliZ("c"), PauliY("b") @ PauliZ("d")],
            (
                [RX(np.pi / 2, wires=["b"])],
                [
                    PauliZ(wires=["a"]) @ PauliZ(wires=["b"]) @ PauliZ(wires=["c"]),
                    PauliZ(wires=["b"]) @ PauliZ(wires=["d"]),
                ],
            ),
        ),
        ([PauliX("a")], ([RY(-np.pi / 2, wires=["a"])], [PauliZ(wires=["a"])])),
    ]

    @pytest.mark.parametrize("qwc_grouping,qwc_sol_tuple", qwc_diagonalization_io)
    def test_diagonalize_qwc_grouping(self, qwc_grouping, qwc_sol_tuple):

        qwc_rot, diag_qwc_grouping = diagonalize_qwc_grouping(qwc_grouping)
        qwc_rot_sol, diag_qwc_grouping_sol = qwc_sol_tuple

        assert all(
            [
                self.are_identical_rotation_gates(qwc_rot[i], qwc_rot_sol[i])
                for i in range(len(qwc_rot))
            ]
        )
        assert all(
            [
                are_identical_pauli_words(diag_qwc_grouping[i], diag_qwc_grouping_sol[i])
                for i in range(len(diag_qwc_grouping))
            ]
        )

    not_qwc_groupings = [
        [PauliX("a"), PauliY("a")],
        [PauliZ(0) @ Identity(1), PauliZ(0) @ PauliZ(1), PauliX(0) @ Identity(1)],
        [PauliX("a") @ PauliX(0), PauliZ(0) @ PauliZ("a")],
    ]

    @pytest.mark.parametrize("not_qwc_grouping", not_qwc_groupings)
    def test_diagonalize_qwc_grouping_catch_when_not_qwc(self, not_qwc_grouping):
        assert pytest.raises(ValueError, diagonalize_qwc_grouping, not_qwc_grouping)
