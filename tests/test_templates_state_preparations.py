# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.template.state_preparations` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import pennylane as qml
from pennylane.templates.state_preparations import BasisStatePreparation

class TestBasisStatePreparation:
    """Tests the template BasisStatePreparation."""

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires,target_wires", [
        ([0], [0], []),
        ([0], [1], []),
        ([1], [0], [0]),
        ([1], [1], [1]),
        ([0, 1], [0, 1], [1]),
        ([1, 0], [1, 4], [1]),
        ([1, 1], [0, 2], [0, 2]),
        ([1, 0], [4, 5], [4]),
        ([0, 0, 1, 0], [1, 2, 3, 4], [3]),
        ([1, 1, 1, 0], [1, 2, 6, 8], [1, 2, 6]),
        ([1, 0, 1, 1], [1, 2, 6, 8], [1, 6, 8]),
    ])
    # fmt: on
    def test_correct_pl_gates(self, tol, basis_state, wires, target_wires):
        """Tests that the template BasisStatePreparation calls the correct
        PennyLane gates on the correct wires."""

        with patch("pennylane.PauliX") as mock:
            BasisStatePreparation(basis_state, wires)

            called_wires = [args[0] for args, kwargs in mock.call_args_list]

            assert len(target_wires) == len(called_wires)
            assert np.array_equal(called_wires, target_wires)

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires,target_state", [
        ([0], [0], [0, 0, 0]),
        ([0], [1], [0, 0, 0]),
        ([1], [0], [1, 0, 0]),
        ([1], [1], [0, 1, 0]),
        ([0, 1], [0, 1], [0, 1, 0]),
        ([1, 1], [0, 2], [1, 0, 1]),
        ([1, 1], [1, 2], [0, 1, 1]),
        ([1, 0], [0, 2], [1, 0, 0]),
        ([1, 1, 0], [0, 1, 2], [1, 1, 0]),
        ([1, 0, 1], [0, 1, 2], [1, 0, 1]),
    ])
    # fmt: on
    def test_state_preparation(self, tol, qubit_device_3_wires, basis_state, wires, target_state):
        """Tests that the template BasisStatePreparation integrates correctly with PennyLane."""

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            BasisStatePreparation(basis_state, wires)

            # Pauli Z gates identify the basis state
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        # Convert from Pauli Z eigenvalues to basis state
        output_state = [0 if x == 1.0 else 1 for x in circuit()]

        assert np.allclose(output_state, target_state, atol=tol, rtol=0)

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires,error_message", [
        ([0], [0, 1], "Number of qubits must be equal to the number of wires"),
        ([0, 1], [0], "Number of qubits must be equal to the number of wires"),
        ([0], 0, "Wires needs to be a list of wires that the embedding uses"),
        ([3], [0], "Basis state must only consist of 0s and 1s"),
        ([1, 0, 2], [0, 1, 2], "Basis state must only consist of 0s and 1s"),
    ])
    # fmt: on
    def test_errors(self, basis_state, wires, error_message):
        """Tests that the correct error messages are raised."""

        with pytest.raises(ValueError, match=error_message):
            BasisStatePreparation(basis_state, wires)

