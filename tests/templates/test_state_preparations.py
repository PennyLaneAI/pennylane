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
Unit tests for the :mod:`pennylane.template.state_preparations` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop

import math
from unittest.mock import patch
import numpy as np
import pytest
import pennylane as qml
from pennylane.templates.state_preparations import (BasisStatePreparation,
                                                    MottonenStatePreparation)
from pennylane.templates.state_preparations.mottonen import gray_code


class TestHelperFunctions:
    """Tests the functionality of helper functions."""

    # fmt: off
    @pytest.mark.parametrize("rank,expected_gray_code", [
        (1, ['0', '1']),
        (2, ['00', '01', '11', '10']),
        (3, ['000', '001', '011', '010', '110', '111', '101', '100']),
    ])
    # fmt: on
    def test_gray_code(self, rank, expected_gray_code):
        """Tests that the function gray_code generates the proper
        Gray code of given rank."""

        assert gray_code(rank) == expected_gray_code


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
    def test_correct_pl_gates(self, basis_state, wires, target_wires):
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
    @pytest.mark.parametrize("basis_state,wires", [
        ([0], [0, 1]),
        ([0, 1], [0]),
    ])
    # fmt: on
    def test_error_num_qubits(self, basis_state, wires):
        """Tests that the correct error message is raised when the number
        of qubits doesn't match the number of wires."""

        with pytest.raises(ValueError, match="'basis_state' must be of shape"):
            BasisStatePreparation(basis_state, wires)

    # fmt: off
    @pytest.mark.parametrize("basis_state,wires", [
        ([3], [0]),
        ([1, 0, 2], [0, 1, 2]),
    ])
    # fmt: on
    def test_error_basis_state_format(self, basis_state, wires):
        """Tests that the correct error messages is raised when
        the basis state contains numbers different from 0 and 1."""

        with pytest.raises(ValueError, match="'basis_state' must only contain"):
            BasisStatePreparation(basis_state, wires)


class TestMottonenStatePreparation:
    """Tests the template MottonenStatePreparation."""

    # fmt: off
    @pytest.mark.parametrize("state_vector,wires,target_state", [
        ([1, 0], [0], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [1], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1], [0], [0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 1], [1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 1], [2], [0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 2], [0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 1], [1, 2], [0, 0, 0, 1, 0, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1j, 0, 0, 0], [0, 1, 2], [0, 0, 0, 0, 1j, 0, 0, 0]),
        ([1/2, 0, 0, 0, 1/2, 1j/2, -1/2, 0], [0, 1, 2], [1/2, 0, 0, 0, 1/2, 1j/2, -1/2, 0]),
        ([1/3, 0, 0, 0, 2j/3, 2j/3, 0, 0], [0, 1, 2], [1/3, 0, 0, 0, 2j/3, 2j/3, 0, 0]),
        ([2/3, 0, 0, 0, 1/3, 0, 0, 2/3], [0, 1, 2], [2/3, 0, 0, 0, 1/3, 0, 0, 2/3]),
        (
            [1/math.sqrt(8), 1j/math.sqrt(8), 1/math.sqrt(8), -1j/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1j/math.sqrt(8)],
            [0, 1, 2],
            [1/math.sqrt(8), 1j/math.sqrt(8), 1/math.sqrt(8), -1j/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1j/math.sqrt(8)],
        ),
        (
            [-0.17133152-0.18777771j, 0.00240643-0.40704011j, 0.18684538-0.36315606j, -0.07096948+0.104501j, 0.30357755-0.23831927j, -0.38735106+0.36075556j, 0.12351096-0.0539908j, 0.27942828-0.24810483j],
            [0, 1, 2],
            [-0.17133152-0.18777771j, 0.00240643-0.40704011j, 0.18684538-0.36315606j, -0.07096948+0.104501j, 0.30357755-0.23831927j, -0.38735106+0.36075556j, 0.12351096-0.0539908j, 0.27942828-0.24810483j],
        ),
        (
            [-0.29972867+0.04964242j, -0.28309418+0.09873227j, 0.00785743-0.37560696j, -0.3825148 +0.00674343j, -0.03008048+0.31119167j, 0.03666351-0.15935903j, -0.25358831+0.35461265j, -0.32198531+0.33479292j],
            [0, 1, 2],
            [-0.29972867+0.04964242j, -0.28309418+0.09873227j, 0.00785743-0.37560696j, -0.3825148 +0.00674343j, -0.03008048+0.31119167j, 0.03666351-0.15935903j, -0.25358831+0.35461265j, -0.32198531+0.33479292j],
        ),
        (
            [-0.39340123+0.05705932j, 0.1980509 -0.24234781j, 0.27265585-0.0604432j, -0.42641249+0.25767258j, 0.40386614-0.39925987j, 0.03924761+0.13193724j, -0.06059103-0.01753834j, 0.21707136-0.15887973j],
            [0, 1, 2],
            [-0.39340123+0.05705932j, 0.1980509 -0.24234781j, 0.27265585-0.0604432j, -0.42641249+0.25767258j, 0.40386614-0.39925987j, 0.03924761+0.13193724j, -0.06059103-0.01753834j, 0.21707136-0.15887973j],
        ),
        (
            [-1.33865287e-01+0.09802308j, 1.25060033e-01+0.16087698j, -4.14678130e-01-0.00774832j, 1.10121136e-01+0.37805482j, -3.21284864e-01+0.21521063j, -2.23121454e-04+0.28417422j, 5.64131205e-02+0.38135286j, 2.32694503e-01+0.41331133j],
            [0, 1, 2],
            [-1.33865287e-01+0.09802308j, 1.25060033e-01+0.16087698j, -4.14678130e-01-0.00774832j, 1.10121136e-01+0.37805482j, -3.21284864e-01+0.21521063j, -2.23121454e-04+0.28417422j, 5.64131205e-02+0.38135286j, 2.32694503e-01+0.41331133j],
        ),
        ([1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0], [0, 1, 2], [1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0]),
        ([1/2, 0, 1j/2, 1j/math.sqrt(2)], [0, 1], [1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0]),
    ])
    # fmt: on
    def test_state_preparation_fidelity(self, tol, qubit_device_3_wires, state_vector, wires, target_state):
        """Tests that the template MottonenStatePreparation integrates correctly with PennyLane
        and produces states with correct fidelity."""

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            MottonenStatePreparation(state_vector, wires)

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        circuit()

        state = qubit_device_3_wires._state
        fidelity = abs(np.vdot(state, target_state))**2

        # We test for fidelity here, because the vector themselves will hardly match
        # due to imperfect state preparation
        assert np.isclose(fidelity, 1, atol=tol, rtol=0)

    # fmt: off
    @pytest.mark.parametrize("state_vector,wires,target_state", [
        ([1, 0], [0], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [1], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([1, 0], [2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 1], [0], [0, 0, 0, 0, 1, 0, 0, 0]),
        ([0, 1], [1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 1], [2], [0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 0], [0, 1], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 2], [0, 0, 0, 0, 0, 1, 0, 0]),
        ([0, 0, 0, 1], [1, 2], [0, 0, 0, 1, 0, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 1j, 0, 0, 0], [0, 1, 2], [0, 0, 0, 0, 1j, 0, 0, 0]),
        ([1/2, 0, 0, 0, 1/2, 1j/2, -1/2, 0], [0, 1, 2], [1/2, 0, 0, 0, 1/2, 1j/2, -1/2, 0]),
        ([1/3, 0, 0, 0, 2j/3, 2j/3, 0, 0], [0, 1, 2], [1/3, 0, 0, 0, 2j/3, 2j/3, 0, 0]),
        ([2/3, 0, 0, 0, 1/3, 0, 0, 2/3], [0, 1, 2], [2/3, 0, 0, 0, 1/3, 0, 0, 2/3]),
        (
            [1/math.sqrt(8), 1j/math.sqrt(8), 1/math.sqrt(8), -1j/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1j/math.sqrt(8)],
            [0, 1, 2],
            [1/math.sqrt(8), 1j/math.sqrt(8), 1/math.sqrt(8), -1j/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1j/math.sqrt(8)],
        ),
        (
            [-0.17133152-0.18777771j, 0.00240643-0.40704011j, 0.18684538-0.36315606j, -0.07096948+0.104501j, 0.30357755-0.23831927j, -0.38735106+0.36075556j, 0.12351096-0.0539908j, 0.27942828-0.24810483j],
            [0, 1, 2],
            [-0.17133152-0.18777771j, 0.00240643-0.40704011j, 0.18684538-0.36315606j, -0.07096948+0.104501j, 0.30357755-0.23831927j, -0.38735106+0.36075556j, 0.12351096-0.0539908j, 0.27942828-0.24810483j],
        ),
        (
            [-0.29972867+0.04964242j, -0.28309418+0.09873227j, 0.00785743-0.37560696j, -0.3825148 +0.00674343j, -0.03008048+0.31119167j, 0.03666351-0.15935903j, -0.25358831+0.35461265j, -0.32198531+0.33479292j],
            [0, 1, 2],
            [-0.29972867+0.04964242j, -0.28309418+0.09873227j, 0.00785743-0.37560696j, -0.3825148 +0.00674343j, -0.03008048+0.31119167j, 0.03666351-0.15935903j, -0.25358831+0.35461265j, -0.32198531+0.33479292j],
        ),
        (
            [-0.39340123+0.05705932j, 0.1980509 -0.24234781j, 0.27265585-0.0604432j, -0.42641249+0.25767258j, 0.40386614-0.39925987j, 0.03924761+0.13193724j, -0.06059103-0.01753834j, 0.21707136-0.15887973j],
            [0, 1, 2],
            [-0.39340123+0.05705932j, 0.1980509 -0.24234781j, 0.27265585-0.0604432j, -0.42641249+0.25767258j, 0.40386614-0.39925987j, 0.03924761+0.13193724j, -0.06059103-0.01753834j, 0.21707136-0.15887973j],
        ),
        (
            [-1.33865287e-01+0.09802308j, 1.25060033e-01+0.16087698j, -4.14678130e-01-0.00774832j, 1.10121136e-01+0.37805482j, -3.21284864e-01+0.21521063j, -2.23121454e-04+0.28417422j, 5.64131205e-02+0.38135286j, 2.32694503e-01+0.41331133j],
            [0, 1, 2],
            [-1.33865287e-01+0.09802308j, 1.25060033e-01+0.16087698j, -4.14678130e-01-0.00774832j, 1.10121136e-01+0.37805482j, -3.21284864e-01+0.21521063j, -2.23121454e-04+0.28417422j, 5.64131205e-02+0.38135286j, 2.32694503e-01+0.41331133j],
        ),
        ([1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0], [0, 1, 2], [1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0]),
        ([1/2, 0, 1j/2, 1j/math.sqrt(2)], [0, 1], [1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0]),
    ])
    # fmt: on
    def test_state_preparation_probability_distribution(self, tol, qubit_device_3_wires, state_vector, wires, target_state):
        """Tests that the template MottonenStatePreparation integrates correctly with PennyLane
        and produces states with correct probability distribution."""

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            MottonenStatePreparation(state_vector, wires)

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        circuit()

        state = qubit_device_3_wires._state

        probabilities = np.abs(state)**2
        target_probabilities = np.abs(target_state)**2

        assert np.allclose(probabilities, target_probabilities, atol=tol, rtol=0)

    # fmt: off
    @pytest.mark.parametrize("state_vector, wires", [
        ([1/2, 1/2], [0]),
        ([2/3, 0, 2j/3, -2/3], [0, 1]),
    ])
    # fmt: on
    def test_error_state_vector_not_normalized(self, state_vector, wires):
        """Tests that the correct error messages is raised if
        the given state vector is not normalized."""

        with pytest.raises(ValueError, match="'state_vector' has to be of length"):
            MottonenStatePreparation(state_vector, wires)

    # fmt: off
    @pytest.mark.parametrize("state_vector,wires", [
        ([0, 1, 0], [0, 1]),
        ([0, 1, 0, 0, 0], [0]),
    ])
    # fmt: on
    def test_error_num_entries(self, state_vector, wires):
        """Tests that the correct error messages is raised  if
        the number of entries in the given state vector does not match
        with the number of wires in the system."""

        with pytest.raises(ValueError, match="'state_vector' must be of shape"):
            MottonenStatePreparation(state_vector, wires)

