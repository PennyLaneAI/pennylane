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
import math

import pennylane as qml
from pennylane.templates.state_preparations import BasisStatePreparation, MottonenStatePreparation

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

class TestMottonenStatePreparation:
    """Tests the template MottonenStatePreparation."""

    
    # def test_basis_state_mapping(self, num_wires):
    #     """Tests that MottonenStatePreparation maps basis states to basis states."""
    #     eye = np.eye(num_wires**2)
    #     dev = qml.device("default.qubit", wires=num_wires)

    #     idx_maps = []
    #     bin_maps = []

    #     for row in eye:
    #         dev.reset()

    #         @qml.qnode(dev)
    #         def circuit():
    #             MottonenStatePreparation(row, [0, 1, 2, 3])

    #             return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))

    #         circuit()

    #         state = dev._state

    #         assert 

    #         print("Circuit maps\n    {}\n -> {}".format(row, np.real(np.round(state, 2))))
    #         idx_maps.append((np.argmax(row), np.argmax(np.real(np.round(state, 2)))))
    #         bin_maps.append(("{0:04b}".format(np.argmax(row)), "{0:04b}".format(np.argmax(np.real(np.round(state, 2))))))

    #     #print(idx_maps)
    #     #print(bin_maps)
    #     for i in range(16):
    #         print("{} -> {}, {} -> {}".format(idx_maps[i][0], idx_maps[i][1], bin_maps[i][0], bin_maps[i][1]))
    #     raise Exception("X")


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
            [1/math.sqrt(8), 1j/math.sqrt(8), 1/math.sqrt(8), -1j/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1/math.sqrt(8), 1j/math.sqrt(8)]
        ),
        (
            [-0.17133152-0.18777771j, 0.00240643-0.40704011j, 0.18684538-0.36315606j, -0.07096948+0.104501j, 0.30357755-0.23831927j, -0.38735106+0.36075556j, 0.12351096-0.0539908j, 0.27942828-0.24810483j],
            [0, 1, 2],
            [-0.17133152-0.18777771j, 0.00240643-0.40704011j, 0.18684538-0.36315606j, -0.07096948+0.104501j, 0.30357755-0.23831927j, -0.38735106+0.36075556j, 0.12351096-0.0539908j, 0.27942828-0.24810483j],
        ),
        (
            [-0.29972867+0.04964242j, -0.28309418+0.09873227j,  0.00785743-0.37560696j,
  -0.3825148 +0.00674343j, -0.03008048+0.31119167j,  0.03666351-0.15935903j,   -0.25358831+0.35461265j, -0.32198531+0.33479292j],
            [0, 1, 2],
            [-0.29972867+0.04964242j, -0.28309418+0.09873227j,  0.00785743-0.37560696j,
  -0.3825148 +0.00674343j, -0.03008048+0.31119167j,  0.03666351-0.15935903j,   -0.25358831+0.35461265j, -0.32198531+0.33479292j],
        ),
        (
            [-0.39340123+0.05705932j,  0.1980509 -0.24234781j,  0.27265585-0.0604432j,
  -0.42641249+0.25767258j,  0.40386614-0.39925987j,  0.03924761+0.13193724j,
  -0.06059103-0.01753834j,  0.21707136-0.15887973j],
            [0, 1, 2],
            [-0.39340123+0.05705932j,  0.1980509 -0.24234781j,  0.27265585-0.0604432j,
  -0.42641249+0.25767258j,  0.40386614-0.39925987j,  0.03924761+0.13193724j,
  -0.06059103-0.01753834j,  0.21707136-0.15887973j]
        ),
        (
            [-1.33865287e-01+0.09802308j,  1.25060033e-01+0.16087698j,
  -4.14678130e-01-0.00774832j,  1.10121136e-01+0.37805482j,
  -3.21284864e-01+0.21521063j, -2.23121454e-04+0.28417422j,
   5.64131205e-02+0.38135286j,  2.32694503e-01+0.41331133j],
            [0, 1, 2],
            [-1.33865287e-01+0.09802308j,  1.25060033e-01+0.16087698j,
  -4.14678130e-01-0.00774832j,  1.10121136e-01+0.37805482j,
  -3.21284864e-01+0.21521063j, -2.23121454e-04+0.28417422j,
   5.64131205e-02+0.38135286j,  2.32694503e-01+0.41331133j],
        ),
        ([1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0], [0, 1, 2], [1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0]),
        ([1/2, 0, 1j/2, 1j/math.sqrt(2)], [0, 1], [1/2, 0, 0, 0, 1j/2, 0, 1j/math.sqrt(2), 0]),
    ])
    # fmt: on
    def test_state_preparation(self, tol, qubit_device_3_wires, state_vector, wires, target_state):
        """Tests that the template MottonenStatePreparation integrates correctly with PennyLane."""

        @qml.qnode(qubit_device_3_wires)
        def circuit():
            MottonenStatePreparation(state_vector, wires)

            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        circuit()

        state = qubit_device_3_wires._state
        fidelity = abs(np.vdot(state, target_state))**2

        # print("state: ", state)
        # print("target: ", target_state)
        # print("fidelity: ", fidelity)

        # We test for fidelity here, because the vector themselves will hardly match
        # due to imperfect state preparation
        assert np.isclose(fidelity, 1, atol=tol, rtol=0)

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

