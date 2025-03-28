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
"""Test the Compact input classes for resource estimation."""
import math

import pytest

import pennylane as qml
import pennylane.labs.resource_estimation as re
from pennylane import numpy as np


class TestCompactState:
    """Test the compact state class"""

    attribute_names = (
        "num_qubits",
        "num_coeffs",
        "precision",
        "num_work_wires",
        "num_bit_flips",
        "positive_and_real",
    )

    @pytest.mark.parametrize(
        "num_mps_mats, max_bond_dim",
        (
            (5, 3),
            (7, 4),
            (10, 10),
        ),
    )
    def test_from_mps(self, num_mps_mats, max_bond_dim):
        """Test that the CompactState is instantiated correctly from mps"""

        expected = re.CompactState(
            num_qubits=num_mps_mats, num_work_wires=math.ceil(math.log2(max_bond_dim))
        )
        computed = re.CompactState.from_mps(
            num_mps_matrices=num_mps_mats, max_bond_dim=max_bond_dim
        )
        assert expected == computed

    @pytest.mark.parametrize(
        "num_qubits, num_bit_flips",
        (
            (5, 16),
            (7, 100),
            (3, 2),
        ),
    )
    def test_from_bitstring(self, num_qubits, num_bit_flips):
        """Test that the CompactState is instantiated correctly from a bitstring"""
        expected = re.CompactState(num_qubits=num_qubits, num_coeffs=1, num_bit_flips=num_bit_flips)
        computed = re.CompactState.from_bitstring(
            num_qubits=num_qubits, num_bit_flips=num_bit_flips
        )
        assert expected == computed

    @pytest.mark.parametrize(
        "num_qubits, num_coeffs, precision, num_work_wires, positive_and_real",
        (
            (5, 32, 1e-3, 2, True),
            (10, 5, 1e-5, 4, False),
            (6, 16, 1e-3, 10, False),
        ),
    )
    def test_from_state_vector(
        self, num_qubits, num_coeffs, precision, num_work_wires, positive_and_real
    ):
        """Test that the CompactState is instantiated correctly from a bitstring"""
        expected = re.CompactState(
            num_qubits=num_qubits,
            num_coeffs=num_coeffs,
            num_work_wires=num_work_wires,
            precision=precision,
            positive_and_real=positive_and_real,
        )
        computed = re.CompactState.from_state_vector(
            num_qubits=num_qubits,
            num_coeffs=num_coeffs,
            precision=precision,
            num_work_wires=num_work_wires,
            positive_and_real=positive_and_real,
        )
        assert expected == computed


class TestIntegration:
    """Test that the compact classes integrate with existing classes"""

    @pytest.mark.parametrize(
        "compact_state, wires",
        (
            (
                re.CompactState.from_state_vector(num_qubits=10, num_coeffs=20),
                range(10),
            ),
            (
                re.CompactState.from_state_vector(
                    num_qubits=4, num_coeffs=16, num_work_wires=1, positive_and_real=True
                ),
                range(4),
            ),
        ),
    )
    def test_stateprep_template(self, compact_state, wires):
        """Test that we can get the resources of ResourceStatePrep with compact states"""

        def circ():
            re.ResourceStatePrep(compact_state, wires)
            return

        def expected_circ():
            state = np.random.rand(2 ** len(wires))
            state = state / np.linalg.norm(state)
            re.ResourceStatePrep(state, wires)
            return

        assert re.get_resources(circ)() == re.get_resources(expected_circ)()

    @pytest.mark.parametrize(
        "compact_state, wires, bit_flips",
        (
            (
                re.CompactState.from_bitstring(num_qubits=10, num_bit_flips=10),
                range(10),
                10,
            ),
            (
                re.CompactState.from_bitstring(num_qubits=4, num_bit_flips=2),
                range(4),
                2,
            ),
        ),
    )
    def test_basis_state_template(self, compact_state, wires, bit_flips):
        """Test that we can get the resources of ResourceBasisState with compact states"""
        state = [1] * bit_flips + ([0] * (len(wires) - bit_flips))

        def circ():
            re.ResourceBasisState(compact_state, wires)
            return

        def expected_circ():
            re.ResourceBasisState(state, wires)
            return

        assert re.get_resources(circ)() == re.get_resources(expected_circ)()

    @pytest.mark.parametrize(
        "compact_state, wires, num_coeffs",
        (
            (
                re.CompactState.from_state_vector(num_qubits=10, num_coeffs=20),
                range(10),
                20,
            ),
            (
                re.CompactState.from_state_vector(
                    num_qubits=4, num_coeffs=16, num_work_wires=1, positive_and_real=True
                ),
                range(4),
                16,
            ),
        ),
    )
    def test_superposition_template(self, compact_state, wires, num_coeffs):
        """Test that we can get the resources of ResourceSuperposition with compact states"""

        def circ():
            re.ResourceSuperposition(state_vect=compact_state, wires=wires, work_wire=["w1"])
            return

        coeffs = np.random.rand(num_coeffs)
        coeffs = coeffs / np.linalg.norm(coeffs)

        bases = []
        for i in range(num_coeffs):
            bin_index = format(i, "b")
            base = [int(char) for char in bin_index[::-1]] + [0] * (len(wires) - len(bin_index))
            bases.append(base)

        def expected_circ():
            re.ResourceSuperposition(coeffs, bases, wires, work_wire=["w1"])
            return

        assert re.get_resources(circ)() == re.get_resources(expected_circ)()

    @pytest.mark.parametrize(
        "compact_state, wires",
        (
            (
                re.CompactState.from_state_vector(num_qubits=10, num_coeffs=20),
                range(10),
            ),
            (
                re.CompactState.from_state_vector(
                    num_qubits=4, num_coeffs=16, num_work_wires=1, positive_and_real=True
                ),
                range(4),
            ),
        ),
    )
    def test_mottonen_template(self, compact_state, wires):
        """Test that we can get the resources of ResourceMottonenStatePreparation with compact states"""

        def circ():
            re.ResourceMottonenStatePreparation(compact_state, wires)
            return

        def expected_circ():
            state = np.random.rand(2 ** len(wires))
            state = state / np.linalg.norm(state)
            re.ResourceMottonenStatePreparation(state, wires)
            return

        assert re.get_resources(circ)() == re.get_resources(expected_circ)()

    @pytest.mark.parametrize(
        "compact_state, mps, wires, work_wires",
        (
            (
                re.CompactState.from_mps(num_mps_matrices=4, max_bond_dim=4),
                [
                    np.array([[0.70710678, 0.0], [0.0, 0.70710678]]),
                    np.array(
                        [
                            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, -0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
                        ]
                    ),
                    np.array(
                        [
                            [[0.00000000e00, 1.74315280e-32], [-7.07106781e-01, -7.07106781e-01]],
                            [[7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                            [[0.00000000e00, 0.00000000e00], [-7.07106781e-01, 7.07106781e-01]],
                            [[-7.07106781e-01, 7.07106781e-01], [0.00000000e00, 0.00000000e00]],
                        ]
                    ),
                    np.array([[1.0, 0.0], [0.0, 1.0]]),
                ],
                range(4),
                list(f"w{j}" for j in range(2)),
            ),
            (
                re.CompactState.from_mps(num_mps_matrices=5, max_bond_dim=4),
                [
                    np.array([[0.53849604, -0.44389787], [-0.59116842, -0.40434711]]),
                    np.array(
                        [
                            [
                                [-6.05052107e-01, 1.34284016e-01, -2.84018989e-01, -1.12416345e-01],
                                [3.60988555e-01, 6.14571922e-01, -1.20681653e-01, -2.89527967e-04],
                            ],
                            [
                                [-1.12393068e-01, 2.11496619e-01, 3.99193070e-01, -3.44891522e-01],
                                [-5.99232567e-01, 3.76491467e-01, 3.04813277e-01, 2.65697349e-01],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [
                                [0.87613189, -0.34254341, -0.12704983, -0.0161698],
                                [-0.20758717, 0.1329479, -0.18184107, 0.06942658],
                            ],
                            [
                                [-0.16499137, -0.13680142, -0.18432824, 0.12950892],
                                [-0.68790868, -0.64141472, -0.12485688, -0.0556177],
                            ],
                            [
                                [0.0352582, -0.37993402, 0.26781956, -0.25935129],
                                [0.04351872, -0.27109361, 0.65111429, 0.4648453],
                            ],
                            [
                                [0.1909576, 0.25461839, -0.07463641, -0.34390477],
                                [-0.21279487, 0.0305474, 0.53420894, -0.66578494],
                            ],
                        ]
                    ),
                    np.array(
                        [
                            [[-0.26771292, -0.00628612], [-0.96316273, 0.02465422]],
                            [[0.96011241, 0.07601506], [-0.2663889, 0.03798452]],
                            [[-0.00727353, 0.4537835], [-0.02374101, -0.89076596]],
                            [[0.08038064, -0.88784161], [-0.02812246, -0.45220057]],
                        ]
                    ),
                    np.array([[-0.97855153, 0.2060022], [0.2060022, 0.97855153]]),
                ],
                range(5),
                list(f"w{j}" for j in range(2)),
            ),
            (
                re.CompactState.from_mps(num_mps_matrices=3, max_bond_dim=2),
                [
                    np.array([[0.0, 0.107], [0.994, 0.0]]),
                    np.array(
                        [
                            [[0.0, 0.0], [1.0, 0.0]],
                            [[0.0, 1.0], [0.0, 0.0]],
                        ]
                    ),
                    np.array([[-1.0, -0.0], [-0.0, -1.0]]),
                ],
                range(3),
                list(f"w{j}" for j in range(1)),
            ),
        ),
    )
    def test_mps_template(self, compact_state, mps, wires, work_wires):
        """Test that we can get the resources of ResourceMPSPrep with compact states"""
        # Add QubitUnitary to gateset:
        gs = {"QubitUnitary"}

        def circ():
            re.ResourceMPSPrep(compact_state, wires, work_wires)
            return

        def expected_circ():
            re.ResourceMPSPrep(mps, wires, work_wires)
            return

        assert (
            re.get_resources(circ, gate_set=gs)() == re.get_resources(expected_circ, gate_set=gs)()
        )
