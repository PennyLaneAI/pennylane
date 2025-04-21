# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for ``optimize_measurements`` function in ``/pauli/grouping/optimize_measurements.py``.
"""
import numpy as np
import pytest

from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane.pauli import are_identical_pauli_words
from pennylane.pauli.grouping.optimize_measurements import optimize_measurements


class TestOptimizeMeasurements:
    """Tests for the ``optimize_measurements`` function."""

    observables_diagonalized = [
        (
            [PauliY(0), PauliX(0) @ PauliX(1), PauliZ(1)],
            [
                [PauliZ(wires=[0]) @ PauliZ(wires=[1])],
                [PauliZ(wires=[0]), PauliZ(wires=[1])],
            ],
        ),
        (
            [
                Identity(0),
                PauliX(1) @ PauliY(2),
                PauliY(3) @ PauliX(1) @ PauliZ(2),
                PauliY(4) @ PauliZ(1),
                PauliZ(2),
            ],
            [
                [Identity(wires=[0]), PauliZ(wires=[1]) @ PauliZ(wires=[2])],
                [
                    PauliZ(wires=[1]) @ PauliZ(wires=[2]) @ PauliZ(wires=[3]),
                    PauliZ(wires=[2]),
                ],
                [PauliZ(wires=[1]) @ PauliZ(wires=[4])],
            ],
        ),
        (
            [
                PauliX("a"),
                PauliX("a") @ PauliY(1),
                PauliZ("b") @ PauliY("c") @ Identity("a"),
                PauliZ("a") @ PauliZ("b"),
            ],
            [
                [
                    PauliZ(wires=["b"]) @ PauliZ(wires=["c"]),
                    PauliZ(wires=["a"]) @ PauliZ(wires=["b"]),
                ],
                [PauliZ(wires=["a"]), PauliZ(wires=["a"]) @ PauliZ(wires=[1])],
            ],
        ),
    ]

    @pytest.mark.parametrize("observables,diagonalized_groupings_sol", observables_diagonalized)
    def test_optimize_measurements_qwc_generic_case(self, observables, diagonalized_groupings_sol):
        """Generic test cases without coefficients."""

        diagonalized_groupings = optimize_measurements(
            observables, grouping="qwc", colouring_method="rlf"
        )[1]

        # assert the correct number of partitions:
        n_partitions = len(diagonalized_groupings_sol)
        assert len(diagonalized_groupings) == n_partitions
        # assert each partition is of the correct length:
        assert all(
            len(diagonalized_groupings[i]) == len(diagonalized_groupings_sol[i])
            for i in range(n_partitions)
        )
        # assert each partition contains the same Pauli terms as the solution partition:
        for i, partition in enumerate(diagonalized_groupings):
            for j, pauli in enumerate(partition):
                assert are_identical_pauli_words(pauli, diagonalized_groupings_sol[i][j])

    def test_optimize_measurements_qwc_generic_case_with_coefficients(self):
        """Tests if coefficients are properly re-structured."""

        observables = [PauliY(0), PauliX(0) @ PauliX(1), PauliZ(1)]
        coefficients = [1.43, 4.21, 0.97]

        grouped_coeffs_sol = [[4.21], [1.43, 0.97]]

        grouped_coeffs = optimize_measurements(
            observables, coefficients, grouping="qwc", colouring_method="rlf"
        )[2]

        assert len(grouped_coeffs) == len(grouped_coeffs_sol)

        assert all(
            np.allclose(grouped_coeffs[i], grouped_coeffs_sol[i])
            for i in range(len(grouped_coeffs_sol))
        )

    @pytest.mark.parametrize(
        "obs",
        [
            [PauliZ(0), PauliZ(0)],
            [PauliX(0) @ PauliX(1), PauliX(0) @ PauliX(1)],
            [PauliX(0) @ PauliX(1), PauliX(1) @ PauliX(0)],
        ],
    )
    def test_optimize_measurements_qwc_term_multiple_times(self, obs):
        """Tests if coefficients are properly re-structured even if the same
        terms appear multiple times.

        Although it should be fair to assume that the terms are unique in the
        Hamiltonian, making sure that grouping happens even with terms appearing
        multiple times can ensure that there is no unexpected behaviour.
        """
        coefficients = [1.43, 4.21]

        grouped_coeffs_sol = [[1.43, 4.21]]

        grouped_coeffs = optimize_measurements(
            obs, coefficients, grouping="qwc", colouring_method="lf"
        )[2]

        assert len(grouped_coeffs) == len(grouped_coeffs)

        assert all(
            np.allclose(grouped_coeffs[i], grouped_coeffs_sol[i])
            for i in range(len(grouped_coeffs_sol))
        )

    def test_optimize_measurements_not_implemented_catch(self):
        """Tests that NotImplementedError is raised for methods other than ``'qwc'``."""

        observables = [PauliY(0), PauliX(0) @ PauliX(1), PauliZ(1)]
        grouping = "anticommuting"

        with pytest.raises(
            NotImplementedError,
            match="Measurement reduction by 'anticommuting' grouping not implemented",
        ):
            optimize_measurements(observables, grouping=grouping)
