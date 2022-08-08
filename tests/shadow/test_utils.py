# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the classical shadows utility functions"""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.shadows import median_of_means, pauli_expval


@pytest.mark.autograd
class TestMedianOfMeans:
    """Test the median of means function"""

    @pytest.mark.parametrize(
        "arr, num_batches, expected",
        [
            (np.array([0.1]), 1, 0.1),
            (np.array([0.1, 0.2]), 1, 0.15),
            (np.array([0.1, 0.2]), 2, 0.15),
            (np.array([0.2, 0.1, 0.4]), 1, 0.7 / 3),
            (np.array([0.2, 0.1, 0.4]), 2, 0.275),
            (np.array([0.2, 0.1, 0.4]), 3, 0.2),
        ],
    )
    def test_output(self, arr, num_batches, expected):
        """Test that the output is correct"""
        actual = median_of_means(arr, num_batches)
        assert actual.shape == ()
        assert np.allclose(actual, expected)


@pytest.mark.autograd
class TestPauliExpval:
    """Test the Pauli expectation value function"""

    @pytest.mark.parametrize("word", [[0, 0, 1], [0, 2, -1], [-1, -1, 1]])
    def test_word_not_present(self, word):
        """Test that the output is 0 if the Pauli word is not present in the recipes"""

        bits = np.array([[0, 0, 0]])
        recipes = np.array([[0, 0, 0]])

        actual = pauli_expval(bits, recipes, np.array(word))
        assert actual.shape == (1,)
        assert actual[0] == 0

    single_bits = np.array([[1, 0, 1]])
    single_recipes = np.array([[0, 1, 2]])

    @pytest.mark.parametrize(
        "word, expected", [([0, 1, 2], 27), ([0, 1, -1], -9), ([-1, -1, 2], -3), ([-1, -1, -1], 1)]
    )
    def test_single_word_present(self, word, expected):
        """Test that the output is correct if the Pauli word appears once in the recipes"""
        actual = pauli_expval(self.single_bits, self.single_recipes, np.array(word))
        assert actual.shape == (1,)
        assert actual[0] == expected

    multi_bits = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])
    multi_recipes = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 0]])

    @pytest.mark.parametrize(
        "word, expected",
        [
            ([0, 1, 2], [27, -27, 0]),
            ([0, 1, -1], [-9, 9, 9]),
            ([-1, -1, 2], [-3, -3, 0]),
            ([-1, -1, -1], [1, 1, 1]),
        ],
    )
    def test_multi_word_present(self, word, expected):
        """Test that the output is correct if the Pauli word appears multiple
        times in the recipes"""
        actual = pauli_expval(self.multi_bits, self.multi_recipes, np.array(word))
        assert actual.shape == (self.multi_bits.shape[0],)
        assert np.all(actual == expected)
