# Copyright 2024 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for arithmetic of Pauli words and sentences with a dense representation."""

import pytest
import numpy as np
from pennylane.labs.lie import ps_to_tensor, product, commutator
from pennylane.pauli import PauliWord, PauliSentence

def unit_tensor(n, idx):
    """Create a unit vector for tensors on n qubits, with idx marking the non-zero entry."""
    assert len(idx) == n
    idx = (4 ** np.arange(n)) @ np.array(idx)[::-1]
    vec = np.eye(4**n)[idx]
    return vec.reshape((4,) * n)

ps_to_tensor_test_cases = [
    (
        PauliSentence({PauliWord({0: "X", 1: "Y"}): 0.3}),
        2,
        unit_tensor(2, (1, 2)) * 0.3,
    ),
    (
        PauliSentence({PauliWord({0: "X", 2: "Y"}): -2.3}),
        4,
        unit_tensor(4, (1, 0, 2, 0)) * -2.3,
    ),
    (
        PauliSentence({PauliWord({0: "X"}): 2.4, PauliWord({1: "Y"}): 0.3}),
        2,
        unit_tensor(2, (1, 0)) * 2.4 + unit_tensor(2, (0, 2)) * 0.3,
    ),
    (
        PauliSentence({PauliWord({2: "X"}): 2.4, PauliWord({0: "Y", 2: "Z"}): 0.3}),
        4,
        unit_tensor(4, (0, 0, 1, 0)) * 2.4 + unit_tensor(4, (2, 0, 3, 0)) * 0.3,
    ),
    (
        PauliSentence({PauliWord({}): -2.3}),
        3,
        unit_tensor(3, (0, 0, 0)) * -2.3,
    ),
]

class TestPsToTensor:
    """Tests for the conversion function ``ps_to_tensor``."""

    @pytest.mark.parametrize("ps, n, expected", ps_to_tensor_test_cases)
    def test_standard_usage(self, ps, n, expected):
        """Test standard application of ps_to_tensor."""
        out = ps_to_tensor(ps, n)
        assert out.shape == expected.shape
        assert np.allclose(out, expected)

    @pytest.mark.parametrize("dtype", [np.int16, np.float16, np.float32, np.float64])
    def test_dtype(self, dtype):
        """Test that a custom dtype is respected."""
        ps = PauliSentence({PauliWord({0: "X"}): 2.4, PauliWord({1: "Y"}): 0.3})
        out = ps_to_tensor(ps, 4, dtype=dtype)
        assert out.dtype == dtype



class TestPauliProduct:
    """Tests for product function."""

    def test_distinct_wires(self):
        """Test product of Paulis that do not share wires."""
        tensor1 = unit_tensor(4, (1, 2, 0, 0)) * 0.5 - unit_tensor(4, (3, 0, 0, 0)) * 1.2
        tensor2 = unit_tensor(4, (0, 0, 3, 2)) * 0.8

        out = product(tensor1, tensor2)
        expected = unit_tensor(4, (1, 2, 3, 2)) * 0.4 - unit_tensor(4, (3, 0, 3, 2)) * 0.96
        assert out.shape == expected.shape == (4,) * 4
        assert np.allclose(out, expected)

    def test_same_wires(self):
        """Test product of fully overlapping Paulis."""
        # Product with real outputs
        # 0.5 * (X Y Z X) - 1.2 (Z X X Y)
        tensor1 = unit_tensor(4, (1, 2, 3, 1)) * 0.5 - unit_tensor(4, (3, 1, 1, 2)) * 1.2
        # 0.8 * (Z Y Z Y)
        tensor2 = unit_tensor(4, (3, 2, 3, 2)) * 0.8

        out = product(tensor1, tensor2)
        expected = unit_tensor(4, (2, 0, 0, 3)) * 0.4 - unit_tensor(4, (0, 3, 2, 0)) * 0.96
        assert out.shape == expected.shape == (4,) * 4
        assert np.allclose(out, expected)

        # Product with complex outputs
        # 0.8 * (Z Z Z Y)
        tensor2 = unit_tensor(4, (3, 3, 3, 2)) * 0.8

        out = product(tensor1, tensor2)
        expected = unit_tensor(4, (2, 1, 0, 3)) * 0.4j + unit_tensor(4, (0, 2, 2, 0)) * 0.96
        assert out.shape == expected.shape == (4,) * 4
        assert np.allclose(out, expected)

    def test_partial_overlap(self):
        """Test product of partially overlapping Paulis."""
        # 0.5 * (I Y Z X) - 1.2 (I I X Y)
        tensor1 = unit_tensor(4, (0, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 0, 1, 2)) * 1.2
        # 0.8 * (Z Y Z I)
        tensor2 = unit_tensor(4, (3, 2, 3, 0)) * 0.8

        out = product(tensor1, tensor2)
        expected = unit_tensor(4, (3, 0, 0, 1)) * 0.4 + unit_tensor(4, (3, 2, 2, 2)) * 0.96j
        assert out.shape == expected.shape == (4,) * 4
        assert np.allclose(out, expected)

    def test_broadcasting_first_factor(self):
        """Test product with broadcasting the first factor."""
        # [0.5 * (I Y Z X) - 1.2 (I I X Y), -1.2 * (X Y Z X) - 1.2 (I Z Y Y)]
        tensor1 = np.stack([
            unit_tensor(4, (0, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 0, 1, 2)) * 1.2,
            unit_tensor(4, (1, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 3, 2, 2)) * 1.2,
        ])
        # 0.8 * (Z Y Z I)
        tensor2 = unit_tensor(4, (3, 2, 3, 0)) * 0.8

        out = product(tensor1, tensor2, broadcasted=[True, False])
        expected = np.stack([
            unit_tensor(4, (3, 0, 0, 1)) * 0.4 + unit_tensor(4, (3, 2, 2, 2)) * 0.96j,
            unit_tensor(4, (2, 0, 0, 1)) * -0.4j - unit_tensor(4, (3, 1, 1, 2)) * 0.96,
        ])
        assert out.shape == expected.shape == (2,) + (4,) * 4
        assert np.allclose(out, expected)

    def test_broadcasting_second_factor(self):
        """Test product with broadcasting the second factor."""
        # 0.8 * (Z Y Z I)
        tensor1 = unit_tensor(4, (3, 2, 3, 0)) * 0.8
        # [0.5 * (I Y Z X) - 1.2 (I I X Y),  -1.2 * (X Y Z X) - 1.2 (I Z Y Y)]
        tensor2 = np.stack([
            unit_tensor(4, (0, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 0, 1, 2)) * 1.2,
            unit_tensor(4, (1, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 3, 2, 2)) * 1.2,
        ])

        out = product(tensor1, tensor2, broadcasted=[False, True])
        expected = np.stack([
            unit_tensor(4, (3, 0, 0, 1)) * 0.4 - unit_tensor(4, (3, 2, 2, 2)) * 0.96j,
            unit_tensor(4, (2, 0, 0, 1)) * 0.4j - unit_tensor(4, (3, 1, 1, 2)) * 0.96,
        ])
        assert out.shape == expected.shape == (2,) + (4,) * 4
        assert np.allclose(out, expected)

    def test_broadcasting_both_factors(self):
        """Test product with broadcasting both factors."""
        tensor1 = np.stack([
            unit_tensor(4, (0, 2, 3, 1)) * 0.5 - unit_tensor(4, (2, 2, 1, 2)) * 1.2,
            unit_tensor(4, (1, 0, 0, 1)) * 0.5 + unit_tensor(4, (0, 1, 1, 2)) * 1.2j,
        ])
        tensor2 = np.stack([
            unit_tensor(4, (0, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 0, 1, 2)) * 1.2,
            unit_tensor(4, (1, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 3, 2, 2)) * 1.2,
        ])

        out = product(tensor1, tensor2, broadcasted=[True, True])
        expected = np.stack([
            np.stack([
                (
                    unit_tensor(4, (0, 0, 0, 0)) * 0.25
                    - unit_tensor(4, (0, 2, 2, 3)) * -0.6
                    - unit_tensor(4, (2, 0, 2, 3)) * -0.6
                    + unit_tensor(4, (2, 2, 0, 0)) * 1.44
                ),
                (
                    unit_tensor(4, (1, 0, 0, 0)) * 0.25
                    - unit_tensor(4, (0, 1, 1, 3)) * 0.6j
                    - unit_tensor(4, (3, 0, 2, 3)) * 0.6j
                    + unit_tensor(4, (2, 1, 3, 0)) * -1.44
                ),
            ]),
            np.stack([
                (
                    unit_tensor(4, (1, 2, 3, 0)) * 0.25
                    - unit_tensor(4, (1, 0, 1, 3)) * 0.6j
                    + unit_tensor(4, (0, 3, 2, 3)) * 0.6
                    - unit_tensor(4, (0, 1, 0, 0)) * 1.44j
                ),
                (
                    unit_tensor(4, (0, 2, 3, 0)) * 0.25
                    - unit_tensor(4, (1, 3, 2, 3)) * 0.6j
                    + unit_tensor(4, (1, 3, 2, 3)) * 0.6
                    - unit_tensor(4, (0, 2, 3, 0)) * 1.44j
                ),
            ]),
        ])
        assert out.shape == expected.shape ==  (2, 2) + (4,) * 4
        assert np.allclose(out, expected)

    def test_broadcasting_both_factors_merged(self):
        """Test product with broadcasting both factors and merging the broadcasting
        axes into one axis."""
        tensor1 = np.stack([
            unit_tensor(4, (0, 2, 3, 1)) * 0.5 - unit_tensor(4, (2, 2, 1, 2)) * 1.2,
            unit_tensor(4, (1, 0, 0, 1)) * 0.5 + unit_tensor(4, (0, 1, 1, 2)) * 1.2j,
        ])
        tensor2 = np.stack([
            unit_tensor(4, (0, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 0, 1, 2)) * 1.2,
            unit_tensor(4, (1, 2, 3, 1)) * 0.5 - unit_tensor(4, (0, 3, 2, 2)) * 1.2,
        ])

        out = product(tensor1, tensor2, broadcasted="merge")
        expected = np.stack([
            (
                unit_tensor(4, (0, 0, 0, 0)) * 0.25
                - unit_tensor(4, (0, 2, 2, 3)) * -0.6
                - unit_tensor(4, (2, 0, 2, 3)) * -0.6
                + unit_tensor(4, (2, 2, 0, 0)) * 1.44
            ),
            (
                unit_tensor(4, (0, 2, 3, 0)) * 0.25
                - unit_tensor(4, (1, 3, 2, 3)) * 0.6j
                + unit_tensor(4, (1, 3, 2, 3)) * 0.6
                - unit_tensor(4, (0, 2, 3, 0)) * 1.44j
            )
        ])
        assert out.shape == expected.shape ==  (2,) + (4,) * 4
        assert np.allclose(out, expected)

class TestCommutator:
    # TODO
    pass
