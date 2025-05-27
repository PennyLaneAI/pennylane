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
This submodule tests mathematical decomposition functions, for example of matrices or integers.
"""
import pytest

from pennylane.math.decomposition import decomp_int_to_powers_of_two


@pytest.mark.parametrize(
    "k, n, exp_R",
    [
        (0, 1, [0]),
        (0, 2, [0, 0]),
        (1, 2, [0, 1]),
        (0, 3, [0, 0, 0]),
        (1, 3, [0, 0, 1]),
        (2, 3, [0, 1, 0]),
        (3, 3, [1, 0, -1]),
        (4, 3, [1, 0, 0]),
        (0, 4, [0, 0, 0, 0]),
        (1, 4, [0, 0, 0, 1]),
        (2, 4, [0, 0, 1, 0]),
        (3, 4, [0, 1, 0, -1]),
        (4, 4, [0, 1, 0, 0]),
        (5, 4, [0, 1, 0, 1]),
        (6, 4, [1, 0, -1, 0]),
        (7, 4, [1, 0, 0, -1]),
        (8, 4, [1, 0, 0, 0]),
        (121, 8, [1, 0, 0, 0, -1, 0, 0, 1]),  # 121=01111001_2
        (245, 10, [0, 1, 0, 0, 0, -1, 0, 1, 0, 1]),  # 245=0011110101_2
        (8716, 15, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0, 0]),  # 8716=010001000001100_2
    ],
)
def test_decomp_int_to_powers_of_two(k, n, exp_R):
    """Tests for ``decomp_int_to_powers_of_two``, which is used
    to decompose, e.g., ``PCPhase`` operations."""

    R = decomp_int_to_powers_of_two(k, n)
    assert R == exp_R, f"\n{R}\n{exp_R}"
    assert sum(val != 0 for val in R) == (k ^ (3 * k)).bit_count()
