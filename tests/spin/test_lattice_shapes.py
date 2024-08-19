# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for computing the lattices of certain shapes.
"""

import pytest

from pennylane.spin import generate_lattice


@pytest.mark.parametrize(
    # expected_edges here were obtained with netket.
    ("shape", "L", "expected_edges"),
    [
        (
            "chAin ",
            [10, 0, 0],
            [
                (0, 1, 0),
                (1, 2, 0),
                (3, 4, 0),
                (2, 3, 0),
                (6, 7, 0),
                (4, 5, 0),
                (8, 9, 0),
                (5, 6, 0),
                (7, 8, 0),
            ],
        ),
        (
            "Square",
            [3, 3],
            [
                (0, 1, 0),
                (1, 2, 0),
                (3, 4, 0),
                (5, 8, 0),
                (0, 3, 0),
                (1, 4, 0),
                (6, 7, 0),
                (4, 5, 0),
                (3, 6, 0),
                (2, 5, 0),
                (4, 7, 0),
                (7, 8, 0),
            ],
        ),
        (
            " Rectangle ",
            [3, 4],
            [
                (0, 1, 0),
                (9, 10, 0),
                (1, 2, 0),
                (0, 4, 0),
                (10, 11, 0),
                (1, 5, 0),
                (3, 7, 0),
                (2, 3, 0),
                (6, 7, 0),
                (4, 5, 0),
                (8, 9, 0),
                (2, 6, 0),
                (5, 6, 0),
                (4, 8, 0),
                (6, 10, 0),
                (5, 9, 0),
                (7, 11, 0),
            ],
        ),
        (
            "honeycomb",
            [2, 2],
            [
                (0, 1, 0),
                (1, 2, 0),
                (1, 4, 0),
                (2, 3, 0),
                (6, 7, 0),
                (4, 5, 0),
                (5, 6, 0),
                (3, 6, 0),
            ],
        ),
        (
            "TRIANGLE",
            [2, 2],
            [(0, 1, 0), (1, 2, 0), (2, 3, 0), (0, 2, 0), (1, 3, 0)],
        ),
    ],
)
def test_edges_for_shapes(shape, L, expected_edges):
    r"""Test that correct edges are obtained for given lattice shapes"""
    lattice = generate_lattice(lattice=shape, n_cells=L)
    assert sorted(lattice.edges) == sorted(expected_edges)
