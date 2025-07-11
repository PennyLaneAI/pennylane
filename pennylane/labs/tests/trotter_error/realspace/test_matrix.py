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
"""Test matrix operations"""


import numpy as np
import pytest

from pennylane.labs.trotter_error.realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
)
from pennylane.labs.trotter_error.realspace.matrix import _op_norm

# pylint: disable=too-many-arguments,too-many-positional-arguments,no-self-use


def _coeffs(states: int, modes: int, order: int):
    """Produce random coefficients for input"""

    phis = []
    symmetric_phis = []
    for i in range(order + 1):
        shape = (states, states) + (modes,) * i
        phi = np.random.random(size=shape)
        phis.append(phi)
        symmetric_phis.append(np.zeros(shape))

    for phi in phis:
        for i in range(states):
            for j in range(states):
                phi[i, j] = (phi[i, j] + phi[i, j].T) / 2

    for phi, symmetric_phi in zip(phis, symmetric_phis):
        for i in range(states):
            for j in range(states):
                symmetric_phi[i, j] = (phi[i, j] + phi[j, i]) / 2

    return np.random.random(size=modes), symmetric_phis


vword0 = RealspaceSum(2, [RealspaceOperator(2, tuple(), RealspaceCoeffs(np.array(0.5)))])
blocks0 = {(0, 0): vword0}
vmat0 = RealspaceMatrix(1, 2, blocks0)

vword1 = RealspaceSum(
    1, [RealspaceOperator(1, ("P",), RealspaceCoeffs(np.array([1]), label="omega"))]
)
blocks1 = {(0, 0): vword1}
vmat1 = RealspaceMatrix(1, 1, blocks1)

vword2a = RealspaceSum(
    2,
    [RealspaceOperator(2, ("P", "P"), RealspaceCoeffs(np.array([[0, 1], [2, 3]]), label="omega"))],
)
vword2b = RealspaceSum(
    2, [RealspaceOperator(2, ("P",), RealspaceCoeffs(np.array([1, 2]), label="omega"))]
)
blocks2 = {(0, 0): vword2a, (1, 1): vword2b}
vmat2 = RealspaceMatrix(2, 2, blocks2)

blocks3 = {(0, 1): vword2a, (1, 0): vword2b}
vmat3 = RealspaceMatrix(2, 2, blocks3)

blocks4 = {(0, 0): vword2a, (0, 1): vword2a, (1, 0): vword2b, (1, 1): vword2b}
vmat4 = RealspaceMatrix(2, 2, blocks4)


class TestMatrix:
    """Test properties of the RealspaceMatrix class"""

    params = [
        (blocks2, 2, 2, 2, False, 6 * _op_norm(2) ** 2),
        (blocks2, 4, 2, 2, False, 6 * _op_norm(4) ** 2),
        (blocks2, 2, 2, 2, True, 6 * _op_norm(2) ** 2),
        (blocks2, 4, 2, 2, True, 6 * _op_norm(4) ** 2),
        (blocks3, 2, 2, 2, False, np.sqrt(18 * _op_norm(2) ** 3)),
        (blocks3, 4, 2, 2, False, np.sqrt(18 * _op_norm(4) ** 3)),
        (blocks3, 2, 2, 2, True, np.sqrt(18 * _op_norm(2) ** 3)),
        (blocks3, 4, 2, 2, True, np.sqrt(18 * _op_norm(4) ** 3)),
        (blocks4, 2, 2, 2, False, 6 * _op_norm(2) ** 2 + np.sqrt(18 * _op_norm(2) ** 3)),
        (blocks4, 4, 2, 2, False, 6 * _op_norm(4) ** 2 + np.sqrt(18 * _op_norm(4) ** 3)),
        (blocks4, 2, 2, 2, True, 6 * _op_norm(2) ** 2 + np.sqrt(18 * _op_norm(2) ** 3)),
        (blocks4, 4, 2, 2, True, 6 * _op_norm(4) ** 2 + np.sqrt(18 * _op_norm(4) ** 3)),
    ]

    @pytest.mark.parametrize("blocks, gridpoints, states, modes, sparse, expected", params)
    def test_norm(
        self,
        blocks: dict[tuple[int], RealspaceSum],
        gridpoints: int,
        states: int,
        modes: int,
        sparse: bool,
        expected: float,
    ):
        """Test that the norm is correct"""

        vmatrix = RealspaceMatrix(states, modes, blocks)

        params = {"gridpoints": gridpoints, "sparse": sparse}

        assert np.isclose(vmatrix.norm(params), expected)

    params = [
        (blocks2, 2, 2, 2, False),
        (blocks2, 4, 2, 2, False),
        (blocks2, 2, 2, 2, True),
        (blocks2, 4, 2, 2, True),
        (blocks3, 2, 2, 2, False),
        (blocks3, 4, 2, 2, False),
        (blocks3, 2, 2, 2, True),
        (blocks3, 4, 2, 2, True),
        (blocks4, 2, 2, 2, False),
        (blocks4, 4, 2, 2, False),
        (blocks4, 2, 2, 2, True),
        (blocks4, 4, 2, 2, True),
    ]

    @pytest.mark.parametrize("blocks, gridpoints, states, modes, sparse", params)
    def test_norm_against_numpy(
        self,
        blocks: dict[tuple[int], RealspaceSum],
        gridpoints: int,
        states: int,
        modes: int,
        sparse: bool,
    ):
        """Test that .norm is an upper bound on the true norm"""

        params = {"gridpoints": gridpoints, "sparse": sparse}

        vmatrix = RealspaceMatrix(states, modes, blocks)
        upper_bound = vmatrix.norm(params)
        norm = np.abs(np.max(np.linalg.eigvals(vmatrix.matrix(gridpoints))))

        assert np.isclose(norm, upper_bound) or norm < upper_bound
