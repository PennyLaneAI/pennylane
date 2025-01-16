"""Test matrix operations"""

from typing import Dict, Tuple

import numpy as np
import pytest

from pennylane.labs.vibronic import Node, VibronicMatrix, VibronicTerm, VibronicWord, op_norm

# pylint: disable=too-many-arguments

vword0 = VibronicWord([VibronicTerm(tuple(), Node.tensor_node(np.array(0.5)))])
blocks0 = {(0, 0): vword0}
vmat0 = VibronicMatrix(1, 1, blocks0)

vword1 = VibronicWord([VibronicTerm(("P",), Node.tensor_node(np.array([1])))])
blocks1 = {(0, 0): vword1}
vmat1 = VibronicMatrix(1, 1, blocks1)

vword2a = VibronicWord([VibronicTerm(("P", "P"), Node.tensor_node(np.array([[0, 1], [2, 3]])))])
vword2b = VibronicWord([VibronicTerm(("P",), Node.tensor_node(np.array([1, 2])))])
blocks2 = {(0, 0): vword2a, (1, 1): vword2b}
vmat2 = VibronicMatrix(2, 2, blocks2)

blocks3 = {(0, 1): vword2a, (1, 0): vword2b}
vmat3 = VibronicMatrix(2, 2, blocks3)

blocks4 = {(0, 0): vword2a, (0, 1): vword2a, (1, 0): vword2b, (1, 1): vword2b}
vmat4 = VibronicMatrix(2, 2, blocks4)


class TestMatrix:
    """Test properties of the VibronicMatrix class"""

    params = [
        (blocks0, 2, 1, 1, False, 0.5),
        (blocks0, 4, 1, 1, False, 0.5),
        (blocks0, 2, 1, 1, True, 0.5),
        (blocks0, 4, 1, 1, True, 0.5),
        (blocks1, 2, 1, 1, False, op_norm(2)),
        (blocks1, 4, 1, 1, False, op_norm(4)),
        (blocks1, 2, 1, 1, True, op_norm(2)),
        (blocks1, 4, 1, 1, True, op_norm(4)),
        (blocks2, 2, 2, 2, False, 6 * op_norm(2) ** 2),
        (blocks2, 4, 2, 2, False, 6 * op_norm(4) ** 2),
        (blocks2, 2, 2, 2, True, 6 * op_norm(2) ** 2),
        (blocks2, 4, 2, 2, True, 6 * op_norm(4) ** 2),
        (blocks3, 2, 2, 2, False, np.sqrt(18 * op_norm(2) ** 3)),
        (blocks3, 4, 2, 2, False, np.sqrt(18 * op_norm(4) ** 3)),
        (blocks3, 2, 2, 2, True, np.sqrt(18 * op_norm(2) ** 3)),
        (blocks3, 4, 2, 2, True, np.sqrt(18 * op_norm(4) ** 3)),
        (blocks4, 2, 2, 2, False, 6 * op_norm(2) ** 2 + np.sqrt(18 * op_norm(2) ** 3)),
        (blocks4, 4, 2, 2, False, 6 * op_norm(4) ** 2 + np.sqrt(18 * op_norm(4) ** 3)),
        (blocks4, 2, 2, 2, True, 6 * op_norm(2) ** 2 + np.sqrt(18 * op_norm(2) ** 3)),
        (blocks4, 4, 2, 2, True, 6 * op_norm(4) ** 2 + np.sqrt(18 * op_norm(4) ** 3)),
    ]

    @pytest.mark.parametrize("blocks, gridpoints, states, modes, sparse, expected", params)
    def test_norm(
        self,
        blocks: Dict[Tuple[int], VibronicWord],
        gridpoints: int,
        states: int,
        modes: int,
        sparse: bool,
        expected: float,
    ):
        """Test that the norm is correct"""

        vmatrix = VibronicMatrix(states, modes, blocks, sparse=sparse)

        assert np.isclose(vmatrix.norm(gridpoints), expected)

    params = [
        (blocks0, 2, 1, 1, False),
        (blocks0, 4, 1, 1, False),
        (blocks0, 2, 1, 1, True),
        (blocks0, 4, 1, 1, True),
        (blocks1, 2, 1, 1, False),
        (blocks1, 4, 1, 1, False),
        (blocks1, 2, 1, 1, True),
        (blocks1, 4, 1, 1, True),
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
        blocks: Dict[Tuple[int], VibronicWord],
        gridpoints: int,
        states: int,
        modes: int,
        sparse: bool,
    ):
        """Test that .norm is an upper bound on the true norm"""
        vmatrix = VibronicMatrix(states, modes, blocks, sparse=sparse)
        upper_bound = vmatrix.norm(gridpoints)
        norm = np.max(np.linalg.eigvals(vmatrix.matrix(gridpoints)))
        assert norm <= upper_bound
