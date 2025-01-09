"""Test matrix operations"""

import numpy as np
import pytest

from pennylane.labs.vibronic import Node, VibronicMatrix, VibronicTerm, VibronicWord, op_norm

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
        (vmat0, 2, 0.5),
        (vmat0, 4, 0.5),
        (vmat1, 2, op_norm(2)),
        (vmat1, 4, op_norm(4)),
        (vmat2, 2, 6 * op_norm(2) ** 2),
        (vmat2, 4, 6 * op_norm(4) ** 2),
        (vmat3, 2, np.sqrt(18 * op_norm(2) ** 3)),
        (vmat3, 4, np.sqrt(18 * op_norm(4) ** 3)),
        (vmat4, 2, 6 * op_norm(2) ** 2 + np.sqrt(18 * op_norm(2) ** 3)),
        (vmat4, 4, 6 * op_norm(4) ** 2 + np.sqrt(18 * op_norm(4) ** 3)),
    ]

    @pytest.mark.parametrize("vmatrix, gridpoints, expected", params)
    def test_norm(self, vmatrix: VibronicMatrix, gridpoints: int, expected: float):
        """Test that the norm is correct"""

        assert np.isclose(vmatrix.norm(gridpoints), expected)

    params = [
        (vmat0, 2),
        (vmat0, 4),
        (vmat1, 2),
        (vmat1, 4),
        (vmat2, 2),
        (vmat2, 4),
        (vmat3, 2),
        (vmat3, 4),
        (vmat4, 2),
        (vmat4, 4),
    ]

    @pytest.mark.parametrize("vmatrix, gridpoints", params)
    def test_norm_against_numpy(self, vmatrix: VibronicMatrix, gridpoints: int):
        """Test that .norm is an upper bound on the true norm"""
        upper_bound = vmatrix.norm(gridpoints)
        norm = np.max(np.linalg.eigvals(vmatrix.matrix(gridpoints)))
        assert norm <= upper_bound
