# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the SparseMatrix class.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.math import SparseMatrix

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

converters = [pnp.array, jnp.array, tf.Variable, torch.tensor]

INIT = [([[0., 3.], [0., 2.]], {(0, 1): 3., (1, 1): 2.}, (2, 2)),
            ([[0., 1.], [0., 0.], [0., 0.]], {(0, 1): 1.}, (3, 2)),
            ([[1.]], {(0, 0): 1.}, (1, 1)),
            ([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], {}, (3, 3)),
            ]

PROPERTIES = [(np.array([[0., 3.], [0., 2.]]), 2, [0, 1], [1, 1]),
              (np.array([[0.], [1.]]), 1, [1], [0])]

ADDITION = [([[3., 0., 0.], [2., 0., 0.]], [[-3., 0., 0.], [2., 0., 0.]], {(1, 0): 4.}),
            ([[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]], {})
            ]

SUBTRACTION = [([[3., 0., 0.], [2., 0., 0.]], [[3., 0., 0.], [-2., 0., 0.]], {(1, 0): 4.}),
               ([[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]], {})
               ]

MULTIPLICATION = [([[3., 0., 0.], [2., 0., 0.]], -1., {(0, 0): -3., (1, 0): -2.}),
                  ([[0., 0.], [0., 0.]], 2., {})
                  ]

EQUALITY = [([[3., 0., 0.], [2., 0., 0.]], [[3., 0., 0.], [2., 0., 0.]], True),
            ([[3., 0., 0.], [2., 0., 0.]], [[3., 0., 0.], [0., 0., 0.]], False)]


KRON = [([[2., 0.], [0., 0.]], [[0., 2.], [-1., 0.]], {(0, 1): 4., (1, 0): -2.}, (4, 4)),
        ([[2., 0.], [0., 0.]], [[2.]], {(0, 0): 4.}, (2, 2)),
        ]


class TestConstruction:
    """Tests the initialization of a sparse matrix"""

    @pytest.mark.parametrize("converter", converters)
    @pytest.mark.parametrize("matrix, expect_data, expect_shape", INIT)
    def test_init_from_autograd(self, matrix, expect_data, expect_shape, converter):
        """Test initialization with autograd tensor."""
        tensor = converter(matrix)
        s = SparseMatrix(tensor)
        assert s.shape == expect_shape
        assert s.data == {k: tensor[k] for k, v in expect_data.items()}

    def test_init_from_shape(self):
        """Test initialization with a tuple"""
        s = SparseMatrix((4, 5))
        assert s.shape == (4, 5)
        assert s.data == {}

    @pytest.mark.parametrize("arg", [(4, ), np.array([0.])])
    def test_shape_error(self, arg):
        """Test error when shape is not 2-d"""
        with pytest.raises(ValueError, match="Expected a 2-dimensional tensor"):
            SparseMatrix(arg)


class TestProperties:
    """Tests the class properties"""

    @pytest.mark.parametrize("matrix, nnz, row, col", PROPERTIES)
    def test_properties(self, matrix, nnz, row, col):
        s = SparseMatrix(matrix)
        assert s.nnz == nnz
        assert s.row == row
        assert s.col == col

    def test_representation(self):
        """Test that representation works."""
        s = SparseMatrix((2, 3))
        assert s.__repr__() == "<SparseMatrix: entries=0, shape=(2, 3)>"


class TestArithmetic:
    """Test arithmetic of sparse matrices"""

    @pytest.mark.parametrize("converter", converters)
    @pytest.mark.parametrize("matrix1, matrix2, res", ADDITION)
    def test_addition(self, converter, matrix1, matrix2, res):
        """Test addition in all frameworks."""
        tensor1 = converter(matrix1)
        tensor2 = converter(matrix2)
        s1 = SparseMatrix(tensor1)
        s2 = SparseMatrix(tensor2)
        s3 = s1 + s2
        assert s3.data == {k: converter(v) for k, v in res.items()}

    def test_addition_shape_error(self):
        """Test that error raised when trying to add matrices of different shapes"""
        tensor1 = np.array([[1., 0.], [0., 0.]])
        tensor2 = np.array([[1., 0., 0.], [0., 0., 0.]])
        s1 = SparseMatrix(tensor1)
        s2 = SparseMatrix(tensor2)
        with pytest.raises(ValueError, match="Cannot add SparseMatrix object of different shape"):
            s1 + s2

    def test_addition_type_error(self):
        """Test that error raised when trying to add matrices of different types"""
        tensor1 = np.array([[1., 0.], [0., 0.]])
        tensor2 = np.array([[1., 0., 0.], [0., 0., 0.]])
        s1 = SparseMatrix(tensor1)
        with pytest.raises(ValueError, match="Cannot add SparseMatrix and"):
            s1 + tensor2

    @pytest.mark.parametrize("converter", converters)
    @pytest.mark.parametrize("matrix1, matrix2, res", SUBTRACTION)
    def test_subtraction(self, converter, matrix1, matrix2, res):
        """Test subtraction in all frameworks."""
        tensor1 = converter(matrix1)
        tensor2 = converter(matrix2)
        s1 = SparseMatrix(tensor1)
        s2 = SparseMatrix(tensor2)
        s3 = s1 - s2
        assert s3.data == {k: converter(v) for k, v in res.items()}

    @pytest.mark.parametrize("converter", converters)
    @pytest.mark.parametrize("matrix1, scalar, res", MULTIPLICATION)
    def test_multiplication(self, converter, matrix1, scalar, res):
        """Test scalar multiplication in all frameworks."""
        tensor1 = converter(matrix1)
        s1 = SparseMatrix(tensor1)
        s3 = s1 * scalar
        s4 = scalar * s1
        assert s3.data == {k: converter(v) for k, v in res.items()}
        assert s4 == s3

    @pytest.mark.parametrize("converter", converters)
    @pytest.mark.parametrize("arg1, arg2, res", EQUALITY)
    def test_equality(self, converter, arg1, arg2, res):
        """Test equality in all frameworks."""
        tensor1 = converter(arg1)
        tensor2 = converter(arg2)
        s1 = SparseMatrix(tensor1)
        s2 = SparseMatrix(tensor2)
        eq = s1 == s2
        assert eq == res

    def test_equality_tuple(self):
        """Test equality when initializing from tuple."""
        s1 = SparseMatrix((2, 3))
        s2 = SparseMatrix((2, 3))
        assert (s1 == s2)

        s1 = SparseMatrix((3, 2))
        s2 = SparseMatrix((2, 3))
        assert not (s1 == s2)

    @pytest.mark.parametrize("converter", converters)
    @pytest.mark.parametrize("matrix1, matrix2, res, shape", KRON)
    def test_kron(self, converter, matrix1, matrix2, res, shape):
        """Test kronecker product in all frameworks."""
        tensor1 = converter(matrix1)
        tensor2 = converter(matrix2)
        s1 = SparseMatrix(tensor1)
        s2 = SparseMatrix(tensor2)
        s3 = s1.kron(s2)
        assert s3.data == {k: converter(v) for k, v in res.items()}
        assert s3.shape == shape

    def test_kron_error(self):
        """Test that error raised when trying to compute the kronecker product of matrices of different types"""
        tensor1 = np.array([[1., 0.], [0., 0.]])
        tensor2 = np.array([[1., 0., 0.], [0., 0., 0.]])
        s1 = SparseMatrix(tensor1)
        with pytest.raises(ValueError, match="an only compute the kronecker product with another SparseMatrix"):
            s1.kron(tensor2)
