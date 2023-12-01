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
Tests for the functions used in the fable template.
"""
from functools import reduce
import random
from copy import copy, deepcopy
import pytest
import numpy as np
from pennylane.pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform
from pennylane.templates.subroutines.fable import *
from pennylane.templates.state_preparations.mottonen import gray_code, _get_alpha_y
from pennylane import numpy as pnp
import pennylane as qml

def generate_random_matrix(rows, cols):
    """
    Generate a random NumPy matrix with the given number of rows and columns.

    Parameters:
    - rows: Number of rows in the matrix.
    - cols: Number of columns in the matrix.

    Returns:
    - Random NumPy matrix.
    """
    random_matrix = np.random.rand(rows, cols)
    return random_matrix

def generate_random_complex_matrix(rows, cols):
    """
    Generate a random complex-valued matrix with the given number of rows and columns.

    Parameters:
    - rows: Number of rows in the matrix.
    - cols: Number of columns in the matrix.

    Returns:
    - Random complex-valued matrix.
    """
    real_part = np.random.rand(rows, cols)
    imag_part = np.random.rand(rows, cols)

    # Combine the real and imaginary parts to form a complex-valued matrix
    complex_matrix = real_part + 1j * imag_part

    return complex_matrix


params = []
for i in range(2,5):
    params.append(generate_gray_sequence(i))
@pytest.mark.parametrize("sequence", params)
def test_find_changed_bits(sequence):
    """Tests for the function find_changed_bit, that is used to determine the correct CNOT controls
    in the maximaly controlled rotation gates"""
    indices = find_changed_bits(sequence)
    assert len(indices) != 0, 'control index sequence must have non zero length'
    assert all(indices) > 0, 'control indices cannot be zero, or negative'

params = []
for i in range (2,4):
    for j in range(2,4):
        params.append(generate_random_matrix(i,j))

@pytest.mark.parametrize("matrix", params)
def test_process_matrix(matrix):
    """Tests for the process_matrix function that is used for pre-processing the matrix to
    be encoded"""
    processed_matrix  = process_matrix(matrix)
    assert np.shape(processed_matrix)[0] == np.shape(processed_matrix)[1], 'Processed matrix is not square.'
    assert np.linalg.norm(processed_matrix) == 1, 'Processed matrix is not normalized'

@pytest.mark.parametrize("rank,expected_gray_code", [
        (1, ['0', '1']),
        (2, ['00', '01', '11', '10']),
        (3, ['000', '001', '011', '010', '110', '111', '101', '100']),
    ])
    # fmt: on
def test_gray_code(rank, expected_gray_code):
    """Tests that the function gray_code generates the proper
    Gray code of given rank."""
    assert gray_code(rank) == expected_gray_code

def test_gray_permutation():
    """Test for the function gray_permutation that shuffles basis states according to grays code."""
    vector = np.random.randint(10, size=(1, 5))

    result = gray_permutation(vector)

    # Expected result based on the known Gray code sequence
    expected_result = [vector[i] for i in generate_gray_sequence(int(np.log2(len(vector))))]

    # Assert that the result matches the expected result
    assert np.array_equal(result, np.array(expected_result)),'Gray permutations are not equal'

params = []
for i in range(2,5):
    matrix = generate_random_complex_matrix(i,i)
    tol = (random.uniform(0, 1))
    params.append((matrix,tol))

@pytest.mark.parametrize('matrix,tolerance',params)
def test_FABLE(matrix,tolerance):
    """Test the FABLE template for block encoding. Test checks to see if FABLE circuit correctly
    block encodes the matrix to be block encoded up to a given tolerance value, where the absolute
    error is at most N^3 * tol, where N is the shape of the matrix to be encoded"""
    y = deepcopy(matrix)
    y = process_matrix(y)
    wiress = range(0, len(y) + 1)
    y = np.array(y)

    dev = qml.device('default.qubit', wires=wiress)
    @qml.qnode(dev)
    def circuit(A, tol):
        qml.FABLE(A, tol)
        return qml.state()

    M = len(y) * qml.matrix(circuit, wire_order=None)(y, tolerance)[0:len(matrix), 0:len(matrix)]

    assert np.allclose(matrix,M,atol= len(matrix)**3 * tolerance), 'block encoded matrix and given matrix are not equal'

class TestWalshHadamardTransform:
    """Test the helper function walsh_hadamard_transform."""

    @pytest.mark.parametrize(
        "inp, exp",
        [
            ([1, 1, 1, 1], [1, 0, 0, 0]),
            ([1, 1.5, 0.5, 1], [1, -0.25, 0.25, 0]),
            ([1, 0, -1, 2.5], [0.625, -0.625, -0.125, 1.125]),
        ],
    )
    def test_compare_analytic_results(self, inp, exp):
        """Test against hard-coded results."""
        inp = np.array(inp)
        output = _walsh_hadamard_transform(inp)
        assert qml.math.allclose(output, exp)

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("provide_n", [True, False])
    def test_compare_matrix_mult(self, n, provide_n):
        """Test against matrix multiplication for a few random inputs."""
        np.random.seed(382)
        inp = np.random.random(2**n)
        output = _walsh_hadamard_transform(inp, n=n if provide_n else None)
        h = np.array([[0.5, 0.5], [0.5, -0.5]])
        h = reduce(np.kron, [h] * n)
        exp = h @ inp
        assert qml.math.allclose(output, exp)

    def test_compare_analytic_results_broadcasted(self):
        """Test against hard-coded results."""
        inp = np.array([[1, 1, 1, 1], [1, 1.5, 0.5, 1], [1, 0, -1, 2.5]])
        exp = [[1, 0, 0, 0], [1, -0.25, 0.25, 0], [0.625, -0.625, -0.125, 1.125]]
        output = _walsh_hadamard_transform(inp)
        assert qml.math.allclose(output, exp)

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("provide_n", [True, False])
    def test_compare_matrix_mult_broadcasted(self, n, provide_n):
        """Test against matrix multiplication for a few random inputs."""
        np.random.seed(382)
        inp = np.random.random((5, 2**n))
        output = _walsh_hadamard_transform(inp, n=n if provide_n else None)
        h = np.array([[0.5, 0.5], [0.5, -0.5]])
        h = reduce(np.kron, [h] * n)
        exp = qml.math.moveaxis(qml.math.tensordot(h, inp, [[1], [1]]), 0, 1)
        assert qml.math.allclose(output, exp)

