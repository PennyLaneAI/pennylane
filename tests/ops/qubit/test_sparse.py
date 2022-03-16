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
"""
Unit tests for the SparseHamiltonian observable.
"""
import pytest
import numpy as np
import scipy.sparse.coo
from scipy.sparse import coo_matrix, csr_matrix
import pennylane as qml
from pennylane import DeviceError
from pennylane.wires import Wires


SPARSE_HAMILTONIAN_TEST_DATA = [(np.array([[1, 0], [-1.5, 0]])), (np.eye(4))]

H_row = np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11, 12, 12, 13, 14, 15])
H_col = np.array([0, 1, 2, 3, 12, 4, 5, 6, 9, 7, 8, 6, 9, 10, 11, 3, 12, 13, 14, 15])
H_data = np.array(
    [
        0.72004228 + 0.0j,
        0.24819411 + 0.0j,
        0.24819411 + 0.0j,
        0.47493347 + 0.0j,
        0.18092703 + 0.0j,
        -0.5363422 + 0.0j,
        -0.52452263 + 0.0j,
        -0.34359561 + 0.0j,
        -0.18092703 + 0.0j,
        0.3668115 + 0.0j,
        -0.5363422 + 0.0j,
        -0.18092703 + 0.0j,
        -0.34359561 + 0.0j,
        -0.52452263 + 0.0j,
        0.3668115 + 0.0j,
        0.18092703 + 0.0j,
        -1.11700225 + 0.0j,
        -0.44058791 + 0.0j,
        -0.44058791 + 0.0j,
        0.93441396 + 0.0j,
    ]
)
H_hydrogen = coo_matrix((H_data, (H_row, H_col)), shape=(16, 16)).toarray()


class TestSparse:
    """Tests for sparse hamiltonian observable"""

    def test_label(self):
        """Test label method returns 𝓗"""
        H = qml.SparseHamiltonian(coo_matrix(np.array([[1, 0], [-1.5, 0]])), 1)
        assert H.label() == "𝓗"

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSE_HAMILTONIAN_TEST_DATA)
    def test_sparse_typeerror(self, sparse_hamiltonian):
        """Test that the SparseHamiltonian class raises a TypeError on incorrect inputs."""
        mat = np.array([[1, 0], [-1.5, 0]])
        sparse_hamiltonian = csr_matrix(mat)

        with pytest.raises(TypeError, match="Observable must be a scipy sparse coo_matrix"):
            qml.SparseHamiltonian(sparse_hamiltonian)

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSE_HAMILTONIAN_TEST_DATA)
    def test_sparse_matrix(self, sparse_hamiltonian, tol):
        """Test that the matrix property of the SparseHamiltonian class returns the correct matrix."""
        num_wires = len(sparse_hamiltonian[0])
        sparse_hamiltonian_coo = coo_matrix(sparse_hamiltonian)
        res_dynamic = qml.SparseHamiltonian(
            sparse_hamiltonian_coo, range(num_wires)
        ).sparse_matrix()
        res_static = qml.SparseHamiltonian.compute_sparse_matrix(sparse_hamiltonian_coo)
        assert isinstance(res_dynamic, coo_matrix)
        assert isinstance(res_static, coo_matrix)
        assert np.allclose(res_dynamic.toarray(), sparse_hamiltonian, atol=tol, rtol=0)
        assert np.allclose(res_static.toarray(), sparse_hamiltonian, atol=tol, rtol=0)

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSE_HAMILTONIAN_TEST_DATA)
    def test_matrix(self, sparse_hamiltonian, tol):
        """Test that the matrix property of the SparseHamiltonian class returns the correct matrix."""
        num_wires = len(sparse_hamiltonian[0])
        sparse_hamiltonian_coo = coo_matrix(sparse_hamiltonian)
        res_dynamic = qml.SparseHamiltonian(sparse_hamiltonian_coo, range(num_wires)).get_matrix()
        res_static = qml.SparseHamiltonian.compute_matrix(sparse_hamiltonian_coo)
        assert isinstance(res_dynamic, np.ndarray)
        assert isinstance(res_static, np.ndarray)
        assert np.allclose(res_dynamic, sparse_hamiltonian, atol=tol, rtol=0)
        assert np.allclose(res_static, sparse_hamiltonian, atol=tol, rtol=0)

    def test_sparse_diffmethod_error(self):
        """Test that an error is raised when the observable is SparseHamiltonian and the
        differentiation method is not parameter-shift."""
        dev = qml.device("default.qubit", wires=2, shots=None)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.RX(param, wires=0)
            return qml.expval(qml.SparseHamiltonian(coo_matrix(np.eye(4)), [0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="SparseHamiltonian observable must be"
            " used with the parameter-shift differentiation method",
        ):
            qml.grad(circuit, argnum=0)([0.5])

    @pytest.mark.parametrize("qubits, hamiltonian, expected_output", [(4, H_hydrogen, -0.18092703)])
    def test_sparse_gradient(self, qubits, hamiltonian, expected_output, tol):
        """Tests that gradients are computed correctly for a SparseHamiltonian observable."""
        dev = qml.device("default.qubit", wires=qubits, shots=None)

        hamiltonian = coo_matrix(hamiltonian)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(param):
            qml.PauliX(0)
            qml.PauliX(1)
            qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
            return qml.expval(qml.SparseHamiltonian(hamiltonian, wires=range(qubits)))

        assert np.allclose(qml.grad(circuit, argnum=0)(0.0), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "qubits, operations, hamiltonian, expected_output",
        [
            (1, [], np.array([[1.0, 0.0], [0.0, 1.0]]), 1.0),
            (
                2,
                [qml.PauliX(0), qml.PauliY(1)],
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                1.0,
            ),
            (
                4,
                [
                    qml.PauliX(0),
                    qml.PauliX(1),
                    qml.DoubleExcitation(0.22350048065138242, wires=[0, 1, 2, 3]),
                ],
                H_hydrogen,
                -1.1373060481,
            ),
        ],
    )
    def test_sparse_hamiltonian_expval(self, qubits, operations, hamiltonian, expected_output, tol):
        """Test that expectation values of sparse hamiltonians are properly calculated."""

        hamiltonian = coo_matrix(hamiltonian)

        dev = qml.device("default.qubit", wires=qubits, shots=None)
        dev.apply(operations)
        expval = dev.expval(qml.SparseHamiltonian(hamiltonian, range(qubits)))[0]

        assert np.allclose(expval, expected_output, atol=tol, rtol=0)

    def test_sparse_expval_error(self):
        """Test that a DeviceError is raised when the observable is SparseHamiltonian and finite
        shots is requested."""
        hamiltonian = coo_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

        dev = qml.device("default.qubit", wires=1, shots=1)

        with pytest.raises(AssertionError, match="SparseHamiltonian must be used with shots=None"):
            dev.expval(qml.SparseHamiltonian(hamiltonian, [0]))[0]
