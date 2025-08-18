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
import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError

SPARSEHAMILTONIAN_TEST_MATRIX = np.array(
    [
        [
            0 + 0j,
            0 - 0j,
            0 + 0j,
            -0 + 0j,
            0 + 0j,
            -0 + 0j,
            -0 + 0j,
            0 + 1.0j,
        ],
        [
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            -0 - 1.0j,
            -0 + 0j,
        ],
        [
            0 + 0j,
            0 - 0j,
            0 + 0j,
            0 - 0j,
            0 + 0j,
            -0 - 1.0j,
            0 + 0j,
            -0 + 0j,
        ],
        [
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 1.0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
        ],
        [
            0 + 0j,
            0 - 0j,
            0 + 0j,
            -0 - 1.0j,
            0 + 0j,
            0 - 0j,
            0 + 0j,
            -0 + 0j,
        ],
        [
            0 + 0j,
            0 + 0j,
            0 + 1.0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
        ],
        [
            -0 + 0j,
            0 + 1.0j,
            0 + 0j,
            0 - 0j,
            0 + 0j,
            0 - 0j,
            0 + 0j,
            0 - 0j,
        ],
        [
            -0 - 1.0j,
            -0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
            0 + 0j,
        ],
    ]
)

SPARSE_HAMILTONIAN_TEST_DATA = [(np.array([[1, 0], [-1.5, 0]])), (np.eye(4))]
SPARSE_MATRIX_FORMATS = [
    ("coo", coo_matrix),
    ("csr", csr_matrix),
    ("lil", lil_matrix),
    ("csc", csc_matrix),
]

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
H_hydrogen = csr_matrix((H_data, (H_row, H_col)), shape=(16, 16)).toarray()


class TestSparse:
    """Tests for sparse hamiltonian observable"""

    def test_label(self):
        """Test label method returns 𝓗"""
        H = qml.SparseHamiltonian(csr_matrix(np.array([[1, 0], [-1.5, 0]])), 1)
        assert H.label() == "𝓗"

    def test_valueeror_shape(self):
        """Test that iniatilization raises a ValueError if the shape is not correct for the number of wires."""
        mat = csr_matrix([[1, 0], [0, 1]])
        with pytest.raises(
            ValueError, match=r"Sparse Matrix must be of shape \(4, 4\). Got \(2, 2\)."
        ):
            qml.SparseHamiltonian(mat, wires=(0, 1))

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSE_HAMILTONIAN_TEST_DATA)
    def test_sparse_typeerror(self, sparse_hamiltonian):
        """Test that the SparseHamiltonian class raises a TypeError on incorrect inputs."""
        mat = np.array([[1, 0], [-1.5, 0]])
        sparse_hamiltonian = coo_matrix(mat)

        with pytest.raises(TypeError, match="Observable must be a scipy sparse csr_matrix"):
            qml.SparseHamiltonian(sparse_hamiltonian, wires=0)

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSE_HAMILTONIAN_TEST_DATA)
    def test_sparse_matrix(self, sparse_hamiltonian, tol):
        """Test that the matrix property of the SparseHamiltonian class returns the correct matrix."""
        num_wires = int(np.log2(len(sparse_hamiltonian[0])))
        sparse_hamiltonian_csr = csr_matrix(sparse_hamiltonian)
        res_dynamic = qml.SparseHamiltonian(
            sparse_hamiltonian_csr, range(num_wires)
        ).sparse_matrix()
        res_static = qml.SparseHamiltonian.compute_sparse_matrix(sparse_hamiltonian_csr)
        assert isinstance(res_dynamic, csr_matrix)
        assert isinstance(res_static, csr_matrix)
        assert np.allclose(res_dynamic.toarray(), sparse_hamiltonian, atol=tol, rtol=0)
        assert np.allclose(res_static.toarray(), sparse_hamiltonian, atol=tol, rtol=0)

    @pytest.mark.parametrize("sparse_matrix_format", SPARSE_MATRIX_FORMATS)
    @pytest.mark.parametrize("sparse_hamiltonian", SPARSE_HAMILTONIAN_TEST_DATA)
    def test_sparse_matrix_format(self, sparse_matrix_format, sparse_hamiltonian, tol):
        """Test that the sparse matrix accepts the format parameter."""
        format, expected_type = sparse_matrix_format

        num_wires = int(np.log2(len(sparse_hamiltonian[0])))
        sparse_hamiltonian_csr = csr_matrix(sparse_hamiltonian)
        res_dynamic = qml.SparseHamiltonian(sparse_hamiltonian_csr, range(num_wires)).sparse_matrix(
            format=format
        )
        assert isinstance(res_dynamic, expected_type)
        assert np.allclose(res_dynamic.toarray(), sparse_hamiltonian, atol=tol, rtol=0)

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSE_HAMILTONIAN_TEST_DATA)
    def test_matrix(self, sparse_hamiltonian, tol):
        """Test that the matrix property of the SparseHamiltonian class returns the correct matrix."""
        num_wires = int(np.log2(len(sparse_hamiltonian[0])))
        sparse_hamiltonian_csr = csr_matrix(sparse_hamiltonian)
        res_dynamic = qml.SparseHamiltonian(sparse_hamiltonian_csr, range(num_wires)).matrix()
        res_static = qml.SparseHamiltonian.compute_matrix(sparse_hamiltonian_csr)
        assert isinstance(res_dynamic, np.ndarray)
        assert isinstance(res_static, np.ndarray)
        assert np.allclose(res_dynamic, sparse_hamiltonian, atol=tol, rtol=0)
        assert np.allclose(res_static, sparse_hamiltonian, atol=tol, rtol=0)

    def test_sparse_diffmethod_error(self):
        """Test that an error is raised when the observable is SparseHamiltonian and the
        differentiation method is not parameter-shift."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.RX(param, wires=0)
            return qml.expval(qml.SparseHamiltonian(csr_matrix(np.eye(4)), [0, 1]))

        with pytest.raises(
            QuantumFunctionError,
            match="does not support backprop with requested circuit.",
        ):
            qml.grad(circuit, argnum=0)([0.5])

    @pytest.mark.parametrize("qubits, hamiltonian, expected_output", [(4, H_hydrogen, -0.18092703)])
    def test_sparse_gradient(self, qubits, hamiltonian, expected_output, tol):
        """Tests that gradients are computed correctly for a SparseHamiltonian observable."""
        dev = qml.device("default.qubit", wires=qubits)

        hamiltonian = csr_matrix(hamiltonian)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(param):
            qml.PauliX(0)
            qml.PauliX(1)
            qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
            return qml.expval(qml.SparseHamiltonian(hamiltonian, wires=range(qubits)))

        assert np.allclose(qml.grad(circuit, argnum=0)(0.0), expected_output, atol=tol, rtol=0)

    def test_sparse_no_all_wires_error(self, tol):
        """Tests that SparseHamiltonian can be used as expected when the operator wires don't cover
        all device wires."""
        dev = qml.device("default.qubit", wires=6)

        hamiltonian = qml.SparseHamiltonian(csr_matrix(H_hydrogen), wires=range(4))

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.PauliX(0)
            qml.PauliX(1)
            qml.PauliZ(1)
            return qml.expval(hamiltonian)

        expected_state = np.zeros((64, 1))
        expected_state[48, 0] = -1

        expected = (
            expected_state.conj().T @ qml.matrix(hamiltonian, wire_order=range(6)) @ expected_state
        )
        assert np.allclose(circuit(), expected, atol=tol, rtol=0)

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
            (
                4,
                [
                    qml.PauliX(0),
                    qml.PauliX(1),
                    qml.DoubleExcitation(
                        [0.22350048065138242, 0.22350048065138242], wires=[0, 1, 2, 3]
                    ),
                ],
                H_hydrogen,
                [-1.1373060481, -1.1373060481],
            ),
        ],
    )
    def test_sparse_hamiltonian_expval(self, qubits, operations, hamiltonian, expected_output, tol):
        """Test that expectation values of sparse hamiltonians are properly calculated."""
        # pylint: disable=too-many-arguments

        hamiltonian = csr_matrix(hamiltonian)

        dev = qml.device("default.qubit", wires=qubits)
        qs = qml.tape.QuantumScript(
            operations, [qml.expval(qml.SparseHamiltonian(hamiltonian, range(qubits)))]
        )
        expval = dev.execute(qs)

        assert np.allclose(expval, expected_output, atol=tol, rtol=0)

    def test_sparse_expval_error(self):
        """Test that a DeviceError is raised when the observable is SparseHamiltonian and finite
        shots is requested."""
        hamiltonian = csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))

        dev = qml.device("default.qubit", wires=1)
        qs = qml.tape.QuantumScript(
            measurements=[qml.expval(qml.SparseHamiltonian(hamiltonian, [0]))], shots=1
        )

        with pytest.raises(qml.operation.DiagGatesUndefinedError):
            dev.execute(qs)

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSEHAMILTONIAN_TEST_MATRIX)
    def test_scalar_multipication(self, sparse_hamiltonian):
        """Tests if the __mul__ method of SparseHamiltonian gives the correct result."""

        value = 0.1234
        num_wires = int(np.log2(len(sparse_hamiltonian)))
        H = csr_matrix(SPARSEHAMILTONIAN_TEST_MATRIX)
        H_multiplied = csr_matrix(SPARSEHAMILTONIAN_TEST_MATRIX * value)

        H_sparse_mul_method = qml.SparseHamiltonian(H, wires=range(num_wires)) * value
        H_sparse_mul_method = H_sparse_mul_method.sparse_matrix()
        H_sparse_multiplied_before = qml.SparseHamiltonian(
            H_multiplied, wires=range(num_wires)
        ).sparse_matrix()

        assert np.allclose(H_sparse_mul_method.toarray(), H_sparse_multiplied_before.toarray())

    @pytest.mark.parametrize("sparse_hamiltonian", SPARSEHAMILTONIAN_TEST_MATRIX)
    def test_scalar_multipication_typeerror(self, sparse_hamiltonian):
        """Tests if the __mul__ method of SparseHamiltonian throws the correct error."""
        value = [0.1234]
        num_wires = int(np.log2(len(sparse_hamiltonian)))
        H = csr_matrix(SPARSEHAMILTONIAN_TEST_MATRIX)

        with pytest.raises(
            TypeError, match="Scalar value must be an int or float. Got <class 'list'>"
        ):
            _ = qml.SparseHamiltonian(H, wires=range(num_wires)) * value
