# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.vqe` submodule.
"""
import pennylane as qml
import numpy as np

H_ONE_QUBIT = np.array([[1., 0.5j],
                        [-0.5j, 2.5]])
H_TWO_QUBITS = np.array([[0.5, 1.0j, 0.0, -3j],
                         [-1.0j, -1.1, 0.0, -0.1],
                         [0.0, 0.0, -0.9, 12.0],
                         [3j, -0.1, 12.0, 0.0]])
H_NONHERMITIAN = np.array([1.0, 0.5j],
                           [0.5j, -1.3])

class TestHamiltonian:
    """Test the Hamiltonian class"""

    @pytest.mark.parametrize("coeffs,ops", [
        ((), (qml.PauliZ(0),)),
        ((), (qml.PauliZ(0), qml.PauliY(1))),
        ((3.5,), ()),
        ((1.2, -0.4), ()),
        ((0.5, 1.2), (qml.PauliZ(0),)),
        (1.0,), (qml.PauliZ(0), qml.PauliY(0))
    ])
    def test_hamiltonian_invalid_init_exception(self, coeffs, ops):
        """Tests that an exception is raised when giving an invalid combination of coefficients and ops"""
        with pytest.raises(ValueError, match="could not create a valid Hamiltonian"):
            H = qml.vqe.Hamiltonian(coeffs, ops)


    @pytest.mark.parametrize("coeffs,ops", [
        ((1.0,), (qml.Hermitian(H_NONHERMITIAN, 0),)),
        ((1j,), (qml.Hermitian(H_ONE_QUBIT, 0),)),
        ((0.5j, -1.2), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 1)))
    ])
    def test_nonhermitian_hamiltonian_exception(self, coeffs, ops):
        """Tests that an exception is raised when attempting to create a non-hermitian Hamiltonian"""
        with pytest.raises(ValueError, match="Hamiltonian must be Hermitian"):
            H = qml.vqe.Hamiltonian(coeffs, ops)


    @pytest.mark.parametrize("coeffs,ops", [
        ((1.0,), (qml.Hermitian(H_TWO_QUBITS, [0, 1]),)),
        ((-0.8,), (qml.PauliZ(0),)),
        ((0.5, -1.6), (qml.PauliX(0), qml.PauliY(1))),
        ((0.5, -1.6), (qml.PauliX(1), qml.PauliY(1))),
        ((1.1, -0.4, 0.333), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 2), qml.PauliZ(2))),
        ((-0.4, -0.4, 0.333), (qml.Hermitian(H_TWO_QUBITS, [0, 2]), qml.PauliZ(1))),
    ])
    def test_hamiltonian_valid_init(self, coeffs, ops):
        """Tests that the Hamiltonian object is created with the correct attributes"""
        H = qml.vqe.Hamiltonian(coeffs, ops)

        assert H.terms == (coeffs, ops)

