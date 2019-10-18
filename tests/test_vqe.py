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
import pytest
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
from pennylane.templates.layers import StronglyEntanglingLayer
from pennylane.init import strong_ent_layer_uniform
import numpy as np

np.random.seed(0)

H_ONE_QUBIT = np.array([[1., 0.5j],
                        [-0.5j, 2.5]])
H_TWO_QUBITS = np.array([[0.5, 1.0j, 0.0, -3j],
                         [-1.0j, -1.1, 0.0, -0.1],
                         [0.0, 0.0, -0.9, 12.0],
                         [3j, -0.1, 12.0, 0.0]])
H_NONHERMITIAN = np.array([[1.0, 0.5j],
                           [0.5j, -1.3]])
OBSERVABLES = [(qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)),
               (qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)),
               (qml.Hermitian(H_TWO_QUBITS, [0,1]),)
    ]
JUNK_INPUTS = [None, [], (,), lambda x: x, 5.0]

def custom_fixed_ansatz(_unused_params):
    qml.RX(0.5, 0)
    qml.RX(-1.2, 1)
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    qml.Hadamard(1)
    qml.CNOT([0, 1])


def custom_var_ansatz(params):
    for p in params:
        qml.RX(p, 0)
    qml.CNOT([0, 1])
    for p in params:
        qml.RX(-p, 1)
    qml.CNOT([0, 1])

def amp_embed(params):
    AmplitudeEmbedding(params, wires=[0,1,2])

def strong_ent_layer(params):
    StronglyEntanglingLayer(params, wires=[0,1,2])

def amp_embed_and_strong_ent_layer(embed_params, layer_params):
    amp_embed(embed_params, wires=[0,1,2])
    strong_ent_layer(layer_params, wires=[0,1,2])

ANSAETZE = [custom_fixed_ansatz, custom_var_ansatz, amp_embed, strong_ent_layer, amp_embed_and_strong_ent_layer]
EMBED_PARAMS = np.cumsum(np.arange(2 ** 3))
EMBED_PARAMS /= np.linalg.norm(EMBED_PARAMS)
LAYER_PARAMS = strong_ent_layer_uniform(n_wires=3)

class TestHamiltonian:
    """Test the Hamiltonian class"""

    @pytest.mark.parametrize("coeffs, ops", [
        ((1.0,), (qml.Hermitian(H_TWO_QUBITS, [0, 1]),)),
        ((-0.8,), (qml.PauliZ(0),)),
        ((0.5, -1.6), (qml.PauliX(0), qml.PauliY(1))),
        ((0.5, -1.6), (qml.PauliX(1), qml.PauliY(1))),
        ((1.1, -0.4, 0.333), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 2), qml.PauliZ(2))),
        ((-0.4, 0.15), (qml.Hermitian(H_TWO_QUBITS, [0, 2]), qml.PauliZ(1))),
        ([1.5, 2.0], [qml.PauliZ(0), qml.PauliY(2)]),
        (np.array([-0.1, 0.5]), [qml.Hermitian(H_TWO_QUBITS, [0,1]), qml.PauliY(0)]),
        ((0.5, 1.2), (qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1)))
    ])
    def test_hamiltonian_valid_init(self, coeffs, ops):
        """Tests that the Hamiltonian object is created with the correct attributes"""
        H = qml.vqe.Hamiltonian(coeffs, ops)
        assert H.terms == (coeffs, ops)

    @pytest.mark.parametrize("coeffs, ops", [
        ((), (qml.PauliZ(0),)),
        ((), (qml.PauliZ(0), qml.PauliY(1))),
        ((3.5,), ()),
        ((1.2, -0.4), ()),
        ((0.5, 1.2), (qml.PauliZ(0),)),
        ((1.0,), (qml.PauliZ(0), qml.PauliY(0))),
    ])
    def test_hamiltonian_invalid_init_exception(self, coeffs, ops):
        """Tests that an exception is raised when giving an invalid combination of coefficients and ops"""
        with pytest.raises(ValueError, match="number of coefficients and operators does not match"):
            H = qml.vqe.Hamiltonian(coeffs, ops)

    @pytest.mark.parametrize("coeffs, ops", [
        ((1.0,), (qml.Hermitian(H_NONHERMITIAN, 0),)),
        ((1j,), (qml.Hermitian(H_ONE_QUBIT, 0),)),
        ((0.5j, -1.2), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 1)))
    ])
    def test_nonhermitian_hamiltonian_exception(self, coeffs, ops):
        """Tests that an exception is raised when attempting to create a non-hermitian Hamiltonian"""
        with pytest.raises(ValueError, match=r"((coefficients are not real-valued)|"
                                             r"(one or more ops are not Hermitian))"):
            H = qml.vqe.Hamiltonian(coeffs, ops)


class TestVQE:
    """Test the core functionality of the VQE module"""
    @pytest.mark.parametrize("ansatz", ANSAETZE)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_vqe_qnodes_valid_init(self, ansatz, observables):
        """Tests that a collection of QNodes is properly created"""
        qnodes = qml.vqe.vqe_qnodes(ansatz, observables)

        assert len(qnodes) == len(observables)
        assert all(isinstance(qml.QNode, q) for q in qnodes)
        assert all(len(q.circuit.observables) == 1 for q in qnodes)
        assert all(q.circuit.observables[0] == observables[idx] for idx, q in enumerate(qnodes))

    @pytest.mark.parametrize("ansatz, params", [
        (amp_embed, EMBED_PARAMS),
        (strong_ent_layer, LAYER_PARAMS),
        (amp_embed_and_strong_ent_layer, (EMBED_PARAMS, LAYER_PARAMS)),
    ])
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_vqe_qnodes_evaluate(self, ansatz, observables, params):
        """Tests that the qnodes returned by vqe_qnodes evaluate properly"""
        qnodes = qml.vqe.vqe_qnodes((ansatz, observables), )
        for idx, q in qnodes:
            val = q(*params)
            assert type(val) == float

    @pytest.mark.parametrize("coeffs, observables, expected", [
        ((1.0,), (qml.PauliZ(0)), 0.0),
        ((1.0,), (qml.PauliX(0), ), 0.5),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), 1.2 * 0.5),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), 1.2 * 0.5),
        ((0.5, 1.2), (qml.PauliX(0), qml.PauliX(0)), 0.5 * 0.5 + 1.2 * 0.5),
        ((0.5, 1.2), (qml.PauliX(0), qml.PauliX(1)), 0.5 * 0.5 + 1.2 * 0.5)
    ])
    def test_vqe_aggregate_expvals(self, coeffs, observables, expected):
        """Tests that the vqe_aggregate function returns correct expectation values"""
        ansatz = lambda x: None
        params = np.zeros(1)
        qnodes = qml.vqe.vqe_qnodes(ansatz, observables)
        for q in qnodes:
            val = q(*params)
            assert val == expected

    @pytest.mark.parametrize("ansatz", ANSAETZE)
    @pytest.mark.parametrize("observables", JUNK_INPUTS)
    def test_vqe_qnodes_no_observables(self, ansatz, observables):
        """Tests that an exception is raised when no observables are supplied to vqe_qnodes"""
        with pytest.raises(ValueError, "no observables were provided"):
            observables = []
            qnodes = qml.vqe.vqe_qnodes(ansatz, observables)

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_vqe_qnodes_no_ansatz(self, ansatz, observables):
        """Tests that an exception is raised when no valid ansatz is supplied to vqe_qnodes"""
        with pytest.raises(ValueError, "no valid ansatz was provided"):
            qnodes = qml.vqe.vqe_qnodes(ansatz, observables)

  def test_vqe_cost_evaluate(self, hamiltonian, ansatz, params):
      """Tests that the vqe_cost function evaluates properly"""
      pass

  def test_vqe_cost_expvals(self, hamiltonian, ansatz, params):
      """Tests that the vqe_cost function gives the correct expectation values"""
      pass

  def test_vqe_cost_invalid_inputs(self, hamiltonian, ansatz, params):
      """Tests that the vqe_cost function raises an exception if the arguments are not compatible"""
      pass