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
COEFFS = [(0.5, 1.2, -0.7),
          (2.2, -0.2, 0.0),
          (0.33,)]
OBSERVABLES = [(qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)),
               (qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)),
               (qml.Hermitian(H_TWO_QUBITS, [0, 1]),)]
JUNK_INPUTS = [None, [], tuple(), 5.0, {"junk": -1}]

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
    AmplitudeEmbedding(params, wires=[0, 1, 2])


def strong_ent_layer(params):
    StronglyEntanglingLayer(*params, wires=[0, 1, 2])


def amp_embed_and_strong_ent_layer(embed_params, layer_params):
    amp_embed(embed_params, wires=[0, 1, 2])
    strong_ent_layer(layer_params, wires=[0, 1, 2])


empty_ansatz = lambda x: None

ANSAETZE = [empty_ansatz, custom_fixed_ansatz, custom_var_ansatz,
            amp_embed, strong_ent_layer, amp_embed_and_strong_ent_layer]
empty_params = []
EMBED_PARAMS = np.array([1 / np.sqrt(2 ** 3)] * 2 ** 3)
LAYER_PARAMS = strong_ent_layer_uniform(n_wires=3)

@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    """A mock instance of the abstract Device class"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())

@pytest.fixture(scope="function")
def mock_device_with_eval(monkeypatch):
    """A mock instance of the abstract Device class"""
    with monkeypatch.context() as m:
        m.setattr(Device, '__abstractmethods__', frozenset())
        m.setattr(Device, 'expval', lambda self, x, y, z: 0.0)
        m.setattr(Device, 'apply', lambda self, x, y, z: None)

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
    def test_qnodes_valid_init(self, ansatz, observables):
        """Tests that a collection of QNodes is properly created"""
        qnodes = qml.vqe.qnodes(ansatz, observables)

        assert len(qnodes) == len(observables)
        assert all(isinstance(qml.QNode, q) for q in qnodes)
        assert all(len(q.circuit.observables) == 1 for q in qnodes)
        assert all(q.circuit.observables[0] == observables[idx] for idx, q in enumerate(qnodes))

    @pytest.mark.parametrize("ansatz, params", [
        (amp_embed, EMBED_PARAMS),
        #(strong_ent_layer, LAYER_PARAMS),
        #(amp_embed_and_strong_ent_layer, (EMBED_PARAMS, LAYER_PARAMS)),
    ])
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_qnodes_evaluate(self, ansatz, observables, params):
        """Tests that the qnodes returned by qnodes evaluate properly"""
        dev = mock_device_with_eval
        dev.num_wires = 3
        qnodes = qml.vqe.qnodes(ansatz, observables, device=dev)
        print("qnodes", qnodes)
        print("params", params)
        for idx, q in enumerate(qnodes):
            val = q(params)
            assert type(val) == float

    @pytest.mark.parametrize("coeffs, observables, expected", [
        ((-0.6,), (qml.PauliZ(0),), -0.6 * 1.0),
        ((1.0,), (qml.PauliX(0),), 0.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), 0.5 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), 0.5 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), 0.5 * 1.0 + 1.2 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), 0.5 * 1.0 + 1.2 * 1.0)
    ])
    def test_qnodes_expvals(self, coeffs, observables, expected):
        """Tests that the qnodes function returns correct expectation values"""
        qnodes = qml.vqe.qnodes(empty_ansatz, observables)
        for q in qnodes:
            val = q(*empty_params)
            assert val == expected

    @pytest.mark.parametrize("ansatz", ANSAETZE)
    @pytest.mark.parametrize("observables", JUNK_INPUTS)
    def test_qnodes_no_observables(self, ansatz, observables):
        """Tests that an exception is raised when no observables are supplied to qnodes"""
        with pytest.raises(ValueError, match="observables are not valid"):
            obs = (observables,)
            qnodes = qml.vqe.qnodes(ansatz, obs, device=mock_device)

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_qnodes_no_ansatz(self, ansatz, observables):
        """Tests that an exception is raised when no valid ansatz is supplied to qnodes"""
        with pytest.raises(ValueError, match="ansatz is not a callable function"):
            qnodes = qml.vqe.qnodes(ansatz, observables, device=mock_device)

    @pytest.mark.parametrize("coeffs, observables, expected", [
        ((-0.6,), (qml.PauliZ(0),), -0.6 * 1.0),
        ((1.0,), (qml.PauliX(0),), 0.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), 0.5 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), 0.5 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), 0.5 * 1.0 + 1.2 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), 0.5 * 1.0 + 1.2 * 1.0)
    ])
    def test_aggregate_expval(self, coeffs, observables, expected):
        """Tests that the aggregate function returns correct expectation values"""
        qnodes = qml.vqe.qnodes(empty_ansatz, observables)
        expval = qml.vqe.aggregate(coeffs, qnodes, empty_params)
        assert expval == expected

    @pytest.mark.parametrize("ansatz, params", [
        (amp_embed, EMBED_PARAMS),
        (strong_ent_layer, LAYER_PARAMS),
        (amp_embed_and_strong_ent_layer, (EMBED_PARAMS, LAYER_PARAMS)),
    ])
    @pytest.mark.parametrize("coeffs, observables", [z for z in zip(COEFFS, OBSERVABLES)])
    def test_cost_evaluate(self, params, ansatz, coeffs, observables):
        """Tests that the cost function evaluates properly"""
        hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
        expval = qml.vqe.cost(params, ansatz, hamiltonian)
        assert type(expval) == float
        assert np.shape(expval) == () # expval should be scalar

    @pytest.mark.parametrize("coeffs, observables, expected", [
        ((-0.6,), (qml.PauliZ(0),), -0.6 * 1.0),
        ((1.0,), (qml.PauliX(0),), 0.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), 0.5 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), 0.5 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), 0.5 * 1.0 + 1.2 * 1.0),
        ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), 0.5 * 1.0 + 1.2 * 1.0)
    ])
    def test_cost_expvals(self, coeffs, observables, expected):
        """Tests that the cost function gives the correct expectation value"""
        hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
        cost = qml.vqe.cost(empty_params, empty_ansatz, hamiltonian)
        assert cost == expected

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    def test_cost_invalid_ansatz(self, ansatz):
        """Tests that the cost function raises an exception if the ansatz is not valid"""
        hamiltonian = qml.vqe.Hamiltonian((1.0,), (qml.PauliZ(0)))
        with pytest.raises(ValueError, match="no valid ansatz was provided"):
            cost = qml.vqe.cost(empty_params, empty_ansatz, hamiltonian)

    @pytest.mark.parametrize("hamiltonian", JUNK_INPUTS)
    def test_cost_invalid_ansatz(self, hamiltonian):
        """Tests that the cost function raises an exception if the Hamiltonian is not valid"""
        with pytest.raises(ValueError, match="the Hamiltonian is invalid"):
            cost = qml.vqe.cost(empty_params, empty_ansatz, hamiltonian)
