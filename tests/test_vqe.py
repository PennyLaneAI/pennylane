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
import numpy as np


@pytest.fixture(scope="function")
def seed():
    """Resets the random seed with every test"""
    np.random.seed(0)

#####################################################
# Hamiltonians


H_ONE_QUBIT = np.array([[1.0, 0.5j], [-0.5j, 2.5]])

H_TWO_QUBITS = np.array(
    [[0.5, 1.0j, 0.0, -3j], [-1.0j, -1.1, 0.0, -0.1], [0.0, 0.0, -0.9, 12.0], [3j, -0.1, 12.0, 0.0]]
)

COEFFS = [(0.5, 1.2, -0.7), (2.2, -0.2, 0.0), (0.33,)]

OBSERVABLES = [
    (qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)),
    (qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0) @ qml.PauliZ(1), qml.PauliZ(1)),
    (qml.Hermitian(H_TWO_QUBITS, [0, 1]),),
]

JUNK_INPUTS = [None, [], tuple(), 5.0, {"junk": -1}]

valid_hamiltonians = [
    ((1.0,), (qml.Hermitian(H_TWO_QUBITS, [0, 1]),)),
    ((-0.8,), (qml.PauliZ(0),)),
    ((0.5, -1.6), (qml.PauliX(0), qml.PauliY(1))),
    ((0.5, -1.6), (qml.PauliX(1), qml.PauliY(1))),
    ((1.1, -0.4, 0.333), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 2), qml.PauliZ(2))),
    ((-0.4, 0.15), (qml.Hermitian(H_TWO_QUBITS, [0, 2]), qml.PauliZ(1))),
    ([1.5, 2.0], [qml.PauliZ(0), qml.PauliY(2)]),
    (np.array([-0.1, 0.5]), [qml.Hermitian(H_TWO_QUBITS, [0, 1]), qml.PauliY(0)]),
    ((0.5, 1.2), (qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1))),
]

invalid_hamiltonians = [
    ((), (qml.PauliZ(0),)),
    ((), (qml.PauliZ(0), qml.PauliY(1))),
    ((3.5,), ()),
    ((1.2, -0.4), ()),
    ((0.5, 1.2), (qml.PauliZ(0),)),
    ((1.0,), (qml.PauliZ(0), qml.PauliY(0))),
]


hamiltonians_with_expvals = [
    ((-0.6,), (qml.PauliZ(0),), -0.6 * 1.0),
    ((1.0,), (qml.PauliX(0),), 0.0),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), 0.5 * 1.0),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), 0.5 * 1.0),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), 0.5 * 1.0 + 1.2 * 1.0),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), 0.5 * 1.0 + 1.2 * 1.0),
]

#####################################################
# Ansatz


def custom_fixed_ansatz(*params, wires=None):
    """Custom fixed ansatz"""
    qml.RX(0.5, wires=0)
    qml.RX(-1.2, wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])


def custom_var_ansatz(*params, wires=None):
    """Custom parametrized ansatz"""
    for p in params:
        qml.RX(p, wires=wires[0])

    qml.CNOT(wires=[wires[0], wires[1]])

    for p in params:
        qml.RX(-p, wires=wires[1])

    qml.CNOT(wires=[wires[0], wires[1]])


def amp_embed_and_strong_ent_layer(*params, wires=None):
    """Ansatz combining amplitude embedding and
    strongly entangling layers"""
    qml.templates.embeddings.AmplitudeEmbedding(*params[0], wires=wires)
    qml.templates.layers.StronglyEntanglingLayers(*params[1], wires=wires)


ANSAETZE = [
    lambda *params, wires=None: None,
    custom_fixed_ansatz,
    custom_var_ansatz,
    qml.templates.embeddings.AmplitudeEmbedding,
    qml.templates.layers.StronglyEntanglingLayers,
    amp_embed_and_strong_ent_layer,
]

#####################################################
# Parameters

EMPTY_PARAMS = []
VAR_PARAMS = [0.5]
EMBED_PARAMS = np.array([1 / np.sqrt(2 ** 3)] * 2 ** 3)
LAYER_PARAMS = qml.init.strong_ent_layers_uniform(n_layers=2, n_wires=3)

CIRCUITS = [
    (lambda *params, wires=None: None, EMPTY_PARAMS),
    (custom_fixed_ansatz, EMPTY_PARAMS),
    (custom_var_ansatz, VAR_PARAMS),
    (qml.templates.layers.StronglyEntanglingLayers, LAYER_PARAMS),
    # FIXME uncomment when https://github.com/XanaduAI/pennylane/issues/365 is addressed
    # (amp_embed, EMBED_PARAMS),
    # (amp_embed_and_strong_ent_layer, (EMBED_PARAMS, LAYER_PARAMS)),
]


#####################################################
# Device


@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(qml.Device, '__abstractmethods__', frozenset())
        m.setattr(qml.Device, '_capabilities', {})
        m.setattr(qml.Device, 'operations', ["RX", "Rot", "CNOT", "Hadamard"])
        m.setattr(qml.Device, 'observables', ["PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian"])
        m.setattr(qml.Device, 'short_name', 'MockDevice')
        m.setattr(qml.Device, 'expval', lambda self, x, y, z: 1)
        m.setattr(qml.Device, 'var', lambda self, x, y, z: 2)
        m.setattr(qml.Device, 'sample', lambda self, x, y, z: 3)
        m.setattr(qml.Device, 'apply', lambda self, x, y, z: None)
        yield qml.Device()


class TestHamiltonian:
    """Test the Hamiltonian class"""

    @pytest.mark.parametrize("coeffs, ops", valid_hamiltonians)
    def test_hamiltonian_valid_init(self, coeffs, ops):
        """Tests that the Hamiltonian object is created with the correct attributes"""
        H = qml.vqe.Hamiltonian(coeffs, ops)
        assert H.terms == (coeffs, ops)

    @pytest.mark.parametrize("coeffs, ops", invalid_hamiltonians)
    def test_hamiltonian_invalid_init_exception(self, coeffs, ops):
        """Tests that an exception is raised when giving an invalid combination of coefficients and ops"""
        with pytest.raises(ValueError, match="number of coefficients and operators does not match"):
            H = qml.vqe.Hamiltonian(coeffs, ops)


class TestVQE:
    """Test the core functionality of the VQE module"""

    @pytest.mark.parametrize("ansatz", ANSAETZE)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_valid_init(self, ansatz, observables, mock_device):
        """Tests that a collection of circuits is properly created by vqe.circuits"""
        circuits = qml.vqe.circuits(ansatz, observables, device=mock_device)

        assert len(circuits) == len(observables)
        assert all(callable(c) for c in circuits)
        assert all(c.device == mock_device for c in circuits)
        assert all(hasattr(c, "jacobian") for c in circuits)

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_evaluate(self, ansatz, observables, params, mock_device, seed):
        """Tests that the circuits returned by ``vqe.circuits`` evaluate properly"""
        mock_device.num_wires = 3
        circuits = qml.vqe.circuits(ansatz, observables, device=mock_device)

        res = [c(params) for c in circuits]
        assert all(val == 1.0 for val in res)

    @pytest.mark.parametrize("coeffs, observables, expected")
    def test_circuits_expvals(self, coeffs, observables, expected):
        """Tests that the vqe.circuits function returns correct expectation values"""
        dev = qml.device("default.qubit", wires=2)
        circuits = qml.vqe.circuits(EMPTY_ANSATZ, observables, dev)
        for c in circuits:
            val = c(*EMPTY_PARAMS)
            assert val == expected

#     @pytest.mark.parametrize("ansatz", ANSAETZE)
#     @pytest.mark.parametrize("observables", JUNK_INPUTS)
#     def test_circuits_no_observables(self, ansatz, observables, mock_device):
#         """Tests that an exception is raised when no observables are supplied to vqe.circuits"""
#         with pytest.raises(ValueError, match="observables are not valid"):
#             obs = (observables,)
#             circuits = qml.vqe.circuits(ansatz, obs, device=mock_device)

#     @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
#     @pytest.mark.parametrize("observables", OBSERVABLES)
#     def test_circuits_no_ansatz(self, ansatz, observables, mock_device):
#         """Tests that an exception is raised when no valid ansatz is supplied to vqe.circuits"""
#         with pytest.raises(ValueError, match="ansatz is not a callable function"):
#             circuits = qml.vqe.circuits(ansatz, observables, device=mock_device)

#     @pytest.mark.parametrize(
#         "coeffs, observables, expected",
#         [
#             ((-0.6,), (qml.PauliZ(0),), -0.6 * 1.0),
#             ((1.0,), (qml.PauliX(0),), 0.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), 0.5 * 1.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), 0.5 * 1.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), 0.5 * 1.0 + 1.2 * 1.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), 0.5 * 1.0 + 1.2 * 1.0),
#         ],
#     )
#     def test_aggregate_expval(self, coeffs, observables, expected):
#         """Tests that the aggregate function returns correct expectation values"""
#         qnodes = qml.vqe.qnodes(EMPTY_ANSATZ, observables)
#         expval = qml.vqe.aggregate(coeffs, qnodes, empty_params)
#         assert expval == expected

#     @pytest.mark.parametrize(
#         "ansatz, params",
#         [
#             (amp_embed, EMBED_PARAMS),
#             (strong_ent_layer, LAYER_PARAMS),
#             (amp_embed_and_strong_ent_layer, (EMBED_PARAMS, LAYER_PARAMS)),
#         ],
#     )
#     @pytest.mark.parametrize("coeffs, observables", [z for z in zip(COEFFS, OBSERVABLES)])
#     def test_cost_evaluate(self, params, ansatz, coeffs, observables):
#         """Tests that the cost function evaluates properly"""
#         hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
#         expval = qml.vqe.cost(params, ansatz, hamiltonian)
#         assert type(expval) == float
#         assert np.shape(expval) == ()  # expval should be scalar

#     @pytest.mark.parametrize(
#         "coeffs, observables, expected",
#         [
#             ((-0.6,), (qml.PauliZ(0),), -0.6 * 1.0),
#             ((1.0,), (qml.PauliX(0),), 0.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), 0.5 * 1.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), 0.5 * 1.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), 0.5 * 1.0 + 1.2 * 1.0),
#             ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), 0.5 * 1.0 + 1.2 * 1.0),
#         ],
#     )
#     def test_cost_expvals(self, coeffs, observables, expected):
#         """Tests that the cost function gives the correct expectation value"""
#         hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
#         cost = qml.vqe.cost(empty_params, EMPTY_ANSATZ, hamiltonian)
#         assert cost == expected

#     @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
#     def test_cost_invalid_ansatz(self, ansatz):
#         """Tests that the cost function raises an exception if the ansatz is not valid"""
#         hamiltonian = qml.vqe.Hamiltonian((1.0,), (qml.PauliZ(0)))
#         with pytest.raises(ValueError, match="no valid ansatz was provided"):
#             cost = qml.vqe.cost(empty_params, EMPTY_ANSATZ, hamiltonian)

#     @pytest.mark.parametrize("hamiltonian", JUNK_INPUTS)
#     def test_cost_invalid_ansatz(self, hamiltonian):
#         """Tests that the cost function raises an exception if the Hamiltonian is not valid"""
#         with pytest.raises(ValueError, match="the Hamiltonian is invalid"):
#             cost = qml.vqe.cost(empty_params, EMPTY_ANSATZ, hamiltonian)
