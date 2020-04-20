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
Unit tests for the :mod:`pennylane.vqe` submodule.
"""
import pytest
import pennylane as qml
import numpy as np


try:
    import torch
except ImportError as e:
    pass


try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        print(tf.__version__)
        import tensorflow.contrib.eager as tfe
        tf.enable_eager_execution()
        Variable = tfe.Variable
    else:
        from tensorflow import Variable
except ImportError as e:
    pass


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
    ((-0.6,), (qml.PauliZ(0),), [-0.6 * 1.0]),
    ((1.0,), (qml.PauliX(0),), [0.0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), [0.5 * 1.0, 0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), [0.5 * 1.0, 0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), [0.5 * 1.0, 1.2 * 1.0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), [0.5 * 1.0, 1.2 * 1.0]),
]

#####################################################
# Ansatz

def custom_fixed_ansatz(params, wires=None):
    """Custom fixed ansatz"""
    qml.RX(0.5, wires=0)
    qml.RX(-1.2, wires=1)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])


def custom_var_ansatz(params, wires=None):
    """Custom parametrized ansatz"""
    for p in params:
        qml.RX(p, wires=wires[0])

    qml.CNOT(wires=[wires[0], wires[1]])

    for p in params:
        qml.RX(-p, wires=wires[1])

    qml.CNOT(wires=[wires[0], wires[1]])


def amp_embed_and_strong_ent_layer(params, wires=None):
    """Ansatz combining amplitude embedding and
    strongly entangling layers"""
    qml.templates.embeddings.AmplitudeEmbedding(params[0], wires=wires)
    qml.templates.layers.StronglyEntanglingLayers(params[1], wires=wires)


ANSAETZE = [
    lambda params, wires=None: None,
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
    (lambda params, wires=None: None, EMPTY_PARAMS),
    (custom_fixed_ansatz, EMPTY_PARAMS),
    (custom_var_ansatz, VAR_PARAMS),
    (qml.templates.layers.StronglyEntanglingLayers, LAYER_PARAMS),
    (qml.templates.embeddings.AmplitudeEmbedding, EMBED_PARAMS),
    (amp_embed_and_strong_ent_layer, (EMBED_PARAMS, LAYER_PARAMS)),
]

#####################################################
# Device

@pytest.fixture(scope="function")
def mock_device(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(qml.Device, "__abstractmethods__", frozenset())
        m.setattr(qml.Device, "_capabilities", {"tensor_observables": True, "model": "qubit"})
        m.setattr(qml.Device, "operations", ["RX", "Rot", "CNOT", "Hadamard", "QubitStateVector"])
        m.setattr(
            qml.Device, "observables", ["PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian"]
        )
        m.setattr(qml.Device, "short_name", "MockDevice")
        m.setattr(qml.Device, "expval", lambda self, x, y, z: 1)
        m.setattr(qml.Device, "var", lambda self, x, y, z: 2)
        m.setattr(qml.Device, "sample", lambda self, x, y, z: 3)
        m.setattr(qml.Device, "apply", lambda self, x, y, z: None)
        yield qml.Device()

#####################################################
# Tests

class TestHamiltonian:
    """Test the Hamiltonian class"""

    @pytest.mark.parametrize("coeffs, ops", valid_hamiltonians)
    def test_hamiltonian_valid_init(self, coeffs, ops):
        """Tests that the Hamiltonian object is created with
        the correct attributes"""
        H = qml.vqe.Hamiltonian(coeffs, ops)
        assert H.terms == (coeffs, ops)

    @pytest.mark.parametrize("coeffs, ops", invalid_hamiltonians)
    def test_hamiltonian_invalid_init_exception(self, coeffs, ops):
        """Tests that an exception is raised when giving an invalid
        combination of coefficients and ops"""
        with pytest.raises(ValueError, match="number of coefficients and operators does not match"):
            H = qml.vqe.Hamiltonian(coeffs, ops)

    @pytest.mark.parametrize("coeffs", [[0.2, -1j], [0.5j, 2-1j]])
    def test_hamiltonian_complex(self, coeffs):
        """Tests that an exception is raised when
        a complex Hamiltonian is given"""
        obs = [qml.PauliX(0), qml.PauliZ(1)]

        with pytest.raises(ValueError, match="coefficients are not real-valued"):
            H = qml.vqe.Hamiltonian(coeffs, obs)

    @pytest.mark.parametrize("obs", [[qml.PauliX(0), qml.CNOT(wires=[0, 1])], [qml.PauliZ, qml.PauliZ(0)]])
    def test_hamiltonian_invalid_observables(self, obs):
        """Tests that an exception is raised when
        a complex Hamiltonian is given"""
        coeffs = [0.1, 0.2]

        with pytest.raises(ValueError, match="observables are not valid"):
            H = qml.vqe.Hamiltonian(coeffs, obs)


class TestVQE:
    """Test the core functionality of the VQE module"""

    @pytest.mark.parametrize("ansatz", ANSAETZE)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_valid_init(self, ansatz, observables, mock_device):
        """Tests that a collection of circuits is properly created by vqe.circuits"""
        circuits = qml.map(ansatz, observables, device=mock_device)

        assert len(circuits) == len(observables)
        assert all(callable(c) for c in circuits)
        assert all(c.device == mock_device for c in circuits)
        assert all(hasattr(c, "jacobian") for c in circuits)

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_evaluate(self, ansatz, observables, params, mock_device, seed):
        """Tests that the circuits returned by ``vqe.circuits`` evaluate properly"""
        mock_device.num_wires = 3
        circuits = qml.map(ansatz, observables, device=mock_device)
        res = circuits(params)
        assert all(val == 1.0 for val in res)

    @pytest.mark.parametrize("coeffs, observables, expected", hamiltonians_with_expvals)
    def test_circuits_expvals(self, coeffs, observables, expected):
        """Tests that the vqe.circuits function returns correct expectation values"""
        dev = qml.device("default.qubit", wires=2)
        circuits = qml.map(lambda params, **kwargs: None, observables, dev)
        res = [a * c([]) for a, c in zip(coeffs, circuits)]
        assert np.all(res == expected)

    @pytest.mark.parametrize("ansatz", ANSAETZE)
    @pytest.mark.parametrize("observables", JUNK_INPUTS)
    def test_circuits_no_observables(self, ansatz, observables, mock_device):
        """Tests that an exception is raised when no observables are supplied to vqe.circuits"""
        with pytest.raises(ValueError, match="observables are not valid"):
            obs = (observables,)
            circuits = qml.map(ansatz, obs, device=mock_device)

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_no_ansatz(self, ansatz, observables, mock_device):
        """Tests that an exception is raised when no valid ansatz is supplied to vqe.circuits"""
        with pytest.raises(ValueError, match="not a callable function"):
            circuits = qml.map(ansatz, observables, device=mock_device)

    @pytest.mark.parametrize("coeffs, observables, expected", hamiltonians_with_expvals)
    def test_aggregate_expval(self, coeffs, observables, expected):
        """Tests that the aggregate function returns correct expectation values"""
        dev = qml.device("default.qubit", wires=2)
        qnodes = qml.map(lambda params, **kwargs: None, observables, dev)
        expval = qml.dot(coeffs, qnodes)
        assert expval([]) == sum(expected)

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("coeffs, observables", [z for z in zip(COEFFS, OBSERVABLES)])
    def test_cost_evaluate(self, params, ansatz, coeffs, observables):
        """Tests that the cost function evaluates properly"""
        hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
        dev = qml.device("default.qubit", wires=3)
        expval = qml.VQECost(ansatz, hamiltonian, dev)
        assert type(expval(params)) == np.float64
        assert np.shape(expval(params)) == ()  # expval should be scalar

    @pytest.mark.parametrize("coeffs, observables, expected", hamiltonians_with_expvals)
    def test_cost_expvals(self, coeffs, observables, expected):
        """Tests that the cost function returns correct expectation values"""
        dev = qml.device("default.qubit", wires=2)
        hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
        cost = qml.VQECost(lambda params, **kwargs: None, hamiltonian, dev)
        assert cost([]) == sum(expected)

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    def test_cost_invalid_ansatz(self, ansatz, mock_device):
        """Tests that the cost function raises an exception if the ansatz is not valid"""
        hamiltonian = qml.vqe.Hamiltonian((1.0,), [qml.PauliZ(0)])
        with pytest.raises(ValueError, match="not a callable function."):
            cost = qml.VQECost(4, hamiltonian, mock_device)

    @pytest.mark.parametrize("coeffs, observables, expected", hamiltonians_with_expvals)
    def test_passing_kwargs(self, coeffs, observables, expected):
        """Test that the step size and order used for the finite differences
        differentiation method were passed to the QNode instances using the
        keyword arguments."""
        dev = qml.device("default.qubit", wires=2)
        hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
        cost = qml.VQECost(lambda params, **kwargs: None, hamiltonian, dev, h=123, order=2)

        # Checking that the qnodes contain the step size and order
        for qnode in cost.qnodes:
            assert qnode.h == 123
            assert qnode.order == 2


class TestAutogradInterface:
    """Tests for the Autograd interface (and the NumPy interface for backward compatibility)"""

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    @pytest.mark.parametrize("interface", ["autograd", "numpy"])
    def test_QNodes_have_right_interface(self, ansatz, observables, params, mock_device, interface):
        """Test that QNodes have the Autograd interface"""
        mock_device.num_wires = 3
        circuits = qml.map(ansatz, observables, device=mock_device, interface=interface)

        assert all(c.interface == "autograd" for c in circuits)
        assert all(c.__class__.__name__ == "AutogradQNode" for c in circuits)

        res = [c(params) for c in circuits]
        assert all(isinstance(val, float) for val in res)

    @pytest.mark.parametrize("interface", ["autograd", "numpy"])
    def test_gradient(self, tol, interface):
        """Test differentiation works"""
        dev = qml.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0), qml.PauliY(0)]

        H = qml.vqe.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = np.array([a, b])

        cost = qml.VQECost(ansatz, H, dev, interface=interface)
        dcost = qml.grad(cost, argnum=[0])
        res = dcost(params)

        expected = [
            -coeffs[0]*np.sin(a)*np.sin(b) - coeffs[1]*np.cos(a),
            coeffs[0]*np.cos(a)*np.cos(b)
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.usefixtures("skip_if_no_torch_support")
class TestTorchInterface:
    """Tests for the PyTorch interface"""

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_QNodes_have_right_interface(self, ansatz, observables, params, mock_device):
        """Test that QNodes have the torch interface"""
        mock_device.num_wires = 3
        circuits = qml.map(ansatz, observables, device=mock_device, interface="torch")
        assert all(c.interface == "torch" for c in circuits)

        res = [c(params) for c in circuits]
        assert all(isinstance(val, torch.Tensor) for val in res)

    def test_gradient(self, tol):
        """Test differentiation works"""
        dev = qml.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0), qml.PauliY(0)]

        H = qml.vqe.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = torch.autograd.Variable(torch.tensor([a, b]), requires_grad=True)

        cost = qml.VQECost(ansatz, H, dev, interface="torch")
        loss = cost(params)
        loss.backward()

        res = params.grad.numpy()

        expected = [
            -coeffs[0]*np.sin(a)*np.sin(b) - coeffs[1]*np.cos(a),
            coeffs[0]*np.cos(a)*np.cos(b)
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)



@pytest.mark.usefixtures("skip_if_no_tf_support")
class TestTFInterface:
    """Tests for the TF interface"""

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_QNodes_have_right_interface(self, ansatz, observables, params, mock_device):
        """Test that QNodes have the tf interface"""
        if ansatz == amp_embed_and_strong_ent_layer:
            pytest.skip("TF doesn't work with ragged arrays")

        mock_device.num_wires = 3
        circuits = qml.map(ansatz, observables, device=mock_device, interface="tf")
        assert all(c.interface == "tf" for c in circuits)

        res = [c(params) for c in circuits]
        assert all(isinstance(val, (Variable, tf.Tensor)) for val in res)

    def test_gradient(self, tol):
        """Test differentiation works"""
        dev = qml.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0), qml.PauliY(0)]

        H = qml.vqe.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = Variable([a, b], dtype=tf.float64)
        cost = qml.VQECost(ansatz, H, dev, interface="tf")

        with tf.GradientTape() as tape:
            loss = cost(params)
            res = np.array(tape.gradient(loss, params))

        expected = [
            -coeffs[0]*np.sin(a)*np.sin(b) - coeffs[1]*np.cos(a),
            coeffs[0]*np.cos(a)*np.cos(b)
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.usefixtures("skip_if_no_tf_support")
@pytest.mark.usefixtures("skip_if_no_torch_support")
class TestMultipleInterfaceIntegration:
    """Tests to ensure that interfaces agree and integrate correctly"""

    def test_all_interfaces_gradient_agree(self, tol):
        """Test the gradient agrees across all interfaces"""
        dev = qml.device("default.qubit", wires=2)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0)@qml.PauliZ(1), qml.PauliY(0)]

        H = qml.vqe.Hamiltonian(coeffs, observables)

        # TensorFlow interface
        params = Variable(qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2, seed=1))
        ansatz = qml.templates.layers.StronglyEntanglingLayers

        cost = qml.VQECost(ansatz, H, dev, interface="tf")

        with tf.GradientTape() as tape:
            loss = cost(params)
            res_tf = np.array(tape.gradient(loss, params))

        # Torch interface
        params = torch.tensor(qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2, seed=1))
        params = torch.autograd.Variable(params, requires_grad=True)
        ansatz = qml.templates.layers.StronglyEntanglingLayers

        cost = qml.VQECost(ansatz, H, dev, interface="torch")
        loss = cost(params)
        loss.backward()
        res_torch = params.grad.numpy()

        # NumPy interface
        params = qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2, seed=1)
        ansatz = qml.templates.layers.StronglyEntanglingLayers
        cost = qml.VQECost(ansatz, H, dev, interface="numpy")
        dcost = qml.grad(cost, argnum=[0])
        res = dcost(params)

        assert np.allclose(res, res_tf, atol=tol, rtol=0)
        assert np.allclose(res, res_torch, atol=tol, rtol=0)
