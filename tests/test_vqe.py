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
import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.fixture(scope="function")
def seed():
    """Resets the random seed with every test"""
    np.random.seed(0)


def catch_warn_ExpvalCost(ansatz, hamiltonian, device, **kwargs):
    """Computes the ExpvalCost and catches the initial deprecation warning."""

    with pytest.warns(UserWarning, match="is deprecated,"):
        res = qml.ExpvalCost(ansatz, hamiltonian, device, **kwargs)
    return res


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

OBSERVABLES_NO_HERMITIAN = [
    (qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)),
    (qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0) @ qml.PauliZ(1), qml.PauliZ(1)),
]

JUNK_INPUTS = [None, [], tuple(), 5.0, {"junk": -1}]

hamiltonians_with_expvals = [
    ((-0.6,), (qml.PauliZ(0),), [-0.6 * 1.0]),
    ((1.0,), (qml.PauliX(0),), [0.0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(0)), [0.5 * 1.0, 0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliX(1)), [0.5 * 1.0, 0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(0)), [0.5 * 1.0, 1.2 * 1.0]),
    ((0.5, 1.2), (qml.PauliZ(0), qml.PauliZ(1)), [0.5 * 1.0, 1.2 * 1.0]),
]

zero_hamiltonians_with_expvals = [
    ([], [], [0]),
    ((0, 0), (qml.PauliZ(0), qml.PauliZ(1)), [0]),
    ((0, 0, 0), (qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0), qml.PauliX(1)), [0]),
]

big_hamiltonian_coeffs = np.array(
    [
        -0.04207898,
        0.17771287,
        0.17771287,
        -0.24274281,
        -0.24274281,
        0.17059738,
        0.04475014,
        -0.04475014,
        -0.04475014,
        0.04475014,
        0.12293305,
        0.16768319,
        0.16768319,
        0.12293305,
        0.17627641,
    ]
)

big_hamiltonian_ops = [
    qml.Identity(wires=[0]),
    qml.PauliZ(wires=[0]),
    qml.PauliZ(wires=[1]),
    qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
    qml.PauliY(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
]

big_hamiltonian = qml.Hamiltonian(big_hamiltonian_coeffs, big_hamiltonian_ops)

big_hamiltonian_grad = (
    np.array(
        [
            [
                [6.52084595e-18, -2.11464420e-02, -1.16576858e-02],
                [-8.22589330e-18, -5.20597922e-02, -1.85365365e-02],
                [-2.73850768e-17, 1.14202988e-01, -5.45041403e-03],
                [-1.27514307e-17, -1.10465531e-01, 5.19489457e-02],
            ],
            [
                [-2.45428288e-02, 8.38921555e-02, -2.00641818e-17],
                [-2.21085973e-02, 7.39332741e-04, -1.25580654e-17],
                [9.62058625e-03, -1.51398765e-01, 2.02129847e-03],
                [1.10020832e-03, -3.49066271e-01, 2.13669117e-03],
            ],
        ]
    ),
)

#####################################################
# Ansatz


# pylint: disable=unused-argument
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
EMBED_PARAMS = np.array([1 / np.sqrt(2**3)] * 2**3)
LAYER_PARAMS = np.random.random(qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3))

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


@pytest.fixture(scope="function", name="mock_device")
def mock_device_fixture(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(qml.Device, "__abstractmethods__", frozenset())
        m.setattr(
            qml.Device, "_capabilities", {"supports_tensor_observables": True, "model": "qubit"}
        )
        m.setattr(qml.Device, "operations", ["RX", "RY", "Rot", "CNOT", "Hadamard", "StatePrep"])
        m.setattr(
            qml.Device, "observables", ["PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian"]
        )
        m.setattr(qml.Device, "short_name", "MockDevice")
        m.setattr(qml.Device, "expval", lambda self, x, y, z: 1)
        m.setattr(qml.Device, "var", lambda self, x, y, z: 2)
        m.setattr(qml.Device, "sample", lambda self, x, y, z: 3)
        m.setattr(qml.Device, "apply", lambda self, x, y, z: None)

        def get_device(wires=1):
            return qml.Device(wires=wires)  # pylint:disable=abstract-class-instantiated

        yield get_device


#####################################################
# Queues

QUEUE_HAMILTONIANS_1 = [
    qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
    qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
]

QUEUE_HAMILTONIANS_2 = [
    qml.Hamiltonian([1], [qml.PauliX(0)]),
    qml.Hamiltonian([5], [qml.PauliX(0) @ qml.PauliZ(1)]),
]

QUEUES = [
    [
        qml.PauliX(0),
        qml.PauliZ(1),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.PauliX(0),
        qml.Hamiltonian([1], [qml.PauliX(0)]),
        qml.Hamiltonian([2, 1], [qml.PauliX(0), qml.PauliZ(1)]),
    ],
    [
        qml.PauliX(0),
        qml.PauliZ(1),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.PauliX(0),
        qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliZ(1),
        qml.Hamiltonian([1], [qml.PauliX(0) @ qml.PauliZ(1)]),
        qml.Hamiltonian([1, 1, 2], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliZ(1)]),
    ],
]

add_queue = zip(QUEUE_HAMILTONIANS_1, QUEUE_HAMILTONIANS_2, QUEUES)

#####################################################
# Tests


class TestVQE:
    """Test the core functionality of the VQE module"""

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("coeffs, observables", list(zip(COEFFS, OBSERVABLES)))
    def test_cost_evaluate(self, params, ansatz, coeffs, observables):
        """Tests that the cost function evaluates properly"""
        hamiltonian = qml.Hamiltonian(coeffs, observables)
        dev = qml.device("default.qubit", wires=3)
        expval = catch_warn_ExpvalCost(ansatz, hamiltonian, dev)
        assert expval(params).dtype == np.float64
        assert np.shape(expval(params)) == ()  # expval should be scalar

    @pytest.mark.parametrize(
        "coeffs, observables, expected", hamiltonians_with_expvals + zero_hamiltonians_with_expvals
    )
    def test_cost_expvals(self, coeffs, observables, expected):
        """Tests that the cost function returns correct expectation values"""
        dev = qml.device("default.qubit", wires=2)
        hamiltonian = qml.Hamiltonian(coeffs, observables)
        cost = catch_warn_ExpvalCost(lambda params, **kwargs: None, hamiltonian, dev)
        assert cost([]) == sum(expected)

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    def test_cost_invalid_ansatz(self, ansatz, mock_device):
        """Tests that the cost function raises an exception if the ansatz is not valid"""
        hamiltonian = qml.Hamiltonian((1.0,), [qml.PauliZ(0)])
        with pytest.raises(ValueError, match="not a callable function."):
            catch_warn_ExpvalCost(4, hamiltonian, mock_device())

    @pytest.mark.autograd
    @pytest.mark.parametrize("coeffs, observables, expected", hamiltonians_with_expvals)
    def test_passing_kwargs(self, coeffs, observables, expected):
        """Test that the step size and order used for the finite differences
        differentiation method were passed to the QNode instances using the
        keyword arguments."""
        dev = qml.device("default.qubit", wires=2)
        hamiltonian = qml.Hamiltonian(coeffs, observables)
        cost = catch_warn_ExpvalCost(
            lambda params, **kwargs: None, hamiltonian, dev, h=123, order=2
        )

        # Checking that the qnodes contain the step size and order
        for qnode in cost.qnodes:
            assert qnode.gradient_kwargs["h"] == 123
            assert qnode.gradient_kwargs["order"] == 2

    # pylint: disable=protected-access
    @pytest.mark.torch
    @pytest.mark.slow
    @pytest.mark.parametrize("shots", [None, [(8000, 5)], [(8000, 5), (9000, 4)]])
    def test_optimize_torch(self, shots):
        """Test that an ExpvalCost with observable optimization gives the same result as another
        ExpvalCost without observable optimization."""

        dev = qml.device("default.qubit", wires=4, shots=shots)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            optimize=True,
            interface="torch",
            diff_method="parameter-shift",
        )
        cost2 = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            optimize=False,
            interface="torch",
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            c1 = cost(w)

        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)

        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 5  # Number of groups in the Hamiltonian
        assert exec_no_opt == 15

        assert np.allclose(c1, c2, atol=1e-1)

    # pylint: disable=protected-access
    @pytest.mark.tf
    @pytest.mark.slow
    @pytest.mark.parametrize("shots", [None, [(8000, 5)], [(8000, 5), (9000, 4)]])
    def test_optimize_tf(self, shots):
        """Test that an ExpvalCost with observable optimization gives the same result as another
        ExpvalCost without observable optimization."""

        dev = qml.device("default.qubit", wires=4, shots=shots)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            optimize=True,
            interface="tf",
            diff_method="parameter-shift",
        )
        cost2 = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            optimize=False,
            interface="tf",
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 5  # Number of groups in the Hamiltonian
        assert exec_no_opt == 15

        assert np.allclose(c1, c2, atol=1e-1)

    # pylint: disable=protected-access
    @pytest.mark.autograd
    @pytest.mark.slow
    @pytest.mark.parametrize("shots", [None, [(8000, 5)], [(8000, 5), (9000, 4)]])
    def test_optimize_autograd(self, shots):
        """Test that an ExpvalCost with observable optimization gives the same result as another
        ExpvalCost without observable optimization."""

        dev = qml.device("default.qubit", wires=4, shots=shots)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            optimize=True,
            interface="autograd",
            diff_method="parameter-shift",
        )
        cost2 = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            optimize=False,
            interface="autograd",
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 5  # Number of groups in the Hamiltonian
        assert exec_no_opt == 15

        assert np.allclose(c1, c2, atol=1e-1)

    # pylint: disable=protected-access
    @pytest.mark.autograd
    def test_optimize_multiple_terms_autograd(self):
        """Test that an ExpvalCost with observable optimization gives the same
        result as another ExpvalCost without observable optimization even when there
        are non-unique Hamiltonian terms."""

        dev = qml.device("default.qubit", wires=5)
        obs = [
            qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[4]),  # <---- These two terms
            qml.PauliZ(wires=[4]) @ qml.PauliZ(wires=[2]),  # <---- are equal
            qml.PauliZ(wires=[1]),
            qml.PauliZ(wires=[2]),
            qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
            qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[0]),
            qml.PauliZ(wires=[3]) @ qml.PauliZ(wires=[1]),
            qml.PauliZ(wires=[4]) @ qml.PauliZ(wires=[3]),
        ]

        coefs = (np.random.rand(len(obs)) - 0.5) * 2
        hamiltonian1 = qml.Hamiltonian(coefs, obs)
        hamiltonian2 = qml.Hamiltonian(coefs, obs)

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            optimize=True,
            interface="autograd",
            diff_method="parameter-shift",
        )
        cost2 = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            optimize=False,
            interface="autograd",
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=5)
        w = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 1  # Number of groups in the Hamiltonian
        assert exec_no_opt == 8

        assert np.allclose(c1, c2)

    # pylint: disable=protected-access
    @pytest.mark.torch
    def test_optimize_multiple_terms_torch(self):
        """Test that an ExpvalCost with observable optimization gives the same
        result as another ExpvalCost without observable optimization even when there
        are non-unique Hamiltonian terms."""

        dev = qml.device("default.qubit", wires=5)
        obs = [
            qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[4]),  # <---- These two terms
            qml.PauliZ(wires=[4]) @ qml.PauliZ(wires=[2]),  # <---- are equal
            qml.PauliZ(wires=[1]),
            qml.PauliZ(wires=[2]),
            qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
            qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[0]),
            qml.PauliZ(wires=[3]) @ qml.PauliZ(wires=[1]),
            qml.PauliZ(wires=[4]) @ qml.PauliZ(wires=[3]),
        ]

        coefs = (np.random.rand(len(obs)) - 0.5) * 2
        hamiltonian1 = qml.Hamiltonian(coefs, obs)
        hamiltonian2 = qml.Hamiltonian(coefs, obs)

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            optimize=True,
            interface="torch",
            diff_method="parameter-shift",
        )
        cost2 = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            optimize=False,
            interface="torch",
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=5)
        w = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        # was 1, 8 on old device
        assert exec_opt == 1  # Number of groups in the Hamiltonian
        assert exec_no_opt == 8

        assert np.allclose(c1, c2)

    # pylint: disable=protected-access
    @pytest.mark.tf
    def test_optimize_multiple_terms_tf(self):
        """Test that an ExpvalCost with observable optimization gives the same
        result as another ExpvalCost without observable optimization even when there
        are non-unique Hamiltonian terms."""

        dev = qml.device("default.qubit", wires=5)
        obs = [
            qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[4]),  # <---- These two terms
            qml.PauliZ(wires=[4]) @ qml.PauliZ(wires=[2]),  # <---- are equal
            qml.PauliZ(wires=[1]),
            qml.PauliZ(wires=[2]),
            qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
            qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[0]),
            qml.PauliZ(wires=[3]) @ qml.PauliZ(wires=[1]),
            qml.PauliZ(wires=[4]) @ qml.PauliZ(wires=[3]),
        ]

        coefs = (np.random.rand(len(obs)) - 0.5) * 2
        hamiltonian1 = qml.Hamiltonian(coefs, obs)
        hamiltonian2 = qml.Hamiltonian(coefs, obs)

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            optimize=True,
            interface="tf",
            diff_method="parameter-shift",
        )
        cost2 = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            optimize=False,
            interface="tf",
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=5)
        w = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        # was 1, 8 on old device
        assert exec_opt == 1  # Number of groups in the Hamiltonian
        assert exec_no_opt == 8

        assert np.allclose(c1, c2)

    # pylint: disable=protected-access
    @pytest.mark.autograd
    def test_optimize_grad(self):
        """Test that the gradient of ExpvalCost is accessible and correct when using observable
        optimization and the autograd interface."""
        dev = qml.device("default.qubit", wires=4)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            optimize=True,
            diff_method="parameter-shift",
        )
        cost2 = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            optimize=False,
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = pnp.random.uniform(low=0, high=2 * np.pi, size=shape, requires_grad=True)

        with qml.Tracker(dev) as tracker:
            dc = qml.grad(cost)(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            dc2 = qml.grad(cost2)(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_no_opt > exec_opt
        assert np.allclose(dc, big_hamiltonian_grad)
        assert np.allclose(dc2, big_hamiltonian_grad)

    @pytest.mark.autograd
    @pytest.mark.parametrize("opt", [True, False])
    def test_grad_zero_hamiltonian(self, opt):
        """Test that the gradient of ExpvalCost is accessible and correct when using observable
        optimization and the autograd interface with a zero Hamiltonian."""
        dev = qml.device("default.qubit", wires=4)
        hamiltonian = qml.Hamiltonian([0], [qml.PauliX(0)])

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian,
            dev,
            optimize=opt,
            diff_method="parameter-shift",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = pnp.random.random(shape, requires_grad=True)

        dc = qml.grad(cost)(w)
        assert np.allclose(dc, 0)

    @pytest.mark.torch
    @pytest.mark.slow
    def test_optimize_grad_torch(self):
        """Test that the gradient of ExpvalCost is accessible and correct when using observable
        optimization and the Torch interface."""
        import torch

        dev = qml.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian,
            dev,
            optimize=True,
            interface="torch",
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = np.random.uniform(low=0, high=2 * np.pi, size=shape)
        w = torch.tensor(w, requires_grad=True)

        res = cost(w)
        res.backward()
        dc = w.grad.detach().numpy()

        assert np.allclose(dc, big_hamiltonian_grad)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_optimize_grad_tf(self):
        """Test that the gradient of ExpvalCost is accessible and correct when using observable
        optimization and the TensorFlow interface."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian

        cost = catch_warn_ExpvalCost(
            qml.templates.StronglyEntanglingLayers, hamiltonian, dev, optimize=True, interface="tf"
        )

        np.random.seed(1967)
        shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = np.random.uniform(low=0, high=2 * np.pi, size=shape)
        w = tf.Variable(w)

        with tf.GradientTape() as tape:
            res = cost(w)

        dc = tape.gradient(res, w).numpy()

        assert np.allclose(dc, big_hamiltonian_grad)

    def test_multiple_devices_opt_true(self):
        """Test if a ValueError is raised when multiple devices are passed when optimize=True."""
        dev = [qml.device("default.qubit", wires=2), qml.device("default.qubit", wires=2)]

        h = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])

        with pytest.raises(ValueError, match="Using multiple devices is not supported when"):
            catch_warn_ExpvalCost(qml.templates.StronglyEntanglingLayers, h, dev, optimize=True)

    def test_variance_error(self):
        """Test that an error is raised if attempting to use ExpvalCost to measure
        variances"""
        dev = qml.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian

        with pytest.raises(ValueError, match="sums of expectation values"):
            catch_warn_ExpvalCost(
                qml.templates.StronglyEntanglingLayers, hamiltonian, dev, measure="var"
            )


# Test data
np.random.seed(1967)
_shape = qml.templates.StronglyEntanglingLayers.shape(2, 4)
PARAMS = np.random.uniform(low=0, high=2 * np.pi, size=_shape)


class TestNewVQE:
    """Test the new VQE syntax of passing the Hamiltonian as an observable."""

    # pylint: disable=cell-var-from-loop
    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES_NO_HERMITIAN)
    def test_circuits_evaluate(self, ansatz, observables, params, tol):
        """Tests simple VQE evaluations."""
        coeffs = [1.0] * len(observables)
        dev = qml.device("default.qubit", wires=3)
        H = qml.Hamiltonian(coeffs, observables)

        # pass H directly
        @qml.qnode(dev)
        def circuit():
            ansatz(params, wires=range(3))
            return qml.expval(H)

        res = circuit()

        res_expected = []
        for obs in observables:

            @qml.qnode(dev)
            def separate_circuit():
                ansatz(params, wires=range(3))
                return qml.expval(obs)

            res_expected.append(separate_circuit())

        res_expected = np.sum([c * r for c, r in zip(coeffs, res_expected)])

        assert np.isclose(res, res_expected, atol=tol)

    def test_acting_on_subcircuit(self, tol):
        """Tests a VQE circuit where the observable does not act on all wires."""
        dev = qml.device("default.qubit", wires=3)
        coeffs = [1.0, 1.0, 1.0]
        np.random.seed(1967)
        w = np.random.random(qml.templates.StronglyEntanglingLayers.shape(n_layers=1, n_wires=2))

        observables1 = [qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)]
        H1 = qml.Hamiltonian(coeffs, observables1)

        @qml.qnode(dev)
        def circuit1():
            qml.templates.StronglyEntanglingLayers(w, wires=range(2))
            return qml.expval(H1)

        observables2 = [qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1) @ qml.Identity(2)]
        H2 = qml.Hamiltonian(coeffs, observables2)

        @qml.qnode(dev)
        def circuit2():
            qml.templates.StronglyEntanglingLayers(w, wires=range(2))
            return qml.expval(H2)

        res1 = circuit1()
        res2 = circuit2()

        assert np.allclose(res1, res2, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("shots, dim", [([(1000, 2)], 2), ([30, 30], 2), ([2, 3, 4], 3)])
    def test_shot_distribution(self, shots, dim):
        """Tests that distributed shots work with the new VQE design."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(weights, coeffs):
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            H = qml.Hamiltonian(coeffs, obs)
            return qml.expval(H)

        obs = [qml.PauliZ(0), qml.PauliX(0) @ qml.PauliZ(1)]
        coeffs = np.array([0.1, 0.2])
        key = jax.random.PRNGKey(42)
        weights = jax.random.uniform(key, [2, 2, 3])

        res = circuit(weights, coeffs)
        grad = jax.jacobian(circuit, argnums=[1])(weights, coeffs)
        assert len(res) == dim
        assert qml.math.shape(grad) == (dim, 1, 2)

    def test_circuit_drawer(self):
        """Test that the circuit drawer displays Hamiltonians well."""
        dev = qml.device("default.qubit", wires=3)
        coeffs = [1.0, 1.0, 1.0]
        observables1 = [qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(2)]
        H1 = qml.Hamiltonian(coeffs, observables1)

        @qml.qnode(dev)
        def circuit1():
            qml.Hadamard(wires=0)
            return qml.expval(H1)

        res = qml.draw(circuit1)()
        expected = "0: ──H─┤ ╭<𝓗(1.00,1.00,1.00)>\n2: ────┤ ╰<𝓗(1.00,1.00,1.00)>"
        assert res == expected

    def test_multiple_expvals(self):
        """Tests that more than one Hamiltonian expval can be evaluated."""

        coeffs = [1.0, 1.0, 1.0]
        dev = qml.device("default.qubit", wires=4)
        H1 = qml.Hamiltonian(coeffs, [qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)])
        H2 = qml.Hamiltonian(coeffs, [qml.PauliZ(2), qml.PauliY(2), qml.PauliZ(3)])
        w = PARAMS

        @qml.qnode(dev)
        def circuit():
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H1), qml.expval(H2)

        res = circuit()

        @qml.qnode(dev)
        def circuit1():
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H1)

        @qml.qnode(dev)
        def circuit2():
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H2)

        assert res[0] == circuit1()
        assert res[1] == circuit2()

    def test_multiple_expvals_same_wires(self):
        """Tests that more than one Hamiltonian expval can be evaluated."""

        coeffs = [1.0, 1.0, 1.0]
        dev = qml.device("default.qubit", wires=4)
        H1 = qml.Hamiltonian(coeffs, [qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)])
        w = PARAMS

        @qml.qnode(dev)
        def circuit():
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H1), qml.expval(H1)

        res = circuit()

        @qml.qnode(dev)
        def circuit1():
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H1)

        assert res[0] == circuit1()
        assert res[1] == circuit1()

    def test_error_var_measurement(self):
        """Tests that error is thrown if var(H) is measured."""
        observables = [qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)]
        coeffs = [1.0] * len(observables)
        dev = qml.device("default.qubit", wires=2)
        H = qml.Hamiltonian(coeffs, observables)

        @qml.qnode(dev)
        def circuit():
            return qml.var(H)

        with pytest.raises(NotImplementedError):
            circuit()

    def test_error_sample_measurement(self):
        """Tests that error is thrown if sample(H) is measured."""
        observables = [qml.PauliZ(0), qml.PauliY(0), qml.PauliZ(1)]
        coeffs = [1.0] * len(observables)
        dev = qml.device("default.qubit", wires=2, shots=10)
        H = qml.Hamiltonian(coeffs, observables)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(H)

        with pytest.raises(qml.operation.DiagGatesUndefinedError):
            circuit()

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "best"])
    def test_grad_autograd(self, diff_method, tol):
        """Tests the VQE gradient in the autograd interface."""
        dev = qml.device("default.qubit", wires=4)
        H = big_hamiltonian
        w = pnp.array(PARAMS, requires_grad=True)

        @qml.qnode(dev, diff_method=diff_method)
        def circuit(w):
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H)

        dc = qml.grad(circuit)(w)
        assert np.allclose(dc, big_hamiltonian_grad, atol=tol)

    @pytest.mark.autograd
    def test_grad_zero_hamiltonian(self, tol):
        """Tests the VQE gradient for a "zero" Hamiltonian."""
        dev = qml.device("default.qubit", wires=4)
        H = qml.Hamiltonian([0], [qml.PauliX(0)])
        w = pnp.array(PARAMS, requires_grad=True)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(w):
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H)

        dc = qml.grad(circuit)(w)
        assert np.allclose(dc, 0, atol=tol)

    @pytest.mark.torch
    @pytest.mark.slow
    def test_grad_torch(self, tol):
        """Tests VQE gradients in the torch interface."""
        import torch

        dev = qml.device("default.qubit", wires=4)
        H = big_hamiltonian

        @qml.qnode(dev)
        def circuit(w):
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H)

        w = torch.tensor(PARAMS, requires_grad=True)

        res = circuit(w)
        res.backward()  # pylint:disable=no-member
        dc = w.grad.detach().numpy()

        assert np.allclose(dc, big_hamiltonian_grad, atol=tol)

    @pytest.mark.tf
    def test_grad_tf(self, tol):
        """Tests VQE gradients in the tf interface."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=4)
        H = big_hamiltonian

        @qml.qnode(dev)
        def circuit(w):
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H)

        w = tf.Variable(PARAMS, dtype=tf.double)

        with tf.GradientTape() as tape:
            res = circuit(w)

        dc = tape.gradient(res, w).numpy()

        assert np.allclose(dc, big_hamiltonian_grad, atol=tol)

    @pytest.mark.jax
    @pytest.mark.slow
    def test_grad_jax(self, tol):
        """Tests VQE gradients in the jax interface."""
        import jax
        from jax import numpy as jnp

        dev = qml.device("default.qubit", wires=4)
        H = big_hamiltonian
        np.random.seed(1967)
        w = jnp.array(PARAMS)

        @qml.qnode(dev)
        def circuit(w):
            qml.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qml.expval(H)

        dc = jax.grad(circuit)(w)
        assert np.allclose(dc, big_hamiltonian_grad, atol=tol)

    def test_specs(self):
        """Test that the specs of a VQE circuit can be computed"""
        dev = qml.device("default.qubit", wires=2)
        H = qml.Hamiltonian([0.1, 0.2], [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1)])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(H)

        res = qml.specs(circuit)()

        assert res["num_observables"] == 1
        assert res["num_diagonalizing_gates"] == 0


class TestInterfaces:
    """Tests for VQE with interfaces."""

    @pytest.mark.autograd
    @pytest.mark.parametrize("interface", ["autograd"])
    def test_gradient_autograd(self, tol, interface):
        """Tests for the Autograd interface (and the NumPy interface for
        backward compatibility)"""
        dev = qml.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0), qml.PauliY(0)]

        H = qml.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = np.array([a, b])

        cost = catch_warn_ExpvalCost(ansatz, H, dev, interface=interface)
        dcost = qml.grad(cost, argnum=[0])
        res = dcost(params)

        expected = [
            -coeffs[0] * np.sin(a) * np.sin(b) - coeffs[1] * np.cos(a),
            coeffs[0] * np.cos(a) * np.cos(b),
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.torch
    def test_gradient_torch(self, tol):
        """Tests for the PyTorch interface"""
        import torch

        dev = qml.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0), qml.PauliY(0)]

        H = qml.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = torch.autograd.Variable(torch.tensor([a, b]), requires_grad=True)

        cost = catch_warn_ExpvalCost(ansatz, H, dev, interface="torch")
        loss = cost(params)
        loss.backward()

        res = params.grad.numpy()

        expected = [
            -coeffs[0] * np.sin(a) * np.sin(b) - coeffs[1] * np.cos(a),
            coeffs[0] * np.cos(a) * np.cos(b),
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_gradient_tf(self, tol):
        """Tests for the TF interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0), qml.PauliY(0)]

        H = qml.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = tf.Variable([a, b], dtype=tf.float64)
        cost = catch_warn_ExpvalCost(ansatz, H, dev, interface="tf")

        with tf.GradientTape() as tape:
            loss = cost(params)
            res = np.array(tape.gradient(loss, params))

        expected = [
            -coeffs[0] * np.sin(a) * np.sin(b) - coeffs[1] * np.cos(a),
            coeffs[0] * np.cos(a) * np.cos(b),
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)

    # Multiple interfaces will be tested with math module
    @pytest.mark.all_interfaces
    def test_all_interfaces_gradient_agree(self, tol):
        """Test the gradient agrees across all interfaces"""
        import tensorflow as tf
        import torch

        dev = qml.device("default.qubit", wires=2)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0)]

        H = qml.Hamiltonian(coeffs, observables)

        np.random.seed(1)
        shape = qml.templates.StronglyEntanglingLayers.shape(3, 2)
        params = np.random.uniform(low=0, high=2 * np.pi, size=shape)

        # TensorFlow interface
        w = tf.Variable(params)
        ansatz = qml.templates.layers.StronglyEntanglingLayers

        cost = catch_warn_ExpvalCost(ansatz, H, dev, interface="tf")

        with tf.GradientTape() as tape:
            loss = cost(w)
            res_tf = np.array(tape.gradient(loss, w))

        # Torch interface
        w = torch.tensor(params, requires_grad=True)
        w = torch.autograd.Variable(w, requires_grad=True)
        ansatz = qml.templates.layers.StronglyEntanglingLayers

        cost = catch_warn_ExpvalCost(ansatz, H, dev, interface="torch")
        loss = cost(w)
        loss.backward()
        res_torch = w.grad.numpy()

        # NumPy interface
        w = params
        ansatz = qml.templates.layers.StronglyEntanglingLayers
        cost = catch_warn_ExpvalCost(ansatz, H, dev, interface="autograd")
        dcost = qml.grad(cost, argnum=[0])
        res = dcost(w)

        assert np.allclose(res, res_tf, atol=tol, rtol=0)
        assert np.allclose(res, res_torch, atol=tol, rtol=0)
