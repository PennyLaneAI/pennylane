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

import pennylane as qp
from pennylane import numpy as pnp


def generate_cost_fn(ansatz, hamiltonian, device, **kwargs):
    """Generates a QNode and computes the expectation value of a cost Hamiltonian with respect
    to the parameters provided to an ansatz"""
    shots = kwargs.pop("shots", None)

    @qp.set_shots(shots)  # Set shots for the QNode
    @qp.qnode(device, **kwargs)
    def res(params):
        ansatz(params, wires=device.wires)
        return qp.expval(hamiltonian)

    return res


#####################################################
# Hamiltonians


H_ONE_QUBIT = np.array([[1.0, 0.5j], [-0.5j, 2.5]])

H_TWO_QUBITS = np.array(
    [[0.5, 1.0j, 0.0, -3j], [-1.0j, -1.1, 0.0, -0.1], [0.0, 0.0, -0.9, 12.0], [3j, -0.1, 12.0, 0.0]]
)

COEFFS = [(0.5, 1.2, -0.7), (2.2, -0.2, 0.0), (0.33,)]

OBSERVABLES = [
    (qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(1)),
    (qp.PauliX(0) @ qp.PauliZ(1), qp.PauliY(0) @ qp.PauliZ(1), qp.PauliZ(1)),
    (qp.Hermitian(H_TWO_QUBITS, [0, 1]),),
]

OBSERVABLES_NO_HERMITIAN = [
    (qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(1)),
    (qp.PauliX(0) @ qp.PauliZ(1), qp.PauliY(0) @ qp.PauliZ(1), qp.PauliZ(1)),
]

hamiltonians_with_expvals = [
    ((-0.6,), (qp.PauliZ(0),), [-0.6 * 1.0]),
    ((1.0,), (qp.PauliX(0),), [0.0]),
    ((0.5, 1.2), (qp.PauliZ(0), qp.PauliX(0)), [0.5 * 1.0, 0]),
    ((0.5, 1.2), (qp.PauliZ(0), qp.PauliX(1)), [0.5 * 1.0, 0]),
    ((0.5, 1.2), (qp.PauliZ(0), qp.PauliZ(0)), [0.5 * 1.0, 1.2 * 1.0]),
    ((0.5, 1.2), (qp.PauliZ(0), qp.PauliZ(1)), [0.5 * 1.0, 1.2 * 1.0]),
]

zero_hamiltonians_with_expvals = [
    ([], [], [0]),
    ((0, 0), (qp.PauliZ(0), qp.PauliZ(1)), [0]),
    ((0, 0, 0), (qp.PauliX(0) @ qp.Identity(1), qp.PauliX(0), qp.PauliX(1)), [0]),
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
    qp.Identity(wires=[0]),
    qp.PauliZ(wires=[0]),
    qp.PauliZ(wires=[1]),
    qp.PauliZ(wires=[2]),
    qp.PauliZ(wires=[3]),
    qp.PauliZ(wires=[0]) @ qp.PauliZ(wires=[1]),
    qp.PauliY(wires=[0]) @ qp.PauliX(wires=[1]) @ qp.PauliX(wires=[2]) @ qp.PauliY(wires=[3]),
    qp.PauliY(wires=[0]) @ qp.PauliY(wires=[1]) @ qp.PauliX(wires=[2]) @ qp.PauliX(wires=[3]),
    qp.PauliX(wires=[0]) @ qp.PauliX(wires=[1]) @ qp.PauliY(wires=[2]) @ qp.PauliY(wires=[3]),
    qp.PauliX(wires=[0]) @ qp.PauliY(wires=[1]) @ qp.PauliY(wires=[2]) @ qp.PauliX(wires=[3]),
    qp.PauliZ(wires=[0]) @ qp.PauliZ(wires=[2]),
    qp.PauliZ(wires=[0]) @ qp.PauliZ(wires=[3]),
    qp.PauliZ(wires=[1]) @ qp.PauliZ(wires=[2]),
    qp.PauliZ(wires=[1]) @ qp.PauliZ(wires=[3]),
    qp.PauliZ(wires=[2]) @ qp.PauliZ(wires=[3]),
]

big_hamiltonian = qp.Hamiltonian(big_hamiltonian_coeffs, big_hamiltonian_ops)

big_hamiltonian_grad = (
    np.array(
        [
            [
                [3.46944695e-17, 2.19990188e-01, -2.30793349e-02],
                [3.28242208e-17, -2.40632771e-02, -3.24974295e-04],
                [3.81639165e-17, 5.36985274e-02, 5.09078210e-02],
                [4.16333634e-17, -1.65286612e-01, 1.00566407e-01],
            ],
            [
                [-1.30075605e-02, 7.64413731e-02, -1.73472348e-17],
                [-3.93930424e-02, -1.41264311e-02, -3.03576608e-18],
                [1.27502468e-02, 2.53562554e-02, -1.93489132e-02],
                [-3.78744735e-02, 1.04547989e-02, 1.86649332e-02],
            ],
        ]
    ),
)

#####################################################
# Ansatz


# pylint: disable=unused-argument
def custom_fixed_ansatz(params, wires=None):
    """Custom fixed ansatz"""
    qp.RX(0.5, wires=0)
    qp.RX(-1.2, wires=1)
    qp.Hadamard(wires=0)
    qp.CNOT(wires=[0, 1])
    qp.Hadamard(wires=1)
    qp.CNOT(wires=[0, 1])


def custom_var_ansatz(params, wires=None):
    """Custom parametrized ansatz"""
    for p in params:
        qp.RX(p, wires=wires[0])

    qp.CNOT(wires=[wires[0], wires[1]])

    for p in params:
        qp.RX(-p, wires=wires[1])

    qp.CNOT(wires=[wires[0], wires[1]])


def amp_embed_and_strong_ent_layer(params, wires=None):
    """Ansatz combining amplitude embedding and
    strongly entangling layers"""
    qp.templates.embeddings.AmplitudeEmbedding(params[0], wires=wires)
    qp.templates.layers.StronglyEntanglingLayers(params[1], wires=wires)


ANSAETZE = [
    lambda params, wires=None: None,
    custom_fixed_ansatz,
    custom_var_ansatz,
    qp.templates.embeddings.AmplitudeEmbedding,
    qp.templates.layers.StronglyEntanglingLayers,
    amp_embed_and_strong_ent_layer,
]

#####################################################
# Parameters

EMPTY_PARAMS = []
VAR_PARAMS = [0.5]
EMBED_PARAMS = np.array([1 / np.sqrt(2**3)] * 2**3)
LAYER_PARAMS = np.random.random(qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=3))

CIRCUITS = [
    (lambda params, wires=None: None, EMPTY_PARAMS),
    (custom_fixed_ansatz, EMPTY_PARAMS),
    (custom_var_ansatz, VAR_PARAMS),
    (qp.templates.layers.StronglyEntanglingLayers, LAYER_PARAMS),
    (qp.templates.embeddings.AmplitudeEmbedding, EMBED_PARAMS),
    (amp_embed_and_strong_ent_layer, (EMBED_PARAMS, LAYER_PARAMS)),
]

#####################################################
# Queues

QUEUE_HAMILTONIANS_1 = [
    qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliZ(1)]),
    qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliZ(1)]),
]

QUEUE_HAMILTONIANS_2 = [
    qp.Hamiltonian([1], [qp.PauliX(0)]),
    qp.Hamiltonian([5], [qp.PauliX(0) @ qp.PauliZ(1)]),
]

QUEUES = [
    [
        qp.PauliX(0),
        qp.PauliZ(1),
        qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliZ(1)]),
        qp.PauliX(0),
        qp.Hamiltonian([1], [qp.PauliX(0)]),
        qp.Hamiltonian([2, 1], [qp.PauliX(0), qp.PauliZ(1)]),
    ],
    [
        qp.PauliX(0),
        qp.PauliZ(1),
        qp.Hamiltonian([1, 1], [qp.PauliX(0), qp.PauliZ(1)]),
        qp.PauliX(0),
        qp.PauliZ(1),
        qp.PauliX(0) @ qp.PauliZ(1),
        qp.Hamiltonian([1], [qp.PauliX(0) @ qp.PauliZ(1)]),
        qp.Hamiltonian([1, 1, 2], [qp.PauliX(0), qp.PauliZ(1), qp.PauliX(0) @ qp.PauliZ(1)]),
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
        hamiltonian = qp.Hamiltonian(coeffs, observables)
        dev = qp.device("default.qubit", wires=3)
        expval = generate_cost_fn(ansatz, hamiltonian, dev)
        assert expval(params).dtype == np.float64
        assert np.shape(expval(params)) == ()  # expval should be scalar

    @pytest.mark.parametrize(
        "coeffs, observables, expected", hamiltonians_with_expvals + zero_hamiltonians_with_expvals
    )
    def test_cost_expvals(self, coeffs, observables, expected):
        """Tests that the cost function returns correct expectation values"""
        dev = qp.device("default.qubit", wires=2)
        hamiltonian = qp.Hamiltonian(coeffs, observables)
        cost = generate_cost_fn(lambda params, **kwargs: None, hamiltonian, dev)
        assert cost([]) == sum(expected)

    # pylint: disable=protected-access
    @pytest.mark.torch
    @pytest.mark.slow
    @pytest.mark.parametrize("shots", [None, [(8000, 5)], [(8000, 5), (9000, 4)]])
    def test_optimize_torch(self, shots, seed):
        """Test that a Hamiltonian cost function is the same with and without
        grouping optimization when using the Torch interface."""

        dev = qp.device("default.qubit", wires=4)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)
        hamiltonian1.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            interface="torch",
            diff_method="parameter-shift",
        )
        cost2 = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            interface="torch",
            diff_method="parameter-shift",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        _rng = np.random.default_rng(seed)
        w = _rng.random(shape)

        with qp.Tracker(dev) as tracker:
            c1 = cost(w)

        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)

        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 5  # Number of groups in the Hamiltonian
        assert exec_no_opt == 8  # Number of wire-based groups

        assert np.allclose(c1, c2, atol=1e-1)

    # pylint: disable=protected-access
    @pytest.mark.tf
    @pytest.mark.slow
    @pytest.mark.parametrize("shots", [None, [(8000, 5)], [(8000, 5), (9000, 4)]])
    def test_optimize_tf(self, shots, seed):
        """Test that a Hamiltonian cost function is the same with and without
        grouping optimization when using the TensorFlow interface."""

        dev = qp.device("default.qubit", wires=4)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)
        hamiltonian1.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            interface="tf",
            diff_method="parameter-shift",
        )
        cost2 = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            interface="tf",
            diff_method="parameter-shift",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        _rng = np.random.default_rng(seed)
        w = _rng.random(shape)

        with qp.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 5  # Number of groups in the Hamiltonian
        assert exec_no_opt == 8  # Number of wire-based groups

        assert np.allclose(c1, c2, atol=1e-1)

    # pylint: disable=protected-access
    @pytest.mark.autograd
    @pytest.mark.slow
    @pytest.mark.parametrize("shots", [None, [(8000, 5)], [(8000, 5), (9000, 4)]])
    def test_optimize_autograd(self, shots, seed):
        """Test that a Hamiltonian cost function is the same with and without
        grouping optimization when using the autograd interface."""

        dev = qp.device("default.qubit", wires=4)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)
        hamiltonian1.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            interface="autograd",
            diff_method="parameter-shift",
            shots=shots,
        )
        cost2 = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            interface="autograd",
            diff_method="parameter-shift",
            shots=shots,
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        _rng = np.random.default_rng(seed)
        w = _rng.random(shape)

        with qp.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 5  # Number of groups in the Hamiltonian
        assert exec_no_opt == 8

        assert np.allclose(c1, c2, atol=1e-1)

    # pylint: disable=protected-access
    @pytest.mark.autograd
    def test_optimize_multiple_terms_autograd(self, seed):
        """Test that a Hamiltonian cost function is the same with and without
        grouping optimization when using the autograd interface, even when
        there are non-unique Hamiltonian terms."""

        dev = qp.device("default.qubit", wires=5)
        obs = [
            qp.PauliZ(wires=[2]) @ qp.PauliZ(wires=[4]),  # <---- These two terms
            qp.PauliZ(wires=[4]) @ qp.PauliZ(wires=[2]),  # <---- are equal
            qp.PauliZ(wires=[1]),
            qp.PauliZ(wires=[2]),
            qp.PauliZ(wires=[1]) @ qp.PauliZ(wires=[2]),
            qp.PauliZ(wires=[2]) @ qp.PauliZ(wires=[0]),
            qp.PauliZ(wires=[3]) @ qp.PauliZ(wires=[1]),
            qp.PauliZ(wires=[4]) @ qp.PauliZ(wires=[3]),
        ]

        coeffs = (np.random.rand(len(obs)) - 0.5) * 2
        hamiltonian1 = qp.Hamiltonian(coeffs, obs)
        hamiltonian2 = qp.Hamiltonian(coeffs, obs)
        hamiltonian1.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            interface="autograd",
            diff_method="parameter-shift",
        )
        cost2 = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            interface="autograd",
            diff_method="parameter-shift",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=5)
        _rng = np.random.default_rng(seed)
        w = _rng.random(shape)

        with qp.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 1  # Number of groups in the Hamiltonian
        assert exec_no_opt == 4  # number of wire-based groups

        assert np.allclose(c1, c2)

    # pylint: disable=protected-access
    @pytest.mark.torch
    def test_optimize_multiple_terms_torch(self, seed):
        """Test that a Hamiltonian cost function is the same with and without
        grouping optimization when using the Torch interface, even when there
        are non-unique Hamiltonian terms."""

        dev = qp.device("default.qubit", wires=5)
        obs = [
            qp.PauliZ(wires=[2]) @ qp.PauliZ(wires=[4]),  # <---- These two terms
            qp.PauliZ(wires=[4]) @ qp.PauliZ(wires=[2]),  # <---- are equal
            qp.PauliZ(wires=[1]),
            qp.PauliZ(wires=[2]),
            qp.PauliZ(wires=[1]) @ qp.PauliZ(wires=[2]),
            qp.PauliZ(wires=[2]) @ qp.PauliZ(wires=[0]),
            qp.PauliZ(wires=[3]) @ qp.PauliZ(wires=[1]),
            qp.PauliZ(wires=[4]) @ qp.PauliZ(wires=[3]),
        ]

        coeffs = (np.random.rand(len(obs)) - 0.5) * 2
        hamiltonian1 = qp.Hamiltonian(coeffs, obs)
        hamiltonian2 = qp.Hamiltonian(coeffs, obs)
        hamiltonian1.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            interface="torch",
            diff_method="parameter-shift",
        )
        cost2 = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            interface="torch",
            diff_method="parameter-shift",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=5)
        _rng = np.random.default_rng(seed)
        w = _rng.random(shape)

        with qp.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 1  # Number of groups in the Hamiltonian
        assert exec_no_opt == 4

        assert np.allclose(c1, c2)

    # pylint: disable=protected-access
    @pytest.mark.tf
    def test_optimize_multiple_terms_tf(self, seed):
        """Test that a Hamiltonian cost function is the same with and without
        grouping optimization when using the TensorFlow interface, even when
        there are non-unique Hamiltonian terms."""

        dev = qp.device("default.qubit", wires=5)
        obs = [
            qp.PauliZ(wires=[2]) @ qp.PauliZ(wires=[4]),  # <---- These two terms
            qp.PauliZ(wires=[4]) @ qp.PauliZ(wires=[2]),  # <---- are equal
            qp.PauliZ(wires=[1]),
            qp.PauliZ(wires=[2]),
            qp.PauliZ(wires=[1]) @ qp.PauliZ(wires=[2]),
            qp.PauliZ(wires=[2]) @ qp.PauliZ(wires=[0]),
            qp.PauliZ(wires=[3]) @ qp.PauliZ(wires=[1]),
            qp.PauliZ(wires=[4]) @ qp.PauliZ(wires=[3]),
        ]

        coeffs = (np.random.rand(len(obs)) - 0.5) * 2
        hamiltonian1 = qp.Hamiltonian(coeffs, obs)
        hamiltonian2 = qp.Hamiltonian(coeffs, obs)
        hamiltonian1.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            interface="tf",
            diff_method="parameter-shift",
        )
        cost2 = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            interface="tf",
            diff_method="parameter-shift",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=5)
        _rng = np.random.default_rng(seed)
        w = _rng.random(shape)

        with qp.Tracker(dev) as tracker:
            c1 = cost(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            c2 = cost2(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_opt == 1  # Number of groups in the Hamiltonian
        assert exec_no_opt == 4

        assert np.allclose(c1, c2)

    # pylint: disable=protected-access
    @pytest.mark.autograd
    def test_optimize_grad(self):
        """Test that the gradient of a Hamiltonian cost function is accessible
        and correct when using observable grouping optimization and the
        autograd interface."""
        dev = qp.device("default.qubit", wires=4)

        hamiltonian1 = copy.copy(big_hamiltonian)
        hamiltonian2 = copy.copy(big_hamiltonian)
        hamiltonian1.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian1,
            dev,
            diff_method="parameter-shift",
        )
        cost2 = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian2,
            dev,
            diff_method="parameter-shift",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        # TODO: This is another case of a magic number in the sense that no other number allows
        #       this test to pass. This is likely because the expected `big_hamiltonian_grad`
        #       was calculated using this exact seed. This test needs to be revisited.
        _rng = pnp.random.default_rng(1967)
        w = _rng.uniform(low=0, high=2 * np.pi, size=shape, requires_grad=True)

        with qp.Tracker(dev) as tracker:
            dc = qp.grad(cost)(w)
        exec_opt = tracker.totals["executions"]

        with tracker:
            dc2 = qp.grad(cost2)(w)
        exec_no_opt = tracker.totals["executions"]

        assert exec_no_opt > exec_opt
        assert np.allclose(dc, big_hamiltonian_grad)
        assert np.allclose(dc2, big_hamiltonian_grad)

    @pytest.mark.autograd
    @pytest.mark.parametrize("opt", [True, False])
    def test_grad_zero_hamiltonian(self, opt):
        """Test that the gradient of a Hamiltonian cost function is accessible
        and correct when using observable grouping optimization and the
        autograd interface, with a zero Hamiltonian."""
        dev = qp.device("default.qubit", wires=4)
        hamiltonian = qp.Hamiltonian([0], [qp.PauliX(0)])
        if opt:
            hamiltonian.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian,
            dev,
            diff_method="parameter-shift",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        w = pnp.random.random(shape, requires_grad=True)

        dc = qp.grad(cost)(w)
        assert np.allclose(dc, 0)

    @pytest.mark.torch
    @pytest.mark.slow
    def test_optimize_grad_torch(self):
        """Test that the gradient of a Hamiltonian cost function is accessible
        and correct when using observable grouping optimization and the Torch
        interface."""
        import torch

        dev = qp.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian
        hamiltonian.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers,
            hamiltonian,
            dev,
            interface="torch",
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        # TODO: This is another case of a magic number in the sense that no other number allows
        #       this test to pass. This is likely because the expected `big_hamiltonian_grad`
        #       was calculated using this exact seed. This test needs to be revisited.
        _rng = np.random.default_rng(1967)
        w = _rng.uniform(low=0, high=2 * np.pi, size=shape)
        w = torch.tensor(w, requires_grad=True)

        res = cost(w)
        res.backward()
        dc = w.grad.detach().numpy()

        assert np.allclose(dc, big_hamiltonian_grad)

    @pytest.mark.tf
    @pytest.mark.slow
    def test_optimize_grad_tf(self):
        """Test that the gradient of a Hamiltonian cost function is accessible
        and correct when using observable grouping optimization and the
        TensorFlow interface."""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian
        hamiltonian.compute_grouping()

        cost = generate_cost_fn(
            qp.templates.StronglyEntanglingLayers, hamiltonian, dev, interface="tf"
        )

        shape = qp.templates.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        # TODO: This is another case of a magic number in the sense that no other number allows
        #       this test to pass. This is likely because the expected `big_hamiltonian_grad`
        #       was calculated using this exact seed. This test needs to be revisited.
        _rng = np.random.default_rng(1967)
        w = _rng.uniform(low=0, high=2 * np.pi, size=shape)
        w = tf.Variable(w)

        with tf.GradientTape() as tape:
            res = cost(w)

        dc = tape.gradient(res, w).numpy()

        assert np.allclose(dc, big_hamiltonian_grad)


# Test data
rng = np.random.default_rng(1967)
_shape = qp.templates.StronglyEntanglingLayers.shape(2, 4)
PARAMS = rng.uniform(low=0, high=2 * np.pi, size=_shape)


class TestNewVQE:
    """Test the new VQE syntax of passing the Hamiltonian as an observable."""

    # pylint: disable=cell-var-from-loop
    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES_NO_HERMITIAN)
    def test_circuits_evaluate(self, ansatz, observables, params, tol):
        """Tests simple VQE evaluations."""

        coeffs = [1.0] * len(observables)
        dev = qp.device("default.qubit", wires=3)
        H = qp.Hamiltonian(coeffs, observables)

        # pass H directly
        @qp.qnode(dev)
        def circuit():
            ansatz(params, wires=range(3))
            return qp.expval(H)

        res = circuit()

        res_expected = []
        for obs in observables:

            @qp.qnode(dev)
            def separate_circuit():
                ansatz(params, wires=range(3))
                return qp.expval(obs)

            res_expected.append(separate_circuit())

        res_expected = np.sum([c * r for c, r in zip(coeffs, res_expected)])

        assert np.isclose(res, res_expected, atol=tol)

    def test_acting_on_subcircuit(self, tol):
        """Tests a VQE circuit where the observable does not act on all wires."""
        dev = qp.device("default.qubit", wires=3)
        coeffs = [1.0, 1.0, 1.0]

        w = np.random.random(qp.templates.StronglyEntanglingLayers.shape(n_layers=1, n_wires=2))

        observables1 = [qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(1)]
        H1 = qp.Hamiltonian(coeffs, observables1)

        @qp.qnode(dev)
        def circuit1():
            qp.templates.StronglyEntanglingLayers(w, wires=range(2))
            return qp.expval(H1)

        observables2 = [qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(1) @ qp.Identity(2)]
        H2 = qp.Hamiltonian(coeffs, observables2)

        @qp.qnode(dev)
        def circuit2():
            qp.templates.StronglyEntanglingLayers(w, wires=range(2))
            return qp.expval(H2)

        res1 = circuit1()
        res2 = circuit2()

        assert np.allclose(res1, res2, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("shots, dim", [([(1000, 2)], 2), ([30, 30], 2), ([2, 3, 4], 3)])
    def test_shot_distribution(self, shots, dim, seed):
        """Tests that distributed shots work with the new VQE design."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qp.device("default.qubit", wires=2)

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit(weights, coeffs):
            qp.templates.StronglyEntanglingLayers(weights, wires=[0, 1])
            H = qp.Hamiltonian(coeffs, obs)
            return qp.expval(H)

        obs = [qp.PauliZ(0), qp.PauliX(0) @ qp.PauliZ(1)]
        coeffs = np.array([0.1, 0.2])
        key = jax.random.PRNGKey(seed)
        weights = jax.random.uniform(key, [2, 2, 3])

        res = circuit(weights, coeffs)
        grad = jax.jacobian(circuit, argnums=[1])(weights, coeffs)
        assert len(res) == dim
        assert qp.math.shape(grad) == (dim, 1, 2)

    def test_circuit_drawer(self):
        """Test that the circuit drawer displays Hamiltonians well."""
        dev = qp.device("default.qubit", wires=3)
        coeffs = [1.0, 1.0, 1.0]
        observables1 = [qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(2)]
        H1 = qp.Hamiltonian(coeffs, observables1)

        @qp.qnode(dev)
        def circuit1():
            qp.Hadamard(wires=0)
            return qp.expval(H1)

        res = qp.draw(circuit1)()
        expected = "0: ──H─┤ ╭<𝓗(1.00,1.00,1.00)>\n2: ────┤ ╰<𝓗(1.00,1.00,1.00)>"
        assert res == expected

    def test_multiple_expvals(self):
        """Tests that more than one Hamiltonian expval can be evaluated."""

        coeffs = [1.0, 1.0, 1.0]
        dev = qp.device("default.qubit", wires=4)
        H1 = qp.Hamiltonian(coeffs, [qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(1)])
        H2 = qp.Hamiltonian(coeffs, [qp.PauliZ(2), qp.PauliY(2), qp.PauliZ(3)])
        w = PARAMS

        @qp.qnode(dev)
        def circuit():
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H1), qp.expval(H2)

        res = circuit()

        @qp.qnode(dev)
        def circuit1():
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H1)

        @qp.qnode(dev)
        def circuit2():
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H2)

        assert res[0] == circuit1()
        assert res[1] == circuit2()

    def test_multiple_expvals_same_wires(self):
        """Tests that more than one Hamiltonian expval can be evaluated."""

        coeffs = [1.0, 1.0, 1.0]
        dev = qp.device("default.qubit", wires=4)
        H1 = qp.Hamiltonian(coeffs, [qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(1)])
        w = PARAMS

        @qp.qnode(dev)
        def circuit():
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H1), qp.expval(H1)

        res = circuit()

        @qp.qnode(dev)
        def circuit1():
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H1)

        assert res[0] == circuit1()
        assert res[1] == circuit1()

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "best"])
    def test_grad_autograd(self, diff_method, tol):
        """Tests the VQE gradient in the autograd interface."""
        dev = qp.device("default.qubit", wires=4)
        H = big_hamiltonian
        w = pnp.array(PARAMS, requires_grad=True)

        @qp.qnode(dev, diff_method=diff_method)
        def circuit(w):
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H)

        dc = qp.grad(circuit)(w)
        assert np.allclose(dc, big_hamiltonian_grad, atol=tol)

    @pytest.mark.autograd
    def test_grad_zero_hamiltonian(self, tol):
        """Tests the VQE gradient for a "zero" Hamiltonian."""
        dev = qp.device("default.qubit", wires=4)
        H = qp.Hamiltonian([0], [qp.PauliX(0)])
        w = pnp.array(PARAMS, requires_grad=True)

        @qp.qnode(dev, diff_method="parameter-shift")
        def circuit(w):
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H)

        dc = qp.grad(circuit)(w)
        assert np.allclose(dc, 0, atol=tol)

    @pytest.mark.torch
    @pytest.mark.slow
    def test_grad_torch(self, tol):
        """Tests VQE gradients in the torch interface."""
        import torch

        dev = qp.device("default.qubit", wires=4)
        H = big_hamiltonian

        @qp.qnode(dev)
        def circuit(w):
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H)

        w = torch.tensor(PARAMS, requires_grad=True)

        res = circuit(w)
        res.backward()  # pylint:disable=no-member
        dc = w.grad.detach().numpy()

        assert np.allclose(dc, big_hamiltonian_grad, atol=tol)

    @pytest.mark.tf
    def test_grad_tf(self, tol):
        """Tests VQE gradients in the tf interface."""
        import tensorflow as tf

        dev = qp.device("default.qubit", wires=4)
        H = big_hamiltonian

        @qp.qnode(dev)
        def circuit(w):
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H)

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

        dev = qp.device("default.qubit", wires=4)
        H = big_hamiltonian

        w = jnp.array(PARAMS)

        @qp.qnode(dev)
        def circuit(w):
            qp.templates.StronglyEntanglingLayers(w, wires=range(4))
            return qp.expval(H)

        dc = jax.grad(circuit)(w)
        assert np.allclose(dc, big_hamiltonian_grad, atol=tol)

    def test_specs(self):
        """Test that the specs of a VQE circuit can be computed"""
        dev = qp.device("default.qubit", wires=2)
        H = qp.Hamiltonian([0.1, 0.2], [qp.PauliZ(0), qp.PauliZ(0) @ qp.PauliX(1)])

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(H)

        res = qp.specs(circuit)()

        assert res["resources"] == qp.resource.SpecsResources(
            num_allocs=2,
            gate_types={"Hadamard": 1, "CNOT": 1},
            gate_sizes={1: 1, 2: 1},
            measurements={"expval(Hamiltonian(num_wires=2, num_terms=2))": 1},
            depth=2,
        )


class TestInterfaces:
    """Tests for VQE with interfaces."""

    @pytest.mark.autograd
    @pytest.mark.parametrize("interface", ["autograd"])
    def test_gradient_autograd(self, tol, interface):
        """Tests for the Autograd interface (and the NumPy interface for
        backward compatibility)"""
        dev = qp.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qp.RX(params[0], wires=0)
            qp.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qp.PauliX(0), qp.PauliY(0)]

        H = qp.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = np.array([a, b])

        cost = generate_cost_fn(ansatz, H, dev, interface=interface)
        dcost = qp.grad(cost, argnums=[0])
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

        dev = qp.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qp.RX(params[0], wires=0)
            qp.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qp.PauliX(0), qp.PauliY(0)]

        H = qp.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = torch.autograd.Variable(torch.tensor([a, b]), requires_grad=True)

        cost = generate_cost_fn(ansatz, H, dev, interface="torch")
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

        dev = qp.device("default.qubit", wires=1)

        def ansatz(params, **kwargs):
            qp.RX(params[0], wires=0)
            qp.RY(params[1], wires=0)

        coeffs = [0.2, 0.5]
        observables = [qp.PauliX(0), qp.PauliY(0)]

        H = qp.Hamiltonian(coeffs, observables)
        a, b = 0.54, 0.123
        params = tf.Variable([a, b], dtype=tf.float64)
        cost = generate_cost_fn(ansatz, H, dev, interface="tf")

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
        import torch

        dev = qp.device("default.qubit", wires=2)

        coeffs = [0.2, 0.5]
        observables = [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliY(0)]

        H = qp.Hamiltonian(coeffs, observables)

        shape = qp.templates.StronglyEntanglingLayers.shape(3, 2)
        params = np.random.uniform(low=0, high=2 * np.pi, size=shape)

        # Torch interface
        w = torch.tensor(params, requires_grad=True)
        w = torch.autograd.Variable(w, requires_grad=True)
        ansatz = qp.templates.layers.StronglyEntanglingLayers

        cost = generate_cost_fn(ansatz, H, dev, interface="torch")
        loss = cost(w)
        loss.backward()
        res_torch = w.grad.numpy()

        # NumPy interface
        w = params
        ansatz = qp.templates.layers.StronglyEntanglingLayers
        cost = generate_cost_fn(ansatz, H, dev, interface="autograd")
        dcost = qp.grad(cost, argnums=[0])
        res = dcost(w)

        assert np.allclose(res, res_torch, atol=tol, rtol=0)
