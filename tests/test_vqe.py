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
import numpy as np
import pytest

import pennylane as qml
from pennylane.wires import Wires
from pennylane.devices import DefaultQubit, DefaultMixed

try:
    import torch
except ImportError as e:
    pass


try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        tf.enable_eager_execution()

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
    ((0.6,), (qml.PauliX(0) @ qml.PauliX(1),)),
    ((0.5, -1.6), (qml.PauliX(0), qml.PauliY(1))),
    ((0.5, -1.6), (qml.PauliX(1), qml.PauliY(1))),
    ((0.5, -1.6), (qml.PauliX("a"), qml.PauliY("b"))),
    ((1.1, -0.4, 0.333), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 2), qml.PauliZ(2))),
    ((-0.4, 0.15), (qml.Hermitian(H_TWO_QUBITS, [0, 2]), qml.PauliZ(1))),
    ([1.5, 2.0], [qml.PauliZ(0), qml.PauliY(2)]),
    (np.array([-0.1, 0.5]), [qml.Hermitian(H_TWO_QUBITS, [0, 1]), qml.PauliY(0)]),
    ((0.5, 1.2), (qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1))),
]

valid_hamiltonians_str = [
    "(1.0) [Hermitian0'1]",
    "(-0.8) [Z0]",
    "(0.6) [X0 X1]",
    "(0.5) [X0]\n+ (-1.6) [Y1]",
    "(0.5) [X1]\n+ (-1.6) [Y1]",
    "(0.5) [Xa]\n+ (-1.6) [Yb]",
    "(1.1) [X0]\n+ (-0.4) [Hermitian2]\n+ (0.333) [Z2]",
    "(-0.4) [Hermitian0'2]\n+ (0.15) [Z1]",
    "(1.5) [Z0]\n+ (2.0) [Y2]",
    "(-0.1) [Hermitian0'1]\n+ (0.5) [Y0]",
    "(0.5) [X0]\n+ (1.2) [X0 X1]",
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

simplify_hamiltonians = [
    (
        qml.Hamiltonian([1, 1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0), qml.PauliX(1)]),
        qml.Hamiltonian([2, 1], [qml.PauliX(0), qml.PauliX(1)]),
    ),
    (
        qml.Hamiltonian(
            [-1, 1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0), qml.PauliX(1)]
        ),
        qml.Hamiltonian([1], [qml.PauliX(1)]),
    ),
    (
        qml.Hamiltonian(
            [1, 0.5],
            [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.Identity(2) @ qml.PauliX(0)],
        ),
        qml.Hamiltonian([1.5], [qml.PauliX(0) @ qml.PauliY(1)]),
    ),
    (
        qml.Hamiltonian(
            [1, 1, 0.5],
            [
                qml.Hermitian(np.array([[1, 0], [0, -1]]), "a"),
                qml.PauliX("b") @ qml.PauliY(1.3),
                qml.PauliY(1.3) @ qml.Identity(-0.9) @ qml.PauliX("b"),
            ],
        ),
        qml.Hamiltonian(
            [1, 1.5],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "a"), qml.PauliX("b") @ qml.PauliY(1.3)],
        ),
    ),
]

equal_hamiltonians = [
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0)]),
        True,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliY(2) @ qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliY(2) @ qml.Identity(1)]),
        True,
    ),
    (
        qml.Hamiltonian(
            [1, 1, 1], [qml.PauliX(0) @ qml.Identity(1), qml.PauliZ(0), qml.Identity(1)]
        ),
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(0)]),
        False,
    ),
    (qml.Hamiltonian([1], [qml.PauliZ(0) @ qml.PauliX(1)]), qml.PauliZ(0) @ qml.PauliX(1), True),
    (qml.Hamiltonian([1], [qml.PauliZ(0)]), qml.PauliZ(0), True),
    (
        qml.Hamiltonian(
            [1, 1, 1],
            [
                qml.Hermitian(np.array([[1, 0], [0, -1]]), "b") @ qml.Identity(7),
                qml.PauliZ(3),
                qml.Identity(1.2),
            ],
        ),
        qml.Hamiltonian(
            [1, 1, 1],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), qml.PauliZ(3), qml.Identity(1.2)],
        ),
        True,
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliZ(3) @ qml.Identity(1.2), qml.PauliZ(3)]),
        qml.Hamiltonian([2], [qml.PauliZ(3)]),
        True,
    ),
]

add_hamiltonians = [
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([0.5, 0.3, 1], [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]),
        qml.Hamiltonian(
            [1.5, 1.2, 1.1, 0.3], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]
        ),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [0.5, 0.3, 1.6], [qml.PauliX(0), qml.PauliX(1) @ qml.PauliX(0), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [1.6, 0.2, 2.3, 0.5],
            [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2), qml.PauliX(0)],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([1.5, 1.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.PauliX(0) @ qml.Identity(1),
        qml.Hamiltonian([2, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hadamard(1),
        qml.Hamiltonian(
            [1.3, 1.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX("b"), qml.PauliZ(3.1), qml.PauliX(1.6)]),
        qml.PauliX("b") @ qml.Identity(5),
        qml.Hamiltonian([2, 1.2, 0.1], [qml.PauliX("b"), qml.PauliZ(3.1), qml.PauliX(1.6)]),
    ),
]

sub_hamiltonians = [
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.Hamiltonian([0.5, 0.3, 1.6], [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]),
        qml.Hamiltonian(
            [0.5, 1.2, -1.5, -0.3], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2), qml.PauliX(1)]
        ),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 1], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [0.5, 0.3, 1], [qml.PauliX(0), qml.PauliX(1) @ qml.PauliX(0), qml.PauliX(2)]
        ),
        qml.Hamiltonian(
            [1, 0.2, -0.5], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(0)]
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliX(0), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)]),
        qml.PauliX(0) @ qml.Identity(1),
        qml.Hamiltonian([1.2, 0.1], [qml.PauliZ(1), qml.PauliX(2)]),
    ),
    (
        qml.Hamiltonian(
            [1.3, 0.2, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
        qml.Hadamard(1),
        qml.Hamiltonian(
            [1.3, -0.8, 0.7], [qml.PauliX(0) @ qml.PauliX(1), qml.Hadamard(1), qml.PauliX(2)]
        ),
    ),
    (
        qml.Hamiltonian([1, 1.2, 0.1], [qml.PauliX("b"), qml.PauliZ(3.1), qml.PauliX(1.6)]),
        qml.PauliX("b") @ qml.Identity(1),
        qml.Hamiltonian([1.2, 0.1], [qml.PauliZ(3.1), qml.PauliX(1.6)]),
    ),
]

mul_hamiltonians = [
    (
        3,
        qml.Hamiltonian([1.5, 0.5], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.Hamiltonian([4.5, 1.5], [qml.PauliX(0), qml.PauliZ(1)]),
    ),
    (
        -1.3,
        qml.Hamiltonian([1, -0.3], [qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]),
        qml.Hamiltonian([-1.3, 0.39], [qml.PauliX(0), qml.PauliZ(1) @ qml.PauliZ(2)]),
    ),
    (
        -1.3,
        qml.Hamiltonian(
            [1, -0.3],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), qml.PauliZ(23) @ qml.PauliZ(0)],
        ),
        qml.Hamiltonian(
            [-1.3, 0.39],
            [qml.Hermitian(np.array([[1, 0], [0, -1]]), "b"), qml.PauliZ(23) @ qml.PauliZ(0)],
        ),
    ),
]

matmul_hamiltonians = [
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.Hamiltonian([0.5, 0.5], [qml.PauliZ(2), qml.PauliZ(3)]),
        qml.Hamiltonian(
            [0.5, 0.5, 0.5, 0.5],
            [
                qml.PauliX(0) @ qml.PauliZ(2),
                qml.PauliX(0) @ qml.PauliZ(3),
                qml.PauliZ(1) @ qml.PauliZ(2),
                qml.PauliZ(1) @ qml.PauliZ(3),
            ],
        ),
    ),
    (
        qml.Hamiltonian([0.5, 0.25], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0)]),
        qml.Hamiltonian([1, 1], [qml.PauliX(3) @ qml.PauliZ(2), qml.PauliZ(2)]),
        qml.Hamiltonian(
            [0.5, 0.5, 0.25, 0.25],
            [
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(3) @ qml.PauliZ(2),
                qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliZ(2),
                qml.PauliZ(0) @ qml.PauliX(3) @ qml.PauliZ(2),
                qml.PauliZ(0) @ qml.PauliZ(2),
            ],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX("b"), qml.Hermitian(np.array([[1, 0], [0, -1]]), 0)]),
        qml.Hamiltonian([2, 2], [qml.PauliZ(1.2), qml.PauliY("c")]),
        qml.Hamiltonian(
            [2, 2, 2, 2],
            [
                qml.PauliX("b") @ qml.PauliZ(1.2),
                qml.PauliX("b") @ qml.PauliY("c"),
                qml.Hermitian(np.array([[1, 0], [0, -1]]), 0) @ qml.PauliZ(1.2),
                qml.Hermitian(np.array([[1, 0], [0, -1]]), 0) @ qml.PauliY("c"),
            ],
        ),
    ),
    (
        qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)]),
        qml.PauliX(2),
        qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.PauliX(2), qml.PauliZ(1) @ qml.PauliX(2)]),
    ),
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

big_hamiltonian = qml.vqe.Hamiltonian(big_hamiltonian_coeffs, big_hamiltonian_ops)

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
        m.setattr(
            qml.Device, "_capabilities", {"supports_tensor_observables": True, "model": "qubit"}
        )
        m.setattr(qml.Device, "operations", ["RX", "Rot", "CNOT", "Hadamard", "QubitStateVector"])
        m.setattr(
            qml.Device, "observables", ["PauliX", "PauliY", "PauliZ", "Hadamard", "Hermitian"]
        )
        m.setattr(qml.Device, "short_name", "MockDevice")
        m.setattr(qml.Device, "expval", lambda self, x, y, z: 1)
        m.setattr(qml.Device, "var", lambda self, x, y, z: 2)
        m.setattr(qml.Device, "sample", lambda self, x, y, z: 3)
        m.setattr(qml.Device, "apply", lambda self, x, y, z: None)

        def get_device(wires=1):
            return qml.Device(wires=wires)

        yield get_device


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

    @pytest.mark.parametrize("coeffs", [[0.2, -1j], [0.5j, 2 - 1j]])
    def test_hamiltonian_complex(self, coeffs):
        """Tests that an exception is raised when
        a complex Hamiltonian is given"""
        obs = [qml.PauliX(0), qml.PauliZ(1)]

        with pytest.raises(ValueError, match="coefficients are not real-valued"):
            H = qml.vqe.Hamiltonian(coeffs, obs)

    @pytest.mark.parametrize(
        "obs", [[qml.PauliX(0), qml.CNOT(wires=[0, 1])], [qml.PauliZ, qml.PauliZ(0)]]
    )
    def test_hamiltonian_invalid_observables(self, obs):
        """Tests that an exception is raised when
        a complex Hamiltonian is given"""
        coeffs = [0.1, 0.2]

        with pytest.raises(ValueError, match="observables are not valid"):
            H = qml.vqe.Hamiltonian(coeffs, obs)

    @pytest.mark.parametrize("coeffs, ops", valid_hamiltonians)
    def test_hamiltonian_wires(self, coeffs, ops):
        """Tests that the Hamiltonian object has correct wires."""
        H = qml.vqe.Hamiltonian(coeffs, ops)
        assert set(H.wires) == set([w for op in H.ops for w in op.wires])

    @pytest.mark.parametrize("terms, string", zip(valid_hamiltonians, valid_hamiltonians_str))
    def test_hamiltonian_str(self, terms, string):
        """Tests that the __str__ function for printing is correct"""
        H = qml.vqe.Hamiltonian(*terms)
        assert H.__str__() == string

    @pytest.mark.parametrize(("old_H", "new_H"), simplify_hamiltonians)
    def test_simplify(self, old_H, new_H):
        """Tests the simplify method"""
        old_H.simplify()
        assert old_H.compare(new_H)

    def test_data(self):
        """Tests the obs_data method"""

        H = qml.Hamiltonian(
            [1, 1, 0.5],
            [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(2) @ qml.Identity(1)],
        )
        data = H._obs_data()

        assert data == {
            (1, frozenset([("PauliZ", Wires(0), ())])),
            (1, frozenset([("PauliZ", Wires(0), ()), ("PauliX", Wires(1), ())])),
            (0.5, frozenset([("PauliX", Wires(2), ())])),
        }

    def test_hamiltonian_equal_error(self):
        """Tests that the correct error is raised when compare() is called on invalid type"""

        H = qml.Hamiltonian([1], [qml.PauliZ(0)])
        with pytest.raises(
            ValueError,
            match=r"Can only compare a Hamiltonian, and a Hamiltonian/Observable/Tensor.",
        ):
            H.compare([[1, 0], [0, -1]])

    @pytest.mark.parametrize(("H1", "H2", "res"), equal_hamiltonians)
    def test_hamiltonian_equal(self, H1, H2, res):
        """Tests that equality can be checked between Hamiltonians"""
        assert H1.compare(H2) == res

    @pytest.mark.parametrize(("H1", "H2", "H"), add_hamiltonians)
    def test_hamiltonian_add(self, H1, H2, H):
        """Tests that Hamiltonians are added correctly"""
        assert H.compare(H1 + H2)

    @pytest.mark.parametrize(("coeff", "H", "res"), mul_hamiltonians)
    def test_hamiltonian_mul(self, coeff, H, res):
        """Tests that scalars and Hamiltonians are multiplied correctly"""
        assert res.compare(coeff * H)
        assert res.compare(H * coeff)

    @pytest.mark.parametrize(("H1", "H2", "H"), sub_hamiltonians)
    def test_hamiltonian_sub(self, H1, H2, H):
        """Tests that Hamiltonians are subtracted correctly"""
        assert H.compare(H1 - H2)

    @pytest.mark.parametrize(("H1", "H2", "H"), matmul_hamiltonians)
    def test_hamiltonian_matmul(self, H1, H2, H):
        """Tests that Hamiltonians are tensored correctly"""
        assert H.compare(H1 @ H2)

    @pytest.mark.parametrize(("H1", "H2", "H"), add_hamiltonians)
    def test_hamiltonian_iadd(self, H1, H2, H):
        """Tests that Hamiltonians are added inline correctly"""
        H1 += H2
        assert H.compare(H1)

    @pytest.mark.parametrize(("coeff", "H", "res"), mul_hamiltonians)
    def test_hamiltonian_imul(self, coeff, H, res):
        """Tests that scalars and Hamiltonians are multiplied inline correctly"""
        H *= coeff
        assert res.compare(H)

    @pytest.mark.parametrize(("H1", "H2", "H"), sub_hamiltonians)
    def test_hamiltonian_isub(self, H1, H2, H):
        """Tests that Hamiltonians are subtracted inline correctly"""
        H1 -= H2
        assert H.compare(H1)

    def test_arithmetic_errors(self):
        """Tests that the arithmetic operations thrown the correct errors"""
        H = qml.Hamiltonian([1], [qml.PauliZ(0)])
        A = [[1, 0], [0, -1]]
        with pytest.raises(ValueError, match="Cannot tensor product Hamiltonian"):
            H @ A
        with pytest.raises(ValueError, match="Cannot add Hamiltonian"):
            H + A
        with pytest.raises(ValueError, match="Cannot multiply Hamiltonian"):
            H * A
        with pytest.raises(ValueError, match="Cannot subtract"):
            H - A
        with pytest.raises(ValueError, match="Cannot add Hamiltonian"):
            H += A
        with pytest.raises(ValueError, match="Cannot multiply Hamiltonian"):
            H *= A
        with pytest.raises(ValueError, match="Cannot subtract"):
            H -= A


@pytest.mark.usefixtures("tape_mode")
class TestVQE:
    """Test the core functionality of the VQE module"""

    @pytest.mark.parametrize("ansatz", ANSAETZE)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_valid_init(self, ansatz, observables, mock_device):
        """Tests that a collection of circuits is properly created by vqe.circuits"""
        dev = mock_device()
        circuits = qml.map(ansatz, observables, device=dev)

        assert len(circuits) == len(observables)
        assert all(callable(c) for c in circuits)
        assert all(c.device == dev for c in circuits)
        assert all(hasattr(c, "jacobian") for c in circuits)

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_evaluate(self, ansatz, observables, params, mock_device, seed):
        """Tests that the circuits returned by ``vqe.circuits`` evaluate properly"""
        dev = mock_device(wires=3)
        circuits = qml.map(ansatz, observables, device=dev)
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
            qml.map(ansatz, obs, device=mock_device())

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_circuits_no_ansatz(self, ansatz, observables, mock_device):
        """Tests that an exception is raised when no valid ansatz is supplied to vqe.circuits"""
        with pytest.raises(ValueError, match="not a callable function"):
            qml.map(ansatz, observables, device=mock_device())

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
        expval = qml.ExpvalCost(ansatz, hamiltonian, dev)
        assert type(expval(params)) == np.float64
        assert np.shape(expval(params)) == ()  # expval should be scalar

    @pytest.mark.parametrize("coeffs, observables, expected", hamiltonians_with_expvals)
    def test_cost_expvals(self, coeffs, observables, expected):
        """Tests that the cost function returns correct expectation values"""
        dev = qml.device("default.qubit", wires=2)
        hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
        cost = qml.ExpvalCost(lambda params, **kwargs: None, hamiltonian, dev)
        assert cost([]) == sum(expected)

    @pytest.mark.parametrize("ansatz", JUNK_INPUTS)
    def test_cost_invalid_ansatz(self, ansatz, mock_device):
        """Tests that the cost function raises an exception if the ansatz is not valid"""
        hamiltonian = qml.vqe.Hamiltonian((1.0,), [qml.PauliZ(0)])
        with pytest.raises(ValueError, match="not a callable function."):
            cost = qml.ExpvalCost(4, hamiltonian, mock_device())

    @pytest.mark.parametrize("coeffs, observables, expected", hamiltonians_with_expvals)
    def test_passing_kwargs(self, coeffs, observables, expected):
        """Test that the step size and order used for the finite differences
        differentiation method were passed to the QNode instances using the
        keyword arguments."""
        dev = qml.device("default.qubit", wires=2)
        hamiltonian = qml.vqe.Hamiltonian(coeffs, observables)
        cost = qml.ExpvalCost(lambda params, **kwargs: None, hamiltonian, dev, h=123, order=2)

        # Checking that the qnodes contain the step size and order
        for qnode in cost.qnodes:
            assert qnode.h == 123
            assert qnode.order == 2

    def test_optimize_outside_tape_mode(self):
        """Test that an error is raised if observable optimization is requested outside of tape
        mode."""
        if qml.tape_mode_active():
            pytest.skip("This test is only intended for non-tape mode")

        dev = qml.device("default.qubit", wires=2)
        hamiltonian = qml.vqe.Hamiltonian([1], [qml.PauliZ(0)])

        with pytest.raises(ValueError, match="Observable optimization is only supported in tape"):
            qml.ExpvalCost(lambda params, **kwargs: None, hamiltonian, dev, optimize=True)

    @pytest.mark.parametrize("interface", ["tf", "torch", "autograd"])
    def test_optimize(self, interface, tf_support, torch_support):
        """Test that an ExpvalCost with observable optimization gives the same result as another
        ExpvalCost without observable optimization."""
        if not qml.tape_mode_active():
            pytest.skip("This test is only intended for tape mode")
        if interface == "tf" and not tf_support:
            pytest.skip("This test requires TensorFlow")
        if interface == "torch" and not torch_support:
            pytest.skip("This test requires Torch")

        dev = qml.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian

        cost = qml.ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian,
            dev,
            optimize=True,
            interface=interface,
            diff_method="parameter-shift"
        )
        cost2 = qml.ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian,
            dev,
            optimize=False,
            interface=interface,
            diff_method="parameter-shift"
        )

        w = qml.init.strong_ent_layers_uniform(2, 4, seed=1967)

        c1 = cost(w)
        exec_opt = dev.num_executions
        dev._num_executions = 0

        c2 = cost2(w)
        exec_no_opt = dev.num_executions

        assert exec_opt == 5  # Number of groups in the Hamiltonian
        assert exec_no_opt == 15

        assert np.allclose(c1, c2)

    def test_optimize_grad(self):
        """Test that the gradient of ExpvalCost is accessible and correct when using observable
        optimization and the autograd interface."""
        if not qml.tape_mode_active():
            pytest.skip("This test is only intended for tape mode")

        dev = qml.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian

        cost = qml.ExpvalCost(
            qml.templates.StronglyEntanglingLayers, hamiltonian, dev, optimize=True, diff_method="parameter-shift"
        )
        cost2 = qml.ExpvalCost(
            qml.templates.StronglyEntanglingLayers, hamiltonian, dev, optimize=False, diff_method="parameter-shift"
        )

        w = qml.init.strong_ent_layers_uniform(2, 4, seed=1967)

        dc = qml.grad(cost)(w)
        exec_opt = dev.num_executions
        dev._num_executions = 0

        dc2 = qml.grad(cost2)(w)
        exec_no_opt = dev.num_executions

        assert exec_no_opt > exec_opt
        assert np.allclose(dc, big_hamiltonian_grad)
        assert np.allclose(dc2, big_hamiltonian_grad)

    def test_optimize_grad_torch(self, torch_support):
        """Test that the gradient of ExpvalCost is accessible and correct when using observable
        optimization and the Torch interface."""
        if not qml.tape_mode_active():
            pytest.skip("This test is only intended for tape mode")
        if not torch_support:
            pytest.skip("This test requires Torch")

        dev = qml.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian

        cost = qml.ExpvalCost(
            qml.templates.StronglyEntanglingLayers,
            hamiltonian,
            dev,
            optimize=True,
            interface="torch",
        )

        w = torch.tensor(qml.init.strong_ent_layers_uniform(2, 4, seed=1967), requires_grad=True)

        res = cost(w)
        res.backward()
        dc = w.grad.detach().numpy()

        assert np.allclose(dc, big_hamiltonian_grad)

    def test_optimize_grad_tf(self, tf_support):
        """Test that the gradient of ExpvalCost is accessible and correct when using observable
        optimization and the TensorFlow interface."""
        if not qml.tape_mode_active():
            pytest.skip("This test is only intended for tape mode")
        if not tf_support:
            pytest.skip("This test requires TensorFlow")

        dev = qml.device("default.qubit", wires=4)
        hamiltonian = big_hamiltonian

        cost = qml.ExpvalCost(
            qml.templates.StronglyEntanglingLayers, hamiltonian, dev, optimize=True, interface="tf"
        )

        w = tf.Variable(qml.init.strong_ent_layers_uniform(2, 4, seed=1967))

        with tf.GradientTape() as tape:
            res = cost(w)

        dc = tape.gradient(res, w).numpy()

        assert np.allclose(dc, big_hamiltonian_grad)

    def test_metric_tensor_tape_mode(self):
        """Test that the metric tensor can be calculated in tape mode, and that it is equal to a
        metric tensor calculated in non-tape mode."""
        if not qml.tape_mode_active():
            pytest.skip("This test is only intended for tape mode")

        dev = qml.device("default.qubit", wires=2)
        p = np.array([1., 1., 1.])

        def ansatz(params, **kwargs):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(params[2], wires=1)

        h = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])
        qnodes = qml.ExpvalCost(ansatz, h, dev)
        mt = qml.metric_tensor(qnodes)(p)
        assert qml.tape_mode_active()  # Check that tape mode is still active

        try:
            qml.disable_tape()

            @qml.qnode(dev)
            def circuit(params):
                qml.RX(params[0], wires=0)
                qml.RY(params[1], wires=0)
                qml.CNOT(wires=[0, 1])
                qml.PhaseShift(params[2], wires=1)
                return qml.expval(qml.PauliZ(0))

            mt2 = circuit.metric_tensor([p])
        finally:
            qml.enable_tape()

        assert np.allclose(mt, mt2)

    def test_multiple_devices(self, mocker):
        """Test that passing multiple devices to ExpvalCost works correctly"""

        dev = [qml.device("default.qubit", wires=2), qml.device("default.mixed", wires=2)]
        spy = mocker.spy(DefaultQubit, "apply")
        spy2 = mocker.spy(DefaultMixed, "apply")

        obs = [qml.PauliZ(0), qml.PauliZ(1)]
        h = qml.Hamiltonian([1, 1], obs)

        qnodes = qml.ExpvalCost(qml.templates.BasicEntanglerLayers, h, dev)
        w = qml.init.basic_entangler_layers_uniform(3, 2, seed=1967)

        res = qnodes(w)

        spy.assert_called_once()
        spy2.assert_called_once()

        mapped = qml.map(qml.templates.BasicEntanglerLayers, obs, dev)
        exp = sum(mapped(w))

        assert np.allclose(res, exp)

        with pytest.warns(UserWarning, match="ExpvalCost was instantiated with multiple devices."):
            qml.metric_tensor(qnodes)(w)

    def test_multiple_devices_opt_true(self):
        """Test if a ValueError is raised when multiple devices are passed when optimize=True."""
        if not qml.tape_mode_active():
            pytest.skip("This test is only intended for tape mode")

        dev = [qml.device("default.qubit", wires=2), qml.device("default.qubit", wires=2)]

        h = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliZ(1)])

        with pytest.raises(ValueError, match="Using multiple devices is not supported when"):
            qml.ExpvalCost(qml.templates.StronglyEntanglingLayers, h, dev, optimize=True)


@pytest.mark.usefixtures("tape_mode")
class TestAutogradInterface:
    """Tests for the Autograd interface (and the NumPy interface for backward compatibility)"""

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    @pytest.mark.parametrize("interface", ["autograd"])
    def test_QNodes_have_right_interface(self, ansatz, observables, params, mock_device, interface):
        """Test that QNodes have the Autograd interface"""
        dev = mock_device(wires=3)
        circuits = qml.map(ansatz, observables, device=dev, interface=interface)

        assert all(c.interface == "autograd" for c in circuits)

        res = [c(params) for c in circuits]
        assert all(isinstance(val, (np.ndarray, float)) for val in res)

    @pytest.mark.parametrize("interface", ["autograd"])
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

        cost = qml.ExpvalCost(ansatz, H, dev, interface=interface)
        dcost = qml.grad(cost, argnum=[0])
        res = dcost(params)

        expected = [
            -coeffs[0] * np.sin(a) * np.sin(b) - coeffs[1] * np.cos(a),
            coeffs[0] * np.cos(a) * np.cos(b),
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.usefixtures("tape_mode")
@pytest.mark.usefixtures("skip_if_no_torch_support")
class TestTorchInterface:
    """Tests for the PyTorch interface"""

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_QNodes_have_right_interface(self, ansatz, observables, params, mock_device):
        """Test that QNodes have the torch interface"""
        dev = mock_device(wires=3)
        circuits = qml.map(ansatz, observables, device=dev, interface="torch")
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

        cost = qml.ExpvalCost(ansatz, H, dev, interface="torch")
        loss = cost(params)
        loss.backward()

        res = params.grad.numpy()

        expected = [
            -coeffs[0] * np.sin(a) * np.sin(b) - coeffs[1] * np.cos(a),
            coeffs[0] * np.cos(a) * np.cos(b),
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.usefixtures("tape_mode")
@pytest.mark.usefixtures("skip_if_no_tf_support")
class TestTFInterface:
    """Tests for the TF interface"""

    @pytest.mark.parametrize("ansatz, params", CIRCUITS)
    @pytest.mark.parametrize("observables", OBSERVABLES)
    def test_QNodes_have_right_interface(self, ansatz, observables, params, mock_device):
        """Test that QNodes have the tf interface"""
        if ansatz == amp_embed_and_strong_ent_layer:
            pytest.skip("TF doesn't work with ragged arrays")

        dev = mock_device(wires=3)
        circuits = qml.map(ansatz, observables, device=dev, interface="tf")
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
        cost = qml.ExpvalCost(ansatz, H, dev, interface="tf")

        with tf.GradientTape() as tape:
            loss = cost(params)
            res = np.array(tape.gradient(loss, params))

        expected = [
            -coeffs[0] * np.sin(a) * np.sin(b) - coeffs[1] * np.cos(a),
            coeffs[0] * np.cos(a) * np.cos(b),
        ]

        assert np.allclose(res, expected, atol=tol, rtol=0)


@pytest.mark.usefixtures("tape_mode")
@pytest.mark.usefixtures("skip_if_no_tf_support")
@pytest.mark.usefixtures("skip_if_no_torch_support")
class TestMultipleInterfaceIntegration:
    """Tests to ensure that interfaces agree and integrate correctly"""

    def test_all_interfaces_gradient_agree(self, tol):
        """Test the gradient agrees across all interfaces"""
        dev = qml.device("default.qubit", wires=2)

        coeffs = [0.2, 0.5]
        observables = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0)]

        H = qml.vqe.Hamiltonian(coeffs, observables)

        # TensorFlow interface
        params = Variable(qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2, seed=1))
        ansatz = qml.templates.layers.StronglyEntanglingLayers

        cost = qml.ExpvalCost(ansatz, H, dev, interface="tf")

        with tf.GradientTape() as tape:
            loss = cost(params)
            res_tf = np.array(tape.gradient(loss, params))

        # Torch interface
        params = torch.tensor(qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2, seed=1))
        params = torch.autograd.Variable(params, requires_grad=True)
        ansatz = qml.templates.layers.StronglyEntanglingLayers

        cost = qml.ExpvalCost(ansatz, H, dev, interface="torch")
        loss = cost(params)
        loss.backward()
        res_torch = params.grad.numpy()

        # NumPy interface
        params = qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2, seed=1)
        ansatz = qml.templates.layers.StronglyEntanglingLayers
        cost = qml.ExpvalCost(ansatz, H, dev, interface="autograd")
        dcost = qml.grad(cost, argnum=[0])
        res = dcost(params)

        assert np.allclose(res, res_tf, atol=tol, rtol=0)
        assert np.allclose(res, res_torch, atol=tol, rtol=0)


def test_vqe_cost():
    """Tests that VQECost raises a DeprecationWarning but otherwise behaves as ExpvalCost"""

    h = qml.Hamiltonian([1], [qml.PauliZ(0)])
    dev = qml.device("default.qubit", wires=1)
    ansatz = qml.templates.StronglyEntanglingLayers

    with pytest.warns(DeprecationWarning, match="Use of VQECost is deprecated"):
        cost = qml.VQECost(ansatz, h, dev)

    assert isinstance(cost, qml.ExpvalCost)
