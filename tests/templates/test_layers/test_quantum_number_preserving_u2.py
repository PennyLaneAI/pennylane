# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the QuantumNumberPreservingU2 template.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        "layers, qubits, init_state, pi_gate_include",
        [
            (1, 4, np.array([1, 1, 0, 0]), False),
            (2, 4, np.array([1, 1, 0, 0]), True),
            (1, 6, np.array([1, 1, 0, 0, 0, 0]), False),
            (2, 6, np.array([1, 1, 0, 0, 0, 0]), True),
            (1, 8, np.array([1, 1, 0, 0, 0, 0, 0, 0]), False),
            (2, 8, np.array([1, 1, 1, 1, 0, 0, 0, 0]), True),
        ],
    )
    def test_operations(self, layers, qubits, init_state, pi_gate_include):
        """Test the correctness of the QuantumNumberPreservingU2 template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""

        weights = np.random.normal(0, 2 * np.pi, (layers, qubits // 2 - 1, 2))

        if not pi_gate_include:
            n_gates = 1 + (qubits - 2) * layers
            exp_gates = (
                ([qml.DoubleExcitation] + [qml.QuantumNumberPreservingOR]) * (qubits // 2 - 1)
            ) * layers
        else:
            n_gates = 1 + 3 * (qubits // 2 - 1) * layers
            exp_gates = (
                (
                    [qml.QuantumNumberPreservingOR]
                    + [qml.DoubleExcitation]
                    + [qml.QuantumNumberPreservingOR]
                )
                * (qubits // 2 - 1)
            ) * layers

        op = qml.templates.QuantumNumberPreservingU2(
            weights, wires=range(qubits), init_state=init_state, pi_gate_include=pi_gate_include
        )
        queue = op.expand().operations
        print(op, n_gates, queue)

        # number of gates
        assert len(queue) == n_gates

        # initialization
        assert isinstance(queue[0], qml.templates.BasisEmbedding)

        # order of gates
        for op1, op2 in zip(queue[1:], exp_gates):
            assert isinstance(op1, op2)

        # gate parameter
        params = np.array(
            [queue[i].parameters for i in range(1, n_gates) if queue[i].parameters != []]
        )

        if pi_gate_include:
            weights = np.insert(weights, 0, [[np.pi] * (qubits // 2 - 1)] * layers, axis=2)

        assert np.allclose(params.flatten(), weights.flatten())

        # gate wires
        wires = range(qubits)
        qwires = [wires[i : i + 4] for i in range(0, len(wires), 4) if len(wires[i : i + 4]) == 4]
        if len(wires) // 2 > 2:
            qwires += [
                wires[i : i + 4] for i in range(2, len(wires), 4) if len(wires[i : i + 4]) == 4
            ]

        exp_wires = []
        for _ in range(layers):
            for wire in qwires:
                if pi_gate_include:
                    exp_wires.append(list(wire))
                exp_wires.append(list(wire))
                exp_wires.append(list(wire))

        res_wires = [queue[i].wires.tolist() for i in range(1, n_gates)]
        assert res_wires == exp_wires

    @pytest.mark.parametrize(
        ("init_state", "exp_state"),
        [
            (
                np.array([0, 0, 0, 0]),
                np.array(
                    [
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([0, 0, 0, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([0, 0, 1, 0]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([0, 1, 0, 0]),
                np.array(
                    [
                        0.0 + 0.0j,
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 0, 0, 0]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([0, 0, 1, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([0, 1, 0, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 0, 0, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 1, 0, 0]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 0, 1, 0]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([0, 1, 1, 0]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 1, 1, 0]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 0, 1, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        -0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([0, 1, 1, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 1, 0, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                np.array([1, 1, 1, 1]),
                np.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                    ]
                ),
            ),
        ],
    )
    def test_decomposition_q(self, init_state, exp_state, tol):
        """Test the decomposition of the Q_{theta, phi}` exchange gate by asserting the prepared
        state."""

        N = 4
        wires = range(N)

        weight = [[[np.pi / 2, np.pi / 2]]]

        dev = qml.device("default.qubit", wires=N)
        print(init_state, exp_state)

        @qml.qnode(dev)
        def circuit(weight):
            qml.templates.layers.QuantumNumberPreservingU2(weight, wires, init_state=init_state)
            return qml.expval(qml.PauliZ(0))

        circuit(weight)

        assert np.allclose(circuit.device.state, exp_state, atol=tol)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 1, 2))
        init_state = np.array([1, 1, 0, 0])

        dev = qml.device("default.qubit", wires=4)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "r"])

        @qml.qnode(dev)
        def circuit():
            qml.templates.QuantumNumberPreservingU2(weights, wires=range(4), init_state=init_state)
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.templates.QuantumNumberPreservingU2(
                weights, wires=["z", "a", "k", "r"], init_state=init_state
            )
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weights", "wires", "msg_match"),
        [
            (
                np.array([[[-0.080, 2.629]]]),
                [0],
                "This template requires the number of qubits to be greater than four",
            ),
            (
                np.array([[[-0.080, 2.629]]]),
                [5],
                "This template requires the number of qubits to be greater than four",
            ),
            (
                np.array([[[-0.080, 2.629, -0.710, 5.383]]]),
                [0, 1, 2, 3],
                "Weights tensor must",
            ),
            (
                np.array(
                    [
                        [
                            [-0.080, 2.629, -0.710, 5.383, 0.646, -2.872],
                            [-0.080, 2.629, -0.710, 5.383, 0.646, -2.872],
                        ]
                    ]
                ),
                [0, 1, 2, 3],
                "Weights tensor must",
            ),
            (
                np.array([-0.080, 2.629, -0.710, 5.383, 0.646, -2.872]),
                [0, 1, 2, 3],
                "Weights tensor must be 3-dimensional",
            ),
            (
                np.array([[[-0.080, 2.629], [-0.710, 5.383]]]),
                [0, 1, 2, 3],
                "Weights tensor must have second dimension of length",
            ),
            (
                np.array([[[-0.080]]]),
                [0, 1, 2, 3],
                "Weights tensor must have third dimension of length 2",
            ),
            (
                np.array([[[-0.080, 2.629]]]),
                [0, 1, 2, 3, 4],
                "This template requires the number of qubits to be multiple of 2",
            ),
        ],
    )
    def test_exceptions(self, weights, wires, msg_match):
        """Test that ParticleConservingU2 throws an exception if the parameters have illegal
        shapes, types or values."""
        N = len(wires)
        init_state = np.array([1, 1, 0, 0])

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit():
            qml.templates.QuantumNumberPreservingU2(
                weights=weights,
                wires=wires,
                init_state=init_state,
            )
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    def test_id(self):
        """Tests that the id attribute can be set."""
        init_state = np.array([1, 1, 0, 0])
        template = qml.templates.QuantumNumberPreservingU2(
            weights=np.random.random(size=(1, 1, 2)), wires=range(4), init_state=init_state, id="a"
        )
        assert template.id == "a"

    def test_init_state_exception(self):
        """Tests that the operation warns if initial state is not provided"""
        with pytest.raises(ValueError, match="Inital state should be provided"):
            qml.templates.QuantumNumberPreservingU2(
                weights=np.random.random(size=(1, 1, 2)), wires=range(4)
            )


class TestAttributes:
    """Tests additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_layers, n_wires, expected_shape",
        [
            (2, 4, (2, 1, 2)),
            (2, 6, (2, 2, 2)),
            (2, 8, (2, 3, 2)),
        ],
    )
    def test_shape(self, n_layers, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor"""

        shape = qml.templates.QuantumNumberPreservingU2.shape(n_layers, n_wires)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(
            ValueError, match="This template requires the number of qubits to be greater than four"
        ):
            qml.templates.QuantumNumberPreservingU2.shape(3, 1)

    def test_shape_exception_not_even_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(
            ValueError, match="This template requires the number of qubits to be multiple of 2"
        ):
            qml.templates.QuantumNumberPreservingU2.shape(1, 5)


def circuit_template(weights):
    qml.templates.QuantumNumberPreservingU2(
        weights,
        range(4 + (weights.shape[1] - 1) * 2),
        init_state=np.array([1, 1, 0, 0] + [0] * (weights.shape[1] - 1) * 2),
    )
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):

    wires = range(4 + (weights.shape[1] - 1) * 2)

    qwires = [wires[i : i + 4] for i in range(0, len(wires), 4) if len(wires[i : i + 4]) == 4]
    if len(wires) // 2 > 2:
        qwires += [wires[i : i + 4] for i in range(2, len(wires), 4) if len(wires[i : i + 4]) == 4]

    qml.BasisState(np.array([1, 1, 0, 0] + [0] * (len(wires) - 4)), wires=wires)

    for layer in range(weights.shape[0]):
        for idx in range(weights.shape[1]):
            qml.DoubleExcitation(weights[layer][idx][0], wires=qwires[idx])
            qml.QuantumNumberPreservingOR(weights[layer][idx][1], wires=qwires[idx])

    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    def test_list_and_tuples(self, tol):
        """Tests common iterables as inputs."""

        weights = [
            [
                [
                    0.1,
                    -1.1,
                ]
            ]
        ]

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        weights_tuple = [tuple(weights[0])]
        res = circuit(weights_tuple)
        res2 = circuit2(weights_tuple)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = np.random.random(size=(1, 1, 2))
        weights = pnp.array(weights)

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = qml.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = qml.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    def test_jax(self, tol):
        """Tests the jax interface."""

        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev, interface="jax")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    def test_tf(self, tol):
        """Tests the tf interface."""

        tf = pytest.importorskip("tensorflow")

        weights = tf.Variable(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev, interface="tf")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="tf")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(weights)
        grads = tape.gradient(res, [weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(weights)
        grads2 = tape2.gradient(res2, [weights])

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    def test_torch(self, tol):
        """Tests the torch interface."""

        torch = pytest.importorskip("torch")

        weights = torch.tensor(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev, interface="torch")
        circuit2 = qml.QNode(circuit_decomposed, dev, interface="torch")

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights)
        res2.backward()
        grads2 = [weights.grad]

        assert np.allclose(grads[0], grads2[0], atol=tol, rtol=0)
