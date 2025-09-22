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
Unit tests for the GateFabric template.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.jax
@pytest.mark.parametrize("include_pi", (True, False))
def test_standard_validity(include_pi):
    """Check the operation using the assert_valid function."""

    layers = 2
    qubits = 6
    init_state = qml.math.array([1, 1, 0, 0, 0, 0])

    weights = np.random.normal(0, 2 * np.pi, (layers, qubits // 2 - 1, 2))

    op = qml.GateFabric(weights, wires=range(qubits), init_state=init_state, include_pi=include_pi)

    qml.ops.functions.assert_valid(op)


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        "layers, qubits, init_state, include_pi",
        [
            (1, 4, qml.math.array([1, 1, 0, 0]), False),
            (2, 4, qml.math.array([1, 1, 0, 0]), True),
            (1, 6, qml.math.array([1, 1, 0, 0, 0, 0]), False),
            (2, 6, qml.math.array([1, 1, 0, 0, 0, 0]), True),
            (1, 8, qml.math.array([1, 1, 0, 0, 0, 0, 0, 0]), False),
            (2, 8, qml.math.array([1, 1, 1, 1, 0, 0, 0, 0]), True),
        ],
    )
    def test_operations(self, layers, qubits, init_state, include_pi):
        """Test the correctness of the GateFabric template including the gate count
        and order, the wires each operation acts on and the correct use of parameters
        in the circuit."""

        weights = np.random.normal(0, 2 * np.pi, (layers, qubits // 2 - 1, 2))

        if not include_pi:
            n_gates = 1 + (qubits - 2) * layers
            exp_gates = (
                ([qml.DoubleExcitation] + [qml.OrbitalRotation]) * (qubits // 2 - 1)
            ) * layers
        else:
            n_gates = 1 + 3 * (qubits // 2 - 1) * layers
            exp_gates = (
                ([qml.OrbitalRotation] + [qml.DoubleExcitation] + [qml.OrbitalRotation])
                * (qubits // 2 - 1)
            ) * layers

        op = qml.GateFabric(
            weights, wires=range(qubits), init_state=init_state, include_pi=include_pi
        )
        queue = op.decomposition()

        # number of gates
        assert len(queue) == n_gates

        # initialization
        assert isinstance(queue[0], qml.BasisEmbedding)

        # order of gates
        for op1, op2 in zip(queue[1:], exp_gates):
            assert isinstance(op1, op2)

        # gate parameter
        params = qml.math.array(
            [queue[i].parameters for i in range(1, n_gates) if queue[i].parameters != []]
        )

        if include_pi:
            weights = qml.math.insert(weights, 0, [[np.pi] * (qubits // 2 - 1)] * layers, axis=2)

        assert qml.math.allclose(params.flatten(), weights.flatten())

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
                if include_pi:
                    exp_wires.append(list(wire))
                exp_wires.append(list(wire))
                exp_wires.append(list(wire))

        res_wires = [queue[i].wires.tolist() for i in range(1, n_gates)]
        assert res_wires == exp_wires

    @pytest.mark.parametrize(
        ("init_state", "exp_state"),
        [
            (
                qml.math.array([0, 0, 0, 0]),
                qml.math.array(
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
                qml.math.array([0, 0, 0, 1]),
                qml.math.array(
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
                qml.math.array([0, 0, 1, 0]),
                qml.math.array(
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
                qml.math.array([0, 1, 0, 0]),
                qml.math.array(
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
                qml.math.array([1, 0, 0, 0]),
                qml.math.array(
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
                qml.math.array([0, 0, 1, 1]),
                qml.math.array(
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
                qml.math.array([0, 1, 0, 1]),
                qml.math.array(
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
                qml.math.array([1, 0, 0, 1]),
                qml.math.array(
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
                qml.math.array([1, 1, 0, 0]),
                qml.math.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
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
                qml.math.array([1, 0, 1, 0]),
                qml.math.array(
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
                qml.math.array([0, 1, 1, 0]),
                qml.math.array(
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.5 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                qml.math.array([1, 1, 1, 0]),
                qml.math.array(
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
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                qml.math.array([1, 0, 1, 1]),
                qml.math.array(
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
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                qml.math.array([0, 1, 1, 1]),
                qml.math.array(
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
                        -0.70710678 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ),
            ),
            (
                qml.math.array([1, 1, 0, 1]),
                qml.math.array(
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
                qml.math.array([1, 1, 1, 1]),
                qml.math.array(
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
        """Test the decomposition of the :math:`Q_{theta, phi}` gate by asserting the prepared state."""

        N = 4
        wires = range(N)

        weight = [[[np.pi / 2, np.pi / 2]]]

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(weight):
            qml.GateFabric(weight, wires, init_state=init_state)
            return qml.state()

        assert qml.math.allclose(circuit(weight), exp_state, atol=tol)

    @pytest.mark.parametrize(
        ("num_qubits", "layers", "exp_state"),
        [
            (
                4,
                4,
                qml.math.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.25,
                        0.0,
                        0.0,
                        -0.10355,
                        0.0,
                        0.0,
                        0.10355,
                        0.0,
                        0.0,
                        0.95711,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
            (
                6,
                6,
                qml.math.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.24264,
                        0.0,
                        0.0,
                        0.31019,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.4068,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.25487,
                        0.0,
                        0.0,
                        -0.02977,
                        0.0,
                        0.0,
                        -0.37703,
                        0.0,
                        0.0,
                        0.49874,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.40826,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.23667,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
        ],
    )
    def test_layers_gate_fabric(self, num_qubits, layers, exp_state, tol):
        """Test that the GateFabric template with multiple layers works correctly asserting the prepared state."""

        wires = range(num_qubits)

        shape = qml.GateFabric.shape(n_layers=layers, n_wires=num_qubits)
        weight = np.pi / 2 * qml.math.ones(shape)

        dev = qml.device("default.qubit", wires=wires)

        init_state = qml.math.array([1 if x < num_qubits // 2 else 0 for x in wires])

        @qml.qnode(dev)
        def circuit(weight):
            qml.GateFabric(weight, wires, init_state=init_state, include_pi=True)
            return qml.state()

        assert qml.math.allclose(circuit(weight), exp_state, atol=tol)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        weights = np.random.random(size=(1, 1, 2))
        init_state = qml.math.array([1, 1, 0, 0])

        dev = qml.device("default.qubit", wires=4)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "r"])

        @qml.qnode(dev)
        def circuit():
            qml.GateFabric(weights, wires=range(4), init_state=init_state)
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.GateFabric(weights, wires=["z", "a", "k", "r"], init_state=init_state)
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("weights", "wires", "msg_match"),
        [
            (
                qml.math.array([[[-0.080, 2.629]]]),
                [0],
                "This template requires the number of qubits to be greater than four",
            ),
            (
                qml.math.array([[[-0.080, 2.629]]]),
                [5],
                "This template requires the number of qubits to be greater than four",
            ),
            (
                qml.math.array([[[-0.080, 2.629]]]),
                [0, 1, 2, 3, 4],
                "This template requires an even number of qubits",
            ),
            (
                qml.math.array([[[-0.080]]]),
                [0, 1, 2, 3],
                "Weights tensor must have third dimension of length 2",
            ),
            (
                qml.math.array([[[-0.080, 2.629, -0.710, 5.383]]]),
                [0, 1, 2, 3],
                "Weights tensor must have third dimension of length 2",
            ),
            (
                qml.math.array(
                    [
                        [
                            [-0.080, 2.629, -0.710, 5.383, 0.646, -2.872],
                            [-0.080, 2.629, -0.710, 5.383, 0.646, -2.872],
                        ]
                    ]
                ),
                [0, 1, 2, 3],
                "Weights tensor must have second dimension of length",
            ),
            (
                qml.math.array([[[-0.080, 2.629], [-0.710, 5.383]]]),
                [0, 1, 2, 3],
                "Weights tensor must have second dimension of length",
            ),
            (
                qml.math.array([-0.080, 2.629, -0.710, 5.383, 0.646, -2.872]),
                [0, 1, 2, 3],
                "Weights tensor must be 3-dimensional",
            ),
        ],
    )
    def test_exceptions(self, weights, wires, msg_match):
        """Test that GateFabric template throws an exception if the parameters have illegal
        shapes, types or values."""
        N = len(wires)
        init_state = qml.math.array([1, 1, 0, 0])

        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit():
            qml.GateFabric(
                weights=weights,
                wires=wires,
                init_state=init_state,
            )
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    def test_id(self):
        """Tests that the id attribute can be set."""
        init_state = qml.math.array([1, 1, 0, 0])
        template = qml.GateFabric(
            weights=np.random.random(size=(1, 1, 2)), wires=range(4), init_state=init_state, id="a"
        )
        assert template.id == "a"


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

        shape = qml.GateFabric.shape(n_layers, n_wires)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(
            ValueError, match="This template requires the number of qubits to be at least four"
        ):
            qml.GateFabric.shape(3, 1)

    def test_shape_exception_not_even_qubits(self):
        """Test that the shape function warns if the number of qubits are not even."""

        with pytest.raises(ValueError, match="This template requires an even number of qubits"):
            qml.GateFabric.shape(1, 5)


def circuit_template(weights):
    qml.GateFabric(weights, range(4), init_state=qml.math.array([1, 1, 0, 0]), include_pi=True)
    return qml.expval(qml.PauliZ(0))


def circuit_decomposed(weights):
    wires = range(4)
    qwires = [wires[i : i + 4] for i in range(0, len(wires), 4) if len(wires[i : i + 4]) == 4]
    if len(wires) > 4:
        qwires += [wires[i : i + 4] for i in range(2, len(wires), 4) if len(wires[i : i + 4]) == 4]
    qml.BasisState(qml.math.array([1, 1, 0, 0]), wires=wires)
    include_pi_param = qml.math.array(np.pi, like=qml.math.get_interface(*weights))
    for layer in range(weights.shape[0]):
        for idx in range(weights.shape[1]):
            qml.OrbitalRotation.compute_decomposition(include_pi_param, wires=qwires[idx])
            qml.DoubleExcitation.compute_decomposition(weights[layer][idx][0], wires=qwires[idx])
            qml.OrbitalRotation.compute_decomposition(weights[layer][idx][1], wires=qwires[idx])

    return qml.expval(qml.PauliZ(0))


class TestInterfaces:
    """Tests that the GateFabric template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self, tol):
        """Tests the autograd interface."""

        weights = pnp.random.random(size=(1, 1, 2), requires_grad=True)

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

        assert qml.math.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qml.math.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self, tol):
        """Tests jit within the jax interface."""

        import jax
        import jax.numpy as jnp

        weights = jnp.array(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = jax.jit(circuit)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        grad_fn = jax.grad(circuit)
        grads = grad_fn(weights)

        grad_fn2 = jax.grad(circuit2)
        grads2 = grad_fn2(weights)

        assert qml.math.allclose(grads, grads2, atol=tol, rtol=0)

    @pytest.mark.tf
    def test_tf(self, tol):
        """Tests the tf interface."""

        import tensorflow as tf

        weights = tf.Variable(np.random.random(size=(1, 1, 2)))

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        with tf.GradientTape() as tape:
            res = circuit(weights)
        grads = tape.gradient(res, [weights])

        with tf.GradientTape() as tape2:
            res2 = circuit2(weights)
        grads2 = tape2.gradient(res2, [weights])

        assert qml.math.allclose(grads[0], grads2[0], atol=tol, rtol=0)

    @pytest.mark.torch
    def test_torch(self, tol):
        """Tests the torch interface."""

        import torch

        weights = torch.tensor(np.random.random(size=(1, 1, 2)), requires_grad=True)

        dev = qml.device("default.qubit", wires=4)

        circuit = qml.QNode(circuit_template, dev)
        circuit2 = qml.QNode(circuit_decomposed, dev)

        res = circuit(weights)
        res2 = circuit2(weights)
        assert qml.math.allclose(res, res2, atol=tol, rtol=0)

        res = circuit(weights)
        res.backward()
        grads = [weights.grad]

        res2 = circuit2(weights)
        res2.backward()
        grads2 = [weights.grad]

        assert qml.math.allclose(grads[0], grads2[0], atol=tol, rtol=0)
