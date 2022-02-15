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
Integration tests for the draw transform
"""
import pytest
from functools import partial

import pennylane as qml
from pennylane import numpy as np

from pennylane.transforms import draw


@qml.qnode(qml.device("default.qubit", wires=(0, "a", 1.234)))
def circuit(x, y, z):
    qml.RX(x, wires=0)
    qml.RY(y, wires="a")
    qml.RZ(z, wires=1.234)
    return qml.expval(qml.PauliZ(0))


class TestLabelling:
    """Test the wire labels."""

    def test_any_wire_labels(self):
        """Test wire labels with different kinds of objects."""

        split_str = draw(circuit)(1.2, 2.3, 3.4).split("\n")
        assert split_str[0][0:6] == "    0:"
        assert split_str[1][0:6] == "    a:"
        assert split_str[2][0:6] == "1.234:"

    def test_wire_order(self):
        """Test wire_order keyword changes order of the wires."""

        split_str = draw(circuit, wire_order=[1.234, "a", 0, "b"])(1.2, 2.3, 3.4).split("\n")
        assert split_str[0][0:6] == "1.234:"
        assert split_str[1][0:6] == "    a:"
        assert split_str[2][0:6] == "    0:"

    def test_show_all_wires(self):
        """Test show_all_wires=True forces empty wires to display."""

        @qml.qnode(qml.device("default.qubit", wires=(0, 1)))
        def circuit():
            return qml.expval(qml.PauliZ(0))

        split_str = draw(circuit, show_all_wires=True)().split("\n")
        assert split_str[0][0:2] == "0:"
        assert split_str[1][0:2] == "1:"

    def test_show_all_wires_and_wire_order(self):
        """Test show_all_wires forces empty wires to display when empty wire is in wire order."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            return qml.expval(qml.PauliZ(0))

        split_str = draw(circuit, wire_order=[0, "a"], show_all_wires=True)().split("\n")
        assert split_str[0][0:2] == "0:"
        assert split_str[1][0:2] == "a:"


class TestDecimals:
    """Test the decimals keyword argument."""

    def test_decimals_None(self):
        """Test that when decimals is ``None``, parameters are omitted."""

        expected = "    0: ──RX─┤  <Z>\n    a: ──RY─┤     \n1.234: ──RZ─┤     "
        assert draw(circuit, decimals=None)(1.234, 2.345, 3.456) == expected

    def test_decimals(self):
        """Test decimals keyword makes the operation parameters included to given precision"""

        expected = "    0: ──RX(1.2)─┤  <Z>\n    a: ──RY(2.3)─┤     \n1.234: ──RZ(3.5)─┤     "
        assert draw(circuit, decimals=1)(1.234, 2.345, 3.456) == expected

    def test_decimals_higher_value(self):
        """Test all decimals places display when requested value is bigger than number precision."""

        out = "    0: ──RX(1.0000)─┤  <Z>\n    a: ──RY(2.0000)─┤     \n1.234: ──RZ(3.0000)─┤     "
        assert qml.draw(circuit, decimals=4)(1, 2, 3) == out

    def test_decimals_multiparameters(self):
        """Test decimals also displays parameters when the operation has multiple parameters."""

        @qml.qnode(qml.device("default.qubit", wires=(0)))
        def circuit(x):
            qml.Rot(*x, wires=0)
            return qml.expval(qml.PauliZ(0))

        expected = "0: ──Rot(1.2,2.3,3.5)─┤  <Z>"
        assert draw(circuit, decimals=1)([1.234, 2.345, 3.456]) == expected

    def test_decimals_0(self):
        """Test decimals=0 rounds to integers."""

        expected = "    0: ──RX(1)─┤  <Z>\n    a: ──RY(2)─┤     \n1.234: ──RZ(3)─┤     "
        assert draw(circuit, decimals=0)(1.234, 2.3456, 3.456) == expected

    def test_qml_numpy_parameters(self):
        """Test numpy parameters display as normal numbers."""

        expected = "    0: ──RX(1.00)─┤  <Z>\n    a: ──RY(2.00)─┤     \n1.234: ──RZ(3.00)─┤     "
        assert draw(circuit)(np.array(1), np.array(2), np.array(3)) == expected

    def test_torch_parameters(self):
        """Test torch parameters display as normal numbers."""

        torch = pytest.importorskip("torch")
        expected = "    0: ──RX(1.2)─┤  <Z>\n    a: ──RY(2.3)─┤     \n1.234: ──RZ(3.5)─┤     "
        out = draw(circuit, decimals=1)(torch.tensor(1.23), torch.tensor(2.34), torch.tensor(3.45))
        assert out == expected

    def test_tensorflow_parameters(self):
        """Test tensorflow parameters display as normal numbers."""
        tf = pytest.importorskip("tensorflow")

        expected = "    0: ──RX(1.2)─┤  <Z>\n    a: ──RY(2.3)─┤     \n1.234: ──RZ(3.5)─┤     "
        out = draw(circuit, decimals=1)(tf.Variable(1.234), tf.Variable(2.345), tf.Variable(3.456))
        assert out == expected

    def test_jax_parameters(self):
        """Test jax parameters in tape display as normal numbers."""
        jnp = pytest.importorskip("jax.numpy")

        expected = "    0: ──RX(1.2)─┤  <Z>\n    a: ──RY(2.3)─┤     \n1.234: ──RZ(3.5)─┤     "
        out = draw(circuit, decimals=1)(jnp.array(1.234), jnp.array(2.345), jnp.array(3.456))
        assert out == expected

    def test_string_decimals(self):
        """Test displays string valued parameters."""

        expected = "    0: ──RX(x)─┤  <Z>\n    a: ──RY(y)─┤     \n1.234: ──RZ(z)─┤     "
        assert draw(circuit)("x", "y", "z") == expected


class TestMaxLength:
    """Test the max_length keyword."""

    def test_max_length_default(self):
        """Test max length default to 100."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def long_circuit():
            for _ in range(100):
                qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        out = draw(long_circuit)()
        assert 95 <= max(len(s) for s in out.split("\n")) <= 100

    @pytest.mark.parametrize("ml", [10, 15, 20])
    def test_setting_max_length(self, ml):
        @qml.qnode(qml.device("default.qubit", wires=1))
        def long_circuit():
            for _ in range(10):
                qml.PauliX(0)
            return [qml.expval(qml.PauliZ(0)) for _ in range(4)]

        out = draw(long_circuit, max_length=ml)()
        assert max(len(s) for s in out.split("\n")) <= ml


class TestLayering:
    """Test operations are placed in the correct locations."""

    def test_adjacent_ops(self):
        """Test non-blocking gates end up on same layer."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circuit():
            [qml.PauliX(i) for i in range(3)]
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        expected = "0: ──X─┤  <Z>\n1: ──X─┤  <Z>\n2: ──X─┤  <Z>"
        assert draw(circuit)() == expected

    def test_blocking_ops(self):
        """Test single qubits gates on the same wire block each other."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            [qml.PauliX(0) for i in range(3)]
            return qml.expval(qml.PauliZ(0))

        expected = "0: ──X──X──X─┤  <Z>"

    def test_blocking_multiwire_gate(self):
        """Test gate gets blocked by multi-wire gate."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circuit():
            qml.PauliX(0)
            qml.IsingXX(1.234, wires=(0, 2))
            qml.PauliX(1)
            return qml.expval(qml.PauliZ(0))

        expect = (
            "0: ──X─╭IsingXX(1.23)────┤  <Z>\n"
            "1: ────│───────────────X─┤     \n"
            "2: ────╰IsingXX(1.23)────┤     "
        )
        assert draw(circuit)() == expect


@pytest.mark.parametrize(
    "transform",
    [qml.gradients.param_shift(shift=0.2), partial(qml.gradients.param_shift, shift=0.2)],
)
def test_draw_batch_transform(transform):
    """Test that drawing a batch transform works correctly."""

    @transform
    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    expected = "0: ──H──RX(0.8)─┤  <Z>\n\n0: ──H──RX(0.4)─┤  <Z>"
    assert draw(circuit, decimals=1)(np.array(0.6, requires_grad=True)) == expected


def test_nested_tapes():
    """Test nested tapes inside the qnode."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit():
        with qml.tape.QuantumTape() as tape1:
            qml.PauliX(0)
            with qml.tape.QuantumTape() as tape2:
                qml.PauliY(0)
        with qml.tape.QuantumTape() as tape3:
            qml.PauliZ(0)
            with qml.tape.QuantumTape() as tape4:
                qml.PauliX(0)
        return qml.expval(qml.PauliZ(0))

    expected = (
        "0: ──Tape:0──Tape:1─┤  <Z>\n\n"
        "Tape:0\n0: ──X──Tape:2─┤  \n\n"
        "Tape:2\n0: ──Y─┤  \n\n"
        "Tape:1\n0: ──Z──Tape:3─┤  \n\n"
        "Tape:3\n0: ──X─┤  "
    )

    assert draw(circuit)() == expected


def test_expansion_strategy():
    """Test expansion strategy keyword modifies tape expansion."""

    H = qml.PauliX(0) + qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliX(1)

    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(t):
        qml.ApproxTimeEvolution(H, t, 2)
        return qml.probs(wires=0)

    expected_gradient = "0: ─╭ApproxTimeEvolution─┤  Probs\n1: ─╰ApproxTimeEvolution─┤       "
    assert draw(circuit, expansion_strategy="gradient", decimals=None)(0.5)

    expected_device = (
        "0: ──H────────MultiRZ──H──H─╭MultiRZ──H──H────────MultiRZ──H──H─╭MultiRZ──H─┤  Probs\n"
        "1: ──MultiRZ──H─────────────╰MultiRZ──H──MultiRZ──H─────────────╰MultiRZ──H─┤       "
    )
    assert draw(circuit, expansion_strategy="device", decimals=None)(0.5)
