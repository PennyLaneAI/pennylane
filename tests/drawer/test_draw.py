# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
# pylint: disable=import-outside-toplevel
from functools import partial

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.drawer import draw


@qml.qnode(qml.device("default.qubit", wires=(0, "a", 1.234)))
def circuit(x, y, z):
    """A quantum circuit on three wires."""
    qml.RX(x, wires=0)
    qml.RY(y, wires="a")
    qml.RZ(z, wires=1.234)
    return qml.expval(qml.PauliZ(0))


class TestLabelling:
    """Test the wire labels."""

    def test_any_wire_labels(self):
        """Test wire labels with different kinds of objects."""

        split_str = draw(circuit)(1.2, 2.3, 3.4).split("\n")
        assert split_str[0][:6] == "    0:"
        assert split_str[1][:6] == "    a:"
        assert split_str[2][:6] == "1.234:"

    def test_wire_order(self):
        """Test wire_order keyword changes order of the wires."""

        split_str = draw(circuit, wire_order=[1.234, "a", 0, "b"])(1.2, 2.3, 3.4).split("\n")
        assert split_str[0][:6] == "1.234:"
        assert split_str[1][:6] == "    a:"
        assert split_str[2][:6] == "    0:"

    def test_show_all_wires(self):
        """Test show_all_wires=True forces empty wires to display."""

        @qml.qnode(qml.device("default.qubit", wires=(0, 1)))
        def circ():
            return qml.expval(qml.PauliZ(0))

        split_str = draw(circ, show_all_wires=True)().split("\n")
        assert split_str[0][:2] == "0:"
        assert split_str[1][:2] == "1:"

    def test_show_all_wires_and_wire_order(self):
        """Test show_all_wires forces empty wires to display when empty wire is in wire order."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circ():
            return qml.expval(qml.PauliZ(0))

        split_str = draw(circ, wire_order=[0, "a"], show_all_wires=True)().split("\n")
        assert split_str[0][:2] == "0:"
        assert split_str[1][:2] == "a:"


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

        @qml.qnode(qml.device("default.qubit", wires=[0]))
        def circ(x):
            qml.Rot(*x, wires=0)
            return qml.expval(qml.PauliZ(0))

        expected = "0: ──Rot(1.2,2.3,3.5)─┤  <Z>"
        assert draw(circ, decimals=1)([1.234, 2.345, 3.456]) == expected

    def test_decimals_0(self):
        """Test decimals=0 rounds to integers."""

        expected = "    0: ──RX(1)─┤  <Z>\n    a: ──RY(2)─┤     \n1.234: ──RZ(3)─┤     "
        assert draw(circuit, decimals=0)(1.234, 2.3456, 3.456) == expected

    def test_qml_numpy_parameters(self):
        """Test numpy parameters display as normal numbers."""

        expected = "    0: ──RX(1.00)─┤  <Z>\n    a: ──RY(2.00)─┤     \n1.234: ──RZ(3.00)─┤     "
        assert draw(circuit)(np.array(1), np.array(2), np.array(3)) == expected

    @pytest.mark.torch
    def test_torch_parameters(self):
        """Test torch parameters display as normal numbers."""

        import torch

        expected = "    0: ──RX(1.2)─┤  <Z>\n    a: ──RY(2.3)─┤     \n1.234: ──RZ(3.5)─┤     "
        out = draw(circuit, decimals=1)(torch.tensor(1.23), torch.tensor(2.34), torch.tensor(3.45))
        assert out == expected

    @pytest.mark.tf
    def test_tensorflow_parameters(self):
        """Test tensorflow parameters display as normal numbers."""
        import tensorflow as tf

        expected = "    0: ──RX(1.2)─┤  <Z>\n    a: ──RY(2.3)─┤     \n1.234: ──RZ(3.5)─┤     "
        out = draw(circuit, decimals=1)(tf.Variable(1.234), tf.Variable(2.345), tf.Variable(3.456))
        assert out == expected

    @pytest.mark.jax
    def test_jax_parameters(self):
        """Test jax parameters in tape display as normal numbers."""
        import jax.numpy as jnp

        expected = "    0: ──RX(1.2)─┤  <Z>\n    a: ──RY(2.3)─┤     \n1.234: ──RZ(3.5)─┤     "
        out = draw(circuit, decimals=1)(jnp.array(1.234), jnp.array(2.345), jnp.array(3.456))
        assert out == expected

    def test_string_decimals(self):
        """Test displays string valued parameters."""

        expected = "    0: ──RX(x)─┤  <Z>\n    a: ──RY(y)─┤     \n1.234: ──RZ(z)─┤     "
        assert draw(circuit)("x", "y", "z") == expected


class TestMatrixParameters:
    """Test that tapes containing matrix-valued parameters are drawn correctly."""

    def test_matrix_parameters(self):
        """Test matrix valued parameters remembered and printed out upon request."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def matrices_circuit():
            qml.StatePrep([1.0, 0.0, 0.0, 0.0], wires=(0, 1))
            qml.QubitUnitary(np.eye(2), wires=0)
            return qml.expval(qml.Hermitian(np.eye(2), wires=0))

        expected1 = "0: ─╭|Ψ⟩──U(M0)─┤  <𝓗(M0)>\n1: ─╰|Ψ⟩────────┤         "

        assert draw(matrices_circuit, show_matrices=False)() == expected1

        expected2 = (
            "0: ─╭|Ψ⟩──U(M0)─┤  <𝓗(M0)>\n"
            "1: ─╰|Ψ⟩────────┤         \n"
            "\n"
            "M0 = \n[[1. 0.]\n [0. 1.]]"
        )
        assert draw(matrices_circuit)() == expected2

    def test_matrix_parameters_batch_transform(self):
        """Test matrix parameters only printed once after a batch transform."""

        @partial(qml.gradients.param_shift, shifts=[(0.2,)])
        @qml.qnode(qml.device("default.qubit", wires=2))
        def matrices_circuit(x):
            qml.StatePrep([1.0, 0.0, 0.0, 0.0], wires=(0, 1))
            qml.QubitUnitary(np.eye(2, requires_grad=False), wires=0)
            qml.RX(x, wires=1)
            return qml.expval(qml.Hermitian(np.eye(2, requires_grad=False), wires=1))

        expected1 = (
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(1.20)─┤  <𝓗(M0)>\n\n"
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(0.80)─┤  <𝓗(M0)>\n\n"
            "M0 = \n[[1. 0.]\n [0. 1.]]"
        )
        output = draw(matrices_circuit)(np.array(1.0, requires_grad=True))
        assert output == expected1

        expected2 = (
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(1.20)─┤  <𝓗(M0)>\n\n"
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(0.80)─┤  <𝓗(M0)>"
        )
        output = draw(matrices_circuit, show_matrices=False)(np.array(1.0, requires_grad=True))
        assert output == expected2


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
        """Test that setting a maximal length works as expected."""

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
        def circ():
            _ = [qml.PauliX(i) for i in range(3)]
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        expected = "0: ──X─┤  <Z>\n1: ──X─┤  <Z>\n2: ──X─┤  <Z>"
        assert draw(circ)() == expected

    def test_blocking_ops(self):
        """Test single qubits gates on the same wire block each other."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circ():
            _ = [qml.PauliX(0) for i in range(3)]
            return qml.expval(qml.PauliZ(0))

        expected = "0: ──X──X──X─┤  <Z>"
        assert draw(circ)() == expected

    def test_blocking_multiwire_gate(self):
        """Test gate gets blocked by multi-wire gate."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circ():
            qml.PauliX(0)
            qml.IsingXX(1.234, wires=(0, 2))
            qml.PauliX(1)
            return qml.expval(qml.PauliZ(0))

        expected = (
            "0: ──X─╭IsingXX(1.23)────┤  <Z>\n"
            "1: ────│───────────────X─┤     \n"
            "2: ────╰IsingXX(1.23)────┤     "
        )
        assert draw(circ)() == expected


class TestMidCircuitMeasurements:
    """Tests for drawing mid-circuit measurements and classical conditions."""

    @pytest.mark.parametrize("device_name", ["default.qubit"])
    def test_qnode_mid_circuit_measurement_not_deferred(self, device_name, mocker):
        """Test that a circuit containing mid-circuit measurements is transformed by the drawer
        to use deferred measurements if the device uses the new device API."""
        dev = qml.device(device_name)

        @qml.qnode(dev)
        def circ():
            qml.PauliX(0)
            qml.measure(0)
            return qml.probs(wires=0)

        draw_qnode = qml.draw(circ)
        spy = mocker.spy(qml.defer_measurements, "_transform")

        drawing = draw_qnode()
        spy.assert_not_called()

        assert drawing == "0: ──X──┤↗├─┤  Probs"

    @pytest.mark.parametrize(
        "postselect, reset, mid_measure_label",
        [
            (None, False, "┤↗├"),
            (None, True, "┤↗│  │0⟩"),
            (0, False, "┤↗₀├"),
            (0, True, "┤↗₀│  │0⟩"),
            (1, False, "┤↗₁├"),
            (1, True, "┤↗₁│  │0⟩"),
        ],
    )
    def test_draw_mid_circuit_measurement(self, postselect, reset, mid_measure_label):
        """Test that mid-circuit measurements are drawn correctly."""

        def func():
            qml.Hadamard(0)
            qml.measure(0, reset=reset, postselect=postselect)
            qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(func)()
        expected_drawing = "0: ──H──" + mid_measure_label + "──X─┤  <Z>"

        assert drawing == expected_drawing

    def test_draw_mid_circuit_measurement_multiple_wires(self):
        """Test that mid-circuit measurements are correctly drawn in circuits
        with multiple wires."""

        def circ(weights):
            qml.RX(weights[0], 0)
            qml.measure(0, reset=True)
            qml.RX(weights[1], 1)
            qml.measure(1)
            qml.CNOT([0, 3])
            qml.measure(3, postselect=0, reset=True)
            qml.RY(weights[2], 2)
            qml.CNOT([1, 2])
            qml.measure(2, postselect=1)
            qml.MultiRZ(0.5, [0, 2])
            return qml.expval(qml.PauliZ(2))

        drawing = qml.draw(circ)(np.array([np.pi, 3.124, 0.456]))
        expected_drawing = (
            "0: ──RX(3.14)──┤↗│  │0⟩─╭●─────────────────────╭MultiRZ(0.50)─┤     \n"
            "1: ──RX(3.12)──┤↗├──────│─────────────╭●───────│──────────────┤     \n"
            "3: ─────────────────────╰X──┤↗₀│  │0⟩─│────────│──────────────┤     \n"
            "2: ──RY(0.46)─────────────────────────╰X──┤↗₁├─╰MultiRZ(0.50)─┤  <Z>"
        )

        assert drawing == expected_drawing

    def test_single_meas_single_cond(self):
        """Test that a circuit with a single classical condition and a single measurement
        is drawn correctly."""

        def circ(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliX)(wires=1)

        drawing = qml.draw(circ)(np.pi)
        expected_drawing = (
            "0: ──RX(3.14)──┤↗├────┤  \n1: ─────────────║───X─┤  \n                ╚═══╝    "
        )

        assert drawing == expected_drawing

    def test_single_meas_single_cond_multi_wire(self):
        """Test that a multi-wire conditional operator relying on a single
        mid-circuit measurement is drawn correctly."""

        def circ(phi, theta):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RY(theta, 2)
            qml.cond(m0, qml.CNOT)(wires=[1, 0])

        drawing = qml.draw(circ)(np.pi, np.pi / 2)
        expected_drawing = (
            "0: ──RX(3.14)──┤↗├───────────╭X─┤  \n"
            "2: ─────────────║───RY(1.57)─│──┤  \n"
            "1: ─────────────║────────────╰●─┤  \n"
            "                ╚═════════════╝    "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_with_reset(self):
        """Test that a circuit where a single mid-circuit measurement is used for multiple
        conditions is drawn correctly"""

        def circ():
            qml.RX(0.5, 0)
            qml.RX(0.5, 1)
            m0 = qml.measure(1, reset=True)
            qml.cond(m0, qml.MultiControlledX)(wires=[1, 2, 0], control_values="10")
            qml.CNOT([3, 2])
            qml.cond(m0, qml.ctrl(qml.MultiRZ, control=[1, 2], control_values=[True, False]))(
                0.5, wires=[0, 3]
            )
            qml.cond(m0, qml.PauliX)(0)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──RX(0.50)───────────╭X────╭MultiRZ(0.50)──X─┤  <Z>\n"
            "1: ──RX(0.50)──┤↗│  │0⟩─├●────├●──────────────║─┤     \n"
            "2: ─────────────║───────╰○─╭X─├○──────────────║─┤     \n"
            "3: ─────────────║────────║─╰●─╰MultiRZ(0.50)──║─┤     \n"
            "                ╚════════╩═════╩══════════════╝       "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_with_postselection(self):
        """Test that a circuit where a single mid-circuit measurement is used for multiple
        conditions is drawn correctly"""

        def circ():
            qml.RX(0.5, 0)
            qml.RX(0.5, 1)
            m0 = qml.measure(1, postselect=1)
            qml.cond(m0, qml.MultiControlledX)(wires=[1, 2, 0], control_values="10")
            qml.CNOT([3, 2])
            qml.cond(m0, qml.ctrl(qml.MultiRZ, control=[1, 2], control_values=[True, False]))(
                0.5, wires=[0, 3]
            )
            qml.cond(m0, qml.PauliX)(0)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──RX(0.50)───────╭X────╭MultiRZ(0.50)──X─┤  <Z>\n"
            "1: ──RX(0.50)──┤↗₁├─├●────├●──────────────║─┤     \n"
            "2: ─────────────║───╰○─╭X─├○──────────────║─┤     \n"
            "3: ─────────────║────║─╰●─╰MultiRZ(0.50)──║─┤     \n"
            "                ╚════╩═════╩══════════════╝       "
        )

        assert drawing == expected_drawing

    def test_multi_meas_single_cond(self):
        """Test that a circuit is drawn correctly when a single conditioned operation relies on
        multiple mid-circuit measurements."""

        def circ():
            qml.RX(0.5, 0)
            qml.RX(0.5, 1)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.CNOT([1, 2])
            m2 = qml.measure(2)
            qml.cond(m0 & m1 & m2, qml.RZ)(1.23, 1)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──RX(0.50)──┤↗├────────────────────────┤  <Z>\n"
            "1: ──RX(0.50)───║───┤↗├─╭●───────RZ(1.23)─┤     \n"
            "2: ─────────────║────║──╰X──┤↗├──║────────┤     \n"
            "                ╚════║═══════║═══╣              \n"
            "                     ╚═══════║═══╣              \n"
            "                             ╚═══╝              "
        )

        assert drawing == expected_drawing

    def test_multi_meas_multi_cond(self):
        """Test that a circuit is drawn correctly when there are multiple conditional operations
        that rely on single or multiple mid-circuit measurements."""

        def circ():
            qml.RX(0.5, 0)
            qml.RX(0.5, 1)

            m0 = qml.measure(0, reset=True, postselect=1)
            m1 = qml.measure(1)
            qml.cond(m0 & m1, qml.MultiControlledX)(wires=[1, 2, 3, 0], control_values="110")
            qml.cond(m1, qml.PauliZ)(2)

            m2 = qml.measure(1)
            qml.CNOT([3, 2])
            qml.cond(m0, qml.ctrl(qml.SWAP, control=[1, 2], control_values=[True, False]))(
                wires=[0, 3]
            )
            qml.cond(m2 & m1, qml.PauliY)(0)

            qml.Toffoli([3, 1, 0])

            m3 = qml.measure(3, postselect=0)
            qml.cond(m3, qml.RX)(1.23, 3)
            qml.cond(m3 & m1, qml.Hadamard)(2)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──RX(0.50)──┤↗₁│  │0⟩──────╭X────────────╭SWAP──Y─╭X────────────────────┤ ╭<Z@Z>\n"
            "1: ──RX(0.50)───║─────────┤↗├─├●─────┤↗├────├●─────║─├●────────────────────┤ │     \n"
            "2: ─────────────║──────────║──├●──Z───║──╭X─├○─────║─│───────────────────H─┤ ╰<Z@Z>\n"
            "3: ─────────────║──────────║──╰○──║───║──╰●─╰SWAP──║─╰●──┤↗₀├──RX(1.23)──║─┤       \n"
            "                ╚══════════║═══╬══║═══║══════╝     ║      ║    ║         ║         \n"
            "                           ╚═══╩══╩═══║════════════╬══════║════║═════════╣         \n"
            "                                      ╚════════════╝      ║    ║         ║         \n"
            "                                                          ╚════╩═════════╝         "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_split_lines(self):
        """Test that a circuit is when multiple lines are needed and the measurement
        and condition are on different lines."""

        def circ():
            m0 = qml.measure(0)
            qml.RX(0.0, 0)
            qml.cond(m0, qml.RX)(0.123, 1)
            qml.RX(0.0, 0)
            qml.cond(m0, qml.RX)(0.123, 1)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(circ, max_length=25)()
        expected_drawing = (
            "0: ──┤↗├──RX(0.00)\n"
            "1: ───║───RX(0.12)\n"
            "      ╚═══╩═══════\n\n"
            "───RX(0.00)─┤  <Z>\n"
            "───RX(0.12)─┤     \n"
            "═══╝              "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_new_line(self):
        """Test that a circuit is when multiple lines are needed and the measurement
        and condition are on the same lines after the first line."""

        def circ():
            qml.RX(0.0, 0)
            qml.RX(0.0, 0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliX)(1)
            qml.cond(m0, qml.PauliX)(1)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(circ, max_length=25)()
        expected_drawing = (
            "0: ──RX(0.00)──RX(0.00)\n"
            "1: ────────────────────\n"
            "                       \n\n"
            "───┤↗├───────┤  <Z>\n"
            "────║───X──X─┤     \n"
            "    ╚═══╩══╝       "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_first_line(self):
        """Test that a circuit is drawn correctly when multiple lines are needed and the
        measurement and condition are on the same lines in the first line."""

        def circ():
            m0 = qml.measure(0)
            qml.cond(m0, qml.RX)(0.123, 1)
            qml.cond(m0, qml.PauliX)(1)
            qml.RX(0.0, 0)
            qml.RX(0.0, 1)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(circ, max_length=25)()
        expected_drawing = (
            "0: ──┤↗├──RX(0.00)───\n"
            "1: ───║───RX(0.12)──X\n"
            "      ╚═══╩═════════╝\n\n"
            "────────────┤  <Z>\n"
            "───RX(0.00)─┤     \n"
            "                  "
        )

        assert drawing == expected_drawing

    def test_multi_meas_single_cond_split_lines(self):
        """Test that a circuit is drawn correctly when multiple lines are needed and the
        measurements and condition are split between the lines."""

        def circ():
            qml.RX(0.5, 0)
            qml.RX(0.5, 1)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            qml.CNOT([1, 2])
            m2 = qml.measure(2)
            qml.cond(m0 & m1 & m2, qml.RZ)(1.23, 1)
            return qml.expval(qml.PauliZ(0))

        drawing = qml.draw(circ, max_length=30)()
        expected_drawing = (
            "0: ──RX(0.50)──┤↗├────────\n"
            "1: ──RX(0.50)───║───┤↗├─╭●\n"
            "2: ─────────────║────║──╰X\n"
            "                ╚════║════\n"
            "                     ╚════\n"
            "                          \n\n"
            "─────────────────┤  <Z>\n"
            "────────RZ(1.23)─┤     \n"
            "───┤↗├──║────────┤     \n"
            "════║═══╣              \n"
            "════║═══╣              \n"
            "    ╚═══╝              "
        )

        assert drawing == expected_drawing

    def test_multi_meas_multi_cond_split_lines(self):
        """Test that a circuit is drawn correctly when multiple lines are needed and the
        measurements and conditions are split between the lines."""

        def circ():
            qml.RX(0.5, 0)
            qml.RX(0.5, 1)

            m0 = qml.measure(0, reset=True, postselect=1)
            m1 = qml.measure(1)
            qml.cond(m0 & m1, qml.MultiControlledX)(wires=[1, 2, 3, 0], control_values="110")
            qml.cond(m1, qml.PauliZ)(2)

            m2 = qml.measure(1)
            qml.CNOT([3, 2])
            qml.cond(m0, qml.ctrl(qml.SWAP, control=[1, 2], control_values=[True, False]))(
                wires=[0, 3]
            )
            qml.cond(m2 & m1, qml.PauliY)(0)

            qml.Toffoli([3, 1, 0])

            m3 = qml.measure(3, postselect=0)
            qml.cond(m3, qml.RX)(1.23, 3)
            qml.cond(m3 & m1, qml.Hadamard)(2)

            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

        drawing = qml.draw(circ, max_length=60)()
        expected_drawing = (
            "0: ──RX(0.50)──┤↗₁│  │0⟩──────╭X────────────╭SWAP──Y─╭X\n"
            "1: ──RX(0.50)───║─────────┤↗├─├●─────┤↗├────├●─────║─├●\n"
            "2: ─────────────║──────────║──├●──Z───║──╭X─├○─────║─│─\n"
            "3: ─────────────║──────────║──╰○──║───║──╰●─╰SWAP──║─╰●\n"
            "                ╚══════════║═══╬══║═══║══════╝     ║   \n"
            "                           ╚═══╩══╩═══║════════════╬═══\n"
            "                                      ╚════════════╝   \n"
            "                                                       \n\n"
            "─────────────────────┤ ╭<Z@Z>\n"
            "─────────────────────┤ │     \n"
            "───────────────────H─┤ ╰<Z@Z>\n"
            "───┤↗₀├──RX(1.23)──║─┤       \n"
            "    ║    ║         ║         \n"
            "════║════║═════════╣         \n"
            "    ║    ║         ║         \n"
            "    ╚════╩═════════╝         "
        )

        assert drawing == expected_drawing

    @pytest.mark.parametrize(
        "mp, mp_label",
        [
            (qml.expval, "<MCM>"),
            (qml.var, "Var[MCM]"),
            (qml.probs, "Probs[MCM]"),
            (qml.sample, "Sample[MCM]"),
            (qml.counts, "Counts[MCM]"),
        ],
    )
    def test_single_meas_stats(self, mp, mp_label):
        """Test that collecting statistics on a single mid-circuit measurement
        works as expected."""

        def circ():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.Hadamard(0)
            return mp(op=m0)

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──H──┤↗├──H─┤  " + " " * len(mp_label) + f"\n         ╚═════╡  {mp_label}"
        )

        assert drawing == expected_drawing

    @pytest.mark.parametrize(
        "mp, mp_label",
        [
            (qml.expval, "<MCM>"),
            (qml.var, "Var[MCM]"),
            (qml.sample, "Sample[MCM]"),
            (qml.counts, "Counts[MCM]"),
        ],
    )
    def test_multi_meas_stats(self, mp, mp_label):
        """Test that collecting statistics on multiple mid-circuit measurements
        works as expected"""

        def circ():
            qml.Hadamard(0)
            qml.Hadamard(1)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            return mp(op=m0 + m1)

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──H──┤↗├──────┤  "
            + " " * len(mp_label)
            + "\n1: ──H───║───┤↗├─┤  "
            + " " * len(mp_label)
            + f"\n         ╚════║══╡ ╭{mp_label}"
            + f"\n              ╚══╡ ╰{mp_label}"
        )

        assert drawing == expected_drawing

    def test_multi_meas_stats_multi_meas(self):
        """Test that collecting statistics on multiple mid-circuit measurements with
        multiple terminal measurement processes works as expected."""

        def circ():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.Hadamard(1)
            m1 = qml.measure(1)
            qml.Hadamard(2)
            m2 = qml.measure(2)
            return qml.expval(m0 * m2), qml.sample(m1)

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──H──┤↗├─────────────────┤                    \n"
            "1: ──────║───H──┤↗├─────────┤                    \n"
            "2: ──────║───────║───H──┤↗├─┤                    \n"
            "         ╚═══════║═══════║══╡ ╭<MCM>             \n"
            "                 ╚═══════║══╡ │       Sample[MCM]\n"
            "                         ╚══╡ ╰<MCM>             "
        )

        assert drawing == expected_drawing

    def test_multi_meas_stats_same_cwire(self):
        """Test that colecting statistics on multiple mid-circuit measurements
        with multiple terminal measurements is drawn correctly"""

        def circ():
            qml.Hadamard(0)
            qml.Hadamard(1)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            return qml.expval(m0 + m1), qml.sample(m0)

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──H──┤↗├──────┤                    \n"
            "1: ──H───║───┤↗├─┤                    \n"
            "         ╚════║══╡ ╭<MCM>  Sample[MCM]\n"
            "              ╚══╡ ╰<MCM>             "
        )

        assert drawing == expected_drawing

    def test_single_cond_single_meas_stats(self):
        """Test that collecting statistics and using classical conditions together
        on the same measurement works as expected."""

        def circ():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliZ)(0)
            return qml.expval(m0)

        drawing = qml.draw(circ)()
        expected_drawing = "0: ──H──┤↗├──Z─┤       \n         ╚═══╩═╡  <MCM>"

        assert drawing == expected_drawing

    def test_multi_cond_multi_meas_stats(self):
        """Test that combining multiple conditionals and multiple mid-circuit
        measurement statistics is drawn correctly."""

        def circ():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            qml.Hadamard(1)
            m1 = qml.measure(1)
            qml.Hadamard(2)
            m2 = qml.measure(2)
            qml.Hadamard(3)
            qml.measure(3)
            qml.cond(m0, qml.PauliX)(0)
            qml.cond(m0 & m1, qml.PauliY)(1)
            return qml.expval(m2), qml.sample([m1, m2])

        drawing = qml.draw(circ)()
        expected_drawing = (
            "0: ──H──┤↗├──────────────────────────X────┤                    \n"
            "1: ──────║───H──┤↗├──────────────────║──Y─┤                    \n"
            "2: ──────║───────║───H──┤↗├──────────║──║─┤                    \n"
            "3: ──────║───────║───────║───H──┤↗├──║──║─┤                    \n"
            "         ╚═══════║═══════║═══════════╩══╣                      \n"
            "                 ╚═══════║══════════════╩═╡        ╭Sample[MCM]\n"
            "                         ╚════════════════╡  <MCM> ╰Sample[MCM]"
        )

        assert drawing == expected_drawing


def test_draw_batch_transform():
    """Test that drawing a batch transform works correctly."""

    @partial(qml.gradients.param_shift, shifts=[(0.2,)])
    @qml.qnode(qml.device("default.qubit", wires=1))
    def circ(x):
        qml.Hadamard(wires=0)
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    expected = "0: ──H──RX(0.8)─┤  <Z>\n\n0: ──H──RX(0.4)─┤  <Z>"
    assert draw(circ, decimals=1)(np.array(0.6, requires_grad=True)) == expected


@pytest.mark.skip("Nested tapes are being deprecated")
def test_nested_tapes():
    """Test nested tapes inside the qnode."""

    @qml.qnode(qml.device("default.qubit", wires=1))
    def circ():
        with qml.queuing.AnnotatedQueue():
            qml.PauliX(0)
            with qml.queuing.AnnotatedQueue():
                qml.PauliY(0)
        with qml.queuing.AnnotatedQueue():
            qml.PauliZ(0)
            with qml.queuing.AnnotatedQueue():
                qml.PauliX(0)
        return qml.expval(qml.PauliZ(0))

    expected = (
        "0: ──Tape:0──Tape:1─┤  <Z>\n\n"
        "Tape:0\n0: ──X──Tape:2─┤  \n\n"
        "Tape:2\n0: ──Y─┤  \n\n"
        "Tape:1\n0: ──Z──Tape:3─┤  \n\n"
        "Tape:3\n0: ──X─┤  "
    )

    assert draw(circ)() == expected


@pytest.mark.parametrize(
    "device",
    [qml.device("default.qubit.legacy", wires=2), qml.devices.DefaultQubit(wires=2)],
)
def test_expansion_strategy(device):
    """Test expansion strategy keyword modifies tape expansion."""

    H = qml.PauliX(0) + qml.PauliZ(1) + 0.5 * qml.PauliX(0) @ qml.PauliX(1)

    @qml.qnode(device)
    def circ(t):
        qml.ApproxTimeEvolution(H, t, 2)
        return qml.probs(wires=0)

    expected_gradient = "0: ─╭ApproxTimeEvolution─┤  Probs\n1: ─╰ApproxTimeEvolution─┤       "
    assert draw(circ, expansion_strategy="gradient", decimals=None)(0.5) == expected_gradient

    expected_device = "0: ──RX─╭RXX──RX─╭RXX─┤  Probs\n1: ──RZ─╰RXX──RZ─╰RXX─┤       "
    assert draw(circ, expansion_strategy="device", decimals=None)(0.5) == expected_device


@pytest.mark.parametrize(
    "device",
    [qml.device("default.qubit.legacy", wires=2), qml.device("default.qubit", wires=2)],
)
def test_applied_transforms(device):
    """Test that any transforms applied to the qnode are included in the output."""

    @qml.transform
    def just_pauli_x(_):
        new_tape = qml.tape.QuantumScript([qml.PauliX(0)])
        return (new_tape,), lambda res: res[0]

    @just_pauli_x
    @qml.qnode(device)
    def my_circuit(x):
        qml.RX(x, wires=0)
        qml.SWAP(wires=(0, 1))
        return qml.probs(wires=(0, 1))

    expected = "0: ──X─┤  "
    assert qml.draw(my_circuit)(1.234) == expected


def test_draw_with_qfunc():
    """Test a non-qnode qfunc can be drawn."""

    def qfunc(x):
        qml.RX(x, wires=[0])
        qml.PauliZ(1)

    assert qml.draw(qfunc)(1.1) == "0: ──RX(1.10)─┤  \n1: ──Z────────┤  "


def test_draw_with_qfunc_with_measurements():
    """Test a non-qnode qfunc with measurements can be drawn."""

    def qfunc(x):
        qml.RX(x, wires=[0])
        qml.CNOT([0, 1])
        return qml.expval(qml.PauliZ(1))

    assert qml.draw(qfunc)(1.1) == "0: ──RX(1.10)─╭●─┤     \n1: ───────────╰X─┤  <Z>"


def test_draw_with_qfunc_warns_with_expansion_strategy():
    """Test that draw warns the user about expansion_strategy being ignored."""

    def qfunc():
        qml.PauliZ(0)

    with pytest.warns(UserWarning, match="the expansion_strategy argument is ignored"):
        _ = qml.draw(qfunc, expansion_strategy="gradient")
