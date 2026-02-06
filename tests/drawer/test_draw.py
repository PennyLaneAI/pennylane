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
import pytest

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.drawer import draw


@qp.qnode(qp.device("default.qubit", wires=(0, "a", 1.234)))
def circuit(x, y, z):
    """A quantum circuit on three wires."""
    qp.RX(x, wires=0)
    qp.RY(y, wires="a")
    qp.RZ(z, wires=1.234)
    return qp.expval(qp.PauliZ(0))


class TestLabelling:
    """Test the wire labels."""

    def test_any_wire_labels(self):
        """Test wire labels with different kinds of objects."""

        split_str = draw(circuit)(1.2, 2.3, 3.4).split("\n")
        assert split_str[0][:6] == "    0:"
        assert split_str[1][:6] == "    a:"
        assert split_str[2][:6] == "1.234:"

    @pytest.mark.parametrize("as_qnode", (True, False))
    def test_wire_order(self, as_qnode):
        """Test wire_order keyword changes order of the wires."""

        def f(x, y, z):
            """A quantum circuit on three wires."""
            qp.RX(x, wires=0)
            qp.RY(y, wires="a")
            qp.RZ(z, wires=1.234)
            return qp.expval(qp.PauliZ(0))

        if as_qnode:
            f = qp.QNode(f, qp.device("default.qubit", wires=(0, "a", 1.234)))

        split_str = draw(f, wire_order=[1.234, "a", 0, "b"])(1.2, 2.3, 3.4).split("\n")
        assert split_str[0][:6] == "1.234:"
        assert split_str[1][:6] == "    a:"
        assert split_str[2][:6] == "    0:"

    def test_show_all_wires(self):
        """Test show_all_wires=True forces empty wires to display."""

        @qp.qnode(qp.device("default.qubit", wires=(0, 1)))
        def circ():
            return qp.expval(qp.PauliZ(0))

        split_str = draw(circ, show_all_wires=True)().split("\n")
        assert split_str[0][:2] == "0:"
        assert split_str[1][:2] == "1:"

    def test_show_all_wires_and_wire_order(self):
        """Test show_all_wires forces empty wires to display when empty wire is in wire order."""

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circ():
            return qp.expval(qp.PauliZ(0))

        split_str = draw(circ, wire_order=[0, "a"], show_all_wires=True)().split("\n")
        assert split_str[0][:2] == "0:"
        assert split_str[1][:2] == "a:"

    def test_hiding_labels(self):
        """Test that printing wire labels can be skipped with show_wire_labels=False."""

        @qp.qnode(qp.device("default.qubit"))
        def circ():
            return qp.expval(qp.Z(0) @ qp.X(1))

        split_str = draw(circ, show_wire_labels=False)().split("\n")
        assert split_str[0].startswith("─")
        assert split_str[1].startswith("─")


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
        assert qp.draw(circuit, decimals=4)(1, 2, 3) == out

    def test_decimals_multiparameters(self):
        """Test decimals also displays parameters when the operation has multiple parameters."""

        @qp.qnode(qp.device("default.qubit", wires=[0]))
        def circ(x):
            qp.Rot(*x, wires=0)
            return qp.expval(qp.PauliZ(0))

        expected = "0: ──Rot(1.2,2.3,3.5)─┤  <Z>"
        assert draw(circ, decimals=1)([1.234, 2.345, 3.456]) == expected

    def test_decimals_0(self):
        """Test decimals=0 rounds to integers."""

        expected = "    0: ──RX(1)─┤  <Z>\n    a: ──RY(2)─┤     \n1.234: ──RZ(3)─┤     "
        assert draw(circuit, decimals=0)(1.234, 2.3456, 3.456) == expected

    def test_qp_numpy_parameters(self):
        """Test numpy parameters display as normal numbers."""

        expected = "    0: ──RX(1.00)─┤  <Z>\n    a: ──RY(2.00)─┤     \n1.234: ──RZ(3.00)─┤     "
        assert draw(circuit)(pnp.array(1), pnp.array(2), pnp.array(3)) == expected

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

        @qp.qnode(qp.device("default.qubit", wires=2))
        def matrices_circuit():
            qp.StatePrep([1.0, 0.0, 0.0, 0.0], wires=(0, 1))
            qp.QubitUnitary(pnp.eye(2), wires=0)
            return qp.expval(qp.Hermitian(pnp.eye(2), wires=0))

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

        @qp.gradients.param_shift(shifts=[(0.2,)])
        @qp.qnode(qp.device("default.qubit", wires=2))
        def matrices_circuit(x):
            qp.StatePrep([1.0, 0.0, 0.0, 0.0], wires=(0, 1))
            qp.QubitUnitary(pnp.eye(2, requires_grad=False), wires=0)
            qp.RX(x, wires=1)
            return qp.expval(qp.Hermitian(pnp.eye(2, requires_grad=False), wires=1))

        expected1 = (
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(1.20)─┤  <𝓗(M0)>\n\n"
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(0.80)─┤  <𝓗(M0)>\n\n"
            "M0 = \n[[1. 0.]\n [0. 1.]]"
        )
        output = draw(matrices_circuit, level="gradient")(pnp.array(1.0, requires_grad=True))
        assert output == expected1

        expected2 = (
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(1.20)─┤  <𝓗(M0)>\n\n"
            "0: ─╭|Ψ⟩──U(M0)────┤         \n"
            "1: ─╰|Ψ⟩──RX(0.80)─┤  <𝓗(M0)>"
        )
        output = draw(matrices_circuit, show_matrices=False)(pnp.array(1.0, requires_grad=True))
        assert output == expected2


class TestMaxLength:
    """Test the max_length keyword."""

    def test_max_length_default(self):
        """Test max length default to 100."""

        @qp.qnode(qp.device("default.qubit", wires=1))
        def long_circuit():
            for _ in range(100):
                qp.PauliX(0)
            return qp.expval(qp.PauliZ(0))

        out = draw(long_circuit)()

        assert 95 <= max(len(s) for s in out.split("\n")) <= 100

    # We choose values of max_length that allow us to include continuation dots
    # when the circuit is partitioned
    @pytest.mark.parametrize("ml", [25, 50, 75])
    def test_setting_max_length(self, ml):
        """Test that setting a maximal length works as expected."""

        @qp.qnode(qp.device("default.qubit", wires=1))
        def long_circuit():
            for _ in range(10):
                qp.PauliX(0)
            return [qp.expval(qp.PauliZ(0)) for _ in range(4)]

        out = draw(long_circuit, max_length=ml)()

        assert max(len(s) for s in out.split("\n")) <= ml


class TestLayering:
    """Test operations are placed in the correct locations."""

    def test_adjacent_ops(self):
        """Test non-blocking gates end up on same layer."""

        @qp.qnode(qp.device("default.qubit", wires=3))
        def circ():
            _ = [qp.PauliX(i) for i in range(3)]
            return [qp.expval(qp.PauliZ(i)) for i in range(3)]

        expected = "0: ──X─┤  <Z>\n1: ──X─┤  <Z>\n2: ──X─┤  <Z>"
        assert draw(circ)() == expected

    def test_blocking_ops(self):
        """Test single qubits gates on the same wire block each other."""

        @qp.qnode(qp.device("default.qubit", wires=1))
        def circ():
            _ = [qp.PauliX(0) for i in range(3)]
            return qp.expval(qp.PauliZ(0))

        expected = "0: ──X──X──X─┤  <Z>"
        assert draw(circ)() == expected

    def test_blocking_multiwire_gate(self):
        """Test gate gets blocked by multi-wire gate."""

        @qp.qnode(qp.device("default.qubit", wires=3))
        def circ():
            qp.PauliX(0)
            qp.IsingXX(1.234, wires=(0, 2))
            qp.PauliX(1)
            return qp.expval(qp.PauliZ(0))

        expected = (
            "0: ──X─╭IsingXX(1.23)────┤  <Z>\n"
            "1: ────│───────────────X─┤     \n"
            "2: ────╰IsingXX(1.23)────┤     "
        )
        assert draw(circ)() == expected


class TestMidCircuitMeasurements:
    """Tests for drawing mid-circuit measurements and classical conditions."""

    # pylint: disable=too-many-public-methods

    @pytest.mark.parametrize("device_name", ["default.qubit"])
    def test_qnode_mid_circuit_measurement_not_deferred(self, device_name, mocker):
        """Test that a circuit containing mid-circuit measurements is transformed by the drawer
        to use deferred measurements if the device uses the new device API."""
        dev = qp.device(device_name)

        @qp.qnode(dev)
        def circ():
            qp.PauliX(0)
            qp.measure(0)
            return qp.probs(wires=0)

        draw_qnode = qp.draw(circ)
        spy = mocker.spy(qp.defer_measurements, "_tape_transform")

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
            qp.Hadamard(0)
            qp.measure(0, reset=reset, postselect=postselect)
            qp.PauliX(0)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(func)()
        expected_drawing = "0: ──H──" + mid_measure_label + "──X─┤  <Z>"

        assert drawing == expected_drawing

    @pytest.mark.parametrize(
        "op, grouped",
        [
            (qp.GlobalPhase(0.1), True),
            (qp.Identity(), True),
            (qp.Snapshot(), False),
            (qp.Barrier(), False),
        ],
    )
    @pytest.mark.parametrize("decimals", [None, 2])
    def test_draw_all_wire_ops(self, op, grouped, decimals):
        """Test that operators acting on all wires are drawn correctly"""

        def func():
            qp.X(0)
            qp.X(1)
            m = qp.measure(0)
            qp.cond(m, qp.X)(0)
            qp.apply(op)
            return qp.expval(qp.Z(0))

        # Stripping to remove trailing white-space because length of white-space at the
        # end of the drawing depends on the length of each individual line
        drawing = qp.draw(func, decimals=decimals)().strip()
        label = op.label(decimals=decimals).replace("\n", "")
        if grouped:
            expected_drawing = (
                f"0: ──X──┤↗├──X─╭{label}─┤  <Z>\n1: ──X───║───║─╰{label}─┤     \n         ╚═══╝"
            )
        else:
            expected_drawing = (
                f"0: ──X──┤↗├──X──{label}─┤  <Z>\n1: ──X───║───║──{label}─┤     \n         ╚═══╝"
            )

        assert drawing == expected_drawing

    @pytest.mark.parametrize(
        "mp, label", [(qp.sample(), "Sample"), (qp.probs(), "Probs"), (qp.counts(), "Counts")]
    )
    def test_draw_all_wire_measurements(self, mp, label):
        """Test that operators acting on all wires are drawn correctly"""

        def func():
            qp.X(0)
            qp.X(1)
            m = qp.measure(0)
            qp.cond(m, qp.X)(0)
            return qp.apply(mp)

        # Stripping to remove trailing white-space because length of white-space at the
        # end of the drawing depends on the length of each individual line
        drawing = qp.draw(func)().strip()
        expected_drawing = f"0: ──X──┤↗├──X─┤  {label}\n1: ──X───║───║─┤  {label}\n         ╚═══╝"

        assert drawing == expected_drawing

    def test_draw_mid_circuit_measurement_multiple_wires(self):
        """Test that mid-circuit measurements are correctly drawn in circuits
        with multiple wires."""

        def circ(weights):
            qp.RX(weights[0], 0)
            qp.measure(0, reset=True)
            qp.RX(weights[1], 1)
            qp.measure(1)
            qp.CNOT([0, 3])
            qp.measure(3, postselect=0, reset=True)
            qp.RY(weights[2], 2)
            qp.CNOT([1, 2])
            qp.measure(2, postselect=1)
            qp.MultiRZ(0.5, [0, 2])
            return qp.expval(qp.PauliZ(2))

        drawing = qp.draw(circ)(pnp.array([pnp.pi, 3.124, 0.456]))
        expected_drawing = (
            "0: ──RX(3.14)──┤↗│  │0⟩─╭●─────────────────────╭MultiRZ(0.50)─┤     \n"
            "1: ──RX(3.12)──┤↗├──────│─────────────╭●───────│──────────────┤     \n"
            "2: ─────────────────────│───RY(0.46)──╰X──┤↗₁├─╰MultiRZ(0.50)─┤  <Z>\n"
            "3: ─────────────────────╰X──┤↗₀│  │0⟩─────────────────────────┤     "
        )

        assert drawing == expected_drawing

    def test_single_meas_single_cond(self):
        """Test that a circuit with a single classical condition and a single measurement
        is drawn correctly."""

        def circ(phi):
            qp.RX(phi, 0)
            m0 = qp.measure(0)
            qp.cond(m0, qp.PauliX)(wires=1)

        drawing = qp.draw(circ)(pnp.pi)
        expected_drawing = (
            "0: ──RX(3.14)──┤↗├────┤  \n1: ─────────────║───X─┤  \n                ╚═══╝    "
        )

        assert drawing == expected_drawing

    def test_single_meas_single_cond_multi_wire(self):
        """Test that a multi-wire conditional operator relying on a single
        mid-circuit measurement is drawn correctly."""

        def circ(phi, theta):
            qp.RX(phi, 0)
            m0 = qp.measure(0)
            qp.RY(theta, 2)
            qp.cond(m0, qp.CNOT)(wires=[1, 0])

        drawing = qp.draw(circ)(pnp.pi, pnp.pi / 2)
        expected_drawing = (
            "0: ──RX(3.14)──┤↗├───────────╭X─┤  \n"
            "1: ─────────────║────────────╰●─┤  \n"
            "2: ─────────────║───RY(1.57)──║─┤  \n"
            "                ╚═════════════╝    "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_with_reset(self):
        """Test that a circuit where a single mid-circuit measurement is used for multiple
        conditions is drawn correctly"""

        def circ():
            qp.RX(0.5, 0)
            qp.RX(0.5, 1)
            m0 = qp.measure(1, reset=True)
            qp.cond(m0, qp.MultiControlledX)(wires=[1, 2, 0], control_values=[1, 0])
            qp.CNOT([3, 2])
            qp.cond(m0, qp.ctrl(qp.MultiRZ, control=[1, 2], control_values=[True, False]))(
                0.5, wires=[0, 3]
            )
            qp.cond(m0, qp.PauliX)(0)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(circ)()
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
            qp.RX(0.5, 0)
            qp.RX(0.5, 1)
            m0 = qp.measure(1, postselect=1)
            qp.cond(m0, qp.MultiControlledX)(wires=[1, 2, 0], control_values=[1, 0])
            qp.CNOT([3, 2])
            qp.cond(m0, qp.ctrl(qp.MultiRZ, control=[1, 2], control_values=[True, False]))(
                0.5, wires=[0, 3]
            )
            qp.cond(m0, qp.PauliX)(0)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(circ)()
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
            qp.RX(0.5, 0)
            qp.RX(0.5, 1)
            m0 = qp.measure(0)
            m1 = qp.measure(1)
            qp.CNOT([1, 2])
            m2 = qp.measure(2)
            qp.cond(m0 & m1 & m2, qp.RZ)(1.23, 1)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(circ)()
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
            qp.RX(0.5, 0)
            qp.RX(0.5, 1)

            m0 = qp.measure(0, reset=True, postselect=1)
            m1 = qp.measure(1)
            qp.cond(m0 & m1, qp.MultiControlledX)(wires=[1, 2, 3, 0], control_values=[1, 1, 0])
            qp.cond(m1, qp.PauliZ)(2)

            m2 = qp.measure(1)
            qp.CNOT([3, 2])
            qp.cond(m0, qp.ctrl(qp.SWAP, control=[1, 2], control_values=[True, False]))(
                wires=[0, 3]
            )
            qp.cond(m2 & m1, qp.PauliY)(0)

            qp.Toffoli([3, 1, 0])

            m3 = qp.measure(3, postselect=0)
            qp.cond(m3, qp.RX)(1.23, 3)
            qp.cond(m3 & m1, qp.Hadamard)(2)

            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(2))

        drawing = qp.draw(circ)()
        expected_drawing = (
            "0: ──RX(0.50)──┤↗₁│  │0⟩──────╭X────────────╭SWAP──Y─╭X────────────────────┤ ╭<Z@Z>\n"
            "1: ──RX(0.50)───║─────────┤↗├─├●─────┤↗├────├●─────║─├●────────────────────┤ │     \n"
            "2: ─────────────║──────────║──├●──Z───║──╭X─├○─────║─│───────────────────H─┤ ╰<Z@Z>\n"
            "3: ─────────────║──────────║──╰○──║───║──╰●─╰SWAP──║─╰●──┤↗₀├──RX(1.23)──║─┤       \n"
            "                ╚══════════║═══╬══║═══║══════╝     ║      ╚════╩═════════╣         \n"
            "                           ╚═══╩══╩═══║════════════╬═════════════════════╝         \n"
            "                                      ╚════════════╝                               "
        )
        assert drawing == expected_drawing

    def test_single_meas_multi_cond_split_lines(self):
        """Test that a circuit is drawn correctly when multiple lines are needed and the measurement
        and condition are on different lines."""

        def circ():
            m0 = qp.measure(0)
            qp.RX(0.0, 0)
            qp.cond(m0, qp.RX)(0.123, 1)
            qp.RX(0.0, 0)
            qp.cond(m0, qp.RX)(0.123, 1)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(circ, max_length=25)()
        expected_drawing = (
            "0: ──┤↗├──RX(0.00) ···\n"
            "1: ───║───RX(0.12) ···\n"
            "      ╚═══╩═══════\n\n"
            "0: ··· ──RX(0.00)─┤  <Z>\n"
            "1: ··· ──RX(0.12)─┤     \n"
            "       ══╝              "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_new_line(self):
        """Test that a circuit is when multiple lines are needed and the measurement
        and condition are on the same lines after the first line."""

        def circ():
            qp.RX(0.0, 0)
            qp.RX(0.0, 0)
            m0 = qp.measure(0)
            qp.cond(m0, qp.PauliX)(1)
            qp.cond(m0, qp.PauliX)(1)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(circ, max_length=27)()
        expected_drawing = (
            "0: ──RX(0.00)──RX(0.00) ···\n"
            "1: ──────────────────── ···\n"
            "                       \n\n"
            "0: ··· ──┤↗├───────┤  <Z>\n"
            "1: ··· ───║───X──X─┤     \n"
            "          ╚═══╩══╝       "
        )

        assert drawing == expected_drawing

    def test_single_meas_multi_cond_first_line(self):
        """Test that a circuit is drawn correctly when multiple lines are needed and the
        measurement and condition are on the same lines in the first line."""

        def circ():
            m0 = qp.measure(0)
            qp.cond(m0, qp.RX)(0.123, 1)
            qp.cond(m0, qp.PauliX)(1)
            qp.RX(0.0, 0)
            qp.RX(0.0, 1)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(circ, max_length=25)()
        expected_drawing = (
            "0: ──┤↗├──RX(0.00)─── ···\n"
            "1: ───║───RX(0.12)──X ···\n"
            "      ╚═══╩═════════╝\n\n"
            "0: ··· ───────────┤  <Z>\n"
            "1: ··· ──RX(0.00)─┤     \n"
            "                        "
        )

        assert drawing == expected_drawing

    def test_multi_meas_single_cond_split_lines(self):
        """Test that a circuit is drawn correctly when multiple lines are needed and the
        measurements and condition are split between the lines."""

        def circ():
            qp.RX(0.5, 0)
            qp.RX(0.5, 1)
            m0 = qp.measure(0)
            m1 = qp.measure(1)
            qp.CNOT([1, 2])
            m2 = qp.measure(2)
            qp.cond(m0 & m1 & m2, qp.RZ)(1.23, 1)
            return qp.expval(qp.PauliZ(0))

        drawing = qp.draw(circ, max_length=30)()
        expected_drawing = (
            "0: ──RX(0.50)──┤↗├──────── ···\n"
            "1: ──RX(0.50)───║───┤↗├─╭● ···\n"
            "2: ─────────────║────║──╰X ···\n"
            "                ╚════║════\n"
            "                     ╚════\n"
            "                          \n\n"
            "0: ··· ────────────────┤  <Z>\n"
            "1: ··· ───────RZ(1.23)─┤     \n"
            "2: ··· ──┤↗├──║────────┤     \n"
            "       ═══║═══╣              \n"
            "       ═══║═══╣              \n"
            "          ╚═══╝              "
        )

        assert drawing == expected_drawing

    def test_multi_meas_multi_cond_split_lines(self):
        """Test that a circuit is drawn correctly when multiple lines are needed and the
        measurements and conditions are split between the lines."""

        def circ():
            qp.RX(0.5, 0)
            qp.RX(0.5, 1)

            m0 = qp.measure(0, reset=True, postselect=1)
            m1 = qp.measure(1)
            qp.cond(m0 & m1, qp.MultiControlledX)(wires=[1, 2, 3, 0], control_values=[1, 1, 0])
            qp.cond(m1, qp.PauliZ)(2)

            m2 = qp.measure(1)
            qp.CNOT([3, 2])
            qp.cond(m0, qp.ctrl(qp.SWAP, control=[1, 2], control_values=[True, False]))(
                wires=[0, 3]
            )
            qp.cond(m2 & m1, qp.PauliY)(0)

            qp.Toffoli([3, 1, 0])

            m3 = qp.measure(3, postselect=0)
            qp.cond(m3, qp.RX)(1.23, 3)
            qp.cond(m3 & m1, qp.Hadamard)(2)

            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(2))

        drawing = qp.draw(circ, max_length=60)()
        expected_drawing = (
            "0: ──RX(0.50)──┤↗₁│  │0⟩──────╭X────────────╭SWAP──Y─╭X ···\n"
            "1: ──RX(0.50)───║─────────┤↗├─├●─────┤↗├────├●─────║─├● ···\n"
            "2: ─────────────║──────────║──├●──Z───║──╭X─├○─────║─│─ ···\n"
            "3: ─────────────║──────────║──╰○──║───║──╰●─╰SWAP──║─╰● ···\n"
            "                ╚══════════║═══╬══║═══║══════╝     ║   \n"
            "                           ╚═══╩══╩═══║════════════╬═══\n"
            "                                      ╚════════════╝   \n\n"
            "0: ··· ────────────────────┤ ╭<Z@Z>\n"
            "1: ··· ────────────────────┤ │     \n"
            "2: ··· ──────────────────H─┤ ╰<Z@Z>\n"
            "3: ··· ──┤↗₀├──RX(1.23)──║─┤       \n"
            "          ╚════╩═════════╣         \n"
            "       ══════════════════╝         \n"
            "                                   "
        )

        assert drawing == expected_drawing

    @pytest.mark.parametrize(
        "mp, mp_label",
        [
            (qp.expval, "<MCM>"),
            (qp.var, "Var[MCM]"),
            (qp.probs, "Probs[MCM]"),
            (qp.sample, "Sample[MCM]"),
            (qp.counts, "Counts[MCM]"),
        ],
    )
    def test_single_meas_stats(self, mp, mp_label):
        """Test that collecting statistics on a single mid-circuit measurement
        works as expected."""

        def circ():
            qp.Hadamard(0)
            m0 = qp.measure(0)
            qp.Hadamard(0)
            return mp(op=m0)

        drawing = qp.draw(circ)()
        expected_drawing = (
            "0: ──H──┤↗├──H─┤  " + " " * len(mp_label) + f"\n         ╚═════╡  {mp_label}"
        )

        assert drawing == expected_drawing

    @pytest.mark.parametrize(
        "mp, mp_label",
        [
            (qp.expval, "<MCM>"),
            (qp.var, "Var[MCM]"),
            (qp.sample, "Sample[MCM]"),
            (qp.counts, "Counts[MCM]"),
        ],
    )
    def test_multi_meas_stats(self, mp, mp_label):
        """Test that collecting statistics on multiple mid-circuit measurements
        works as expected"""

        def circ():
            qp.Hadamard(0)
            qp.Hadamard(1)
            m0 = qp.measure(0)
            m1 = qp.measure(1)
            return mp(op=m0 + m1)

        drawing = qp.draw(circ)()
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
            qp.Hadamard(0)
            m0 = qp.measure(0)
            qp.Hadamard(1)
            m1 = qp.measure(1)
            qp.Hadamard(2)
            m2 = qp.measure(2)
            return qp.expval(m0 * m2), qp.sample(m1)

        drawing = qp.draw(circ)()
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
            qp.Hadamard(0)
            qp.Hadamard(1)
            m0 = qp.measure(0)
            m1 = qp.measure(1)
            return qp.expval(m0 + m1), qp.sample(m0)

        drawing = qp.draw(circ)()
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
            qp.Hadamard(0)
            m0 = qp.measure(0)
            qp.cond(m0, qp.PauliZ)(0)
            return qp.expval(m0)

        drawing = qp.draw(circ)()
        expected_drawing = "0: ──H──┤↗├──Z─┤       \n         ╚═══╩═╡  <MCM>"

        assert drawing == expected_drawing

    def test_multi_cond_multi_meas_stats(self):
        """Test that combining multiple conditionals and multiple mid-circuit
        measurement statistics is drawn correctly."""

        def circ():
            qp.Hadamard(0)
            m0 = qp.measure(0)
            qp.Hadamard(1)
            m1 = qp.measure(1)
            qp.Hadamard(2)
            m2 = qp.measure(2)
            qp.Hadamard(3)
            qp.measure(3)
            qp.cond(m0, qp.PauliX)(0)
            qp.cond(m0 & m1, qp.PauliY)(1)
            return qp.expval(m2), qp.sample([m1, m2])

        drawing = qp.draw(circ)()
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


class TestPauliMeasure:
    """Tests PauliMeasure in a circuit drawing."""

    def test_pauli_measure_single_wire(self):
        """Tests drawing a pauli measurement on a single wire."""

        def circ():
            qp.H(0)
            qp.pauli_measure("X", wires=0)
            return qp.probs()

        expected = "0: ──H──┤↗X├─┤  Probs"
        assert draw(circ)() == expected

    def test_pauli_measure_multi_wires(self):
        """Tests drawing a pauli measurement on multiple wires."""

        def circ():
            qp.H(0)
            qp.pauli_measure("XY", wires=[0, 1])
            qp.CNOT([1, 2])
            return qp.expval(qp.Z(2))

        expected = "0: ──H─╭┤↗X├────┤     \n1: ────╰┤↗Y├─╭●─┤     \n2: ──────────╰X─┤  <Z>"
        assert draw(circ)() == expected

    @pytest.mark.parametrize("postselect", [0, 1])
    def test_pauli_measure_postselect(self, postselect):
        """Tests drawing a pauli measurement with postselect."""

        postselect_script = "₁" if postselect == 1 else "₀"

        def circ():
            qp.H(0)
            qp.pauli_measure("XY", wires=[0, 1], postselect=postselect)
            qp.CNOT([1, 2])
            return qp.expval(qp.Z(2))

        expected = (
            f"0: ──H─╭┤↗{postselect_script}X├────┤     \n"
            f"1: ────╰┤↗{postselect_script}Y├─╭●─┤     \n"
            "2: ───────────╰X─┤  <Z>"
        )
        assert draw(circ)() == expected

    def test_pauli_measure_multi_non_adjacent_wires(self):
        """Tests when the pauli measure skips wires."""

        def circ():
            qp.H(0)
            qp.H(1)
            qp.pauli_measure("XYZ", wires=[3, 0, 2])
            qp.X(1)
            return qp.probs()

        expected = (
            "0: ──H─╭┤↗Y├────┤  Probs\n"
            "1: ──H─│──────X─┤  Probs\n"
            "2: ────├┤↗Z├────┤  Probs\n"
            "3: ────╰┤↗X├────┤  Probs"
        )
        assert draw(circ)() == expected

    def test_conditional_on_pauli_measure(self):
        """Tests drawing using PPMs as classical control."""

        def circ():
            qp.H(0)
            qp.H(1)
            qp.H(2)
            qp.H(3)
            qp.H(4)
            m0 = qp.pauli_measure("XYZ", wires=[3, 0, 2])
            qp.cond(m0, qp.X)(1)
            return qp.probs()

        expected = (
            "0: ──H─╭┤↗Y├────┤  Probs\n"
            "1: ──H─│──────X─┤  Probs\n"
            "2: ──H─├┤↗Z├──║─┤  Probs\n"
            "3: ──H─╰┤↗X├──║─┤  Probs\n"
            "4: ──H───║────║─┤  Probs\n"
            "         ╚════╝         "
        )
        assert draw(circ)() == expected

    def test_terminal_measure_of_pauli_measure(self):
        """Tests drawing a terminal measurement of a pauli product measurement."""

        def circ():
            qp.H(0)
            qp.H(1)
            m0 = qp.pauli_measure("XY", wires=[1, 0])
            return qp.expval(m0)

        expected = "0: ──H─╭┤↗Y├─┤       \n1: ──H─╰┤↗X├─┤       \n         ╚═══╡  <PPM>"
        assert draw(circ)() == expected


class TestLevelExpansionStrategy:
    """Tests for the level expansion strategy in the draw function."""

    @pytest.fixture
    def transforms_circuit(self):
        """Fixture for a circuit with transforms applied."""

        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("default.qubit"), diff_method="parameter-shift")
        def circ(weights, order):
            qp.RandomLayers(weights, wires=(0, 1))
            qp.Permute(order, wires=(0, 1, 2))
            qp.PauliX(0)
            qp.PauliX(0)
            qp.RX(0.1, wires=0)
            qp.RX(-0.1, wires=0)
            return qp.expval(qp.PauliX(0))

        return circ

    @pytest.mark.parametrize(
        "var1,var2,expected",
        [
            (
                0,
                "top",
                "0: ─╭RandomLayers(M0)─╭Permute──X──X──RX(0.10)──RX(-0.10)─┤  <X>\n"
                "1: ─╰RandomLayers(M0)─├Permute────────────────────────────┤     \n"
                "2: ───────────────────╰Permute────────────────────────────┤     ",
            ),
            (
                2,
                "user",
                "0: ─╭RandomLayers(M0)─╭Permute─┤  <X>\n"
                "1: ─╰RandomLayers(M0)─├Permute─┤     \n"
                "2: ───────────────────╰Permute─┤     ",
            ),
            (
                3,
                "gradient",
                "0: ──RY(1.00)──╭Permute─┤  <X>\n"
                "1: ──RX(20.00)─├Permute─┤     \n"
                "2: ────────────╰Permute─┤     ",
            ),
            (
                8,
                "device",
                "0: ──RY(1.00)──╭SWAP─┤  <X>\n"
                "1: ──RX(20.00)─│─────┤     \n"
                "2: ────────────╰SWAP─┤     ",
            ),
        ],
    )
    def test_equivalent_levels(self, transforms_circuit, var1, var2, expected):
        """Test that drawing the circuit at different levels produces equivalent results."""
        order = [2, 1, 0]
        weights = pnp.array([[1.0, 20]])

        out1 = qp.draw(transforms_circuit, level=var1, show_matrices=False)(weights, order)
        out2 = qp.draw(transforms_circuit, level=var2, show_matrices=False)(weights, order)

        assert out1 == out2 == expected

    def test_draw_at_level_1(self, transforms_circuit):
        """Test that at level one the first transform has been applied, cancelling inverses."""

        order = [2, 1, 0]
        weights = pnp.array([[1.0, 20]])

        out = qp.draw(transforms_circuit, level=1, show_matrices=False)(weights, order)

        expected = (
            "0: ─╭RandomLayers(M0)─╭Permute──RX(0.10)──RX(-0.10)─┤  <X>\n"
            "1: ─╰RandomLayers(M0)─├Permute──────────────────────┤     \n"
            "2: ───────────────────╰Permute──────────────────────┤     "
        )
        assert out == expected

    def test_draw_with_qfunc_warns_with_level(self):
        """Test that draw warns the user about level being ignored."""

        def qfunc():
            qp.PauliZ(0)

        with pytest.warns(UserWarning, match="the level argument is ignored"):
            qp.draw(qfunc, level=None)

    def test_custom_level(self):
        """Test that we can draw at a custom level."""

        @qp.transforms.merge_rotations
        @qp.marker(level="my_level")
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("null.qubit"))
        def c():
            qp.RX(0.2, 0)
            qp.X(0)
            qp.X(0)
            qp.RX(0.2, 0)
            return qp.state()

        expected = "0: ──RX(0.20)──RX(0.20)─┤  State"
        assert qp.draw(c, level="my_level")() == expected


def test_draw_batch_transform():
    """Test that drawing a batch transform works correctly."""

    @qp.gradients.param_shift(shifts=[(0.2,)])
    @qp.qnode(qp.device("default.qubit", wires=1))
    def circ(x):
        qp.Hadamard(wires=0)
        qp.RX(x, wires=0)
        return qp.expval(qp.PauliZ(0))

    expected = "0: ──H──RX(0.8)─┤  <Z>\n\n0: ──H──RX(0.4)─┤  <Z>"
    assert draw(circ, decimals=1)(pnp.array(0.6, requires_grad=True)) == expected


def test_applied_transforms():
    """Test that any transforms applied to the qnode are included in the output."""

    @qp.transform
    def just_pauli_x(_):
        new_tape = qp.tape.QuantumScript([qp.PauliX(0)])
        return (new_tape,), lambda res: res[0]

    @just_pauli_x
    @qp.qnode(qp.device("default.qubit", wires=2))
    def my_circuit(x):
        qp.RX(x, wires=0)
        qp.SWAP(wires=(0, 1))
        return qp.probs(wires=(0, 1))

    expected = "0: ──X─┤  "
    assert qp.draw(my_circuit)(1.234) == expected


def test_draw_with_qfunc():
    """Test a non-qnode qfunc can be drawn."""

    def qfunc(x):
        qp.RX(x, wires=[0])
        qp.PauliZ(1)

    assert qp.draw(qfunc)(1.1) == "0: ──RX(1.10)─┤  \n1: ──Z────────┤  "


def test_draw_with_qfunc_with_measurements():
    """Test a non-qnode qfunc with measurements can be drawn."""

    def qfunc(x):
        qp.RX(x, wires=[0])
        qp.CNOT([0, 1])
        return qp.expval(qp.PauliZ(1))

    assert qp.draw(qfunc)(1.1) == "0: ──RX(1.10)─╭●─┤     \n1: ───────────╰X─┤  <Z>"


@pytest.mark.parametrize("use_qnode", [True, False])
def test_sort_wires(use_qnode):
    """Test that drawing a qnode with no wire order or device wires sorts the wires automatically."""

    def func():
        qp.X(4)
        qp.X(2)
        qp.X(0)
        return qp.expval(qp.Z(0))

    if use_qnode:
        func = qp.QNode(func, qp.device("default.qubit"))

    expected = "0: ──X─┤  <Z>\n2: ──X─┤     \n4: ──X─┤     "
    assert qp.draw(func)() == expected


@pytest.mark.parametrize("use_qnode", [True, False])
def test_sort_wires_fallback(use_qnode):
    """Test that drawing a qnode with no wire order or device wires falls back to tape wires if
    sorting fails."""

    def func():
        qp.X(4)
        qp.X("a")
        qp.X(0)
        return qp.expval(qp.Z(0))

    if use_qnode:
        func = qp.QNode(func, qp.device("default.qubit"))

    expected = "4: ──X─┤     \na: ──X─┤     \n0: ──X─┤  <Z>"
    assert qp.draw(func)() == expected
