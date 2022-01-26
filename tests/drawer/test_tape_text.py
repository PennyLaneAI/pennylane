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
Unit tests for the `pennylane.draw_text` function.
"""

import pytest
import pennylane as qml

from pennylane.drawer import tape_text
from pennylane.drawer.tape_text import _add_grouping_symbols, _add_op, _add_measurement
from pennylane.tape import QuantumTape

default_wire_map = {0: 0, 1: 1, 2: 2, 3: 3}

with QuantumTape() as tape:
    qml.RX(1.23456, wires=0)
    qml.RY(2.3456, wires="a")
    qml.RZ(3.4567, wires=1.234)


class TestHelperFunctions:
    @pytest.mark.parametrize(
        "op, out",
        [
            (qml.PauliX(0), ["", "", "", ""]),
            (qml.CNOT(wires=(0, 2)), ["╭", "│", "╰", ""]),
            (qml.CSWAP(wires=(0, 2, 3)), ["╭", "│", "├", "╰"]),
        ],
    )
    def test_add_grouping_symbols(self, op, out):
        assert out == _add_grouping_symbols(op, ["", "", "", ""], default_wire_map)

    @pytest.mark.parametrize(
        "op, out",
        [
            (qml.expval(qml.PauliX(0)), ["<X>", "", "", ""]),
            (qml.probs(wires=(0, 2)), ["╭Probs", "│", "╰Probs", ""]),
            (qml.var(qml.PauliX(1)), ["", "Var[X]", "", ""]),
            (qml.state(), ["State", "State", "State", "State"]),
            (qml.sample(), ["Sample", "Sample", "Sample", "Sample"]),
        ],
    )
    def test_add_measurements(self, op, out):
        """Test private _add_measurement function renders as expected."""
        assert out == _add_measurement(op, [""] * 4, default_wire_map, None)

    @pytest.mark.parametrize(
        "op, out",
        [
            (qml.PauliX(0), ["─X", "─", "─", "─"]),
            (qml.CNOT(wires=(0, 2)), ["╭C", "│", "╰X", "─"]),
            (qml.Toffoli(wires=(0, 1, 3)), ["╭C", "├C", "│", "╰X"]),
            (qml.IsingXX(1.23, wires=(0, 2)), ["╭IsingXX", "│", "╰IsingXX", "─"]),
        ],
    )
    def test_add_op(self, op, out):
        """Test adding the first operation to array of strings"""
        assert out == _add_op(op, ["─"] * 4, default_wire_map, None)

    @pytest.mark.parametrize(
        "op, out",
        [
            (qml.PauliY(1), ["─X", "─Y", "─", "─"]),
            (qml.CNOT(wires=(1, 2)), ["─X", "╭C", "╰X", "─"]),
            (qml.CRX(1.23, wires=(2, 3)), ["─X", "─", "╭C", "╰RX"]),
        ],
    )
    def test_add_second_op(self, op, out):
        """Test adding a second operation to the array of strings"""
        start = _add_op(qml.PauliX(0), ["─"] * 4, default_wire_map, None)
        assert out == _add_op(op, start, default_wire_map, None)


class TestEmptyTapes:
    def test_empty_tape(self):
        """Test using an empty tape returns a blank string"""
        assert tape_text(QuantumTape()) == ""

    def test_empty_tape_wire_order(self):
        """Test wire order and show_all_wires shows wires with empty tape."""
        expected = "a: ───┤  \nb: ───┤  "
        out = tape_text(QuantumTape(), wire_order=["a", "b"], show_all_wires=True)
        assert expected == out


class TestLabeling:
    def test_any_wire_labels(self):
        """Test wire labels with different kinds of objects."""

        split_str = tape_text(tape).split("\n")
        assert split_str[0][0:6] == "    0:"
        assert split_str[1][0:6] == "    a:"
        assert split_str[2][0:6] == "1.234:"

    def test_wire_order(self):
        """Test wire_order keyword changes order of the wires"""

        split_str = tape_text(tape, wire_order=[1.234, "a", 0, "b"]).split("\n")
        assert split_str[2][0:6] == "    0:"
        assert split_str[1][0:6] == "    a:"
        assert split_str[0][0:6] == "1.234:"

    def test_show_all_wires(self):
        """Test wire_order constains unused wires, show_all_wires
        forces them to display."""

        split_str = tape_text(tape, wire_order=["b"], show_all_wires=True).split("\n")

        assert split_str[0][0:6] == "    b:"
        assert split_str[1][0:6] == "    0:"
        assert split_str[2][0:6] == "    a:"
        assert split_str[3][0:6] == "1.234:"


class TestDecimals:
    """Test the decimals keyword argument."""

    def test_decimals(self):
        """Test that the decimals keyword makes the operation parameters included."""

        expected = "    0: ──RX(1.23)─┤  \n    a: ──RY(2.35)─┤  \n1.234: ──RZ(3.46)─┤  "

        assert tape_text(tape, decimals=2) == expected

    def test_decimals_multiparameters(self):
        """Tests decimals also displays parameters when the operation has multiple parameters."""

        with QuantumTape() as tape_rot:
            qml.Rot(1.2345, 2.3456, 3.4566, wires=0)

        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_rot, decimals=2) == expected

    def test_decimals_0(self):
        """Test decimals=0 rounds to integers"""

        expected = "    0: ──RX(1)─┤  \n" "    a: ──RY(2)─┤  \n" "1.234: ──RZ(3)─┤  "

        assert tape_text(tape, decimals=0) == expected

    def test_torch_parameters(self):
        """Test torch parameters in tape display as normal numbers."""
        torch = pytest.importorskip("torch")
        with QuantumTape() as tape_torch:
            qml.Rot(torch.tensor(1.234), torch.tensor(2.345), torch.tensor(3.456), wires=0)

        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_torch, decimals=2) == expected

    def test_tensorflow_parameters(self):
        """Test tensorflow parameters display as normal numbers."""
        tf = pytest.importorskip("tensorflow")
        with QuantumTape() as tape_tf:
            qml.Rot(tf.Variable(1.234), tf.Variable(2.345), tf.Variable(3.456), wires=0)

        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_tf, decimals=2) == expected

    def test_jax_parameters(self):
        """Test jax parameters in tape display as normal numbers."""
        jnp = pytest.importorskip("jax.numpy")

        with QuantumTape() as tape_jax:
            qml.Rot(jnp.array(1.234), jnp.array(2.345), jnp.array(3.456), wires=0)

        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_jax, decimals=2) == expected


class TestMaxLength:
    """Test the max_length keyword."""

    def test_max_length_default(self):
        """Test max length defaults to 100."""
        with QuantumTape() as tape_ml:
            for _ in range(50):
                qml.PauliX(0)
                qml.PauliY(1)

            for _ in range(3):
                qml.sample()

        out = tape_text(tape_ml)
        assert 95 <= max(len(s) for s in out.split("\n")) <= 100

    @pytest.mark.parametrize("ml", [10, 15, 20])
    def test_setting_max_length(self, ml):
        """Test several custom max_length parameters change the wrapping length."""

        with QuantumTape() as tape_ml:
            for _ in range(10):
                qml.PauliX(0)
                qml.PauliY(1)

            for _ in range(3):
                qml.sample()

        out = tape_text(tape, max_length=ml)
        assert max(len(s) for s in out.split("\n")) <= ml


single_op_tests_data = [
    (qml.CNOT(wires=(0, 1)), "0: ─╭C─┤  \n1: ─╰X─┤  "),
    (qml.Toffoli(wires=(0, 1, 2)), "0: ─╭C─┤  \n1: ─├C─┤  \n2: ─╰X─┤  "),
    (qml.Barrier(wires=(0, 1, 2)), "0: ─╭||─┤  \n1: ─├||─┤  \n2: ─╰||─┤  "),
    (qml.CSWAP(wires=(0, 1, 2)), "0: ─╭C────┤  \n1: ─├SWAP─┤  \n2: ─╰SWAP─┤  "),
    (
        qml.DoubleExcitationPlus(1.23, wires=(0, 1, 2, 3)),
        "0: ─╭G²₊(1.23)─┤  \n1: ─├G²₊(1.23)─┤  \n2: ─├G²₊(1.23)─┤  \n3: ─╰G²₊(1.23)─┤  ",
    ),
    (qml.QubitUnitary(qml.numpy.eye(4), wires=(0, 1)), "0: ─╭U─┤  \n1: ─╰U─┤  "),
    (qml.QubitSum(wires=(0, 1, 2)), "0: ─╭Σ─┤  \n1: ─├Σ─┤  \n2: ─╰Σ─┤  "),
    (qml.AmplitudeDamping(0.98, wires=0), "0: ──AmplitudeDamping(0.98)─┤  "),
    (
        qml.QubitStateVector([0, 1, 0, 0], wires=(0, 1)),
        "0: ─╭QubitStateVector─┤  \n1: ─╰QubitStateVector─┤  ",
    ),
    (qml.Kerr(1.234, wires=0), "0: ──Kerr(1.23)─┤  "),
    (
        qml.GroverOperator(wires=(0, 1, 2)),
        "0: ─╭GroverOperator─┤  \n1: ─├GroverOperator─┤  \n2: ─╰GroverOperator─┤  ",
    ),
    (qml.RX(1.234, wires=0).inv(), "0: ──RX⁻¹(1.23)─┤  "),
    (qml.expval(qml.PauliZ(0)), "0: ───┤  <Z>"),
    (qml.var(qml.PauliZ(0)), "0: ───┤  Var[Z]"),
    (qml.probs(wires=0), "0: ───┤  Probs"),
    (qml.probs(op=qml.PauliZ(0)), "0: ───┤  Probs[Z]"),
    (qml.sample(wires=0), "0: ───┤  Sample"),
    (qml.sample(op=qml.PauliX(0)), "0: ───┤  Sample[X]"),
    (qml.expval(0.1 * qml.PauliX(0) @ qml.PauliY(1)), "0: ───┤ ╭<𝓗(0.10)>\n1: ───┤ ╰<𝓗(0.10)>"),
    (
        qml.expval(
            0.1 * qml.PauliX(0) + 0.2 * qml.PauliY(1) + 0.3 * qml.PauliZ(0) + 0.4 * qml.PauliZ(1)
        ),
        "0: ───┤ ╭<𝓗>\n1: ───┤ ╰<𝓗>",
    ),
]


@pytest.mark.parametrize("op, expected", single_op_tests_data)
def test_single_ops(op, expected):
    """Tests a variety of different single operation tapes render as expected."""

    with QuantumTape() as tape:
        qml.apply(op)

    assert tape_text(tape, decimals=2) == expected


class TestLayering:
    """Test operations are placed in the correct locations."""

    def test_adjacent_ops(self):
        """Test non-blocking gates end up on same layer."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(1)
            qml.PauliX(2)

        assert tape_text(tape) == "0: ──X─┤  \n1: ──X─┤  \n2: ──X─┤  "

    def test_blocking_ops(self):
        """Test single qubit gates on same wire line up."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        assert tape_text(tape) == "0: ──X──X──X─┤  "

    def test_blocking_multiwire_gate(self):
        """Tests gate gets blocked by multi-wire gate."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.IsingXX(1.2345, wires=(0, 2))
            qml.PauliX(1)

        expected = "0: ──X─╭IsingXX────┤  \n1: ────│─────────X─┤  \n2: ────╰IsingXX────┤  "

        assert tape_text(tape, wire_order=[0, 1, 2]) == expected


class TestNestedTapes:
    """Test situations with nested tapes."""

    def test_cache_keyword_tape_offset(self):
        """Test that tape numbering is determined by the `tape_offset` keyword of the cache."""

        with QuantumTape() as tape:
            with QuantumTape() as tape_inner:
                qml.PauliX(0)

        expected = "0: ──Tape:3─┤  \n" "\nTape:3\n" "0: ──X─┤  "

        assert tape_text(tape, cache={"tape_offset": 3}) == expected

    def test_multiple_nested_tapes(self):
        """Test numbers consistent with multiple nested tapes and
        multiple levels of nesting."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            with QuantumTape() as tape0:
                qml.PauliY(0)
                qml.PauliZ(0)
                with QuantumTape() as tape2:
                    qml.PauliX(0)
            with QuantumTape() as tape1:
                qml.PauliY(0)
                with QuantumTape() as tape3:
                    qml.PauliZ(0)

        expected = (
            "0: ──X──Tape:0──Tape:1─┤  \n"
            "\nTape:0\n"
            "0: ──Y──Z──Tape:2─┤  \n"
            "\nTape:2\n"
            "0: ──X─┤  \n"
            "\nTape:1\n"
            "0: ──Y──Tape:3─┤  \n"
            "\nTape:3\n"
            "0: ──Z─┤  "
        )

        assert tape_text(tape) == expected

    def test_nested_tapes_decimals(self):
        """Test decimals keyword passed to nested tapes."""

        with QuantumTape() as tape:
            qml.RX(1.2345, wires=0)
            with QuantumTape() as tape0:
                qml.Rot(1.2345, 2.3456, 3.456, wires=0)

        expected = "0: ──RX(1.2)──Tape:0─┤  \n" "\nTape:0\n" "0: ──Rot(1.2,2.3,3.5)─┤  "

        assert tape_text(tape, decimals=1) == expected

    def test_nested_tapes_wire_order(self):
        """Test wire order preserved in nested tapes."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            qml.PauliY(1)
            with QuantumTape() as tape0:
                qml.PauliX(0)
                qml.PauliY(1)

        expected = (
            "1: ──Y─╭Tape:0─┤  \n" "0: ──X─╰Tape:0─┤  \n" "\nTape:0\n" "1: ──Y─┤  \n0: ──X─┤  "
        )

        assert tape_text(tape, wire_order=[1, 0]) == expected

    def test_nested_tapes_max_length(self):
        """Test max length passes to recursive tapes."""

        with QuantumTape() as tape:
            qml.PauliX(0)
            with QuantumTape() as tape0:
                for _ in range(10):
                    qml.PauliX(0)

        expected = (
            "0: ──X──Tape:0─┤  \n" "\nTape:0\n" "0: ──X──X──X──X──X\n" "\n───X──X──X──X──X─┤  "
        )

        out = tape_text(tape, max_length=20)
        assert out == expected
        assert max(len(s) for s in out.split("\n")) <= 20
