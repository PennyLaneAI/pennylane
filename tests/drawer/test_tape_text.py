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
# pylint: disable=import-outside-toplevel

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.drawer import tape_text
from pennylane.drawer.tape_text import _add_grouping_symbols, _add_measurement, _add_op
from pennylane.tape import QuantumScript, QuantumTape

default_wire_map = {0: 0, 1: 1, 2: 2, 3: 3}

with qml.queuing.AnnotatedQueue() as q_tape:
    qml.RX(1.23456, wires=0)
    qml.RY(2.3456, wires="a")
    qml.RZ(3.4567, wires=1.234)


tape = qml.tape.QuantumScript.from_queue(q_tape)


class TestHelperFunctions:
    """Test helper functions for the tape text."""

    @pytest.mark.parametrize(
        "op, out",
        [
            (qml.PauliX(0), ["", "", "", ""]),
            (qml.CNOT(wires=(0, 2)), ["╭", "│", "╰", ""]),
            (qml.CSWAP(wires=(0, 2, 3)), ["╭", "│", "├", "╰"]),
        ],
    )
    def test_add_grouping_symbols(self, op, out):
        """Test private _add_grouping_symbols function renders as expected."""
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
        assert out == _add_measurement(op, [""] * 4, default_wire_map, None, None)

    def test_add_measurements_cache(self):
        """Test private _add_measurement function with a matrix cache."""
        cache = {"matrices": []}
        op = qml.expval(qml.Hermitian(np.eye(2), wires=0))
        assert _add_measurement(op, ["", ""], {0: 0, 1: 1}, None, cache) == ["<𝓗(M0)>", ""]

        assert qml.math.allclose(cache["matrices"][0], np.eye(2))

        op2 = qml.expval(qml.Hermitian(np.eye(2), wires=1))
        # new op with same matrix, should have same M0 designation
        assert _add_measurement(op2, ["", ""], {0: 0, 1: 1}, None, cache) == ["", "<𝓗(M0)>"]

    @pytest.mark.parametrize(
        "op, out",
        [
            (qml.PauliX(0), ["─X", "─", "─", "─"]),
            (qml.CNOT(wires=(0, 2)), ["╭●", "│", "╰X", "─"]),
            (qml.Toffoli(wires=(0, 1, 3)), ["╭●", "├●", "│", "╰X"]),
            (qml.IsingXX(1.23, wires=(0, 2)), ["╭IsingXX", "│", "╰IsingXX", "─"]),
            (qml.Snapshot(), ["─|Snap|", "─|Snap|", "─|Snap|", "─|Snap|"]),
            (qml.Barrier(), ["─||", "─||", "─||", "─||"]),
            (qml.S(0) @ qml.T(0), ["─S@T", "─", "─", "─"]),
        ],
    )
    def test_add_op(self, op, out):
        """Test adding the first operation to array of strings"""
        assert out == _add_op(op, ["─"] * 4, default_wire_map, None, None)

    @pytest.mark.parametrize(
        "op, out",
        [
            (qml.PauliY(1), ["─X", "─Y", "─", "─"]),
            (qml.CNOT(wires=(1, 2)), ["─X", "╭●", "╰X", "─"]),
            (qml.CRX(1.23, wires=(2, 3)), ["─X", "─", "╭●", "╰RX"]),
        ],
    )
    def test_add_second_op(self, op, out):
        """Test adding a second operation to the array of strings"""
        start = _add_op(qml.PauliX(0), ["─"] * 4, default_wire_map, None, None)
        assert out == _add_op(op, start, default_wire_map, None, None)

    def test_add_op_cache(self):
        """Test private _add_op method functions with a matrix cache."""
        cache = {"matrices": []}
        op1 = qml.QubitUnitary(np.eye(2), wires=0)
        assert _add_op(op1, ["", ""], {0: 0, 1: 1}, None, cache) == ["U(M0)", ""]

        assert qml.math.allclose(cache["matrices"][0], np.eye(2))
        op2 = qml.QubitUnitary(np.eye(2), wires=1)
        assert _add_op(op2, ["", ""], {0: 0, 1: 1}, None, cache) == ["", "U(M0)"]


class TestEmptyTapes:
    """Test that the text for empty tapes is correct."""

    def test_empty_tape(self):
        """Test using an empty tape returns a blank string"""
        assert tape_text(QuantumScript()) == ""

    def test_empty_tape_wire_order(self):
        """Test wire order and show_all_wires shows wires with empty tape."""
        expected = "a: ───┤  \nb: ───┤  "
        out = tape_text(QuantumScript(), wire_order=["a", "b"], show_all_wires=True)
        assert expected == out


class TestLabeling:
    """Test that wire labels work correctly."""

    def test_any_wire_labels(self):
        """Test wire labels with different kinds of objects."""

        split_str = tape_text(tape).split("\n")
        assert split_str[0][:6] == "    0:"
        assert split_str[1][:6] == "    a:"
        assert split_str[2][:6] == "1.234:"

    def test_wire_order(self):
        """Test wire_order keyword changes order of the wires"""

        split_str = tape_text(tape, wire_order=[1.234, "a", 0, "b"]).split("\n")
        assert split_str[2][:6] == "    0:"
        assert split_str[1][:6] == "    a:"
        assert split_str[0][:6] == "1.234:"

    def test_show_all_wires(self):
        """Test wire_order constains unused wires, show_all_wires
        forces them to display."""

        split_str = tape_text(tape, wire_order=["b"], show_all_wires=True).split("\n")

        assert split_str[0][:6] == "    b:"
        assert split_str[1][:6] == "    0:"
        assert split_str[2][:6] == "    a:"
        assert split_str[3][:6] == "1.234:"


class TestDecimals:
    """Test the decimals keyword argument."""

    def test_decimals(self):
        """Test that the decimals keyword makes the operation parameters included."""

        expected = "    0: ──RX(1.23)─┤  \n    a: ──RY(2.35)─┤  \n1.234: ──RZ(3.46)─┤  "

        assert tape_text(tape, decimals=2) == expected

    def test_decimals_multiparameters(self):
        """Tests decimals also displays parameters when the operation has multiple parameters."""

        with qml.queuing.AnnotatedQueue() as q_tape_rot:
            qml.Rot(1.2345, 2.3456, 3.4566, wires=0)

        tape_rot = qml.tape.QuantumScript.from_queue(q_tape_rot)
        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_rot, decimals=2) == expected

    def test_decimals_0(self):
        """Test decimals=0 rounds to integers"""

        expected = "    0: ──RX(1)─┤  \n    a: ──RY(2)─┤  \n1.234: ──RZ(3)─┤  "

        assert tape_text(tape, decimals=0) == expected

    @pytest.mark.torch
    def test_torch_parameters(self):
        """Test torch parameters in tape display as normal numbers."""
        import torch

        with qml.queuing.AnnotatedQueue() as q_tape_torch:
            qml.Rot(torch.tensor(1.234), torch.tensor(2.345), torch.tensor(3.456), wires=0)

        tape_torch = qml.tape.QuantumScript.from_queue(q_tape_torch)
        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_torch, decimals=2) == expected

    @pytest.mark.tf
    def test_tensorflow_parameters(self):
        """Test tensorflow parameters display as normal numbers."""
        import tensorflow as tf

        with qml.queuing.AnnotatedQueue() as q_tape_tf:
            qml.Rot(tf.Variable(1.234), tf.Variable(2.345), tf.Variable(3.456), wires=0)

        tape_tf = qml.tape.QuantumScript.from_queue(q_tape_tf)
        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_tf, decimals=2) == expected

    @pytest.mark.jax
    def test_jax_parameters(self):
        """Test jax parameters in tape display as normal numbers."""
        import jax.numpy as jnp

        with qml.queuing.AnnotatedQueue() as q_tape_jax:
            qml.Rot(jnp.array(1.234), jnp.array(2.345), jnp.array(3.456), wires=0)

        tape_jax = qml.tape.QuantumScript.from_queue(q_tape_jax)
        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_jax, decimals=2) == expected


class TestMaxLength:
    """Test the max_length keyword."""

    def test_max_length_default(self):
        """Test max length defaults to 100."""
        with qml.queuing.AnnotatedQueue() as q_tape_ml:
            for _ in range(50):
                qml.PauliX(0)
                qml.PauliY(1)

            for _ in range(3):
                qml.sample()

        tape_ml = qml.tape.QuantumScript.from_queue(q_tape_ml)
        out = tape_text(tape_ml)
        assert 95 <= max(len(s) for s in out.split("\n")) <= 100

    @pytest.mark.parametrize("ml", [10, 15, 20])
    def test_setting_max_length(self, ml):
        """Test several custom max_length parameters change the wrapping length."""

        with qml.queuing.AnnotatedQueue() as q_tape_ml:
            for _ in range(10):
                qml.PauliX(0)
                qml.PauliY(1)

            for _ in range(3):
                qml.sample()

        tape_ml = qml.tape.QuantumScript.from_queue(q_tape_ml)
        out = tape_text(tape_ml, max_length=ml)
        assert max(len(s) for s in out.split("\n")) <= ml


single_op_tests_data = [
    (
        qml.MultiControlledX(wires=[0, 1, 2, 3], control_values="010"),
        "0: ─╭○─┤  \n1: ─├●─┤  \n2: ─├○─┤  \n3: ─╰X─┤  ",
    ),
    (
        # pylint:disable=no-member
        qml.ops.op_math.Controlled(qml.PauliY(3), (0, 1, 2), [0, 1, 0]),
        "0: ─╭○─┤  \n1: ─├●─┤  \n2: ─├○─┤  \n3: ─╰Y─┤  ",
    ),
    (qml.CNOT(wires=(0, 1)), "0: ─╭●─┤  \n1: ─╰X─┤  "),
    (qml.Toffoli(wires=(0, 1, 2)), "0: ─╭●─┤  \n1: ─├●─┤  \n2: ─╰X─┤  "),
    (qml.Barrier(wires=(0, 1, 2)), "0: ─╭||─┤  \n1: ─├||─┤  \n2: ─╰||─┤  "),
    (qml.CSWAP(wires=(0, 1, 2)), "0: ─╭●────┤  \n1: ─├SWAP─┤  \n2: ─╰SWAP─┤  "),
    (
        qml.DoubleExcitationPlus(1.23, wires=(0, 1, 2, 3)),
        "0: ─╭G²₊(1.23)─┤  \n1: ─├G²₊(1.23)─┤  \n2: ─├G²₊(1.23)─┤  \n3: ─╰G²₊(1.23)─┤  ",
    ),
    (qml.QubitUnitary(qml.numpy.eye(4), wires=(0, 1)), "0: ─╭U(M0)─┤  \n1: ─╰U(M0)─┤  "),
    (qml.QubitSum(wires=(0, 1, 2)), "0: ─╭Σ─┤  \n1: ─├Σ─┤  \n2: ─╰Σ─┤  "),
    (qml.AmplitudeDamping(0.98, wires=0), "0: ──AmplitudeDamping(0.98)─┤  "),
    (
        qml.QubitStateVector([0, 1, 0, 0], wires=(0, 1)),
        "0: ─╭|Ψ⟩─┤  \n1: ─╰|Ψ⟩─┤  ",
    ),
    (qml.Kerr(1.234, wires=0), "0: ──Kerr(1.23)─┤  "),
    (
        qml.GroverOperator(wires=(0, 1, 2)),
        "0: ─╭GroverOperator─┤  \n1: ─├GroverOperator─┤  \n2: ─╰GroverOperator─┤  ",
    ),
    (
        qml.adjoint(qml.RX(1.234, wires=0)),
        "0: ──RX(1.23)†─┤  ",
    ),
    (
        qml.RX(1.234, wires=0) ** -1,
        "0: ──RX(1.23)⁻¹─┤  ",
    ),
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
    # Operations (both regular and controlled) and nested multi-valued controls
    (qml.ctrl(qml.PauliX(wires=2), control=[0, 1]), "0: ─╭●─┤  \n1: ─├●─┤  \n2: ─╰X─┤  "),
    (qml.ctrl(qml.CNOT(wires=[1, 2]), control=0), "0: ─╭●─┤  \n1: ─├●─┤  \n2: ─╰X─┤  "),
    (
        qml.ctrl(qml.CRZ(0.2, wires=[1, 2]), control=[3, 0]),
        "3: ─╭●────────┤  \n0: ─├●────────┤  \n1: ─├●────────┤  \n2: ─╰RZ(0.20)─┤  ",
    ),
    (
        qml.ctrl(qml.CH(wires=[0, 3]), control=[2, 1], control_values=[False, True]),
        "2: ─╭○─┤  \n1: ─├●─┤  \n0: ─├●─┤  \n3: ─╰H─┤  ",
    ),
    (
        qml.ctrl(
            qml.ctrl(qml.CY(wires=[3, 4]), control=[1, 2], control_values=[True, False]),
            control=0,
            control_values=[False],
        ),
        "0: ─╭○─┤  \n1: ─├●─┤  \n2: ─├○─┤  \n3: ─├●─┤  \n4: ─╰Y─┤  ",
    ),
]


@pytest.mark.parametrize("op, expected", single_op_tests_data)
def test_single_ops(op, expected):
    """Tests a variety of different single operation tapes render as expected."""

    with qml.queuing.AnnotatedQueue() as q:
        qml.apply(op)

    _tape = qml.tape.QuantumScript.from_queue(q)
    assert tape_text(_tape, decimals=2, show_matrices=False) == expected


class TestLayering:
    """Test operations are placed in the correct locations."""

    def test_adjacent_ops(self):
        """Test non-blocking gates end up on same layer."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(0)
            qml.PauliX(1)
            qml.PauliX(2)

        _tape = qml.tape.QuantumScript.from_queue(q)
        assert tape_text(_tape) == "0: ──X─┤  \n1: ──X─┤  \n2: ──X─┤  "

    def test_blocking_ops(self):
        """Test single qubit gates on same wire line up."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(0)
            qml.PauliX(0)
            qml.PauliX(0)

        _tape = qml.tape.QuantumScript.from_queue(q)
        assert tape_text(_tape) == "0: ──X──X──X─┤  "

    def test_blocking_multiwire_gate(self):
        """Tests gate gets blocked by multi-wire gate."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(0)
            qml.IsingXX(1.2345, wires=(0, 2))
            qml.PauliX(1)

        _tape = qml.tape.QuantumScript.from_queue(q)
        expected = "0: ──X─╭IsingXX────┤  \n1: ────│─────────X─┤  \n2: ────╰IsingXX────┤  "

        assert tape_text(_tape, wire_order=[0, 1, 2]) == expected


tape_matrices = qml.tape.QuantumScript(
    prep=[qml.QubitStateVector([1.0, 0.0, 0.0, 0.0], wires=(0, 1))],
    ops=[qml.QubitUnitary(np.eye(2), wires=0)],
    measurements=[qml.expval(qml.Hermitian(np.eye(2), wires=0))],
)


class TestShowMatrices:
    """Test the handling of matrix-valued parameters."""

    def test_default_shows_matrix_parameters(self):
        """Test matrices numbered but not included by default."""

        expected = (
            "0: ─╭|Ψ⟩──U(M0)─┤  <𝓗(M0)>\n"
            "1: ─╰|Ψ⟩────────┤         \n"
            "M0 = \n[[1. 0.]\n [0. 1.]]"
        )

        assert tape_text(tape_matrices) == expected

    def test_do_not_show_matrices(self):
        """Test matrices included when requested."""

        expected = "0: ─╭|Ψ⟩──U(M0)─┤  <𝓗(M0)>\n1: ─╰|Ψ⟩────────┤         "

        assert tape_text(tape_matrices, show_matrices=False) == expected

    def test_matrix_parameters_provided_cache(self):
        """Providing an existing matrix cache determines numbering order of matrices.
        All matrices printed out regardless of use."""

        cache = {"matrices": [np.eye(2), -np.eye(3)]}

        expected = (
            "0: ─╭|Ψ⟩──U(M0)─┤  <𝓗(M0)>\n"
            "1: ─╰|Ψ⟩────────┤         \n"
            "M0 = \n[[1. 0.]\n [0. 1.]]\n"
            "M1 = \n[[-1. -0. -0.]\n [-0. -1. -0.]\n [-0. -0. -1.]]"
        )

        assert tape_text(tape_matrices, show_matrices=True, cache=cache) == expected


# @pytest.mark.skip("Nested tapes are being deprecated")
class TestNestedTapes:
    """Test situations with nested tapes."""

    def test_cache_keyword_tape_offset(self):
        """Test that tape numbering is determined by the `tape_offset` keyword of the cache."""

        with QuantumTape() as _tape:
            with QuantumTape():
                qml.PauliX(0)

        expected = "0: ──Tape:3─┤  \n\nTape:3\n0: ──X─┤  "

        assert tape_text(_tape, cache={"tape_offset": 3}) == expected

    def test_multiple_nested_tapes(self):
        """Test numbers consistent with multiple nested tapes and
        multiple levels of nesting."""

        with QuantumTape() as _tape:
            qml.PauliX(0)
            with QuantumTape():
                qml.PauliY(0)
                qml.PauliZ(0)
                with QuantumTape():
                    qml.PauliX(0)
            with QuantumTape():
                qml.PauliY(0)
                with QuantumTape():
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

        assert tape_text(_tape) == expected

    def test_nested_tapes_decimals(self):
        """Test decimals keyword passed to nested tapes."""

        with QuantumTape() as _tape:
            qml.RX(1.2345, wires=0)
            with QuantumTape():
                qml.Rot(1.2345, 2.3456, 3.456, wires=0)

        expected = "0: ──RX(1.2)──Tape:0─┤  \n\nTape:0\n0: ──Rot(1.2,2.3,3.5)─┤  "

        assert tape_text(_tape, decimals=1) == expected

    def test_nested_tapes_wire_order(self):
        """Test wire order preserved in nested tapes."""

        with QuantumTape() as _tape:
            qml.PauliX(0)
            qml.PauliY(1)
            with QuantumTape():
                qml.PauliX(0)
                qml.PauliY(1)

        expected = "1: ──Y─╭Tape:0─┤  \n0: ──X─╰Tape:0─┤  \n\nTape:0\n1: ──Y─┤  \n0: ──X─┤  "

        assert tape_text(_tape, wire_order=[1, 0]) == expected

    def test_nested_tapes_max_length(self):
        """Test max length passes to recursive tapes."""

        with QuantumTape() as _tape:
            qml.PauliX(0)
            with QuantumTape():
                for _ in range(10):
                    qml.PauliX(0)

        expected = "0: ──X──Tape:0─┤  \n\nTape:0\n0: ──X──X──X──X──X\n\n───X──X──X──X──X─┤  "

        out = tape_text(_tape, max_length=20)
        assert out == expected
        assert max(len(s) for s in out.split("\n")) <= 20
