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

from copy import copy

import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.drawer import tape_text
from pennylane.drawer._add_obj import (
    _add_cond_grouping_symbols,
    _add_cwire_measurement,
    _add_cwire_measurement_grouping_symbols,
    _add_grouping_symbols,
    _add_measurement,
    _add_mid_measure_grouping_symbols,
    _add_obj,
)
from pennylane.drawer.tape_text import _Config
from pennylane.tape import QuantumScript

default_wire_map = {0: 0, 1: 1, 2: 2, 3: 3}
default_bit_map = {}

default_mid_measure_1 = qp.ops.MidMeasure(0, id="1")
default_mid_measure_2 = qp.ops.MidMeasure(0, id="2")
default_mid_measure_3 = qp.ops.MidMeasure(0, id="3")
default_measurement_value_1 = qp.ops.MeasurementValue([default_mid_measure_1], lambda v: v)
default_measurement_value_2 = qp.ops.MeasurementValue([default_mid_measure_2], lambda v: v)
default_measurement_value_3 = qp.ops.MeasurementValue([default_mid_measure_3], lambda v: v)
cond_bit_map_1 = {default_mid_measure_1: 0}
cond_bit_map_2 = {default_mid_measure_1: 0, default_mid_measure_2: 1}
stats_bit_map_1 = {default_mid_measure_1: 0, default_mid_measure_2: 1, default_mid_measure_3: 2}


def get_conditional_op(mv, true_fn, *args, **kwargs):
    """Helper to get conditional operator."""

    with qp.queuing.AnnotatedQueue() as q:
        qp.cond(mv, true_fn)(*args, **kwargs)

    return q.queue[0]


with qp.queuing.AnnotatedQueue() as q_tape:
    qp.RX(1.23456, wires=0)
    qp.RY(2.3456, wires="a")
    qp.RZ(3.4567, wires=1.234)


tape = qp.tape.QuantumScript.from_queue(q_tape)


def test_error_if_unsupported_object_in_tape():
    """Test an error is raised if there's an unsupported object in the tape."""

    # pylint: disable=too-few-public-methods
    class DummyObj:
        wires = qp.wires.Wires(2)

    _tape = qp.tape.QuantumScript([DummyObj()], [])

    with pytest.raises(NotImplementedError, match="unable to draw object"):
        qp.drawer.tape_text(_tape)


class TestHelperFunctions:  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """Test helper functions for the tape text."""

    @pytest.mark.parametrize(
        "op, out",
        [
            (qp.PauliX(0), ["", "", "", ""]),
            (qp.CNOT(wires=(0, 2)), ["╭", "│", "╰", ""]),
            (qp.CSWAP(wires=(0, 2, 3)), ["╭", "│", "├", "╰"]),
        ],
    )
    def test_add_grouping_symbols(self, op, out):
        """Test private _add_grouping_symbols function renders as expected."""
        config = _Config(
            wire_map=default_wire_map, bit_map=default_bit_map, num_op_layers=4, cur_layer=0
        )
        assert out == _add_grouping_symbols(op.wires, ["", "", "", ""], config)

    @pytest.mark.parametrize(
        "op, bit_map, layer_str, out",
        [
            (default_mid_measure_1, default_bit_map, ["", "", "", ""], ["", "", "", ""]),
            (
                default_mid_measure_1,
                cond_bit_map_1,
                ["", "", "", "", ""],
                ["", "─║", "─║", "─║", " ╚"],
            ),
            (
                default_mid_measure_2,
                cond_bit_map_2,
                ["─", "─", "─", "─", " ", " "],
                ["─", "──║", "──║", "──║", "  ║", "  ╚"],
            ),
        ],
    )
    def test_add_mid_measure_grouping_symbols(self, op, layer_str, bit_map, out):
        """Test private _add_grouping_symbols function renders as expected for MidMeasures."""
        config = _Config(wire_map=default_wire_map, bit_map=bit_map, num_op_layers=4, cur_layer=1)
        assert out == _add_mid_measure_grouping_symbols(op, layer_str, config)

    @pytest.mark.parametrize(
        "cond_op, args, kwargs, out, bit_map, mv, cur_layer",
        [
            (
                qp.PauliX,
                [],
                {"wires": 0},
                ["─", "─║", "─║", "─║", "═╩"],
                cond_bit_map_1,
                default_measurement_value_1,
                1,
            ),
            (
                qp.MultiRZ,
                [0.5],
                {"wires": [0, 1]},
                ["─", "─", "─║", "─║", "═╩"],
                cond_bit_map_1,
                default_measurement_value_1,
                1,
            ),
            (
                qp.Toffoli,
                [],
                {"wires": [0, 1, 2]},
                ["─", "─", "─", "─║", " ║", "═╝"],
                cond_bit_map_2,
                default_measurement_value_2,
                1,
            ),
            (
                qp.Toffoli,
                [],
                {"wires": [0, 1, 2]},
                ["─", "─", "─", "─║", "═╬", "═╝"],
                cond_bit_map_2,
                default_measurement_value_1 & default_measurement_value_2,
                1,
            ),
            (
                qp.Toffoli,
                [],
                {"wires": [0, 1, 2]},
                ["─", "─", "─", "─║", "═╣", "═╩"],
                cond_bit_map_2,
                default_measurement_value_1 & default_measurement_value_2,
                0,
            ),
        ],
    )
    def test_add_cond_grouping_symbols(self, cond_op, bit_map, mv, cur_layer, args, kwargs, out):
        """Test private _add_grouping_symbols function renders as expected for Conditionals."""
        op = get_conditional_op(mv, cond_op, *args, **kwargs)
        layer_str = ["─", "─", "─", ""] + [" "] * len(bit_map)

        config = _Config(
            wire_map=default_wire_map,
            bit_map=bit_map,
            cur_layer=cur_layer,
            cwire_layers={0: [[0]], 1: [[1]]},
            num_op_layers=4,
        )

        assert out == _add_cond_grouping_symbols(op, layer_str, config)

    @pytest.mark.parametrize(
        "mps, bit_map, out",
        [
            ([default_mid_measure_1], stats_bit_map_1, [" "] * (4 + len(stats_bit_map_1))),
            (
                [default_mid_measure_1, default_mid_measure_3],
                stats_bit_map_1,
                [" "] * len(default_wire_map) + ["╭", "│", "╰"],
            ),
            (
                [default_mid_measure_1, default_mid_measure_2, default_mid_measure_3],
                stats_bit_map_1,
                [" "] * len(default_wire_map) + ["╭", "├", "╰"],
            ),
        ],
    )
    def test_add_cwire_measurement_grouping_symbols(self, mps, bit_map, out):
        """Test private _add_cwire_measurement_grouping_symbols renders as expected."""
        layer_str = [" "] * (len(default_wire_map) + len(bit_map))
        config = _Config(wire_map=default_wire_map, bit_map=bit_map, num_op_layers=4, cur_layer=1)

        assert out == _add_cwire_measurement_grouping_symbols(mps, layer_str, config)

    @pytest.mark.parametrize(
        "mp, bit_map, out",
        [
            (
                qp.sample([default_measurement_value_1]),
                stats_bit_map_1,
                [" "] * len(default_wire_map) + [" Sample[MCM]", " ", " "],
            ),
            (
                qp.expval(default_measurement_value_1 * default_measurement_value_3),
                stats_bit_map_1,
                [" "] * len(default_wire_map) + ["╭<MCM>", "│", "╰<MCM>"],
            ),
            (
                qp.counts(
                    default_measurement_value_1
                    + default_measurement_value_2
                    - default_measurement_value_3
                ),
                stats_bit_map_1,
                [" "] * len(default_wire_map) + ["╭Counts[MCM]", "├Counts[MCM]", "╰Counts[MCM]"],
            ),
        ],
    )
    def test_add_cwire_measurement(self, mp, bit_map, out):
        """Test private _add_cwire_measurement renders as expected."""
        layer_str = [" "] * (len(default_wire_map) + len(bit_map))
        config = _Config(wire_map=default_wire_map, bit_map=bit_map, num_op_layers=4, cur_layer=1)
        assert out == _add_cwire_measurement(mp, layer_str, config)

    @pytest.mark.parametrize(
        "op, out",
        [
            (qp.expval(qp.PauliX(0)), ["<X>", "", "", ""]),
            (qp.probs(wires=(0, 2)), ["╭Probs", "│", "╰Probs", ""]),
            (qp.var(qp.PauliX(1)), ["", "Var[X]", "", ""]),
            (qp.state(), ["State", "State", "State", "State"]),
            (qp.sample(), ["Sample", "Sample", "Sample", "Sample"]),
            (qp.purity(0), ["purity", "", "", ""]),
            (qp.vn_entropy([2, 1]), ["", "╭vnentropy", "╰vnentropy", ""]),
            (qp.mutual_info(3, 1), ["", "╭mutualinfo", "│", "╰mutualinfo"]),
        ],
    )
    def test_add_measurements(self, op, out):
        """Test private _add_measurement function renders as expected."""
        config = _Config(
            wire_map=default_wire_map, bit_map=default_bit_map, num_op_layers=4, cur_layer=1
        )
        assert out == _add_measurement(op, [""] * 4, config)

    def test_add_measurements_cache(self):
        """Test private _add_measurement function with a matrix cache."""
        cache = {"matrices": []}
        op = qp.expval(qp.Hermitian(np.eye(2), wires=0))
        config = _Config(
            wire_map={0: 0, 1: 1},
            bit_map=default_bit_map,
            cache=cache,
            num_op_layers=4,
            cur_layer=1,
        )
        assert _add_measurement(op, ["", ""], config) == ["<𝓗(M0)>", ""]

        assert qp.math.allclose(cache["matrices"][0], np.eye(2))

        op2 = qp.expval(qp.Hermitian(np.eye(2), wires=1))
        # new op with same matrix, should have same M0 designation
        assert _add_measurement(op2, ["", ""], config) == ["", "<𝓗(M0)>"]

    @pytest.mark.parametrize(
        "op, out",
        [
            (qp.PauliX(0), ["─X", "─", "─", "─"]),
            (qp.CNOT(wires=(0, 2)), ["╭●", "│", "╰X", "─"]),
            (qp.Toffoli(wires=(0, 1, 3)), ["╭●", "├●", "│", "╰X"]),
            (qp.IsingXX(1.23, wires=(0, 2)), ["╭IsingXX", "│", "╰IsingXX", "─"]),
            (qp.Snapshot(), ["─|Snap|", "─|Snap|", "─|Snap|", "─|Snap|"]),
            (qp.Barrier(), ["─||", "─||", "─||", "─||"]),
            (qp.S(0) @ qp.T(0), ["─S@T", "─", "─", "─"]),
            (qp.TemporaryAND([0, 1, 3]), ["╭●", "├●", "│", "╰⊕"]),
            (qp.TemporaryAND([1, 0, 3], control_values=(0, 1)), ["╭●", "├○", "│", "╰⊕"]),
            (qp.ctrl(qp.TemporaryAND([0, 1, 2]), control=[3]), ["╭●", "├●", "├⊕", "╰●"]),
        ],
    )
    def test_add_obj(self, op, out):
        """Test adding the first operation to array of strings"""
        config = _Config(
            wire_map=default_wire_map, bit_map=default_bit_map, num_op_layers=4, cur_layer=1
        )
        assert out == _add_obj(op, ["─"] * 4, config)

    @pytest.mark.parametrize(
        "op, bit_map, layer_str, out",
        [
            (default_mid_measure_1, default_bit_map, ["─", "─", "─", "─"], ["─┤↗├", "─", "─", "─"]),
            (
                default_mid_measure_1,
                cond_bit_map_1,
                ["─", "─", "─", "─", " "],
                ["─┤↗├", "──║", "──║", "──║", "  ╚"],
            ),
            (
                default_mid_measure_2,
                cond_bit_map_2,
                ["─", "─", "─", "─", " ", " "],
                ["─┤↗├", "──║", "──║", "──║", "  ║", "  ╚"],
            ),
        ],
    )
    def test_add_mid_measure_op(self, op, layer_str, bit_map, out):
        """Test adding the first MidMeasure to array of strings"""
        config = _Config(wire_map=default_wire_map, bit_map=bit_map, num_op_layers=4, cur_layer=0)
        assert out == _add_obj(op, layer_str, config)

    @pytest.mark.parametrize(
        "cond_op, args, kwargs, out, bit_map, mv",
        [
            (
                qp.MultiRZ,
                [0.5],
                {"wires": [0, 1]},
                ["╭MultiRZ", "╰MultiRZ", "─║", "─║", "═╩"],
                cond_bit_map_1,
                default_measurement_value_1,
            ),
            (
                qp.Toffoli,
                [],
                {"wires": [0, 1, 2]},
                ["╭●", "├●", "╰X", "─║", "═╩"],
                cond_bit_map_1,
                default_measurement_value_1,
            ),
            (
                qp.PauliX,
                [],
                {"wires": 1},
                ["─", "─X", "─║", "─║", " ║", "═╝"],
                cond_bit_map_2,
                default_measurement_value_2,
            ),
        ],
    )
    def test_add_cond_op(self, cond_op, bit_map, mv, args, kwargs, out):
        """Test adding the first Conditional to array of strings"""
        op = get_conditional_op(mv, cond_op, *args, **kwargs)
        layer_str = ["─", "─", "─", ""] + [" "] * len(bit_map)
        config = _Config(
            wire_map=default_wire_map,
            bit_map=bit_map,
            cur_layer=1,
            cwire_layers={0: [[0]], 1: [[1]]},
            num_op_layers=4,
        )

        assert out == _add_obj(op, layer_str, config)

    @pytest.mark.parametrize(
        "op, out",
        [
            (qp.PauliY(1), ["─X", "─Y", "─", "─"]),
            (qp.CNOT(wires=(1, 2)), ["─X", "╭●", "╰X", "─"]),
            (qp.CRX(1.23, wires=(2, 3)), ["─X", "─", "╭●", "╰RX"]),
        ],
    )
    def test_add_second_op(self, op, out):
        """Test adding a second operation to the array of strings"""
        config = _Config(
            wire_map=default_wire_map, bit_map=default_bit_map, num_op_layers=4, cur_layer=1
        )
        start = _add_obj(qp.PauliX(0), ["─"] * 4, config)
        assert out == _add_obj(op, start, config)

    def test_add_obj_cache(self):
        """Test private _add_obj method functions with a matrix cache."""
        cache = {"matrices": []}
        op1 = qp.QubitUnitary(np.eye(2), wires=0)
        config = _Config(
            wire_map={0: 0, 1: 1},
            bit_map=default_bit_map,
            cache=cache,
            num_op_layers=4,
            cur_layer=1,
        )
        assert _add_obj(op1, ["", ""], config) == ["U(M0)", ""]

        assert qp.math.allclose(cache["matrices"][0], np.eye(2))
        op2 = qp.QubitUnitary(np.eye(2), wires=1)
        assert _add_obj(op2, ["", ""], config) == ["", "U(M0)"]

    @pytest.mark.parametrize("wires", [tuple(), (0, 1), (0, 1, 2, 3)])
    @pytest.mark.parametrize("wire_map", [default_wire_map, {0: 0, 1: 1}])
    @pytest.mark.parametrize("cls, label", [(qp.GlobalPhase, "GlobalPhase"), (qp.Identity, "I")])
    def test_add_global_op(self, wires, wire_map, cls, label):
        """Test that adding a global op works as expected."""
        data = [0.5124][: cls.num_params]
        op = cls(*data, wires=wires)
        # Expected output does not depend on the wires of GlobalPhase but just
        # on the number of drawn wires as dictated by the config!
        n_wires = len(wire_map)
        expected = [f"╭{label}"] + [f"├{label}"] * (n_wires - 2) + [f"╰{label}"]
        config = _Config(wire_map=wire_map, bit_map=default_bit_map, num_op_layers=4, cur_layer=1)
        out = _add_obj(op, ["─"] * n_wires, config)
        assert expected == out

    @pytest.mark.parametrize(
        "wires, control_wires, expected",
        [
            (tuple(), (0,), ["╭●", "├label", "├label", "╰label"]),
            (tuple(), (2,), ["╭label", "├label", "├●", "╰label"]),
            ((2,), (0, 1, 3), ["╭●", "├●", "├label", "╰●"]),
            ((0, 1), (3,), ["╭label", "├label", "├label", "╰●"]),
            ((0, 2), (1, 3), ["╭label", "├●", "├label", "╰●"]),
            ((0, 1, 3), (2,), ["╭label", "├label", "├●", "╰label"]),
        ],
    )
    @pytest.mark.parametrize("wire_map", [default_wire_map, {i: i for i in range(6)}])
    @pytest.mark.parametrize("cls, label", [(qp.GlobalPhase, "GlobalPhase"), (qp.Identity, "I")])
    def test_add_controlled_global_op(self, wires, control_wires, expected, wire_map, cls, label):
        """Test that adding a controlled global op works as expected."""
        expected = copy(expected)
        data = [0.5124][: cls.num_params]
        op = qp.ctrl(cls(*data, wires=wires), control=control_wires)
        n_wires = len(wire_map)
        if n_wires > 4:
            expected[-1] = "├" + expected[-1][1:]
            expected.extend(["├label"] * (n_wires - 5))
            expected.append("╰label")

        expected = [line.replace("label", label) for line in expected]
        config = _Config(wire_map=wire_map, bit_map=default_bit_map, num_op_layers=4, cur_layer=1)
        out = _add_obj(op, ["─"] * n_wires, config)
        assert expected == out


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

    def test_hiding_labels(self):
        """Test that printing wire labels can be skipped with show_wire_labels=False."""

        split_str = tape_text(tape, show_wire_labels=False).split("\n")
        assert split_str[0].startswith("─")
        assert split_str[1].startswith("─")
        assert split_str[2].startswith("─")


class TestDecimals:
    """Test the decimals keyword argument."""

    def test_decimals(self):
        """Test that the decimals keyword makes the operation parameters included."""

        expected = "    0: ──RX(1.23)─┤  \n    a: ──RY(2.35)─┤  \n1.234: ──RZ(3.46)─┤  "

        assert tape_text(tape, decimals=2) == expected

    def test_decimals_multiparameters(self):
        """Tests decimals also displays parameters when the operation has multiple parameters."""

        with qp.queuing.AnnotatedQueue() as q_tape_rot:
            qp.Rot(1.2345, 2.3456, 3.4566, wires=0)

        tape_rot = qp.tape.QuantumScript.from_queue(q_tape_rot)
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

        with qp.queuing.AnnotatedQueue() as q_tape_torch:
            qp.Rot(torch.tensor(1.234), torch.tensor(2.345), torch.tensor(3.456), wires=0)

        tape_torch = qp.tape.QuantumScript.from_queue(q_tape_torch)
        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_torch, decimals=2) == expected

    @pytest.mark.tf
    def test_tensorflow_parameters(self):
        """Test tensorflow parameters display as normal numbers."""
        import tensorflow as tf

        with qp.queuing.AnnotatedQueue() as q_tape_tf:
            qp.Rot(tf.Variable(1.234), tf.Variable(2.345), tf.Variable(3.456), wires=0)

        tape_tf = qp.tape.QuantumScript.from_queue(q_tape_tf)
        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_tf, decimals=2) == expected

    @pytest.mark.jax
    def test_jax_parameters(self):
        """Test jax parameters in tape display as normal numbers."""
        import jax.numpy as jnp

        with qp.queuing.AnnotatedQueue() as q_tape_jax:
            qp.Rot(jnp.array(1.234), jnp.array(2.345), jnp.array(3.456), wires=0)

        tape_jax = qp.tape.QuantumScript.from_queue(q_tape_jax)
        expected = "0: ──Rot(1.23,2.35,3.46)─┤  "
        assert tape_text(tape_jax, decimals=2) == expected


class TestMaxLength:
    """Test the max_length keyword."""

    def test_max_length_default(self):
        """Test max length defaults to 100."""
        with qp.queuing.AnnotatedQueue() as q_tape_ml:
            for _ in range(50):
                qp.PauliX(0)
                qp.PauliY(1)

            for _ in range(3):
                qp.sample()

        tape_ml = qp.tape.QuantumScript.from_queue(q_tape_ml)
        out = tape_text(tape_ml)

        assert 95 <= max(len(s) for s in out.split("\n")) <= 100

    # We choose values of max_length that allow us to include continuation dots
    # when the circuit is partitioned
    @pytest.mark.parametrize("ml", [25, 50, 75])
    def test_setting_max_length(self, ml):
        """Test several custom max_length parameters change the wrapping length."""

        with qp.queuing.AnnotatedQueue() as q_tape_ml:
            for _ in range(50):
                qp.PauliX(0)
                qp.PauliY(1)

            for _ in range(3):
                qp.sample()

        tape_ml = qp.tape.QuantumScript.from_queue(q_tape_ml)
        out = tape_text(tape_ml, max_length=ml)

        assert max(len(s) for s in out.split("\n")) <= ml


single_op_tests_data = [
    (
        qp.MultiControlledX(wires=[0, 1, 2, 3], control_values=[0, 1, 0]),
        "0: ─╭○─┤  \n1: ─├●─┤  \n2: ─├○─┤  \n3: ─╰X─┤  ",
    ),
    (
        # pylint:disable=no-member
        qp.ops.op_math.Controlled(qp.PauliY(3), (0, 1, 2), [0, 1, 0]),
        "0: ─╭○─┤  \n1: ─├●─┤  \n2: ─├○─┤  \n3: ─╰Y─┤  ",
    ),
    (qp.CNOT(wires=(0, 1)), "0: ─╭●─┤  \n1: ─╰X─┤  "),
    (qp.Toffoli(wires=(0, 1, 2)), "0: ─╭●─┤  \n1: ─├●─┤  \n2: ─╰X─┤  "),
    (qp.Barrier(wires=(0, 1, 2)), "0: ─╭||─┤  \n1: ─├||─┤  \n2: ─╰||─┤  "),
    (qp.CSWAP(wires=(0, 1, 2)), "0: ─╭●────┤  \n1: ─├SWAP─┤  \n2: ─╰SWAP─┤  "),
    (
        qp.DoubleExcitationPlus(1.23, wires=(0, 1, 2, 3)),
        "0: ─╭G²₊(1.23)─┤  \n1: ─├G²₊(1.23)─┤  \n2: ─├G²₊(1.23)─┤  \n3: ─╰G²₊(1.23)─┤  ",
    ),
    (qp.QubitUnitary(qp.numpy.eye(4), wires=(0, 1)), "0: ─╭U(M0)─┤  \n1: ─╰U(M0)─┤  "),
    (qp.QubitSum(wires=(0, 1, 2)), "0: ─╭Σ─┤  \n1: ─├Σ─┤  \n2: ─╰Σ─┤  "),
    (qp.AmplitudeDamping(0.98, wires=0), "0: ──AmplitudeDamping(0.98)─┤  "),
    (
        qp.StatePrep([0, 1, 0, 0], wires=(0, 1)),
        "0: ─╭|Ψ⟩─┤  \n1: ─╰|Ψ⟩─┤  ",
    ),
    (qp.Kerr(1.234, wires=0), "0: ──Kerr(1.23)─┤  "),
    (
        qp.GroverOperator(wires=(0, 1, 2)),
        "0: ─╭GroverOperator─┤  \n1: ─├GroverOperator─┤  \n2: ─╰GroverOperator─┤  ",
    ),
    (
        qp.adjoint(qp.RX(1.234, wires=0)),
        "0: ──RX(1.23)†─┤  ",
    ),
    (
        qp.RX(1.234, wires=0) ** -1,
        "0: ──RX(1.23)⁻¹─┤  ",
    ),
    (qp.expval(qp.PauliZ(0)), "0: ───┤  <Z>"),
    (qp.var(qp.PauliZ(0)), "0: ───┤  Var[Z]"),
    (qp.probs(wires=0), "0: ───┤  Probs"),
    (qp.probs(op=qp.PauliZ(0)), "0: ───┤  Probs[Z]"),
    (qp.sample(wires=0), "0: ───┤  Sample"),
    (qp.sample(op=qp.PauliX(0)), "0: ───┤  Sample[X]"),
    (
        qp.expval(0.1 * qp.PauliX(0) @ qp.PauliY(1)),
        "0: ───┤ ╭<(0.10*X)@Y>\n1: ───┤ ╰<(0.10*X)@Y>",
    ),
    (
        qp.expval(
            0.1 * qp.PauliX(0) + 0.2 * qp.PauliY(1) + 0.3 * qp.PauliZ(0) + 0.4 * qp.PauliZ(1)
        ),
        "0: ───┤ ╭<𝓗>\n1: ───┤ ╰<𝓗>",
    ),
    # Operations (both regular and controlled) and nested multi-valued controls
    (qp.ctrl(qp.PauliX(wires=2), control=[0, 1]), "0: ─╭●─┤  \n1: ─├●─┤  \n2: ─╰X─┤  "),
    (qp.ctrl(qp.CNOT(wires=[1, 2]), control=0), "0: ─╭●─┤  \n1: ─├●─┤  \n2: ─╰X─┤  "),
    (
        qp.ctrl(qp.CRZ(0.2, wires=[1, 2]), control=[3, 0]),
        "3: ─╭●────────┤  \n0: ─├●────────┤  \n1: ─├●────────┤  \n2: ─╰RZ(0.20)─┤  ",
    ),
    (
        qp.ctrl(qp.CH(wires=[0, 3]), control=[2, 1], control_values=[False, True]),
        "2: ─╭○─┤  \n1: ─├●─┤  \n0: ─├●─┤  \n3: ─╰H─┤  ",
    ),
    (
        qp.ctrl(
            qp.ctrl(qp.CY(wires=[3, 4]), control=[1, 2], control_values=[True, False]),
            control=0,
            control_values=[False],
        ),
        "0: ─╭○─┤  \n1: ─├●─┤  \n2: ─├○─┤  \n3: ─├●─┤  \n4: ─╰Y─┤  ",
    ),
    (
        qp.TemporaryAND([3, 0, 2], control_values=(1, 0)),
        "3: ─╭●─┤  \n0: ─├○─┤  \n2: ─╰⊕─┤  ",
    ),
    (
        qp.adjoint(qp.TemporaryAND([3, 0, 2], control_values=(0, 1))),
        "3: ──○╮─┤  \n0: ──●┤─┤  \n2: ──⊕╯─┤  ",
    ),
]


@pytest.mark.parametrize("op, expected", single_op_tests_data)
def test_single_ops(op, expected):
    """Tests a variety of different single operation tapes render as expected."""

    with qp.queuing.AnnotatedQueue() as q:
        qp.apply(op)

    _tape = qp.tape.QuantumScript.from_queue(q)
    assert tape_text(_tape, decimals=2, show_matrices=False) == expected


class TestLayering:
    """Test operations are placed in the correct locations."""

    def test_adjacent_ops(self):
        """Test non-blocking gates end up on same layer."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.PauliX(0)
            qp.PauliX(1)
            qp.PauliX(2)

        _tape = qp.tape.QuantumScript.from_queue(q)
        assert tape_text(_tape) == "0: ──X─┤  \n1: ──X─┤  \n2: ──X─┤  "

    def test_blocking_ops(self):
        """Test single qubit gates on same wire line up."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.PauliX(0)
            qp.PauliX(0)
            qp.PauliX(0)

        _tape = qp.tape.QuantumScript.from_queue(q)
        assert tape_text(_tape) == "0: ──X──X──X─┤  "

    def test_blocking_multiwire_gate(self):
        """Tests gate gets blocked by multi-wire gate."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.PauliX(0)
            qp.IsingXX(1.2345, wires=(0, 2))
            qp.PauliX(1)

        _tape = qp.tape.QuantumScript.from_queue(q)
        expected = "0: ──X─╭IsingXX────┤  \n1: ────│─────────X─┤  \n2: ────╰IsingXX────┤  "

        assert tape_text(_tape, wire_order=[0, 1, 2]) == expected

    def test_multiple_elbows(self):
        """Test that multiple elbows are drawn correctly."""
        _tape = qp.tape.QuantumScript(
            [
                qp.TemporaryAND(["a", "b", "c"]),
                qp.adjoint(qp.TemporaryAND(["f", "d", "e"])),
                qp.adjoint(qp.TemporaryAND(["a", "d", "b"], control_values=(0, 0))),
                qp.TemporaryAND(["e", "h", "f"], control_values=(0, 1)),
            ]
        )
        expected = (
            "a: ─╭●───○╮─┤  \n"
            "b: ─├●───⊕┤─┤  \n"
            "c: ─╰⊕────│─┤  \n"
            "d: ──●╮──○╯─┤  \n"
            "e: ──⊕┤─╭○──┤  \n"
            "f: ──●╯─├⊕──┤  \n"
            "g: ─────│───┤  \n"
            "h: ─────╰●──┤  "
        )
        out = tape_text(
            _tape, wire_order=["a", "b", "c", "d", "e", "f", "g", "h"], show_all_wires=True
        )
        assert out == expected


tape_matrices = qp.tape.QuantumScript(
    ops=[qp.StatePrep([1.0, 0.0, 0.0, 0.0], wires=(0, 1)), qp.QubitUnitary(np.eye(2), wires=0)],
    measurements=[qp.expval(qp.Hermitian(np.eye(2), wires=0))],
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


def test_nested_tapes():
    """Test nested tapes inside the qnode."""

    def circ():
        with qp.tape.QuantumTape():
            qp.PauliX(0)
            with qp.tape.QuantumTape():
                qp.PauliY(0)
        with qp.tape.QuantumTape():
            qp.PauliZ(0)
            with qp.tape.QuantumTape():
                qp.PauliX(0)
        return qp.expval(qp.PauliZ(0))

    expected = (
        "0: ──Tape:0──Tape:1─┤  <Z>\n\n"
        "Tape:0\n0: ──X──Tape:2─┤  \n\n"
        "Tape:2\n0: ──Y─┤  \n\n"
        "Tape:1\n0: ──Z──Tape:3─┤  \n\n"
        "Tape:3\n0: ──X─┤  "
    )

    assert qp.draw(circ)() == expected
