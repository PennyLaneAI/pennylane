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
Unit tests for the pennylane.drawer.drawable_layers` module.
"""

import sys

import pytest

import pennylane as qp
from pennylane.drawer.drawable_layers import (
    _recursive_find_layer,
    _recursive_find_mcm_stats_layer,
    drawable_layers,
)
from pennylane.ops import MidMeasure
from pennylane.queuing import AnnotatedQueue


class TestRecursiveFindLayer:
    """Tests for ``_recursive_find_layer`` and ``_recursive_find_mcm_stats_layer``"""

    def test_first_layer(self):
        """Test operation remains in 0th layer if not blocked"""
        out = _recursive_find_layer(
            layer_to_check=0, op_occupied_wires={0}, occupied_wires_per_layer=[{1}]
        )
        assert out == 0

    def test_blocked_layer(self):
        """Test operation moved to higher layer if blocked on 0th layer."""
        out = _recursive_find_layer(
            layer_to_check=0, op_occupied_wires={0}, occupied_wires_per_layer=[{0}]
        )
        assert out == 1

    def test_recursion_no_block(self):
        """Test recursion to zero if start in higher layer and not blocked"""
        out = _recursive_find_layer(
            layer_to_check=2, op_occupied_wires={0}, occupied_wires_per_layer=[{1}, {1}, {1}]
        )
        assert out == 0

    def test_recursion_block(self):
        """Test blocked on layer 1 gives placement on layer 2"""
        out = _recursive_find_layer(
            layer_to_check=3, op_occupied_wires={0}, occupied_wires_per_layer=[{1}, {0}, {1}, {1}]
        )
        assert out == 2

    def test_first_mcm_stats_layer(self):
        """Test operation remains in 0th layer if not blocked"""
        out = _recursive_find_mcm_stats_layer(
            layer_to_check=0, op_occupied_cwires={0}, used_cwires_per_layer=[{1}]
        )
        assert out == 0

    def test_blocked_mcm_stats_layer(self):
        """Test operation moved to higher layer if blocked on 0th layer."""
        out = _recursive_find_mcm_stats_layer(
            layer_to_check=0, op_occupied_cwires={0}, used_cwires_per_layer=[{0}]
        )
        assert out == 1

    def test_recursion_no_block_mcm_stats(self):
        """Test recursion to zero if start in higher layer and not blocked"""
        out = _recursive_find_mcm_stats_layer(
            layer_to_check=2,
            op_occupied_cwires={0},
            used_cwires_per_layer=[{1}, {1}, {1}],
        )
        assert out == 0

    def test_recursion_block_mcm_stats(self):
        """Test blocked on layer 1 gives placement on layer 2"""
        out = _recursive_find_mcm_stats_layer(
            layer_to_check=3,
            op_occupied_cwires={0},
            used_cwires_per_layer=[{1}, {0}, {1}, {1}],
        )
        assert out == 2


class TestDrawableLayers:
    """Tests for `drawable_layers`"""

    def test_recursion_error(self):
        """Test that extremely deep circuits are handled with an informative message."""

        ops = [qp.X(0)] * (sys.getrecursionlimit() + 1) + [qp.X(1)]

        with pytest.raises(RecursionError, match=r"which is too deep to handle"):
            drawable_layers(ops)

    def test_single_wires_no_blocking(self):
        """Test simple case where nothing blocks each other"""

        ops = [qp.PauliX(0), qp.PauliX(1), qp.PauliX(2)]

        layers = drawable_layers(ops)

        assert layers == [ops]

    def test_single_wires_blocking(self):
        """Test single wire gates blocking each other"""

        ops = [qp.PauliX(0), qp.PauliX(0), qp.PauliX(0)]

        layers = drawable_layers(ops)

        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    def test_barrier_only_visual(self):
        """Test the barrier is always drawn"""

        ops = [
            qp.PauliX(0),
            qp.Barrier(wires=0),
            qp.Barrier(only_visual=True, wires=0),
            qp.PauliX(0),
        ]
        layers = drawable_layers(ops)
        assert layers == [[ops[0]], [ops[1]], [ops[2]], [ops[3]]]

    def test_barrier_block(self):
        """Test the barrier blocking operators"""

        ops = [qp.PauliX(0), qp.Barrier(wires=[0, 1]), qp.PauliX(1)]
        layers = drawable_layers(ops)
        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    def test_wirecut_block(self):
        """Test the wirecut blocking operators"""

        ops = [qp.PauliX(0), qp.WireCut(wires=[0, 1]), qp.PauliX(1)]
        layers = drawable_layers(ops)
        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    @pytest.mark.parametrize(
        "multiwire_gate",
        (
            qp.CNOT(wires=(0, 2)),
            qp.CNOT(wires=(2, 0)),
            qp.Toffoli(wires=(0, 2, 3)),
            qp.Toffoli(wires=(2, 3, 0)),
            qp.Toffoli(wires=(3, 0, 2)),
            qp.Toffoli(wires=(0, 3, 2)),
            qp.Toffoli(wires=(3, 2, 0)),
            qp.Toffoli(wires=(2, 0, 3)),
        ),
    )
    def test_multiwire_blocking(self, multiwire_gate):
        """Test multi-wire gate blocks on unused wire"""

        wire_map = {0: 0, 1: 1, 2: 2, 3: 3}
        ops = [qp.PauliZ(1), multiwire_gate, qp.PauliX(1)]

        layers = drawable_layers(ops, wire_map=wire_map)

        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    @pytest.mark.parametrize("measurement", (qp.state(), qp.sample()))
    def test_all_wires_measurement(self, measurement):
        """Test measurements that act on all wires also block on all available wires."""

        ops = [qp.PauliX(0), measurement, qp.PauliY(1)]

        layers = drawable_layers(ops)

        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    def test_mid_measure_custom_wires(self):
        """Test that custom wires do not break the drawing of mid-circuit measurements."""
        mp0 = MidMeasure("A", id="foo")
        mp1 = MidMeasure("a", id="bar")
        m0 = qp.ops.MeasurementValue([mp0], lambda v: v)
        m1 = qp.ops.MeasurementValue([mp1], lambda v: v)

        def teleport(state):
            qp.StatePrep(state, wires=["A"])
            qp.Hadamard(wires="a")
            qp.CNOT(wires=["a", "B"])
            qp.CNOT(wires=["A", "a"])
            qp.Hadamard(wires="A")
            qp.apply(mp0)
            qp.apply(mp1)
            qp.cond(m1, qp.PauliX)("B")
            qp.cond(m0, qp.PauliZ)("B")

        tape_custom = qp.tape.make_qscript(teleport)([0, 1])
        [tape_standard], _ = qp.map_wires(tape_custom, {"A": 0, "a": 1, "B": 2})
        ops = tape_standard.operations
        bit_map = {MidMeasure(0, id="foo"): None, MidMeasure(1, id="bar"): None}
        layers = drawable_layers(ops, bit_map=bit_map)
        assert layers == [ops[:2]] + [[op] for op in ops[2:]]


def test_basic_mid_measure():
    """Tests a simple case with mid-circuit measurement."""
    with AnnotatedQueue() as q:
        m0 = qp.measure(0)
        qp.cond(m0, qp.PauliX)(1)

    bit_map = {q.queue[0]: None}

    assert drawable_layers(q.queue, bit_map=bit_map) == [[q.queue[0]], [q.queue[1]]]
