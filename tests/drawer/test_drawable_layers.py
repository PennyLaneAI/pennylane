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

import pytest

import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.queuing import AnnotatedQueue
from pennylane.drawer.drawable_layers import (
    _recursive_find_layer,
    drawable_layers,
)


class TestRecursiveFindLayer:
    """Tests for `_recursive_find_layer`"""

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


class TestDrawableLayers:
    """Tests for `drawable_layers`"""

    def test_single_wires_no_blocking(self):
        """Test simple case where nothing blocks each other"""

        ops = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]

        layers = drawable_layers(ops)

        assert layers == [ops]

    def test_single_wires_blocking(self):
        """Test single wire gates blocking each other"""

        ops = [qml.PauliX(0), qml.PauliX(0), qml.PauliX(0)]

        layers = drawable_layers(ops)

        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    def test_barrier_only_visual(self):
        """Test the barrier is always drawn"""

        ops = [
            qml.PauliX(0),
            qml.Barrier(wires=0),
            qml.Barrier(only_visual=True, wires=0),
            qml.PauliX(0),
        ]
        layers = drawable_layers(ops)
        assert layers == [[ops[0]], [ops[1]], [ops[2]], [ops[3]]]

    def test_barrier_block(self):
        """Test the barrier blocking operators"""

        ops = [qml.PauliX(0), qml.Barrier(wires=[0, 1]), qml.PauliX(1)]
        layers = drawable_layers(ops)
        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    def test_wirecut_block(self):
        """Test the wirecut blocking operators"""

        ops = [qml.PauliX(0), qml.WireCut(wires=[0, 1]), qml.PauliX(1)]
        layers = drawable_layers(ops)
        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    @pytest.mark.parametrize(
        "multiwire_gate",
        (
            qml.CNOT(wires=(0, 2)),
            qml.CNOT(wires=(2, 0)),
            qml.Toffoli(wires=(0, 2, 3)),
            qml.Toffoli(wires=(2, 3, 0)),
            qml.Toffoli(wires=(3, 0, 2)),
            qml.Toffoli(wires=(0, 3, 2)),
            qml.Toffoli(wires=(3, 2, 0)),
            qml.Toffoli(wires=(2, 0, 3)),
        ),
    )
    def test_multiwire_blocking(self, multiwire_gate):
        """Test multi-wire gate blocks on unused wire"""

        wire_map = {0: 0, 1: 1, 2: 2, 3: 3}
        ops = [qml.PauliZ(1), multiwire_gate, qml.PauliX(1)]

        layers = drawable_layers(ops, wire_map=wire_map)

        assert layers == [[ops[0]], [ops[1]], [ops[2]]]

    @pytest.mark.parametrize("measurement", (qml.state(), qml.sample()))
    def test_all_wires_measurement(self, measurement):
        """Test measurements that act on all wires also block on all available wires."""

        ops = [qml.PauliX(0), measurement, qml.PauliY(1)]

        layers = drawable_layers(ops)

        assert layers == [[ops[0]], [ops[1]], [ops[2]]]


class TestMidMeasure:
    """Tests the various changes from mid-circuit measurements."""

    def test_basic_mid_measure(self):
        """Tests a simple case with mid-circuit measurement."""
        with AnnotatedQueue() as q:
            m0 = qml.measure(0)
            qml.cond(m0, qml.PauliX)(1)

        assert drawable_layers(q.queue) == [[q.queue[0]], [q.queue[1]]]

    def test_mid_measure_promoted_if_needed(self):
        """Tests promotion of MidMeasureMPs if used for the same conditional."""
        with AnnotatedQueue() as q:
            m0 = qml.measure(0)
            qml.Hadamard(1)
            m1 = qml.measure(1)
            qml.cond(m0 + m1, qml.PauliZ)(2)

        ops = q.queue
        assert drawable_layers(ops) == [[ops[1]], [ops[2], ops[0]], [ops[3]]]

    def test_empty_layers_are_pruned(self):
        """Tests that no empty layers are returned after MidMeasure promotion."""
        with AnnotatedQueue() as q:
            m0 = qml.measure(1)
            qml.CNOT([0, 2])
            m1 = qml.measure(0)
            qml.cond(m0 + m1, qml.PauliZ)(2)

        ops = q.queue
        layers = drawable_layers(ops, wire_map={i: i for i in range(3)})
        assert layers == [[ops[1]], [ops[2], ops[0]], [ops[3]]]

    def test_cannot_draw_multi_wire_MidMeasureMP(self):
        """Tests that MidMeasureMP is only supported with one wire."""
        with pytest.raises(ValueError, match="mid-circuit measurements with more than one wire."):
            drawable_layers([MidMeasureMP([0, 1])])
