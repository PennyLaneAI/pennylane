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
Unit tests for the pennylane.drawer.utils` module.
"""

import pytest

import pennylane as qml
from pennylane.drawer.utils import (
    convert_wire_order,
    cwire_connections,
    default_bit_map,
    default_wire_map,
    unwrap_controls,
)
from pennylane.wires import Wires


class TestDefaultWireMap:
    """Tests ``_default_wire_map`` helper function."""

    def test_empty(self):
        """Test creating an empty wire map"""

        wire_map = default_wire_map([])
        assert wire_map == {}

    def test_simple(self):
        """Test creating a wire map with wires that do not have successive ordering"""

        ops = [qml.PauliX(0), qml.PauliX(2), qml.PauliX(1)]

        wire_map = default_wire_map(ops)
        assert wire_map == {0: 0, 2: 1, 1: 2}

    def test_string_wires(self):
        """Test wire map works with string labelled wires."""

        ops = [qml.PauliY("a"), qml.CNOT(wires=("b", "c"))]

        wire_map = default_wire_map(ops)
        assert wire_map == {"a": 0, "b": 1, "c": 2}


class TestDefaultBitMap:
    """Tests ``default_bit_map`` helper function."""

    def test_empty(self):
        """Test creating empty bit map"""
        # pylint: disable=use-implicit-booleaness-not-comparison
        bit_map = default_bit_map([])
        assert bit_map == {}

        bit_map = default_bit_map([qml.measurements.MidMeasureMP(0)])
        assert bit_map == {}

    def test_simple(self):
        """Test that the bit_map contains only measurements that are used."""

        m0 = qml.measure(0)
        m1 = qml.measure(1)
        m2 = qml.measure(2)
        cond0 = qml.ops.Conditional(m0, qml.S(0))
        mp0 = qml.expval(m1)

        queue = [m0.measurements[0], m1.measurements[0], m2.measurements[0], cond0, mp0]
        bit_map = default_bit_map(queue)
        assert bit_map == {m0.measurements[0]: 0, m1.measurements[0]: 1}


class TestConvertWireOrder:
    """Tests the ``convert_wire_order`` utility function."""

    def test_no_wire_order(self):
        """Test that a wire map is produced if no wire order is passed."""

        ops = [qml.PauliX(0), qml.PauliX(2), qml.PauliX(1)]

        wire_map = convert_wire_order(ops)

        assert wire_map == {0: 0, 2: 1, 1: 2}

    def test_wire_order_ints(self):
        """Tests wire map produced when initial wires are integers."""

        ops = [qml.PauliX(0), qml.PauliX(2), qml.PauliX(1)]
        wire_order = [2, 1, 0]

        wire_map = convert_wire_order(ops, wire_order)
        assert wire_map == {2: 0, 1: 1, 0: 2}

    def test_wire_order_str(self):
        """Test wire map produced when initial wires are strings."""

        ops = [qml.CNOT(wires=("a", "b")), qml.PauliX("c")]
        wire_order = ("c", "b", "a")

        wire_map = convert_wire_order(ops, wire_order)
        assert wire_map == {"c": 0, "b": 1, "a": 2}

    def test_show_all_wires_false(self):
        """Test when `show_all_wires` is set to `False` only used wires are in the map."""

        ops = [qml.PauliX("a"), qml.PauliY("c")]
        wire_order = ["a", "b", "c", "d"]

        wire_map = convert_wire_order(ops, wire_order, show_all_wires=False)
        assert wire_map == {"a": 0, "c": 1}

    def test_show_all_wires_true(self):
        """Test when `show_all_wires` is set to `True` everything in ``wire_order`` is included."""

        ops = [qml.PauliX("a"), qml.PauliY("c")]
        wire_order = ["a", "b", "c", "d"]

        wire_map = convert_wire_order(ops, wire_order, show_all_wires=True)
        assert wire_map == {"a": 0, "b": 1, "c": 2, "d": 3}


class TestUnwrapControls:
    """Tests the ``unwrap_controls`` utility function."""

    # pylint:disable=too-few-public-methods

    @pytest.mark.parametrize(
        "op,expected_control_wires,expected_control_values",
        [
            (qml.PauliX(wires="a"), Wires([]), None),
            (qml.CNOT(wires=["a", "b"]), Wires("a"), [True]),
            (qml.ctrl(qml.PauliX(wires="b"), control="a"), Wires("a"), [True]),
            (
                qml.ctrl(qml.PauliX(wires="b"), control=["a", "c", "d"]),
                Wires(["a", "c", "d"]),
                [True, True, True],
            ),
            (
                qml.ctrl(qml.PauliZ(wires="c"), control=["a", "d"], control_values=[True, False]),
                Wires(["a", "d"]),
                [True, False],
            ),
            (
                qml.ctrl(
                    qml.CRX(0.3, wires=["c", "e"]),
                    control=["a", "b", "d"],
                    control_values=[True, False, False],
                ),
                Wires(["a", "b", "d", "c"]),
                [True, False, False, True],
            ),
            (
                qml.ctrl(qml.CNOT(wires=["c", "d"]), control=["a", "b"]),
                Wires(["a", "b", "c"]),
                [True, True, True],
            ),
            (
                qml.ctrl(qml.ctrl(qml.CNOT(wires=["c", "d"]), control=["a", "b"]), control=["e"]),
                Wires(["e", "a", "b", "c"]),
                [True, True, True, True],
            ),
            (
                qml.ctrl(
                    qml.ctrl(
                        qml.CNOT(wires=["c", "d"]), control=["a", "b"], control_values=[False, True]
                    ),
                    control=["e"],
                    control_values=[False],
                ),
                Wires(["e", "a", "b", "c"]),
                [False, False, True, True],
            ),
        ],
    )
    def test_multi_defined_control_values(
        self, op, expected_control_wires, expected_control_values
    ):
        """Test a multi-controlled single-qubit operation with defined control values."""
        control_wires, control_values = unwrap_controls(op)

        assert control_wires == expected_control_wires
        assert control_values == expected_control_values


# pylint: disable=use-implicit-booleaness-not-comparison
class TestCwireConnections:
    """Tests for the cwire_connections helper method."""

    def test_null_circuit(self):
        """Test null behavior with an empty circuit."""
        bit_map, layers, wires = cwire_connections([[]], {})
        assert layers == {}
        assert wires == {}
        assert bit_map == {}

    def test_single_measure(self):
        """Test a single meassurment that does not have a conditional."""
        bit_map, layers, wires = cwire_connections([qml.measure(0).measurements], {})
        assert layers == {}
        assert wires == {}
        assert bit_map == {}

    def test_single_measure_single_cond(self):
        """Test a case with a single measurement and a single conditional."""
        m = qml.measure(0)
        cond = qml.ops.Conditional(m, qml.PauliX(0))
        layers = [m.measurements, [cond]]
        bit_map = {m.measurements[0]: 0}

        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 1]]}
        assert wires == {0: [[0, 0]]}
        assert new_bit_map == bit_map

    def test_multiple_measure_multiple_cond(self):
        """Test a case with multiple measurements and multiple conditionals."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        m2_nonused = qml.measure(2)

        cond0 = qml.ops.Conditional(m0 + m1, qml.PauliX(1))
        cond1 = qml.ops.Conditional(m1, qml.PauliY(2))
        bit_map = {m0.measurements[0]: 0, m1.measurements[0]: 1}

        layers = [m0.measurements, m1.measurements, [cond0], m2_nonused.measurements, [cond1]]
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 2]], 1: [[1, 2, 4]]}
        assert wires == {0: [[0, 1]], 1: [[1, 1, 2]]}
        assert new_bit_map == bit_map

    def test_measurements_layer(self):
        """Test cwire_connections works if measurement layers are appended at the end."""

        m0 = qml.measure(0)
        cond0 = qml.ops.Conditional(m0, qml.S(0))
        layers = [m0.measurements, [cond0], [qml.expval(qml.PauliX(0))]]
        bit_map = {m0.measurements[0]: 0}
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 1]]}
        assert wires == {0: [[0, 0]]}
        assert new_bit_map == bit_map

    def test_mid_measure_stats_layer(self):
        """Test cwire_connections works if layers contain terminal measurements using measurement
        values"""

        m0 = qml.measure(0)
        layers = [m0.measurements, [qml.expval(m0)]]
        bit_map = {m0.measurements[0]: 0}
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 1]]}
        assert wires == {0: [[0]]}
        assert new_bit_map == bit_map

    def test_single_mid_measure_cond_and_stats_layer(self):
        """Test cwire_connections works if layers contain terminal measurements using measurement
        values"""

        m0 = qml.measure(1)
        cond0 = qml.ops.Conditional(m0, qml.X(0))
        layers = [m0.measurements, [cond0], [qml.expval(m0)]]
        bit_map = {m0.measurements[0]: 0}
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 1, 2]]}
        assert wires == {0: [[1, 0]]}
        assert new_bit_map == bit_map

    def test_multi_mid_measure_stats_layer(self):
        """Test cwire_connections works if layers contain multiple MCMs, no conditionals,
        and one or multiple terminal measurements using measurement values"""

        m0 = qml.measure(0)
        m1 = qml.measure(1)
        m2 = qml.measure(0)

        # final expvals prevent wire reusing
        layers = [
            m0.measurements,
            m1.measurements,
            m2.measurements,
            [qml.expval(m0), qml.expval(m1), qml.expval(m2)],
        ]
        bit_map = {m0.measurements[0]: 0, m1.measurements[0]: 1, m2.measurements[0]: 2}
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 3]], 1: [[1, 3]], 2: [[2, 3]]}
        assert wires == {0: [[0]], 1: [[1]], 2: [[0]]}
        assert new_bit_map == bit_map

        # should not draw cwire for m2 if there is no usage of it
        layers = [
            m0.measurements,
            m1.measurements,
            m2.measurements,
            [qml.expval(m0), qml.expval(m1)],
        ]
        bit_map = {m0.measurements[0]: 0, m1.measurements[0]: 1}
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 3]], 1: [[1, 3]]}
        assert wires == {0: [[0]], 1: [[1]]}
        assert new_bit_map == bit_map

        # should not draw cwire for m1 if there is no usage of it
        layers = [
            m0.measurements,
            m1.measurements,
            m2.measurements,
            [qml.expval(m0), qml.expval(m2)],
        ]
        bit_map = {m0.measurements[0]: 0, m2.measurements[0]: 1}
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 3]], 1: [[2, 3]]}
        assert wires == {0: [[0]], 1: [[0]]}
        assert new_bit_map == bit_map

    def test_multi_mid_measure_cond_and_stats_layer(self):
        """Test cwire_connections works if layers contain multiple MCMs, multiple conditionals,
        and one or multiple terminal measurements using measurement values"""

        m0 = qml.measure(0)
        m1 = qml.measure(1)
        cond0 = qml.ops.Conditional(m0, qml.X(1))
        cond1 = qml.ops.Conditional(m1, qml.S(1))
        bit_map = {m0.measurements[0]: 0, m1.measurements[0]: 1}

        # final expval prevents wire reusing
        layers = [m0.measurements, [cond0], m1.measurements, [qml.X(0), cond1], [qml.expval(m0)]]
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 1, 4]], 1: [[2, 3]]}
        assert wires == {0: [[0, 1]], 1: [[1, 1]]}
        assert new_bit_map == bit_map

        # Nested measuring + cond already prevents wire reusing
        layers = [m0.measurements, m1.measurements, [cond0], [qml.X(0), cond1], [qml.expval(m1)]]
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 2]], 1: [[1, 3, 4]]}
        assert wires == {0: [[0, 1]], 1: [[1, 1]]}
        assert new_bit_map == bit_map

        # Wire can be reused
        layers = [m0.measurements, [cond0], m1.measurements, [qml.X(0), cond1], [qml.expval(m1)]]
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 1], [2, 3, 4]]}
        assert wires == {0: [[0, 1], [1, 1]]}
        assert new_bit_map == {key: 0 for key in bit_map}

        # Wire could be reused if it wasn't for the second expval
        layers = [
            m0.measurements,
            [cond0],
            m1.measurements,
            [qml.X(0), cond1],
            [qml.expval(m1), qml.expval(m0)],
        ]
        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)
        assert clayers == {0: [[0, 1, 4]], 1: [[2, 3, 4]]}
        assert wires == {0: [[0, 1]], 1: [[1, 1]]}
        assert new_bit_map == bit_map

    @pytest.mark.parametrize("rep", [2, 10])
    def test_reuse_cwire_many_times(self, rep):
        """Test that a measure + conditional executed multiple times only uses one cwire."""
        m = [qml.measure(0) for _ in range(rep)]
        conds = [qml.ops.Conditional(_m, qml.RX(i * 0.1, 2)) for i, _m in enumerate(m)]
        bit_map = {_m.measurements[0]: i for i, _m in enumerate(m)}
        layers = sum(([_m.measurements, [_c]] for _m, _c in zip(m, conds)), start=[])

        new_bit_map, clayers, wires = cwire_connections(layers, bit_map)

        assert clayers == {0: [[2 * i, 2 * i + 1] for i in range(rep)]}
        assert wires == {0: [[0, 2]] * rep}
        assert new_bit_map == {_m.measurements[0]: 0 for _m in m}
