# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the RewriteContext class and associated utilities."""

from uuid import UUID

import pytest
from xdsl.dialects import builtin, test
from xdsl.ir import SSAValue

from pennylane.compiler.python_compiler.dialects import quantum
from pennylane.compiler.python_compiler.rewriter import AbstractWire, SSAQubitMap

pytestmark = pytest.mark.external


def _create_ssa_values(result_types) -> list[SSAValue]:
    """Create SSAValues of the specified types."""
    test_op = test.TestOp(result_types=result_types)
    return list(test_op.results)


def _create_populated_ssa_qubit_map(wires=None) -> tuple[SSAQubitMap, quantum.QubitSSAValue]:
    """Create a populated SSAQubitMap instance for testing. If no wires are provided, a mix
    of static and abstract wires are used to initialize the map."""
    default_wires = (0, 1, 2, 3, 4, 5, AbstractWire(), AbstractWire())
    ssa_qubit_map = SSAQubitMap(wires=wires)

    # Populate map with qubits
    map_wires = wires or default_wires
    # We put more than one qubit per wire in the map since that is valid and needs to be tested.
    ssa_qubits = _create_ssa_values([quantum.QubitType()] * int(1.5 * len(map_wires)))
    for i, q in enumerate(ssa_qubits):
        ssa_qubit_map[map_wires[i % len(map_wires)]] = q

    return ssa_qubit_map, ssa_qubits


class TestAbstractWire:
    """Unit tests for the AbstractWire class."""

    def test_init(self):
        """Test that AbstractWires are initialized correctly."""
        wire1 = AbstractWire()
        assert isinstance(wire1.uuid, UUID)
        assert wire1.idx is None

        int_ssa = _create_ssa_values([builtin.i64])[0]
        qreg_ssa = _create_ssa_values([quantum.QuregType()])[0]
        wire2 = AbstractWire(idx=int_ssa, qreg=qreg_ssa)
        assert isinstance(wire2.uuid, UUID)
        assert wire2.idx == int_ssa
        assert wire2.qreg == qreg_ssa

    def test_equality(self):
        """Test that comparing two AbstractWires works correctly."""
        wire1 = AbstractWire()
        wire2 = AbstractWire()
        assert wire1 != wire2

        int_ssa = _create_ssa_values([builtin.i64])[0]
        qreg_ssa = _create_ssa_values([quantum.QuregType()])[0]
        wire3 = AbstractWire(idx=int_ssa, qreg=qreg_ssa)
        wire4 = AbstractWire(idx=int_ssa, qreg=qreg_ssa)
        assert wire3 == wire4

    def test_hash(self):
        """Test that the hash of different AbstractWires is unique."""
        wire1 = AbstractWire()
        wire2 = AbstractWire()
        assert hash(wire1) != hash(wire2)

        wire_dict = {}
        wire_dict[wire1] = "foo"
        wire_dict[wire2] = "bar"
        assert len(wire_dict) == 2

        int_ssa = _create_ssa_values([builtin.i64])[0]
        qreg_ssa = _create_ssa_values([quantum.QuregType()])[0]
        wire3 = AbstractWire(idx=int_ssa, qreg=qreg_ssa)
        wire4 = AbstractWire(idx=int_ssa, qreg=qreg_ssa)
        assert hash(wire3) == hash(wire4)

        wire_dict1 = {}
        wire_dict1[wire3] = "foo"
        wire_dict1[wire4] = "bar"
        assert len(wire_dict1) == 1
        assert wire_dict1[wire3] == wire_dict1[wire4] == "bar"


class TestSSAQubitMap:
    """Unit tests for the SSAQubitMap class."""

    def test_init(self):
        """Test that the SSAQubitMap's initialization is correct."""
        # pylint: disable=protected-access
        ssa_qubit_map = SSAQubitMap()
        assert ssa_qubit_map.wires is None
        assert ssa_qubit_map._map == {}

        wires = (1, 2, 3)
        ssa_qubit_map1 = SSAQubitMap(wires=wires)
        assert ssa_qubit_map1.wires == wires
        assert ssa_qubit_map1._map == {}

    def test_contains(self):
        """Test that the __contains__ dunder method works correctly."""
        wires = [0, 1, 2]
        ssa_qubit_map, ssa_qubits = _create_populated_ssa_qubit_map(wires=wires)

        for k in wires + ssa_qubits:
            assert k in ssa_qubit_map

        dummy_qubit = _create_ssa_values([quantum.QubitType()])[0]
        assert 4 not in ssa_qubit_map
        assert dummy_qubit not in ssa_qubit_map

    def test_getitem(self):
        """Test that the __getitem__ dunder method works correctly."""
        wires = [0, 1, 2, 3]
        ssa_qubit_map, ssa_qubits = _create_populated_ssa_qubit_map(wires=wires)

        for i, w in enumerate(wires):
            qubits = ssa_qubit_map[w]
            # We initialized the ssa_qubit_map to contain qubits that are
            # 1.5x the number of wires
            expected_qubits = [ssa_qubits[i]]
            if (next_idx := i + len(wires)) < len(ssa_qubits):
                expected_qubits.append(ssa_qubits[next_idx])
            assert qubits == expected_qubits

        for i, q in enumerate(ssa_qubits):
            _w = ssa_qubit_map[q]
            # We initialized the ssa_qubit_map to contain qubits that are
            # 1.5x the number of wires
            expected_w = wires[i % len(wires)]
            assert _w == expected_w

    def test_setitem(self):
        """Test that the __setitem__ dunder method works correctly."""
        wires = [0, 1, 2, 3]
        ssa_qubit_map = SSAQubitMap(wires=wires)
        qubits1 = _create_ssa_values([quantum.QubitType()] * len(wires))
        qubits2 = _create_ssa_values([quantum.QubitType()] * len(wires))

        for w, q in zip(wires, qubits1, strict=True):
            ssa_qubit_map[w] = q
            assert ssa_qubit_map[w] == [q]
            assert ssa_qubit_map[q] == w

        for w, q1, q2 in zip(wires, qubits1, qubits2, strict=True):
            ssa_qubit_map[q2] = w
            assert ssa_qubit_map[w] == [q1, q2]
            assert ssa_qubit_map[q1] == w

    def test_setitem_invalid_type(self):
        """Test that an error is raised if the input to __setitem__ is not a valid key
        or value type."""
        ssa_qubit_map = SSAQubitMap()
        invalid_type = "foo"

        with pytest.raises(
            AssertionError, match=f"{invalid_type} is not a valid key or value for an SSAQubitMap"
        ):
            ssa_qubit_map[invalid_type] = 0

        with pytest.raises(
            AssertionError, match=f"{invalid_type} is not a valid key or value for an SSAQubitMap"
        ):
            ssa_qubit_map[0] = invalid_type

    def test_setitem_invalid_wire_label(self):
        """Test that an error is raised by __setitem__ if the key or value that is
        a wire label is not valid."""
        wires = (0,)
        ssa_qubit_map = SSAQubitMap(wires=wires)
        invalid_wire = 1
        qubit = _create_ssa_values([quantum.QubitType()])[0]

        with pytest.raises(AssertionError, match=f"{invalid_wire} is not a valid wire label"):
            ssa_qubit_map[qubit] = invalid_wire

        with pytest.raises(AssertionError, match=f"{invalid_wire} is not a valid wire label"):
            ssa_qubit_map[invalid_wire] = qubit

    def test_setitem_same_qubit_error(self):
        """Test that an error is raised by __setitem__ if the key or value is a qubit that is
        already in the map."""
        ssa_qubit_map = SSAQubitMap()
        q1 = _create_ssa_values([quantum.QubitType()])[0]
        ssa_qubit_map[0] = q1

        with pytest.raises(KeyError, match="Cannot update qubits that are already in the map."):
            ssa_qubit_map[q1] = 1

        with pytest.raises(ValueError, match="Cannot update qubits that are already in the map."):
            ssa_qubit_map[1] = q1

    def test_get(self):
        """Test that the get method works correctly."""
        wires = (0, 1, 2)
        ssa_qubit_map, ssa_qubits = _create_populated_ssa_qubit_map(wires=wires)

        for i, w in enumerate(wires):
            # We initialized the ssa_qubit_map to contain qubits that are
            # 1.5x the number of wires
            expected_qubits = [ssa_qubits[i]]
            if (next_idx := i + len(wires)) < len(ssa_qubits):
                expected_qubits.append(ssa_qubits[next_idx])
            assert ssa_qubit_map.get(w) == expected_qubits

        for i, q in enumerate(ssa_qubits):
            # We initialized the ssa_qubit_map to contain qubits that are
            # 1.5x the number of wires
            expected_w = wires[i % len(wires)]
            assert ssa_qubit_map.get(q) == expected_w

        assert ssa_qubit_map.get(-1) is None
        assert ssa_qubit_map.get(-1, "foo") == "foo"

    def test_pop_from_wires(self):
        """Test that the pop method works correctly when extracting qubits using wire labels."""
        wires = (0, 1, 2)
        ssa_qubit_map, ssa_qubits = _create_populated_ssa_qubit_map(wires=wires)

        for i, w in enumerate(wires):
            # We initialized the ssa_qubit_map to contain qubits that are
            # 1.5x the number of wires
            expected_qubits = [ssa_qubits[i]]
            if (next_idx := i + len(wires)) < len(ssa_qubits):
                expected_qubits.append(ssa_qubits[next_idx])
            assert ssa_qubit_map.pop(w) == expected_qubits

            assert w not in ssa_qubit_map
            for q in expected_qubits:
                assert q not in ssa_qubit_map

        assert ssa_qubit_map.pop(-1) is None
        assert ssa_qubit_map.pop(-1, "foo") == "foo"

    def test_pop_from_qubits(self):
        """Test that the pop method works correctly when extracting wire labels from qubits."""
        wires = (0, 1, 2, 3)
        ssa_qubit_map = SSAQubitMap(wires=wires)
        ssa_qubits = _create_ssa_values([quantum.QubitType()] * 2 * len(wires))

        # Make all wires map to two qubits
        for i, w in enumerate(wires):
            ssa_qubit_map[w] = ssa_qubits[i]
            ssa_qubit_map[w] = ssa_qubits[i + len(wires)]

        for i, w in enumerate(wires):
            cur_qubits = [ssa_qubits[i], ssa_qubits[i + len(wires)]]

            # ``w`` maps to two qubits. We pop one of them, so that qubit should
            # no longer be in the map. But, ``w`` still maps to one other qubit,
            # so it should still be in the map.
            w1 = ssa_qubit_map.pop(cur_qubits[0])
            assert w1 == w
            assert w in ssa_qubit_map
            assert cur_qubits[0] not in ssa_qubit_map
            assert cur_qubits[0] not in ssa_qubit_map[w]

            # ``w`` now maps to one qubit. We pop it, so that qubit should no longer
            # be in the map. Since ``w`` no longer maps to any qubits, it should also
            # no longer be in the map.
            w2 = ssa_qubit_map.pop(cur_qubits[1])
            assert w2 == w
            assert cur_qubits[1] not in ssa_qubit_map
            assert w not in ssa_qubit_map

        missing_qubit = _create_ssa_values([quantum.QubitType()])[0]
        assert ssa_qubit_map.pop(missing_qubit) is None
        assert ssa_qubit_map.pop(missing_qubit, "foo") == "foo"

    def test_assert_valid_type(self):
        """Test that the _assert_valid_type method works correctly."""
        # pylint: disable=protected-access
        valid_types = (4, AbstractWire(), _create_ssa_values([quantum.QubitType()])[0])
        ssa_qubit_map = SSAQubitMap()

        for v in valid_types:
            ssa_qubit_map._assert_valid_type(v)

        invalid_type = "foo"
        with pytest.raises(AssertionError, match=f"{invalid_type} is not a valid key or value"):
            ssa_qubit_map._assert_valid_type(invalid_type)

    def test_assert_valid_wire_label(self):
        """Test that the _assert_valid_wire_label method works correctly."""
        # pylint: disable=protected-access
        wires = (0, 1, 2)
        ssa_qubit_map = SSAQubitMap(wires=wires)

        valid_wires = wires + (AbstractWire(),)
        for v in valid_wires:
            ssa_qubit_map._assert_valid_wire_label(v)

        invalid_wire = 10
        with pytest.raises(AssertionError, match=f"{invalid_wire} is not a valid wire label"):
            ssa_qubit_map._assert_valid_wire_label(invalid_wire)

    def test_verify(self):
        """Test that a valid SSAQubitMap's verify method does not raise an error."""
        wires = (0, 1, 2, 3, 4, 5)
        ssa_qubit_map, _ = _create_populated_ssa_qubit_map(wires=wires)
        ssa_qubit_map.verify()

    def test_verify_invalid_value_for_wire_key(self):
        """Test that a valid SSAQubitMap's verify method raises an error if the wire keys
        map to values that are invalid."""
        # pylint: disable=protected-access
        ssa_qubit_map = SSAQubitMap()
        ssa_qubit_map._map[0] = "foo"

        with pytest.raises(AssertionError, match=f"The key {0} maps to an invalid type {'foo'}."):
            ssa_qubit_map.verify()

    def test_verify_different_wire_keys_values_errors(self):
        """Test that a valid SSAQubitMap's verify method raises an error if the wire keys
        do not match the wire values."""
        # pylint: disable=protected-access
        ssa_qubit_map = SSAQubitMap()
        q0, q1 = _create_ssa_values([quantum.QubitType()] * 2)
        ssa_qubit_map[0] = q0
        # Wire 1 maps to q1, but q1 does not map to wire 1
        ssa_qubit_map._map[1] = [q1]

        with pytest.raises(
            AssertionError, match="The wire label keys do not match the wire label values."
        ):
            ssa_qubit_map.verify()

    def test_verify_different_qubit_keys_values_errors(self):
        """Test that a valid SSAQubitMap's verify method raises an error if the qubit keys do
        not match the qubit values."""
        # pylint: disable=protected-access
        ssa_qubit_map = SSAQubitMap()
        q0, q1 = _create_ssa_values([quantum.QubitType()] * 2)
        ssa_qubit_map[0] = q0
        # q1 maps to wire 0, but wire 0 does not map to q1
        ssa_qubit_map._map[q1] = 0

        with pytest.raises(AssertionError, match="The qubit keys do not match the qubit values."):
            ssa_qubit_map.verify()

    def test_verify_multiple_wires_for_same_qubit_errors(self):
        """Test that a valid SSAQubitMap's verify method raises an error if multiple wires
        map to the same qubit."""
        # pylint: disable=protected-access
        ssa_qubit_map = SSAQubitMap()
        q0, q1 = _create_ssa_values([quantum.QubitType()] * 2)
        # Both wire 0 and 1 map to q1, which is invalid
        ssa_qubit_map._map[0] = [q0, q1]
        ssa_qubit_map._map[1] = [q1]
        ssa_qubit_map._map[q0] = 0
        ssa_qubit_map._map[q1] = 1

        with pytest.raises(
            AssertionError, match="Multiple wires are being mapped to the same qubit."
        ):
            ssa_qubit_map.verify()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
