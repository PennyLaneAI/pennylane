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
from xdsl.dialects import arith, builtin
from xdsl.dialects import stablehlo as xstablehlo
from xdsl.dialects import tensor, test
from xdsl.ir import SSAValue

from pennylane.compiler.python_compiler.dialects import quantum
from pennylane.compiler.python_compiler.rewriter import AbstractWire, RewriteContext, WireQubitMap

pytestmark = pytest.mark.external


def _create_ssa_values(result_types) -> list[SSAValue]:
    """Create SSAValues of the specified types."""
    test_op = test.TestOp(result_types=result_types)
    return list(test_op.results)


def _create_populated_wire_qubit_map(wires=None) -> WireQubitMap:
    """Create a populated WireQubitMap instance for testing. If no wires are provided, a mix
    of static and abstract wires are used to initialize the map."""
    default_wires = (0, 1, 2, 3, 4, 5, AbstractWire(), AbstractWire())
    wqmap = WireQubitMap(wires=wires)

    # Populate map with qubits
    map_wires = wires or default_wires
    ssa_qubits = _create_ssa_values([quantum.QubitType() for _ in range(len(map_wires))])
    for w, q in zip(map_wires, ssa_qubits, strict=True):
        wqmap[w] = q

    return wqmap


def _create_arith_constant_int(value) -> SSAValue:
    """Create a constant SSAValue using arith.constant."""
    op = arith.ConstantOp(
        value=builtin.IntegerAttr(value=value, value_type=builtin.IntegerType(64))
    )
    return op.results[0]


def _create_stablehlo_constant_int(value) -> SSAValue:
    """Create a constant SSAValue using stablehlo.constant and tensor.extract."""
    dense_repr = builtin.DenseIntOrFPElementsAttr(
        type=builtin.TensorType(builtin.IntegerType(64), shape=()), data=[value]
    )
    cst_op = xstablehlo.ConstantOp(dense_repr)
    extract_op = tensor.ExtractOp(tensor=cst_op.results[0], indices=())
    return extract_op.results[0]


class TestAbstractWire:
    """Unit tests for the AbstractWire class."""

    def test_init(self):
        """Test that AbstractWires are initialized correctly."""
        wire1 = AbstractWire()
        assert isinstance(wire1.id, UUID)
        assert wire1.idx is None

        int_ssa = _create_ssa_values([builtin.i64])[0]
        wire2 = AbstractWire(idx=int_ssa)
        assert isinstance(wire2.id, UUID)
        assert wire2.idx == int_ssa

    def test_equality(self):
        """Test that comparing two AbstractWires works correctly."""
        wire1 = AbstractWire()
        wire2 = AbstractWire()
        assert wire1 != wire2

        int_ssa = _create_ssa_values([builtin.i64])[0]
        wire3 = AbstractWire(idx=int_ssa)
        wire4 = AbstractWire(idx=int_ssa)
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
        wire3 = AbstractWire(idx=int_ssa)
        wire4 = AbstractWire(idx=int_ssa)
        assert hash(wire3) == hash(wire4)

        wire_dict1 = {}
        wire_dict1[wire3] = "foo"
        wire_dict1[wire4] = "bar"
        assert len(wire_dict1) == 1
        assert wire_dict1[wire3] == wire_dict1[wire4] == "bar"


class TestWireQubitMap:
    """Unit tests for the WireQubitMap class."""

    def test_init(self):
        """Test that the WireQubitMap's initialization is correct."""
        # pylint: disable=protected-access
        wqmap = WireQubitMap()
        assert wqmap.wires is None
        assert wqmap._wire_to_qubit_map == {}
        assert wqmap._qubit_to_wire_map == {}

        wires = (1, 2, 3)
        wqmap1 = WireQubitMap(wires=wires)
        assert wqmap1.wires == wires
        assert wqmap1._wire_to_qubit_map == {}
        assert wqmap1._qubit_to_wire_map == {}

    def test_contains(self):
        """Test that the __contains__ dunder method works correctly."""
        # pylint: disable=protected-access
        wires = (0, 1, 2)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        qubits = tuple(wqmap._qubit_to_wire_map.keys())

        for key in wires + qubits:
            assert key in wqmap

        dummy_qubit = _create_ssa_values([quantum.QubitType()])[0]
        assert 4 not in wqmap
        assert dummy_qubit not in wqmap

    def test_len(self):
        """Test that the __len__ dunder method works correctly."""
        wires = (0, 1, 2)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        assert len(wqmap) == len(wires)

    def test_len_invalid_length(self):
        """Test that __len__ raises an error if the lengths of the dictionaries do not match."""
        # pylint: disable=protected-access
        wqmap = WireQubitMap()
        wqmap._wire_to_qubit_map[0] = _create_ssa_values([quantum.QubitType()])[0]

        with pytest.raises(
            AssertionError, match="The lengths of the wire and qubit maps do not match"
        ):
            _ = len(wqmap)

    def test_getitem(self):
        """Test that the __getitem__ dunder method works correctly."""
        # pylint: disable=protected-access
        wires = (0, 1, 2)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        qubits = list(wqmap._qubit_to_wire_map)

        for w, q in zip(wires, qubits, strict=True):
            assert wqmap[w] == q
            assert wqmap[q] == w

    def test_setitem(self):
        """Test that the __setitem__ dunder method works correctly."""
        # pylint: disable=protected-access
        wires = (0, 1, 2, 3)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        qubits = list(wqmap._qubit_to_wire_map)
        new_qubits = _create_ssa_values([quantum.QubitType() for _ in range(len(wires))])

        for w, q, nq in zip(wires, qubits, new_qubits, strict=True):
            wqmap[w] = nq
            assert wqmap[w] == nq
            assert wqmap[nq] == w
            assert q not in wqmap

        new_qubits1 = _create_ssa_values([quantum.QubitType() for _ in range(len(wires))])
        abstract_wires = [AbstractWire() for _ in range(len(new_qubits1))]
        for nq, w in zip(new_qubits1, abstract_wires, strict=True):
            wqmap[nq] = w
            assert wqmap[w] == nq
            assert wqmap[nq] == w

    def test_setitem_invalid_ssa_value(self):
        """Test that an error is raised if the input to __setitem__ is an SSAValue
        but its type is not QubitType."""
        wqmap = WireQubitMap()
        invalid_ssa_value = _create_ssa_values([builtin.i64])[0]

        with pytest.raises(KeyError, match="Expected key to be a QubitType SSAValue"):
            wqmap[invalid_ssa_value] = 0

    def test_setitem_ssa_key_with_invalid_val(self):
        """Test that an error is raised by __setitem__ if the key is a valid SSAValue but
        the value is not a valid wire label."""
        wires = (0,)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        qubit = _create_ssa_values([quantum.QubitType()])[0]
        invalid_wire = 1

        with pytest.raises(ValueError, match=f"{invalid_wire} is not an available wire"):
            wqmap[qubit] = invalid_wire

    def test_setitem_wire_key_not_in_wires(self):
        """Test that an error is raised by __setitem__ if the key is a wire label that is
        not in the wires tuple."""
        wires = (0,)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        qubit = _create_ssa_values([quantum.QubitType()])[0]
        invalid_wire = 1

        with pytest.raises(KeyError, match=f"{invalid_wire} is not an available wire"):
            wqmap[1] = qubit

    def test_setitem_wire_key_invalid_val(self):
        """Test that an error is raised by __setitem__ if the key is a valid wire label
        but the value is not a qubit SSAValue."""
        wqmap = WireQubitMap()
        invalid_ssa_value = _create_ssa_values([builtin.i64])[0]

        with pytest.raises(ValueError, match="Expected value to be a QubitType SSAValue"):
            wqmap[0] = invalid_ssa_value

    def test_setitem_invalid_key_type(self):
        """Test that an error is raised by __setitem__ if the key is not a qubit or a wire label."""
        wqmap = WireQubitMap()
        invalid_key = "foo"

        with pytest.raises(
            KeyError, match=f"{invalid_key} is not a valid wire label or QubitType SSAValue"
        ):
            wqmap[invalid_key] = 1

    def test_get(self):
        """Test that the get method works correctly."""
        # pylint: disable=protected-access
        wires = (0, 1, 2)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        qubits = list(wqmap._qubit_to_wire_map.keys())

        for w, q in zip(wires, qubits, strict=True):
            assert wqmap.get(w) == q
            assert wqmap.get(q) == w

        assert wqmap.get(-1) is None
        assert wqmap.get(-1, "foo") == "foo"

    def test_pop(self):
        """Test that the pop method works correctly."""
        # pylint: disable=protected-access
        wires = (0, 1, 2, 3)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        qubits = list(wqmap._qubit_to_wire_map.keys())

        for w, q in zip(wires[:2], qubits[:2], strict=True):
            assert wqmap.pop(w) == q
            assert w not in wqmap
            assert q not in wqmap

        for w, q in zip(wires[2:], qubits[2:], strict=True):
            assert wqmap.pop(q) == w
            assert w not in wqmap
            assert q not in wqmap

        assert wqmap.pop(-1) is None
        assert wqmap.pop(-1, "foo") == "foo"

    def test_update_qubit(self):
        """Test that the update_qubit method works correctly."""
        wires = (0, 1, 2)
        wqmap = _create_populated_wire_qubit_map(wires=wires)

        old_qubit = wqmap[0]
        new_qubit = _create_ssa_values([quantum.QubitType()])[0]
        wqmap.update_qubit(old_qubit, new_qubit)

        assert old_qubit not in wqmap
        assert wqmap[0] == new_qubit
        assert wqmap[new_qubit] == 0

    def test_update_qubit_invalid(self):
        """Test that an error is raised by update_qubit if the old qubit is not in the map."""
        wqmap = WireQubitMap()
        old_qubit = _create_ssa_values([quantum.QubitType()])[0]
        new_qubit = _create_ssa_values([quantum.QubitType()])[0]

        with pytest.raises(KeyError, match="is not in the WireQubitMap"):
            wqmap.update_qubit(old_qubit, new_qubit)


in_qubits: list[quantum.QubitSSAValue] = _create_ssa_values(
    [quantum.QubitType() for _ in range(10)]
)
out_qubits: list[quantum.QubitSSAValue] = _create_ssa_values(
    [quantum.QubitType() for _ in range(10)]
)
test_qreg: quantum.QuregSSAValue = _create_ssa_values([quantum.QuregType()])[0]


@pytest.fixture(scope="function")
def wire_qubit_map():
    wires = list(range(len(in_qubits)))
    wqmap = WireQubitMap(wires=wires)
    for w, q in zip(wires, in_qubits):
        wqmap[w] = q

    yield wqmap


# pylint: disable=redefined-outer-name,too-many-arguments
class TestRewriteContext:
    """Unit tests for the RewriteContext class."""

    def test_init(self):
        """Test that the initialization of RewriteContext is correct."""
        context = RewriteContext()
        assert context.shots is None
        assert context.nqubits is None
        assert context.qreg is None
        assert context.wire_qubit_map == WireQubitMap()

        wqmap = _create_populated_wire_qubit_map(wires=(0, 1))
        qreg = _create_ssa_values([quantum.QuregType()])[0]
        context = RewriteContext(nqubits=4, shots=10, qreg=qreg, wire_qubit_map=wqmap)
        assert context.shots == 10
        assert context.nqubits == 4
        assert context.qreg == qreg
        assert context.wire_qubit_map == wqmap

    @pytest.mark.parametrize(
        "op, updated_keys, updated_values, updated_wires_and_qubits",
        [
            # DeviceInitOp
            (),
            (),
            (),
            # ExtractOp
            (),
            (),
            (),
            (),
            # InsertOp
            (),
            # AllocOp
            (),
            (),
            (),
            # AllocQubitOp
            (),
            # DeallocOp
            (),
            # DeallocQubitOp
            (),
            # NumQubitsOp
            (),
            # MeasureOp
            (),
            # MeasureInBasisOp
            (),
        ],
    )
    def test_update_from_op_dispatches(
        self, op, updated_keys, updated_values, updated_wires_and_qubits, wire_qubit_map
    ):
        """Test that the dispatches for RewriteContext.update_from_op update
        the context as expected."""

    def test_update_from_op_default(self):
        """Test that the default RewriteContext.update_from_op updates the
        context as expected."""
        # pylint: disable=protected-access
        wires = (0, 1)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        context = RewriteContext(wire_qubit_map=wqmap)
        old_qubits = [wqmap[0], wqmap[1]]

        op = test.TestOp(
            operands=old_qubits,
            result_types=[quantum.QuregType(), quantum.QubitType(), quantum.QubitType()],
        )
        qreg, *new_qubits = op.results
        context.update_from_op(op)

        assert context.wire_qubit_map[0] == new_qubits[0]
        assert context.wire_qubit_map[1] == new_qubits[1]
        assert context.qreg == qreg
        assert context.nqubits is None
        assert context.shots is None

    def test_update_from_op_default_no_qubit_outs(self):
        """Test that the default RewriteContext.update_from_op updates the
        context as expected when the op has no qubit outputs."""
        # pylint: disable=protected-access
        wires = (0, 1)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        context = RewriteContext(wire_qubit_map=wqmap)
        old_qubits = [wqmap[0], wqmap[1]]

        op = test.TestOp(
            operands=old_qubits,
        )
        context.update_from_op(op)
        assert context.wire_qubit_map[0] == old_qubits[0]
        assert context.wire_qubit_map[1] == old_qubits[1]
        assert context.qreg is None
        assert context.nqubits is None
        assert context.shots is None

    def test_update_from_op_default_different_qubit_in_out_lengths(self):
        """Test that the default RewriteContext.update_from_op updates the
        context as expected when the op has qubit outputs but the number of
        input and output qubits is different."""
        # pylint: disable=protected-access
        wires = (0, 1)
        wqmap = _create_populated_wire_qubit_map(wires=wires)
        context = RewriteContext(wire_qubit_map=wqmap)
        old_qubits = [wqmap[0], wqmap[1]]

        op = test.TestOp(operands=old_qubits, result_types=(quantum.QubitType(),))
        context.update_from_op(op)
        assert context.wire_qubit_map[0] == old_qubits[0]
        assert context.wire_qubit_map[1] == old_qubits[1]
        assert context.qreg is None
        assert context.nqubits is None
        assert context.shots is None

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("region_first", [False, True])
    @pytest.mark.parametrize("update", [False, True])
    def test_walk(self, reverse, region_first, update):
        """Test that RewriteContext.walk works as expected."""


if __name__ == "__main__":
    pytest.main(["-x", __file__])
