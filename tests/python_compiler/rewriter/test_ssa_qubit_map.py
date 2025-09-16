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
from pennylane.compiler.python_compiler.rewriter import AbstractWire, SSAQubitMap

pytestmark = pytest.mark.external


def _create_ssa_values(result_types) -> list[SSAValue]:
    """Create SSAValues of the specified types."""
    test_op = test.TestOp(result_types=result_types)
    return list(test_op.results)


def _create_populated_ssa_qubit_map(wires=None) -> SSAQubitMap:
    """Create a populated SSAQubitMap instance for testing. If no wires are provided, a mix
    of static and abstract wires are used to initialize the map."""
    default_wires = (0, 1, 2, 3, 4, 5, AbstractWire(), AbstractWire())
    ssa_qubit_map = SSAQubitMap(wires=wires)

    # Populate map with qubits
    map_wires = wires or default_wires
    ssa_qubits = _create_ssa_values([quantum.QubitType() for _ in range(len(map_wires))])
    for w, q in zip(map_wires, ssa_qubits, strict=True):
        ssa_qubit_map[w] = q

    return ssa_qubit_map


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
        # pylint: disable=protected-access
        wires = (0, 1, 2)
        ssa_qubit_map = _create_populated_ssa_qubit_map(wires=wires)

        for k in ssa_qubit_map._map:
            assert k in ssa_qubit_map

        dummy_qubit = _create_ssa_values([quantum.QubitType()])[0]
        assert 4 not in ssa_qubit_map
        assert dummy_qubit not in ssa_qubit_map

    # def test_getitem(self):
    #     """Test that the __getitem__ dunder method works correctly."""
    #     # pylint: disable=protected-access
    #     wires = (0, 1, 2)
    #     ssa_qubit_map = _create_populated_ssa_qubit_map(wires=wires)
    #     qubits = [q for q in ssa_qubit_map._map if isinstance(q, SSAValue)]

    #     for w, q in zip(wires, qubits, strict=True):
    #         assert ssa_qubit_map[w] == q
    #         assert ssa_qubit_map[q] == w

    # def test_setitem(self):
    #     """Test that the __setitem__ dunder method works correctly."""
    #     # pylint: disable=protected-access
    #     wires = (0, 1, 2, 3)
    #     ssa_qubit_map = _create_populated_ssa_qubit_map(wires=wires)
    #     qubits = list(ssa_qubit_map._qubit_to_wire_map)
    #     new_qubits = _create_ssa_values([quantum.QubitType() for _ in range(len(wires))])

    #     for w, q, nq in zip(wires, qubits, new_qubits, strict=True):
    #         ssa_qubit_map[w] = nq
    #         assert ssa_qubit_map[w] == nq
    #         assert ssa_qubit_map[nq] == w
    #         assert q not in ssa_qubit_map

    #     new_qubits1 = _create_ssa_values([quantum.QubitType() for _ in range(len(wires))])
    #     abstract_wires = [AbstractWire() for _ in range(len(new_qubits1))]
    #     for nq, w in zip(new_qubits1, abstract_wires, strict=True):
    #         ssa_qubit_map[nq] = w
    #         assert ssa_qubit_map[w] == nq
    #         assert ssa_qubit_map[nq] == w

    # def test_setitem_invalid_ssa_value(self):
    #     """Test that an error is raised if the input to __setitem__ is an SSAValue
    #     but its type is not QubitType."""
    #     ssa_qubit_map = SSAQubitMap()
    #     invalid_ssa_value = _create_ssa_values([builtin.i64])[0]

    #     with pytest.raises(KeyError, match="Expected key to be a QubitType SSAValue"):
    #         ssa_qubit_map[invalid_ssa_value] = 0

    # def test_setitem_ssa_key_with_invalid_val(self):
    #     """Test that an error is raised by __setitem__ if the key is a valid SSAValue but
    #     the value is not a valid wire label."""
    #     wires = (0,)
    #     ssa_qubit_map = _create_populated_ssa_qubit_map(wires=wires)
    #     qubit = _create_ssa_values([quantum.QubitType()])[0]
    #     invalid_wire = 1

    #     with pytest.raises(ValueError, match=f"{invalid_wire} is not an available wire"):
    #         ssa_qubit_map[qubit] = invalid_wire

    # def test_setitem_wire_key_not_in_wires(self):
    #     """Test that an error is raised by __setitem__ if the key is a wire label that is
    #     not in the wires tuple."""
    #     wires = (0,)
    #     ssa_qubit_map = _create_populated_ssa_qubit_map(wires=wires)
    #     qubit = _create_ssa_values([quantum.QubitType()])[0]
    #     invalid_wire = 1

    #     with pytest.raises(KeyError, match=f"{invalid_wire} is not an available wire"):
    #         ssa_qubit_map[1] = qubit

    # def test_setitem_wire_key_invalid_val(self):
    #     """Test that an error is raised by __setitem__ if the key is a valid wire label
    #     but the value is not a qubit SSAValue."""
    #     ssa_qubit_map = SSAQubitMap()
    #     invalid_ssa_value = _create_ssa_values([builtin.i64])[0]

    #     with pytest.raises(ValueError, match="Expected value to be a QubitType SSAValue"):
    #         ssa_qubit_map[0] = invalid_ssa_value

    # def test_setitem_invalid_key_type(self):
    #     """Test that an error is raised by __setitem__ if the key is not a qubit or a wire label."""
    #     ssa_qubit_map = SSAQubitMap()
    #     invalid_key = "foo"

    #     with pytest.raises(
    #         KeyError, match=f"{invalid_key} is not a valid wire label or QubitType SSAValue"
    #     ):
    #         ssa_qubit_map[invalid_key] = 1

    # def test_get(self):
    #     """Test that the get method works correctly."""
    #     # pylint: disable=protected-access
    #     wires = (0, 1, 2)
    #     ssa_qubit_map = _create_populated_ssa_qubit_map(wires=wires)
    #     qubits = list(ssa_qubit_map._qubit_to_wire_map.keys())

    #     for w, q in zip(wires, qubits, strict=True):
    #         assert ssa_qubit_map.get(w) == q
    #         assert ssa_qubit_map.get(q) == w

    #     assert ssa_qubit_map.get(-1) is None
    #     assert ssa_qubit_map.get(-1, "foo") == "foo"

    # def test_pop(self):
    #     """Test that the pop method works correctly."""
    #     # pylint: disable=protected-access
    #     wires = (0, 1, 2, 3)
    #     ssa_qubit_map = _create_populated_ssa_qubit_map(wires=wires)
    #     qubits = list(ssa_qubit_map._qubit_to_wire_map.keys())

    #     for w, q in zip(wires[:2], qubits[:2], strict=True):
    #         assert ssa_qubit_map.pop(w) == q
    #         assert w not in ssa_qubit_map
    #         assert q not in ssa_qubit_map

    #     for w, q in zip(wires[2:], qubits[2:], strict=True):
    #         assert ssa_qubit_map.pop(q) == w
    #         assert w not in ssa_qubit_map
    #         assert q not in ssa_qubit_map

    #     assert ssa_qubit_map.pop(-1) is None
    #     assert ssa_qubit_map.pop(-1, "foo") == "foo"

    def test_assert_valid_type(self):
        """Test that the _assert_valid_type method works correctly."""

    def test_assert_valid_wire_label(self):
        """Test that the _assert_valid_wire_label method works correctly."""

    def test_verify(self):
        """Test that valid SSAQubitMaps' verify method does not raise an error."""

    def test_verify_errors(self):
        """Test that valid SSAQubitMaps' verify method does not raise an error."""
