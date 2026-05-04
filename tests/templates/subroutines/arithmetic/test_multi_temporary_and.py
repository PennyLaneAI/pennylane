# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Tests for the ``MultiTemporaryAND`` operation.
"""

# pylint: disable = protected-access

import pytest

import pennylane as qp
from pennylane.ops.functions.assert_valid import _check_decomposition_new, assert_valid


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_valid_decomp(n):
    """Test that the registered decomposition rule for ``MultiTemporaryAND`` is
    consistent with the operator and yields the correct resources."""

    registers = qp.registers({"target_wire": 1, "control_wires": n, "work_wires": n - 2})

    op = qp.MultiTemporaryAND(
        **registers,
        control_values=None,
        work_wire_type="zeroed",
    )
    _check_decomposition_new(op)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_assert_valid(n):
    """Test that ``MultiTemporaryAND`` satisfies the standard operator validation
    checks (``_flatten``/``_unflatten``, pickling, wire mapping, decomposition, etc.)."""
    registers = qp.registers({"target_wire": 1, "control_wires": n})

    op = qp.MultiTemporaryAND(
        **registers,
        control_values=None,
    )
    assert_valid(op)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_assert_valid_with_work_wires(n):
    """Same as ``test_assert_valid`` but with some zeroed work wires provided"""
    registers = qp.registers({"target_wire": 1, "control_wires": n, "work_wires": n - 2})

    op = qp.MultiTemporaryAND(
        **registers,
        control_values=None,
        work_wire_type="zeroed",
    )
    assert_valid(op)


class TestConstruction:
    """Test construction, validation, and accessors of ``MultiTemporaryAND``."""

    def test_basic_construction(self):
        op = qp.MultiTemporaryAND(control_wires=[0, 1, 2], target_wire=3)
        assert op.control_wires == qp.wires.Wires([0, 1, 2])
        assert op.target_wire == qp.wires.Wires([3])
        assert op.target_wires == qp.wires.Wires([3])
        assert op.control_values == (True, True, True)
        assert op.work_wires == qp.wires.Wires([])
        assert op.work_wire_type == "borrowed"
        assert op.wires == qp.wires.Wires([0, 1, 2, 3])

    def test_construction_with_work_wires(self):
        op = qp.MultiTemporaryAND(
            control_wires=[0, 1, 2],
            target_wire=3,
            work_wires=[4, 5],
            work_wire_type="zeroed",
        )
        assert op.work_wires == qp.wires.Wires([4, 5])
        assert op.work_wire_type == "zeroed"
        assert op.wires == qp.wires.Wires([0, 1, 2, 3, 4, 5])

    def test_control_values_string(self):
        op = qp.MultiTemporaryAND(control_wires=[0, 1, 2], target_wire=3, control_values="101")
        assert op.control_values == (True, False, True)

    def test_control_values_list_of_ints(self):
        op = qp.MultiTemporaryAND(control_wires=[0, 1, 2], target_wire=3, control_values=[1, 0, 0])
        assert op.control_values == (True, False, False)

    def test_empty_control_wires_raises(self):
        with pytest.raises(ValueError, match="at least one control wire"):
            qp.MultiTemporaryAND(control_wires=[], target_wire=0)

    def test_multi_target_wire_raises(self):
        with pytest.raises(ValueError, match="exactly one target wire"):
            qp.MultiTemporaryAND(control_wires=[0, 1], target_wire=[2, 3])

    def test_target_overlaps_controls_raises(self):
        with pytest.raises(ValueError, match="Target wire must be different"):
            qp.MultiTemporaryAND(control_wires=[0, 1], target_wire=0)

    def test_work_overlaps_controls_raises(self):
        with pytest.raises(ValueError, match="Work wires must be different"):
            qp.MultiTemporaryAND(control_wires=[0, 1], target_wire=2, work_wires=[0])

    def test_bad_work_wire_type_raises(self):
        with pytest.raises(ValueError, match="work_wire_type must be either"):
            qp.MultiTemporaryAND(control_wires=[0, 1], target_wire=2, work_wire_type="spam")

    def test_bad_control_values_raises(self):
        with pytest.raises(ValueError, match="Length of control values"):
            qp.MultiTemporaryAND(control_wires=[0, 1, 2], target_wire=3, control_values="11")

    def test_repr(self):
        op = qp.MultiTemporaryAND(control_wires=[0, 1], target_wire=2)
        assert repr(op) == "MultiTemporaryAND(control_wires=[0, 1], target_wire=[2])"

        op2 = qp.MultiTemporaryAND(control_wires=[0, 1], target_wire=2, control_values=[0, 1])
        assert (
            repr(op2)
            == "MultiTemporaryAND(control_wires=[0, 1], target_wire=[2], control_values=[0, 1])"
        )


class TestSerialization:
    """Tests that the operator round-trips through ``_flatten``/``_unflatten``, pickling, and map_wires."""

    def test_flatten_unflatten_roundtrip(self):
        op = qp.MultiTemporaryAND(
            control_wires=[0, 1, 2],
            target_wire=3,
            control_values=[1, 0, 1],
            work_wires=[4, 5],
            work_wire_type="zeroed",
        )
        data, metadata = op._flatten()
        clone = qp.MultiTemporaryAND._unflatten(data, metadata)
        qp.assert_equal(op, clone)

    def test_map_wires(self):
        op = qp.MultiTemporaryAND(
            control_wires=[0, 1],
            target_wire=2,
            work_wires=[3],
            work_wire_type="zeroed",
        )
        mapped = op.map_wires({0: "a", 1: "b", 2: "c", 3: "d"})
        assert mapped.control_wires == qp.wires.Wires(["a", "b"])
        assert mapped.target_wire == qp.wires.Wires(["c"])
        assert mapped.work_wires == qp.wires.Wires(["d"])
        assert mapped.work_wire_type == "zeroed"
