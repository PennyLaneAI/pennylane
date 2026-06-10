# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Unit tests for the extended ``registers`` class.
"""

# pylint: disable=use-implicit-booleaness-not-comparison

import pytest

from pennylane.labs.registers import registers
from pennylane.wires import Wires


class TestConstruction:
    """Construction parity with ``qml.registers`` plus the Wires-mapping path."""

    def test_flat_spec(self):
        """A flat int spec allocates contiguous, non-overlapping wire ranges."""
        reg = registers({"a": 2, "b": 2})
        assert isinstance(reg, dict)
        assert reg["a"].tolist() == [0, 1]
        assert reg["b"].tolist() == [2, 3]

    def test_nested_spec(self):
        """A nested spec flattens children while the parent spans their wires."""
        reg = registers({"x": {"a": 2, "b": 2}, "c": 3})
        assert reg["a"].tolist() == [0, 1]
        assert reg["b"].tolist() == [2, 3]
        assert reg["x"].tolist() == [0, 1, 2, 3]
        assert reg["c"].tolist() == [4, 5, 6]

    def test_wires_mapping_passthrough(self):
        """An already-built name->Wires mapping is stored verbatim."""
        reg = registers({"a": Wires([0, 1]), "b": Wires([2, 3])})
        assert reg["a"].tolist() == [0, 1]
        assert reg["b"].tolist() == [2, 3]

    def test_none_is_empty(self):
        """Constructing with ``None`` yields an empty register set."""
        assert registers(None) == {}

    def test_non_dict_raises(self):
        """A non-dict argument raises ``ValueError``."""
        with pytest.raises(ValueError, match="Expected a dict"):
            registers([("a", 2)])

    def test_verbose_default_true(self):
        """``verbose`` defaults to ``True`` and is overridable."""
        assert registers({"a": 2}).verbose is True
        assert registers({"a": 2}, verbose=False).verbose is False


class TestHelpers:
    """Internal helpers: distinct-wire count and the reindex offset."""

    def test_n_wires_flat(self):
        """``n_wires`` counts the distinct wires of a flat register set."""
        assert registers({"a": 2, "b": 2}).n_wires == 4

    def test_n_wires_nested_counts_distinct(self):
        """``n_wires`` counts distinct wires, not summed register lengths."""
        # Parent key 'x' shares wires with its children; distinct count is 7.
        assert registers({"x": {"a": 2, "b": 2}, "c": 3}).n_wires == 7

    def test_offset_empty(self):
        """An empty register set has a zero reindex offset."""
        assert registers(None)._offset() == 0  # pylint: disable=protected-access

    def test_offset_flat(self):
        """The offset clears the largest existing integer wire label."""
        assert registers({"a": 2, "b": 2})._offset() == 4  # pylint: disable=protected-access


class TestAddDisjoint:
    """The headline behaviour: reindex the right operand on concatenation."""

    def test_basic_reindex(self):
        """Adding disjoint register sets offsets the right operand's wires."""
        reg1 = registers({"a": 2, "b": 2})
        reg2 = registers({"c": 2, "d": 2})
        out = reg1 + reg2
        assert {k: v.tolist() for k, v in out.items()} == {
            "a": [0, 1],
            "b": [2, 3],
            "c": [4, 5],
            "d": [6, 7],
        }

    def test_returns_registers_instance(self):
        """The sum is itself a ``registers`` instance, not a plain dict."""
        out = registers({"a": 2}) + registers({"b": 2})
        assert isinstance(out, registers)

    def test_operands_unmodified(self):
        """Addition does not mutate either operand."""
        reg1 = registers({"a": 2, "b": 2})
        reg2 = registers({"c": 2, "d": 2})
        _ = reg1 + reg2
        assert reg1["a"].tolist() == [0, 1]
        assert reg2["c"].tolist() == [0, 1]  # right operand untouched

    def test_add_non_dict_returns_notimplemented(self):
        """Adding a non-dict falls back to a ``TypeError`` via NotImplemented."""
        with pytest.raises(TypeError):
            _ = registers({"a": 2}) + 5

    def test_radd_supports_sum(self):
        """``__radd__`` enables ``sum`` with a ``registers`` start value."""
        regs = [registers({"a": 2}), registers({"b": 2}), registers({"c": 3})]
        out = sum(regs, start=registers())
        assert {k: v.tolist() for k, v in out.items()} == {"a": [0, 1], "b": [2, 3], "c": [4, 5, 6]}


class TestAddNameClash:
    """Resolution rules when the same register name appears in both operands."""

    def test_identical_wires_kept_once_no_warning(self, recwarn):
        """Identical clashing wires are kept once and emit no warning."""
        left = registers({"a": 2, "b": 2})
        right = registers({"a": 2, "x": 3})
        out = left + right
        assert out["a"].tolist() == [0, 1]  # identical -> kept once
        assert out["b"].tolist() == [2, 3]
        assert out["x"].tolist() == [6, 7, 8]  # non-clashing key still offset
        assert len(recwarn) == 0

    def test_different_size_raises(self):
        """Clashing names with different sizes raise ``ValueError``."""
        with pytest.raises(ValueError, match="different sizes"):
            _ = registers({"a": 2}) + registers({"a": 3})

    def test_different_wires_keep_disjoint_and_warn(self):
        """A same-size clash keeps the wire set disjoint from the rest, with a warning."""
        left = registers({"a": Wires([0, 1]), "b": Wires([2, 3])})
        right = registers({"a": Wires([2, 3])})  # collides with left['b']
        with pytest.warns(UserWarning, match="merged by keeping"):
            out = left + right
        assert out["a"].tolist() == [0, 1]  # existing is disjoint -> kept

    def test_both_disjoint_prefers_left(self):
        """When both clashing wire sets are safe, the left operand wins."""
        left = registers({"a": Wires([0, 1])})
        right = registers({"a": Wires([5, 6])})  # both disjoint (no other regs)
        with pytest.warns(UserWarning, match="merged by keeping"):
            out = left + right
        assert out["a"].tolist() == [0, 1]

    def test_ambiguous_raises(self):
        """A clash where neither wire set is disjoint raises ``ValueError``."""
        # For key 'a', others={1,2}; existing [0,1] hits 1, incoming [2,3] hits 2.
        left = registers({"a": Wires([0, 1]), "b": Wires([1, 2])})
        right = registers({"a": Wires([2, 3])})
        with pytest.raises(ValueError, match="ambiguous"):
            _ = left + right

    def test_verbose_false_silences_warning(self, recwarn):
        """``verbose=False`` suppresses the merge warning."""
        left = registers({"a": Wires([0, 1]), "b": Wires([2, 3])}, verbose=False)
        right = registers({"a": Wires([2, 3])})
        out = left + right
        assert out["a"].tolist() == [0, 1]
        assert len(recwarn) == 0


class TestRepr:  # pylint: disable=too-few-public-methods
    """String representation of a register set."""

    def test_repr_roundtrips_dict(self):
        """``repr`` is prefixed with the class name."""
        reg = registers({"a": 2})
        assert repr(reg).startswith("registers(")
