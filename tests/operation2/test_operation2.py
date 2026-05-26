# Copyright 2026 Xanadu Quantum Technologies Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic unit tests for ``Operator2``."""

# pylint: disable=redefined-outer-name,protected-access,too-few-public-methods

import copy

import numpy as np
import pytest

from pennylane.operation import _UNSET_BATCH_SIZE
from pennylane.operation2 import Operator2
from pennylane.pytrees.pytrees import flatten_registrations, unflatten_registrations
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires


class TestInitSubclass:
    """Tests for the validation performed in ``Operator2.__init_subclass__``."""

    def test_str_argnames_converted_to_tuple(self):
        """Test that string argnames are converted into one-tuples."""

        class Op(Operator2):
            dynamic_argnames = "phi"
            static_argnames = "pw"
            hybrid_argnames = "wires"

            def __init__(self, phi, pw, wires):
                super().__init__(phi, pw, wires=wires)

        assert Op.dynamic_argnames == ("phi",)
        assert Op.static_argnames == ("pw",)
        assert Op.hybrid_argnames == ("wires",)
        assert Op.wire_argnames == ("wires",)
        assert Op.compilable_argnames == ()

    def test_static_and_compilable_both_set_raises(self):
        """Test that declaring both ``static_argnames`` and ``compilable_argnames``
        is not allowed."""

        with pytest.raises(
            TypeError, match="only contain 'static_argnames' or 'compilable_argnames'"
        ):
            # pylint: disable=unused-variable
            class Op(Operator2):
                static_argnames = ("a",)
                compilable_argnames = ("b",)

                def __init__(self, a, b, wires):
                    super().__init__(a, b, wires=wires)

    @pytest.mark.parametrize(
        "first, second",
        [
            ("dynamic_argnames", "wire_argnames"),
            ("dynamic_argnames", "static_argnames"),
            ("dynamic_argnames", "compilable_argnames"),
            ("static_argnames", "wire_argnames"),
            ("compilable_argnames", "wire_argnames"),
        ],
    )
    def test_pairwise_overlap_raises(self, first, second):
        """Test that dynamic, wire, static, and compilable argnames must be disjoint."""

        def __init__(self, x):
            Operator2.__init__(self, x)

        attrs = {first: ("x",), second: ("x",), "__init__": __init__}
        # If neither overlapping group is ``wire_argnames``, blank out the
        # default ``("wires",)`` since this signature has no ``wires`` param.
        if "wire_argnames" not in (first, second):
            attrs["wire_argnames"] = ()

        with pytest.raises(TypeError, match="must not overlap"):
            # This creates a class while allowing us to parametrize the attributes
            # that we want to test.
            _ = type("Op", (Operator2,), attrs)

    def test_hybrid_may_overlap_with_wires(self):
        """Test that ``hybrid_argnames`` is allowed to overlap with ``wire_argnames``."""

        class Op(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires=wires)

        assert Op.hybrid_argnames == ("wires",)

    @pytest.mark.parametrize(
        "other_group", ["dynamic_argnames", "static_argnames", "compilable_argnames"]
    )
    def test_hybrid_overlap_with_non_wire_raises(self, other_group):
        """Test that ``hybrid_argnames`` may not overlap with dynamic, static,
        or compilable argnames."""

        def __init__(self, x, wires):
            Operator2.__init__(self, x, wires=wires)

        attrs = {"hybrid_argnames": ("x",), other_group: ("x",), "__init__": __init__}

        with pytest.raises(TypeError, match="overlap with dynamic, static"):
            # This creates a class while allowing us to parametrize the attributes
            # that we want to test.
            type("Op", (Operator2,), attrs)

    def test_unclassified_signature_parameter_raises(self):
        """Test that every parameter in the signature must appear in some ``**_argnames`` tuple."""

        with pytest.raises(TypeError, match="not classified in any argnames"):
            # pylint: disable=unused-variable
            class Op(Operator2):
                # ``phi`` is not in any argnames tuple
                def __init__(self, phi, wires):
                    super().__init__(phi, wires=wires)

    def test_signature_captured(self):
        """Test that ``cls._sig`` is set to the subclass's __init__ signature."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert list(Op._sig.parameters) == ["phi", "wires"]
        assert "self" not in Op._sig.parameters

    def test_class_registered_as_pytree(self):
        """Test that subclasses are automatically registered as pytrees."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        assert Op in flatten_registrations
        assert Op in unflatten_registrations


@pytest.fixture
def DynOp():
    """A simple operator with one dynamic param and wires."""

    class _DynOp(Operator2):
        dynamic_argnames = ("phi",)

        def __init__(self, phi, wires):
            super().__init__(phi, wires=wires)

    return _DynOp


@pytest.fixture
def FullOp():
    """An operator using all argname groups."""

    class _FullOp(Operator2):
        dynamic_argnames = ("phi",)
        static_argnames = ("static",)
        hybrid_argnames = ("hybrid",)

        def __init__(self, phi, static, hybrid, wires):
            super().__init__(phi, static, hybrid, wires=wires)

    return _FullOp


class TestOperatorInit:
    """Tests for ``Operator2.__init__``."""

    def test_arguments_bound(self, DynOp):
        """Test that constructor positional/keyword arguments are bound into ``arguments``."""

        op = DynOp(0.5, wires=0)
        assert op.arguments == {"phi": 0.5, "wires": Wires([0])}

    def test_defaults_applied(self):
        """Test that missing kwargs receive their default values."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            static_argnames = ("method",)

            # pylint: disable=unused-argument
            def __init__(self, phi, wires, method="auto"):
                super().__init__(phi, Wires(wires))

        op = Op(0.5, wires=0)
        assert op.arguments["method"] == "auto"

    def test_pauli_rep_default_is_none(self, DynOp):
        """Test that ``_pauli_rep`` is initialized to ``None``."""

        op = DynOp(0.5, wires=0)
        assert op.pauli_rep is None

    def test_wires_collected_from_wire_argnames(self, DynOp):
        """Test that the ``_wires`` attribute combines the contents of ``wire_argnames``."""

        op = DynOp(0.5, wires=[0, 2])
        assert op.wires == Wires([0, 2])

    def test_wires_collected_from_multiple_wire_args(self):
        """Test that multiple wire argnames are unioned, preserving order."""

        class TwoWiresArgsOp(Operator2):
            wire_argnames = ("wires", "ctrl_wires")

            def __init__(self, wires, ctrl_wires):
                super().__init__(Wires(wires), Wires(ctrl_wires))

        op = TwoWiresArgsOp(wires=[0, 1], ctrl_wires=2)
        assert op.wires == Wires([0, 1, 2])

    def test_work_wires_excluded_from_wires(self):
        """Test that the ``work_wires``/``work_wire`` argnames are not added to ``_wires``."""

        class WithWorkWires(Operator2):
            wire_argnames = ("wires", "work_wires", "work_wire")

            def __init__(self, wires, work_wires, work_wire):
                super().__init__(Wires(wires), Wires(work_wires), Wires(work_wire))

        op = WithWorkWires(wires=[0, 1], work_wires=[2, 3], work_wire=4)
        assert op.wires == Wires([0, 1])

    def test_hybrid_arg_operator_wires_collected(self, DynOp):
        """Test that operators inside ``hybrid_argnames`` contribute their wires."""

        class Composite(Operator2):
            hybrid_argnames = ("ops", "pytree_wires")
            wire_argnames = ("wires", "pytree_wires")

            def __init__(self, ops, pytree_wires, wires):
                ptw = [Wires(w) for w in pytree_wires]
                super().__init__(ops, ptw, Wires(wires))

        inner = [DynOp(0.1, wires=7), DynOp(0.2, wires=8)]
        op = Composite(inner, [0, [1, 2], [3, 4]], [5, 6])
        # Wires are ordered using wire_argnames, so `wires` come before `pytree_wires`
        assert op.wires == Wires([5, 6, 0, 1, 2, 3, 4, 7, 8])

    def test_non_hybrid_wire_arg_auto_wrapped_in_constructor(self):
        """Test that non-hybrid wire arguments are wrapped in ``Wires`` by the
        ``Operator2`` constructor even if the subclass forwards a raw value."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                # No Wires() wrapping here — the constructor should do it.
                super().__init__(phi, wires=wires)

        op = Op(0.5, wires=0)
        assert isinstance(op.arguments["wires"], Wires)
        assert op.arguments["wires"] == Wires([0])

    @pytest.mark.parametrize(
        "raw_wires, expected",
        [
            (0, Wires([0])),
            ([0, 1, 2], Wires([0, 1, 2])),
            ((0, 1), Wires([0, 1])),
            (range(3), Wires([0, 1, 2])),
            (Wires([3, 4]), Wires([3, 4])),
        ],
    )
    def test_non_hybrid_wire_arg_auto_wrapped_various_inputs(self, raw_wires, expected):
        """Test that the constructor accepts a variety of raw inputs for wire
        arguments and canonicalizes them to a ``Wires`` instance."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=raw_wires)
        assert op.arguments["wires"] == expected
        assert op.wires == expected

    def test_hybrid_wire_arg_with_non_wires_leaf_raises(self):
        """Test that a hybrid wire argument whose leaves are not ``Wires``
        instances raises a ``TypeError``."""

        class Op(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                # Forwards the raw list without wrapping each leaf in Wires.
                super().__init__(wires=wires)

        with pytest.raises(TypeError, match="Hybrid wires argument 'wires' have not been cast"):
            _ = Op(wires=[0, [1, 2]])

    def test_op_is_queued_on_init(self, DynOp):
        """Test that instantiating an operator appends it to the active queue."""

        with AnnotatedQueue() as q:
            op = DynOp(0.5, wires=0)

        assert len(q) == 1
        assert list(q.keys())[0].obj is op


class TestProperties:
    """Tests for public properties of ``Operator2``."""

    def test_arguments(self, FullOp):
        """Test that ``arguments`` maps all arguments to their values."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.arguments == {
            "phi": 0.5,
            "static": "info",
            "hybrid": [],
            "wires": Wires([0]),
        }

    def test_dynamic_args(self, FullOp):
        """Test that ``dynamic_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.dynamic_args == {"phi": 0.5}

    def test_static_args(self, FullOp):
        """Test that ``static_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.static_args == {"static": "info"}

    def test_wire_args(self, FullOp):
        """Test that ``wire_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.wire_args == {"wires": Wires([0])}

    def test_hybrid_args(self, FullOp):
        """Test that ``hybrid_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.hybrid_args == {"hybrid": []}

    def test_compilable_args(self):
        """Test that ``compilable_args`` is set correctly."""

        class Op(Operator2):
            compilable_argnames = ("pw",)

            def __init__(self, pw, wires):
                super().__init__(pw, wires=wires)

        op = Op("XY", wires=[0, 1])
        assert op.compilable_args == {"pw": "XY"}

    def test_name(self, DynOp):
        """Test that ``name`` is the same as the class name."""
        op = DynOp(0.5, wires=0)
        assert op.name == op.__class__.__name__

    def test_wires(self):
        """Test ``wires`` is set correctly."""

        class Op(Operator2):
            wire_argnames = ("wires1", "wires")

            def __init__(self, wires, wires1):
                super().__init__(Wires(wires), Wires(wires1))

        op = Op([0, 1], [2, 3, 4])
        assert op.wires == Wires([2, 3, 4, 0, 1])

    def test_queue_category(self, DynOp):
        op = DynOp(0.5, wires=0)
        assert op._queue_category == "_ops"


class TestBroadcasting:
    """Tests for parameter broadcasting."""

    @pytest.mark.parametrize("data, exp_batch_size", [(1.1, None), ([1.1, 2.2], 2)])
    def test_batch_size(self, data, exp_batch_size):
        """Test that ``batch_size`` is correct when ndim_params are specified
        during class definition."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (0,)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(data, wires=0)
        assert op._batch_size is _UNSET_BATCH_SIZE
        assert op.batch_size == exp_batch_size

    @pytest.mark.parametrize(
        "data", [(np.empty((2, 2)), np.empty((4, 4, 4, 4))), (np.array(1.5), np.empty((4,)))]
    )
    def test_inferred_ndim_params(self, data):
        """Test that ``ndim_params`` is assumed to be the same as the number of dimensions of
        input dynamic parameters if not set as a class variable."""

        class Op(Operator2):
            dynamic_argnames = ("a", "b")

            def __init__(self, a, b, wires):
                super().__init__(a, b, Wires(wires))

        op = Op(*data, wires=0)
        expected_ndims = tuple(d.ndim for d in data)
        assert op._ndim_params is _UNSET_BATCH_SIZE
        assert op.ndim_params == expected_ndims

    def test_wrong_ndim_raises(self):
        """Test that parameters with wrong dimensionality raise an error."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            ndim_params = (0,)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op([[[0.5]]], wires=0)
        with pytest.raises(ValueError, match=r"wrong number\(s\) of dimensions"):
            _ = op.batch_size

    def test_mismatched_broadcasting_raises(self):
        """Test that different broadcast dimensions across parameters raise an error."""

        class Op(Operator2):
            dynamic_argnames = ("a", "b")
            ndim_params = (0, 0)

            def __init__(self, a, b, wires):
                super().__init__(a, b, wires=wires)

        op = Op([0.3] * 4, [0.4] * 3, wires=0)
        with pytest.raises(ValueError, match="Broadcasting was attempted"):
            _ = op.batch_size


class TestPytreeMethods:
    """Tests for ``_flatten`` and ``_unflatten``."""

    def test_static_arg_in_metadata(self):
        """Test that flattening and unflattening work correctly when there are static_args."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            static_argnames = ("static",)

            def __init__(self, phi, static, wires):
                super().__init__(phi, static, wires=wires)

        op = Op(0.5, "bar", wires=0)
        data, metadata = op._flatten()

        assert data == ([0.5], [Wires([0])], [])
        assert metadata == ("bar",)

        new_op = Op._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires

    def test_compilable_arg_in_metadata(self):
        """Test that flattening and unflattening work correctly when there are compilable_args."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            compilable_argnames = ("static",)

            def __init__(self, phi, static, wires):
                super().__init__(phi, static, wires=wires)

        op = Op(1.5, "bar", wires=[0, 1])
        data, metadata = op._flatten()

        assert data == ([1.5], [Wires([0, 1])], [])
        assert metadata == ("bar",)

        new_op = Op._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires

    def test_flatten_order_dynamic_wire_hybrid(self, FullOp):
        """Test that data is ordered as dynamic, then wires, then non-wire hybrid."""

        op = FullOp(0.5, "static", [-1, -2, -3], wires=0)
        data, metadata = op._flatten()

        assert data == ([0.5], [Wires([0])], [[-1, -2, -3]])
        assert metadata == ("static",)

        new_op = FullOp._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires

    def test_hybrid_overlapping_with_wire_not_duplicated(self):
        """Test that hybrid wire arguments are flattened and unflattened correctly."""

        class Op(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                wires = [Wires(w) for w in wires]
                super().__init__(wires=wires)

        op = Op(wires=[0, [1, 2], (3,)])
        data, metadata = op._flatten()

        assert data == ([], [[Wires([0]), Wires([1, 2]), Wires([3])]], [])
        assert metadata == ()

        new_op = Op._unflatten(data, metadata)
        assert new_op.arguments == op.arguments
        assert new_op.wires == op.wires


class TestDynamicProperties:
    """Tests for the dynamic properties generated using operator parameters."""

    def test_non_existent_attr(self, DynOp):
        """Test that an attribute error is raised if trying to access a property
        that doesn't exist."""
        op = DynOp(phi=1.5, wires=0)
        with pytest.raises(AttributeError, match="object has no attribute"):
            _ = op.invalid_attr

    def test_signature_parameter_property(self, FullOp):
        """Test that operator parameters are exposed as properties."""
        op = FullOp(phi=0.5, static="static", hybrid=[], wires=0)
        assert op.phi == 0.5
        assert op.static == "static"
        assert op.hybrid == []

    def test_explicit_class_attribute_not_overridden(self):
        """Test that attributes present in the class are not overriden by dynamic properties."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)
            phi = -1

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op = Op(phi=100, wires=0)
        assert op.phi == -1


class TestDunderMethods:
    """Tests for ``Operator2`` dunder methods."""

    def test_repr_with_dynamic_args(self, DynOp):
        """Test that __repr__ includes dynamic parameters if present."""
        op = DynOp(0.5, wires=[0, 1])
        assert repr(op) == "_DynOp(0.5, wires=[0, 1])"

    def test_repr_without_dynamic_args(self):
        """Test that __repr__ prints without dynamic parameters if there are none."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=0)
        assert repr(op) == "Op(wires=[0])"

    def test_copy(self, FullOp):
        """Test that shallow copies of operators are created correctly."""
        op = FullOp(0.5, static="static", hybrid=[], wires=0)
        op_copy = copy.copy(op)

        assert op_copy is not op
        assert type(op_copy) is type(op)

        for attr in ("phi", "static", "hybrid", "wires", "arguments"):
            assert getattr(op_copy, attr) == getattr(op, attr)
            # Shallow copy so stored attributes must be the same references
            assert getattr(op_copy, attr) is getattr(op, attr)

    def test_deepcopy(self, FullOp):
        """Test that deep copies of operators are created correctly."""
        # Putting phi in a container because integer pointers have special handling in Python
        phi = {0.5}
        static = {"foo": 1.5}
        hybrid = [1, 2, 3]

        op = FullOp(phi=phi, static=static, hybrid=hybrid, wires=0)
        op_deep = copy.deepcopy(op)

        assert op_deep is not op

        for attr in ("phi", "static", "hybrid", "wires", "arguments"):
            assert getattr(op_deep, attr) == getattr(op, attr)
            # Deep copy so stored attributes must NOT be the same references
            assert getattr(op_deep, attr) is not getattr(op, attr)
