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

from pennylane.exceptions import AdjointUndefinedError, PowUndefinedError
from pennylane.operation import _UNSET_BATCH_SIZE
from pennylane.operation2 import Operator2
from pennylane.pauli import PauliSentence, PauliWord
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


class DynOp(Operator2):
    """A simple operator with one dynamic param and wires."""

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(phi, wires=wires)


class FullOp(Operator2):
    """An operator using all argname groups."""

    dynamic_argnames = ("phi",)
    static_argnames = ("static",)
    hybrid_argnames = ("hybrid",)

    def __init__(self, phi, static, hybrid, wires):
        super().__init__(phi, static, hybrid, wires=wires)


class TestOperatorInit:
    """Tests for ``Operator2.__init__``."""

    def test_arguments_bound(self):
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

    def test_wires_collected_from_wire_argnames(self):
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

    def test_hybrid_arg_operator_wires_collected(self):
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

    def test_op_is_queued_on_init(self):
        """Test that instantiating an operator appends it to the active queue."""

        with AnnotatedQueue() as q:
            op = DynOp(0.5, wires=0)

        assert len(q) == 1
        assert list(q.keys())[0].obj is op


class TestProperties:
    """Tests for public properties of ``Operator2``."""

    def test_arguments(self):
        """Test that ``arguments`` maps all arguments to their values."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.arguments == {
            "phi": 0.5,
            "static": "info",
            "hybrid": [],
            "wires": Wires([0]),
        }

    def test_dynamic_args(self):
        """Test that ``dynamic_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.dynamic_args == {"phi": 0.5}

    def test_static_args(self):
        """Test that ``static_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.static_args == {"static": "info"}

    def test_wire_args(self):
        """Test that ``wire_args`` is set correctly."""
        op = FullOp(0.5, "info", [], wires=0)
        assert op.wire_args == {"wires": Wires([0])}

    def test_hybrid_args(self):
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

    def test_name(self):
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

    def test_queue_category(self):
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


class TestEquality:
    """Tests for operator equality. Since ``Operator2.__eq__`` uses ``qp.equal``
    under the hood, the bulk of testing for operator equality are elsewhere."""

    def test_eq_identical_instance(self):
        """Test that an operator is equal to itself."""
        op = DynOp(0.5, wires=0)
        assert op == op  # pylint: disable=comparison-with-itself

    def test_eq_equal_operators(self):
        """Test that distinct operators with the same arguments are equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=0)
        assert op1 == op2
        assert op2 == op1

    def test_eq_different_dynamic_args(self):
        """Test that operators with different dynamic args are not equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.6, wires=0)
        assert op1 != op2
        assert op2 != op1

    def test_eq_different_wires(self):
        """Test that operators acting on different wires are not equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=1)
        assert op1 != op2
        assert op2 != op1

    def test_eq_different_type(self):
        """Test that two ``Operator2`` subclasses are not equal."""

        class Other(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op1 = DynOp(0.5, wires=0)
        op2 = Other(0.5, wires=0)
        assert op1 != op2
        assert op2 != op1

    @pytest.mark.parametrize("other", [42, "string", 0.5, None, [DynOp(0.5, wires=0)]])
    def test_eq_against_non_operator(self, other):
        """Test that comparing an operator with a non-``Operator2`` value returns False."""
        op = DynOp(0.5, wires=0)
        assert op != other


class TestHash:
    """Tests for ``Operator2`` hashing."""

    def test_op_is_hashable(self):
        """Test that ``Operator2`` is hashable."""
        op = DynOp(0.5, wires=0)
        assert isinstance(hash(op), int)

    def test_op_can_be_dict_key(self):
        """Test that an ``Operator2`` instance can be used as a dict key."""
        op = DynOp(0.5, wires=0)
        d = {op: "value"}
        assert d[DynOp(0.5, wires=0)] == "value"

    def test_set_deduplicates_equal_operators(self):
        """Test that equal operators do not make unique entries when collected into a set."""
        ops = {DynOp(0.5, wires=0) for _ in range(5)}
        assert len(ops) == 1

    def test_equal_ops_hash(self):
        """Test the hash-equality invariant: ``a == b`` implies ``hash(a) == hash(b)``."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=0)
        assert op1 == op2
        assert hash(op1) == hash(op2)

    def test_close_floats_same_hash(self):
        """Test that float values that round to the same 10 decimal places hash equally."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5 + 1e-12, wires=0)
        assert hash(op1) == hash(op2)

    def test_different_args_hash(self):
        """Test that differing arguments produces different hashes."""
        op1 = FullOp(phi=0.5, static="a", hybrid=[0, 1], wires=0)
        op2 = FullOp(phi=0.6, static="ab", hybrid=[1, 1], wires=1)
        assert hash(op1) != hash(op2)

    def test_different_types_different_hash(self):
        """Test that operators of different types produce different hashes."""

        class Other(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=wires)

        op1 = DynOp(0.5, wires=0)
        op2 = Other(0.5, wires=0)
        assert hash(op1) != hash(op2)

    def test_different_hybrid_operator_args_different_hash(self):
        """Test that differing hybrid arguments that have operator leaves produces different
        hashes."""
        op1 = FullOp(0.5, "a", [DynOp(0.5, wires=0)], wires=0)
        op2 = FullOp(0.5, "a", [DynOp(0.6, wires=0)], wires=0)
        assert hash(op1) != hash(op2)

    def test_hybrid_with_wires_leaf_equal_hash(self):
        """Test that hybrid arguments with ``Wires`` leaves hash equally for the same values."""

        class HybridWireOp(Operator2):
            wire_argnames = ("pytree_wires",)
            hybrid_argnames = ("pytree_wires",)

            def __init__(self, pytree_wires):
                super().__init__([Wires(w) for w in pytree_wires])

        op1 = HybridWireOp([[0, 1], [2]])
        op2 = HybridWireOp([[0, 1], [2]])
        assert hash(op1) == hash(op2)

    def test_hybrid_different_pytree_structure_different_hash(self):
        """Test that hybrid arguments with different pytree structures hash differently."""
        op1 = FullOp(0.5, "a", [0.1, 0.2], wires=0)
        op2 = FullOp(0.5, "a", [0.1, [0.2]], wires=0)
        assert hash(op1) != hash(op2)

    @pytest.mark.parametrize("name", ["RX", "RY", "RZ", "PhaseShift", "Rot"])
    def test_rotation_gate_modulo_2pi(self, name):
        """Test that operators with designated rotation gate names hash modulo 2 * pi."""

        Op = type(
            name,
            (Operator2,),
            {
                "dynamic_argnames": ("phi",),
                "__init__": lambda self, phi, wires: Operator2.__init__(self, phi, wires=wires),
            },
        )

        op1 = Op(0.5, wires=0)
        op2 = Op(0.5 + 2 * np.pi, wires=0)
        assert hash(op1) == hash(op2)

    def test_non_rotation_no_modulo(self):
        """Test that non-rotation operators do not apply modulo 2 * pi hashing."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5 + 2 * np.pi, wires=0)
        assert hash(op1) != hash(op2)


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

    def test_flatten_order_dynamic_wire_hybrid(self):
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

    def test_non_existent_attr(self):
        """Test that an attribute error is raised if trying to access a property
        that doesn't exist."""
        op = DynOp(phi=1.5, wires=0)
        with pytest.raises(AttributeError, match="object has no attribute"):
            _ = op.invalid_attr

    def test_signature_parameter_property(self):
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

    def test_repr_with_dynamic_args(self):
        """Test that __repr__ includes dynamic parameters if present."""
        op = DynOp(0.5, wires=[0, 1])
        assert repr(op) == "DynOp(0.5, wires=[0, 1])"

    def test_repr_without_dynamic_args(self):
        """Test that __repr__ prints without dynamic parameters if there are none."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=wires)

        op = Op(wires=0)
        assert repr(op) == "Op(wires=[0])"

    def test_copy(self):
        """Test that shallow copies of operators are created correctly."""
        op = FullOp(0.5, static="static", hybrid=[], wires=0)
        op_copy = copy.copy(op)

        assert op_copy is not op
        assert type(op_copy) is type(op)

        for attr in ("phi", "static", "hybrid", "wires", "arguments"):
            assert getattr(op_copy, attr) == getattr(op, attr)
            # Shallow copy so stored attributes must be the same references
            assert getattr(op_copy, attr) is getattr(op, attr)

    def test_deepcopy(self):
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


class TestPublicProperties:
    """Tests for public properties of ``Operator2``."""

    def test_arithmetic_depth_default(self):
        """Test that the default ``arithmetic_depth`` is 0."""
        op = DynOp(0.5, wires=0)
        assert op.arithmetic_depth == 0

    def test_is_verified_hermitian_default(self):
        """Test that the default ``is_verified_hermitian`` is False."""
        op = DynOp(0.5, wires=0)
        assert op.is_verified_hermitian is False

    def test_pauli_rep_default(self):
        """Test that ``pauli_rep`` is ``None`` by default."""

        op = DynOp(0.5, wires=0)
        assert op.pauli_rep is None


class TestLabel:
    """Tests for the ``label`` method."""

    def test_no_dynamic_args(self):
        """Test that ``label`` returns just the class name when there
        are no dynamic args."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

        assert Op(wires=0).label() == "Op"

    @pytest.mark.parametrize(
        "category", ["static_argnames", "compilable_argnames", "hybrid_argnames"]
    )
    def test_non_dynamic_not_included(self, category):
        """Test that arguments not in ``dynamic_argnames`` aren't included
        in the label."""

        def __init__(self, phi, arg, wires):
            Operator2.__init__(self, phi, arg, Wires(wires))

        attrs = {
            "dynamic_argnames": ("phi",),
            "wire_argnames": ("wires",),
            category: ("arg",),
            "__init__": __init__,
        }
        # This creates a class while allowing us to parametrize the attributes
        # that we want to test.
        OpClass = type("OpClass", (Operator2,), attrs)

        assert OpClass(phi=1.5, arg=[1, 2, 3], wires=0).label() == "OpClass"

    def test_no_dynamic_args_with_base_label(self):
        """Test that ``base_label`` overrides the class name even with
        no dynamic args."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

        assert Op(wires=0).label(base_label="custom") == "custom"

    def test_scalar_decimals_none_excludes_param(self):
        """Test that scalar parameters are excluded from the label when
        ``decimals`` is ``None``."""
        op = DynOp(1.23456, wires=0)
        assert op.label() == "DynOp"

    def test_scalar_decimals_formats_param(self):
        """Test that scalar parameters are formatted to ``decimals`` places."""
        op = DynOp(1.23456, wires=0)
        assert op.label(decimals=2) == "DynOp\n(1.23)"

    def test_base_label_with_decimals(self):
        """Test that ``base_label`` and ``decimals`` can be combined."""
        op = DynOp(1.23456, wires=0)
        assert op.label(decimals=3, base_label="custom") == "custom\n(1.235)"

    def test_multiple_dynamic_args(self):
        """Test that multiple dynamic params are joined with ``,\\n``."""

        class TwoArgs(Operator2):
            dynamic_argnames = ("phi", "theta")

            def __init__(self, phi, theta, wires):
                super().__init__(phi, theta, wires=Wires(wires))

        op = TwoArgs(1.23, 4.567, wires=0)
        assert op.label(decimals=2) == "TwoArgs\n(1.23,\n4.57)"

    def test_matrix_param_no_cache_omitted(self):
        """Test that a matrix parameter is omitted from the label when no
        cache is provided."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        op = MatOp(np.eye(2), wires=0)
        # Without a cache, matrix-valued params produce empty strings.
        assert op.label(decimals=2) == "MatOp"

    def test_matrix_param_with_cache(self):
        """Test that matrix parameters are recorded in the cache and referenced
        as ``M{index}``."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        cache = {"matrices": []}
        op = MatOp(np.eye(2), wires=0)

        assert op.label(cache=cache) == "MatOp\n(M0)"
        assert len(cache["matrices"]) == 1
        assert np.allclose(cache["matrices"][0], np.eye(2))

    def test_matrix_param_cache_reuse(self):
        """Test that the same matrix is reused from the cache rather than appended again."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        cache = {"matrices": []}
        op1 = MatOp(np.eye(2), wires=0)
        op2 = MatOp(np.eye(2), wires=1)

        assert op1.label(cache=cache) == "MatOp\n(M0)"
        assert op2.label(cache=cache) == "MatOp\n(M0)"
        assert len(cache["matrices"]) == 1

    def test_matrix_param_cache_growth(self):
        """Test that distinct matrices are appended with incrementing indices."""

        class MatOp(Operator2):
            dynamic_argnames = ("U",)

            def __init__(self, U, wires):
                super().__init__(U, wires=Wires(wires))

        cache = {"matrices": []}
        op1 = MatOp(np.eye(2), wires=0)
        op2 = MatOp(np.eye(4), wires=[0, 1])

        assert op1.label(cache=cache) == "MatOp\n(M0)"
        assert op2.label(cache=cache) == "MatOp\n(M1)"
        assert len(cache["matrices"]) == 2

    def test_subclass_override(self):
        """Test that a subclass can override ``label``."""

        class Op(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

            # pylint: disable=unused-argument
            def label(self, decimals=None, base_label=None, cache=None):
                return "custom_label"

        assert Op(wires=0).label() == "custom_label"


class TestGeneralMethods:
    """Tests for various general methods in ``Operator2``."""

    def test_pow_zero_returns_empty(self):
        """Test that ``op.pow(0)`` returns an empty list."""
        op = DynOp(0.5, wires=0)
        assert op.pow(0) == []

    def test_pow_one_returns_single(self):
        """Test that ``op.pow(1)`` returns a single-element list with a copy of the operator."""
        op = DynOp(0.5, wires=0)
        result = op.pow(1)

        assert len(result) == 1
        assert result[0] == op
        assert result[0] is not op

    def test_pow_positive_integer(self):
        """Test that ``op.pow(z)`` for positive integer ``z`` returns ``z`` copies."""
        op = DynOp(0.5, wires=0)
        result = op.pow(3)

        assert len(result) == 3
        assert all(r == op for r in result)
        assert all(r is not op for r in result)

        # Each op must be a new copy
        assert len({id(r) for r in result}) == 3

    def test_pow_queuing(self):
        """Test that ``op.pow(z)`` inside a queuing context queues ``z`` additional copies."""
        with AnnotatedQueue() as q:
            op = DynOp(0.5, wires=0)
            result = op.pow(2)

        # 1 entry from original op + 2 entries from pow.
        assert len(q) == 3
        assert len(result) == 2

    @pytest.mark.parametrize("z", [-1, -2, 1.5, 0.5])
    def test_pow_invalid_raises(self, z):
        """Test that ``op.pow`` raises ``PowUndefinedError`` for negative or non-integer exponents."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(PowUndefinedError):
            _ = op.pow(z)

    def test_subclass_pow_override(self):
        """Test that a subclass can override ``pow``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=Wires(wires))

            def pow(self, z):
                return [type(self)(self.phi * z, wires=self.wires)]

        op = Op(2.0, wires=0)
        result = op.pow(3)
        assert len(result) == 1
        assert result[0].phi == 6.0

    def test_map_wires_basic(self):
        """Test that ``map_wires`` maps the wires of the operator according
        to the wire map."""
        op = DynOp(0.5, wires=0)
        new_op = op.map_wires({0: "a"})

        # Check that the original op is unchanged
        assert op.wires == Wires([0])
        assert new_op == DynOp(0.5, wires="a")

    def test_map_wires_unmapped_labels_preserved(self):
        """Test that wire labels missing from the wire map are kept as-is."""
        op = DynOp(0.5, wires=[0, 1, 2])
        new_op = op.map_wires({0: "a", 2: "c"})

        assert new_op == DynOp(0.5, wires=["a", 1, "c"])

    def test_map_wires_multiple_wire_args(self):
        """Test that ``map_wires`` maps wires across multiple wire arguments."""

        class TwoWireOp(Operator2):
            wire_argnames = ("wires", "ctrl_wires")

            def __init__(self, wires, ctrl_wires):
                super().__init__(Wires(wires), Wires(ctrl_wires))

        op = TwoWireOp(wires=[0, 1], ctrl_wires=[2])
        new_op = op.map_wires({0: "a", 1: "b", 2: "c"})

        assert new_op == TwoWireOp(wires=["a", "b"], ctrl_wires=["c"])

    def test_map_wires_pytree_hybrid_wires(self):
        """Test that ``map_wires`` correctly maps pytree-structured wires inside a hybrid arg."""

        class PytreeWiresOp(Operator2):
            wire_argnames = ("wires",)
            hybrid_argnames = ("wires",)

            def __init__(self, wires):
                wires = [Wires(w) for w in wires]
                super().__init__(wires=wires)

        op = PytreeWiresOp(wires=[[0], [1, 2]])
        new_op = op.map_wires({0: "a", 1: "b"})

        assert new_op == PytreeWiresOp(wires=[["a"], ["b", 2]])

    def test_map_wires_pauli_rep(self):
        """Test that ``Operator2.map_wires`` maps the ``pauli_rep`` correctly."""
        op = DynOp(1.5, wires=[0, 1])
        op._pauli_rep = PauliSentence({PauliWord({0: "X", 1: "Y"}): 1.0})

        new_op = op.map_wires({0: "a", 1: "b"})
        assert new_op.pauli_rep == PauliSentence({PauliWord({"a": "X", "b": "Y"}): 1.0})

    def test_simplify_default(self):
        """Test that ``simplify`` returns the operator itself by default."""
        op = DynOp(0.5, wires=0)
        assert op.simplify() is op

    def test_subclass_simplify_override(self):
        """Test that a subclass can override ``simplify``."""

        class Op(Operator2):
            dynamic_argnames = ("phi",)

            def __init__(self, phi, wires):
                super().__init__(phi, wires=Wires(wires))

            def simplify(self):
                return type(self)(0.0, wires=self.wires)

        op = Op(0.5, wires=0)
        simplified = op.simplify()

        # Assert that the original op is left unchanged
        assert op.phi == 0.5
        assert simplified == Op(0.0, wires=0)


class TestHasRepresentations:
    """Tests for the ``has_**`` representation properties."""

    def test_has_adjoint_default(self):
        """Test that ``has_adjoint`` is False by default."""
        assert DynOp.has_adjoint is False
        assert Operator2.has_adjoint is False

    def test_has_adjoint_true_when_adjoint_overridden(self):
        """Test that ``has_adjoint`` is True when the subclass overrides ``adjoint``."""

        class SelfAdj(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

            def adjoint(self):
                return self

        assert SelfAdj.has_adjoint is True


class TestRepresentations:
    """Tests for the various operator representation methods
    (and their corresponding ``compute_**`` static methods)."""

    def test_adjoint_default_raises(self):
        """Test that the default ``adjoint`` raises ``AdjointUndefinedError``."""
        op = DynOp(0.5, wires=0)
        with pytest.raises(AdjointUndefinedError):
            op.adjoint()

    def test_subclass_adjoint_override(self):
        """Test that a subclass can override ``adjoint``."""

        class SelfAdj(Operator2):
            def __init__(self, wires):
                super().__init__(wires=Wires(wires))

            def adjoint(self):
                return type(self)(wires=self.wires)

        op = SelfAdj(wires=0)
        adj = op.adjoint()
        assert adj == op
