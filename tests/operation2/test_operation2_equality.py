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
"""Tests for ``qp.equal`` with ``Operator2``."""

# pylint: disable=too-few-public-methods

import numpy as np
import pytest

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.operation2 import Operator2
from pennylane.wires import Wires

# ---------------- Operator2 subclasses for testing ----------------


class DynOp(Operator2):
    """Operator with one dynamic parameter and wires."""

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(phi, wires=wires)


class TwoDynOp(Operator2):
    """Operator with two dynamic parameters."""

    dynamic_argnames = ("phi", "theta")

    def __init__(self, phi, theta, wires):
        super().__init__(phi, theta, wires=wires)


class StaticOp(Operator2):
    """Operator with a static argument."""

    static_argnames = ("label",)

    def __init__(self, label, wires):
        super().__init__(label, wires=wires)


class CompOp(Operator2):
    """Operator with a compilable static argument."""

    compilable_argnames = ("n",)

    def __init__(self, n, wires):
        super().__init__(n, wires=wires)


class MultiWireOp(Operator2):
    """Operator with two wire arguments."""

    wire_argnames = ("wires", "ctrl_wires")

    def __init__(self, wires, ctrl_wires):
        super().__init__(wires=wires, ctrl_wires=ctrl_wires)


class HybridOp(Operator2):
    """Operator with a hybrid argument that can contain Operator2 leaves."""

    hybrid_argnames = ("ops",)

    def __init__(self, ops, wires):
        super().__init__(ops, wires=wires)


class HybridWireOp(Operator2):
    """Operator with a wire argument that is also a hybrid argument."""

    wire_argnames = ("pytree_wires",)
    hybrid_argnames = ("pytree_wires",)

    def __init__(self, pytree_wires):
        super().__init__([Wires(w) for w in pytree_wires])


class FullOp(Operator2):
    """Operator with all argname groups."""

    dynamic_argnames = ("phi",)
    static_argnames = ("label",)
    hybrid_argnames = ("hybrid",)

    def __init__(self, phi, label, hybrid, wires):
        super().__init__(phi, label, hybrid, wires=wires)


# ---------------------- Tests ----------------------


class TestEqualBasic:
    """Basic tests for ``qp.equal`` with ``Operator2``."""

    def test_equal_identical(self):
        """Test that ``qp.equal`` with an operator compared to itself returns ``True``."""
        op = DynOp(0.5, wires=0)
        assert qp.equal(op, op) is True
        qp.assert_equal(op, op)

    def test_equal_same_args(self):
        """Test that two operators with identical arguments are equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=0)
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_different_types_not_equal(self):
        """Test that two operators of different ``Operator2`` subclasses are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = StaticOp("a", wires=0)
        assert qp.equal(op1, op2) is False

    def test_assert_equal_raises_for_unequal_ops(self):
        """Test that ``qp.assert_equal`` raises an informative error for unequal operators."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.6, wires=0)
        with pytest.raises(AssertionError, match="different values for 'phi'"):
            qp.assert_equal(op1, op2)


class TestDynamicArgs:
    """Tests for equality of dynamic arguments."""

    def test_different_values_not_equal(self):
        """Test that operators with different dynamic values are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.6, wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'phi'"):
            qp.assert_equal(op1, op2)

    def test_close_values_equal_within_tolerance(self):
        """Test that operators with dynamic values within the absolute tolerance are equal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5 + 1e-10, wires=0)
        assert qp.equal(op1, op2, atol=1e-9, rtol=0) is True
        qp.assert_equal(op1, op2, atol=1e-9, rtol=0)

    def test_close_values_not_equal_outside_tolerance(self):
        """Test that operators with dynamic values outside the absolute tolerance are
        unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5 + 1e-6, wires=0)
        assert qp.equal(op1, op2, atol=1e-9, rtol=0) is False

    def test_scalar_versus_batched_dynamic_arg_not_equal(self):
        """Test that operators with a scalar dynamic value and a broadcast-compatible batched
        dynamic value are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(np.array([0.5, 0.5, 0.5]), wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'phi'"):
            qp.assert_equal(op1, op2)

    def test_different_batch_shapes_dynamic_arg_not_equal(self):
        """Test that operators with batched dynamic values of different shapes are unequal."""
        op1 = DynOp(np.array([0.5, 0.5]), wires=0)
        op2 = DynOp(np.array([0.5, 0.5, 0.5]), wires=0)
        assert qp.equal(op1, op2) is False

    def test_multiple_dynamic_one_arg_differs(self):
        """Test that operators with one different dynamic argument being different are unequal."""
        op1 = TwoDynOp(0.5, 1.0, wires=0)
        op2 = TwoDynOp(0.5, 1.1, wires=0)
        with pytest.raises(AssertionError, match="different values for 'theta'"):
            qp.assert_equal(op1, op2)

    def test_different_interfaces_not_equal(self):
        """Test that operators with the same value but different interfaces are unequal."""
        op1 = DynOp(np.array(0.5), wires=0)
        op2 = DynOp(pnp.array(0.5, requires_grad=False), wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different interfaces for 'phi'"):
            qp.assert_equal(op1, op2)

    def test_different_interfaces_equal_when_disabled(self):
        """Test that ``check_interface=False`` ignores interface differences."""
        op1 = DynOp(np.array(0.5), wires=0)
        op2 = DynOp(pnp.array(0.5, requires_grad=False), wires=0)
        assert qp.equal(op1, op2, check_interface=False) is True
        qp.assert_equal(op1, op2, check_interface=False)

    def test_different_trainability_not_equal(self):
        """Test that operators differing only in trainability are unequal."""
        op1 = DynOp(pnp.array(0.5, requires_grad=True), wires=0)
        op2 = DynOp(pnp.array(0.5, requires_grad=False), wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="differ in trainability for 'phi'"):
            qp.assert_equal(op1, op2)

    def test_different_trainability_equal_when_disabled(self):
        """Test that ``check_trainability=False`` ignores trainability differences."""
        op1 = DynOp(pnp.array(0.5, requires_grad=True), wires=0)
        op2 = DynOp(pnp.array(0.5, requires_grad=False), wires=0)
        assert qp.equal(op1, op2, check_trainability=False) is True
        qp.assert_equal(op1, op2, check_trainability=False)


class TestStaticArgs:
    """Tests for equality of static arguments."""

    def test_equal_same_static(self):
        """Test that operators with the same static argument are equal."""
        op1 = StaticOp("a", wires=0)
        op2 = StaticOp("a", wires=0)
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_different_static_not_equal(self):
        """Test that operators with different static arguments are unequal."""
        op1 = StaticOp("a", wires=0)
        op2 = StaticOp("b", wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'label'"):
            qp.assert_equal(op1, op2)


class TestCompilableArgs:
    """Tests for equality of compilable arguments."""

    def test_equal_same_compilable(self):
        """Test that operators with the same compilable argument are equal."""
        op1 = CompOp(3, wires=0)
        op2 = CompOp(3, wires=0)
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_different_compilable_not_equal(self):
        """Test that operators with different compilable arguments are unequal."""
        op1 = CompOp(3, wires=0)
        op2 = CompOp(4, wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'n'"):
            qp.assert_equal(op1, op2)


class TestWireArgs:
    """Tests for equality of wire arguments."""

    def test_same_wires_equal(self):
        """Test that operators acting on the same wires are equal."""
        op1 = DynOp(0.5, wires=[0, 1])
        op2 = DynOp(0.5, wires=[0, 1])
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_different_wires_not_equal_single_arg(self):
        """Test that operators on different wires raise an informative message."""
        op1 = DynOp(0.5, wires=0)
        op2 = DynOp(0.5, wires=1)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different wires"):
            qp.assert_equal(op1, op2)

    def test_different_wire_order_not_equal(self):
        """Test that operators with the same wires but different order are unequal."""
        op1 = DynOp(0.5, wires=[0, 1])
        op2 = DynOp(0.5, wires=[1, 0])
        assert qp.equal(op1, op2) is False

    def test_same_multi_wire_args_equal(self):
        """Test equality across multiple wire arguments."""
        op1 = MultiWireOp(wires=[0, 1], ctrl_wires=2)
        op2 = MultiWireOp(wires=[0, 1], ctrl_wires=2)
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_different_multi_wire_arg_names_message(self):
        """Test that a difference in a wire argument names that argument
        in the error message."""
        op1 = MultiWireOp(wires=[0, 1], ctrl_wires=2)
        op2 = MultiWireOp(wires=[0, 1], ctrl_wires=3)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different wires for the 'ctrl_wires' argument"):
            qp.assert_equal(op1, op2)


class TestHybridArgs:
    """Tests for equality of hybrid arguments."""

    def test_equal_empty_hybrid(self):
        """Test equality when the hybrid argument is empty."""
        op1 = HybridOp([], wires=0)
        op2 = HybridOp([], wires=0)
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_equal_with_operator_leaves(self):
        """Test equality when hybrid arguments contain matching ``Operator2`` leaves."""
        op1 = HybridOp([DynOp(0.5, wires=1), DynOp(0.7, wires=2)], wires=0)
        op2 = HybridOp([DynOp(0.5, wires=1), DynOp(0.7, wires=2)], wires=0)
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_different_operator_leaves_not_equal(self):
        """Test that mismatched ``Operator2`` leaves produce an informative message."""
        op1 = HybridOp([DynOp(0.5, wires=1)], wires=0)
        op2 = HybridOp([DynOp(0.6, wires=1)], wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError) as err:
            qp.assert_equal(op1, op2)

        assert "different values for 'ops'" in str(err.value)
        assert "different values for 'phi'" in str(err.value)

    def test_different_pytree_structure_not_equal(self):
        """Test that hybrid arguments with different pytree structures compare unequal."""
        op1 = HybridOp([DynOp(0.5, wires=1)], wires=0)
        op2 = HybridOp([DynOp(0.5, wires=1), DynOp(0.7, wires=2)], wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'ops'"):
            qp.assert_equal(op1, op2)

    def test_operator_leaf_versus_non_operator_leaf(self):
        """Test that an operator leaf on one side and a non-operator leaf on the
        other (with matching pytree structure) compare unequal."""
        op1 = HybridOp([DynOp(0.5, wires=1)], wires=0)
        op2 = HybridOp([0.5], wires=0)

        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'ops'"):
            qp.assert_equal(op1, op2)

        # Check reverse order
        assert qp.equal(op2, op1) is False
        with pytest.raises(AssertionError, match="different values for 'ops'"):
            qp.assert_equal(op2, op1)

    def test_hybrid_wires(self):
        """Test that operators with a hybrid argument that is also a wire argument
        that has the same leaves are equal."""
        op1 = HybridWireOp([[0, 1], [2]])
        op2 = HybridWireOp([[0, 1], [2]])
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_hybrid_wires_different_wires(self):
        """Test that operators with a hybrid wire argument with different contents are unequal."""
        op1 = HybridWireOp([[0, 1], [2]])
        op2 = HybridWireOp([[0, 1], [3]])
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different wires for the 'pytree_wires' argument"):
            qp.assert_equal(op1, op2)

    def test_equal_with_numeric_leaves(self):
        """Test equality for hybrid arguments containing plain numeric leaves."""
        op1 = HybridOp([0.5, 1.0], wires=0)
        op2 = HybridOp([0.5, 1.0], wires=0)
        assert qp.equal(op1, op2) is True
        qp.assert_equal(op1, op2)

    def test_different_numeric_leaves_not_equal(self):
        """Test that operators with different numeric leaves in hybrid arguments are unequal."""
        op1 = HybridOp([0.5, 1.0], wires=0)
        op2 = HybridOp([0.5, 2.0], wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'ops'"):
            qp.assert_equal(op1, op2)

    def test_numeric_leaves_within_tolerance(self):
        """Test that operators with close numeric leaves in hybrid arguments are equal
        within tolerance."""
        op1 = HybridOp([0.5], wires=0)
        op2 = HybridOp([0.5 + 1e-10], wires=0)
        assert qp.equal(op1, op2, atol=1e-9, rtol=0) is True
        qp.assert_equal(op1, op2, atol=1e-9, rtol=0)

    def test_numeric_leaves_different_interfaces_not_equal(self):
        """Test that operators with the same value but different interfaces are unequal."""
        op1 = FullOp(0.5, label="", hybrid=[np.array(0.5)], wires=0)
        op2 = FullOp(0.5, label="", hybrid=[pnp.array(0.5, requires_grad=False)], wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different interfaces for 'hybrid'"):
            qp.assert_equal(op1, op2)

    def test_numeric_leaves_different_interfaces_equal_when_disabled(self):
        """Test that ``check_interface=False`` ignores interface differences."""
        op1 = FullOp(0.5, label="", hybrid=[np.array(0.5)], wires=0)
        op2 = FullOp(0.5, label="", hybrid=[pnp.array(0.5, requires_grad=False)], wires=0)
        assert qp.equal(op1, op2, check_interface=False) is True
        qp.assert_equal(op1, op2, check_interface=False)

    def test_numeric_leaves_different_trainability_not_equal(self):
        """Test that operators differing only in trainability are unequal."""
        op1 = FullOp(0.5, label="", hybrid=[pnp.array(0.5, requires_grad=True)], wires=0)
        op2 = FullOp(0.5, label="", hybrid=[pnp.array(0.5, requires_grad=False)], wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="differ in trainability for 'hybrid'"):
            qp.assert_equal(op1, op2)

    def test_numeric_leaves_different_trainability_equal_when_disabled(self):
        """Test that ``check_trainability=False`` ignores trainability differences."""
        op1 = FullOp(0.5, label="", hybrid=[pnp.array(0.5, requires_grad=True)], wires=0)
        op2 = FullOp(0.5, label="", hybrid=[pnp.array(0.5, requires_grad=False)], wires=0)
        assert qp.equal(op1, op2, check_trainability=False) is True
        qp.assert_equal(op1, op2, check_trainability=False)

    def test_op_leaves_within_tolerance(self):
        """Test that operators with operator leaves in hybrid arguments with dynamic
        data within tolerance are equal."""
        op1 = HybridOp([DynOp(0.5, wires=1)], wires=0)
        op2 = HybridOp([DynOp(0.5 + 1e-10, wires=1)], wires=0)
        assert qp.equal(op1, op2, atol=1e-9, rtol=0) is True
        qp.assert_equal(op1, op2, atol=1e-9, rtol=0)

    def test_scalar_versus_batched_hybrid_leaf_not_equal(self):
        """Test that operators with a scalar numeric leaf and a broadcast-compatible
        batched leaf in a hybrid argument are unequal.
        """
        op1 = HybridOp([0.5], wires=0)
        op2 = HybridOp([np.array([0.5, 0.5, 0.5])], wires=0)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different values for 'ops'"):
            qp.assert_equal(op1, op2)

    def test_nested_operator_trainability_propagates(self):
        """Test that operator-leaf trainability differences are reported."""
        op1 = HybridOp([DynOp(pnp.array(0.5, requires_grad=True), wires=1)], wires=0)
        op2 = HybridOp([DynOp(pnp.array(0.5, requires_grad=False), wires=1)], wires=0)
        assert qp.equal(op1, op2) is False
        assert qp.equal(op1, op2, check_trainability=False) is True
        qp.assert_equal(op1, op2, check_trainability=False)

    def test_nested_operator_interface_propagates(self):
        """Test that operator-leaf interface differences are reported."""
        op1 = HybridOp([DynOp(np.array(0.5), wires=1)], wires=0)
        op2 = HybridOp([DynOp(pnp.array(0.5, requires_grad=False), wires=1)], wires=0)
        assert qp.equal(op1, op2) is False
        assert qp.equal(op1, op2, check_interface=False) is True
        qp.assert_equal(op1, op2, check_interface=False)


class TestEqualFullOperator:
    """Tests for operators with all types of arguments."""

    def test_all_equal(self):
        """Test that operators with matching arguments across all groups are equal."""
        args = {"phi": 0, "label": "a", "hybrid": [], "wires": 0}
        op1 = FullOp(**args)
        op2 = FullOp(**args)
        assert qp.equal(op1, op2) is True

    @pytest.mark.parametrize(
        "diff, match",
        [
            ({"phi": 1.5}, "different values for 'phi'"),
            ({"label": "b"}, "different values for 'label'"),
            ({"hybrid": [DynOp(0.5, wires=2)]}, "different values for 'hybrid'"),
            ({"wires": 1}, "different wires"),
        ],
    )
    def test_each_group_difference_detected(self, diff, match):
        """Test that a difference in any single argument group is detected correctly."""
        args = {"phi": 0, "label": "a", "hybrid": [], "wires": 0}
        op1 = FullOp(**args)

        args.update(diff)
        op2 = FullOp(**args)

        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match=match):
            qp.assert_equal(op1, op2)


@pytest.mark.jax
class TestAbstractDynamicArgs:
    """Tests for checking for equality of traced dynamic arguments."""

    def test_abstract_dynamic_arg_not_equal(self):
        """Test that operators with traced dynamic arguments are unequal."""
        import jax

        result = jax.jit(qp.equal)(DynOp(0.5, wires=0), DynOp(0.5, wires=0))
        assert bool(result) is False

    def test_abstract_dynamic_arg_assert_equal_message(self):
        """Test that ``assert_equal`` on operators with traced dynamic arguments
        raises the correct error."""
        import jax

        with pytest.raises(AssertionError, match="tracer value for 'phi'"):
            jax.jit(qp.assert_equal)(DynOp(0.5, wires=0), DynOp(0.5, wires=0))

    def test_abstract_hybrid_leaf_not_equal(self):
        """Test that operators with traced numeric leaves inside hybrid arguments
        are unequal."""
        import jax

        def _check(x, y):
            return qp.equal(HybridOp([x], wires=0), HybridOp([y], wires=0))

        result = jax.jit(_check)(0.5, 0.5)
        assert bool(result) is False

    def test_abstract_hybrid_leaf_assert_equal(self):
        """Test that ``assert_equal`` on operators with traced numeric leaves
        inside hybrid arguments raises the correct error."""
        import jax

        def _check(x, y):
            qp.assert_equal(HybridOp([x], wires=0), HybridOp([y], wires=0))

        with pytest.raises(AssertionError, match="tracer value for 'ops'"):
            jax.jit(_check)(0.5, 0.5)


class ExtendedDynOp(DynOp):
    """A subclass of ``DynOp`` that adds an extra dynamic argument. An instance of
    this class passes ``isinstance(op, DynOp)``, but has different ``dynamic_argnames``."""

    dynamic_argnames = ("phi", "theta")

    def __init__(self, phi, theta, wires):
        # pylint: disable=non-parent-init-called,super-init-not-called
        Operator2.__init__(self, phi, theta, wires=wires)


class StaticDynOp(DynOp):
    """A subclass of ``DynOp`` that adds a static argument. An instance of this class
    passes ``isinstance(op, DynOp)``, but has different ``static_argnames``."""

    static_argnames = ("label",)

    def __init__(self, phi, label, wires):
        # pylint: disable=non-parent-init-called,super-init-not-called
        Operator2.__init__(self, phi, label, wires=wires)


class CompilableDynOp(DynOp):
    """A subclass of ``DynOp`` that adds a compilable argument. An instance of this class
    passes ``isinstance(op, DynOp)``, but has different ``compilable_argnames``."""

    compilable_argnames = ("n",)

    def __init__(self, phi, n, wires):
        # pylint: disable=non-parent-init-called,super-init-not-called
        Operator2.__init__(self, phi, n, wires=wires)


class WireDynOp(DynOp):
    """A subclass of ``DynOp`` that adds a wire argument. An instance of this class
    passes ``isinstance(op, DynOp)``, but has different ``wire_argnames``."""

    wire_argnames = ("wires", "ctrl_wires")

    def __init__(self, phi, wires, ctrl_wires):
        # pylint: disable=non-parent-init-called,super-init-not-called
        Operator2.__init__(self, phi, wires=wires, ctrl_wires=ctrl_wires)


class HybridDynOp(DynOp):
    """A subclass of ``DynOp`` that adds a hybrid argument. An instance of this class
    passes ``isinstance(op, DynOp)``, but has different ``hybrid_argnames``."""

    hybrid_argnames = ("ops",)

    def __init__(self, phi, ops, wires):
        # pylint: disable=non-parent-init-called,super-init-not-called
        Operator2.__init__(self, phi, ops, wires=wires)


class TestArgnames:
    """Tests that operators whose ``*_argnames`` differ are not equal.

    A subclass passes the ``isinstance(op2, type(op1))`` check used by ``qp.equal``,
    so two related operators may reach ``_equal_operator2`` with differing argnames.
    """

    def test_different_dynamic_argnames_not_equal(self):
        """Test that operators with different ``dynamic_argnames`` are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = ExtendedDynOp(0.5, 0.7, wires=0)
        # Sanity check: op2 is an instance of op1's type, so the top-level type
        # check in ``qp.equal`` passes and dispatch reaches ``_equal_operator2``.
        assert isinstance(op2, DynOp)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different 'dynamic_argnames'"):
            qp.assert_equal(op1, op2)

    def test_different_static_argnames_not_equal(self):
        """Test that operators with different ``static_argnames`` are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = StaticDynOp(0.5, "label", wires=0)
        assert isinstance(op2, DynOp)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different 'static_argnames'"):
            qp.assert_equal(op1, op2)

    def test_different_compilable_argnames_not_equal(self):
        """Test that operators with different ``compilable_argnames`` are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = CompilableDynOp(0.5, 3, wires=0)
        assert isinstance(op2, DynOp)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different 'compilable_argnames'"):
            qp.assert_equal(op1, op2)

    def test_different_wire_argnames_not_equal(self):
        """Test that operators with different ``wire_argnames`` are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = WireDynOp(0.5, wires=0, ctrl_wires=1)
        assert isinstance(op2, DynOp)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different 'wire_argnames'"):
            qp.assert_equal(op1, op2)

    def test_different_hybrid_argnames_not_equal(self):
        """Test that operators with different ``hybrid_argnames`` are unequal."""
        op1 = DynOp(0.5, wires=0)
        op2 = HybridDynOp(0.5, [], wires=0)
        assert isinstance(op2, DynOp)
        assert qp.equal(op1, op2) is False
        with pytest.raises(AssertionError, match="different 'hybrid_argnames'"):
            qp.assert_equal(op1, op2)
