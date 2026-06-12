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
"""Tests for Operator2's metaclass."""

import pytest
from operator2_utils import DynOp, MultiWireOp, TwoDynOp

import pennylane as qp
from pennylane.core.operator import Operator2
from pennylane.core.operator.meta import (
    ArgType,
    _canonicalize_abstract_type,
    _canonicalize_wire_leaf,
    _contains_abstract_type,
)
from pennylane.core.operator.operator2 import operator_p
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractWires, Wires

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]


@pytest.mark.parametrize(
    "leaf, expected",
    [
        (0, AbstractWires(1)),
        ("a", AbstractWires(1)),
        (Wires([0, 1, 2]), AbstractWires(3)),
        ([0, 1], AbstractWires(2)),
        ((0, 1, 2, 3), AbstractWires(4)),
        (range(5), AbstractWires(5)),
    ],
)
def test_wire_leaf_maps_to_abstract_wires(leaf, expected):
    """Direct tests for '_canonicalize_wire_leaf'."""

    assert _canonicalize_wire_leaf(leaf) == expected


class TestCanonicalizeAbstractTypeHelper:
    """Tests '_canonicalize_abstract_type' helper."""

    # =========================================================================
    # Unit tests when 'kind=ArgType.WIRES'
    # =========================================================================

    @pytest.mark.parametrize(
        "val, expected",
        [
            (0, AbstractWires(1)),
            ([0, 1], AbstractWires(2)),
            (Wires([0, 1, 2]), AbstractWires(3)),
        ],
    )
    def test_concrete_wires_are_promoted(self, val, expected):
        """Tests that concrete wires are promoted to abstract."""
        assert _canonicalize_abstract_type(val, kind=ArgType.WIRES) == expected

    def test_abstract_wires_pass_through(self):
        """Tests that already abstract wires are passed through unchanged."""

        assert _canonicalize_abstract_type(AbstractWires(5), kind=ArgType.WIRES) == AbstractWires(5)

    # =========================================================================
    # Unit tests when 'kind=ArgType.DYN'
    # =========================================================================

    @pytest.mark.parametrize(
        "val, expected",
        [
            (0, AbstractArray((), int)),
            ([0.0, 1.0], AbstractArray((2,), float)),
            ([[0], [1], [2]], AbstractArray((3, 1), int)),
        ],
    )
    def test_concrete_inputs_are_promoted_when_kind_is_dyn(self, val, expected):
        """Tests that inputs are properly canonicalized."""

        assert _canonicalize_abstract_type(val, kind=ArgType.DYN) == expected

    # =========================================================================
    # Unit tests when 'kind=ArgType.HYBRID'
    # =========================================================================

    @pytest.mark.parametrize(
        "val, expected",
        [
            (0, AbstractArray((), int)),
            ([0.0, 1.0], [AbstractArray((), float), AbstractArray((), float)]),
            (
                {"a": [0, 1], "b": 1.5},
                {
                    "a": [AbstractArray((), int), AbstractArray((), int)],
                    "b": AbstractArray((), float),
                },
            ),
        ],
    )
    def test_concrete_inputs_are_promoted_when_kind_is_hybrid(self, val, expected):
        """Tests that inputs are properly canonicalized."""

        assert _canonicalize_abstract_type(val, kind=ArgType.HYBRID) == expected


class TestContainsAbstractTypeHelper:
    """Tests the '_contains_abstract_type' helper."""

    @pytest.mark.parametrize(
        "val",
        [
            AbstractArray((1,), float),
            AbstractWires(2),
            [0.5, AbstractArray((1,), int)],
            {"a": AbstractWires(1)},
            (1.0, [AbstractArray((1,), float)]),
        ],
    )
    def test_abstract_leaves_are_flagged(self, val):
        """Tests that pytrees with at least one abstract value return True."""

        assert _contains_abstract_type(val)

    @pytest.mark.parametrize(
        "val",
        [
            0.5,
            [0.5, 1.0],
            {"a": 1, "b": 2},
            Wires([0, 1, 2]),
            (1.0, 2.0),
        ],
    )
    def test_concrete_leaves_are_not_flagged(self, val):
        """Tests that pytrees containing purely concrete values are not falgged."""

        assert not _contains_abstract_type(val)

    def test_jax_tracer_is_not_flagged(self):
        """Tests that JAX tracers are not considered 'abstract'."""

        captured = {}

        def f(x):
            captured["is_abstract"] = _contains_abstract_type(x)

        _ = jax.make_jaxpr(f)(0.5)
        assert captured["is_abstract"] is False


class DynCanonOp(Operator2):  # pylint: disable=too-few-public-methods
    """Operator with a dynamic parameter and wires that performs canonicalization."""

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(2 * phi, wires)


class TestOperatorConcreteInputs:
    """Tests that the child constructor is run and the primitive is bound if inputs are concrete."""

    def test_child_constructor_runs_when_concrete(self):
        """Tests a concrete input will trigger the child's constructor."""

        op = DynCanonOp(phi=2.0, wires=0)
        # __init__ is hit so phi is doubled
        assert op.phi == 4.0
        assert op.wires == Wires(0)

    def test_concrete_inputs_triggers_bind(self):
        """Tests that a concrete construction under capture will bind the primitive."""

        cjaxpr = jax.make_jaxpr(lambda x: DynOp(x, wires=0))(2.0)
        relevant_eqns = [e for e in cjaxpr.eqns if e.primitive is operator_p]
        assert len(relevant_eqns) == 1


class TestOperatorAbstractInputs:
    """Tests that the metaclass canonicalizes abstract operators."""

    def test_child_init_is_skipped(self):
        """Tests that the child constructor is skipped."""

        aa = AbstractArray((1,), float)
        op = DynCanonOp(phi=aa, wires=[0])
        assert op.phi is aa
        # The 0 got replaced with a AbstractWires
        assert op.wires == AbstractWires(1)

    def test_bind_isnt_trigger_for_abstract_array(self):
        """Tests that no operator equation enters the jaxpr for abstract inputs."""

        def f():
            DynCanonOp(phi=AbstractArray((1,), float), wires=0)

        cjaxpr = jax.make_jaxpr(f)()
        relevant_eqns = [e for e in cjaxpr.eqns if e.primitive is operator_p]
        assert len(relevant_eqns) == 0

    @pytest.mark.parametrize(
        "concrete_theta, abstract_theta",
        [
            (1.0, AbstractArray((), float)),
            (qp.numpy.ones((2, 3)), AbstractArray((2, 3), float)),
            ([0, 1], AbstractArray((2,), int)),
        ],
    )
    @pytest.mark.parametrize(
        "concrete_wires, abstract_wires",
        [(0, AbstractWires(1)), ([0], AbstractWires(1)), ([0, 1], AbstractWires(2))],
    )
    def test_canonicalize_all_inputs_when_some_are_abstract(
        self, concrete_theta, abstract_theta, concrete_wires, abstract_wires
    ):
        """Tests that it takes at least one abstract argument to skip the init and canonicalize inputs."""

        aa = AbstractArray((1,), int)
        op = TwoDynOp(phi=aa, theta=concrete_theta, wires=concrete_wires)
        # __init__ is hit so phi is doubled
        assert op.phi is aa
        assert op.theta == abstract_theta
        assert op.wires == abstract_wires

    def test_abstract_wires_skips_init(self):
        """Tests that the presence of abstract wires also skips init."""

        aw = Wires[1]
        op = MultiWireOp(wires=aw, ctrl_wires=0)
        assert op.wires == AbstractWires(2)

    def test_bind_isnt_trigger_for_abstract_wires(self):
        """Tests that no operator equation enters the jaxpr for abstract wires."""

        def f():
            MultiWireOp(Wires[1], 0)

        cjaxpr = jax.make_jaxpr(f)()
        relevant_eqns = [e for e in cjaxpr.eqns if e.primitive is operator_p]
        assert len(relevant_eqns) == 0

    def test_mixed_arg_op_correctly_abstractifies_arguments(self):
        """Tests that different types of arguments canonicalize differently."""

        class MixedArgOp(Operator2):
            """Operator with static, dynamic and hybrid argnames."""

            static_argnames = ("static_arg",)
            dynamic_argnames = ("dynamic_arg",)
            hybrid_argnames = ("hybrid_arg",)

            def __init__(self, static_arg, dynamic_arg, hybrid_arg, wires):
                super().__init__(static_arg, dynamic_arg, hybrid_arg, wires)

        op = MixedArgOp(
            static_arg="blah", dynamic_arg=[0, 1], hybrid_arg=[0, 1], wires=AbstractWires(1)
        )
        assert op.static_arg == "blah"
        assert op.dynamic_arg == AbstractArray((2,), int)
        assert op.hybrid_arg == [AbstractArray((), int), AbstractArray((), int)]
        assert op.wires == AbstractWires(1)
