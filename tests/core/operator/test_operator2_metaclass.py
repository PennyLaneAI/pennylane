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

from pennylane.core.operator import Operator2
from pennylane.core.operator.capture import contains_abstract_type
from pennylane.core.operator.operator2 import operator_p
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractWires, Wires

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]


class TestContainsAbstractTypeHelper:
    """Tests the 'contains_abstract_type' helper."""

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

        assert contains_abstract_type(val)

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

        assert not contains_abstract_type(val)

    def test_jax_tracer_is_not_flagged(self):
        """Tests that JAX tracers are not considered 'abstract'."""

        captured = {}

        def f(x):
            captured["is_abstract"] = contains_abstract_type(x)

        _ = jax.make_jaxpr(f)(0.5)
        assert captured["is_abstract"] is False


class DynCanonOp(Operator2):
    """Operator with a dynamic parameter and wires that performs canonicalization."""

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(2 * phi, wires)


class TestConcreteInputs:
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


class TestAbstractInputs:
    """Tests that the Operator2 constructor is run if abstract inputs are detected."""

    def test_child_init_is_skipped(self):
        """Tests that the child constructor is skipped."""

        aa = AbstractArray((1,), float)
        op = DynCanonOp(phi=aa, wires=0)
        assert op.phi is aa
        assert op.wires == Wires(0)

    def test_bind_isnt_trigger_for_abstract_array(self):
        """Tests that no operator equation enters the jaxpr for abstract inputs."""

        def f():
            DynCanonOp(phi=AbstractArray((1,), float), wires=0)

        cjaxpr = jax.make_jaxpr(f)()
        relevant_eqns = [e for e in cjaxpr.eqns if e.primitive is operator_p]
        assert len(relevant_eqns) == 0

    def test_some_inputs_are_abstract(self):
        """Tests that it takes at least one abstract argument to skip the init."""

        aa = AbstractArray((1,), float)
        op = TwoDynOp(phi=aa, theta=1.0, wires=0)
        # __init__ is hit so phi is doubled
        assert op.phi is aa
        assert op.theta == 1.0
        assert op.wires == Wires(0)

    def test_abstract_wires_skips_init(self):
        """Tests that the presence of abstract wires also skips init."""

        aw = Wires[1]
        op = MultiWireOp(wires=aw, ctrl_wires=0)
        assert op.wires[0] is aw
        assert op.wires[1] == 0

    def test_bind_isnt_trigger_for_abstract_wires(self):
        """Tests that no operator equation enters the jaxpr for abstract wires."""

        def f():
            MultiWireOp(Wires[1], 0)

        cjaxpr = jax.make_jaxpr(f)()
        relevant_eqns = [e for e in cjaxpr.eqns if e.primitive is operator_p]
        assert len(relevant_eqns) == 0
