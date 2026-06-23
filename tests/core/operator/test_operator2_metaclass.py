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

import numpy as np
import pytest
from operator2_utils import DynOp, MultiWireOp, TwoDynOp

import pennylane as qp
from pennylane.core.operator import Operator2
from pennylane.core.operator.operator2 import operator_p
from pennylane.queuing import AnnotatedQueue
from pennylane.typing import AbstractArray, AbstractWires
from pennylane.wires import Wires

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]


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
        # Make sure the operator primitive is in thie JAXPR
        assert len([e for e in cjaxpr.eqns if e.primitive is operator_p]) == 1


class TestOperatorAbstractInputs:
    """Tests that the metaclass canonicalizes abstract operators."""

    def test_child_init_is_skipped(self):
        """Tests that the child constructor is skipped."""

        # only AbstractArray

        aa = AbstractArray((1,), float)
        op = DynCanonOp(phi=aa, wires=[0])
        assert isinstance(op.phi, AbstractArray)
        assert op.wires == AbstractWires(1)

        # only AbstractWires

        aw = AbstractWires(1)
        op = MultiWireOp(wires=aw, ctrl_wires=0)
        assert op.wires == AbstractWires(2)

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
        [
            (0, AbstractWires(1)),
            (0.0, AbstractWires(1)),
            ([0], AbstractWires(1)),
            ([0, 1], AbstractWires(2)),
            ({"a": 0, "b": 1}, AbstractWires(2)),
            ("a", AbstractWires(1)),
            ("blah", AbstractWires(1)),
            (Wires([0, 1, 2]), AbstractWires(3)),
        ],
    )
    def test_canonicalize_all_inputs_when_some_are_abstract(
        self, concrete_theta, abstract_theta, concrete_wires, abstract_wires
    ):
        """Tests that it takes at least one abstract argument to skip the init and canonicalize inputs."""

        aa = AbstractArray((1,), int)
        op = TwoDynOp(phi=aa, theta=concrete_theta, wires=concrete_wires)
        # Phi stays the same, theta and wires are abstractified
        assert op.phi is aa
        assert op.theta == abstract_theta
        assert op.wires == abstract_wires

    @pytest.mark.parametrize(
        "hybrid_in, hybrid_out",
        [
            # Standard pytrees
            (0, AbstractArray((), int)),
            ([0.0, 1.0], [AbstractArray((), float), AbstractArray((), float)]),
            ((0.0, 1.0), (AbstractArray((), float), AbstractArray((), float))),
            (
                {"a": [0, 1], "b": 1.5},
                {
                    "a": [AbstractArray((), int), AbstractArray((), int)],
                    "b": AbstractArray((), float),
                },
            ),
            # Ensures nested arrays don't get flattened
            (
                {"my_array": qp.math.array([[0, 1], [1, 0]], dtype=int)},
                {"my_array": AbstractArray((2, 2), int)},
            ),
        ],
    )
    def test_mixed_arg_op(self, hybrid_in, hybrid_out):
        """Tests that different types of arguments canonicalize differently."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            static_argnames = ("static_arg",)
            dynamic_argnames = ("dynamic_arg",)
            hybrid_argnames = ("hybrid_arg",)

            def __init__(self, static_arg, dynamic_arg, hybrid_arg, wires):
                super().__init__(static_arg, dynamic_arg, hybrid_arg, wires=wires)

        op = MixedArgOp(
            static_arg="blah",
            dynamic_arg=[0, 1],
            hybrid_arg=hybrid_in,
            wires=AbstractWires(1),
        )
        assert op.static_arg == "blah"
        # Cast to abstract array
        assert op.dynamic_arg == AbstractArray((2,), int)
        assert op.hybrid_arg == hybrid_out

    @pytest.mark.parametrize(
        "hybrid_in, hybrid_out",
        [
            # Ensures nested abstract wires are handled
            (
                {"aw1": AbstractWires(2), "aw2": AbstractWires(1)},
                {"aw1": AbstractWires(2), "aw2": AbstractWires(1)},
            ),
            # Ensures wires as hybrid args are handled
            (
                {"reg1": Wires([0, 1]), "reg2": Wires([2, 3])},
                {"reg1": AbstractWires(2), "reg2": AbstractWires(2)},
            ),
            (
                qp.registers({"a": 2, "b": 3}),
                {"a": AbstractWires(2), "b": AbstractWires(3)},
            ),
        ],
    )
    def test_op_with_hybrid_wires(self, hybrid_in, hybrid_out):
        """Tests that different types of arguments canonicalize differently."""

        class HybridWiresOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            hybrid_argnames = ("hybrid_arg",)
            # Hybrid arg is a static structure of Wires objects
            wire_argnames = ("hybrid_arg", "wires")

            def __init__(self, hybrid_arg, wires):
                super().__init__(hybrid_arg, wires=wires)

        op = HybridWiresOp(
            hybrid_arg=hybrid_in,
            wires=AbstractWires(1),
        )
        assert op.hybrid_arg == hybrid_out

    @pytest.mark.parametrize(
        "hybrid_wires, base_wires, expected_wires",
        (
            (
                {"reg1": Wires([0, 1]), "reg2": Wires([2, 3, 4])},
                AbstractWires(1),
                AbstractWires(6),  # 2 + 3 + 1
            ),
            (
                {"areg1": AbstractWires(2), "areg2": AbstractWires(3)},
                AbstractWires(1),
                AbstractWires(6),  # 2 + 3 + 1
            ),
            (
                Wires([0, 1]),
                AbstractWires(3),
                AbstractWires(5),  # 3 + 2
            ),
            (
                qp.registers({"alice": 2, "bob": 4}),
                AbstractWires(5),
                AbstractWires(11),  # 5 + 2 + 4
            ),
        ),
    )
    def test_operator_correctly_calculates_total_abstract_wires(
        self, hybrid_wires, base_wires, expected_wires
    ):
        """Tests that the final op.wires is the sum of all abstract wires."""

        class WireTrackingOp(Operator2):  # pylint: disable=too-few-public-methods
            hybrid_argnames = ("hybrid_wires",)
            wire_argnames = ("wires", "hybrid_wires")

            def __init__(self, hybrid_wires, wires):
                super().__init__(hybrid_wires, wires=wires)

        op = WireTrackingOp(hybrid_wires, base_wires)
        assert op.wires == expected_wires

    def test_abstract_operator_doesnt_queue(self):
        """Ensures that an abstract operator doesn't get queued."""

        with AnnotatedQueue() as q:
            op = DynOp(0.5, wires=AbstractWires(1))

        assert op.wires == AbstractWires(1)
        assert op.phi == AbstractArray((), float)

        assert len(q) == 0

    @pytest.mark.parametrize("builtin_type", (float, int, bool, complex, np.float32, np.int_))
    def test_abstract_operator_construction_with_python_builtin_types(self, builtin_type):
        """Tests that you can construct an abstract operator with builtin Python types."""

        class MixedArgOp(Operator2):  # pylint: disable=too-few-public-methods
            """Operator with static, dynamic and hybrid argnames."""

            dynamic_argnames = ("dynamic_arg",)
            hybrid_argnames = ("hybrid_arg",)

            def __init__(self, dynamic_arg, hybrid_arg, wires):
                super().__init__(dynamic_arg, hybrid_arg, wires=wires)

        op = MixedArgOp(builtin_type, [builtin_type, builtin_type], 0)
        assert op.dynamic_arg == AbstractArray((), builtin_type)
        assert op.hybrid_arg == [AbstractArray((), builtin_type), AbstractArray((), builtin_type)]

    @pytest.mark.parametrize("input", ([float], [float, float]))
    def test_array_of_types_for_dynamic_arg(self, input):
        """Tests that a NotImplementedError gets raised if an array of types is used as input for dynamic arg."""

        with pytest.raises(
            NotImplementedError,
            match="sequence of types for a dynamic argument is not currently supported",
        ):
            _ = DynOp(input, AbstractWires(1))


class TestOperatorAbstractInputsCapture:
    """Tests the capture of operators with abstract inputs."""

    def test_bind_isnt_triggered_for_abstract_wires(self):
        """Tests that no operator equation enters the jaxpr for abstract wires."""

        def f():
            MultiWireOp(AbstractWires(1), 0)

        cjaxpr = jax.make_jaxpr(f)()
        assert len([e for e in cjaxpr.eqns if e.primitive is operator_p]) == 0

    def test_bind_isnt_triggered_for_abstract_array(self):
        """Tests that no operator equation enters the jaxpr for abstract inputs."""

        def f():
            DynCanonOp(phi=AbstractArray((1,), float), wires=0)

        cjaxpr = jax.make_jaxpr(f)()
        # Empty JAXPR
        assert len([e for e in cjaxpr.eqns if e.primitive is operator_p]) == 0
