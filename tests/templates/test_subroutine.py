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
"""Tests for Subroutine and SubroutineOp"""
# pylint: disable=unused-argument
import inspect
from functools import partial

import pytest

import pennylane as qml
from pennylane.templates import Subroutine, SubroutineOp


class TestInitialization:

    def test_default_creation(self):
        """Test the creation of a simple subroutine."""

        def Simple(x, wires):
            qml.RX(x, wires=wires)

        S = Subroutine(Simple)

        assert S.name == "Simple"
        assert S.signature == inspect.signature(Simple)

        a, k = S.setup_inputs("a", wires="b")
        assert a == ("a",)
        assert k == {"wires": "b"}

        assert S.static_argnames == tuple()
        assert S.wire_argnames == ("wires",)
        assert S.dynamic_argnames == ("x",)

        with qml.queuing.AnnotatedQueue() as q:
            S.definition(0.5, 0)

        assert len(q.queue) == 1
        qml.assert_equal(q.queue[0], qml.RX(0.5, 0))

    def test_wire_argnames(self):
        """Test that wire argnames can be specified."""

        def WireArgnames(x, reg1, reg2, y):
            pass

        S = Subroutine(WireArgnames, wire_argnames=("reg1", "reg2"))

        assert S.wire_argnames == (
            "reg1",
            "reg2",
        )
        assert S.dynamic_argnames == ("x", "y")

    def test_static_argnames_and_setup_inputs(self):
        """Test static argnames and setup_inputs can be provided."""

        def setup_inputs(x, y, wires, z):
            return (x, y), {"wires": wires, "z": tuple(z)}

        def T(x, y, wires, z):
            pass

        S = Subroutine(T, setup_inputs=setup_inputs, static_argnames=("z",))
        assert S.static_argnames == ("z",)
        assert S.dynamic_argnames == (
            "x",
            "y",
        )

        a, k = S.setup_inputs(0.5, 1.2, "a", [1, 2, 3])
        assert a == (0.5, 1.2)
        assert k == {"wires": "a", "z": (1, 2, 3)}

    def test_string_static_argnames_static_argnums(self):
        """Test that if a string is provided to static argnums or static argnames, it's wrapped in a tuple."""

        def f(metadata, register):
            pass

        S = Subroutine(f, static_argnames="metadata", wire_argnames="register")

        assert S.wire_argnames == ("register",)
        assert S.static_argnames == ("metadata",)


def Example1(x, y, reg1, reg2, pauli_words):
    qml.PauliRot(x, pauli_words[0], reg1)
    qml.PauliRot(y, pauli_words[1], reg2)
    return 2


def Example1SetupInputs(x, y, reg1, reg2, pauli_words):
    return (x, y, reg1, reg2, tuple(pauli_words)), {}


Example1Subroutine = Subroutine(
    Example1,
    wire_argnames=("reg1", "reg2"),
    static_argnames=("pauli_words",),
    setup_inputs=Example1SetupInputs,
)


def generate_subroutine_op_example(*args, **kwargs):
    bound_args = Example1Subroutine.signature.bind(*args, **kwargs)

    with qml.queuing.AnnotatedQueue() as q:
        output = Example1(*args, **kwargs)
    return SubroutineOp(Example1Subroutine, bound_args, decomposition=q.queue, output=output)


class TestSubroutineOp:

    op1 = generate_subroutine_op_example(
        0.5, 0.6, qml.wires.Wires((0, "a")), qml.wires.Wires(("a", 1)), ("XY", "YZ")
    )

    def test_no_primitive_bind_call(self):
        """Test that SubroutineOp has no _primitive_bind_call implementation."""

        with pytest.raises(ValueError, match="SubroutineOp's should never be directly"):
            SubroutineOp._primitive_bind_call()  # pylint: disable=protected-access

    def test_isinstance_with_Subroutine(self):
        """Test that SubroutineOp's are instances of their corresponding Subroutine."""

        # pylint: disable=isinstance-second-argument-not-valid-type
        assert isinstance(self.op1, Example1Subroutine)

    def test_basic_validity(self):
        """Test that subroutine op passes basic validity checks."""

        # What do we want for the behaviour of deep copy? Do we want to copy
        # the function as well?
        qml.ops.functions.assert_valid(self.op1, skip_pickle=True, skip_deepcopy=True)

    def test_repr(self):
        """Test that SubroutineOp has a clean repr."""

        assert (
            repr(self.op1)
            == "<Example1(x=0.5, y=0.6, reg1=Wires([0, 'a']), reg2=Wires(['a', 1]), pauli_words=('XY', 'YZ'))>"
        )

    def test_set_wires(self):
        """Test that the wires for a subroutineop are set."""
        assert self.op1.wires == qml.wires.Wires([0, "a", 1])

    def test_set_data(self):
        """Test that data can automatically be extracted."""
        assert self.op1.data == (0.5, 0.6)

    def test_output(self):
        """Test accessing the output."""
        assert self.op1.output == 2

    def test_label(self):
        """Test that the label uses the name of the subroutine itself."""

        assert self.op1.label() == "Example1"

    def test_accessing_subroutine(self):
        """Test the subroutine can be accessed."""
        assert self.op1.subroutine == Example1Subroutine

    def test_map_wires(self):
        """Test that map_wires can handle multiple registers."""

        new_op = self.op1.map_wires({0: 3, "a": 4})
        assert new_op.wires == qml.wires.Wires([3, 4, 1])
        decomp = new_op.decomposition()
        qml.assert_equal(decomp[0], qml.PauliRot(0.5, "XY", (3, 4)))
        qml.assert_equal(decomp[1], qml.PauliRot(0.6, "YZ", (4, 1)))

    def test_decomposition(self):
        """Test that decomposition behaves like normal."""

        with qml.queuing.AnnotatedQueue() as q:
            decomp = self.op1.decomposition()

        decomp2 = self.op1.decomposition()  # no queuing context

        for d in [q.queue, decomp, decomp2]:
            qml.assert_equal(d[0], qml.PauliRot(0.5, "XY", (0, "a")))
            qml.assert_equal(d[1], qml.PauliRot(0.6, "YZ", ("a", 1)))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_decompose_integration(self):
        """Test that SubroutineOp can be used with the decompose transform in a basic use case."""

        tape = qml.tape.QuantumScript([self.op1])

        [new_tape], _ = qml.transforms.decompose(tape, gate_set={qml.PauliRot})

        qml.assert_equal(new_tape[0], qml.PauliRot(0.5, "XY", (0, "a")))
        qml.assert_equal(new_tape[1], qml.PauliRot(0.6, "YZ", ("a", 1)))


class TestSubroutineCall:

    def test_setup_inputs_run(self):
        """Test that setup_inputs is run and can make user inputs hashable."""

        def my_setup_inputs(wires, pauli_words):
            return (), {"wires": wires, "pauli_words": tuple(pauli_words)}

        @partial(Subroutine, setup_inputs=my_setup_inputs)
        def f(wires, pauli_words):
            assert isinstance(pauli_words, tuple)
            qml.PauliRot(0.5, pauli_words[0], wires)

        op = f(0, ["X"])
        assert op.bound_args.arguments["pauli_words"] == ("X",)

    def test_fill_in_default_values(self):
        """Test that we will in default values when binding."""

        @partial(Subroutine, static_argnames="metadata")
        def f(wires, metadata="default_value"):
            pass


@pytest.mark.capture
class TestSubroutineCapture:

    def test_setup_inputs_program_capture(self):
        """Test that setup_inputs can make inputs hashable for use with program capture."""

        import jax  # pylint: disable=import-outside-toplevel

        def my_setup_inputs(wires, pauli_words):
            return (), {"wires": wires, "pauli_words": tuple(pauli_words)}

        @partial(Subroutine, setup_inputs=my_setup_inputs, static_argnames="pauli_words")
        def f(wires, pauli_words):
            assert isinstance(pauli_words, tuple)
            qml.PauliRot(0.5, pauli_words[0], wires)

        def w():
            f(0, ["X"])

        jaxpr = jax.make_jaxpr(w)()
        assert jaxpr.eqns[-1].primitive == qml.capture.primitives.quantum_subroutine_prim
        inner_jaxpr = jaxpr.eqns[-1].params["jaxpr"]
        # pylint: disable=protected-access
        assert inner_jaxpr.eqns[-1].primitive == qml.PauliRot._primitive
        assert inner_jaxpr.eqns[-1].params["pauli_word"] == "X"

    def test_different_forms_of_wires(self):
        """Test that wires can be provided as literal integers, traced integers, lists, tuple, and arrays."""

        import jax

        @qml.templates.Subroutine
        def f(wires):
            assert wires.shape == (1,)
            qml.X(wires[0])

        jaxpr1 = jax.make_jaxpr(f)(0)

        def w1():
            return f(0)

        jaxpr2 = jax.make_jaxpr(w1)()

        jaxpr3 = jax.make_jaxpr(f)([0])
        jaxpr4 = jax.make_jaxpr(f)((0,))

        def w2():
            return f([0])

        jaxpr5 = jax.make_jaxpr(w2)()

        for jaxpr in [jaxpr1, jaxpr2, jaxpr3, jaxpr4, jaxpr5]:
            assert jaxpr.eqns[-1].primitive == qml.capture.primitives.quantum_subroutine_prim

            assert jaxpr.eqns[-1].invars[0].aval.shape == (1,)
            assert "int" in jaxpr.eqns[-1].invars[0].aval.dtype.name
