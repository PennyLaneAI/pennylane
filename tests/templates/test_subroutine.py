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

        assert S.static_argnames == frozenset()
        assert S.wire_argnames == frozenset({"wires"})
        assert S.dynamic_argnames == frozenset({"x"})

        with qml.queuing.AnnotatedQueue() as q:
            S.definition(0.5, 0)

        assert len(q.queue) == 1
        qml.assert_equal(q.queue[0], qml.RX(0.5, 0))

    def test_wire_argnames(self):
        """Test that wire argnames can be specified."""

        def WireArgnames(x, reg1, reg2, y):
            pass

        S = Subroutine(WireArgnames, wire_argnames={"reg1", "reg2"})

        assert S.wire_argnames == frozenset({"reg1", "reg2"})
        assert S.dynamic_argnames == frozenset({"x", "y"})

    def test_static_argnames_and_setup_inputs(self):
        """Test static argnames and setup_inputs can be provided."""

        def setup_inputs(x, y, wires, z):
            return (x, y), {"wires": wires, "z": tuple(z)}

        def T(x, y, wires, z):
            pass

        S = Subroutine(T, setup_inputs=setup_inputs, static_argnames={"z"})
        assert S.static_argnames == frozenset({"z"})
        assert S.dynamic_argnames == frozenset({"x", "y"})

        a, k = S.setup_inputs(0.5, 1.2, "a", [1, 2, 3])
        assert a == (0.5, 1.2)
        assert k == {"wires": "a", "z": (1, 2, 3)}


def Example1(x, y, reg1, reg2, pauli_words):
    qml.PauliRot(x, pauli_words[0], reg1)
    qml.PauliRot(y, pauli_words[1], reg2)
    return 2


def Example1SetupInputs(x, y, reg1, reg2, pauli_words):
    return (x, y, reg1, reg2, tuple(pauli_words)), {}


Example1Subroutine = Subroutine(
    Example1,
    wire_argnames={"reg1", "reg2"},
    static_argnames={"pauli_words"},
    setup_inputs=Example1SetupInputs,
)


def generate_subroutine_op_example(*args, **kwargs):
    bound_args = Example1Subroutine.signature.bind(*args, **kwargs)

    with qml.queuing.AnnotatedQueue() as q:
        output = Example1(*args, **kwargs)
    return SubroutineOp(Example1Subroutine, bound_args, decomposition=q.queue, output=output)


class TestSubroutineOp:

    op1 = generate_subroutine_op_example(0.5, 0.6, [0, "a"], ["a", 1], ["XY", "YZ"])

    def test_no_primitive_bind_call(self):
        """Test that SubroutineOp has no _primitive_bind_call implementation."""

        with pytest.raises(ValueError, match="SubroutineOp's should never be directly"):
            SubroutineOp._primitive_bind_call()  # pylint: disable=protected-access

    def test_set_wires(self):
        """Test that the wires for a subroutineop are set."""
        assert self.op1.wires == qml.wires.Wires([0, "a", 1])

    def test_set_data(self):
        """Test that data can automatically be extracted."""
        assert self.op1.data == (0.5, 0.6)

    def test_output(self):
        """Test accessing the output."""
        assert self.op1.output == 2

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
