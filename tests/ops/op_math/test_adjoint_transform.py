# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import numpy as np

import pennylane as qml
from pennylane import adjoint
from pennylane.ops.op_math import Adjoint

noncallable_objects = [
    [qml.Hadamard(1), qml.RX(-0.2, wires=1)],
    qml.tape.QuantumTape(),
]


@pytest.mark.parametrize("obj", noncallable_objects)
def test_error_adjoint_on_noncallable(obj):
    """Test that an error is raised if qml.adjoint is applied to an object that
    is not callable, as it silently does not have any effect on those."""
    with pytest.raises(ValueError, match=f"{type(obj)} is not callable."):
        adjoint(obj)


class TestPreconstructedOp:
    """Test providing an already initalized operator to the transform."""

    @pytest.mark.parametrize(
        "base", (qml.IsingXX(1.23, wires=("c", "d")), qml.QFT(wires=(0, 1, 2)))
    )
    def test_single_op(self, base):
        """Test passing a single preconstructed op in a queuing context."""
        with qml.tape.QuantumTape() as tape:
            base.queue()
            out = adjoint(base)

        assert isinstance(out, Adjoint)
        assert out.base is base
        assert len(tape) == 1
        assert len(tape._queue) == 2
        assert tape._queue[base] == {"owner": out}
        assert tape._queue[out] == {"owns": base}

    def test_single_op_defined_outside_queue_eager(self):
        """Test if base is defined outside context and the function eagerly simplifies
        the adjoint, the base is not added to queue."""
        base = qml.RX(1.2, wires=0)
        with qml.tape.QuantumTape() as tape:
            out = adjoint(base, lazy=False)

        assert isinstance(out, qml.RX)
        assert out.data == [-1.2]
        assert len(tape) == 1
        assert tape[0] is out

        assert len(tape._queue) == 1
        assert tape._queue[out] == {"owns": base}

    def test_single_observable(self):
        """Test passing a single preconstructed observable in a queuing context."""

        with qml.tape.QuantumTape() as tape:
            base = qml.PauliX(0) @ qml.PauliY(1)
            out = adjoint(base)

        assert len(tape) == 0
        assert out.base is base
        assert isinstance(out, Adjoint)


class TestDifferentCallableTypes:
    """Test the adjoint transform on a variety of possible inputs."""

    def test_adjoint_single_op_function(self):
        """Test the adjoint transform on a single operation."""

        with qml.tape.QuantumTape() as tape:
            out = adjoint(qml.RX)(1.234, wires="a")

        assert out == tape[0]
        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qml.RX
        assert out.data == [1.234]
        assert out.wires == qml.wires.Wires("a")

    def test_adjoint_template(self):
        """Test the adjoint transform on a template."""

        with qml.tape.QuantumTape() as tape:
            out = adjoint(qml.QFT)(wires=(0, 1, 2))

        assert len(tape) == 1
        assert out == tape[0]
        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qml.QFT
        assert out.wires == qml.wires.Wires((0, 1, 2))

    def test_adjoint_on_function(self):
        """Test adjoint transform on a function"""

        def func(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(z, wires=0)

        x = 1.23
        y = 2.34
        z = 3.45
        with qml.tape.QuantumTape() as tape:
            out = adjoint(func)(x, y, z)

        assert out == tape.circuit

        for op in tape:
            assert isinstance(op, Adjoint)

        # check order reversed
        assert tape[0].base.__class__ is qml.RZ
        assert tape[1].base.__class__ is qml.RY
        assert tape[2].base.__class__ is qml.RX

        # check parameters assigned correctly
        assert tape[0].data == [z]
        assert tape[1].data == [y]
        assert tape[2].data == [x]

    def test_nested_adjoint(self):
        """Test the adjoint transform on an adjoint transform."""
        x = 4.321
        with qml.tape.QuantumTape() as tape:
            out = adjoint(adjoint(qml.RX))(x, wires="b")

        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert isinstance(out.base, Adjoint)
        assert out.base.base.__class__ is qml.RX
        assert out.data == [x]
        assert out.wires == qml.wires.Wires("b")


class TestNonLazyExecution:
    """Test the lazy=False keyword."""

    def test_single_decomposeable_op(self):
        """Test lazy=False for a single op that gets decomposed."""

        x = 1.23
        with qml.tape.QuantumTape() as tape:
            base = qml.RX(x, wires="b")
            out = adjoint(base, lazy=False)

        assert len(tape) == 1
        assert out is tape[0]
        assert tape._queue[base] == {"owner": out}
        assert tape._queue[out] == {"owns": base}

        assert isinstance(out, qml.RX)
        assert out.data == [-1.23]

    def test_single_nondecomposable_op(self):
        """Test lazy=false for a single op that can't be decomposed."""
        with qml.tape.QuantumTape() as tape:
            base = qml.S(0)
            out = adjoint(base, lazy=False)

        assert len(tape) == 1
        assert out is tape[0]
        assert len(tape._queue) == 2
        assert tape._queue[base] == {"owner": out}
        assert tape._queue[out] == {"owns": base}

        assert isinstance(out, Adjoint)
        assert isinstance(out.base, qml.S)

    def test_single_decomposable_op_function(self):
        """Test lazy=False for a single op callable that gets decomposed."""
        x = 1.23
        with qml.tape.QuantumTape() as tape:
            out = adjoint(qml.RX, lazy=False)(x, wires="b")

        assert out is tape[0]
        assert not isinstance(out, Adjoint)
        assert isinstance(out, qml.RX)
        assert out.data == [-x]

    def test_single_nondecomposable_op_function(self):
        """Test lazy=False for a single op function that can't be decomposed."""
        with qml.tape.QuantumTape() as tape:
            out = adjoint(qml.S, lazy=False)(0)

        assert out is tape[0]
        assert isinstance(out, Adjoint)
        assert isinstance(out.base, qml.S)

    def test_mixed_function(self):
        """Test lazy=False with a function that applies operations of both types."""
        x = 1.23

        def qfunc(x):
            qml.RZ(x, wires="b")
            qml.T("b")

        with qml.tape.QuantumTape() as tape:
            out = adjoint(qfunc, lazy=False)(x)

        assert len(tape) == len(out) == 2
        assert isinstance(tape[0], Adjoint)
        assert isinstance(tape[0].base, qml.T)

        assert isinstance(tape[1], qml.RZ)
        assert tape[1].data[0] == -x


class TestOutsideofQueuing:
    """Test the behaviour of the adjoint transform when not called in a queueing context."""

    def test_single_op(self):
        """Test providing a single op outside of a queuing context."""

        x = 1.234
        out = adjoint(qml.RZ(x, wires=0))

        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qml.RZ
        assert out.data == [1.234]
        assert out.wires == qml.wires.Wires(0)

    def test_single_op_eager(self):
        """Test a single op that can be decomposed in eager mode outside of a queuing context."""

        x = 1.234
        base = qml.RX(x, wires=0)
        out = adjoint(base, lazy=False)

        assert isinstance(out, qml.RX)
        assert out.data == [-x]

    def test_observable(self):
        """Test providing a preconstructed Observable outside of a queuing context."""

        base = 1.0 * qml.PauliX(0)
        obs = adjoint(base)

        assert isinstance(obs, Adjoint)
        assert isinstance(obs, qml.operation.Observable)
        assert obs.base is base

    def test_single_op_function(self):
        """Test the transform on a single op as a callable outside of a queuing context."""
        x = 1.234
        out = adjoint(qml.IsingXX)(x, wires=(0, 1))

        assert isinstance(out, Adjoint)
        assert out.base.__class__ is qml.IsingXX
        assert out.data == [1.234]
        assert out.wires == qml.wires.Wires((0, 1))

    def test_function(self):
        """Test the transform on a function outside of a queuing context."""

        def func(wire):
            qml.S(wire)
            qml.SX(wire)

        wire = 1.234
        out = adjoint(func)(wire)

        assert len(out) == 2
        assert all(isinstance(op, Adjoint) for op in out)
        assert all(op.wires == qml.wires.Wires(wire) for op in out)

    def test_nonlazy_op_function(self):
        """Test non-lazy mode on a simplifiable op outside of a queuing context."""

        out = adjoint(qml.PauliX, lazy=False)(0)

        assert not isinstance(out, Adjoint)
        assert isinstance(out, qml.PauliX)


class TestIntegration:
    """Test circuit execution and gradients with the adjoint transform."""

    def test_single_op(self):
        """Test the adjoint of a single op against analytically expected results."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circ():
            qml.PauliX(0)
            adjoint(qml.S)(0)
            return qml.state()

        res = circ()
        expected = np.array([0, -1j])

        assert np.allclose(res, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_autograd(self, diff_method):
        """Test gradients through the adjoint transform with autograd."""
        import autograd

        @qml.qnode(
            qml.device("default.qubit", wires=1), diff_method=diff_method, interface="autograd"
        )
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = autograd.numpy.array(0.234)
        expected_res = np.sin(x)
        expected_grad = np.cos(x)
        assert qml.math.allclose(circ(x), expected_res)
        assert qml.math.allclose(autograd.grad(circ)(x), expected_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_jax(self, diff_method):
        """Test gradients through the adjoint transform with jax."""
        import jax

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method, interface="jax")
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = jax.numpy.array(0.234)
        expected_res = jax.numpy.sin(x)
        expected_grad = jax.numpy.cos(x)
        assert qml.math.allclose(circ(x), expected_res)
        assert qml.math.allclose(jax.grad(circ)(x), expected_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_torch(self, diff_method):
        """Test gradients through the adjoint transform with torch."""
        import torch

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method, interface="torch")
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = torch.tensor(0.234, requires_grad=True)
        y = circ(x)
        y.backward()

        assert qml.math.allclose(y, torch.sin(x))
        assert qml.math.allclose(x.grad, torch.cos(x))

    @pytest.mark.tf
    @pytest.mark.parametrize("diff_method", ("backprop", "adjoint", "parameter-shift"))
    def test_gradient_torch(self, diff_method):
        """Test gradients through the adjoint transform with tensorflow."""

        import tensorflow as tf

        @qml.qnode(qml.device("default.qubit", wires=1), diff_method=diff_method, interface="tf")
        def circ(x):
            adjoint(qml.RX)(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = tf.Variable(0.234)
        with tf.GradientTape() as tape:
            y = circ(x)

        grad = tape.gradient(y, x)

        assert qml.math.allclose(y, tf.sin(x))
        assert qml.math.allclose(grad, tf.cos(x))
