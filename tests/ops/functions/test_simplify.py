# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the qml.simplify function
"""
# pylint: disable=too-few-public-methods
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.tape import QuantumScript


def build_op():
    """Return function to build nested operator."""

    return qml.adjoint(
        qml.prod(
            qml.RX(1, 0) ** 1,
            qml.RY(1, 0),
            qml.prod(qml.adjoint(qml.PauliX(0)), qml.RZ(1, 0)),
            qml.RX(1, 0),
        )
    )


simplified_op = qml.prod(
    qml.RX(4 * np.pi - 1, 0),
    qml.RZ(4 * np.pi - 1, 0),
    qml.PauliX(0),
    qml.RY(4 * np.pi - 1, 0),
    qml.RX(4 * np.pi - 1, 0),
)


class TestSimplifyOperators:
    """Tests for the qml.simplify method used with operators."""

    def test_simplify_method_with_default_depth(self):
        """Test simplify method with default depth."""
        op = build_op()

        s_op = qml.simplify(op)
        assert isinstance(s_op, qml.ops.Prod)  # pylint: disable=no-member
        assert s_op.data == simplified_op.data
        assert s_op.wires == simplified_op.wires
        assert s_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_method_with_queuing(self):
        """Test the simplify method while queuing."""
        with qml.queuing.AnnotatedQueue() as q:
            op = build_op()
            s_op = qml.simplify(op)

        assert len(q) == 1
        assert q.queue[0] is s_op

    def test_simplify_unsupported_object_raises_error(self):
        """Test that an error is raised when trying to simplify an unsupported object."""
        with pytest.raises(ValueError, match="Cannot simplify the object"):
            qml.simplify("unsupported type")

    @pytest.mark.jax
    def test_jit_simplification(self):
        """Test that simplification can be jitted."""

        import jax

        sum_op = qml.sum(qml.PauliX(0), qml.PauliX(0))
        simp_op = jax.jit(qml.simplify)(sum_op)

        qml.assert_equal(
            simp_op, qml.s_prod(2.0, qml.PauliX(0)), check_interface=False, check_trainability=False
        )


class TestSimplifyTapes:
    """Tests for the qml.simplify method used with tapes."""

    @pytest.mark.parametrize("shots", [None, 100])
    def test_simplify_tape(self, shots):
        """Test the simplify method with a tape."""
        with qml.queuing.AnnotatedQueue() as q_tape:
            build_op()

        tape = QuantumScript.from_queue(q_tape, shots=shots)
        [s_tape], _ = qml.simplify(tape)
        assert len(s_tape) == 1
        s_op = s_tape[0]
        assert isinstance(s_op, qml.ops.Prod)  # pylint: disable=no-member
        assert s_op.data == simplified_op.data
        assert s_op.wires == simplified_op.wires
        assert s_op.arithmetic_depth == simplified_op.arithmetic_depth
        assert tape.shots == s_tape.shots

    def test_execute_simplified_tape(self):
        """Test the execution of a simplified tape."""
        dev = qml.device("default.qubit", wires=2)
        with qml.queuing.AnnotatedQueue() as q_tape:
            qml.prod(qml.prod(qml.PauliX(0) ** 1, qml.PauliX(0)), qml.PauliZ(1))
            qml.expval(op=qml.PauliZ(1))

        tape = QuantumScript.from_queue(q_tape)
        simplified_tape_op = qml.PauliZ(1)
        [s_tape], _ = qml.simplify(tape)
        s_op = s_tape.operations[0]
        assert isinstance(s_op, qml.PauliZ)
        assert s_op.data == simplified_tape_op.data
        assert s_op.wires == simplified_tape_op.wires
        assert s_op.arithmetic_depth == simplified_tape_op.arithmetic_depth
        assert dev.execute(tape) == dev.execute(s_tape)


class TestSimplifyQNodes:
    """Tests for the qml.simplify method used with qnodes."""

    def test_simplify_qnode(self):
        """Test the simplify method with a qnode."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def qnode():
            qml.prod(qml.prod(qml.PauliX(0) ** 1, qml.PauliX(0)), qml.PauliZ(1))
            return qml.expval(
                op=qml.prod(qml.prod(qml.PauliX(0) ** 1, qml.PauliX(0)), qml.PauliZ(1))
            )

        simplified_tape_op = qml.PauliZ(1)

        s_qnode = qml.simplify(qnode)
        assert s_qnode() == qnode()

        [s_tape], _ = s_qnode.transform_program([s_qnode.tape])
        assert len(s_tape) == 2

        s_op = s_tape.operations[0]
        s_obs = s_tape.observables[0]
        assert isinstance(s_op, qml.PauliZ)
        assert s_op.data == simplified_tape_op.data
        assert s_op.wires == simplified_tape_op.wires
        assert s_op.arithmetic_depth == simplified_tape_op.arithmetic_depth
        assert isinstance(s_obs, qml.PauliZ)
        assert s_obs.data == simplified_tape_op.data
        assert s_obs.wires == simplified_tape_op.wires
        assert s_obs.arithmetic_depth == simplified_tape_op.arithmetic_depth


class TestSimplifyCallables:
    """Tests for the qml.simplify method used with callables."""

    def test_simplify_qfunc(self):
        """Test the simplify method with a qfunc."""
        dev = qml.device("default.qubit", wires=2)

        def qfunc():
            qml.prod(qml.prod(qml.PauliX(0) ** 1, qml.PauliX(0)), qml.PauliZ(1))
            return qml.probs(wires=0)

        simplified_tape_op = qml.PauliZ(1)

        s_qfunc = qml.simplify(qfunc)

        qnode = qml.QNode(qfunc, dev)
        s_qnode = qml.QNode(s_qfunc, dev)

        assert (s_qnode() == qnode()).all()
        assert len(s_qnode.tape) == 2
        s_op = s_qnode.tape.operations[0]
        assert isinstance(s_op, qml.PauliZ)
        assert s_op.data == simplified_tape_op.data
        assert s_op.wires == simplified_tape_op.wires
        assert s_op.arithmetic_depth == simplified_tape_op.arithmetic_depth

    @pytest.mark.jax
    def test_jitting_simplified_qfunc(self):
        """Test that we can jit qnodes that have a simplified quantum function."""

        import jax

        @jax.jit
        @qml.qnode(qml.device("default.qubit", wires=1))
        @qml.simplify
        def circuit(x):
            qml.adjoint(qml.RX(x, wires=0))
            _ = qml.PauliX(0) ** 2
            return qml.expval(qml.PauliY(0))

        x = jax.numpy.array(4 * jax.numpy.pi + 0.1)
        res = circuit(x)
        assert qml.math.allclose(res, jax.numpy.sin(x))

        grad = jax.grad(circuit)(x)
        assert qml.math.allclose(grad, jax.numpy.cos(x))
