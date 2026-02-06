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
Unit tests for the qp.simplify function
"""

# pylint: disable=too-few-public-methods
import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.tape import QuantumScript


def build_op():
    """Return function to build nested operator."""

    return qp.adjoint(
        qp.prod(
            qp.RX(1, 0) ** 1,
            qp.RY(1, 0),
            qp.prod(qp.adjoint(qp.PauliX(0)), qp.RZ(1, 0)),
            qp.RX(1, 0),
        )
    )


simplified_op = qp.prod(
    qp.RX(4 * np.pi - 1, 0),
    qp.RZ(4 * np.pi - 1, 0),
    qp.PauliX(0),
    qp.RY(4 * np.pi - 1, 0),
    qp.RX(4 * np.pi - 1, 0),
)


class TestSimplifyOperators:
    """Tests for the qp.simplify method used with operators."""

    def test_simplify_method_with_default_depth(self):
        """Test simplify method with default depth."""
        op = build_op()

        s_op = qp.simplify(op)
        assert isinstance(s_op, qp.ops.Prod)  # pylint: disable=no-member
        assert s_op.data == simplified_op.data
        assert s_op.wires == simplified_op.wires
        assert s_op.arithmetic_depth == simplified_op.arithmetic_depth

    def test_simplify_method_with_queuing(self):
        """Test the simplify method while queuing."""
        with qp.queuing.AnnotatedQueue() as q:
            op = build_op()
            s_op = qp.simplify(op)

        assert len(q) == 1
        assert q.queue[0] is s_op

    def test_simplify_unsupported_object_raises_error(self):
        """Test that an error is raised when trying to simplify an unsupported object."""
        with pytest.raises(ValueError, match="Cannot simplify the object"):
            qp.simplify("unsupported type")

    @pytest.mark.jax
    def test_jit_simplification(self):
        """Test that simplification can be jitted."""

        import jax

        sum_op = qp.sum(qp.PauliX(0), qp.PauliX(0))
        simp_op = jax.jit(qp.simplify)(sum_op)

        qp.assert_equal(
            simp_op, qp.s_prod(2.0, qp.PauliX(0)), check_interface=False, check_trainability=False
        )


class TestSimplifyTapes:
    """Tests for the qp.simplify method used with tapes."""

    @pytest.mark.parametrize("shots", [None, 100])
    def test_simplify_tape(self, shots):
        """Test the simplify method with a tape."""
        with qp.queuing.AnnotatedQueue() as q_tape:
            build_op()

        tape = QuantumScript.from_queue(q_tape, shots=shots)
        [s_tape], _ = qp.simplify(tape)
        assert len(s_tape) == 1
        s_op = s_tape[0]
        assert isinstance(s_op, qp.ops.Prod)  # pylint: disable=no-member
        assert s_op.data == simplified_op.data
        assert s_op.wires == simplified_op.wires
        assert s_op.arithmetic_depth == simplified_op.arithmetic_depth
        assert tape.shots == s_tape.shots

    def test_execute_simplified_tape(self):
        """Test the execution of a simplified tape."""
        dev = qp.device("default.qubit", wires=2)
        with qp.queuing.AnnotatedQueue() as q_tape:
            qp.prod(qp.prod(qp.PauliX(0) ** 1, qp.PauliX(0)), qp.PauliZ(1))
            qp.expval(op=qp.PauliZ(1))

        tape = QuantumScript.from_queue(q_tape)
        simplified_tape_op = qp.PauliZ(1)
        [s_tape], _ = qp.simplify(tape)
        s_op = s_tape.operations[0]
        assert isinstance(s_op, qp.PauliZ)
        assert s_op.data == simplified_tape_op.data
        assert s_op.wires == simplified_tape_op.wires
        assert s_op.arithmetic_depth == simplified_tape_op.arithmetic_depth
        assert dev.execute(tape) == dev.execute(s_tape)


class TestSimplifyQNodes:
    """Tests for the qp.simplify method used with qnodes."""

    def test_simplify_qnode(self):
        """Test the simplify method with a qnode."""
        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def qnode():
            qp.prod(qp.prod(qp.PauliX(0) ** 1, qp.PauliX(0)), qp.PauliZ(1))
            return qp.expval(
                op=qp.prod(qp.prod(qp.PauliX(0) ** 1, qp.PauliX(0)), qp.PauliZ(1))
            )

        simplified_tape_op = qp.PauliZ(1)

        s_qnode = qp.simplify(qnode)
        assert s_qnode() == qnode()

        tape = qp.workflow.construct_tape(s_qnode)()
        [s_tape], _ = s_qnode.transform_program([tape])
        assert len(s_tape) == 2

        s_op = s_tape.operations[0]
        s_obs = s_tape.observables[0]
        assert isinstance(s_op, qp.PauliZ)
        assert s_op.data == simplified_tape_op.data
        assert s_op.wires == simplified_tape_op.wires
        assert s_op.arithmetic_depth == simplified_tape_op.arithmetic_depth
        assert isinstance(s_obs, qp.PauliZ)
        assert s_obs.data == simplified_tape_op.data
        assert s_obs.wires == simplified_tape_op.wires
        assert s_obs.arithmetic_depth == simplified_tape_op.arithmetic_depth


class TestSimplifyCallables:
    """Tests for the qp.simplify method used with callables."""

    def test_simplify_qfunc(self):
        """Test the simplify method with a qfunc."""
        dev = qp.device("default.qubit", wires=2)

        def qfunc():
            qp.prod(qp.prod(qp.PauliX(0) ** 1, qp.PauliX(0)), qp.PauliZ(1))
            return qp.probs(wires=0)

        simplified_tape_op = qp.PauliZ(1)

        s_qfunc = qp.simplify(qfunc)

        qnode = qp.QNode(qfunc, dev)
        s_qnode = qp.QNode(s_qfunc, dev)

        assert (s_qnode() == qnode()).all()
        s_tape = qp.workflow.construct_tape(s_qnode)()
        assert len(s_tape) == 2
        s_op = s_tape.operations[0]
        assert isinstance(s_op, qp.PauliZ)
        assert s_op.data == simplified_tape_op.data
        assert s_op.wires == simplified_tape_op.wires
        assert s_op.arithmetic_depth == simplified_tape_op.arithmetic_depth

    @pytest.mark.jax
    def test_jitting_simplified_qfunc(self):
        """Test that we can jit qnodes that have a simplified quantum function."""

        import jax

        @jax.jit
        @qp.qnode(qp.device("default.qubit", wires=1))
        @qp.simplify
        def circuit(x):
            qp.adjoint(qp.RX(x, wires=0))
            _ = qp.PauliX(0) ** 2
            return qp.expval(qp.PauliY(0))

        x = jax.numpy.array(4 * jax.numpy.pi + 0.1)
        res = circuit(x)
        assert qp.math.allclose(res, jax.numpy.sin(x))

        grad = jax.grad(circuit)(x)
        assert qp.math.allclose(grad, jax.numpy.cos(x))
