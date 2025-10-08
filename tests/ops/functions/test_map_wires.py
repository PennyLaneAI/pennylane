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
Unit tests for the qml.map_wires function
"""

# pylint: disable=too-few-public-methods
from functools import partial

import pytest

import pennylane as qml
from pennylane.ops import Prod
from pennylane.tape import QuantumScript
from pennylane.wires import Wires


def build_op():
    """Return function to build nested operator."""

    return qml.prod(
        qml.RX(1, 0) ** 1,
        qml.RY(1, 1),
        qml.prod(qml.adjoint(qml.PauliX(2)), qml.RZ(1, 3)),
        qml.RX(1, 4),
    )


wire_map = {0: 4, 1: 3, 3: 1, 4: 0}

mapped_op = qml.prod(
    qml.RX(1, 4) ** 1,
    qml.RY(1, 3),
    qml.prod(qml.adjoint(qml.PauliX(2)), qml.RZ(1, 1)),
    qml.RX(1, 0),
)


class TestMapWiresOperators:
    """Tests for the qml.map_wires method used with operators."""

    def test_map_wires_with_operator(self):
        """Test the map_wires method with an operator."""
        op = build_op()

        m_op = qml.map_wires(op, wire_map=wire_map)
        assert isinstance(m_op, qml.ops.Prod)  # pylint:disable=no-member
        assert m_op.data == mapped_op.data
        assert m_op.wires == mapped_op.wires
        assert m_op.arithmetic_depth == mapped_op.arithmetic_depth

    def test_map_wires_without_queuing(self):
        """Test the map_wires method while queuing with `queue = False`."""
        with qml.queuing.AnnotatedQueue() as q_tape:
            op = build_op()
            _ = qml.map_wires(op, wire_map=wire_map, queue=False)
        tape = QuantumScript.from_queue(q_tape)
        assert len(tape.circuit) == 1
        assert tape.circuit[0] is op
        assert q_tape.get_info(op).get("owner", None) is None

    def test_map_wires_with_queuing_and_without_replacing(self):
        """Test the map_wires method while queuing with `queue=True` and `replace=False`."""
        with qml.queuing.AnnotatedQueue() as q_tape:
            op = build_op()
            m_op = qml.map_wires(op, wire_map=wire_map, queue=True, replace=False)
        tape = QuantumScript.from_queue(q_tape)
        assert len(tape.circuit) == 2
        assert tape.circuit[0] is op
        assert tape.circuit[1] is m_op
        assert q_tape.get_info(op).get("owner", None) is None

    def test_map_wires_with_queuing_and_with_replacing(self):
        """Test the map_wires method while queuing with `queue = True` and `replace=True`."""
        with qml.queuing.AnnotatedQueue() as q:
            op = build_op()
            m_op = qml.map_wires(op, wire_map=wire_map, queue=True, replace=True)

        assert len(q) == 1
        assert q.queue[0] is m_op

    def test_map_wires_unsupported_object_raises_error(self):
        """Test that an error is raised when trying to map the wires of an unsupported object."""
        with pytest.raises(qml.transforms.core.TransformError, match="Decorating a QNode with"):
            qml.map_wires("unsupported type", wire_map=wire_map)


class TestMapWiresTapes:
    """Tests for the qml.map_wires method used with tapes."""

    @pytest.mark.parametrize("shots", [None, 100])
    def test_map_wires_tape(self, shots):
        """Test the map_wires method with a tape."""

        with qml.queuing.AnnotatedQueue() as q_tape:
            build_op()
            qml.expval(op=qml.PauliZ(1))

        tape = QuantumScript.from_queue(q_tape, shots=shots)
        tape.trainable_params = [0, 2]

        [s_tape], _ = qml.map_wires(tape, wire_map=wire_map)
        assert len(s_tape) == 2
        assert s_tape.trainable_params == [0, 2]
        assert s_tape.shots == tape.shots
        # check ops
        s_op = s_tape[0]
        qml.assert_equal(s_op, mapped_op)
        # check observables
        s_obs = s_tape.observables
        assert len(s_obs) == 1
        assert s_obs[0].wires == Wires(wire_map[1])

    def test_map_wires_batch(self):
        """Test that map_wires can be applied to a batch of tapes."""

        t1 = qml.tape.QuantumScript([qml.X(0)], [qml.expval(qml.Z(0))])
        t2 = qml.tape.QuantumScript([qml.Y(1)], [qml.probs(wires=1)])

        batch, _ = qml.map_wires((t1, t2), {0: "a", 1: "b"})

        expected1 = qml.tape.QuantumScript([qml.X("a")], [qml.expval(qml.Z("a"))])
        expected2 = qml.tape.QuantumScript([qml.Y("b")], [qml.probs(wires="b")])
        qml.assert_equal(batch[0], expected1)
        qml.assert_equal(batch[1], expected2)


class TestMapWiresQNodes:
    """Tests for the qml.map_wires method used with qnodes."""

    def test_map_wires_qnode(self):
        """Test the map_wires method with a qnode."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def qnode():
            build_op()
            return qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)))

        mapped_obs = qml.prod(qml.PauliX(4), qml.PauliY(3), qml.PauliZ(2))

        m_qnode = qml.map_wires(qnode, wire_map=wire_map)
        assert m_qnode() == qnode()
        m_tape = qml.workflow.construct_tape(m_qnode)()
        assert len(m_tape) == 2

        m_op = m_tape.operations
        m_obs = m_tape.observables
        qml.assert_equal(m_op[0], mapped_op)
        qml.assert_equal(m_obs[0], mapped_obs)


class TestMapWiresCallables:
    """Tests for the qml.map_wires method used with callables."""

    def test_map_wires_qfunc(self):
        """Test the map_wires method with a qfunc."""
        dev = qml.device("default.qubit", wires=5)

        def qfunc():
            build_op()
            qml.prod(qml.PauliX(0), qml.PauliY(1))
            return qml.probs(wires=0), qml.probs(wires=1)

        m_qfunc = qml.map_wires(qfunc, wire_map=wire_map)
        mapped_op_2 = qml.prod(qml.PauliX(4), qml.PauliY(3))
        qnode = qml.QNode(qfunc, dev)
        m_qnode = qml.QNode(m_qfunc, dev)
        m_tape = qml.workflow.construct_tape(m_qnode)()

        assert qml.math.allclose(m_qnode(), qnode())
        assert len(m_tape) == 4
        m_ops = m_tape.operations
        assert isinstance(m_ops[0], Prod)
        assert isinstance(m_ops[1], Prod)
        qml.assert_equal(m_ops[0], mapped_op)
        qml.assert_equal(m_ops[1], mapped_op_2)
        assert m_tape.observables[0].wires == Wires(wire_map[0])
        assert m_tape.observables[1].wires == Wires(wire_map[1])

    @pytest.mark.jax
    def test_jitting_simplified_qfunc(self):
        """Test that we can jit qnodes that have a mapped quantum function."""
        import jax

        @jax.jit
        @partial(qml.map_wires, wire_map=wire_map)
        @qml.qnode(qml.device("default.qubit", wires=5))
        def circuit(x):
            qml.adjoint(qml.RX(x, wires=0))
            _ = qml.PauliX(0) ** 2
            return qml.expval(qml.PauliY(0))

        x = jax.numpy.array(4 * jax.numpy.pi + 0.1)
        res = circuit(x)
        assert qml.math.allclose(res, jax.numpy.sin(x))

        grad = jax.grad(circuit)(x)
        assert qml.math.allclose(grad, jax.numpy.cos(x))
