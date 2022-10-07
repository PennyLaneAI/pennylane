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
import pytest

import pennylane as qml
from pennylane.tape import QuantumTape
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
        assert isinstance(m_op, qml.ops.Prod)
        assert m_op.data == mapped_op.data
        assert m_op.wires == mapped_op.wires
        assert m_op.arithmetic_depth == mapped_op.arithmetic_depth

    def test_map_wires_without_queuing(self):
        """Test the map_wires method while queuing with `queue = False`."""
        tape = QuantumTape()
        with tape:
            op = build_op()
            _ = qml.map_wires(op, wire_map=wire_map, queue=False)
        assert len(tape.circuit) == 1
        assert tape.circuit[0] is op
        assert tape._queue[op].get("owner", None) is None

    def test_map_wires_with_queuing_and_without_replacing(self):
        """Test the map_wires method while queuing with `queue=True` and `replace=False`."""
        tape = QuantumTape()
        with tape:
            op = build_op()
            m_op = qml.map_wires(op, wire_map=wire_map, queue=True, replace=False)
        assert len(tape.circuit) == 2
        assert tape.circuit[0] is op
        assert tape.circuit[1] is m_op
        assert tape._queue[op].get("owner", None) is None

    def test_map_wires_with_queuing_and_with_replacing(self):
        """Test the map_wires method while queuing with `queue = True` and `replace=True`."""
        tape = QuantumTape()
        with tape:
            op = build_op()
            m_op = qml.map_wires(op, wire_map=wire_map, queue=True, replace=True)
        assert len(tape.circuit) == 1
        assert tape.circuit[0] is m_op
        assert tape._queue[op].get("owner", None) is m_op

    def test_map_wires_unsupported_object_raises_error(self):
        """Test that an error is raised when trying to map the wires of an unsupported object."""
        with pytest.raises(ValueError, match="Cannot map wires of object"):
            qml.map_wires("unsupported type", wire_map=wire_map)


class TestMapWiresTapes:
    """Tests for the qml.map_wires method used with tapes."""

    def test_map_wires_tape(self):
        """Test the map_wires method with a tape."""
        tape = QuantumTape()
        with tape:
            build_op()

        # TODO: Use qml.equal when supported

        s_tape = qml.map_wires(tape, wire_map=wire_map)
        assert len(s_tape) == 1
        s_op = s_tape[0]
        assert isinstance(s_op, qml.ops.Prod)
        assert s_op.data == mapped_op.data
        assert s_op.wires == mapped_op.wires
        assert s_op.arithmetic_depth == mapped_op.arithmetic_depth

    def test_execute_mapped_tape(self):
        """Test the execution of a mapped tape."""
        dev = qml.device("default.qubit", wires=5)
        tape = QuantumTape()
        with tape:
            build_op()
            qml.expval(op=qml.PauliZ(1))

        # TODO: Use qml.equal when supported

        m_tape = qml.map_wires(tape, wire_map=wire_map)
        m_op = m_tape.operations[0]
        m_obs = m_tape.observables[0]
        assert isinstance(m_op, qml.ops.Prod)
        assert m_op.data == mapped_op.data
        assert m_op.wires == mapped_op.wires
        assert m_op.arithmetic_depth == mapped_op.arithmetic_depth
        assert m_obs.wires == Wires(wire_map[1])
        assert qml.math.allclose(dev.execute(tape), dev.execute(m_tape))


class TestMapWiresQNodes:
    """Tests for the qml.map_wires method used with qnodes."""

    def test_map_wires_qnode(self):
        """Test the map_wires method with a qnode."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def qnode():
            build_op()
            return qml.expval(op=build_op())

        # TODO: Use qml.equal when supported

        m_qnode = qml.map_wires(qnode, wire_map=wire_map)
        assert m_qnode() == qnode()
        assert len(m_qnode.tape) == 2
        m_op = m_qnode.tape.operations[0]
        m_obs = m_qnode.tape.observables[0]
        assert isinstance(m_op, qml.ops.Prod)
        assert m_op.data == mapped_op.data
        assert m_op.wires == mapped_op.wires
        assert m_op.arithmetic_depth == mapped_op.arithmetic_depth
        assert isinstance(m_obs, qml.ops.Prod)
        assert m_obs.data == mapped_op.data
        assert m_obs.wires == mapped_op.wires
        assert m_obs.arithmetic_depth == mapped_op.arithmetic_depth


class TestMapWiresCallables:
    """Tests for the qml.map_wires method used with callables."""

    def test_map_wires_qfunc(self):
        """Test the map_wires method with a qfunc."""
        dev = qml.device("default.qubit", wires=5)

        def qfunc():
            build_op()
            return qml.probs(wires=0)

        m_qfunc = qml.map_wires(qfunc, wire_map=wire_map)

        qnode = qml.QNode(qfunc, dev)
        m_qnode = qml.QNode(m_qfunc, dev)

        # TODO: Use qml.equal when supported

        assert (m_qnode() == qnode()).all()
        assert len(m_qnode.tape) == 2
        m_op = m_qnode.tape.operations[0]
        assert isinstance(m_op, qml.ops.Prod)
        assert m_op.data == mapped_op.data
        assert m_op.wires == mapped_op.wires
        assert m_op.arithmetic_depth == mapped_op.arithmetic_depth
        assert m_qnode.tape.observables[0].wires == Wires(wire_map[0])

    @pytest.mark.jax
    def test_jitting_simplified_qfunc(self):
        """Test that we can jit qnodes that have a mapped quantum function."""
        # TODO: Support @qml.map_wires(wire_map) decorator

        import jax

        @qml.qnode(qml.device("default.qubit.jax", wires=5), interface="jax")
        def circuit(x):
            qml.adjoint(qml.RX(x, wires=0))
            qml.PauliX(0) ** 2
            return qml.expval(qml.PauliY(0))

        m_circuit = jax.jit(qml.map_wires(circuit, wire_map=wire_map))

        x = jax.numpy.array(4 * jax.numpy.pi + 0.1)
        res = m_circuit(x)
        assert qml.math.allclose(res, jax.numpy.sin(x))

        grad = jax.grad(m_circuit)(x)
        assert qml.math.allclose(grad, jax.numpy.cos(x))
