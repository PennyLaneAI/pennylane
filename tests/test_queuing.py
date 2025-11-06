# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`QueuingManager` class.
"""
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import QueuingError
from pennylane.queuing import AnnotatedQueue, QueuingManager, WrappedObj


# pylint: disable=use-implicit-booleaness-not-comparison, unnecessary-dunder-call
class TestStopRecording:
    """Test the stop_recording method of QueuingManager."""

    def test_stop_recording_on_function_inside_QNode(self):
        """Test that the stop_recording transform when applied to a function
        is not recorded by a QNode"""
        dev = qml.device("default.qubit", wires=1)

        @QueuingManager.stop_recording()
        def my_op():
            return [qml.RX(0.123, wires=0), qml.RY(2.32, wires=0), qml.RZ(1.95, wires=0)]

        res = []

        @qml.qnode(dev)
        def my_circuit():
            res.extend(my_op())
            return qml.expval(qml.PauliZ(0))

        tape = qml.workflow.construct_tape(my_circuit)()

        assert len(tape.operations) == 0
        assert len(res) == 3

    def test_stop_recording_directly_on_op(self):
        """Test that stop_recording transform works when directly applied to an op"""
        dev = qml.device("default.qubit", wires=1)
        res = []

        @qml.qnode(dev)
        def my_circuit():
            op1 = QueuingManager.stop_recording()(qml.RX)(np.pi / 4.0, wires=0)
            op2 = qml.RY(np.pi / 4.0, wires=0)
            res.extend([op1, op2])
            return qml.expval(qml.PauliZ(0))

        tape = qml.workflow.construct_tape(my_circuit)()

        assert len(tape.operations) == 1
        assert tape.operations[0] == res[1]
        assert len(res) == 2

    def test_nested_stop_recording_on_function(self):
        """Test that stop_recording works when nested with other stop_recordings"""

        @QueuingManager.stop_recording()
        @QueuingManager.stop_recording()
        def my_op():
            return [
                qml.RX(0.123, wires=0),
                qml.RY(2.32, wires=0),
                qml.RZ(1.95, wires=0),
            ]

        # the stop_recording function will still work outside of any queuing contexts
        res = my_op()
        assert len(res) == 3

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def my_circuit():
            my_op()

            with QueuingManager.stop_recording():
                qml.PauliX(wires=0)
                my_op()

            qml.Hadamard(wires=0)
            my_op()
            return qml.state()

        tape = qml.workflow.construct_tape(my_circuit)()

        assert len(tape.operations) == 1
        assert tape.operations[0].name == "Hadamard"

    def test_stop_recording_qnode(self):
        """A stop_recording QNode is unaffected"""
        dev = qml.device("default.qubit", wires=1)

        @QueuingManager.stop_recording()
        @qml.qnode(dev)
        def my_circuit():
            qml.RX(np.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = my_circuit()
        assert result == -1.0

    def test_stop_recording_within_tape_cleans_up(self):
        """Test if some error is raised within a stop_recording context, the previously
        active contexts are still returned to avoid popping from an empty deque"""

        with pytest.raises(ValueError):
            with AnnotatedQueue():
                with QueuingManager.stop_recording():
                    raise ValueError


class TestQueuingManager:
    """Test the logic associated with the QueuingManager class."""

    def test_append_no_context(self):
        """Test that append does not fail when no context is present."""

        QueuingManager.append(qml.PauliZ(0))

    def test_remove_no_context(self):
        """Test that remove does not fail when no context is present."""

        QueuingManager.remove(qml.PauliZ(0))

    def test_no_active_context(self):
        """Test that if there are no active contexts, active_context() returns None"""
        assert QueuingManager.active_context() is None


class TestAnnotatedQueue:
    """Tests for the annotated queue class"""

    def test_remove_not_in_queue(self):
        """Test that remove passes silently if the object is not in the queue"""

        q2 = AnnotatedQueue()
        q2.remove(qml.PauliX(0))

    def test_append_qubit_gates(self):
        """Test that gates are successfully appended to the queue."""
        with AnnotatedQueue() as q:
            ops = [
                qml.RX(0.5, wires=0),
                qml.RY(-10.1, wires=1),
                qml.CNOT(wires=[0, 1]),
                qml.PhaseShift(-1.1, wires=18),
                qml.T(wires=99),
            ]
        assert q.queue == ops

    def test_append_qubit_observables(self):
        """Test that ops that are also observables are successfully
        appended to the queue."""
        with AnnotatedQueue() as q:
            # wire repetition is deliberate, Queue contains no checks/logic
            # for circuits
            ops = [
                qml.Hadamard(wires=0),
                qml.PauliX(wires=1),
                qml.PauliY(wires=1),
                qml.Hermitian(np.ones([2, 2]), wires=7),
            ]
        assert q.queue == ops

    def test_append_prod_ops_overloaded(self):
        """Test that Prod ops created using `@`
        are successfully added to the queue, as well as the `Prod` object."""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            prod_op = A @ B
        assert q.queue == [prod_op]
        assert prod_op.operands == (A, B)

    def test_get_info(self):
        """Test that get_info correctly returns an annotation"""
        A = qml.RZ(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(A, inv=True)

        assert q.get_info(A) == {"inv": True}

    def test_get_info_error(self):
        """Test that an exception is raised if get_info is called
        for a non-existent object"""

        with AnnotatedQueue() as q:
            qml.PauliZ(0)

        B = qml.PauliY(1)

        with pytest.raises(QueuingError, match="not in the queue"):
            q.get_info(B)

    def test_get_info_none(self):
        """Test that get_info returns None if there is no active queuing context"""
        A = qml.RZ(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(A, inv=True)

        assert QueuingManager.get_info(A) is None

    def test_update_info(self):
        """Test that update_info correctly updates an annotation"""
        A = qml.RZ(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(A, inv=True)
            assert QueuingManager.get_info(A) == {"inv": True}

            QueuingManager.update_info(A, key="value1")

        # should pass silently because no longer recording
        QueuingManager.update_info(A, key="value2")

        assert q.get_info(A) == {"inv": True, "key": "value1"}

        q.update_info(A, inv=False, owner=None)
        assert q.get_info(A) == {"inv": False, "owner": None, "key": "value1"}

    def test_update_info_not_in_queue(self):
        """Test that no exception is raised if get_info is called
        for a non-existent object."""

        with AnnotatedQueue() as q:
            qml.PauliZ(0)

        B = qml.PauliY(1)

        q.update_info(B, inv=True)
        assert len(q.queue) == 1

    def test_parallel_queues_are_isolated(self):
        """Tests that parallel queues do not queue each other's constituents."""
        q1 = AnnotatedQueue()
        q2 = AnnotatedQueue()
        n = 10000

        def queue_pauli(arg):
            q, pauli = arg
            with q:
                for _ in range(n):
                    pauli(0)

        args = [(q1, qml.PauliX), (q2, qml.PauliY)]
        ThreadPool(2).map(queue_pauli, args)
        assert len(q1) == n
        assert len(q2) == n
        for queue, expected_op in args:
            assert all(isinstance(op, expected_op) for op in queue.queue)


test_observables = [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.Hamiltonian(
        [0.1, 0.2, 0.3], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(1), qml.Identity(2)]
    ),
]


class TestApplyOp:
    """Tests for the apply function"""

    def test_error(self):
        """Test that applying an operation without an active
        context raises an error"""
        with pytest.raises(RuntimeError, match="No queuing context"):
            qml.apply(qml.PauliZ(0))

    def test_default_queue_operation_inside(self):
        """Test applying an operation instantiated within the queuing
        context to the existing active queue"""
        with qml.queuing.AnnotatedQueue() as q:
            op1 = qml.PauliZ(0)
            op2 = qml.apply(op1)

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.operations == [op1, op2]

    def test_default_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to an existing active queue"""
        op = qml.PauliZ(0)

        with qml.queuing.AnnotatedQueue() as q:
            qml.apply(op)

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.operations == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to an existing active queue"""
        op = qml.expval(obs)

        with qml.queuing.AnnotatedQueue() as q:
            qml.apply(op)

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.measurements == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_inside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to an existing active queue"""

        with qml.queuing.AnnotatedQueue() as q:
            op1 = qml.expval(obs)
            op2 = qml.apply(op1)

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.measurements == [op1, op2]

    def test_different_queue_operation_inside(self):
        """Test applying an operation instantiated within the queuing
        context to a specfied queuing context"""
        with qml.queuing.AnnotatedQueue() as q1:
            with qml.queuing.AnnotatedQueue() as q2:
                op1 = qml.PauliZ(0)
                op2 = qml.apply(op1, q1)

            tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1 = qml.tape.QuantumScript.from_queue(q1)
        assert tape1.operations == [op2]
        assert tape2.operations == [op1]

    def test_different_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to a specfied queuing context"""
        op = qml.PauliZ(0)

        with qml.queuing.AnnotatedQueue() as q1:
            with qml.queuing.AnnotatedQueue() as q2:
                qml.apply(op, q1)

            tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1 = qml.tape.QuantumScript.from_queue(q1)
        assert tape1.operations == [op]
        assert tape2.operations == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to a specfied queuing context"""
        op = qml.expval(obs)

        with qml.queuing.AnnotatedQueue() as q1:
            with qml.queuing.AnnotatedQueue() as q2:
                qml.apply(op, q1)

            tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1 = qml.tape.QuantumScript.from_queue(q1)
        assert tape1.measurements == [op]
        assert tape2.measurements == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_inside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to a specfied queuing context"""

        with qml.queuing.AnnotatedQueue() as q1:
            with qml.queuing.AnnotatedQueue() as q2:
                op1 = qml.expval(obs)
                op2 = qml.apply(op1, q1)

            tape2 = qml.tape.QuantumScript.from_queue(q2)
        tape1 = qml.tape.QuantumScript.from_queue(q1)
        assert tape1.measurements == [op2]
        assert tape2.measurements == [op1]

    def test_apply_no_queue_method(self):
        """Test that an object with no queue method is still
        added to the queuing context"""
        with qml.queuing.AnnotatedQueue() as q1:
            with qml.queuing.AnnotatedQueue() as q2:
                op1 = qml.apply(5)
                op2 = qml.apply(6, q1)

        assert q1.queue == [op2]
        assert q2.queue == [op1]

    def test_apply_plus_dequeuing(self):
        """Test that operations queued with qml.apply don't get dequeued by subsequent ops."""

        h = qml.H(0)
        with qml.queuing.AnnotatedQueue() as q1:
            op1 = qml.apply(h)
            op2 = qml.adjoint(h)

        assert q1.queue == [op1, op2]


class TestWrappedObj:
    """Tests for the ``WrappedObj`` class"""

    @pytest.mark.parametrize(
        "obj", [qml.PauliX(0), qml.expval(qml.PauliZ(0)), [0, 1, 2], ("a", "b")]
    )
    def test_wrapped_obj_init(self, obj):
        """Test that ``WrappedObj`` is initialized correctly"""
        wo = WrappedObj(obj)
        assert wo.obj is obj

    @pytest.mark.parametrize(
        "obj1, obj2",
        [(qml.PauliX(0), qml.PauliZ(0)), (qml.PauliX(0), qml.PauliX(0)), ((1,), (1, 2))],
    )
    def test_wrapped_obj_eq_false(self, obj1, obj2):
        """Test that ``WrappedObj.__eq__`` returns False when expected."""
        wo1 = WrappedObj(obj1)
        wo2 = WrappedObj(obj2)
        assert wo1 != wo2

    def test_wrapped_obj_eq_false_other_obj(self):
        """Test that WrappedObj.__eq__ returns False when the object being compared is not
        a WrappedObj."""
        op = qml.PauliX(0)
        wo = WrappedObj(op)
        assert wo != op

    def test_wrapped_obj_eq_true(self):
        """Test that ``WrappedObj.__eq__`` returns True when expected."""
        op = qml.PauliX(0)
        assert WrappedObj(op) == WrappedObj(op)

    @pytest.mark.parametrize(
        "obj", [qml.PauliX(0), qml.expval(qml.PauliZ(0)), [0, 1, 2], ("a", "b")]
    )
    def test_wrapped_obj_hash(self, obj):
        """Test that ``WrappedObj.__hash__`` is the object id."""
        wo = WrappedObj(obj)
        assert wo.__hash__() == id(obj)

    def test_wrapped_obj_repr(self):
        """Test that the ``WrappedObj` representation is equivalent to the repr of the
        object it wraps."""

        class Dummy:  # pylint: disable=too-few-public-methods
            """Dummy class with custom repr"""

            def __repr__(self):
                return "test_repr"

        obj = Dummy()
        wo = WrappedObj(obj)
        assert wo.__repr__() == "Wrapped(test_repr)"


def test_process_queue_error_if_not_operator_or_measurement():
    """Test that a QueuingError is raised if process queue encounters an object that does not have a
    _queue_category property
    """
    q = AnnotatedQueue()
    q.append(1)
    with pytest.raises(QueuingError, match="not an object that can be processed"):
        qml.queuing.process_queue(q)
