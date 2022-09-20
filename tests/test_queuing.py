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
import pytest
import pennylane as qml
import numpy as np

from pennylane.queuing import (
    AnnotatedQueue,
    QueuingManager,
    QueuingError,
)


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

        my_circuit.construct([], {})
        tape = my_circuit.qtape

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

        my_circuit.construct([], {})
        tape = my_circuit.qtape

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

        my_circuit.construct([], {})
        tape = my_circuit.qtape

        assert len(tape.operations) == 1
        assert tape.operations[0].name == "Hadamard"

    def test_stop_recording_qnode_qfunc(self):
        """A QNode with a stop_recording qfunc will result in no quantum measurements."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        @QueuingManager.stop_recording()
        def my_circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(0))

        result = my_circuit()
        assert len(result) == 0

        tape = my_circuit.qtape
        assert len(tape.operations) == 0
        assert len(tape.measurements) == 0

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


def test_name_change_warning():
    """Test that a warning is raised when QueuingContext is requested from the queuing module."""
    with pytest.warns(
        UserWarning, match=r"QueuingContext has been renamed qml.queuing.QueuingManager"
    ):
        out = qml.queuing.QueuingContext
    assert out is QueuingManager


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
        """Test that remove fails when the object to be removed is not in the queue."""

        with AnnotatedQueue() as q1:
            op1 = qml.PauliZ(0)
            op2 = qml.PauliZ(1)
            q1.append(op1)
            q1.append(op2)

        with AnnotatedQueue() as q2:
            q2.append(op1)
            with pytest.raises(KeyError):
                q2.remove(op2)

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

    def test_append_tensor_ops(self):
        """Test that ops which are used as inputs to `Tensor`
        are successfully added to the queue, as well as the `Tensor` object."""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = qml.operation.Tensor(A, B)
        assert q.queue == [A, B, tensor_op]
        assert tensor_op.obs == [A, B]

    def test_append_tensor_ops_overloaded(self):
        """Test that Tensor ops created using `@`
        are successfully added to the queue, as well as the `Tensor` object."""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = A @ B
        assert q.queue == [A, B, tensor_op]
        assert tensor_op.obs == [A, B]

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
            A = qml.PauliZ(0)

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

    def test_update_error(self):
        """Test that an exception is raised if get_info is called
        for a non-existent object"""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)

        B = qml.PauliY(1)

        with pytest.raises(QueuingError, match="not in the queue"):
            q.update_info(B, inv=True)

    def test_safe_update_info_queued(self):
        """Test the `safe_update_info` method if the object is already queued."""
        op = qml.RX(0.5, wires=1)

        with AnnotatedQueue() as q:
            q.append(op, key="value1")
            assert q.get_info(op) == {"key": "value1"}
            QueuingManager.safe_update_info(op, key="value2")

        QueuingManager.safe_update_info(op, key="no changes here")
        assert q.get_info(op) == {"key": "value2"}

        q.safe_update_info(op, key="value3")
        assert q.get_info(op) == {"key": "value3"}

    def test_safe_update_info_not_queued(self):
        """Tests the safe_update_info method passes silently if the object is
        not already queued."""
        op = qml.RX(0.5, wires=1)

        with AnnotatedQueue() as q:
            QueuingManager.safe_update_info(op, key="value2")
        QueuingManager.safe_update_info(op, key="no changes here")

        assert len(q.queue) == 0

        q.safe_update_info(op, key="value3")
        assert len(q.queue) == 0

    def test_append_annotating_object(self):
        """Test appending an object that writes annotations when queuing itself"""

        with AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = qml.operation.Tensor(A, B)

        assert q.queue == [A, B, tensor_op]
        assert q.get_info(A) == {"owner": tensor_op}
        assert q.get_info(B) == {"owner": tensor_op}
        assert q.get_info(tensor_op) == {"owns": (A, B)}


test_observables = [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.operation.Tensor(qml.PauliZ(0), qml.PauliX(1)),
    qml.operation.Tensor(qml.PauliZ(0), qml.PauliX(1)) @ qml.Hadamard(2),
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
        with qml.tape.QuantumTape() as tape:
            op1 = qml.PauliZ(0)
            op2 = qml.apply(op1)

        assert tape.operations == [op1, op2]

    def test_default_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to an existing active queue"""
        op = qml.PauliZ(0)

        with qml.tape.QuantumTape() as tape:
            qml.apply(op)

        assert tape.operations == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to an existing active queue"""
        op = qml.expval(obs)

        with qml.tape.QuantumTape() as tape:
            qml.apply(op)

        assert tape.measurements == [op]

    @pytest.mark.parametrize("obs", test_observables)
    def test_default_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to an existing active queue"""

        with qml.tape.QuantumTape() as tape:
            op1 = qml.expval(obs)
            op2 = qml.apply(op1)

        assert tape.measurements == [op1, op2]

    def test_different_queue_operation_inside(self):
        """Test applying an operation instantiated within the queuing
        context to a specfied queuing context"""
        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                op1 = qml.PauliZ(0)
                op2 = qml.apply(op1, tape1)

        assert tape1.operations == [tape2, op2]
        assert tape2.operations == [op1]

    def test_different_queue_operation_outside(self):
        """Test applying an operation instantiated outside a queuing context
        to a specfied queuing context"""
        op = qml.PauliZ(0)

        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                qml.apply(op, tape1)

        assert tape1.operations == [tape2, op]
        assert tape2.operations == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated outside a queuing context
        to a specfied queuing context"""
        op = qml.expval(obs)

        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                qml.apply(op, tape1)

        assert tape1.measurements == [op]
        assert tape2.measurements == []

    @pytest.mark.parametrize("obs", test_observables)
    def test_different_queue_measurements_outside(self, obs):
        """Test applying a measurement instantiated inside a queuing context
        to a specfied queuing context"""

        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                op1 = qml.expval(obs)
                op2 = qml.apply(op1, tape1)

        assert tape1.measurements == [op2]
        assert tape2.measurements == [op1]

    def test_apply_no_queue_method(self):
        """Test that an object with no queue method is still
        added to the queuing context"""
        with qml.tape.QuantumTape() as tape1:
            with qml.tape.QuantumTape() as tape2:
                op1 = qml.apply(5)
                op2 = qml.apply(6, tape1)

        assert tape1.queue == [tape2, op2]
        assert tape2.queue == [op1]

        # note that tapes don't know how to process integers,
        # so they are not included after queue processing
        assert tape1.operations == [tape2]
        assert tape2.operations == []
