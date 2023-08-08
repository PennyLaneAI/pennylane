# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit and integration tests for the transform dispatcher."""
from typing import Sequence, Callable
from functools import partial

import pytest
import pennylane as qml
from pennylane.transforms.core import transform, TransformError

# TODO: Replace with default qubit 2
dev = qml.device("default.qubit", wires=2)

with qml.tape.QuantumTape() as tape_circuit:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(0.42, wires=1)
    qml.expval(qml.PauliZ(wires=0))


def qfunc_circuit(a):
    """Qfunc circuit/"""
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(a, wires=1)
    return qml.expval(qml.PauliZ(wires=0))


##########################################
# Non-valid transforms

non_callable = tape_circuit


def no_tape_transform(
    circuit: qml.tape.QuantumTape, index: int
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Transform without tape."""
    circuit._ops.pop(index)  # pylint:disable=protected-access
    return [circuit], lambda x: x


def no_quantum_tape_transform(
    tape: qml.operation.Operator, index: int
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Transform with wrong hinting."""
    tape._ops.pop(index)  # pylint:disable=protected-access
    return [tape], lambda x: x


def no_processing_fn_transform(tape: qml.tape.QuantumTape) -> Sequence[qml.tape.QuantumTape]:
    """Transform without processing fn."""
    tape_copy = tape.copy()
    return [tape, tape_copy]


def no_tape_sequence_transform(tape: qml.tape.QuantumTape) -> (qml.tape.QuantumTape, Callable):
    """Transform wihtout Sequence return."""
    return tape, lambda x: x


def no_callable_return(
    tape: qml.tape.QuantumTape,
) -> (Sequence[qml.tape.QuantumTape], qml.tape.QuantumTape):
    """Transform without callable return."""
    return list(tape), tape


non_valid_transforms = [
    non_callable,
    no_processing_fn_transform,
    no_tape_sequence_transform,
    no_tape_transform,
    no_quantum_tape_transform,
    no_callable_return,
]


##########################################
# Valid transforms


def first_valid_transform(
    tape: qml.tape.QuantumTape, index: int
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid transform."""
    tape._ops.pop(index)  # pylint:disable=protected-access
    return [tape], lambda x: x


def second_valid_transform(
    tape: qml.tape.QuantumTape, index: int
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid trasnform."""
    tape1 = tape.copy()
    tape2 = tape._ops.pop(index)  # pylint:disable=protected-access

    def fn(results):
        return qml.math.sum(results)

    return [tape1, tape2], fn


valid_transforms = [first_valid_transform, second_valid_transform]


##########################################
# Non-valid expand transform
def multiple_args_expand_transform(
    tape: qml.tape.QuantumTape, index: int
) -> (Sequence[qml.tape.QuantumTape], Callable):
    """Multiple args expand fn."""
    tape._ops.pop(index)  # pylint:disable=protected-access
    return [tape], lambda x: x


# Valid expand transform
def expand_transform(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
    """A valid expand transform."""
    return [tape], lambda x: x


class TestTransformDispatcher:
    """Test the transform function (validate and dispatch)."""

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_integration_dispatcher_with_valid_transform(self, valid_transform):
        """Test that no error is raised with the transform function and that the transform dispatcher returns
        the right object."""

        dispatched_transform = transform(valid_transform)

        # Applied on a tape
        tapes, fn = dispatched_transform(tape_circuit, 0)

        assert isinstance(tapes, Sequence)
        assert callable(fn)

        # Applied on a qfunc (return a qfunc)
        qfunc = dispatched_transform(qfunc_circuit, 0)
        assert callable(qfunc)

        # Applied on a qnode (return a qnode with populated the program)
        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        qnode_transformed = dispatched_transform(qnode_circuit, 0)
        assert not qnode_circuit.transform_program

        assert isinstance(qnode_transformed, qml.QNode)
        assert isinstance(qnode_transformed.transform_program, qml.transforms.core.TransformProgram)
        assert isinstance(
            qnode_transformed.transform_program.pop_front(), qml.transforms.core.TransformContainer
        )
        assert not dispatched_transform.is_informative

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_integration_dispatcher_with_valid_transform_decorator_partial(self, valid_transform):
        """Test that no error is raised with the transform function and that the transform dispatcher returns
        the right object."""

        dispatched_transform = transform(valid_transform)
        targs = [0]

        @partial(dispatched_transform, targs=targs)
        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        assert isinstance(qnode_circuit, qml.QNode)
        assert isinstance(qnode_circuit.transform_program, qml.transforms.core.TransformProgram)
        assert isinstance(
            qnode_circuit.transform_program.pop_front(), qml.transforms.core.TransformContainer
        )

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_integration_dispatcher_with_valid_transform_decorator(self, valid_transform):
        """Test that no error is raised with the transform function and that the transform dispatcher returns
        the right object."""

        dispatched_transform = transform(valid_transform)
        targs = [0]

        @dispatched_transform(targs=targs)
        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        assert isinstance(qnode_circuit, qml.QNode)
        assert isinstance(qnode_circuit.transform_program, qml.transforms.core.TransformProgram)
        assert isinstance(
            qnode_circuit.transform_program.pop_front(), qml.transforms.core.TransformContainer
        )

    def test_queuing_qfunc_transform(self):
        """Test that queuing works with the transformed quantum function."""

        dispatched_transform = transform(first_valid_transform)

        # Applied on a tape
        tapes, fn = dispatched_transform(tape_circuit, 0)

        assert isinstance(tapes, Sequence)
        assert callable(fn)

        # Applied on a qfunc (return a qfunc)
        qfunc_transformed = dispatched_transform(qfunc_circuit, 0)
        assert callable(qfunc_transformed)

        with qml.tape.QuantumTape() as transformed_tape:
            qfunc_transformed(0.42)

        assert isinstance(transformed_tape, qml.tape.QuantumTape)
        assert transformed_tape.circuit is not None
        assert len(transformed_tape.circuit) == 4

        with qml.tape.QuantumTape() as tape:
            qfunc_circuit(0.42)

        assert isinstance(transformed_tape, qml.tape.QuantumTape)
        assert tape.circuit is not None
        assert len(tape.circuit) == 5

    def test_qnode_with_expand_transform(self):
        """Test qnode with a transform program and expand transform."""

        dispatched_transform = transform(first_valid_transform, expand_transform=expand_transform)

        # Applied on a tape
        tapes, fn = dispatched_transform(tape_circuit, 0)

        assert isinstance(tapes, Sequence)
        assert callable(fn)

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        # Applied on a qfunc (return a qfunc)
        qnode_transformed = dispatched_transform(qnode_circuit, 0)

        assert isinstance(qnode_transformed.transform_program, qml.transforms.core.TransformProgram)
        expand_transform_container = qnode_transformed.transform_program.pop_front()
        assert isinstance(expand_transform_container, qml.transforms.core.TransformContainer)
        assert expand_transform_container.args == []
        assert expand_transform_container.kwargs == {}
        assert expand_transform_container.classical_cotransform is None
        assert not expand_transform_container.is_informative

        transform_container = qnode_transformed.transform_program.pop_front()

        assert isinstance(transform_container, qml.transforms.core.TransformContainer)
        assert transform_container.args == [0]
        assert transform_container.kwargs == {}
        assert transform_container.classical_cotransform is None
        assert not expand_transform_container.is_informative

    @pytest.mark.parametrize("non_valid_transform", non_valid_transforms)
    def test_dispatcher_signature_non_valid_transform(self, non_valid_transform):
        """Test the non-valid transforms raises a Transform error."""

        with pytest.raises(TransformError):
            transform(non_valid_transform)

    def test_error_not_callable_transform(self):
        """Test that a non-callable is not a valid transforms."""

        with pytest.raises(TransformError, match="The function to register, "):
            transform(non_callable)

    def test_error_no_tape_transform(self):
        """Test that a transform without tape as arg is not valid."""

        with pytest.raises(TransformError, match="The first argument of a transform must be tape."):
            transform(no_tape_transform)

    def test_error_no_quantumtape_transform(self):
        """Test that a transform needs tape to be a quantum tape in order to be valid."""

        with pytest.raises(
            TransformError, match="The type of the tape argument must be a QuantumTape."
        ):
            transform(no_quantum_tape_transform)

    def test_error_no_processing_fn_transform(self):
        """Test that a transform without processing fn return is not valid."""

        with pytest.raises(TransformError, match="The return of a transform must match"):
            transform(no_processing_fn_transform)

    def test_error_no_tape_sequence_transform(self):
        """Test that a transform not returning a sequence of tape is not valid."""

        with pytest.raises(
            TransformError, match="The first return of a transform must be a sequence of tapes:"
        ):
            transform(no_tape_sequence_transform)

    def test_error_no_callable_return(self):
        """Test that a transform not returning a callable is not valid."""

        with pytest.raises(
            TransformError, match="The second return of a transform must be a callable"
        ):
            transform(no_callable_return)

    def test_expand_transform_not_callable(self):
        """Test that an expand transform must be a callable otherwise it is not valid."""

        with pytest.raises(
            TransformError, match="The expand function must be a valid Python function."
        ):
            transform(first_valid_transform, expand_transform=non_callable)

    def test_multiple_args_expand_transform(self):
        """Test that an expand transform must take a single argument which is the tape."""

        with pytest.raises(
            TransformError,
            match="The expand transform does not support arg and kwargs other than tape.",
        ):
            transform(first_valid_transform, expand_transform=multiple_args_expand_transform)

    def test_cotransform_not_implemented(self):
        """Test that a co-transform must be a callable."""

        with pytest.raises(
            NotImplementedError, match="Classical cotransforms are not yet integrated."
        ):
            transform(first_valid_transform, classical_cotransform=non_callable)

    def test_qfunc_transform_multiple_tapes(self):
        """Test that quantum function is not compatible with multiple tapes."""
        dispatched_transform = transform(second_valid_transform)
        with pytest.raises(
            TransformError, match="Impossible to dispatch your transform on quantum function"
        ):
            dispatched_transform(qfunc_circuit, 0)(0.42)

    def test_dispatched_transform_attribute(self):
        """Test the dispatcher attributes."""
        dispatched_transform = transform(first_valid_transform)

        assert dispatched_transform.transform is first_valid_transform
        assert dispatched_transform.expand_transform is None
        assert dispatched_transform.classical_cotransform is None

    def test_the_transform_container_attributes(self):
        """Test the transform container attributes."""
        container = qml.transforms.core.TransformContainer(
            first_valid_transform, args=[0], kwargs={}, classical_cotransform=None
        )

        q_transform, args, kwargs, cotransform, is_informative = container

        assert q_transform is first_valid_transform
        assert args == [0]
        assert kwargs == {}
        assert cotransform is None
        assert not is_informative

        assert container.transform is first_valid_transform
        assert container.args == [0]
        assert not container.kwargs
        assert container.classical_cotransform is None
        assert not container.is_informative
