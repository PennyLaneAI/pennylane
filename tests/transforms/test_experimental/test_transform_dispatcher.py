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

import pytest
import pennylane as qml
from pennylane.transforms.core import transform, TransformError
from collections.abc import Sequence

# TODO: Replace with default qubit 2
dev = qml.device("default.qubit", wires=2)

with qml.tape.QuantumTape() as tape_circuit:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(0.42, wires=1)
    qml.expval(qml.PauliZ(wires=0))


def qfunc_circuit(a):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(a, wires=1)
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(device=dev)
def qnode_circuit(a):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(a, wires=1)
    return qml.expval(qml.PauliZ(wires=0))


##########################################
# Non-valid transforms

non_callable = tape_circuit


def no_processing_fn_transform(tape: qml.tape.QuantumTape) -> Sequence[qml.tape.QuantumTape]:
    tape_copy = tape.copy()
    return [tape, tape_copy]


def no_tape_sequence_transform(tape: qml.tape.QuantumTape) -> (qml.tape.QuantumTape, callable):
    return tape, lambda x: x


non_valid_transforms = [non_callable, no_processing_fn_transform, no_tape_sequence_transform]


##########################################
# Valid transforms


def first_valid_transform(tape: qml.tape.QuantumTape, index: int) -> (Sequence[qml.tape.QuantumTape], callable):
    tape.circuit.pop(index)
    return [tape], lambda x: x


def second_valid_transform(tape: qml.tape.QuantumTape, index: int) -> (Sequence[qml.tape.QuantumTape], callable):
    tape1 = tape.copy()
    tape2 = tape.circuit.pop(index)

    def fn(results):
        return qml.math.sum(results)

    return [tape1, tape2], fn


valid_transforms = [first_valid_transform, second_valid_transform]


##########################################
# Non-valid expand transforms


def no_processing_fn_expand_transform(tape: qml.tape.QuantumTape) -> qml.tape.QuantumTape:
    tape_copy = tape.copy()
    return [tape, tape_copy]


def no_tape_sequence_expand_transform(tape: qml.tape.QuantumTape) -> qml.tape.QuantumTape:
    return tape, lambda x: x


non_valid_expand_transforms = [
    non_callable,
    no_processing_fn_expand_transform,
    no_tape_sequence_expand_transform,
]


##########################################
# Valid expand transforms


def a_valid_expand_transform(tape: qml.tape.QuantumTape) -> (qml.tape.QuantumTape, callable):
    return [tape], lambda x: x


valid_expand_transforms = [a_valid_expand_transform]


class TestTransformDispatcher:
    """Test the transform function (validate and dispatch)."""

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_dispatcher_with_valid_transform(self, valid_transform):
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
        qnode = dispatched_transform(qnode_circuit, 0)
        assert isinstance(qnode, qml.QNode)
        assert isinstance(qnode.transform_program, list)
        assert isinstance(qnode.transform_program[0], qml.transforms.core.TransformContainer)

    @pytest.mark.parametrize("non_valid_transform", non_valid_transforms)
    def test_dispatcher_signature_non_valid_transform(self, non_valid_transform):
        """Test the non-valid transforms raises a Transform error."""

        with pytest.raises(TransformError):
            transform(non_valid_transform)
