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
"""Unit and integration tests for the transform program."""
from typing import Sequence, Callable

import pytest
import pennylane as qml
from pennylane.transforms.core import (
    transform,
    TransformProgram,
    TransformError,
    TransformContainer,
)


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


class TestTransformProgram:
    """Test the transform program class and its method."""

    def test_empty_program(self):
        """Test an empty program."""
        program = TransformProgram()
        assert program.is_empty()
        assert len(program) == 0

        with pytest.raises(
            TransformError,
            match="The transform program is empty and you cannot get the last "
            "transform container.",
        ):
            program.get_last()

    def test_basic_program(self):
        """Test to push back multiple transforms into a program and also the different methods of a program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=first_valid_transform)
        transform_program.push_back(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform2 = TransformContainer(transform=second_valid_transform)
        transform_program.push_back(transform2)

        assert not transform_program.is_empty()
        assert len(transform_program) == 2
        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is second_valid_transform

        transform_program.push_back(transform1)
        transform_program.push_back(transform2)

        sub_program_transforms = transform_program[2:]
        assert len(sub_program_transforms) == 2
        assert sub_program_transforms[0] is transform1
        assert sub_program_transforms[1] is transform2

    def test_pop_front(self):
        """Test the pop front method of the transform program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=first_valid_transform)
        transform_program.push_back(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform_container = transform_program.pop_front()

        assert transform_program.is_empty()
        assert transform_container is transform1

    def test_insert_front(self):
        """Test to insert a transform at the beginning of a transform program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=first_valid_transform)
        transform_program.push_back(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform2 = TransformContainer(transform=second_valid_transform)
        transform_program.insert_front(transform2)

        assert not transform_program.is_empty()
        assert len(transform_program) == 2
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0] is transform2
        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1] is transform1

        transform3 = TransformContainer(transform=second_valid_transform, is_informative=True)

        with pytest.raises(
            TransformError,
            match="Informative transforms can only be added at the end of the program.",
        ):
            transform_program.insert_front(transform3)

    def test_iter_program(self):
        """Test iteration over the transform program."""
        transform_program = TransformProgram()
        transform1 = TransformContainer(transform=first_valid_transform)

        for _ in range(0, 10):
            transform_program.push_back(transform1)

        assert len(transform_program) == 10

        for elem in transform_program:
            assert isinstance(elem, TransformContainer)
            assert elem.transform is first_valid_transform

    def test_repr_program(self):
        """Test the string representation of a program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=first_valid_transform)
        transform2 = TransformContainer(transform=second_valid_transform)

        transform_program.push_back(transform1)
        transform_program.push_back(transform2)

        str_program = repr(transform_program)
        assert (
            str_program
            == "TransformProgram("
            + str(first_valid_transform.__name__)
            + ", "
            + str(second_valid_transform.__name__)
            + ")"
        )

    def test_valid_transforms(self):
        """Test that that it is only possible to create valid transforms."""
        transform_program = TransformProgram()
        transform1 = TransformContainer(transform=first_valid_transform, is_informative=True)
        transform_program.push_back(transform1)

        with pytest.raises(
            TransformError, match="The transform program already has an informative transform."
        ):
            transform_program.push_back(transform1)

        transform2 = TransformContainer(transform=second_valid_transform, is_informative=False)

        with pytest.raises(
            TransformError, match="The transform program already has an informative transform."
        ):
            transform_program.push_back(transform2)


class TestTransformProgramIntegration:
    """Test the transform program and its integration with QNodes"""

    def test_qnode_integration(self):
        """Test the integration with QNode wiht two similar transforms."""

        dispatched_transform = transform(first_valid_transform)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        new_qnode = dispatched_transform(dispatched_transform(qnode_circuit, 0), 0)

        program = new_qnode.transform_program
        transformed_qnode_rep = repr(program)
        assert (
            transformed_qnode_rep
            == "TransformProgram("
            + str(first_valid_transform.__name__)
            + ", "
            + str(first_valid_transform.__name__)
            + ")"
        )

        assert not program.is_empty()
        assert len(program) == 2
        assert program[0].transform is first_valid_transform
        assert program[1].transform is first_valid_transform

    def test_qnode_integration_different_transforms(self):
        """Test the integration with QNode with two different transforms."""

        dispatched_transform_1 = transform(first_valid_transform)
        dispatched_transform_2 = transform(second_valid_transform)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        new_qnode = dispatched_transform_2(dispatched_transform_1(qnode_circuit, 0), 0)

        program = new_qnode.transform_program
        transformed_qnode_rep = repr(program)
        assert (
            transformed_qnode_rep
            == "TransformProgram("
            + str(first_valid_transform.__name__)
            + ", "
            + str(second_valid_transform.__name__)
            + ")"
        )

        assert not program.is_empty()
        assert len(program) == 2
        assert program[0].transform is first_valid_transform
        assert program[1].transform is second_valid_transform
