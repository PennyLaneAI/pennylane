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
# pylint: disable=no-member

import pytest
import rustworkx as rx

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import (
    TransformContainer,
    TransformError,
    TransformProgram,
    transform,
)
from pennylane.transforms.core.transform_program import (
    CotransformCache,
    _apply_postprocessing_stack,
    _batch_postprocessing,
    null_postprocessing,
)
from pennylane.typing import PostprocessingFn, Result, ResultBatch


def first_valid_transform(
    tape: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid transform."""
    tape = tape.copy()
    tape._ops.pop(index)  # pylint:disable=protected-access
    return [tape], lambda x: x


def expand_transform(
    tape: QuantumScript, index: int  # pylint:disable=unused-argument
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid expand transform."""
    return [tape], lambda x: x


def second_valid_transform(
    tape: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid trasnform."""
    tape1 = tape.copy()
    tape2 = tape.copy()
    tape2 = tape._ops.pop(index)  # pylint:disable=protected-access

    def fn(results):
        return qml.math.sum(results)

    return [tape1, tape2], fn


def informative_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid informative transform"""

    def fn(results):
        return len(results[0].operations)

    return [tape], fn


class TestUtilityHelpers:
    """test the private functions used in post processing."""

    def test_batch_postprocessing(self):
        """Test the _batch_postprocessing helper function."""
        results = (1.0, 2.0, 3.0, 4.0)

        def postprocessing1(results):
            return results[0] + results[1]

        def postprocessing2(results):
            return results[0] + 1

        out = _batch_postprocessing(
            results, (postprocessing1, postprocessing2), [slice(0, 2), slice(2, 4)]
        )
        assert out == (3.0, 4.0)

    def test_postprocessing_stack(self):
        """Tests the _apply_postprocessing_stack helper function."""

        results = (1.0, 2.0, 3.0, 4.0)

        def postprocessing1(results):
            return (results[0] + results[1], results[2] + results[3])

        def postprocessing2(results):
            return (results[0] + 1, results[1] + 2)

        out1 = _apply_postprocessing_stack(results, [postprocessing1])
        assert out1 == (3.0, 7.0)

        out2 = _apply_postprocessing_stack(results, [postprocessing2, postprocessing1])
        assert out2 == (4.0, 9.0)


class TestTransformProgramDunders:
    """Test the dunder methods."""

    def test_bool(self):
        """Check that a transform program is falsy if empty and truthy if not."""
        empty_prog = TransformProgram()
        assert not empty_prog

        transform1 = TransformContainer(qml.transform(first_valid_transform))
        populated_prog = TransformProgram((transform1,))
        assert populated_prog

    def test_iter_program(self):
        """Test iteration over the transform program."""
        transform_program = TransformProgram()
        transform1 = TransformContainer(qml.transform(first_valid_transform))

        for _ in range(10):
            transform_program.push_back(transform1)

        assert len(transform_program) == 10

        for elem in transform_program:
            assert isinstance(elem, TransformContainer)
            assert elem.transform is first_valid_transform

    def test_getitem(self):
        """Tests for the getitem dunder."""

        t0 = TransformContainer(qml.transform(first_valid_transform))
        t1 = TransformContainer(transform=qml.transform(second_valid_transform))
        t2 = TransformContainer(transform=qml.transform(informative_transform))
        program = TransformProgram([t0, t1, t2])

        assert program[0] == t0
        assert program[1] == t1
        assert program[2] == t2

        assert program[:2] == TransformProgram([t0, t1])
        assert program[::-1] == TransformProgram([t2, t1, t0])

    def test_contains(self):
        """Test that we can check whether a transform or transform container exists in a transform."""

        t0 = TransformContainer(transform=qml.transform(first_valid_transform))
        t1 = TransformContainer(transform=qml.transform(second_valid_transform))
        t2 = TransformContainer(transform=qml.transform(informative_transform))
        program = TransformProgram([t0, t1, t2])

        assert t0 in program
        assert t1 in program
        assert t2 in program
        assert qml.compile not in program

        assert t0 in program
        assert t1 in program
        assert t2 in program

        t_not = TransformContainer(transform=qml.compile)
        assert t_not not in program

        assert "a" not in program

    def test_add_single_programs(self):
        """Test adding two transform programs"""
        transform_program1 = TransformProgram()
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform_program1.push_back(transform1)

        transform_program2 = TransformProgram()
        transform2 = TransformContainer(transform=qml.transform(second_valid_transform))
        transform_program2.push_back(transform2)

        transform_program = transform_program1 + transform_program2

        assert len(transform_program) == 2

        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is second_valid_transform

        transform_program = transform_program2 + transform_program1

        assert len(transform_program) == 2

        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is second_valid_transform

        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is first_valid_transform

    def test_add_two_programs(self):
        """Test adding two transform programs"""
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(transform=qml.transform(second_valid_transform))

        transform_program1 = TransformProgram()
        transform_program1.push_back(transform1)
        transform_program1.push_back(transform1)
        transform_program1.push_back(transform1)

        transform_program2 = TransformProgram()
        transform_program1.push_back(transform2)
        transform_program1.push_back(transform2)

        transform_program = transform_program1 + transform_program2

        assert len(transform_program) == 5

        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is first_valid_transform

        assert isinstance(transform_program[2], TransformContainer)
        assert transform_program[2].transform is first_valid_transform

        assert isinstance(transform_program[3], TransformContainer)
        assert transform_program[3].transform is second_valid_transform

        assert isinstance(transform_program[4], TransformContainer)
        assert transform_program[4].transform is second_valid_transform

    def test_add_both_final_transform_programs(self):
        """Test that an error is raised if two programs are added when both have
        terminal transforms"""
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )

        transform_program1 = TransformProgram()
        transform_program1.push_back(transform1)
        transform_program1.push_back(transform2)

        transform_program2 = TransformProgram()
        transform_program2.push_back(transform1)
        transform_program2.push_back(transform2)

        with pytest.raises(
            TransformError, match="The transform program already has a terminal transform"
        ):
            _ = transform_program1 + transform_program2

    def test_add_programs_with_one_final_transform(self):
        """Test that transform programs are added correctly when one of them has a terminal
        transform."""
        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )

        transform_program1 = TransformProgram()
        transform_program1.push_back(transform1)

        transform_program2 = TransformProgram()
        transform_program2.push_back(transform1)
        transform_program2.push_back(transform2)

        merged_program1 = transform_program1 + transform_program2
        assert len(merged_program1) == 3

        assert isinstance(merged_program1[0], TransformContainer)
        assert merged_program1[0].transform is first_valid_transform

        assert isinstance(merged_program1[1], TransformContainer)
        assert merged_program1[1].transform is first_valid_transform

        assert isinstance(merged_program1[2], TransformContainer)
        assert merged_program1[2].transform is second_valid_transform

        merged_program2 = transform_program2 + transform_program1
        assert len(merged_program2) == 3

        assert isinstance(merged_program2[0], TransformContainer)
        assert merged_program2[0].transform is first_valid_transform

        assert isinstance(merged_program2[1], TransformContainer)
        assert merged_program2[1].transform is first_valid_transform

        assert isinstance(merged_program2[2], TransformContainer)
        assert merged_program2[2].transform is second_valid_transform

    def test_repr_program(self):
        """Test the string representation of a program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=qml.transform(first_valid_transform))
        transform2 = TransformContainer(transform=qml.transform(second_valid_transform))

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

    def test_equality(self):
        """Tests that we can compare TransformProgram objects with the '==' and '!=' operators."""
        t1 = TransformContainer(qml.transforms.compile, kwargs={"num_passes": 2})
        t2 = TransformContainer(qml.transforms.compile, kwargs={"num_passes": 2})
        t3 = TransformContainer(qml.transforms.transpile, kwargs={"coupling_map": [(0, 1), (1, 2)]})

        p1 = TransformProgram([t1, t3])
        p2 = TransformProgram([t2, t3])
        p3 = TransformProgram([t3, t2])

        # test for equality of identical objects
        assert p1 == p2
        # test for inequality of different objects
        assert p1 != p3
        assert p1 != t1

        # Test inequality with different transforms
        t4 = TransformContainer(qml.transforms.transpile, kwargs={"coupling_map": [(0, 1), (2, 3)]})
        p4 = TransformProgram([t1, t4])
        assert p1 != p4


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

    def test_get_last(self):
        """Tests the get_last method"""
        program = TransformProgram()
        program.add_transform(transform(first_valid_transform))
        program.add_transform(transform(second_valid_transform))
        assert program.get_last() == TransformContainer(transform=transform(second_valid_transform))

    def test_push_back(self):
        """Test to push back multiple transforms into a program and also the different methods of a program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=transform(first_valid_transform))
        transform_program.push_back(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform2 = TransformContainer(transform=transform(second_valid_transform))
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

        with pytest.raises(
            TransformError,
            match="Only transform container can be added to the transform program.",
        ):
            transform_program.push_back(10.0)

    def test_add_transform(self):
        """Test to add multiple transforms into a program and also the different methods of a program."""
        transform_program = TransformProgram()

        transform1 = transform(first_valid_transform)
        transform_program.add_transform(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform2 = transform(second_valid_transform)
        transform_program.add_transform(transform2)

        assert not transform_program.is_empty()
        assert len(transform_program) == 2
        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is second_valid_transform

        transform_program.add_transform(transform1)
        transform_program.add_transform(transform2)

        sub_program_transforms = transform_program[2:]
        assert len(sub_program_transforms) == 2
        assert sub_program_transforms[0].transform is first_valid_transform
        assert sub_program_transforms[1].transform is second_valid_transform

        with pytest.raises(
            TransformError,
            match="Only transform dispatcher can be added to the transform program.",
        ):
            transform_program.add_transform(10.0)

    def test_add_transform_with_expand(self):
        """Test to add a transform with expand into a program."""
        transform_program = TransformProgram()

        transform1 = transform(first_valid_transform, expand_transform=expand_transform)
        transform_program.add_transform(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 2
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is expand_transform

        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is first_valid_transform

    def test_pop_front(self):
        """Test the pop front method of the transform program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=transform(first_valid_transform))
        transform_program.push_back(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform_container = transform_program.pop_front()

        assert transform_program.is_empty()
        assert transform_container is transform1

    def test_insert_front(self):
        """Test to insert a transform (container) at the beginning of a transform program."""
        transform_program = TransformProgram()

        transform1 = TransformContainer(transform=transform(first_valid_transform))
        transform_program.push_back(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform2 = TransformContainer(transform=transform(second_valid_transform))
        transform_program.insert_front(transform2)

        assert not transform_program.is_empty()
        assert len(transform_program) == 2
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0] is transform2
        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1] is transform1

        transform3 = TransformContainer(
            transform=transform(second_valid_transform, is_informative=True)
        )

        with pytest.raises(
            TransformError,
            match="Informative transforms can only be added at the end of the program.",
        ):
            transform_program.insert_front(transform3)

    def test_insert_transform(self):
        """Test to insert a transform (dispatcher) at the beginning of a transform program."""
        transform_program = TransformProgram()

        transform1 = transform(first_valid_transform)
        transform_program.insert_front_transform(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 1
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is first_valid_transform

        transform2 = transform(second_valid_transform)
        transform_program.insert_front_transform(transform2)

        assert not transform_program.is_empty()
        assert len(transform_program) == 2
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is second_valid_transform
        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is first_valid_transform

        transform3 = transform(second_valid_transform, is_informative=True)

        with pytest.raises(
            TransformError,
            match="Informative transforms can only be added at the end of the program.",
        ):
            transform_program.insert_front_transform(transform3)

    def test_insert_transform_with_expand(self):
        """Test to insert front a transform with expand into a program."""
        transform_program = TransformProgram()

        transform1 = transform(first_valid_transform, expand_transform=expand_transform)
        transform_program.insert_front_transform(transform1)

        assert not transform_program.is_empty()
        assert len(transform_program) == 2
        assert isinstance(transform_program[0], TransformContainer)
        assert transform_program[0].transform is expand_transform

        assert isinstance(transform_program[1], TransformContainer)
        assert transform_program[1].transform is first_valid_transform

    def test_valid_transforms(self):
        """Test adding transforms to a program with a terminal transform."""
        transform_program = TransformProgram()
        transform1 = TransformContainer(qml.transform(first_valid_transform, is_informative=True))
        transform_program.push_back(transform1)

        t_normal = TransformContainer(qml.transform(second_valid_transform))
        transform_program.push_back(t_normal)
        print(transform_program)
        assert len(transform_program) == 2
        assert transform_program[0] is t_normal
        assert transform_program[1] is transform1

        t_normal2 = TransformContainer(qml.transform(first_valid_transform))
        transform_program.push_back(t_normal2)
        assert transform_program[0] is t_normal
        assert transform_program[1] is t_normal2
        assert transform_program[2] is transform1

        with pytest.raises(
            TransformError, match="The transform program already has a terminal transform."
        ):
            transform_program.push_back(transform1)

        transform2 = TransformContainer(
            transform=qml.transform(second_valid_transform, final_transform=True)
        )
        with pytest.raises(
            TransformError, match="The transform program already has a terminal transform."
        ):
            transform_program.push_back(transform2)


class TestClassicalCotransfroms:
    """Test for handling the classical cotransform component."""

    def test_classical_cotransform_caching(self):
        """Tests for setting the classical cotransform."""

        @qml.qnode(qml.device("default.qubit"))
        def f(*_, **__):
            return qml.state()

        program1 = TransformProgram()  # no hybrid transforms
        assert program1.cotransform_cache is None
        program1.set_classical_component(f, (1,), {"a": 2})
        assert program1.cotransform_cache is None

        new_t = qml.transform(
            qml.gradients.param_shift.transform, classical_cotransform=lambda *args: 0
        )

        hybrid_t = TransformContainer(new_t, (), {"hybrid": True})
        program2 = TransformProgram((hybrid_t,))
        program2.set_classical_component(f, (1,), {"a": 2})
        assert program2.cotransform_cache == CotransformCache(f, (1,), {"a": 2})

        program3 = program1 + program2
        assert program3.cotransform_cache == CotransformCache(f, (1,), {"a": 2})

        program4 = program2 + program1
        assert program4.cotransform_cache == CotransformCache(f, (1,), {"a": 2})

        with pytest.raises(ValueError, match=r"programs with cotransform caches"):
            _ = program2 + program2

    @pytest.mark.parametrize("arg", (0.5, rx.PyGraph()))
    def test_error_on_numpy_qnode(self, arg):
        """Test an error is raised if there are no trainable parameters for a hybrid program."""

        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        circuit = qml.gradients.param_shift(circuit, hybrid=True)
        circuit.transform_program.set_classical_component(circuit, (arg,), {})

        tape = qml.tape.QuantumScript([], [])
        with pytest.raises(QuantumFunctionError, match="No trainable parameters"):
            circuit.transform_program((tape,))


class TestTransformProgramCall:
    """Tests for calling a TransformProgram on a batch of quantum tapes."""

    def test_call_on_empty_program(self):
        """Tests that an empty program returns input tapes with the null postprocessing function."""

        batch = qml.tape.QuantumScript([], [qml.state()])

        prog = TransformProgram()
        new_batch, postprocessing = prog(batch)

        assert new_batch is batch
        assert postprocessing is null_postprocessing

        obj = [1, 2, 3, "b"]
        assert null_postprocessing(obj) is obj

    def test_single_transform_program(self):
        """Basic test with a single transform that only modifies the tape but not the results."""

        def single_null_postprocessing(results):
            return results[0]

        def remove_operation_at_index(
            tape: QuantumScript, index: int
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            """A valid transform."""
            new_ops = list(tape.operations)
            new_ops.pop(index)  # pylint:disable=protected-access
            return (
                qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots),
            ), single_null_postprocessing

        container = TransformContainer(transform(remove_operation_at_index), kwargs={"index": 1})
        prog = TransformProgram((container,))

        tape0 = qml.tape.QuantumScript(
            [qml.S(0), qml.T(1), qml.SX(2)], [qml.expval(qml.PauliZ(0))], shots=100
        )
        batch = (tape0,)
        new_batch, fn = prog(batch)

        assert len(new_batch) == 1
        expected = [qml.S(0), qml.SX(2), qml.expval(qml.PauliZ(0))]
        for op1, op2 in zip(expected, new_batch[0]):
            qml.assert_equal(op1, op2)
        assert new_batch[0].shots == qml.measurements.Shots(100)

        assert fn.func is _apply_postprocessing_stack
        assert fn.args == tuple()

        assert len(fn.keywords["postprocessing_stack"]) == 1
        postprocessing0 = fn.keywords["postprocessing_stack"][0]
        assert postprocessing0.func is _batch_postprocessing
        assert postprocessing0.args == tuple()
        assert postprocessing0.keywords["individual_fns"] == [single_null_postprocessing]
        assert postprocessing0.keywords["slices"] == [slice(0, 1)]

        results = (2.0,)
        assert fn(results) == (2.0,)

    def test_chain_two_postprocessings(self):
        """Test postprocessing functions applied in reverse order."""

        def add_one(results):
            return results[0] + 1.0

        def scale_two(results):
            return results[0] * 2.0

        def transform_add(tape: qml.tape.QuantumTape):
            """A valid transform."""
            return (tape,), add_one

        def transform_mul(tape: qml.tape.QuantumTape):
            return (tape,), scale_two

        container1 = TransformContainer(transform(transform_add))
        container2 = TransformContainer(transform(transform_mul))
        prog = TransformProgram((container1, container2))

        tape0 = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))], shots=100)
        batch = (tape0,)
        new_batch, fn = prog(batch)

        assert len(new_batch) == 1
        assert new_batch[0] is tape0

        assert fn.func is _apply_postprocessing_stack
        assert fn.args == tuple()
        assert len(fn.keywords["postprocessing_stack"]) == 2

        postprocessing0 = fn.keywords["postprocessing_stack"][0]
        assert postprocessing0.func is _batch_postprocessing
        assert postprocessing0.args == tuple()
        assert postprocessing0.keywords["individual_fns"] == [
            add_one,
        ]
        assert postprocessing0.keywords["slices"] == [slice(0, 1)]

        postprocessing1 = fn.keywords["postprocessing_stack"][1]
        assert postprocessing1.func is _batch_postprocessing
        assert postprocessing1.args == tuple()
        assert postprocessing1.keywords["individual_fns"] == [
            scale_two,
        ]
        assert postprocessing1.keywords["slices"] == [slice(0, 1)]

        results = (1.0,)
        expected = (3.0,)  # 2.0 * 1.0 + 1.0
        assert fn(results) == expected

        # Test reverse direction

        prog_reverse = TransformProgram((container2, container1))
        new_batch, fn = prog_reverse(batch)

        assert len(new_batch) == 1
        assert new_batch[0] is tape0

        assert fn.func is _apply_postprocessing_stack
        assert fn.args == tuple()
        assert len(fn.keywords["postprocessing_stack"]) == 2

        postprocessing0 = fn.keywords["postprocessing_stack"][0]
        assert postprocessing0.func is _batch_postprocessing
        assert postprocessing0.args == tuple()
        assert postprocessing0.keywords["individual_fns"] == [
            scale_two,
        ]
        assert postprocessing0.keywords["slices"] == [slice(0, 1)]

        postprocessing1 = fn.keywords["postprocessing_stack"][1]
        assert postprocessing1.func is _batch_postprocessing
        assert postprocessing1.args == tuple()
        assert postprocessing1.keywords["individual_fns"] == [
            add_one,
        ]
        assert postprocessing1.keywords["slices"] == [slice(0, 1)]

        results = (1.0,)
        expected = (4.0,)  # (1.0 + 1.0) * 2.0
        assert fn(results) == expected

    def test_postprocessing_batch_circuit_ragged(self):
        """Tests postprocessing when the input is a batch and the transform outputs different sizes of batches
        for each input tape.
        """

        # note this does not work for partitioned shots
        def sum_measurements(results: ResultBatch) -> Result:
            return sum(results)

        def split_sum_terms(tape):
            sum_obj = tape.measurements[0].obs
            new_tapes = tuple(
                QuantumScript(tape.operations, [qml.expval(o)], shots=tape.shots) for o in sum_obj
            )

            return new_tapes, sum_measurements

        container = TransformContainer(transform(split_sum_terms))
        prog = TransformProgram((container,))

        op = qml.Rot(1.2, 2.3, 3.4, wires=0)

        orig1 = qml.tape.QuantumScript([op], [qml.expval(qml.sum(qml.PauliX(0), qml.PauliZ(0)))])
        orig2 = qml.tape.QuantumScript(
            [op], [qml.expval(qml.sum(qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)))]
        )
        orig3 = qml.tape.QuantumScript(
            [op], [qml.expval(qml.sum(*(qml.PauliX(i) for i in range(5))))]
        )  # contributes 5 terms

        batch, fn = prog((orig1, orig2, orig3))

        assert len(batch) == 10

        assert fn.func is _apply_postprocessing_stack
        assert not fn.args
        fn_stack = fn.keywords["postprocessing_stack"]
        assert len(fn_stack) == 1

        assert fn_stack[0].func is _batch_postprocessing
        assert fn_stack[0].keywords["individual_fns"] == [
            sum_measurements,
            sum_measurements,
            sum_measurements,
        ]
        assert fn_stack[0].keywords["slices"] == [slice(0, 2), slice(2, 5), slice(5, 10)]

        dummy_results = (1, 2, 3, 4, 5, 1, 1, 1, 1, 1)
        assert fn(dummy_results) == (3, 12, 5)

    @pytest.mark.capture
    def test_call_jaxpr_empty(self):
        """Test that calling an empty TransformProgram with jaxpr returns untransformed ClosedJaxpr."""
        # pylint: disable=import-outside-toplevel
        import jax

        program = TransformProgram()
        const = jax.numpy.array(3.5)

        def f(x, n):
            qml.IsingXX(x, [0, 1])

            @qml.for_loop(n)
            def loop_fn(i):
                qml.Hadamard(i)
                qml.RX(const, i)

            loop_fn()
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.5, 5)
        transformed_jaxpr = program(jaxpr.jaxpr, jaxpr.consts, 1.5, 5)
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
        assert transformed_jaxpr.consts == jaxpr.consts

        for eqn1, eqn2 in zip(jaxpr.eqns, transformed_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive
            # Jaxpr equality is based on identity and so since they were constructed
            # seperately, they will not be equal (hence the string check)
            assert str(eqn1.params) == str(eqn2.params)

    @pytest.mark.capture
    def test_call_jaxpr_single_transform(self):
        """Test that calling a TransformProgram with a single transform with jaxpr works correctly."""
        # pylint: disable=import-outside-toplevel
        import jax

        program = TransformProgram()
        program.add_transform(qml.transforms.cancel_inverses)

        def f():
            qml.H(0)
            qml.X(1)
            qml.H(0)
            qml.X(1)
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        transformed_jaxpr = program(jaxpr.jaxpr, jaxpr.consts)
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)
        assert transformed_jaxpr.consts == jaxpr.consts

        assert len(transformed_jaxpr.eqns) == 2
        # pylint: disable=protected-access
        assert transformed_jaxpr.eqns[0].primitive == qml.PauliZ._primitive
        assert transformed_jaxpr.eqns[1].primitive == qml.measurements.ExpectationMP._obs_primitive

    @pytest.mark.capture
    def test_call_jaxpr_multiple_transforms(self):
        """Test that calling a TransformProgram with multiple transforms with jaxpr works correctly."""
        # pylint: disable=import-outside-toplevel
        import jax

        program = TransformProgram()
        program.add_transform(qml.transforms.cancel_inverses)
        program.add_transform(qml.transforms.defer_measurements, num_wires=3)
        program.add_transform(
            qml.transforms.decompose,
            gate_set=lambda op: op.name != "IsingXX",
        )

        def f():
            qml.H(0)
            qml.H(0)
            qml.measure(1)
            qml.IsingXX(0.5, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        jaxpr = jax.make_jaxpr(f)()
        transformed_jaxpr = program(jaxpr.jaxpr, jaxpr.consts)
        assert isinstance(transformed_jaxpr, jax.extend.core.ClosedJaxpr)

        # pylint: disable=protected-access
        isingxx_decomp = [qml.CNOT(wires=[0, 1]), qml.RX(0.5, wires=[0]), qml.CNOT(wires=[0, 1])]
        expected_primitives = [
            qml.CNOT._primitive,
            *[op._primitive for op in isingxx_decomp],
            qml.PauliZ._primitive,
            qml.measurements.ExpectationMP._obs_primitive,
        ]
        for eqn, expected_primitive in zip(
            transformed_jaxpr.eqns, expected_primitives, strict=True
        ):
            assert eqn.primitive == expected_primitive


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

    def test_qnode_integration_informative_transform(self):
        """Test the integration with QNode with two transforms, one of which is
        informative."""
        dispatched_transform_1 = transform(first_valid_transform)
        dispatched_transform_2 = transform(informative_transform, is_informative=True)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        new_qnode = dispatched_transform_2(dispatched_transform_1(qnode_circuit, 0))

        program = new_qnode.transform_program
        transformed_qnode_rep = repr(program)
        assert (
            transformed_qnode_rep
            == "TransformProgram("
            + str(first_valid_transform.__name__)
            + ", "
            + str(informative_transform.__name__)
            + ")"
        )

        assert not program.is_empty()
        assert len(program) == 2
        assert program[0].transform is first_valid_transform
        assert program[1].transform is informative_transform

        result = new_qnode(0.1)
        assert result == (3,)

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
