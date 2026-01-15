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
# pylint: disable=unused-argument
import inspect
from collections.abc import Callable, Sequence

import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.transforms.core import (
    BoundTransform,
    Transform,
    TransformError,
)
from pennylane.typing import PostprocessingFn, TensorLike

dev = qml.device("default.qubit", wires=2)

with QuantumTape() as tape_circuit:
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=0)
    qml.RZ(0.42, wires=1)
    qml.expval(qml.PauliZ(wires=0))


def qfunc_circuit(a: qml.typing.TensorLike):
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
    circuit: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transform without tape."""
    circuit = circuit.copy()
    circuit._ops.pop(index)  # pylint:disable=protected-access
    return [circuit], lambda x: x


def no_quantum_tape_transform(
    tape: qml.operation.Operator, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transform with wrong hinting."""
    tape = tape.copy()
    tape._ops.pop(index)  # pylint:disable=protected-access
    return [tape], lambda x: x


def no_processing_fn_transform(tape: QuantumScript) -> QuantumScriptBatch:
    """Transform without processing fn."""
    tape_copy = tape.copy()
    return [tape, tape_copy]


def no_tape_sequence_transform(tape: QuantumScript) -> tuple[QuantumScript, PostprocessingFn]:
    """Transform wihtout Sequence return."""
    return tape, lambda x: x


def no_callable_return(tape: QuantumScript) -> tuple[QuantumScriptBatch, QuantumScript]:
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
    tape: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid transform."""
    tape = tape.copy()
    tape._ops.pop(index)  # pylint:disable=protected-access
    _ = (qml.PauliX(0), qml.S(0))
    return [tape], lambda x: x


def second_valid_transform(
    tape: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid trasnform."""
    tape1 = tape.copy()
    tape2 = tape.copy()
    tape._ops.pop(index)  # pylint:disable=protected-access

    def fn(results):
        return qml.math.sum(results)

    return [tape1, tape2], fn


valid_transforms = [first_valid_transform, second_valid_transform]


##########################################
# Valid expand transform
def expand_transform(
    tape: QuantumScript, index: int
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Multiple args expand fn."""
    tape._ops.pop(index)  # pylint:disable=protected-access
    return [tape], lambda x: x


# Non-valid expand transform
def non_valid_expand_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid expand transform."""
    return [tape], lambda x: x


##########################################
# Valid informative transform
def informative_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """A valid informative transform"""

    def fn(results):
        return len(results[0].operations)

    return [tape], fn


class TestBoundTransform:
    """Tests for the BoundTransform dataclass."""

    def test_repr(self):
        """Tests for the repr of a transform container."""
        t1 = BoundTransform(qml.transforms.merge_rotations, kwargs={"atol": 1e-6})
        assert repr(t1) == "<merge_rotations(atol=1e-06)>"

        t2 = BoundTransform(qml.transforms.merge_rotations, args=(1e-6,))
        assert repr(t2) == "<merge_rotations(1e-06)>"

        t3 = BoundTransform(
            qml.transforms.merge_rotations, args=(1e-6,), kwargs={"include_gates": ["RX"]}
        )
        assert repr(t3) == "<merge_rotations(1e-06, include_gates=['RX'])>"

    def test_equality_and_hash(self):
        """Tests that we can compare BoundTransform objects with the '==' and '!=' operators."""

        t1 = BoundTransform(qml.transforms.compile, kwargs={"num_passes": 2})
        t2 = BoundTransform(qml.transforms.compile, kwargs={"num_passes": 2})
        t3 = BoundTransform(qml.transforms.transpile, kwargs={"coupling_map": ((0, 1), (1, 2))})

        t5 = BoundTransform(qml.transforms.merge_rotations, args=(1e-6,))
        t6 = BoundTransform(qml.transforms.merge_rotations, args=(1e-7,))

        my_name1 = qml.transform(pass_name="my_name1")
        my_name2 = qml.transform(pass_name="my_name2")
        t7 = BoundTransform(my_name1, args=(0.5,))
        t7_duplicate = BoundTransform(my_name1, args=(0.5,))
        t8 = BoundTransform(my_name2, args=(0.5,))

        # test for equality of identical transformers
        assert t1 == t2

        # test for inequality of different transformers
        assert t1 != t3
        assert t2 != t3
        assert t1 != 2
        assert t5 != t6
        assert t5 != t1
        assert t7 != t8
        assert t7 == t7_duplicate

        assert hash(t1) == hash(t2)
        assert hash(t1) != hash(t3)
        assert hash(t5) != hash(t6)
        assert hash(t7) == hash(t7_duplicate)
        assert hash(t7) != hash(t8)

        # Test equality with the same args
        t5_copy = BoundTransform(qml.transforms.merge_rotations, args=(1e-6,))
        assert t5 == t5_copy

    @pytest.mark.jax  # needs jax to have non-none plxpr transform
    def test_the_transform_container_attributes(self):
        """Test the transform container attributes."""
        container = qml.transforms.core.BoundTransform(
            qml.transform(first_valid_transform), args=[0], kwargs={}
        )

        q_transform, args, kwargs, cotransform, plxpr_transform, is_informative, final_transform = (
            container
        )

        assert q_transform is first_valid_transform
        assert args == (0,)
        assert kwargs == {}
        assert cotransform is None
        assert plxpr_transform is not None  # fallback
        assert not is_informative
        assert not final_transform

        assert container.tape_transform is first_valid_transform
        assert container.args == (0,)
        assert not container.kwargs
        assert container.classical_cotransform is None
        assert container.plxpr_transform is not None  # tape fallback
        assert not container.is_informative
        assert not container.is_final_transform

    def test_dispatch_container(self):
        """Test that transform containers can be called on objects with their various stored args and kwargs."""

        def postprocessing(results):
            return results[0]

        @qml.transform
        def repeat_ops(tape, n, new_ops=()):
            return (tape.copy(ops=tape.operations * n + list(new_ops)),), postprocessing

        container = BoundTransform(repeat_ops, kwargs={"n": 3, "new_ops": [qml.X(0)]})

        tape = qml.tape.QuantumScript([qml.X(0)])
        expected = qml.tape.QuantumScript([qml.X(0), qml.X(0), qml.X(0), qml.X(0)])
        [out], fn = container(tape)
        assert fn is postprocessing
        qml.assert_equal(expected, out)

        @qml.qnode(qml.device("default.qubit"))
        def c():
            qml.Y(0)
            return qml.state()

        new_c = container(c)
        assert container == new_c.transform_program[0]

    def test_construction_fallback(self):
        """Test that a BoundTransform can still be constructed in the old way."""

        c = BoundTransform(first_valid_transform, is_informative=True)

        # pylint: disable=protected-access
        assert isinstance(c._transform, Transform)
        assert c.is_informative
        assert c._transform.is_informative  # pylint: disable=protected-access

    def test_error_if_extra_kwargs_when_dispatcher(self):
        """Test that a ValueError is raised if extra kwargs are passed when a Transform is provided."""

        with pytest.raises(ValueError, match="cannot be passed if a transform is provided"):
            _ = BoundTransform(qml.transform(first_valid_transform), is_informative=True)


class TestTransformExtension:
    @pytest.mark.parametrize("explicit_type", (True, False))
    def test_generic_register(self, explicit_type):
        """Test that generic_register can register behavior for a new object."""

        # pylint: disable=too-few-public-methods
        class Subroutine:
            def __init__(self, ops):
                self.ops = ops

        def subroutine_func(obj: Subroutine, transform, *targs, **tkwargs):
            tape = qml.tape.QuantumScript(obj.ops)
            [new_tape], _ = transform(tape, *targs, **tkwargs)
            return Subroutine(new_tape.operations)

        if explicit_type:
            Transform.generic_register(Subroutine)(subroutine_func)
        else:
            Transform.generic_register(subroutine_func)

        @qml.transform
        def dummy_transform(tape, op, n_times):
            tape = qml.tape.QuantumScript(tape.operations + [op for _ in range(n_times)])
            return (tape,), lambda res: res[0]

        new_subroutine = dummy_transform(Subroutine([qml.X(0), qml.X(0)]), qml.Y(1), 3)
        assert isinstance(new_subroutine, Subroutine)
        assert new_subroutine.ops == [qml.X(0), qml.X(0), qml.Y(1), qml.Y(1), qml.Y(1)]

        new_subroutine = dummy_transform.generic_apply_transform(
            Subroutine([qml.X(0), qml.X(0)]), qml.Y(1), 3
        )
        assert isinstance(new_subroutine, Subroutine)
        assert new_subroutine.ops == [qml.X(0), qml.X(0), qml.Y(1), qml.Y(1), qml.Y(1)]

    def test_register(self):
        """Test that transform specific behavior."""

        @qml.transform
        def dummy_transform(tape):
            return (tape.copy(ops=tape.operations[:3]),), lambda x: x[0]

        @dummy_transform.register
        def _(
            tape: qml.tape.QuantumScript,
        ):  # pylint: disable=redefined-outer-name, unused-argument
            return (tape.copy(ops=tape.operations[:1]),), lambda x: x[0]

        input = qml.tape.QuantumScript([qml.X(0), qml.X(1), qml.X(2), qml.X(3), qml.X(4), qml.X(5)])

        [overridden], _ = dummy_transform(input)
        qml.assert_equal(overridden, qml.tape.QuantumScript([qml.X(0)]))

        # propagates to other applications
        tape2 = qml.tape.QuantumScript([qml.Y(0), qml.Y(1), qml.Y(2), qml.Y(3)])
        [overridden1, overridden2], _ = dummy_transform((input, tape2))
        qml.assert_equal(overridden1, qml.tape.QuantumScript([qml.X(0)]))
        qml.assert_equal(overridden2, qml.tape.QuantumScript([qml.Y(0)]))

        # generic apply transform still works
        [generic_output], _ = dummy_transform.generic_apply_transform(input)
        qml.assert_equal(generic_output, qml.tape.QuantumScript([qml.X(0), qml.X(1), qml.X(2)]))


class TestTransform:  # pylint: disable=too-many-public-methods
    """Test the transform function (validate and dispatch)."""

    @pytest.mark.catalyst
    @pytest.mark.external
    def test_error_on_qjit(self):
        """Test that an error is raised on when applying a transform to a qjit object."""

        pytest.importorskip("catalyst")

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def cost():
            qml.RY(0.1, wires=0)
            qml.RY(0.1, wires=0)
            return qml.expval(qml.Z(0))

        dispatched_transform = qml.transform(first_valid_transform)

        with pytest.raises(
            TransformError,
            match=r"Functions that are wrapped / decorated with qjit cannot subsequently",
        ):
            dispatched_transform(cost)

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_integration_dispatcher_with_valid_transform(self, valid_transform):
        """Test that no error is raised with the transform function and that the transform dispatcher returns
        the right object."""

        dispatched_transform = qml.transform(valid_transform)

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

        assert qnode_transformed.device is qnode_circuit.device

        with dev.tracker:
            qnode_circuit(0.1)
        assert dev.tracker.totals["executions"] == 1

        assert isinstance(qnode_transformed, qml.QNode)
        assert isinstance(qnode_transformed.transform_program, qml.CompilePipeline)
        assert isinstance(
            qnode_transformed.transform_program.pop(0), qml.transforms.core.BoundTransform
        )
        assert dispatched_transform.is_informative is False

    def test_integration_dispatcher_with_informative_transform(self):
        """Test that no error is raised with the transform function and that the transform dispatcher returns
        the right object when an informative transform is applied."""

        dispatched_transform = qml.transform(informative_transform, is_informative=True)

        # Applied on a tape (return processed results)
        expected = len(tape_circuit.operations)
        num_ops = dispatched_transform(tape_circuit)
        assert num_ops == expected

        # Applied on a qfunc (return a qfunc)
        qfunc = dispatched_transform(qfunc_circuit)
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

        qnode_transformed = dispatched_transform(qnode_circuit)
        assert not qnode_circuit.transform_program

        assert qnode_transformed(0.1) == 4
        assert isinstance(qnode_transformed, qml.QNode)
        assert isinstance(qnode_transformed.transform_program, qml.CompilePipeline)
        assert isinstance(
            qnode_transformed.transform_program.pop(0), qml.transforms.core.BoundTransform
        )
        assert dispatched_transform.is_informative

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_integration_dispatcher_with_valid_transform_decorator_partial(self, valid_transform):
        """Test that no error is raised with the transform function and that the transform dispatcher returns
        the right object."""

        dispatched_transform = qml.transform(valid_transform)
        targs = [0]

        @dispatched_transform(*targs)
        @qml.qnode(device=dev)
        def qnode_circuit(a):
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.PauliZ(wires=0))

        assert isinstance(qnode_circuit, qml.QNode)
        assert isinstance(qnode_circuit.transform_program, qml.CompilePipeline)
        assert isinstance(
            qnode_circuit.transform_program.pop(0), qml.transforms.core.BoundTransform
        )

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_integration_dispatcher_with_invalid_dispatch_target_returns_container(
        self, valid_transform
    ):
        """Test that calling a transform dispatcher with args that are not a valid dispatch target
        returns a BoundTransform with the supplied args and kwargs."""

        dispatched_transform = qml.transform(valid_transform)
        targs = [0]

        # Calling with invalid dispatch target should return a BoundTransform
        container = dispatched_transform(*targs)
        assert isinstance(container, qml.transforms.core.BoundTransform)
        assert container.args == ()
        assert container.kwargs == {"index": 0}

        # Test with kwargs as well
        container_with_kwargs = dispatched_transform(index=0)
        assert isinstance(container_with_kwargs, qml.transforms.core.BoundTransform)
        assert container_with_kwargs.args == ()
        assert container_with_kwargs.kwargs == {"index": 0}

    def test_combining_dispatcher_and_container(self):
        """Test that a dispatcher can be combined with a container using the + operator."""

        @qml.transform
        def dispatched_transform(tape, key, another):
            return (tape,), lambda res: res[0]

        kwargs_container = dispatched_transform(key="value", another="kwarg")

        program = dispatched_transform + kwargs_container
        assert isinstance(program, qml.CompilePipeline)
        assert len(program) == 2
        assert program[0].args == ()
        assert program[1].kwargs == {"key": "value", "another": "kwarg"}

        @qml.transform
        def transform2(tape, x):
            return (tape,), lambda res: res[0]

        args_container = BoundTransform(transform2, args=(0,))

        program = args_container + dispatched_transform
        assert isinstance(program, qml.CompilePipeline)
        assert len(program) == 2
        assert program[0].args == (0,)
        assert program[1].args == ()

    def test_kwargs_only_returns_container(self):
        """Test that calling a transform dispatcher with only kwargs returns a BoundTransform.

        This enables patterns like:
            decompose(gate_set=gate_set) + merge_rotations(1e-6)
        where decompose might be called with only keyword arguments.
        """

        @qml.transform
        def dispatched_transform(tape, key, another):
            return (tape,), lambda res: res[0]

        # Calling with only kwargs should return a BoundTransform
        container = dispatched_transform(key="value", another="kwarg")
        assert isinstance(container, qml.transforms.core.BoundTransform)
        assert container.args == ()
        assert container.kwargs == {"key": "value", "another": "kwarg"}

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_missing_obj_without_kwargs_errors(self, valid_transform):
        """Test that calling a dispatcher without arguments raises the expected TypeError."""

        dispatched_transform = qml.transform(valid_transform)

        with pytest.raises(
            TypeError,
            match="requires at least one argument",
        ):
            dispatched_transform()

    def test_queuing_qfunc_transform(self):
        """Test that queuing works with the transformed quantum function."""

        dispatched_transform = qml.transform(first_valid_transform)

        # Applied on a tape
        tapes, fn = dispatched_transform(tape_circuit, 0)

        assert isinstance(tapes, Sequence)
        assert callable(fn)

        # Applied on a qfunc (return a qfunc)
        qfunc_transformed = dispatched_transform(qfunc_circuit, 0)
        assert callable(qfunc_transformed)

        assert inspect.signature(qfunc_transformed) == inspect.signature(qfunc_circuit)

        with QuantumTape() as transformed_tape:
            qfunc_transformed(0.42)

        assert isinstance(transformed_tape, QuantumScript)
        assert transformed_tape.circuit is not None
        assert len(transformed_tape.circuit) == 4

        with QuantumTape() as tape:
            qfunc_circuit(0.42)

        assert isinstance(transformed_tape, QuantumScript)
        assert tape.circuit is not None
        assert len(tape.circuit) == 5

    def test_qnode_with_expand_transform(self):
        """Test qnode with a transform program and expand transform."""

        dispatched_transform = qml.transform(
            first_valid_transform, expand_transform=expand_transform
        )

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

        assert isinstance(qnode_transformed.transform_program, qml.CompilePipeline)
        expand_transform_container = qnode_transformed.transform_program.pop(0)
        assert isinstance(expand_transform_container, qml.transforms.core.BoundTransform)
        assert expand_transform_container.args == ()
        assert expand_transform_container.kwargs == {"index": 0}
        assert expand_transform_container.classical_cotransform is None
        assert not expand_transform_container.is_informative

        transform_container = qnode_transformed.transform_program.pop(0)

        assert isinstance(transform_container, qml.transforms.core.BoundTransform)
        assert transform_container.args == ()
        assert transform_container.kwargs == {"index": 0}
        assert transform_container.classical_cotransform is None
        assert not expand_transform_container.is_informative

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_dispatcher_signature_classical_cotransform(self, valid_transform):
        """Test that  valid transforms with non-valid co transform raises a Transform error."""

        with pytest.raises(
            TransformError, match="The classical co-transform must be a valid Python function."
        ):
            qml.transform(valid_transform, classical_cotransform=3)

    def test_error_not_callable_transform(self):
        """Test that a non-callable is not a valid transforms."""

        with pytest.raises(TransformError, match="The function to register, "):
            qml.transform(non_callable)

    def test_expand_transform_not_callable(self):
        """Test that an expand transform must be a callable otherwise it is not valid."""

        with pytest.raises(
            TransformError, match="The expand function must be a valid Python function."
        ):
            qml.transform(first_valid_transform, expand_transform=non_callable)

    def test_qfunc_transform_multiple_tapes(self):
        """Test that quantum function is not compatible with multiple tapes."""
        dispatched_transform = qml.transform(second_valid_transform)
        with pytest.raises(
            TransformError, match="Impossible to dispatch your transform on quantum function"
        ):
            dispatched_transform(qfunc_circuit, 0)(0.42)

    def test_informative_transform_tape_return(self):
        """Test that disaptched informative transforms return processed results instead of
        a list of tapes and processing function."""
        tape = qml.tape.QuantumScript(
            [qml.PauliX(0), qml.CNOT([0, 1]), qml.RX(0.234, 1), qml.Hadamard(1)]
        )
        dispatched_transform = qml.transform(informative_transform, is_informative=True)

        num_ops = dispatched_transform(tape)
        assert num_ops == 4

    def test_dispatched_transform_attribute(self):
        """Test the dispatcher attributes."""
        dispatched_transform = qml.transform(first_valid_transform)

        assert dispatched_transform.tape_transform is first_valid_transform
        assert dispatched_transform.expand_transform is None
        assert dispatched_transform.classical_cotransform is None

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    @pytest.mark.parametrize("batch_type", (tuple, list))
    def test_batch_transform(self, valid_transform, batch_type, num_margin=1e-8):
        """Test that dispatcher can dispatch onto a batch of tapes."""

        def check_batch(batch):
            return isinstance(batch, Sequence) and all(
                isinstance(tape, qml.tape.QuantumScript) for tape in batch
            )

        def comb_postproc(results: TensorLike, fn1: Callable, fn2: Callable):
            return fn1(fn2(results))

        # Create a simple device and tape
        tmp_dev = qml.device("default.qubit", wires=3)

        H = qml.Hamiltonian(
            [0.5, 1.0, 1.0], [qml.PauliZ(2), qml.PauliY(2) @ qml.PauliZ(1), qml.PauliZ(1)]
        )
        measur = [qml.expval(H)]
        ops = [qml.Hadamard(0), qml.RX(0.2, 0), qml.RX(0.6, 0), qml.CNOT((0, 1))]
        tape = QuantumScript(ops, measur)

        ############################################################
        ### Test with two elementary user-defined transforms
        ############################################################

        dispatched_transform1 = qml.transform(valid_transform)
        dispatched_transform2 = qml.transform(valid_transform)

        batch1, fn1 = dispatched_transform1(tape, index=0)
        assert check_batch(batch1)

        batch2, fn2 = dispatched_transform2(batch1, index=0)
        assert check_batch(batch2)

        result = tmp_dev.execute(batch2)
        assert isinstance(result, TensorLike)

        ############################################################
        ### Test with two `concrete` transforms
        ############################################################

        tape = QuantumScript(ops, measur)

        batch1, fn1 = qml.transforms.split_non_commuting(tape)
        assert check_batch(batch1)

        batch2, fn2 = qml.transforms.merge_rotations(batch1)
        assert check_batch(batch2)

        result = tmp_dev.execute(batch2)
        assert isinstance(result, TensorLike)

        # check that final batch and post-processing functions are what we expect after the two transforms
        fin_ops = [qml.Hadamard(0), qml.RX(0.8, 0), qml.CNOT([0, 1])]
        tp1 = QuantumScript(fin_ops, [qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(1))])
        tp2 = QuantumScript(fin_ops, [qml.expval(qml.PauliY(2) @ qml.PauliZ(1))])
        fin_batch = batch_type([tp1, tp2])

        for tapeA, tapeB in zip(fin_batch, batch2):
            qml.assert_equal(tapeA, tapeB)
        assert abs(comb_postproc(result, fn1, fn2).item() - 0.5) < num_margin

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_custom_qnode_transform(self, valid_transform):
        """Test that the custom qnode transform is correctly executed"""

        dispatched_transform = qml.transform(valid_transform)

        history = []

        @dispatched_transform.custom_qnode_transform
        def _custom_qnode_transform(self, qnode, targs, tkwargs):
            history.append((targs, tkwargs))
            return self.default_qnode_transform(qnode, targs, tkwargs)

        @dispatched_transform(index=0)
        @qml.qnode(dev)
        def qnode1():
            """QNode circuit."""
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        assert isinstance(qnode1, qml.QNode)
        assert isinstance(qnode1.transform_program, qml.CompilePipeline)
        assert isinstance(qnode1.transform_program.pop(0), qml.transforms.core.BoundTransform)

        @qml.qnode(dev)
        def qnode2():
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0))

        qnode2 = dispatched_transform(qnode2, 1)

        assert isinstance(qnode2, qml.QNode)
        assert isinstance(qnode2.transform_program, qml.CompilePipeline)
        assert isinstance(qnode2.transform_program.pop(0), qml.transforms.core.BoundTransform)

        # check that the custom qnode transform was called
        assert history == [((), {"index": 0}), ((), {"index": 1})]

    @pytest.mark.parametrize(
        "fn, type_",
        [(list, list), (tuple, tuple), (qml.numpy.array, qml.numpy.ndarray)],
    )
    def test_qfunc_transform_multiple_measurements(self, fn, type_):
        """Ensure that return type is preserved with qfunc transforms."""

        def qfunc():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            qml.PauliZ(1)
            return fn([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))])

        dispatched_transform = qml.transform(first_valid_transform)
        transformed_qfunc = dispatched_transform(qfunc, 2)
        qnode = qml.QNode(transformed_qfunc, qml.device("default.qubit"))
        result = qnode()
        assert isinstance(result, type_)

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_device_transform(self, valid_transform):
        """Test a device transform."""

        class DummyDev(qml.devices.Device):
            # pylint: disable=unused-argument
            def preprocess_transforms(self, execution_config=None):
                prog = qml.CompilePipeline()
                prog.add_transform(qml.defer_measurements)
                prog.add_transform(qml.compile)
                return prog

            def execute(self, circuits, execution_config=None):
                return [0] * len(circuits)

        _dev = DummyDev()

        dispatched_transform = qml.transform(valid_transform)
        new_dev = dispatched_transform(_dev, index=0)

        assert new_dev.original_device is _dev
        assert repr(new_dev).startswith("Transformed Device")

        program = _dev.preprocess_transforms()
        new_program = new_dev.preprocess_transforms()

        assert isinstance(program, qml.CompilePipeline)
        assert isinstance(new_program, qml.CompilePipeline)

        assert len(program) == 2
        assert len(new_program) == 3

        assert new_program[-1].tape_transform is valid_transform

        @qml.qnode(new_dev)
        def circuit():
            qml.PauliX(0)
            return qml.state()

        circuit()

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_old_device_transform(self, valid_transform):
        """Test a device transform."""
        device = qml.devices.LegacyDeviceFacade(
            DefaultQubitLegacy(wires=2)
        )  # pylint: disable=redefined-outer-name

        dispatched_transform = qml.transform(valid_transform)
        new_dev = dispatched_transform(device, index=0)

        assert new_dev.original_device is device
        assert repr(new_dev).startswith("Transformed Device")

        config = device.setup_execution_config()
        program = device.preprocess_transforms(config)
        config = new_dev.setup_execution_config()
        new_program = new_dev.preprocess_transforms(config)

        assert isinstance(program, qml.CompilePipeline)
        assert isinstance(new_program, qml.CompilePipeline)

        assert len(program) == 3
        assert len(new_program) == 4

        assert new_program[-1].tape_transform is valid_transform

        @qml.qnode(new_dev)
        def circuit():
            qml.PauliX(0)
            return qml.state()

        circuit()

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_device_transform_error(self, valid_transform):
        """Test that the device transform returns errors."""

        with pytest.raises(
            TransformError, match="Device transform does not support informative transforms."
        ):
            dispatched_transform = qml.transform(valid_transform, is_informative=True)
            dispatched_transform(dev, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support final transforms."
        ):
            dispatched_transform = qml.transform(valid_transform, final_transform=True)
            dispatched_transform(dev, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support expand transforms."
        ):
            dispatched_transform = qml.transform(valid_transform, expand_transform=valid_transform)
            dispatched_transform(dev, index=0)

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_old_device_transform_error(self, valid_transform):
        """Test that the old device transform returns errors."""
        device = qml.devices.LegacyDeviceFacade(DefaultQubitLegacy(wires=2))

        with pytest.raises(
            TransformError, match="Device transform does not support informative transforms."
        ):
            dispatched_transform = qml.transform(valid_transform, is_informative=True)
            dispatched_transform(device, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support final transforms."
        ):
            dispatched_transform = qml.transform(valid_transform, final_transform=True)
            dispatched_transform(device, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support expand transforms."
        ):
            dispatched_transform = qml.transform(valid_transform, expand_transform=valid_transform)
            dispatched_transform(device, index=0)

    def test_sphinx_build(self, monkeypatch):
        """Test that transforms are not created during Sphinx builds"""
        monkeypatch.setenv("SPHINX_BUILD", "1")

        with pytest.warns(UserWarning, match="Transforms have been disabled, as a Sphinx"):

            @qml.transform
            def custom_transform(  # pylint:disable=unused-variable
                tape: QuantumScript, index: int
            ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
                """A valid transform."""
                tape = tape.copy()
                tape._ops.pop(index)  # pylint:disable=protected-access
                return [tape], lambda x: x


def dummy_fn():
    return qml.state()


dummy_qnode = qml.QNode(dummy_fn, qml.device("default.qubit"))


class TestSetupInputs:

    def test_default_applies_defaults_args(self):
        """Test that the default implementation of setup_inputs fills in default inputs."""

        @qml.transform
        def f(tape, x=1, b=2):
            return (tape,), lambda res: res[0]

        bound_t = f(5)
        assert bound_t.args == ()
        assert bound_t.kwargs == {"x": 5, "b": 2}

    def test_eager_error_on_bad_input(self):
        """Test that an eager error is provided on binding a transform with bad inputs."""

        @qml.transform
        def f(tape, val=1):
            return (tape,), lambda res: res[0]

        with pytest.raises(TypeError, match="got an unexpected keyword argument 'bad'"):
            f(bad=3)

    @pytest.mark.parametrize(
        "target", (dummy_qnode, qml.device("default.qubit"), qml.CompilePipeline())
    )
    def test_eager_error_on_bad_input_dispatch(self, target):
        """Test that an eager error is provided on binding a transform with bad inputs when dispatched onto various objects.."""

        @qml.transform
        def f(tape, val=1):
            return (tape,), lambda res: res[0]

        with pytest.raises(TypeError, match="got an unexpected keyword argument 'bad'"):
            f(target, bad=3)

    def test_use_setup_input(self):
        """Test that custom setup_input functions can be provided and are run at dispatch time."""

        def setup_inputs(x):
            if not isinstance(x, int):
                raise ValueError("not an int")
            return (x,), {}

        def func(tape, x):
            return (tape,), lambda res: res[0]

        t = qml.transform(func, setup_inputs=setup_inputs)

        with pytest.raises(ValueError, match="not an int"):
            t(x="a")

        bound_t = t(x=1)
        assert bound_t.args == (1,)
        assert bound_t.kwargs == {}

        bound_t = t(1)
        assert bound_t.args == (1,)
        assert bound_t.kwargs == {}


class TestPassName:

    def test_no_pass_name_or_tape_def(self):
        """Test that an error is raised if neither a tape def or pass name are provided."""

        with pytest.raises(ValueError, match="must define either a tape transform or a pass_name"):
            qml.transform()

    def test_providing_pass_name_with_tape_def(self):
        """Test a pass_name and a tape def can both be applied."""

        def my_tape_def(tape):
            return (tape,), lambda x: x[0]

        t = qml.transform(my_tape_def, "my_pass_name")

        assert t.tape_transform == my_tape_def
        assert t.pass_name == "my_pass_name"

        assert repr(t) == "<transform: my_tape_def>"

        c = BoundTransform(t)
        assert repr(c) == "<my_tape_def()>"

    def test_providing_pass_name_without_tape_def(self):
        """Test that a transform can be defined by a pass_name without a tape based transform."""

        t = qml.transform(pass_name="my_pass_name")
        assert t.tape_transform is None
        assert t.pass_name == "my_pass_name"

        assert repr(t) == "<transform: my_pass_name>"

        tape = qml.tape.QuantumScript()
        with pytest.raises(NotImplementedError, match="has no defined tape implementation"):
            t(tape)

        with pytest.raises(NotImplementedError, match="has no defined tape implementation"):
            t((tape, tape))

        @t
        @qml.qnode(qml.device("null.qubit"))
        def c():
            return qml.expval(qml.Z(0))

        expected_container = BoundTransform(t)
        assert expected_container.pass_name == "my_pass_name"
        assert repr(expected_container) == "<my_pass_name()>"
        assert expected_container.tape_transform is None
        assert c.transform_program[-1] == expected_container
        assert repr(c.transform_program) == "CompilePipeline(my_pass_name)"

        with pytest.raises(NotImplementedError, match="has no defined tape transform"):
            c.transform_program((tape,))

        with pytest.raises(NotImplementedError, match="has no defined tape transform"):
            c()
