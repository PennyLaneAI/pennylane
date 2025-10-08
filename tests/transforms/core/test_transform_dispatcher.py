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
import inspect
from collections.abc import Callable, Sequence
from functools import partial

import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumTape
from pennylane.transforms.core import (
    TransformContainer,
    TransformDispatcher,
    TransformError,
    transform,
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


class TestTransformContainer:
    """Tests for the TransformContainer dataclass."""

    def test_repr(self):
        """Tests for the repr of a transform container."""
        t1 = qml.transforms.core.TransformContainer(
            qml.transforms.compile, kwargs={"num_passes": 2}
        )
        assert repr(t1) == "<compile((), {'num_passes': 2})>"

    def test_equality(self):
        """Tests that we can compare TransformContainer objects with the '==' and '!=' operators."""

        t1 = TransformContainer(qml.transforms.compile, kwargs={"num_passes": 2})
        t2 = TransformContainer(qml.transforms.compile, kwargs={"num_passes": 2})
        t3 = TransformContainer(qml.transforms.transpile, kwargs={"coupling_map": [(0, 1), (1, 2)]})

        t5 = TransformContainer(qml.transforms.merge_rotations, args=(1e-6,))
        t6 = TransformContainer(qml.transforms.merge_rotations, args=(1e-7,))

        # test for equality of identical transformers
        assert t1 == t2

        # test for inequality of different transformers
        assert t1 != t3
        assert t2 != t3
        assert t1 != 2
        assert t5 != t6
        assert t5 != t1

        # Test equality with the same args
        t5_copy = TransformContainer(qml.transforms.merge_rotations, args=(1e-6,))
        assert t5 == t5_copy

    @pytest.mark.jax  # needs jax to have non-none plxpr transform
    def test_the_transform_container_attributes(self):
        """Test the transform container attributes."""
        container = qml.transforms.core.TransformContainer(
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

        assert container.transform is first_valid_transform
        assert container.args == (0,)
        assert not container.kwargs
        assert container.classical_cotransform is None
        assert container.plxpr_transform is not None  # tape fallback
        assert not container.is_informative
        assert not container.final_transform

    def test_dispatch_container(self):
        """Test that transform containers can be called on objects with their various stored args and kwargs."""

        def postprocessing(results):
            return results[0]

        @qml.transform
        def repeat_ops(tape, n, new_ops=()):
            return (tape.copy(ops=tape.operations * n + list(new_ops)),), postprocessing

        container = TransformContainer(repeat_ops, (3,), {"new_ops": [qml.X(0)]})

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
        """Test that a TransformContainer can still be constructed in the old way."""

        c = TransformContainer(first_valid_transform, is_informative=True)

        # pylint: disable=protected-access
        assert isinstance(c._transform_dispatcher, TransformDispatcher)
        assert c.is_informative
        assert c._transform_dispatcher.is_informative  # pylint: disable=protected-access

    def test_error_if_extra_kwargs_when_dispatcher(self):
        """Test that a ValueError is raised if extra kwargs are passed when a TransformDispatcher is provided."""

        with pytest.raises(
            ValueError, match="cannot be passed if a TransformDispatcher is provided"
        ):
            _ = TransformContainer(qml.transform(first_valid_transform), is_informative=True)


class TestTransformDispatcher:  # pylint: disable=too-many-public-methods
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

        dispatched_transform = transform(first_valid_transform)

        with pytest.raises(
            TransformError,
            match=r"Functions that are wrapped / decorated with qjit cannot subsequently",
        ):
            dispatched_transform(cost)

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

        assert qnode_transformed.device is qnode_circuit.device

        with dev.tracker:
            qnode_circuit(0.1)
        assert dev.tracker.totals["executions"] == 1

        assert isinstance(qnode_transformed, qml.QNode)
        assert isinstance(qnode_transformed.transform_program, qml.transforms.core.TransformProgram)
        assert isinstance(
            qnode_transformed.transform_program.pop_front(), qml.transforms.core.TransformContainer
        )
        assert dispatched_transform.is_informative is False

    def test_integration_dispatcher_with_informative_transform(self):
        """Test that no error is raised with the transform function and that the transform dispatcher returns
        the right object when an informative transform is applied."""

        dispatched_transform = transform(informative_transform, is_informative=True)

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
        assert isinstance(qnode_transformed.transform_program, qml.transforms.core.TransformProgram)
        assert isinstance(
            qnode_transformed.transform_program.pop_front(), qml.transforms.core.TransformContainer
        )
        assert dispatched_transform.is_informative

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
    def test_integration_dispatcher_with_valid_transform_decorator_fails(self, valid_transform):
        """Test that an error is raised with the transform function."""

        dispatched_transform = transform(valid_transform)
        targs = [0]

        msg = r"Decorating a QNode with @transform_fn\(\*\*transform_kwargs\) has been removed"
        with pytest.raises(TransformError, match=msg):

            @dispatched_transform(targs)
            @qml.qnode(device=dev)
            def qnode_circuit(a):  # pylint:disable=unused-variable
                """QNode circuit."""
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=0)
                qml.RZ(a, wires=1)
                return qml.expval(qml.PauliZ(wires=0))

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
        assert expand_transform_container.args == (0,)
        assert expand_transform_container.kwargs == {}
        assert expand_transform_container.classical_cotransform is None
        assert not expand_transform_container.is_informative

        transform_container = qnode_transformed.transform_program.pop_front()

        assert isinstance(transform_container, qml.transforms.core.TransformContainer)
        assert transform_container.args == (0,)
        assert transform_container.kwargs == {}
        assert transform_container.classical_cotransform is None
        assert not expand_transform_container.is_informative

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_dispatcher_signature_classical_cotransform(self, valid_transform):
        """Test that  valid transforms with non-valid co transform raises a Transform error."""

        with pytest.raises(
            TransformError, match="The classical co-transform must be a valid Python function."
        ):
            transform(valid_transform, classical_cotransform=3)

    def test_error_not_callable_transform(self):
        """Test that a non-callable is not a valid transforms."""

        with pytest.raises(TransformError, match="The function to register, "):
            transform(non_callable)

    def test_expand_transform_not_callable(self):
        """Test that an expand transform must be a callable otherwise it is not valid."""

        with pytest.raises(
            TransformError, match="The expand function must be a valid Python function."
        ):
            transform(first_valid_transform, expand_transform=non_callable)

    def test_multiple_args_expand_transform(self):
        """Test that an expand transform must match the signature of the transform"""

        with pytest.raises(
            TransformError,
            match="The expand transform must have the same signature as the transform",
        ):
            transform(first_valid_transform, expand_transform=non_valid_expand_transform)

    def test_qfunc_transform_multiple_tapes(self):
        """Test that quantum function is not compatible with multiple tapes."""
        dispatched_transform = transform(second_valid_transform)
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
        dispatched_transform = transform(informative_transform, is_informative=True)

        num_ops = dispatched_transform(tape)
        assert num_ops == 4

    def test_dispatched_transform_attribute(self):
        """Test the dispatcher attributes."""
        dispatched_transform = transform(first_valid_transform)

        assert dispatched_transform.transform is first_valid_transform
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

        dispatched_transform1 = transform(valid_transform)
        dispatched_transform2 = transform(valid_transform)

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

        dispatched_transform = transform(valid_transform)

        history = []

        @dispatched_transform.custom_qnode_transform
        def _custom_qnode_transform(self, qnode, targs, tkwargs):
            history.append((targs, tkwargs))
            return self.default_qnode_transform(qnode, targs, tkwargs)

        @partial(dispatched_transform, index=0)
        @qml.qnode(dev)
        def qnode1():
            """QNode circuit."""
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        assert isinstance(qnode1, qml.QNode)
        assert isinstance(qnode1.transform_program, qml.transforms.core.TransformProgram)
        assert isinstance(
            qnode1.transform_program.pop_front(), qml.transforms.core.TransformContainer
        )

        @qml.qnode(dev)
        def qnode2():
            """QNode circuit."""
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0))

        qnode2 = dispatched_transform(qnode2, 1)

        assert isinstance(qnode2, qml.QNode)
        assert isinstance(qnode2.transform_program, qml.transforms.core.TransformProgram)
        assert isinstance(
            qnode2.transform_program.pop_front(), qml.transforms.core.TransformContainer
        )

        # check that the custom qnode transform was called
        assert history == [([], {"index": 0}), ([1], {})]

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

        dispatched_transform = transform(first_valid_transform)
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
                prog = qml.transforms.core.TransformProgram()
                prog.add_transform(qml.defer_measurements)
                prog.add_transform(qml.compile)
                return prog

            def execute(self, circuits, execution_config=None):
                return [0] * len(circuits)

        _dev = DummyDev()

        dispatched_transform = transform(valid_transform)
        new_dev = dispatched_transform(_dev, index=0)

        assert new_dev.original_device is _dev
        assert repr(new_dev).startswith("Transformed Device")

        program = _dev.preprocess_transforms()
        new_program = new_dev.preprocess_transforms()

        assert isinstance(program, qml.transforms.core.TransformProgram)
        assert isinstance(new_program, qml.transforms.core.TransformProgram)

        assert len(program) == 2
        assert len(new_program) == 3

        assert new_program[-1].transform is valid_transform

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

        dispatched_transform = transform(valid_transform)
        new_dev = dispatched_transform(device, index=0)

        assert new_dev.original_device is device
        assert repr(new_dev).startswith("Transformed Device")

        program = device.preprocess_transforms()
        new_program = new_dev.preprocess_transforms()

        assert isinstance(program, qml.transforms.core.TransformProgram)
        assert isinstance(new_program, qml.transforms.core.TransformProgram)

        assert len(program) == 3
        assert len(new_program) == 4

        assert new_program[-1].transform is valid_transform

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
            dispatched_transform = transform(valid_transform, is_informative=True)
            dispatched_transform(dev, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support final transforms."
        ):
            dispatched_transform = transform(valid_transform, final_transform=True)
            dispatched_transform(dev, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support expand transforms."
        ):
            dispatched_transform = transform(valid_transform, expand_transform=valid_transform)
            dispatched_transform(dev, index=0)

    @pytest.mark.parametrize("valid_transform", valid_transforms)
    def test_old_device_transform_error(self, valid_transform):
        """Test that the old device transform returns errors."""
        device = qml.devices.LegacyDeviceFacade(DefaultQubitLegacy(wires=2))

        with pytest.raises(
            TransformError, match="Device transform does not support informative transforms."
        ):
            dispatched_transform = transform(valid_transform, is_informative=True)
            dispatched_transform(device, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support final transforms."
        ):
            dispatched_transform = transform(valid_transform, final_transform=True)
            dispatched_transform(device, index=0)

        with pytest.raises(
            TransformError, match="Device transform does not support expand transforms."
        ):
            dispatched_transform = transform(valid_transform, expand_transform=valid_transform)
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
