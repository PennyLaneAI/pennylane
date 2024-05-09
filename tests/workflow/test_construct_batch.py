# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains tests for the `qml.workflow.get_transform_program` getter and `construct_batch`.

"""
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms.core.transform_dispatcher import TransformContainer
from pennylane.transforms.core.transform_program import TransformProgram
from pennylane.workflow import construct_batch, get_transform_program
from pennylane.workflow.construct_batch import expand_fn_transform


def test_expand_fn_transform():
    """Tests the expand_fn_transform."""

    def my_expand_fn(tape, op1, op2=qml.S(0), op3=qml.S(0)):
        """my docstring."""
        return qml.tape.QuantumScript(
            tape.operations + [op1, op2, op3], tape.measurements, tape.shots
        )

    t = expand_fn_transform(my_expand_fn)

    assert isinstance(t, qml.transforms.core.TransformDispatcher)
    tape = qml.tape.QuantumScript([qml.S(0)], [qml.expval(qml.PauliZ(0))], shots=50)

    batch, fn = t(tape, qml.PauliX(0), op3=qml.T(0))
    assert len(batch) == 1
    expected = qml.tape.QuantumScript(
        [qml.S(0), qml.PauliX(0), qml.S(0), qml.T(0)], [qml.expval(qml.PauliZ(0))], shots=50
    )
    assert qml.equal(batch[0], expected)
    assert fn(("a",)) == "a"

    assert repr(t) == "<transform: my_expand_fn>"
    assert t.__doc__ == "my docstring."


class TestTransformProgramGetter:
    def test_bad_string_key(self):
        """Test a value error is raised if a bad string key is provided."""

        @qml.qnode(qml.device("default.qubit"))
        def circuit():
            return qml.state()

        with pytest.raises(ValueError, match=r"level bah not recognized."):
            get_transform_program(circuit, level="bah")

    def test_get_transform_program_gradient_fn_transform(self):
        """Tests for the transform program when the gradient_fn is a transform."""

        dev = qml.device("default.qubit", wires=4)

        @partial(qml.transforms.compile, num_passes=2)
        @partial(qml.transforms.merge_rotations, atol=1e-5)
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift", shifts=2)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        expected_p0 = TransformContainer(qml.transforms.cancel_inverses.transform)
        expected_p1 = TransformContainer(
            qml.transforms.merge_rotations.transform, kwargs={"atol": 1e-5}
        )
        expected_p2 = TransformContainer(qml.transforms.compile.transform, kwargs={"num_passes": 2})

        ps_expand_fn = TransformContainer(
            qml.gradients.param_shift.expand_transform, kwargs={"shifts": 2}
        )

        p0 = get_transform_program(circuit, level=0)
        assert isinstance(p0, TransformProgram)
        assert len(p0) == 0

        p0 = get_transform_program(circuit, level="top")
        assert isinstance(p0, TransformProgram)
        assert len(p0) == 0

        p_grad = get_transform_program(circuit, level="gradient")
        assert isinstance(p_grad, TransformProgram)
        assert len(p_grad) == 4
        assert p_grad == TransformProgram([expected_p0, expected_p1, expected_p2, ps_expand_fn])

        p_dev = get_transform_program(circuit, level="device")
        assert isinstance(p_grad, TransformProgram)
        p_default = get_transform_program(circuit)
        p_none = get_transform_program(circuit, None)
        assert p_dev == p_default
        assert p_none == p_dev
        assert len(p_dev) == 9
        assert p_dev == p_grad + dev.preprocess()[0]

        # slicing
        p_sliced = get_transform_program(circuit, slice(2, 7, 2))
        assert len(p_sliced) == 3
        assert p_sliced[0].transform == qml.compile.transform
        assert p_sliced[1].transform == qml.devices.preprocess.validate_device_wires.transform
        assert p_sliced[2].transform == qml.devices.preprocess.decompose.transform

    def test_gradient_fn_device_gradient(self):
        """Test that if level="gradient" but the gradient does not have preprocessing, the program is strictly user transforms."""

        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("default.qubit"), diff_method="backprop")
        def circuit():
            return qml.state()

        prog = get_transform_program(circuit, level="gradient")
        assert len(prog) == 1
        assert qml.transforms.cancel_inverses in prog

    def test_get_transform_program_device_gradient(self):
        """Test the trnsform program contents when using a device derivative."""

        dev = qml.device("default.qubit")

        @qml.transforms.sum_expand
        @qml.qnode(dev, diff_method="adjoint", device_vjp=False)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        full_prog = get_transform_program(circuit)
        assert len(full_prog) == 13

        config = qml.devices.ExecutionConfig(
            gradient_method="adjoint", use_device_jacobian_product=False
        )
        dev_program = dev.preprocess(config)[0]

        expected = TransformProgram()
        expected.add_transform(qml.transforms.sum_expand)
        expected += dev_program
        assert full_prog == expected

    def test_get_transform_program_legacy_device_interface(self):
        """Test the contents of the transform program with the legacy device interface."""

        dev = qml.device("default.qubit.legacy", wires=5)

        @qml.transforms.merge_rotations
        @qml.qnode(dev, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        program = get_transform_program(circuit)

        m1 = TransformContainer(qml.transforms.merge_rotations.transform)
        m2 = TransformContainer(dev.batch_transform)
        assert program[0:2] == TransformProgram([m1, m2])

        # a little hard to check the contents of a expand_fn_transform
        # this is the best proxy I can find
        assert program[2].transform.__wrapped__ == dev.expand_fn

    def test_get_transform_program_final_transform(self):
        """Test that gradient preprocessing and device transform occur before a final transform."""

        @qml.metric_tensor
        @qml.compile
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
        def circuit():
            qml.IsingXX(1.234, wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

        user_program = get_transform_program(circuit, level="user")
        assert len(user_program) == 2
        assert user_program[0].transform == qml.compile.transform
        assert user_program[1].transform == qml.metric_tensor.expand_transform

        grad_program = get_transform_program(circuit, level="gradient")
        assert len(grad_program) == 3
        assert grad_program[0].transform == qml.compile.transform
        assert grad_program[1].transform == qml.metric_tensor.expand_transform
        assert grad_program[2].transform == qml.gradients.param_shift.expand_transform

        dev_program = get_transform_program(circuit, level="device")
        assert len(dev_program) == 3 + len(circuit.device.preprocess()[0])  # currently 8
        assert qml.metric_tensor not in dev_program

        full = get_transform_program(circuit)
        assert full[-1].transform == qml.metric_tensor.transform


@qml.transforms.merge_rotations
@qml.transforms.cancel_inverses
@qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
def circuit1(weights, order):
    qml.RandomLayers(weights, wires=(0, 1))
    qml.Permute(order, wires=(0, 1, 2))
    qml.PauliX(0)
    qml.PauliX(0)
    qml.RX(0.1, wires=0)
    qml.RX(-0.1, wires=0)
    return qml.expval(qml.PauliX(0))


class TestConstructBatch:
    """Tests for the construct_batch function."""

    def test_level_zero(self):
        """Test that level zero is purely the queued circuit."""

        order = [2, 1, 0]
        weights = np.array([[1.0, 20]])
        batch, fn = construct_batch(circuit1, level=0)(weights, order, shots=10)

        assert len(batch) == 1
        expected_ops = [
            qml.RandomLayers(weights, wires=(0, 1)),
            qml.Permute(order, wires=(0, 1, 2)),
            qml.PauliX(0),
            qml.PauliX(0),
            qml.RX(0.1, wires=0),
            qml.RX(-0.1, wires=0),
        ]

        expected = qml.tape.QuantumScript(
            expected_ops,
            [qml.expval(qml.PauliX(0))],
            shots=10,
        )
        assert qml.equal(batch[0], expected)

        assert fn(("a",)) == ("a",)

    def test_first_transform(self):
        """Test that the first user transform can be selected by level=1"""

        weights = np.array([[1.0, 2.0]])
        order = [2, 1, 0]

        batch, fn = construct_batch(circuit1, level=1)(weights, order=order, shots=50)
        assert len(batch) == 1

        expected_ops = [
            qml.RandomLayers(weights, wires=(0, 1)),
            qml.Permute(order, wires=(0, 1, 2)),
            # cancel inverses
            qml.RX(0.1, wires=0),
            qml.RX(-0.1, wires=0),
        ]

        expected = qml.tape.QuantumScript(expected_ops, [qml.expval(qml.PauliX(0))], shots=50)
        assert qml.equal(batch[0], expected)
        assert fn(("a",)) == ("a",)

    @pytest.mark.parametrize("level", (2, "user"))
    def test_all_user_transforms(self, level):
        """Test that all user transforms can be selected and run."""

        weights = np.array([[1.0, 2.0]])
        order = [2, 1, 0]

        batch, fn = construct_batch(circuit1, level=level)(weights, order=order, shots=50)
        assert len(batch) == 1

        expected_ops = [
            qml.RandomLayers(weights, wires=(0, 1)),
            qml.Permute(order, wires=(0, 1, 2)),
            # cancel inverses
            # merge rotations
        ]

        expected = qml.tape.QuantumScript(expected_ops, [qml.expval(qml.PauliX(0))], shots=50)
        assert qml.equal(batch[0], expected)
        assert fn(("a",)) == ("a",)

    @pytest.mark.parametrize("level", (3, "gradient"))
    def test_gradient_transforms(self, level):
        """Test that the gradient transform can be selected with an integer or keyword."""
        weights = qml.numpy.array([[1.0, 2.0]], requires_grad=True)
        order = [2, 1, 0]
        batch, fn = construct_batch(circuit1, level=level)(weights=weights, order=order)

        expected = qml.tape.QuantumScript(
            [
                qml.RY(qml.numpy.array(1), 0),
                qml.RX(qml.numpy.array(2), 1),
                qml.Permute(order, (0, 1, 2)),
            ],
            [qml.expval(qml.PauliX(0))],
        )
        assert qml.equal(batch[0], expected)
        assert len(batch) == 1
        assert fn(("a",)) == ("a",)

    @pytest.mark.parametrize("level", ("device", None))
    def test_device_transforms(self, level):
        """Test that all device transforms can be run with the device keyword."""

        weights = np.array([[1.0, 2.0]])
        order = [2, 1, 0]

        batch, fn = construct_batch(circuit1, level=level)(weights, order)

        expected = qml.tape.QuantumScript(
            [qml.RY(1, 0), qml.RX(2, 1), qml.SWAP((0, 2))], [qml.expval(qml.PauliX(0))]
        )
        assert qml.equal(batch[0], expected)
        assert len(batch) == 1
        assert fn(("a",)) == ("a",)

    @pytest.mark.parametrize("level", ("device", None))
    def test_device_transforms_legacy_interface(self, level):
        """Test that the device transforms can be selected with level=device or None without trainable parameters"""

        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("default.qubit.legacy", wires=2, shots=50))
        def circuit(order):
            qml.Permute(order, wires=(0, 1, 2))
            qml.X(0)
            qml.X(0)
            return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0))]

        batch, fn = qml.workflow.construct_batch(circuit, level=level)((2, 1, 0))

        expected0 = qml.tape.QuantumScript(
            [qml.SWAP((0, 2))], [qml.expval(qml.PauliX(0))], shots=50
        )
        assert qml.equal(expected0, batch[0])
        expected1 = qml.tape.QuantumScript(
            [qml.SWAP((0, 2))], [qml.expval(qml.PauliY(0))], shots=50
        )
        assert qml.equal(expected1, batch[1])
        assert len(batch) == 2

        assert fn((1.0, 2.0)) == ((1.0, 2.0),)

    def test_final_transform(self):
        """Test that the final transform is included when level=None."""

        @qml.gradients.param_shift
        @qml.transforms.merge_rotations
        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RX(x, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        batch, fn = construct_batch(circuit, level=None)(0.5)
        assert len(batch) == 2
        expected0 = qml.tape.QuantumScript(
            [qml.RX(1.0 + np.pi / 2, 0)], [qml.expval(qml.PauliZ(0))]
        )
        assert qml.equal(batch[0], expected0)
        expected1 = qml.tape.QuantumScript(
            [qml.RX(1.0 - np.pi / 2, 0)], [qml.expval(qml.PauliZ(0))]
        )
        assert qml.equal(batch[1], expected1)

        dummy_res = (1.0, 2.0)
        expected_res = (1.0 - 2.0) / 2
        assert qml.numpy.allclose(fn(dummy_res)[0], expected_res)

    def test_user_transform_multiple_tapes(self):
        """Test a user transform that creates multiple tapes."""

        @qml.transforms.split_non_commuting
        @qml.qnode(qml.device("default.qubit", shots=10))
        def circuit():
            qml.S(0)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

        batch, fn = construct_batch(circuit, level="user")()

        assert len(batch) == 2
        expected0 = qml.tape.QuantumScript(
            [qml.S(0)], [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))], shots=10
        )
        assert qml.equal(expected0, batch[0])

        expected1 = qml.tape.QuantumScript([qml.S(0)], [qml.expval(qml.PauliZ(0))], shots=10)
        assert qml.equal(expected1, batch[1])

        dummy_res = (("x0", "x1"), "z0")
        expected_res = (("x0", "z0", "x1"),)
        assert fn(dummy_res) == expected_res

    def test_slicing_level(self):
        """Test that the level can be a slice."""

        @qml.transforms.merge_rotations
        @qml.qnode(qml.device("default.qubit"))
        def circuit(x):
            qml.RX(x, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        # by slicing starting at one, we do not run the merge rotations transform
        batch, fn = construct_batch(circuit, slice(1, None))(0.5)

        assert len(batch) == 1
        expected = qml.tape.QuantumScript(
            [qml.RX(0.5, 0), qml.RX(0.5, 0)], [qml.expval(qml.PauliZ(0))]
        )
        assert qml.equal(batch[0], expected)
        assert fn(("a",)) == ("a",)

    def test_qfunc_with_shots_arg(self):
        """Test that the tape uses device shots only when qfunc has a shots kwarg"""

        dev = qml.device("default.qubit", shots=100)

        @qml.qnode(dev)
        def circuit(shots):
            for _ in range(shots):
                qml.S(0)
            return qml.expval(qml.PauliZ(0))

        batch, fn = construct_batch(circuit, level=None)(shots=2)
        assert len(batch) == 1
        expected = qml.tape.QuantumScript(
            [qml.S(0), qml.S(0)], [qml.expval(qml.PauliZ(0))], shots=100
        )
        assert qml.equal(batch[0], expected)
        assert fn(("a",)) == ("a",)
