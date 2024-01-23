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
Contains tests for the `qml.workflow.transform_program` getter.

"""
from functools import partial

import numpy as np

import pennylane as qml
from pennylane.transforms.core.transform_dispatcher import TransformContainer
from pennylane.transforms.core.transform_program import TransformProgram
from pennylane.workflow import transform_program, construct_batch


class TestTransformProgramGetter:
    def test_transform_program_gradient_fn(self):
        """Tests for the transform program when the gradient_fn is a transform."""

        dev = qml.device("default.qubit", wires=4)

        @partial(qml.transforms.compile, num_passes=2)
        @partial(qml.transforms.merge_rotations, atol=1e-5)
        @qml.transforms.cancel_inverses
        @qml.qnode(dev, diff_method="parameter-shift", shifts=2)
        def circuit():
            return qml.expval(qml.PauliZ(0))

        expected_p0 = qml.transforms.core.TransformContainer(
            qml.transforms.cancel_inverses.transform
        )
        expected_p1 = qml.transforms.core.TransformContainer(
            qml.transforms.merge_rotations.transform, kwargs={"atol": 1e-5}
        )
        expected_p2 = qml.transforms.core.TransformContainer(
            qml.transforms.compile.transform, kwargs={"num_passes": 2}
        )

        ps_expand_fn = qml.transforms.core.TransformContainer(
            qml.gradients.param_shift.expand_transform, kwargs={"shifts": 2}
        )

        p0 = transform_program(circuit, level=0)
        assert isinstance(p0, TransformProgram)
        assert len(p0) == 0

        p0 = transform_program(circuit, level="top")
        assert isinstance(p0, TransformProgram)
        assert len(p0) == 0

        p_grad = transform_program(circuit, level="gradient")
        assert isinstance(p_grad, TransformProgram)
        assert len(p_grad) == 4
        assert p_grad == TransformProgram([expected_p0, expected_p1, expected_p2, ps_expand_fn])

        p_dev = transform_program(circuit, level="device")
        assert isinstance(p_grad, TransformProgram)
        p_default = transform_program(circuit)
        p_none = transform_program(circuit, None)
        assert p_dev == p_default
        assert p_none == p_dev
        assert len(p_dev) == 9
        assert p_dev == p_grad + dev.preprocess()[0]

        # slicing
        p_sliced = transform_program(circuit, slice(2, 7, 2))
        assert len(p_sliced) == 3
        assert p_sliced[0].transform == qml.compile.transform
        assert p_sliced[1].transform == qml.devices.preprocess.validate_device_wires.transform
        assert p_sliced[2].transform == qml.devices.preprocess.decompose.transform

    def test_transform_program_device_gradient(self):
        """Test the trnsform program contents when using a device derivative."""

        dev = qml.device("default.qubit")

        @qml.transforms.sum_expand
        @qml.qnode(dev, diff_method="adjoint", device_vjp=False)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        full_prog = transform_program(circuit)
        assert len(full_prog) == 13

        config = qml.devices.ExecutionConfig(
            gradient_method="adjoint", use_device_jacobian_product=False
        )
        dev_program = dev.preprocess(config)[0]

        expected = TransformProgram()
        expected.add_transform(qml.transforms.sum_expand)
        expected += dev_program
        assert full_prog == expected

    def test_transform_program_legacy_device_interface(self):
        """Test the contents of the transform program with the legacy device interface."""

        dev = qml.device("default.qubit.legacy", wires=5)

        @qml.transforms.merge_rotations
        @qml.qnode(dev, diff_method="backprop")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        program = transform_program(circuit)

        m1 = TransformContainer(qml.transforms.merge_rotations.transform)
        m2 = TransformContainer(dev.batch_transform)
        assert program[0:2] == TransformProgram([m1, m2])

        # a little hard to check the contents of a expand_fn_transform
        # this is the best proxy I can find
        assert program[2].transform.__wrapped__ == dev.expand_fn

    def test_transform_program_final_transform(self):
        """Test that gradient preprocessing and device transform occur before a final transform."""

        @qml.metric_tensor
        @qml.compile
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
        def circuit():
            qml.IsingXX(1.234, wires=(0, 1))
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))

        user_program = transform_program(circuit, level="user")
        assert len(user_program) == 2
        assert user_program[0].transform == qml.compile.transform
        assert user_program[1].transform == qml.metric_tensor.expand_transform

        grad_program = transform_program(circuit, level="gradient")
        assert len(grad_program) == 3
        assert grad_program[0].transform == qml.compile.transform
        assert grad_program[1].transform == qml.metric_tensor.expand_transform
        assert grad_program[2].transform == qml.gradients.param_shift.expand_transform

        dev_program = transform_program(circuit, level="device")
        assert len(dev_program) == 3 + len(circuit.device.preprocess()[0])  # currently 8
        assert qml.metric_tensor not in dev_program

        full = transform_program(circuit)
        assert full[-1].transform == qml.metric_tensor.transform


class TestConstructBatch:
    """Tests for the construct_batch function."""

    def test_level_zero(self):
        """Test that level zero is purely the queued circuit."""

        @qml.compile
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
        def circuit(order):
            qml.Permute(order, wires=(0, 1, 2))
            qml.PauliX(0)
            qml.PauliX(0)
            return qml.state()

        batch, fn = construct_batch(circuit, level=0)([2, 1, 0], shots=10)

        assert len(batch) == 1
        expected = qml.tape.QuantumScript(
            [qml.Permute([2, 1, 0], wires=(0, 1, 2)), qml.PauliX(0), qml.PauliX(0)],
            [qml.state()],
            shots=10,
        )
        assert qml.equal(batch[0], expected)

        assert fn(("a,")) == "a"

    def test_user_transforms(self):
        """Test that user transforms can be selected and applied."""

        @qml.transforms.cancel_inverses
        @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
        def circuit(weights):
            qml.RandomLayers(weights, wires=(0, 1))
            qml.PauliX(0)
            qml.PauliX(0)
            return qml.state()

        weights = np.array([[1.0, 2.0]])

        for level in ("user", 1):
            batch, fn = construct_batch(circuit, level=level)(weights, shots=50)
            assert len(batch) == 1

            expected = qml.tape.QuantumScript(
                [qml.RandomLayers(weights, wires=(0, 1))], [qml.state()], shots=50
            )
            assert qml.equal(batch[0], expected)
            assert fn(("a",)) == "a"
