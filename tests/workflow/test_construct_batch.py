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
Contains tests for the `qp.workflow.get_transform_program` getter and `construct_batch`.

"""

import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qp
from pennylane.transforms.core import BoundTransform, CompilePipeline
from pennylane.workflow import construct_batch, get_transform_program


class TestMarker:

    def test_level_not_found(self):
        """Test the error message when a requested level is not found."""

        @qp.marker(level="something")
        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        expected = (
            r"level bla not found in transform program. "
            r"Builtin options are 'top', 'user', 'device', and 'gradient'."
            r" Custom levels are \['something'\]."
        )
        with pytest.raises(ValueError, match=expected):
            construct_batch(c, level="bla")()

    def test_accessing_custom_level(self):
        """Test that custom levels can be specified and accessed."""

        @qp.transforms.merge_rotations
        @qp.marker(level="my_level")
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("null.qubit"))
        def c():
            qp.RX(0.2, 0)
            qp.X(0)
            qp.X(0)
            qp.RX(0.2, 0)
            return qp.state()

        (tape,), _ = construct_batch(c, level="my_level")()
        expected = qp.tape.QuantumScript([qp.RX(0.2, 0), qp.RX(0.2, 0)], [qp.state()])
        qp.assert_equal(tape, expected)

    def test_custom_level_as_arg(self):
        """Test that custom levels can be specified as positional arg and accessed."""

        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("null.qubit"))
        def c():
            qp.RX(0.2, 0)
            qp.X(0)
            qp.X(0)
            qp.RX(0.2, 0)
            return qp.state()

        c = qp.marker(c, "my_level")
        c = qp.transforms.merge_rotations(c)

        (tape,), _ = construct_batch(c, level="my_level")()
        expected = qp.tape.QuantumScript([qp.RX(0.2, 0), qp.RX(0.2, 0)], [qp.state()])
        qp.assert_equal(tape, expected)

    def test_execution_with_marker_transform(self):
        """Test that the marker transform does not effect execution results."""

        @qp.marker(level="my_level")
        @qp.qnode(qp.device("default.qubit"))
        def c(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        res = c(0.5)
        assert qp.math.allclose(res, np.cos(0.5))

    def test_tape_application(self):
        """Test that the tape transform leaves the input unaffected."""

        input = qp.tape.QuantumScript([qp.X(0)], [qp.state()])
        (out,), fn = qp.marker(input, level="level")
        assert input is out
        assert fn(("a",)) == "a"

    def test_uniqueness_checking(self):
        """Test an error is raised if a level is not unique."""

        @qp.marker(level="something")
        @qp.marker(level="something")
        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        with pytest.raises(ValueError, match="Found multiple markers for level something"):
            construct_batch(c)()

    def test_protected_levels(self):
        """Test an error is raised for using a protected level."""

        @qp.marker(level="gradient")
        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        with pytest.raises(ValueError, match="Found marker for protected level gradient."):
            construct_batch(c)()


class TestCompilePipelineGetter:
    def test_bad_string_key(self):
        """Test a value error is raised if a bad string key is provided."""

        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            return qp.state()

        with pytest.raises(ValueError, match=r"level bla not found in transform program."):
            get_transform_program(circuit, level="bla")

    def test_bad_other_key(self):
        """Test a value error is raised if a bad, unrecognized key is provided."""

        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            return qp.state()

        with pytest.raises(ValueError, match=r"not recognized."):
            get_transform_program(circuit, level=["bah"])

    def test_get_transform_program_diff_method_transform(self):
        """Tests for the transform program when the diff_method is a transform."""

        dev = qp.device("default.qubit", wires=4)

        @qp.transforms.compile(num_passes=2)
        @qp.transforms.merge_rotations(atol=1e-5)
        @qp.transforms.cancel_inverses
        @qp.qnode(dev, diff_method="parameter-shift", gradient_kwargs={"shifts": 2})
        def circuit():
            return qp.expval(qp.PauliZ(0))

        expected_p0 = BoundTransform(qp.transforms.cancel_inverses)
        expected_p1 = BoundTransform(qp.transforms.merge_rotations, kwargs={"atol": 1e-5})
        expected_p2 = BoundTransform(qp.transforms.compile, kwargs={"num_passes": 2})

        ps_expand_fn = BoundTransform(
            qp.transform(qp.gradients.param_shift.expand_transform), kwargs={"shifts": 2}
        )

        p0 = get_transform_program(circuit, level=0)
        assert isinstance(p0, CompilePipeline)
        assert len(p0) == 0

        p0 = get_transform_program(circuit, level="top")
        assert isinstance(p0, CompilePipeline)
        assert len(p0) == 0

        p_grad = get_transform_program(circuit, level="gradient")
        assert isinstance(p_grad, CompilePipeline)
        assert len(p_grad) == 4
        assert p_grad == CompilePipeline(expected_p0, expected_p1, expected_p2, ps_expand_fn)

        p_dev = get_transform_program(circuit, level="device")
        assert isinstance(p_grad, CompilePipeline)
        p_default = get_transform_program(circuit)
        assert p_dev == p_default

        assert len(p_dev) == 10
        config = qp.devices.ExecutionConfig(
            interface=getattr(circuit, "interface", None),
            mcm_config=qp.devices.MCMConfig(mcm_method="deferred"),
        )
        assert p_dev == p_grad + dev.preprocess_transforms(config)

        # slicing
        p_sliced = get_transform_program(circuit, slice(2, 7, 2))
        assert len(p_sliced) == 3
        assert p_sliced[0].tape_transform == qp.compile.tape_transform
        assert (
            p_sliced[2].tape_transform
            == qp.devices.preprocess.device_resolve_dynamic_wires.tape_transform
        )
        assert p_sliced[1].tape_transform == qp.defer_measurements.tape_transform

    def test_diff_method_device_gradient(self):
        """Test that if level="gradient" but the gradient does not have preprocessing, the program is strictly user transforms."""

        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("default.qubit"), diff_method="backprop")
        def circuit():
            return qp.state()

        prog = get_transform_program(circuit, level="gradient")
        assert len(prog) == 1
        assert qp.transforms.cancel_inverses in prog

    def test_get_transform_program_device_gradient(self):
        """Test the trnsform program contents when using a device derivative."""

        dev = qp.device("default.qubit")

        @qp.transforms.split_non_commuting
        @qp.qnode(dev, diff_method="adjoint", device_vjp=False)
        def circuit(x):
            qp.RX(x, 0)
            return qp.expval(qp.PauliZ(0))

        full_prog = get_transform_program(circuit)
        assert len(full_prog) == 14

        config = qp.devices.ExecutionConfig(
            interface=getattr(circuit, "interface", None),
            gradient_method="adjoint",
            use_device_jacobian_product=False,
        )
        config = dev.setup_execution_config(config)
        dev_program = dev.preprocess_transforms(config)

        expected = CompilePipeline()
        expected.add_transform(qp.transforms.split_non_commuting)
        expected += dev_program
        assert full_prog == expected

    def test_get_transform_program_legacy_device_interface(self):
        """Test the contents of the transform program with the legacy device interface."""

        dev = DefaultQubitLegacy(wires=5)

        @qp.transforms.merge_rotations
        @qp.qnode(dev, diff_method="backprop")
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0))

        program = get_transform_program(circuit)

        m1 = BoundTransform(qp.transforms.merge_rotations)
        assert program[:1] == CompilePipeline([m1])

        m2 = BoundTransform(qp.devices.legacy_facade.legacy_device_batch_transform)
        assert program[1].tape_transform == m2.tape_transform
        assert program[1].kwargs["device"] == dev

        # a little hard to check the contents of a expand_fn transform
        # this is the best proxy I can find
        assert (
            program[2].tape_transform
            == qp.devices.legacy_facade.legacy_device_expand_fn.tape_transform
        )

    def test_get_transform_program_final_transform(self):
        """Test that gradient preprocessing and device transform occur before a final transform."""

        @qp.metric_tensor
        @qp.compile
        @qp.qnode(qp.device("default.qubit"), diff_method="parameter-shift")
        def circuit():
            qp.IsingXX(1.234, wires=(0, 1))
            return qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliX(0))

        user_program = get_transform_program(circuit, level="user")
        assert len(user_program) == 3
        assert user_program[0].tape_transform == qp.compile.tape_transform
        assert user_program[1].tape_transform == qp.metric_tensor.expand_transform
        assert user_program[2].tape_transform == qp.metric_tensor.tape_transform

        grad_program = get_transform_program(circuit, level="gradient")
        assert len(grad_program) == 4
        assert grad_program[0].tape_transform == qp.compile.tape_transform
        assert grad_program[1].tape_transform == qp.gradients.param_shift.expand_transform
        assert grad_program[2].tape_transform == qp.metric_tensor.expand_transform
        assert grad_program[3].tape_transform == qp.metric_tensor.tape_transform

        dev_program = get_transform_program(circuit, level="device")
        config = qp.devices.ExecutionConfig(interface=getattr(circuit, "interface", None))
        config = qp.device("default.qubit").setup_execution_config(config)
        assert len(dev_program) == 4 + len(
            circuit.device.preprocess_transforms(config)
        )  # currently 8
        assert dev_program[-1].tape_transform == qp.metric_tensor.tape_transform

        full_program = get_transform_program(circuit)
        assert full_program[-1].tape_transform == qp.metric_tensor.tape_transform

        assert dev_program == full_program


@qp.transforms.merge_rotations
@qp.transforms.cancel_inverses
@qp.qnode(qp.device("default.qubit"), diff_method="parameter-shift")
def circuit1(weights, order):
    qp.RandomLayers(weights, wires=(0, 1))
    qp.Permute(order, wires=(0, 1, 2))
    qp.PauliX(0)
    qp.PauliX(0)
    qp.RX(0.1, wires=0)
    qp.RX(-0.1, wires=0)
    return qp.expval(qp.PauliX(0))


class TestConstructBatch:
    """Tests for the construct_batch function."""

    def test_level_zero(self):
        """Test that level zero is purely the queued circuit."""

        order = [2, 1, 0]
        weights = np.array([[1.0, 20]])
        batch, fn = construct_batch(qp.set_shots(circuit1, shots=10), level=0)(weights, order)

        assert len(batch) == 1
        expected_ops = [
            qp.RandomLayers(weights, wires=(0, 1)),
            qp.Permute(order, wires=(0, 1, 2)),
            qp.PauliX(0),
            qp.PauliX(0),
            qp.RX(0.1, wires=0),
            qp.RX(-0.1, wires=0),
        ]

        expected = qp.tape.QuantumScript(
            expected_ops, [qp.expval(qp.PauliX(0))], shots=10, trainable_params=[]
        )
        qp.assert_equal(batch[0], expected)

        assert fn(("a",)) == ("a",)

    def test_first_transform(self):
        """Test that the first user transform can be selected by level=1"""

        weights = np.array([[1.0, 2.0]])
        order = [2, 1, 0]

        batch, fn = construct_batch(qp.set_shots(circuit1, shots=50), level=1)(
            weights, order=order
        )
        assert len(batch) == 1

        expected_ops = [
            qp.RandomLayers(weights, wires=(0, 1)),
            qp.Permute(order, wires=(0, 1, 2)),
            # cancel inverses
            qp.RX(0.1, wires=0),
            qp.RX(-0.1, wires=0),
        ]

        expected = qp.tape.QuantumScript(expected_ops, [qp.expval(qp.PauliX(0))], shots=50)
        qp.assert_equal(batch[0], expected)
        assert fn(("a",)) == ("a",)

    @pytest.mark.parametrize("level", (2, "user"))
    def test_all_user_transforms(self, level):
        """Test that all user transforms can be selected and run."""

        weights = np.array([[1.0, 2.0]])
        order = [2, 1, 0]

        batch, fn = construct_batch(qp.set_shots(circuit1, shots=50), level=level)(
            weights, order=order
        )
        assert len(batch) == 1

        expected_ops = [
            qp.RandomLayers(weights, wires=(0, 1)),
            qp.Permute(order, wires=(0, 1, 2)),
            # cancel inverses
            # merge rotations
        ]

        expected = qp.tape.QuantumScript(expected_ops, [qp.expval(qp.PauliX(0))], shots=50)
        qp.assert_equal(batch[0], expected)
        assert fn(("a",)) == ("a",)

    @pytest.mark.parametrize("level", (3, "gradient"))
    def test_gradient_transforms(self, level):
        """Test that the gradient transform can be selected with an integer or keyword."""
        weights = qp.numpy.array([[1.0, 2.0]], requires_grad=True)
        order = [2, 1, 0]
        batch, fn = construct_batch(circuit1, level=level)(weights=weights, order=order)

        expected = qp.tape.QuantumScript(
            [
                qp.RY(qp.numpy.array(1), 0),
                qp.RX(qp.numpy.array(2), 1),
                qp.Permute(order, (0, 1, 2)),
            ],
            [qp.expval(qp.PauliX(0))],
        )
        qp.assert_equal(batch[0], expected)
        assert len(batch) == 1
        assert fn(("a",)) == ("a",)

    def test_device_transforms(self):
        """Test that all device transforms can be run with the device keyword."""

        weights = np.array([[1.0, 2.0]])
        order = [2, 1, 0]

        batch, fn = construct_batch(circuit1, level="device")(weights, order)

        expected = qp.tape.QuantumScript(
            [qp.RY(1, 0), qp.RX(2, 1), qp.SWAP((0, 2))], [qp.expval(qp.PauliX(0))]
        )
        qp.assert_equal(batch[0], expected)
        assert len(batch) == 1
        assert fn(("a",)) == ("a",)

    def test_device_transforms_legacy_interface(self):
        """Test that the device transforms can be selected with level=device or None without trainable parameters"""

        @qp.transforms.cancel_inverses
        @qp.set_shots(50)
        @qp.qnode(DefaultQubitLegacy(wires=2))
        def circuit(order):
            qp.Permute(order, wires=(0, 1, 2))
            qp.X(0)
            qp.X(0)
            return [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliY(0))]

        batch, fn = qp.workflow.construct_batch(circuit, level="device")((2, 1, 0))

        expected0 = qp.tape.QuantumScript(
            [qp.SWAP((0, 2))], [qp.expval(qp.PauliX(0))], shots=50
        )
        qp.assert_equal(expected0, batch[0])
        expected1 = qp.tape.QuantumScript(
            [qp.SWAP((0, 2))], [qp.expval(qp.PauliY(0))], shots=50
        )
        qp.assert_equal(expected1, batch[1])
        assert len(batch) == 2

        assert fn((1.0, 2.0)) == ((1.0, 2.0),)

    def test_final_transform(self):
        """Test that the final transform is included when level="device"."""

        @qp.gradients.param_shift
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("default.qubit"))
        def circuit(x):
            qp.RX(x, 0)
            qp.RX(x, 0)
            return qp.expval(qp.PauliZ(0))

        batch, fn = construct_batch(circuit, level="device")(0.5)
        assert len(batch) == 2
        expected0 = qp.tape.QuantumScript(
            [qp.RX(1.0 + np.pi / 2, 0)], [qp.expval(qp.PauliZ(0))]
        )
        qp.assert_equal(batch[0], expected0)
        expected1 = qp.tape.QuantumScript(
            [qp.RX(1.0 - np.pi / 2, 0)], [qp.expval(qp.PauliZ(0))]
        )
        qp.assert_equal(batch[1], expected1)

        dummy_res = (1.0, 2.0)
        expected_res = (1.0 - 2.0) / 2
        assert qp.numpy.allclose(fn(dummy_res)[0], expected_res)

    def test_user_transform_multiple_tapes(self):
        """Test a user transform that creates multiple tapes."""

        @qp.transforms.split_non_commuting
        @qp.set_shots(shots=10)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.S(0)
            return qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliX(1))

        batch, fn = construct_batch(circuit, level="user")()

        assert len(batch) == 2
        expected0 = qp.tape.QuantumScript(
            [qp.S(0)], [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliX(1))], shots=10
        )
        qp.assert_equal(expected0, batch[0])

        expected1 = qp.tape.QuantumScript([qp.S(0)], [qp.expval(qp.PauliZ(0))], shots=10)
        qp.assert_equal(expected1, batch[1])

        dummy_res = (("x0", "x1"), "z0")
        expected_res = (("x0", "z0", "x1"),)
        assert fn(dummy_res) == expected_res

    def test_slicing_level(self):
        """Test that the level can be a slice."""

        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("default.qubit"))
        def circuit(x):
            qp.RX(x, 0)
            qp.RX(x, 0)
            return qp.expval(qp.PauliZ(0))

        # by slicing starting at one, we do not run the merge rotations transform
        batch, fn = construct_batch(circuit, slice(1, None))(0.5)

        assert len(batch) == 1
        expected = qp.tape.QuantumScript(
            [qp.RX(0.5, 0), qp.RX(0.5, 0)], [qp.expval(qp.PauliZ(0))], trainable_params=[]
        )

        qp.assert_equal(batch[0], expected)
        assert fn(("a",)) == ("a",)

    def test_qfunc_with_shots_arg(self):
        """Test that the tape uses device shots only when qfunc has a shots kwarg"""

        dev = qp.device("default.qubit")

        with pytest.warns(UserWarning, match="Detected 'shots' as an argument"):

            @qp.set_shots(shots=100)
            @qp.qnode(dev)
            def circuit(shots):
                for _ in range(shots):
                    qp.S(0)
                return qp.expval(qp.PauliZ(0))

        batch, fn = construct_batch(circuit, level="device")(shots=2)

        assert len(batch) == 1
        expected = qp.tape.QuantumScript(
            [qp.S(0), qp.S(0)], [qp.expval(qp.PauliZ(0))], shots=100
        )
        qp.assert_equal(batch[0], expected)
        assert fn(("a",)) == ("a",)

    @pytest.mark.parametrize(
        "mcm_method, expected_op",
        [("deferred", qp.CNOT), ("tree-traversal", qp.ops.MidMeasure)],
    )
    def test_mcm_method(self, mcm_method, expected_op):
        """Test that the tape is constructed using the mcm_method specified on the QNode"""

        @qp.qnode(qp.device("default.qubit"), mcm_method=mcm_method)
        def circuit():
            qp.measure(0)
            return qp.expval(qp.Z(0))

        (tape,), _ = qp.workflow.construct_batch(circuit, level="device")()

        assert len(tape.operations) == 1
        assert isinstance(tape.operations[0], expected_op)
