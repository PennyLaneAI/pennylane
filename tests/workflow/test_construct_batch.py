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
Contains tests for `construct_batch`.

"""

import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qp
from pennylane.exceptions import PennyLaneDeprecationWarning
from pennylane.workflow import construct_batch


class TestMarker:
    """Tests the integration with the QNode."""

    def test_level_not_found(self):
        """Test the error message when a requested level is not found."""

        @qp.marker("something")
        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        expected = (
            r"Level bla not found in transform program. "
            r"Builtin options are 'top', 'user', 'device', and 'gradient'."
            r" Custom levels are \['something'\]."
        )
        with pytest.raises(ValueError, match=expected):
            construct_batch(c, level="bla")()

    def test_accessing_custom_level(self):
        """Test that custom levels can be specified and accessed."""

        @qp.transforms.merge_rotations
        @qp.marker("my_level")
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

        @qp.marker("my_level")
        @qp.qnode(qp.device("default.qubit"))
        def c(x):
            qp.RX(x, 0)
            return qp.expval(qp.Z(0))

        res = c(0.5)
        assert qp.math.allclose(res, np.cos(0.5))


def test_get_transform_program_is_deprecated():
    """Tests that the 'get_transform_program' function is deprecated."""

    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        return qp.state()

    with pytest.warns(
        PennyLaneDeprecationWarning, match="The 'get_transform_program' function is deprecated"
    ):
        _ = qp.workflow.get_transform_program(circuit)


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

    @pytest.mark.parametrize("level", [[], 0.5, True])
    def test_level_not_recognized(self, level):
        """Tests when the level is not recognized."""

        @qp.qnode(qp.device("null.qubit"))
        def c():
            return qp.state()

        with pytest.raises(ValueError, match=r"level .* not recognized"):
            construct_batch(c, level=level)()

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

        batch, fn = construct_batch(qp.set_shots(circuit1, shots=50), level=1)(weights, order=order)
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

        expected0 = qp.tape.QuantumScript([qp.SWAP((0, 2))], [qp.expval(qp.PauliX(0))], shots=50)
        qp.assert_equal(expected0, batch[0])
        expected1 = qp.tape.QuantumScript([qp.SWAP((0, 2))], [qp.expval(qp.PauliY(0))], shots=50)
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
        expected0 = qp.tape.QuantumScript([qp.RX(1.0 + np.pi / 2, 0)], [qp.expval(qp.PauliZ(0))])
        qp.assert_equal(batch[0], expected0)
        expected1 = qp.tape.QuantumScript([qp.RX(1.0 - np.pi / 2, 0)], [qp.expval(qp.PauliZ(0))])
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
        expected = qp.tape.QuantumScript([qp.S(0), qp.S(0)], [qp.expval(qp.PauliZ(0))], shots=100)
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
