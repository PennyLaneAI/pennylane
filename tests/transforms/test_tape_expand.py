# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for tape expansion stopping criteria and expansion functions.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane.wires import Wires


class TestCreateExpandFn:
    """Test creating expansion functions from stopping criteria."""

    crit_0 = (~qml.operation.is_trainable) | (qml.operation.has_gen & qml.operation.is_trainable)
    doc_0 = "Test docstring."
    with qml.tape.JacobianTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RY(qml.numpy.array(2.1, requires_grad=True), wires=1)
        qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

    def test_create_expand_fn(self):
        """Test creation of expand_fn."""
        expand_fn = qml.transforms.create_expand_fn(
            depth=10,
            stop_at=self.crit_0,
            docstring=self.doc_0,
        )
        assert expand_fn.__doc__ == "Test docstring."

    def test_create_expand_fn_expansion(self):
        """Test expansion with created expand_fn."""
        expand_fn = qml.transforms.create_expand_fn(depth=10, stop_at=self.crit_0)
        new_tape = expand_fn(self.tape)
        assert new_tape.operations[0] == self.tape.operations[0]
        assert new_tape.operations[1] == self.tape.operations[1]
        assert [op.name for op in new_tape.operations[2:]] == ["RZ", "RY", "RZ"]
        assert np.allclose([op.data for op in new_tape.operations[2:]], [[0.5], [0.2], [-0.1]])
        assert [op.wires for op in new_tape.operations[2:]] == [qml.wires.Wires(0)] * 3

    def test_create_expand_fn_dont_expand(self):
        """Test expansion is skipped with depth=0."""
        expand_fn = qml.transforms.create_expand_fn(depth=0, stop_at=self.crit_0)

        new_tape = expand_fn(self.tape)
        assert new_tape.operations == self.tape.operations

    def test_device_and_stopping_expansion(self, mocker):
        """Test that passing a device alongside a stopping condition ensures
        that all operations are expanded to match the devices default gate
        set"""
        dev = qml.device("default.qubit", wires=1)
        expand_fn = qml.transforms.create_expand_fn(device=dev, depth=10, stop_at=self.crit_0)

        with qml.tape.JacobianTape() as tape:
            qml.U1(0.2, wires=0)
            qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

        spy_device = mocker.spy(dev, "supports_operation")
        new_tape = expand_fn(tape)
        spy_device.assert_called()

        assert new_tape.operations[0].name == "PhaseShift"
        assert [op.name for op in new_tape.operations[1:]] == ["RZ", "RY", "RZ"]

    def test_device_only_expansion(self, mocker):
        """Test that passing a device ensures that all operations are expanded
        to match the devices default gate set"""
        dev = qml.device("default.qubit", wires=1)
        expand_fn = qml.transforms.create_expand_fn(device=dev, depth=10)

        with qml.tape.JacobianTape() as tape:
            qml.U1(0.2, wires=0)
            qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

        spy_device = mocker.spy(dev, "supports_operation")
        new_tape = expand_fn(tape)
        spy_device.assert_called()

        assert len(new_tape.operations) == 2
        assert new_tape.operations[0].name == "PhaseShift"
        assert new_tape.operations[1].name == "Rot"

    def test_depth_only_expansion(self):
        """Test that passing a depth simply expands to that depth"""
        dev = qml.device("default.qubit", wires=0)

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.RY(qml.numpy.array(2.1, requires_grad=True), wires=1)
            qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)
            qml.templates.StronglyEntanglingLayers(
                qml.numpy.ones([2, 2, 3], requires_grad=True), wires=[0, 1]
            )

        expand_fn = qml.transforms.create_expand_fn(depth=0)
        new_tape = expand_fn(tape)
        assert new_tape is tape

        expand_fn = qml.transforms.create_expand_fn(depth=10)
        new_tape = expand_fn(tape)
        assert new_tape.operations[0] == tape.operations[0]
        assert new_tape.operations[1] == tape.operations[1]
        assert [op.name for op in new_tape.operations[2:5]] == ["RZ", "RY", "RZ"]
        assert len(new_tape.operations[6:]) == 15


class TestExpandMultipar:
    """Test the expansion of multi-parameter gates."""

    def test_expand_multipar(self):
        """Test that a multi-parameter gate is decomposed correctly.
        And that single-parameter gates are not decomposed."""
        dev = qml.device("default.qubit", wires=3)

        class _CRX(qml.CRX):
            name = "_CRX"

            @staticmethod
            def decomposition(theta, wires):
                raise NotImplementedError()

        with qml.tape.QuantumTape() as tape:
            qml.RX(1.5, wires=0)
            qml.Rot(-2.1, 0.2, -0.418, wires=1)
            _CRX(1.5, wires=[0, 2])

        new_tape = qml.transforms.expand_multipar(tape)
        new_ops = new_tape.operations

        assert [op.name for op in new_ops] == ["RX", "RZ", "RY", "RZ", "_CRX"]

    def test_no_generator_expansion(self):
        """Test that a gate is decomposed correctly if it has
        generator[0]==None."""
        dev = qml.device("default.qubit", wires=3)

        class _CRX(qml.CRX):
            def generator(self):
                raise qml.operations.GeneratorUndefinedError()

        with qml.tape.QuantumTape() as tape:
            qml.RX(1.5, wires=0)
            qml.RZ(-2.1, wires=1)
            qml.RY(0.2, wires=1)
            qml.RZ(-0.418, wires=1)
            _CRX(1.5, wires=[0, 2])

        new_tape = qml.transforms.expand_multipar(tape)
        new_ops = new_tape.operations
        expected = ["RX", "RZ", "RY", "RZ", "RZ", "RY", "CNOT", "RY", "CNOT", "RZ"]
        assert [op.name for op in new_ops] == expected


class TestExpandNonunitaryGen:
    """Test the expansion of operations without a unitary generator."""

    def test_do_not_expand(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""
        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            qml.PauliRot(0.9, "XY", wires=[0, 1])
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        new_tape = qml.transforms.expand_nonunitary_gen(tape)

        assert tape.operations == new_tape.operations

    def test_expand_multi_par(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""
        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            qml.Rot(0.9, 1.2, -0.6, wires=0)
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        new_tape = qml.transforms.expand_nonunitary_gen(tape)
        expanded = [
            qml.RZ(0.9, wires=0),
            qml.RY(1.2, wires=0),
            qml.RZ(-0.6, wires=0),
        ]

        assert tape.operations[:2] == new_tape.operations[:2]
        assert all(exp.name == new.name for exp, new in zip(expanded, new_tape.operations[2:5]))
        assert all(exp.data == new.data for exp, new in zip(expanded, new_tape.operations[2:5]))
        assert all(exp.wires == new.wires for exp, new in zip(expanded, new_tape.operations[2:5]))
        assert tape.operations[3:] == new_tape.operations[5:]

    def test_expand_missing_generator(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""

        class _PhaseShift(qml.PhaseShift):
            def generator(self):
                return None

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            _PhaseShift(2.1, wires=1)
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        new_tape = qml.transforms.expand_nonunitary_gen(tape)
        assert tape.operations[:2] == new_tape.operations[:2]
        exp_op = new_tape.operations[2]
        assert exp_op.name == "RZ" and exp_op.data == [2.1] and exp_op.wires == qml.wires.Wires(1)
        assert tape.operations[3:] == new_tape.operations[3:]

    def test_expand_nonunitary_generator(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            qml.PhaseShift(2.1, wires=1)
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        new_tape = qml.transforms.expand_nonunitary_gen(tape)

        assert tape.operations[:2] == new_tape.operations[:2]
        exp_op = new_tape.operations[2]
        assert exp_op.name == "RZ" and exp_op.data == [2.1] and exp_op.wires == qml.wires.Wires(1)
        assert tape.operations[3:] == new_tape.operations[3:]


class TestExpandInvalidTrainable:
    """Tests for the gradient expand function"""

    def test_no_expansion(self, mocker):
        """Test that a circuit with differentiable
        operations is not expanded"""
        x = qml.numpy.array(0.2, requires_grad=True)
        y = qml.numpy.array(0.1, requires_grad=True)

        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(tape, "expand")
        new_tape = qml.transforms.expand_invalid_trainable(tape)

        assert new_tape is tape
        spy.assert_not_called()

    def test_trainable_nondiff_expansion(self, mocker):
        """Test that a circuit with non-differentiable
        trainable operations is expanded"""
        x = qml.numpy.array(0.2, requires_grad=True)
        y = qml.numpy.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qml.PhaseShift):
            grad_method = None

        with qml.tape.QuantumTape() as tape:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(tape, "expand")
        new_tape = qml.transforms.expand_invalid_trainable(tape)

        assert new_tape is not tape
        spy.assert_called()

        new_tape.operations[0].name == "RZ"
        new_tape.operations[0].grad_method == "A"
        new_tape.operations[1].name == "RY"
        new_tape.operations[2].name == "CNOT"

    def test_nontrainable_nondiff(self, mocker):
        """Test that a circuit with non-differentiable
        non-trainable operations is not expanded"""
        x = qml.numpy.array(0.2, requires_grad=False)
        y = qml.numpy.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qml.PhaseShift):
            grad_method = None

        with qml.tape.QuantumTape() as tape:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)
        assert tape.trainable_params == [1]

        spy = mocker.spy(tape, "expand")
        new_tape = qml.transforms.expand_invalid_trainable(tape)

        assert new_tape is tape
        spy.assert_not_called()

    def test_trainable_numeric(self, mocker):
        """Test that a circuit with numeric differentiable
        trainable operations is *not* expanded"""
        x = qml.numpy.array(0.2, requires_grad=True)
        y = qml.numpy.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qml.PhaseShift):
            grad_method = "F"

        with qml.tape.QuantumTape() as tape:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        spy = mocker.spy(tape, "expand")
        new_tape = qml.transforms.expand_invalid_trainable(tape)

        assert new_tape is tape
        spy.assert_not_called()


# Custom decomposition functions for testing.
def custom_cnot(wires):
    return [
        qml.Hadamard(wires=wires[1]),
        qml.CZ(wires=[wires[0], wires[1]]),
        qml.Hadamard(wires=wires[1]),
    ]


def custom_hadamard(wires):
    return [qml.RZ(np.pi, wires=wires), qml.RY(np.pi / 2, wires=wires)]


# Incorrect, for testing purposes only
def custom_rx(params, wires):
    return [qml.RY(params, wires=wires), qml.Hadamard(wires=wires)]


# To test the gradient; use circuit identity RY(theta) = X RY(-theta) X
def custom_rot(phi, theta, omega, wires):
    return [
        qml.RZ(phi, wires=wires),
        qml.PauliX(wires=wires),
        qml.RY(-theta, wires=wires),
        qml.PauliX(wires=wires),
        qml.RZ(omega, wires=wires),
    ]


# Decompose a template into another template
def custom_basic_entangler_layers(weights, wires):
    return [
        qml.AngleEmbedding(weights, wires=wires),
        qml.broadcast(qml.CNOT, pattern="ring", wires=wires),
    ]


class TestCreateCustomDecompExpandFn:
    """Tests for the gradient expand function"""

    def test_no_custom_decomp(self):
        """Test that sending an empty dictionary results in no decompositions."""

        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        original_dev = qml.device("default.qubit", wires=3)
        decomp_dev = qml.device("default.qubit", wires=3, custom_decomps={})

        original_qnode = qml.QNode(circuit, original_dev, expansion_strategy="device")
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")

        original_res = original_qnode()
        decomp_res = decomp_qnode()

        assert np.isclose(original_res, decomp_res)
        assert [
            orig_op.name == decomp_op.name
            for orig_op, decomp_op in zip(
                original_qnode.qtape.operations, decomp_qnode.qtape.operations
            )
        ]

    def test_no_custom_decomp_template(self):
        """Test that sending an empty dictionary results in no decomposition
        when a template is involved, except the decomposition expected from the device."""

        def circuit():
            qml.BasicEntanglerLayers([[0.1, 0.2]], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        original_dev = qml.device("default.qubit", wires=3)
        decomp_dev = qml.device("default.qubit", wires=3, custom_decomps={})

        original_qnode = qml.QNode(circuit, original_dev, expansion_strategy="device")
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")

        original_res = original_qnode()
        decomp_res = decomp_qnode()

        assert np.isclose(original_res, decomp_res)
        assert [
            orig_op.name == decomp_op.name
            for orig_op, decomp_op in zip(
                original_qnode.qtape.operations, decomp_qnode.qtape.operations
            )
        ]

    @pytest.mark.parametrize("device_name", ["default.qubit", "lightning.qubit"])
    def test_one_custom_decomp(self, device_name):
        """Test that specifying a single custom decomposition works as expected."""

        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        custom_decomps = {"Hadamard": custom_hadamard}
        decomp_dev = qml.device(device_name, wires=2, custom_decomps=custom_decomps)
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 3

        assert decomp_ops[0].name == "RZ"
        assert np.isclose(decomp_ops[0].parameters[0], np.pi)

        assert decomp_ops[1].name == "RY"
        assert np.isclose(decomp_ops[1].parameters[0], np.pi / 2)

        assert decomp_ops[2].name == "CNOT"

    def test_no_decomp_with_depth_zero(self):
        """Test that specifying a single custom decomposition works as expected."""

        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        custom_decomps = {"Hadamard": custom_hadamard, "CNOT": custom_cnot}
        decomp_dev = qml.device(
            "default.qubit", wires=2, custom_decomps=custom_decomps, decomp_depth=0
        )
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 2
        assert decomp_ops[0].name == "Hadamard"
        assert decomp_ops[1].name == "CNOT"

    def test_one_custom_decomp_gradient(self):
        """Test that gradients are still correctly computed after a decomposition
        that performs transpilation."""

        def circuit(x):
            qml.Hadamard(wires=0)
            qml.Rot(x[0], x[1], x[2], wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        original_dev = qml.device("default.qubit", wires=3)
        decomp_dev = qml.device("default.qubit", wires=3, custom_decomps={"Rot": custom_rot})

        original_qnode = qml.QNode(circuit, original_dev, expansion_strategy="device")
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")

        x = qml.numpy.array([0.2, 0.3, 0.4], requires_grad=True)

        original_res = original_qnode(x)
        decomp_res = decomp_qnode(x)
        assert np.allclose(original_res, decomp_res)

        original_grad = qml.grad(original_qnode)(x)
        decomp_grad = qml.grad(decomp_qnode)(x)
        assert np.allclose(original_grad, decomp_grad)

        expected_ops = ["Hadamard", "RZ", "PauliX", "RY", "PauliX", "RZ", "Hadamard"]
        print(decomp_qnode.qtape.operations)
        assert all(
            [op.name == name for op, name in zip(decomp_qnode.qtape.operations, expected_ops)]
        )

    def test_nested_custom_decomp(self):
        """Test that specifying two custom decompositions that have interdependence
        works as expected."""

        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        custom_decomps = {"Hadamard": custom_hadamard, qml.CNOT: custom_cnot}
        decomp_dev = qml.device("default.qubit", wires=2, custom_decomps=custom_decomps)
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 7

        # Check the RZ gates are in the correct place
        for idx in [0, 2, 5]:
            assert decomp_ops[idx].name == "RZ"
            assert np.isclose(decomp_ops[idx].parameters[0], np.pi)

        assert decomp_ops[0].wires == Wires(0)
        assert decomp_ops[2].wires == Wires(1)
        assert decomp_ops[5].wires == Wires(1)

        # Check RY are in the correct place
        for idx in [1, 3, 6]:
            assert decomp_ops[idx].name == "RY"
            assert np.isclose(decomp_ops[idx].parameters[0], np.pi / 2)

        assert decomp_ops[1].wires == Wires(0)
        assert decomp_ops[3].wires == Wires(1)
        assert decomp_ops[6].wires == Wires(1)

        assert decomp_ops[4].name == "CZ"

    def test_nested_custom_decomp_with_template(self):
        """Test that specifying two custom decompositions that have interdependence
        works as expected even when there is a template."""

        def circuit():
            # -RX(0.1)-C- -> -RX(0.1)---C--- -> -RX(0.1)-----------------C----------------
            # -RX(0.2)-X- -> -RX(0.2)-H-Z-H- -> -RX(0.2)-RZ(pi)-RY(pi/2)-Z-RY(pi/2)-RZ(pi)-
            qml.BasicEntanglerLayers([[0.1, 0.2]], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        custom_decomps = {"Hadamard": custom_hadamard, qml.CNOT: custom_cnot}
        decomp_dev = qml.device("default.qubit", wires=2, custom_decomps=custom_decomps)
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 7

        assert decomp_ops[0].name == "RX"
        assert decomp_ops[0].parameters[0] == 0.1
        assert decomp_ops[0].wires == Wires(0)

        assert decomp_ops[1].name == "RX"
        assert decomp_ops[1].parameters[0] == 0.2
        assert decomp_ops[1].wires == Wires(1)

        assert decomp_ops[2].name == "RZ"
        assert np.isclose(decomp_ops[2].parameters[0], np.pi)
        assert decomp_ops[2].wires == Wires(1)

        assert decomp_ops[3].name == "RY"
        assert np.isclose(decomp_ops[3].parameters[0], np.pi / 2)
        assert decomp_ops[3].wires == Wires(1)

        assert decomp_ops[4].name == "CZ"
        assert decomp_ops[4].wires == Wires([0, 1])

        assert decomp_ops[5].name == "RZ"
        assert np.isclose(decomp_ops[5].parameters[0], np.pi)
        assert decomp_ops[5].wires == Wires(1)

        assert decomp_ops[6].name == "RY"
        assert np.isclose(decomp_ops[6].parameters[0], np.pi / 2)
        assert decomp_ops[6].wires == Wires(1)

    def test_custom_decomp_template_to_template(self):
        """Test that decomposing a template into another template and some
        gates yields the correct results."""

        def circuit():
            qml.BasicEntanglerLayers([[0.1, 0.2]], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # BasicEntanglerLayers custom decomposition involves AngleEmbedding
        custom_decomps = {"BasicEntanglerLayers": custom_basic_entangler_layers, "RX": custom_rx}
        decomp_dev = qml.device("default.qubit", wires=2, custom_decomps=custom_decomps)
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 5

        assert decomp_ops[0].name == "RY"
        assert decomp_ops[0].parameters[0] == 0.1
        assert decomp_ops[0].wires == Wires(0)

        assert decomp_ops[1].name == "Hadamard"
        assert decomp_ops[1].wires == Wires(0)

        assert decomp_ops[2].name == "RY"
        assert np.isclose(decomp_ops[2].parameters[0], 0.2)
        assert decomp_ops[2].wires == Wires(1)

        assert decomp_ops[3].name == "Hadamard"
        assert decomp_ops[3].wires == Wires(1)

        assert decomp_ops[4].name == "CNOT"
        assert decomp_ops[4].wires == Wires([0, 1])

    def test_custom_decomp_different_depth(self):
        """Test that alternative expansion depths can be specified."""

        def circuit():
            qml.BasicEntanglerLayers([[0.1, 0.2]], wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        # BasicEntanglerLayers custom decomposition involves AngleEmbedding. If
        # expansion depth is 1, the AngleEmbedding will still be decomposed into
        # RX (since it's not a supported operation on the device), but the RX will
        # not be further decomposed even though the custom decomposition is specified.
        custom_decomps = {"BasicEntanglerLayers": custom_basic_entangler_layers, "RX": custom_rx}
        decomp_dev = qml.device(
            "default.qubit", wires=2, custom_decomps=custom_decomps, decomp_depth=1
        )
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 3

        assert decomp_ops[0].name == "RX"
        assert np.isclose(decomp_ops[0].parameters[0], 0.1)
        assert decomp_ops[0].wires == Wires(0)

        assert decomp_ops[1].name == "RX"
        assert np.isclose(decomp_ops[1].parameters[0], 0.2)
        assert decomp_ops[1].wires == Wires(1)

        assert decomp_ops[2].name == "CNOT"
        assert decomp_ops[2].wires == Wires([0, 1])

    def test_custom_decomp_with_adjoint(self):
        """Test that applying an adjoint in the circuit results in the adjoint
        undergoing the custom decomposition."""

        def circuit():
            # Adjoint is RX(-0.2), so expect RY(-0.2) H
            qml.adjoint(qml.RX)(0.2, wires="a")
            return qml.expval(qml.PauliZ("a"))

        custom_decomps = {qml.RX: custom_rx}
        decomp_dev = qml.device("default.qubit", wires="a", custom_decomps=custom_decomps)
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 2

        assert decomp_ops[0].name == "RY"
        assert decomp_ops[0].parameters[0] == -0.2
        assert decomp_ops[0].wires == Wires("a")

        assert decomp_ops[1].name == "Hadamard"
        assert decomp_ops[1].wires == Wires("a")

    def test_custom_decomp_with_control(self):
        """Test that applying a controlled version of a gate results in the
        controlled version of a decomposition."""

        def circuit():
            qml.ctrl(qml.Hadamard, control=0)(wires=1)
            return qml.expval(qml.PauliZ(0))

        custom_decomps = {qml.Hadamard: custom_hadamard}
        decomp_dev = qml.device("default.qubit", wires=2, custom_decomps=custom_decomps)
        decomp_qnode = qml.QNode(circuit, decomp_dev, expansion_strategy="device")
        _ = decomp_qnode()
        decomp_ops = decomp_qnode.qtape.operations

        assert len(decomp_ops) == 2

        assert decomp_ops[0].name == "CRZ"
        assert np.isclose(decomp_ops[0].parameters[0], np.pi)
        assert decomp_ops[0].wires == Wires([0, 1])

        assert decomp_ops[1].name == "CRY"
        assert np.isclose(decomp_ops[1].parameters[0], np.pi / 2)
        assert decomp_ops[1].wires == Wires([0, 1])

    def test_custom_decomp_in_separate_context(self):
        """Test that the set_decomposition context manager works."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, expansion_strategy="device")
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0))

        # Initial test
        _ = circuit()

        assert len(circuit.qtape.operations) == 1
        assert circuit.qtape.operations[0].name == "CNOT"
        assert dev.custom_expand_fn is None

        # Test within the context manager
        with qml.transforms.set_decomposition({qml.CNOT: custom_cnot}, dev):
            _ = circuit()
            ops_in_context = circuit.qtape.operations

            assert dev.custom_expand_fn is not None

        assert len(ops_in_context) == 3
        assert ops_in_context[0].name == "Hadamard"
        assert ops_in_context[1].name == "CZ"
        assert ops_in_context[2].name == "Hadamard"

        # Check that afterwards, the device has gone back to normal
        _ = circuit()

        assert len(circuit.qtape.operations) == 1
        assert circuit.qtape.operations[0].name == "CNOT"
        assert dev.custom_expand_fn is None
