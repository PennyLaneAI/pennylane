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
            generator = [None, 1]

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
            generator = [None, 1]

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
        assert tape.trainable_params == {1}

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
