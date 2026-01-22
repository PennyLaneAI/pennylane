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
# pylint: disable=too-few-public-methods, invalid-unary-operand-type, no-member,
# pylint: disable=arguments-differ, arguments-renamed,

import numpy as np
import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml

pytest.mark.usefixtures("disable_graph_decomposition")


def crit_0(op: qml.operation.Operator):
    return not any(qml.math.requires_grad(d) for d in op.data) or (
        op.has_generator and any(qml.math.requires_grad(d) for d in op.data)
    )


class PreprocessDevice(qml.devices.Device):

    def __init__(self, wires=None, shots=None):
        self.target_dev = qml.devices.DefaultQubit(wires=wires, shots=shots)
        super().__init__(wires=wires, shots=shots)

    def preprocess(self, execution_config=None):
        return self.target_dev.preprocess(execution_config)

    def execute(self, circuits, execution_config=None):
        return self.target_dev.execute(circuits, execution_config)


class TestCreateExpandFn:
    """Test creating expansion functions from stopping criteria."""

    doc_0 = "Test docstring."
    with qml.queuing.AnnotatedQueue() as q:
        qml.RX(0.2, wires=0)
        qml.RY(qml.numpy.array(2.1, requires_grad=True), wires=1)
        qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

    tape = qml.tape.QuantumScript.from_queue(q)

    def test_create_expand_fn(self):
        """Test creation of expand_fn."""
        expand_fn = qml.transforms.create_expand_fn(
            depth=10,
            stop_at=crit_0,
            docstring=self.doc_0,
        )
        assert expand_fn.__doc__ == "Test docstring."

    def test_create_expand_fn_expansion(self):
        """Test expansion with created expand_fn."""
        expand_fn = qml.transforms.create_expand_fn(depth=10, stop_at=crit_0)
        new_tape = expand_fn(self.tape)
        assert new_tape.operations[0] == self.tape.operations[0]
        assert new_tape.operations[1] == self.tape.operations[1]
        assert [op.name for op in new_tape.operations[2:]] == ["RZ", "RY", "RZ"]
        assert np.allclose([op.data for op in new_tape.operations[2:]], [[0.5], [0.2], [-0.1]])
        assert [op.wires for op in new_tape.operations[2:]] == [qml.wires.Wires(0)] * 3

    def test_create_expand_fn_dont_expand(self):
        """Test expansion is skipped with depth=0."""
        expand_fn = qml.transforms.create_expand_fn(depth=0, stop_at=crit_0)

        new_tape = expand_fn(self.tape)
        assert new_tape.operations == self.tape.operations

    def test_device_and_stopping_expansion(self):
        """Test that passing a device alongside a stopping condition ensures
        that all operations are expanded to match the devices default gate
        set"""
        dev = DefaultQubitLegacy(wires=1)
        expand_fn = qml.transforms.create_expand_fn(device=dev, depth=10, stop_at=crit_0)

        with qml.queuing.AnnotatedQueue() as q:
            qml.U1(0.2, wires=0)
            qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = expand_fn(tape)

        assert new_tape.operations[0].name == "U1"
        assert [op.name for op in new_tape.operations[1:]] == ["RZ", "RY", "RZ"]

    def test_device_only_expansion(self):
        """Test that passing a device ensures that all operations are expanded
        to match the devices default gate set"""
        dev = DefaultQubitLegacy(wires=1)
        expand_fn = qml.transforms.create_expand_fn(device=dev, depth=10)

        with qml.queuing.AnnotatedQueue() as q:
            qml.U1(0.2, wires=0)
            qml.Rot(*qml.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = expand_fn(tape)

        assert len(new_tape.operations) == 2
        assert new_tape.operations[0].name == "U1"
        assert new_tape.operations[1].name == "Rot"


class TestExpandMultipar:
    """Test the expansion of multi-parameter gates."""

    def test_expand_multipar(self):
        """Test that a multi-parameter gate is decomposed correctly.
        And that single-parameter gates are not decomposed."""

        class _CRX(qml.CRX):
            name = "_CRX"

            @staticmethod
            def decomposition(theta, wires):
                raise NotImplementedError()

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.5, wires=0)
            qml.Rot(-2.1, 0.2, -0.418, wires=1)
            _CRX(1.5, wires=[0, 2])

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = qml.transforms.expand_multipar(tape)
        new_ops = new_tape.operations

        assert [op.name for op in new_ops] == ["RX", "RZ", "RY", "RZ", "_CRX"]

    def test_no_generator_expansion(self):
        """Test that a gate is decomposed correctly if it has
        generator[0]==None."""

        # pylint: disable=invalid-overridden-method
        class _CRX(qml.CRX):
            @property
            def has_generator(self):
                return False

            def generator(self):
                raise qml.operations.GeneratorUndefinedError()

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.5, wires=0)
            qml.RZ(-2.1, wires=1)
            qml.RY(0.2, wires=1)
            qml.RZ(-0.418, wires=1)
            _CRX(1.5, wires=[0, 2])

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = qml.transforms.expand_multipar(tape)
        new_ops = new_tape.operations
        expected = ["RX", "RZ", "RY", "RZ", "RZ", "RY", "CNOT", "RY", "CNOT", "RZ"]
        assert [op.name for op in new_ops] == expected


class TestExpandNonunitaryGen:
    """Test the expansion of operations without a unitary generator."""

    def test_do_not_expand(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            qml.PauliRot(0.9, "XY", wires=[0, 1])
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = qml.transforms.expand_nonunitary_gen(tape)

        assert tape.operations == new_tape.operations

    def test_expand_multi_par(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            qml.Rot(0.9, 1.2, -0.6, wires=0)
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qml.tape.QuantumScript.from_queue(q)
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

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            _PhaseShift(2.1, wires=1)
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = qml.transforms.expand_nonunitary_gen(tape)
        assert tape.operations[:2] == new_tape.operations[:2]
        exp_op, gph_op = new_tape.operations[2:4]
        assert exp_op.name == "RZ" and exp_op.data == (2.1,) and exp_op.wires == qml.wires.Wires(1)
        assert gph_op.name == "GlobalPhase" and gph_op.data == (-2.1 * 0.5,)
        assert tape.operations[3:] == new_tape.operations[4:]

    def test_expand_nonunitary_generator(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.2, wires=0)
            qml.Hadamard(0)
            qml.PhaseShift(2.1, wires=1)
            qml.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = qml.transforms.expand_nonunitary_gen(tape)

        assert tape.operations[:2] == new_tape.operations[:2]
        exp_op, gph_op = new_tape.operations[2:4]
        assert exp_op.name == "RZ" and exp_op.data == (2.1,) and exp_op.wires == qml.wires.Wires(1)
        assert gph_op.name == "GlobalPhase" and gph_op.data == (-2.1 * 0.5,)
        assert tape.operations[3:] == new_tape.operations[4:]

    def test_decompose_all_nonunitary_generator(self):
        """Test that decompositions actually only contain unitarily
        generated operators (Bug #4055)."""

        # Corrected list of unitarily generated operators
        unitarily_generated = [
            "RX",
            "RY",
            "RZ",
            "MultiRZ",
            "PauliRot",
            "IsingXX",
            "IsingYY",
            "IsingZZ",
            "SingleExcitationMinus",
            "SingleExcitationPlus",
            "DoubleExcitationMinus",
            "DoubleExcitationPlus",
            "GlobalPhase",
        ]

        with qml.queuing.AnnotatedQueue() as q:
            # All do not have unitary generator, but previously had this
            # attribute
            qml.IsingXY(0.35, wires=[0, 1])
            qml.SingleExcitation(1.61, wires=[1, 0])
            qml.DoubleExcitation(0.56, wires=[0, 1, 2, 3])
            qml.OrbitalRotation(1.15, wires=[1, 0, 3, 2])
            qml.FermionicSWAP(0.3, wires=[2, 3])

        tape = qml.tape.QuantumScript.from_queue(q)
        new_tape = qml.transforms.expand_nonunitary_gen(tape)

        for op in new_tape.operations:
            assert op.name in unitarily_generated or op.num_params == 0


class TestExpandInvalidTrainable:
    """Tests for the gradient expand function"""

    def test_no_expansion(self, mocker):
        """Test that a circuit with differentiable
        operations is not expanded"""
        x = qml.numpy.array(0.2, requires_grad=True)
        y = qml.numpy.array(0.1, requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
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

        with qml.queuing.AnnotatedQueue() as q:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        spy = mocker.spy(qml.transforms, "decompose")
        new_tape = qml.transforms.expand_invalid_trainable(tape)

        assert new_tape is not tape
        spy.assert_called()

        assert new_tape.operations[0].name == "RZ"
        assert new_tape.operations[0].grad_method == "A"
        assert new_tape.operations[1].name == "GlobalPhase"
        assert new_tape.operations[2].name == "RY"
        assert new_tape.operations[3].name == "CNOT"

    def test_nontrainable_nondiff(self, mocker):
        """Test that a circuit with non-differentiable
        non-trainable operations is not expanded"""
        x = qml.numpy.array(0.2, requires_grad=False)
        y = qml.numpy.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qml.PhaseShift):
            grad_method = None

        with qml.queuing.AnnotatedQueue() as q:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
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

        with qml.queuing.AnnotatedQueue() as q:
            NonDiffPhaseShift(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        spy = mocker.spy(tape, "expand")
        new_tape = qml.transforms.expand_invalid_trainable(tape)

        assert new_tape is tape
        spy.assert_not_called()


# Custom decomposition functions for testing.
def custom_cnot(wires, **_):
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
def custom_basic_entangler_layers(weights, wires, **kwargs):
    # pylint: disable=unused-argument
    def cnot_circuit(wires):
        n_wires = len(wires)

        if n_wires == 2:
            qml.CNOT(wires)
            return

        for wire in wires:
            op_wires = [wire % n_wires, (wire + 1) % n_wires]
            qml.CNOT(op_wires)

    cnot_broadcast = qml.tape.make_qscript(cnot_circuit)(wires)
    return [
        qml.AngleEmbedding(weights[0], wires=wires),
        *cnot_broadcast,
    ]
