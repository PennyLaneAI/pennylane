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

import pennylane as qp
from pennylane.exceptions import PennyLaneDeprecationWarning

pytestmark = pytest.mark.usefixtures("disable_graph_decomposition")


def crit_0(op: qp.operation.Operator):
    return not any(qp.math.requires_grad(d) for d in op.data) or (
        op.has_generator and any(qp.math.requires_grad(d) for d in op.data)
    )


class PreprocessDevice(qp.devices.Device):

    def __init__(self, wires=None, shots=None):
        self.target_dev = qp.devices.DefaultQubit(wires=wires, shots=shots)
        super().__init__(wires=wires, shots=shots)

    def preprocess(self, execution_config=None):
        return self.target_dev.preprocess(execution_config)

    def execute(self, circuits, execution_config=None):
        return self.target_dev.execute(circuits, execution_config)


class TestCreateExpandFn:
    """Test creating expansion functions from stopping criteria."""

    doc_0 = "Test docstring."
    with qp.queuing.AnnotatedQueue() as q:
        qp.RX(0.2, wires=0)
        qp.RY(qp.numpy.array(2.1, requires_grad=True), wires=1)
        qp.Rot(*qp.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

    tape = qp.tape.QuantumScript.from_queue(q)

    def test_create_expand_fn(self):
        """Test creation of expand_fn."""
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            expand_fn = qp.transforms.create_expand_fn(
                depth=10,
                stop_at=crit_0,
                docstring=self.doc_0,
            )
        assert expand_fn.__doc__ == "Test docstring."

    def test_create_expand_fn_deprecated(self):
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            _ = qp.transforms.create_expand_fn(
                depth=10,
                stop_at=crit_0,
                docstring=self.doc_0,
            )

    def test_create_expand_fn_expansion(self):
        """Test expansion with created expand_fn."""
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            expand_fn = qp.transforms.create_expand_fn(depth=10, stop_at=crit_0)
        new_tape = expand_fn(self.tape)
        assert new_tape.operations[0] == self.tape.operations[0]
        assert new_tape.operations[1] == self.tape.operations[1]
        assert [op.name for op in new_tape.operations[2:]] == ["RZ", "RY", "RZ"]
        assert np.allclose([op.data for op in new_tape.operations[2:]], [[0.5], [0.2], [-0.1]])
        assert [op.wires for op in new_tape.operations[2:]] == [qp.wires.Wires(0)] * 3

    def test_create_expand_fn_dont_expand(self):
        """Test expansion is skipped with depth=0."""
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            expand_fn = qp.transforms.create_expand_fn(depth=0, stop_at=crit_0)

        new_tape = expand_fn(self.tape)
        assert new_tape.operations == self.tape.operations

    def test_device_and_stopping_expansion(self):
        """Test that passing a device alongside a stopping condition ensures
        that all operations are expanded to match the devices default gate
        set"""
        dev = DefaultQubitLegacy(wires=1)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            expand_fn = qp.transforms.create_expand_fn(device=dev, depth=10, stop_at=crit_0)

        with qp.queuing.AnnotatedQueue() as q:
            qp.U1(0.2, wires=0)
            qp.Rot(*qp.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        new_tape = expand_fn(tape)

        assert new_tape.operations[0].name == "U1"
        assert [op.name for op in new_tape.operations[1:]] == ["RZ", "RY", "RZ"]

    def test_device_only_expansion(self):
        """Test that passing a device ensures that all operations are expanded
        to match the devices default gate set"""
        dev = DefaultQubitLegacy(wires=1)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            expand_fn = qp.transforms.create_expand_fn(device=dev, depth=10)

        with qp.queuing.AnnotatedQueue() as q:
            qp.U1(0.2, wires=0)
            qp.Rot(*qp.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        new_tape = expand_fn(tape)

        assert len(new_tape.operations) == 2
        assert new_tape.operations[0].name == "U1"
        assert new_tape.operations[1].name == "Rot"


class TestExpandTrainableMultipar:
    """Test the expansion of trainable multi-parameter gates."""

    def test_expand_trainable_multipar(self):
        """Test that a trainable multi-parameter gate is decomposed correctly.
        And that non-trainable multi-parameter gates, single-parameter gates are not decomposed."""

        theta = qp.numpy.array(1.0, requires_grad=True)

        class _CRX(qp.CRX):
            name = "_CRX"

            @staticmethod
            def decomposition(theta, wires):
                raise NotImplementedError()

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(1.5, wires=0)
            qp.Rot(-2.1, 1.0, -0.418, wires=1)
            qp.Rot(-2.1, theta, -0.418, wires=1)
            _CRX(1.5, wires=[0, 2])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_trainable_multipar(tape)
        new_ops = new_tape.operations

        assert [op.name for op in new_ops] == ["RX", "Rot", "RZ", "RY", "RZ", "_CRX"]


class TestExpandMultipar:
    """Test the expansion of multi-parameter gates."""

    def test_expand_multipar(self):
        """Test that a multi-parameter gate is decomposed correctly.
        And that single-parameter gates are not decomposed."""

        class _CRX(qp.CRX):
            name = "_CRX"

            @staticmethod
            def decomposition(theta, wires):
                raise NotImplementedError()

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(1.5, wires=0)
            qp.Rot(-2.1, 0.2, -0.418, wires=1)
            _CRX(1.5, wires=[0, 2])

        tape = qp.tape.QuantumScript.from_queue(q)

        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_multipar(tape)
        new_ops = new_tape.operations

        assert [op.name for op in new_ops] == ["RX", "RZ", "RY", "RZ", "_CRX"]

    def test_no_generator_expansion(self):
        """Test that a gate is decomposed correctly if it has
        generator[0]==None."""

        # pylint: disable=invalid-overridden-method
        class _CRX(qp.CRX):
            @property
            def has_generator(self):
                return False

            def generator(self):
                raise qp.operations.GeneratorUndefinedError()

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(1.5, wires=0)
            qp.RZ(-2.1, wires=1)
            qp.RY(0.2, wires=1)
            qp.RZ(-0.418, wires=1)
            _CRX(1.5, wires=[0, 2])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_multipar(tape)
        new_ops = new_tape.operations
        expected = ["RX", "RZ", "RY", "RZ", "RZ", "RY", "CNOT", "RY", "CNOT", "RZ"]
        assert [op.name for op in new_ops] == expected


class TestExpandNonunitaryGen:
    """Test the expansion of operations without a unitary generator."""

    def test_do_not_expand(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(0.2, wires=0)
            qp.Hadamard(0)
            qp.PauliRot(0.9, "XY", wires=[0, 1])
            qp.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_nonunitary_gen(tape)

        assert tape.operations == new_tape.operations

    def test_expand_multi_par(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""
        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(0.2, wires=0)
            qp.Hadamard(0)
            qp.Rot(0.9, 1.2, -0.6, wires=0)
            qp.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_nonunitary_gen(tape)
        expanded = [
            qp.RZ(0.9, wires=0),
            qp.RY(1.2, wires=0),
            qp.RZ(-0.6, wires=0),
        ]

        assert tape.operations[:2] == new_tape.operations[:2]
        assert all(exp.name == new.name for exp, new in zip(expanded, new_tape.operations[2:5]))
        assert all(exp.data == new.data for exp, new in zip(expanded, new_tape.operations[2:5]))
        assert all(exp.wires == new.wires for exp, new in zip(expanded, new_tape.operations[2:5]))
        assert tape.operations[3:] == new_tape.operations[5:]

    def test_expand_missing_generator(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""

        class _PhaseShift(qp.PhaseShift):
            def generator(self):
                return None

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(0.2, wires=0)
            qp.Hadamard(0)
            _PhaseShift(2.1, wires=1)
            qp.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_nonunitary_gen(tape)
        assert tape.operations[:2] == new_tape.operations[:2]
        exp_op, gph_op = new_tape.operations[2:4]
        assert exp_op.name == "RZ" and exp_op.data == (2.1,) and exp_op.wires == qp.wires.Wires(1)
        assert gph_op.name == "GlobalPhase" and gph_op.data == (-2.1 * 0.5,)
        assert tape.operations[3:] == new_tape.operations[4:]

    def test_expand_nonunitary_generator(self):
        """Test that a tape with single-parameter operations with
        unitary generators and non-parametric operations is not touched."""

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(0.2, wires=0)
            qp.Hadamard(0)
            qp.PhaseShift(2.1, wires=1)
            qp.SingleExcitationPlus(-1.2, wires=[1, 0])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_nonunitary_gen(tape)

        assert tape.operations[:2] == new_tape.operations[:2]
        exp_op, gph_op = new_tape.operations[2:4]
        assert exp_op.name == "RZ" and exp_op.data == (2.1,) and exp_op.wires == qp.wires.Wires(1)
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

        with qp.queuing.AnnotatedQueue() as q:
            # All do not have unitary generator, but previously had this
            # attribute
            qp.IsingXY(0.35, wires=[0, 1])
            qp.SingleExcitation(1.61, wires=[1, 0])
            qp.DoubleExcitation(0.56, wires=[0, 1, 2, 3])
            qp.OrbitalRotation(1.15, wires=[1, 0, 3, 2])
            qp.FermionicSWAP(0.3, wires=[2, 3])

        tape = qp.tape.QuantumScript.from_queue(q)
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_nonunitary_gen(tape)

        for op in new_tape.operations:
            assert op.name in unitarily_generated or op.num_params == 0


class TestExpandInvalidTrainable:
    """Tests for the gradient expand function"""

    def test_no_expansion(self, mocker):
        """Test that a circuit with differentiable
        operations is not expanded"""
        x = qp.numpy.array(0.2, requires_grad=True)
        y = qp.numpy.array(0.1, requires_grad=True)

        with qp.queuing.AnnotatedQueue() as q:
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        spy = mocker.spy(tape, "expand")
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_invalid_trainable(tape)

        assert new_tape is tape
        spy.assert_not_called()

    def test_trainable_nondiff_expansion(self, mocker):
        """Test that a circuit with non-differentiable
        trainable operations is expanded"""
        x = qp.numpy.array(0.2, requires_grad=True)
        y = qp.numpy.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qp.PhaseShift):
            grad_method = None

        with qp.queuing.AnnotatedQueue() as q:
            NonDiffPhaseShift(x, wires=0)
            qp.RY(y, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        spy = mocker.spy(qp.transforms, "decompose")
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_invalid_trainable(tape)

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
        x = qp.numpy.array(0.2, requires_grad=False)
        y = qp.numpy.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qp.PhaseShift):
            grad_method = None

        with qp.queuing.AnnotatedQueue() as q:
            NonDiffPhaseShift(x, wires=0)
            qp.RY(y, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qp.math.get_trainable_indices(params)
        assert tape.trainable_params == [1]

        spy = mocker.spy(tape, "expand")
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_invalid_trainable(tape)

        assert new_tape is tape
        spy.assert_not_called()

    def test_trainable_numeric(self, mocker):
        """Test that a circuit with numeric differentiable
        trainable operations is *not* expanded"""
        x = qp.numpy.array(0.2, requires_grad=True)
        y = qp.numpy.array(0.1, requires_grad=True)

        class NonDiffPhaseShift(qp.PhaseShift):
            grad_method = "F"

        with qp.queuing.AnnotatedQueue() as q:
            NonDiffPhaseShift(x, wires=0)
            qp.RY(y, wires=1)
            qp.CNOT(wires=[0, 1])
            qp.expval(qp.PauliZ(0))

        tape = qp.tape.QuantumScript.from_queue(q)
        spy = mocker.spy(tape, "expand")
        with pytest.warns(PennyLaneDeprecationWarning, match="expand"):
            new_tape = qp.transforms.expand_invalid_trainable(tape)

        assert new_tape is tape
        spy.assert_not_called()


# Custom decomposition functions for testing.
def custom_cnot(wires, **_):
    return [
        qp.Hadamard(wires=wires[1]),
        qp.CZ(wires=[wires[0], wires[1]]),
        qp.Hadamard(wires=wires[1]),
    ]


def custom_hadamard(wires):
    return [qp.RZ(np.pi, wires=wires), qp.RY(np.pi / 2, wires=wires)]


# Incorrect, for testing purposes only
def custom_rx(params, wires):
    return [qp.RY(params, wires=wires), qp.Hadamard(wires=wires)]


# To test the gradient; use circuit identity RY(theta) = X RY(-theta) X
def custom_rot(phi, theta, omega, wires):
    return [
        qp.RZ(phi, wires=wires),
        qp.PauliX(wires=wires),
        qp.RY(-theta, wires=wires),
        qp.PauliX(wires=wires),
        qp.RZ(omega, wires=wires),
    ]


# Decompose a template into another template
def custom_basic_entangler_layers(weights, wires, **kwargs):
    # pylint: disable=unused-argument
    def cnot_circuit(wires):
        n_wires = len(wires)

        if n_wires == 2:
            qp.CNOT(wires)
            return

        for wire in wires:
            op_wires = [wire % n_wires, (wire + 1) % n_wires]
            qp.CNOT(op_wires)

    cnot_broadcast = qp.tape.make_qscript(cnot_circuit)(wires)
    return [
        qp.AngleEmbedding(weights[0], wires=wires),
        *cnot_broadcast,
    ]
