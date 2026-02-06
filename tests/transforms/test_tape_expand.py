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
from pennylane.wires import Wires


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
        expand_fn = qp.transforms.create_expand_fn(
            depth=10,
            stop_at=crit_0,
            docstring=self.doc_0,
        )
        assert expand_fn.__doc__ == "Test docstring."

    def test_create_expand_fn_expansion(self):
        """Test expansion with created expand_fn."""
        expand_fn = qp.transforms.create_expand_fn(depth=10, stop_at=crit_0)
        new_tape = expand_fn(self.tape)
        assert new_tape.operations[0] == self.tape.operations[0]
        assert new_tape.operations[1] == self.tape.operations[1]
        assert [op.name for op in new_tape.operations[2:]] == ["RZ", "RY", "RZ"]
        assert np.allclose([op.data for op in new_tape.operations[2:]], [[0.5], [0.2], [-0.1]])
        assert [op.wires for op in new_tape.operations[2:]] == [qp.wires.Wires(0)] * 3

    def test_create_expand_fn_dont_expand(self):
        """Test expansion is skipped with depth=0."""
        expand_fn = qp.transforms.create_expand_fn(depth=0, stop_at=crit_0)

        new_tape = expand_fn(self.tape)
        assert new_tape.operations == self.tape.operations

    def test_device_and_stopping_expansion(self):
        """Test that passing a device alongside a stopping condition ensures
        that all operations are expanded to match the devices default gate
        set"""
        dev = DefaultQubitLegacy(wires=1)
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
        expand_fn = qp.transforms.create_expand_fn(device=dev, depth=10)

        with qp.queuing.AnnotatedQueue() as q:
            qp.U1(0.2, wires=0)
            qp.Rot(*qp.numpy.array([0.5, 0.2, -0.1], requires_grad=True), wires=0)

        tape = qp.tape.QuantumScript.from_queue(q)
        new_tape = expand_fn(tape)

        assert len(new_tape.operations) == 2
        assert new_tape.operations[0].name == "U1"
        assert new_tape.operations[1].name == "Rot"


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
        new_tape = qp.transforms.expand_invalid_trainable(tape)

        assert new_tape is not tape
        spy.assert_called()

        assert new_tape.operations[0].name == "RZ"
        assert new_tape.operations[0].grad_method == "A"
        assert new_tape.operations[1].name == "RY"
        assert new_tape.operations[2].name == "CNOT"

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


class TestCreateCustomDecompExpandFn:
    """Tests for the custom_decomps argument for devices"""

    @pytest.mark.parametrize("device_name", ["default.qutrit"])
    def test_legacy_custom_decomps(self, device_name):
        """Test that the custom_decomps correctly dispatch the legacy device
        Maintain the param list to one of the newest, existing legacy device.
        If there is no existing legacy device, consider removing the
        corresponding logic in device constructor.
        """

        custom_decomps = {"Hadamard": custom_hadamard, qp.CNOT: custom_cnot}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires=2, custom_decomps=custom_decomps)

        # NOTE: don't try to construct; they might cause infinite recursion
        assert isinstance(decomp_dev, qp.devices.LegacyDeviceFacade)
        assert decomp_dev.target_device.short_name == device_name

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_string_and_operator_allowed(self, device_name):
        """Test that the custom_decomps dictionary accepts both strings and operator classes as keys."""

        custom_decomps = {"Hadamard": custom_hadamard, qp.CNOT: custom_cnot}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires=2, custom_decomps=custom_decomps)

        @qp.qnode(decomp_dev)
        def circuit():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        tape = qp.workflow.construct_batch(circuit, level="device")()[0][0]
        decomp_ops = tape.operations

        assert len(decomp_ops) == 7
        for op in decomp_ops:
            assert op.name != "Hadamard"
            assert op.name != "CNOT"

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_no_custom_decomp(self, device_name):
        """Test that sending an empty dictionary results in no decompositions."""

        def circuit():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        original_dev = qp.device(device_name, wires=3)

        decomp_dev = qp.device(device_name, wires=3)

        original_qnode = qp.QNode(circuit, original_dev)
        decomp_qnode = qp.QNode(circuit, decomp_dev)

        original_res = original_qnode()
        decomp_res = decomp_qnode()

        original_ops = qp.workflow.construct_batch(original_qnode, level="device")()[0][
            0
        ].operations
        decomp_ops = qp.workflow.construct_batch(decomp_qnode, level="device")()[0][0].operations

        assert np.isclose(original_res, decomp_res)
        assert [
            orig_op.name == decomp_op.name for orig_op, decomp_op in zip(original_ops, decomp_ops)
        ]

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_no_custom_decomp_template(self, device_name):
        """Test that sending an empty dictionary results in no decomposition
        when a template is involved, except the decomposition expected from the device."""

        def circuit():
            qp.BasicEntanglerLayers([[0.1, 0.2]], wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        original_dev = qp.device(device_name, wires=3)
        decomp_dev = qp.device(device_name, wires=3)

        original_qnode = qp.QNode(circuit, original_dev)
        decomp_qnode = qp.QNode(circuit, decomp_dev)

        original_ops = qp.workflow.construct_batch(original_qnode, level="device")()[0][
            0
        ].operations
        decomp_ops = qp.workflow.construct_batch(decomp_qnode, level="device")()[0][0].operations

        original_res = original_qnode()
        decomp_res = decomp_qnode()

        assert np.isclose(original_res, decomp_res)
        assert [
            orig_op.name == decomp_op.name for orig_op, decomp_op in zip(original_ops, decomp_ops)
        ]

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed", "lightning.qubit"])
    def test_one_custom_decomp(self, device_name):
        """Test that specifying a single custom decomposition works as expected."""

        custom_decomps = {"Hadamard": custom_hadamard}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires=2, custom_decomps=custom_decomps)

        @qp.qnode(decomp_dev)
        def circuit():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        decomp_ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

        assert len(decomp_ops) == 3

        assert decomp_ops[0].name == "RZ"
        assert np.isclose(decomp_ops[0].parameters[0], np.pi)

        assert decomp_ops[1].name == "RY"
        assert np.isclose(decomp_ops[1].parameters[0], np.pi / 2)

        assert decomp_ops[2].name == "CNOT"

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_one_custom_decomp_gradient(self, device_name):
        """Test that gradients are still correctly computed after a decomposition
        that performs transpilation."""

        def circuit(x):
            qp.Hadamard(wires=0)
            qp.Rot(x[0], x[1], x[2], wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        original_dev = qp.device(device_name, wires=3)
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires=3, custom_decomps={"Rot": custom_rot})

        original_qnode = qp.QNode(circuit, original_dev)
        decomp_qnode = qp.QNode(circuit, decomp_dev)

        x = qp.numpy.array([0.2, 0.3, 0.4], requires_grad=True)

        original_res = original_qnode(x)
        decomp_res = decomp_qnode(x)
        assert np.allclose(original_res, decomp_res)

        original_grad = qp.grad(original_qnode)(x)
        decomp_grad = qp.grad(decomp_qnode)(x)
        assert np.allclose(original_grad, decomp_grad)

        expected_ops = ["Hadamard", "RZ", "PauliX", "RY", "PauliX", "RZ", "Hadamard"]
        decomp_ops = qp.workflow.construct_batch(decomp_qnode, level="device")(x)[0][0].operations
        assert all(op.name == name for op, name in zip(decomp_ops, expected_ops))

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_nested_custom_decomp(self, device_name):
        """Test that specifying two custom decompositions that have interdependence
        works as expected."""

        custom_decomps = {"Hadamard": custom_hadamard, qp.CNOT: custom_cnot}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires=2, custom_decomps=custom_decomps)

        @qp.qnode(decomp_dev)
        def circuit():
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        decomp_ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

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

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_nested_custom_decomp_with_template(self, device_name):
        """Test that specifying two custom decompositions that have interdependence
        works as expected even when there is a template."""

        custom_decomps = {"Hadamard": custom_hadamard, qp.CNOT: custom_cnot}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires=2, custom_decomps=custom_decomps)

        @qp.qnode(decomp_dev)
        def circuit():
            # -RX(0.1)-C- -> -RX(0.1)---C--- -> -RX(0.1)-----------------C----------------
            # -RX(0.2)-X- -> -RX(0.2)-H-Z-H- -> -RX(0.2)-RZ(pi)-RY(pi/2)-Z-RY(pi/2)-RZ(pi)-
            qp.BasicEntanglerLayers([[0.1, 0.2]], wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        decomp_ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

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

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_custom_decomp_template_to_template(self, device_name):
        """Test that decomposing a template into another template and some
        gates yields the correct results."""

        # BasicEntanglerLayers custom decomposition involves AngleEmbedding
        custom_decomps = {"BasicEntanglerLayers": custom_basic_entangler_layers, "RX": custom_rx}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires=2, custom_decomps=custom_decomps)

        @qp.qnode(decomp_dev)
        def circuit():
            qp.BasicEntanglerLayers([[0.1, 0.2]], wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        decomp_ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

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

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.mixed"])
    def test_custom_decomp_with_adjoint(self, device_name):
        """Test that applying an adjoint in the circuit results in the adjoint
        undergoing the custom decomposition."""

        custom_decomps = {qp.RX: custom_rx}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device(device_name, wires="a", custom_decomps=custom_decomps)

        @qp.qnode(decomp_dev)
        def circuit():
            # Adjoint is RX(-0.2), so expect RY(-0.2) H
            qp.adjoint(qp.RX, lazy=False)(0.2, wires="a")
            return qp.expval(qp.PauliZ("a"))

        decomp_ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

        assert len(decomp_ops) == 2

        assert decomp_ops[0].name == "RY"
        assert decomp_ops[0].parameters[0] == -0.2
        assert decomp_ops[0].wires == Wires("a")

        assert decomp_ops[1].name == "Hadamard"
        assert decomp_ops[1].wires == Wires("a")

    @pytest.mark.parametrize("device_name", ["default.qubit", "default.qubit.legacy"])
    def test_custom_decomp_with_control(self, device_name):
        """Test that decomposing a controlled version of a gate uses the custom decomposition
        for the base gate."""

        class CustomOp(qp.operation.Operation):
            num_wires = 1

            @staticmethod
            def compute_decomposition(wires):
                return [qp.S(wires)]

        original_decomp = CustomOp(0).decomposition()

        custom_decomps = {CustomOp: lambda wires: [qp.T(wires), qp.T(wires)]}
        if device_name == "default.qubit.legacy":
            decomp_dev = DefaultQubitLegacy(wires=2)
            expand_fn = qp.transforms.create_decomp_expand_fn(custom_decomps, decomp_dev)
            decomp_dev.custom_expand(expand_fn)
        else:
            with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
                decomp_dev = qp.device(device_name, wires=2, custom_decomps=custom_decomps)

        @qp.qnode(decomp_dev)
        def circuit():
            qp.ctrl(CustomOp, control=1)(0)
            return qp.expval(qp.PauliZ(0))

        decomp_ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

        assert len(decomp_ops) == 2

        for op in decomp_ops:
            assert isinstance(op, qp.ops.op_math.Controlled)
            qp.assert_equal(op.base, qp.T(0))

        # check that new instances of the operator are not affected by the modifications made to get the decomposition
        assert [op1 == op2 for op1, op2 in zip(CustomOp(0).decomposition(), original_decomp)]

    def test_custom_decomp_in_separate_context_legacy_opmath(self):
        """Test that the set_decomposition context manager works."""

        dev = qp.devices.LegacyDeviceFacade(DefaultQubitLegacy(wires=2))

        @qp.qnode(dev)
        def circuit():
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(wires=0))

        # Initial test
        ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

        assert len(ops) == 1
        assert ops[0].name == "CNOT"
        assert dev.custom_expand_fn is None

        # Test within the context manager
        with qp.transforms.set_decomposition({qp.CNOT: custom_cnot}, dev):
            ops_in_context = qp.workflow.construct_batch(circuit, level="device")()[0][
                0
            ].operations

            assert dev.custom_expand_fn is not None

        assert len(ops_in_context) == 3
        assert ops_in_context[0].name == "Hadamard"
        assert ops_in_context[1].name == "CZ"
        assert ops_in_context[2].name == "Hadamard"

        # Check that afterwards, the device has gone back to normal
        ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

        assert len(ops) == 1
        assert ops[0].name == "CNOT"
        assert dev.custom_expand_fn is None

    @pytest.mark.parametrize("device", (qp.devices.DefaultQubit, PreprocessDevice))
    def test_custom_decomp_in_separate_context(self, mocker, device):
        """Test that the set_decomposition context manager works for the new device API."""

        dev = device(wires=2)
        spy = mocker.spy(dev, "execute")

        @qp.qnode(dev)
        def circuit():
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(wires=0))

        # Initial test
        _ = circuit()

        tape = spy.call_args_list[0][0][0][0]
        assert len(tape.operations) == 1
        assert tape.operations[0].name == "CNOT"

        ind = 0 if device == qp.devices.DefaultQubit else 1

        assert dev.preprocess_transforms()[ind].tape_transform.__name__ == "decompose"
        assert dev.preprocess_transforms()[ind].kwargs.get("decomposer", None) is None

        # Test within the context manager
        with qp.transforms.set_decomposition({qp.CNOT: custom_cnot}, dev):
            _ = circuit()

            assert dev.preprocess_transforms()[ind].tape_transform.__name__ == "decompose"
            assert dev.preprocess_transforms()[ind].kwargs.get("decomposer", None) is not None

        tape = spy.call_args_list[1][0][0][0]
        ops_in_context = tape.operations
        assert len(ops_in_context) == 3
        assert ops_in_context[0].name == "Hadamard"
        assert ops_in_context[1].name == "CZ"
        assert ops_in_context[2].name == "Hadamard"

        # Check that afterwards, the device has gone back to normal
        _ = circuit()

        tape = spy.call_args_list[2][0][0][0]
        ops_in_context = tape.operations
        assert len(tape.operations) == 1
        assert tape.operations[0].name == "CNOT"
        assert dev.preprocess_transforms()[ind].tape_transform.__name__ == "decompose"
        assert dev.preprocess_transforms()[ind].kwargs.get("decomposer", None) is None

    # pylint: disable=cell-var-from-loop

    def test_custom_decomp_used_twice(self):
        """Test that creating a custom decomposition includes overwriting the
        correct method under the hood and produces expected results."""
        res = []
        for _ in range(2):
            custom_decomps = {"MultiRZ": qp.MultiRZ.compute_decomposition}
            with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
                dev = qp.device("lightning.qubit", wires=2, custom_decomps=custom_decomps)

            @qp.qnode(dev, diff_method="adjoint")
            def cost(theta):
                qp.Hadamard(wires=0)
                qp.Hadamard(wires=1)
                qp.MultiRZ(theta, wires=[1, 0])
                return qp.expval(qp.PauliX(1))

            x = np.array(0.5)
            res.append(cost(x))

        assert res[0] == res[1]

    @pytest.mark.parametrize("shots", [None, 100])
    def test_custom_decomp_with_mcm(self, shots):
        """Test that specifying a single custom decomposition works as expected."""

        custom_decomps = {"Hadamard": custom_hadamard}
        with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning):
            decomp_dev = qp.device("default.qubit", custom_decomps=custom_decomps)

        @qp.set_shots(shots)
        @qp.qnode(decomp_dev)
        def circuit():
            qp.Hadamard(wires=0)
            _ = qp.measure(0)
            qp.CNOT(wires=[0, 1])
            _ = qp.measure(1)
            return qp.expval(qp.PauliZ(0))

        decomp_ops = qp.workflow.construct_batch(circuit, level="device")()[0][0].operations

        assert len(decomp_ops) == 4 if shots is None else 5

        assert decomp_ops[0].name == "RZ"
        assert np.isclose(decomp_ops[0].parameters[0], np.pi)

        assert decomp_ops[1].name == "RY"
        assert np.isclose(decomp_ops[1].parameters[0], np.pi / 2)

        if shots:
            assert decomp_ops[2].name == "MidMeasureMP"
            assert decomp_ops[3].name == "CNOT"
            assert decomp_ops[4].name == "MidMeasureMP"
        else:
            assert decomp_ops[2].name == "CNOT"
            assert decomp_ops[3].name == "CNOT"
