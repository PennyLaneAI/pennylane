# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for preprocess in devices/qubit."""

import pytest

import pennylane as qml
from pennylane.operation import Operation
from pennylane.tape import QuantumScript
from pennylane import DeviceError

# pylint: disable=too-few-public-methods

from pennylane.devices.preprocess import (
    no_sampling,
    validate_device_wires,
    validate_multiprocessing_workers,
    warn_about_trainable_observables,
    _operator_decomposition_gen,
    decompose,
    validate_observables,
    validate_measurements,
)


class NoMatOp(Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


class NoMatNoDecompOp(Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


class TestPrivateHelpers:
    """Test the private helpers for preprocessing."""

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.RX(1.2, wires=0), qml.QFT(wires=range(3))))
    def test_operator_decomposition_gen_accepted_operator(self, op):
        """Test the _operator_decomposition_gen function on an operator that is accepted."""

        def stopping_condition(op):
            return op.has_matrix

        casted_to_list = list(_operator_decomposition_gen(op, stopping_condition))
        assert len(casted_to_list) == 1
        assert casted_to_list[0] is op

    def test_operator_decomposition_gen_decomposed_operators_single_nesting(self):
        """Assert _operator_decomposition_gen turns into a list with the operators decomposition
        when only a single layer of expansion is necessary."""

        def stopping_condition(op):
            return op.has_matrix

        op = NoMatOp("a")
        casted_to_list = list(_operator_decomposition_gen(op, stopping_condition))
        assert len(casted_to_list) == 2
        assert qml.equal(casted_to_list[0], qml.PauliX("a"))
        assert qml.equal(casted_to_list[1], qml.PauliY("a"))

    def test_operator_decomposition_gen_decomposed_operator_ragged_nesting(self):
        """Test that _operator_decomposition_gen handles a decomposition that requires different depths of decomposition."""

        def stopping_condition(op):
            return op.has_matrix

        class RaggedDecompositionOp(Operation):
            """class with a ragged decomposition."""

            num_wires = 1

            def decomposition(self):
                return [NoMatOp(self.wires), qml.S(self.wires), qml.adjoint(NoMatOp(self.wires))]

        op = RaggedDecompositionOp("a")
        final_decomp = list(_operator_decomposition_gen(op, stopping_condition))
        assert len(final_decomp) == 5
        assert qml.equal(final_decomp[0], qml.PauliX("a"))
        assert qml.equal(final_decomp[1], qml.PauliY("a"))
        assert qml.equal(final_decomp[2], qml.S("a"))
        assert qml.equal(final_decomp[3], qml.adjoint(qml.PauliY("a")))
        assert qml.equal(final_decomp[4], qml.adjoint(qml.PauliX("a")))

    def test_error_from_unsupported_operation(self):
        """Test that a device error is raised if the operator cant be decomposed and doesn't have a matrix."""
        op = NoMatNoDecompOp("a")
        with pytest.raises(
            DeviceError,
            match=r"not supported on abc and does",
        ):
            tuple(_operator_decomposition_gen(op, lambda op: op.has_matrix, name="abc"))


def test_no_sampling():
    """Tests for the no_sampling transform."""

    tape1 = qml.tape.QuantumScript(shots=None)
    batch, _ = no_sampling(tape1)
    assert batch[0] is tape1

    tape2 = qml.tape.QuantumScript(shots=2)
    with pytest.raises(qml.DeviceError, match="Finite shots are not supported with abc"):
        no_sampling(tape2, name="abc")


def test_warn_about_trainable_observables():
    """Tests warning raised for warn_about_trainable_observables."""
    tape = qml.tape.QuantumScript([], [qml.expval(2 * qml.PauliX(0))])
    with pytest.warns(UserWarning, match="Differentiating with respect to the input "):
        warn_about_trainable_observables(tape)


class TestValidateDeviceWires:
    def test_error(self):
        """Tests for the error raised by validate_device_wires transform."""

        tape1 = qml.tape.QuantumScript([qml.S("a")])
        with pytest.raises(qml.wires.WireError, match="on abc as they contain wires"):
            validate_device_wires(tape1, wires=qml.wires.Wires((0,)), name="abc")

    def test_null_if_no_wires_provided(self):
        """Test that nothing happens if no wires are provided to the transform."""

        tape1 = qml.tape.QuantumScript([qml.S("b")], [qml.expval(qml.PauliZ(0))])
        batch, _ = validate_device_wires(tape1)
        assert batch[0] is tape1

    def test_fill_in_wires(self):
        """Tests that if the wires are provided, measurements without wires take them gain them."""
        tape1 = qml.tape.QuantumScript([qml.S("b")], [qml.state(), qml.probs()], shots=52)

        wires = qml.wires.Wires(["a", "b", "c"])
        batch, _ = validate_device_wires(tape1, wires=wires)
        assert batch[0][1].wires == wires
        assert batch[0][2].wires == wires
        assert batch[0].operations == tape1.operations
        assert batch[0].shots == tape1.shots


class TestDecomposeValidation:
    """Unit tests for helper functions in qml.devices.qubit.preprocess"""

    def test_error_if_invalid_op(self):
        """Test that expand_fn throws an error when an operation is does not define a matrix or decomposition."""

        tape = QuantumScript(ops=[NoMatNoDecompOp(0)], measurements=[qml.expval(qml.Hadamard(0))])
        with pytest.raises(DeviceError, match="not supported on abc"):
            decompose(tape, lambda op: op.has_matrix, name="abc")

    def test_decompose(self):
        """Test that expand_fn doesn't throw any errors for a valid circuit"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.RZ(0.123, wires=0)], measurements=[qml.state()]
        )
        decompose(tape, lambda obj: obj.has_matrix)

    def test_infinite_decomposition_loop(self):
        """Test that a device error is raised if decomposition enters an infinite loop."""

        class InfiniteOp(qml.operation.Operation):
            """An op with an infinite decomposition."""

            num_wires = 1

            def decomposition(self):
                return [InfiniteOp(*self.parameters, self.wires)]

        qs = qml.tape.QuantumScript([InfiniteOp(1.23, 0)])
        with pytest.raises(DeviceError, match=r"Reached recursion limit trying to decompose"):
            decompose(qs, lambda obj: obj.has_matrix)


class TestValidateObservables:
    """Tests for the validate observables transform."""

    def test_invalid_observable(self):
        """Test that expand_fn throws an error when an observable is invalid."""
        tape = QuantumScript(
            ops=[qml.PauliX(0)], measurements=[qml.expval(qml.GellMann(wires=0, index=1))]
        )
        with pytest.raises(DeviceError, match=r"Observable GellMann1 not supported on abc"):
            validate_observables(tape, lambda obs: obs.name == "PauliX", name="abc")

    def test_invalid_tensor_observable(self):
        """Test that expand_fn throws an error when a tensor includes invalid obserables"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.PauliY(1)],
            measurements=[qml.expval(qml.PauliX(0) @ qml.GellMann(wires=1, index=2))],
        )
        with pytest.raises(DeviceError, match="Observable expval"):
            validate_observables(tape, lambda obj: obj.name == "PauliX")

    def test_valid_tensor_observable(self):
        """Test that a valid tensor ovservable passes without error."""
        tape = QuantumScript([], [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))])
        assert (
            validate_measurements(tape, lambda obs: obs.name in {"PauliZ", "PauliY"})[0][0] is tape
        )


class TestValidateMeasurements:
    """Tests for the validate measurements transform."""

    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.state()],
            [qml.expval(qml.PauliZ(0))],
            [qml.state(), qml.expval(qml.PauliZ(0)), qml.probs(0)],
            [qml.state(), qml.vn_entropy(0), qml.mutual_info(0, 1)],
        ],
    )
    def test_only_state_measurements(self, measurements):
        """Test that an analytic circuit containing only StateMeasurements works"""
        tape = QuantumScript([], measurements, shots=None)
        validate_measurements(tape, lambda obj: True)

    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.sample(wires=0)],
            [qml.expval(qml.PauliZ(0))],
            [qml.sample(wires=0), qml.expval(qml.PauliZ(0)), qml.probs(0)],
            [qml.classical_shadow(wires=[0])],
            [qml.shadow_expval(qml.PauliZ(0))],
        ],
    )
    def test_only_sample_measurements(self, measurements):
        """Test that a circuit with finite shots containing only SampleMeasurements works"""
        tape = QuantumScript([], measurements, shots=100)
        validate_measurements(tape, sample_measurements=lambda obj: True)

    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.sample(wires=0)],
            [qml.state(), qml.sample(wires=0)],
            [qml.sample(wires=0), qml.expval(qml.PauliZ(0))],
            [qml.classical_shadow(wires=[0])],
            [qml.shadow_expval(qml.PauliZ(0))],
        ],
    )
    def test_analytic_with_samples(self, measurements):
        """Test that an analytic circuit containing SampleMeasurements raises an error"""
        tape = QuantumScript([], measurements, shots=None)

        msg = "not accepted for analytic simulation on device"
        with pytest.raises(DeviceError, match=msg):
            validate_measurements(tape)

    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.state()],
            [qml.sample(wires=0), qml.state()],
            [qml.expval(qml.PauliZ(0)), qml.state(), qml.sample(wires=0)],
        ],
    )
    def test_finite_shots_with_state(self, measurements):
        """Test that a circuit with finite shots containing StateMeasurements raises an error"""
        tape = QuantumScript([], measurements, shots=100)

        msg = "not accepted with finite shots on device"
        with pytest.raises(DeviceError, match=msg):
            validate_measurements(tape, lambda obj: True)


class TestExpandFnTransformations:
    """Tests for the behavior of the `expand_fn` helper."""

    @pytest.mark.parametrize("shots", [None, 100])
    def test_decompose_expand_unsupported_op(self, shots):
        """Test that expand_fn expands the tape when unsupported operators are present"""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements, shots=shots)

        expanded_tapes, _ = decompose(tape, lambda obj: obj.has_matrix)
        expanded_tape = expanded_tapes[0]
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        for op, exp in zip(expanded_tape.circuit, expected + measurements):
            assert qml.equal(op, exp)

        assert tape.shots == expanded_tape.shots

    def test_decompose_no_expansion(self):
        """Test that expand_fn does nothing to a fully supported quantum script."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)
        expanded_tapes, _ = decompose(tape, lambda obj: obj.has_matrix)
        expanded_tape = expanded_tapes[0]

        for op, exp in zip(expanded_tape.circuit, ops + measurements):
            assert qml.equal(op, exp)

    @pytest.mark.parametrize("validation_transform", (validate_measurements, validate_observables))
    def test_valdiate_measurements_non_commuting_measurements(self, validation_transform):
        """Test that validate_measurements and validate_observables works when non commuting measurements exist in the circuit."""

        qs = QuantumScript([NoMatOp("a")], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0))])

        new_qs, _ = validation_transform(qs, lambda obj: True)
        new_qs = new_qs[0]
        assert new_qs.measurements == qs.measurements

    @pytest.mark.parametrize(
        "prep_op",
        (
            qml.BasisState([1], wires=0),
            qml.StatePrep([0, 1], wires=1),
            qml.AmplitudeEmbedding([0, 1], wires=1),
        ),
    )
    def test_decompose_state_prep_skip_first(self, prep_op):
        """Test that the decompose only expands mid-circuit instances of StatePrepBase if requested."""
        ops = [
            prep_op,
            qml.Hadamard(wires=0),
            qml.StatePrep([0, 1], wires=1),
            qml.BasisState([1], wires=0),
            qml.RZ(0.123, wires=1),
            qml.AmplitudeEmbedding([0, 1, 0, 0], wires=[0, 1]),
        ]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tapes, _ = decompose(
            tape, lambda obj: obj.has_matrix, skip_initial_state_prep=True
        )
        expanded_tape = expanded_tapes[0]
        expected = [
            prep_op,
            qml.Hadamard(0),
            qml.RY(3.14159265, wires=1),  # decomposition of StatePrep
            qml.PauliX(wires=0),  # decomposition of BasisState
            qml.RZ(0.123, wires=1),
            qml.RY(1.57079633, wires=[1]),  # decomposition of AmplitudeEmbedding
            qml.CNOT(wires=[0, 1]),
            qml.RY(1.57079633, wires=[1]),
            qml.CNOT(wires=[0, 1]),
        ]

        assert expanded_tape.circuit == expected + measurements

    @pytest.mark.parametrize(
        "prep_op",
        (
            qml.BasisState([1], wires=0),
            qml.StatePrep([0, 1], wires=1),
            qml.AmplitudeEmbedding([0, 1], wires=1),
        ),
    )
    def test_decompose_initial_state_prep_if_requested(self, prep_op):
        """Test that initial state prep operations are decomposed if skip_initial_state_prep is False."""

        tape = qml.tape.QuantumScript([prep_op])
        batch, _ = decompose(tape, lambda obj: obj.has_matrix, skip_initial_state_prep=False)
        new_tape = batch[0]

        assert new_tape[0] != prep_op


def test_validate_multiprocessing_workers_None():
    """Test that validation does not fail when max_workers is None"""
    qs = QuantumScript(
        [qml.Rot(0.1, 0.2, 0.3, wires=0), qml.CNOT([0, 1])],
        [qml.expval(qml.PauliZ(1))],
    )
    device = qml.devices.DefaultQubit()
    validate_multiprocessing_workers(qs, None, device)
