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
import warnings
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.preprocess import (
    _operator_decomposition_gen,
    decompose,
    device_resolve_dynamic_wires,
    measurements_from_counts,
    measurements_from_samples,
    mid_circuit_measurements,
    no_analytic,
    no_sampling,
    null_postprocessing,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_multiprocessing_workers,
    validate_observables,
)
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.measurements import CountsMP, SampleMP
from pennylane.operation import Operation
from pennylane.tape import QuantumScript

# pylint: disable=too-few-public-methods


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


class InfiniteOp(qml.operation.Operation):
    """An op with an infinite decomposition."""

    num_wires = 1

    def decomposition(self):
        return [InfiniteOp(*self.parameters, self.wires)]


class TestPrivateHelpers:
    """Test the private helpers for preprocessing."""

    @staticmethod
    def decomposer(op):
        return op.decomposition()

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.RX(1.2, wires=0), qml.QFT(wires=range(3))))
    def test_operator_decomposition_gen_accepted_operator(self, op):
        """Test the _operator_decomposition_gen function on an operator that is accepted."""

        def stopping_condition(op):
            return op.has_matrix

        casted_to_list = list(
            _operator_decomposition_gen(op, stopping_condition, custom_decomposer=self.decomposer)
        )
        assert len(casted_to_list) == 1
        assert casted_to_list[0] is op

    def test_operator_decomposition_gen_decomposed_operators_single_nesting(self):
        """Assert _operator_decomposition_gen turns into a list with the operators decomposition
        when only a single layer of expansion is necessary."""

        def stopping_condition(op):
            return op.has_matrix

        op = NoMatOp("a")
        casted_to_list = list(
            _operator_decomposition_gen(op, stopping_condition, custom_decomposer=self.decomposer)
        )
        assert len(casted_to_list) == 2
        qml.assert_equal(casted_to_list[0], qml.PauliX("a"))
        qml.assert_equal(casted_to_list[1], qml.PauliY("a"))

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
        final_decomp = list(
            _operator_decomposition_gen(op, stopping_condition, custom_decomposer=self.decomposer)
        )
        assert len(final_decomp) == 5
        qml.assert_equal(final_decomp[0], qml.PauliX("a"))
        qml.assert_equal(final_decomp[1], qml.PauliY("a"))
        qml.assert_equal(final_decomp[2], qml.S("a"))
        qml.assert_equal(final_decomp[3], qml.adjoint(qml.PauliY("a")))
        qml.assert_equal(final_decomp[4], qml.adjoint(qml.PauliX("a")))


def test_no_sampling():
    """Tests for the no_sampling transform."""

    tape1 = qml.tape.QuantumScript(shots=None)
    batch, _ = no_sampling(tape1)
    assert batch[0] is tape1

    tape2 = qml.tape.QuantumScript(shots=2)
    with pytest.raises(DeviceError, match="Finite shots are not supported with abc"):
        no_sampling(tape2, name="abc")


def test_no_analytic():
    """Test for the no_anayltic transform"""

    tape1 = qml.tape.QuantumScript(shots=2)
    batch, _ = no_analytic(tape1)
    assert batch[0] is tape1

    tape2 = qml.tape.QuantumScript(shots=None)
    with pytest.raises(DeviceError, match="Analytic execution is not supported with abc"):
        no_analytic(tape2, name="abc")


def test_validate_adjoint_trainable_params_obs_warning():
    """Tests warning raised for validate_adjoint_trainable_params with trainable observables."""

    params = qml.numpy.array(0.123)
    tape = qml.tape.QuantumScript([], [qml.expval(2 * qml.RX(params, wires=0))])
    with pytest.warns(UserWarning, match="Differentiating with respect to the input "):
        validate_adjoint_trainable_params(tape)

    params_non_trainable = qml.numpy.array(0.123, requires_grad=False)
    tape = qml.tape.QuantumScript([], [qml.expval(2 * qml.RX(params_non_trainable, wires=0))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # assert no warning raised
        validate_adjoint_trainable_params(tape)


def test_validate_adjoint_trainable_params_state_prep_error():
    """Tests error raised for validate_adjoint_trainable_params with trainable state-preps."""
    tape = qml.tape.QuantumScript([qml.StatePrep(qml.numpy.array([1.0, 0.0]), wires=[0])])
    with pytest.raises(QuantumFunctionError, match="Differentiating with respect to"):
        validate_adjoint_trainable_params(tape)


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
        """Tests that if the wires are provided, measurements without wires gain them."""
        tape1 = qml.tape.QuantumScript([qml.S("b")], [qml.state(), qml.probs()], shots=52)

        wires = qml.wires.Wires(["a", "b", "c"])
        batch, _ = validate_device_wires(tape1, wires=wires)
        assert batch[0][1].wires == wires
        assert batch[0][2].wires == wires
        assert batch[0].operations == tape1.operations
        assert batch[0].shots == tape1.shots

    @pytest.mark.jax
    def test_error_abstract_wires_tape(self):
        """Tests that an error is raised if abstract wires are present in the tape."""

        import jax

        def jit_wires_tape(wires):
            tape_with_abstract_wires = QuantumScript([qml.CNOT(wires=qml.wires.Wires(wires))])
            validate_device_wires(tape_with_abstract_wires, name="fictional_device")

        with pytest.raises(
            qml.wires.WireError,
            match="on fictional_device as abstract wires are present in the tape",
        ):
            jax.jit(jit_wires_tape)([0, 1])

    @pytest.mark.jax
    def test_error_abstract_wires_dev(self):
        """Tests that an error is raised if abstract wires are present in the device."""

        import jax

        def jit_wires_dev(wires):
            validate_device_wires(QuantumScript([]), wires=wires, name="fictional_device")

        with pytest.raises(
            qml.wires.WireError,
            match="on fictional_device as abstract wires are present in the device",
        ):
            jax.jit(jit_wires_dev)([0, 1])

    def test_fill_in_wires_on_snapshots(self):
        """Test that validate_device_wires also fills in the wires on snapshots."""

        tape = qml.tape.QuantumScript([qml.Snapshot(), qml.Snapshot(measurement=qml.probs())])

        (output,), _ = validate_device_wires(tape, wires=qml.wires.Wires((0, 1, 2)))
        mp0 = qml.measurements.StateMP(wires=qml.wires.Wires((0, 1, 2)))
        qml.assert_equal(output[0], qml.Snapshot(measurement=mp0))
        qml.assert_equal(output[1], qml.Snapshot(measurement=qml.probs(wires=(0, 1, 2))))


class TestDecomposeValidation:
    """Unit tests for helper functions in qml.devices.qubit.preprocess"""

    def test_error_if_invalid_op(self):
        """Test that expand_fn throws an error when an operation does not define a matrix or decomposition."""

        tape = QuantumScript(ops=[NoMatNoDecompOp(0)], measurements=[qml.expval(qml.Hadamard(0))])
        with pytest.raises(DeviceError, match="not supported with abc"):
            decompose(tape, lambda op: op.has_matrix, name="abc")

    def test_error_if_invalid_op_decomposer(self):
        """Test that expand_fn throws an error when an operation does not define a matrix or decomposition."""

        tape = QuantumScript(ops=[NoMatNoDecompOp(0)], measurements=[qml.expval(qml.Hadamard(0))])
        with pytest.raises(DeviceError, match="not supported with abc"):
            decompose(
                tape, lambda op: op.has_matrix, decomposer=lambda op: op.decomposition(), name="abc"
            )

    def test_decompose(self):
        """Test that expand_fn doesn't throw any errors for a valid circuit"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.RZ(0.123, wires=0)], measurements=[qml.state()]
        )
        decompose(tape, lambda obj: obj.has_matrix)

    def test_infinite_decomposition_loop(self):
        """Test that a device error is raised if decomposition enters an infinite loop."""

        qs = qml.tape.QuantumScript([InfiniteOp(1.23, 0)])
        with pytest.raises(DeviceError, match=r"Reached recursion limit trying to decompose"):
            decompose(qs, lambda obj: obj.has_matrix)

    @pytest.mark.parametrize(
        "error_type", [RuntimeError, qml.operation.DecompositionUndefinedError]
    )
    def test_error_type_can_be_set(self, error_type):
        """Test that passing a class of Error the ``decompose`` transform allows raising another type
        of error instead of the default ``DeviceError``."""

        decomp_error_tape = QuantumScript(
            ops=[NoMatNoDecompOp(0)], measurements=[qml.expval(qml.Hadamard(0))]
        )
        recursion_error_tape = qml.tape.QuantumScript([InfiniteOp(1.23, 0)])

        with pytest.raises(error_type, match="not supported with abc"):
            decompose(decomp_error_tape, lambda op: op.has_matrix, name="abc", error=error_type)

        with pytest.raises(error_type, match=r"Reached recursion limit trying to decompose"):
            decompose(recursion_error_tape, lambda obj: obj.has_matrix, error=error_type)


class TestValidateObservables:
    """Tests for the validate observables transform."""

    def test_invalid_observable(self):
        """Test that expand_fn throws an error when an observable is invalid."""
        tape = QuantumScript(
            ops=[qml.PauliX(0)], measurements=[qml.expval(qml.GellMann(wires=0, index=1))]
        )
        with pytest.raises(DeviceError, match=r"not supported on abc"):
            validate_observables(tape, lambda obs: obs.name == "PauliX", name="abc")

    def test_invalid_tensor_observable(self):
        """Test that expand_fn throws an error when a tensor includes invalid obserables"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.PauliY(1)],
            measurements=[qml.expval(qml.PauliX(0) @ qml.GellMann(wires=1, index=2))],
        )
        with pytest.raises(DeviceError, match="not supported on device"):
            validate_observables(tape, lambda obj: obj.name == "PauliX")


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


class TestDecomposeTransformations:
    """Tests for the behavior of the `decompose` helper."""

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize("shots", [None, 100])
    def test_decompose_expand_unsupported_op(self, shots):
        """Test that decompose expands the tape when unsupported operators are present"""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements, shots=shots)

        expanded_tapes, _ = decompose(tape, lambda obj: obj.has_matrix)
        expanded_tape = expanded_tapes[0]
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        for op, exp in zip(expanded_tape.circuit, expected + measurements):
            qml.assert_equal(op, exp)

        assert tape.shots == expanded_tape.shots

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_decompose_no_expansion(self):
        """Test that expand_fn does nothing to a fully supported quantum script."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)
        expanded_tapes, _ = decompose(tape, lambda obj: obj.has_matrix)
        expanded_tape = expanded_tapes[0]

        for op, exp in zip(expanded_tape.circuit, ops + measurements):
            qml.assert_equal(op, exp)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize("validation_transform", (validate_measurements, validate_observables))
    def test_valdiate_measurements_non_commuting_measurements(self, validation_transform):
        """Test that validate_measurements and validate_observables works when non commuting measurements exist in the circuit."""

        qs = QuantumScript([NoMatOp("a")], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0))])

        new_qs, _ = validation_transform(qs, lambda obj: True)
        new_qs = new_qs[0]
        assert new_qs.measurements == qs.measurements

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
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

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
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

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_decompose_with_device_wires_and_target_gates(self):
        """Test that decompose works correctly with device_wires and target_gates parameters."""
        # Mock a simple target gate set
        target_gates = {"Hadamard", "CNOT", "RZ", "RX", "RY", "GlobalPhase"}
        device_wires = qml.wires.Wires([0, 1, 2, 3])

        # Create a tape with an operation that needs decomposition
        tape = qml.tape.QuantumScript([qml.QFT(wires=[0, 1])], [qml.expval(qml.Z(0))])

        # Test with device_wires and target_gates
        batch, _ = decompose(
            tape,
            lambda obj: obj.name in target_gates,
            device_wires=device_wires,
            target_gates=target_gates,
        )
        new_tape = batch[0]

        # QFT should be decomposed into supported gates
        assert len(new_tape.operations) > len(tape.operations)
        assert all(op.name in target_gates for op in new_tape.operations)

        # Test without device_wires and target_gates (backward compatibility)
        batch_legacy, _ = decompose(tape, lambda obj: obj.name in target_gates)
        legacy_tape = batch_legacy[0]

        # Should still work correctly
        assert len(legacy_tape.operations) > len(tape.operations)
        assert all(op.name in target_gates for op in legacy_tape.operations)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        "device_wires_list,tape_wires_list,expected_work_wires",
        [
            # (device_wires, tape_wires, expected_work_wires)
            ([0, 1, 2, 3], [0, 1], 2),  # Normal case
            ([0, 1, 2], [0, 1], 1),  # Some overlap
            ([0, 1], [0, 1], 0),  # No work wires available
            (None, [0, 1], None),  # No device constraint
        ],
    )
    def test_decompose_work_wire_calculation(
        self, device_wires_list, tape_wires_list, expected_work_wires, mocker
    ):
        """Test that work wire calculation is correct and passed to decomposition logic."""

        ops = [qml.IsingXX(1.2, wires=tape_wires_list)]

        tape = qml.tape.QuantumScript(ops, [qml.expval(qml.Z(tape_wires_list[0]))])
        device_wires = qml.wires.Wires(device_wires_list) if device_wires_list else None

        # Directly test the work wire calculation logic
        if device_wires is None:
            computed_work_wires = None  # No constraint on work wires
        else:
            # This is the actual calculation from the decompose function
            computed_work_wires = len(set(device_wires) - set(tape.wires))

        # Verify our calculation matches expected
        assert computed_work_wires == expected_work_wires

        # Use a spy to verify the parameter is passed correctly to the decomposition logic
        spy = mocker.spy(qml.devices.preprocess, "_operator_decomposition_gen")

        # Test decompose with real functionality
        target_gates = {"CNOT", "RX", "RY", "RZ", "Hadamard", "GlobalPhase"}

        batch, _ = decompose(
            tape,
            lambda obj: obj.name in target_gates,
            device_wires=device_wires,
            target_gates=target_gates,
        )
        new_tape = batch[0]

        # Basic sanity check: decomposition should produce valid operations
        assert len(new_tape.operations) >= len(tape.operations)
        assert all(op.name in target_gates for op in new_tape.operations)

        # Verify the spy captured the correct parameter
        if spy.called:
            # Look through all calls to find one with the expected parameter
            found_correct_call = False
            for call in spy.call_args_list:
                call_kwargs = call[1]  # Get keyword arguments
                if "max_work_wires" in call_kwargs:
                    if call_kwargs["max_work_wires"] == expected_work_wires:
                        found_correct_call = True
                        break

            assert (
                found_correct_call
            ), f"Expected max_work_wires={expected_work_wires} not found in calls"


class TestGraphModeExclusiveFeatures:
    """Tests that only work when graph mode is enabled.

    NOTE: All tests in this suite will auto-enable graph mode via fixture.
    """

    @pytest.fixture(autouse=True)
    def enable_graph_mode_only(self):
        """Auto-enable graph mode for all tests in this class."""
        qml.decomposition.enable_graph()
        yield
        qml.decomposition.disable_graph()

    def test_work_wire_unavailability_causes_fallback(self):
        """Test that decompositions requiring more work wires than available are discarded.

        This addresses the reviewer's question: if a device has 1 wire but a decomposition
        requires 5 burnable work wires, that decomposition should be discarded.
        """

        class MyOp(qml.operation.Operator):
            num_wires = 1

        # Fallback decomposition (no work wires needed)
        @qml.register_resources({qml.H: 2})
        def decomp_fallback(wires):
            qml.H(wires)
            qml.H(wires)

        # Work wire decomposition (needs 5 burnable wires)
        @qml.register_resources({qml.X: 1}, work_wires={"burnable": 5})
        def decomp_with_work_wire(wires):
            qml.X(wires)

        qml.add_decomps(MyOp, decomp_fallback, decomp_with_work_wire)

        tape = qml.tape.QuantumScript([MyOp(0)])
        device_wires = qml.wires.Wires(1)  # Only 1 wire, insufficient for 5 burnable
        target_gates = {"Hadamard", "PauliX"}

        (out_tape,), _ = decompose(
            tape,
            lambda obj: obj.name in target_gates,
            device_wires=device_wires,
            target_gates=target_gates,
        )

        # Should use fallback decomposition (2 Hadamards)
        assert len(out_tape.operations) == 2
        assert all(op.name == "Hadamard" for op in out_tape.operations)


class TestMidCircuitMeasurements:
    """Unit tests for the mid_circuit_measurements preprocessing transform"""

    @pytest.mark.parametrize(
        "mcm_method, shots, expected_transform",
        [
            ("deferred", 10, qml.defer_measurements),
            ("deferred", None, qml.defer_measurements),
            (None, None, qml.defer_measurements),
            (None, 10, qml.dynamic_one_shot),
            ("one-shot", 10, qml.dynamic_one_shot),
        ],
    )
    def test_mcm_method(self, mcm_method, shots, expected_transform, mocker):
        """Test that the preprocessing transform adheres to the specified transform"""
        dev = qml.device("default.qubit")
        mcm_config = {"postselect_mode": None, "mcm_method": mcm_method}
        tape = QuantumScript([qml.measurements.MidMeasureMP(0)], [], shots=shots)
        spy = mocker.spy(expected_transform, "_transform")

        _, _ = mid_circuit_measurements(tape, dev, mcm_config)
        spy.assert_called_once()

    @pytest.mark.parametrize("mcm_method", ["device", "tree-traversal"])
    @pytest.mark.parametrize("shots", [10, None])
    def test_device_mcm_method(self, mcm_method, shots):
        """Test that no transform is applied by mid_circuit_measurements when the
        mcm method is handled by the device"""
        dev = qml.device("default.qubit")
        mcm_config = {"postselect_mode": None, "mcm_method": mcm_method}
        tape = QuantumScript([qml.measurements.MidMeasureMP(0)], [], shots=shots)

        (new_tape,), post_processing_fn = mid_circuit_measurements(tape, dev, mcm_config)

        assert qml.equal(tape, new_tape)
        assert post_processing_fn == null_postprocessing

    def test_error_incompatible_mcm_method(self):
        """Test that an error is raised if requesting the one-shot transform without shots"""
        dev = qml.device("default.qubit")
        shots = None
        mcm_config = {"postselect_mode": None, "mcm_method": "one-shot"}
        tape = QuantumScript([qml.measurements.MidMeasureMP(0)], [], shots=shots)

        with pytest.raises(
            QuantumFunctionError,
            match="dynamic_one_shot is only supported with finite shots.",
        ):
            _, _ = mid_circuit_measurements(tape, dev, mcm_config)


class TestMeasurementsFromCountsOrSamples:
    """Tests for transforms modifying measurements to be derived from either counts or samples"""

    @pytest.mark.parametrize(
        "meas_transform", (measurements_from_counts, measurements_from_samples)
    )
    def test_error_without_valid_wires(self, meas_transform):
        """Test that a clear error is raised if the transform fails because a measurement
        without wires is passed to it"""

        tape = qml.tape.QuantumScript([], measurements=[qml.probs()], shots=10)

        with pytest.raises(
            RuntimeError,
            match="Please apply validate_device_wires transform before measurements_from_",
        ):
            meas_transform(tape)

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "meas_transform", (measurements_from_counts, measurements_from_samples)
    )
    @pytest.mark.parametrize(
        "input_measurement, expected_res",
        [
            (
                qml.expval(qml.PauliY(wires=0) @ qml.PauliY(wires=1)),
                lambda theta: np.sin(theta) * np.sin(theta / 2),
            ),
            (qml.var(qml.Y(wires=1)), lambda theta: 1 - np.sin(theta / 2) ** 2),
            (
                qml.probs(),
                lambda theta: np.outer(
                    np.outer(
                        [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2],
                        [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
                    ),
                    [1, 0, 0, 0],
                ).flatten(),
            ),
            (
                qml.probs(wires=[1]),
                lambda theta: [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
            ),
        ],
    )
    @pytest.mark.parametrize("shots", [3000, (3000, 3000), (3000, 3500, 4000)])
    def test_measurements_from_samples_or_counts(
        self,
        meas_transform,
        input_measurement,
        expected_res,
        shots,
    ):
        """Test the test_measurements_from_samples and measurements_from_counts transforms with a
        single measurement, and compare outcome to the analytic result.
        """

        theta = 2.5
        dev = qml.device("default.qubit", wires=4)

        tape = qml.tape.QuantumScript(
            [qml.RX(theta, 0), qml.RX(theta / 2, 1)],
            measurements=[input_measurement],
            shots=shots,
        )
        (validated_tape,), _ = validate_device_wires(tape, dev.wires)
        tapes, fn = meas_transform(validated_tape)

        assert len(tapes) == 1
        assert len(tapes[0].measurements) == 1
        expected_type = SampleMP if meas_transform == measurements_from_samples else CountsMP
        assert isinstance(tapes[0].measurements[0], expected_type)

        output = qml.execute(tapes, device=dev)
        res = fn(output)

        if dev.shots.has_partitioned_shots:
            assert len(res) == dev.shots.num_copies

        assert np.allclose(res, expected_res(theta), atol=0.05)

    @pytest.mark.parametrize(
        "counts_kwargs",
        [
            {},
            {"wires": [2, 3]},
            {"op": qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)},
        ],
    )
    @pytest.mark.parametrize(
        "meas_transform", [measurements_from_counts, measurements_from_samples]
    )
    def test_with_counts_output(self, counts_kwargs, meas_transform):
        """Test that returning counts works as expected for all-wires, specific wires, or an observable,
        when using both the measurements_from_counts and measurements_from_samples transforms."""

        dev = qml.device("default.qubit", wires=4)

        @partial(validate_device_wires, wires=dev.wires)
        @qml.set_shots(5000)
        @qml.qnode(dev)
        def basic_circuit(theta: float):
            qml.RY(theta, 0)
            qml.RY(theta / 2, 1)
            qml.RY(2 * theta, 2)
            qml.RY(theta, 3)
            return qml.counts(**counts_kwargs)

        transformed_circuit = meas_transform(basic_circuit)

        theta = 1.9

        res = transformed_circuit(theta)
        expected = basic_circuit(theta)

        # +/- 200 shots is pretty reasonable with 5000 shots total
        assert len(res) == len(expected)
        assert res.keys() == expected.keys()
        for key in res.keys():
            assert np.isclose(res[key], expected[key], atol=200)

    @pytest.mark.parametrize(
        "sample_kwargs",
        [
            {},
            {"wires": [2, 3]},
            {"op": qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)},
        ],
    )
    @pytest.mark.parametrize(
        "meas_transform", [measurements_from_counts, measurements_from_samples]
    )
    def test_with_sample_output(self, sample_kwargs, meas_transform, seed):
        """Test that returning sample works as expected for all-wires, specific wires, or an observable,
        when using both the measurements_from_counts and measurements_from_samples transforms."""

        dev = qml.device("default.qubit", wires=4, seed=seed)

        @partial(validate_device_wires, wires=dev.wires)
        @qml.set_shots(5000)
        @qml.qnode(dev)
        def basic_circuit(theta: float):
            qml.RY(theta, 0)
            qml.RY(theta / 2, 1)
            qml.RY(2 * theta, 2)
            qml.RY(theta, 3)
            return qml.sample(**sample_kwargs)

        transformed_circuit = meas_transform(basic_circuit)

        theta = 1.9

        res = transformed_circuit(theta)
        expected = basic_circuit(theta)

        assert np.isclose(np.mean(res), np.mean(expected), atol=0.05)
        assert res.shape == expected.shape
        assert set(res.flatten()) == set(expected.flatten())

    @pytest.mark.parametrize(
        "counts_kwargs",
        [
            {},
            {"wires": [2, 3]},
            {"op": qml.Z(wires=0) @ qml.Z(wires=1) @ qml.Z(wires=2)},
        ],
    )
    @pytest.mark.parametrize("all_outcomes", [True, False])
    @pytest.mark.parametrize(
        "meas_transform", [measurements_from_counts, measurements_from_samples]
    )
    def test_counts_all_outcomes(self, counts_kwargs, all_outcomes, meas_transform):
        """Test that the measurements with counts when only some states have non-zero counts,
        and confirm that all_counts returns the expected entries"""

        dev = qml.device("default.qubit", wires=4)

        @partial(validate_device_wires, wires=dev.wires)
        @qml.set_shots(5000)
        @qml.qnode(dev)
        def basic_circuit():
            return qml.counts(**counts_kwargs, all_outcomes=all_outcomes)

        transformed_circuit = meas_transform(basic_circuit)

        res = transformed_circuit()
        expected = basic_circuit()

        # +/- 200 shots is pretty reasonable with 5000 shots total
        assert res.keys() == expected.keys()
        for key in res.keys():
            assert np.isclose(res[key], expected[key], atol=200)

    @pytest.mark.parametrize(
        "meas_transform", (measurements_from_counts, measurements_from_samples)
    )
    def test_multiple_measurements(self, meas_transform, seed):
        """Test the results of applying measurements_from_counts/measurements_from_samples with
        multiple measurements"""

        dev = qml.device("default.qubit", wires=4, seed=seed)

        @qml.set_shots(5000)
        @qml.qnode(dev)
        def basic_circuit(theta: float):
            qml.RY(theta, 0)
            qml.RY(theta / 2, 1)
            qml.RY(2 * theta, 2)
            qml.RY(theta, 3)
            return (
                qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                qml.var(qml.PauliX(wires=2)),
                qml.counts(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
                qml.sample(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
                qml.probs(wires=[3]),
            )

        theta = 1.9
        tape = qml.workflow.construct_tape(basic_circuit)(theta)
        (validated_tape,), _ = validate_device_wires(tape, dev.wires)
        tapes, fn = meas_transform(validated_tape)

        sample_output = qml.execute(tapes, device=dev)
        expval_res, var_res, counts_res, sample_res, probs_res = fn(sample_output)

        expval_expected = np.sin(theta) * np.sin(theta / 2)
        var_expected = 1 - np.sin(2 * theta) ** 2
        counts_expected = basic_circuit(theta)[2]
        sample_expected = basic_circuit(theta)[3]
        probs_expected = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2]

        assert np.isclose(expval_res, expval_expected, atol=0.05)
        assert np.isclose(var_res, var_expected, atol=0.05)
        assert np.allclose(probs_res, probs_expected, atol=0.05)

        # +/- 200 shots is pretty reasonable with 5000 shots total
        assert len(counts_res) == len(counts_expected)
        assert counts_res.keys() == counts_expected.keys()
        for key in counts_res.keys():
            assert np.isclose(counts_res[key], counts_expected[key], atol=200)

        # # sample comparison
        assert np.isclose(np.mean(sample_res), np.mean(sample_expected), atol=0.05)
        assert len(sample_res) == len(sample_expected)
        assert set(np.array(sample_res)) == set(sample_expected)

    @pytest.mark.parametrize("shots", [None, 10])
    @pytest.mark.parametrize(
        "meas_transform", (measurements_from_counts, measurements_from_samples)
    )
    def test_only_applied_if_no_shots(self, meas_transform, shots):
        """Test that the transform is only applied if shots are being used"""

        tape = qml.tape.QuantumScript(
            [], measurements=[qml.expval(qml.Z(0)), qml.var(qml.X(1))], shots=shots
        )

        (new_tape,), fn = meas_transform(tape)

        if shots is None:
            assert qml.equal(new_tape, tape)
            assert fn == null_postprocessing
        else:
            assert len(new_tape.measurements) == 1
            assert isinstance(new_tape.measurements[0], (SampleMP, CountsMP))
            assert fn != null_postprocessing


def test_validate_multiprocessing_workers_None():
    """Test that validation does not fail when max_workers is None"""
    qs = QuantumScript(
        [qml.Rot(0.1, 0.2, 0.3, wires=0), qml.CNOT([0, 1])],
        [qml.expval(qml.PauliZ(1))],
    )
    device = qml.devices.DefaultQubit()
    validate_multiprocessing_workers(qs, None, device)


class TestDeviceResolveDynamicWires:

    def test_many_allocations_no_wires(self):
        """Test that min integer will keep incrementing to higher numbers."""

        allocations = [qml.allocation.Allocate.from_num_wires(1) for _ in range(10)]
        ops = [qml.X(op.wires) for op in allocations]
        tape = qml.tape.QuantumScript(allocations + ops)

        [new_tape], fn = device_resolve_dynamic_wires(tape, wires=None)

        assert fn(("a",)) == "a"
        for op, wire in zip(new_tape.operations, range(0, 10)):
            qml.assert_equal(op, qml.X(wire))

    def test_error_on_not_enough_available_wires(self):
        """Test that an error is raised if there are not enough available wires on the device."""

        allocation = qml.allocation.Allocate.from_num_wires(2)
        ops = [qml.X(w) for w in allocation.wires]
        tape = qml.tape.QuantumScript([allocation] + ops)

        with pytest.raises(
            qml.exceptions.AllocationError, match=r"Not enough available wires on device"
        ):
            device_resolve_dynamic_wires(tape, wires=qml.wires.Wires([0]))

    def test_many_allocations_device_wires(self):
        """Test that provided device wires are used properly."""

        allocations = [qml.allocation.Allocate.from_num_wires(1) for _ in range(10)]
        ops = [qml.X(op.wires) for op in allocations]
        tape = qml.tape.QuantumScript(allocations + ops)

        wires = qml.wires.Wires(list(range(10)))
        [new_tape], fn = device_resolve_dynamic_wires(tape, wires=wires)

        assert fn(("a",)) == "a"
        for op, wire in zip(new_tape.operations, wires):
            qml.assert_equal(op, qml.X(wire))

    def test_min_int_non_integer_algorithmic_wires(self):
        """Test that a min int can be calculated when the tape contains non-integer wires."""

        allocation = qml.allocation.Allocate.from_num_wires(1)
        ops = [qml.Z("a"), qml.Z("b"), qml.Z("my_wire"), qml.X(allocation.wires)]
        tape = qml.tape.QuantumScript([allocation] + ops)

        [new_tape], _ = device_resolve_dynamic_wires(tape, wires=None)
        qml.assert_equal(new_tape[-1], qml.X(0))

    def test_min_int_unsorted_wires(self):
        """Test that the min_int is the smallest integer larger than all integers in operation wires."""
        allocation = qml.allocation.Allocate.from_num_wires(1)
        ops = [qml.Z(5), qml.Z(11), qml.Z(4), qml.X(allocation.wires)]
        tape = qml.tape.QuantumScript([allocation] + ops)

        [new_tape], _ = device_resolve_dynamic_wires(tape, wires=None)
        qml.assert_equal(new_tape[-1], qml.X(12))

    def test_error_if_no_zeroed_wires_left_no_reset(self):
        """Test that an error is raised if all zeroed wires have been used and cant reset."""

        alloc1 = qml.allocation.Allocate.from_num_wires(1)
        dealloc1 = qml.allocation.Deallocate(alloc1.wires)
        alloc2 = qml.allocation.Allocate.from_num_wires(1)
        dealloc2 = qml.allocation.Deallocate(alloc2.wires)

        tape = qml.tape.QuantumScript(
            [qml.Z(0), alloc1, qml.X(alloc1.wires), dealloc1, alloc2, qml.Y(alloc2.wires), dealloc2]
        )

        with pytest.raises(
            qml.exceptions.AllocationError, match="Not enough available wires on device"
        ):
            device_resolve_dynamic_wires(tape, wires=(0, 1), allow_resets=False)

    def test_no_resets_min_int(self):
        """Test that if a min_int is specified along with allow_resets=False, then fresh wires keep getting added."""

        alloc1 = qml.allocation.Allocate.from_num_wires(1)
        dealloc1 = qml.allocation.Deallocate(alloc1.wires)
        alloc2 = qml.allocation.Allocate.from_num_wires(1)
        dealloc2 = qml.allocation.Deallocate(alloc2.wires)

        tape = qml.tape.QuantumScript(
            [qml.Z(0), alloc1, qml.X(alloc1.wires), dealloc1, alloc2, qml.Y(alloc2.wires), dealloc2]
        )

        [new_tape], _ = device_resolve_dynamic_wires(tape, wires=None, allow_resets=False)

        expected = qml.tape.QuantumScript([qml.Z(0), qml.X(1), qml.Y(2)])
        qml.assert_equal(expected, new_tape)
