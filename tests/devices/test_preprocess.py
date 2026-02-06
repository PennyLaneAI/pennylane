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

import numpy as np
import pytest

import pennylane as qp
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
from pennylane.exceptions import DeviceError, PennyLaneDeprecationWarning, QuantumFunctionError
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
        return [qp.PauliX(self.wires), qp.PauliY(self.wires)]


class NoMatNoDecompOp(Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


class InfiniteOp(qp.operation.Operation):
    """An op with an infinite decomposition."""

    num_wires = 1

    def decomposition(self):
        return [InfiniteOp(*self.parameters, self.wires)]


class TestPrivateHelpers:
    """Test the private helpers for preprocessing."""

    @staticmethod
    def decomposer(op):
        return op.decomposition()

    @pytest.mark.parametrize("op", (qp.PauliX(0), qp.RX(1.2, wires=0), qp.QFT(wires=range(3))))
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
        qp.assert_equal(casted_to_list[0], qp.PauliX("a"))
        qp.assert_equal(casted_to_list[1], qp.PauliY("a"))

    def test_operator_decomposition_gen_decomposed_operator_ragged_nesting(self):
        """Test that _operator_decomposition_gen handles a decomposition that requires different depths of decomposition."""

        def stopping_condition(op):
            return op.has_matrix

        class RaggedDecompositionOp(Operation):
            """class with a ragged decomposition."""

            num_wires = 1

            def decomposition(self):
                return [NoMatOp(self.wires), qp.S(self.wires), qp.adjoint(NoMatOp(self.wires))]

        op = RaggedDecompositionOp("a")
        final_decomp = list(
            _operator_decomposition_gen(op, stopping_condition, custom_decomposer=self.decomposer)
        )
        assert len(final_decomp) == 5
        qp.assert_equal(final_decomp[0], qp.PauliX("a"))
        qp.assert_equal(final_decomp[1], qp.PauliY("a"))
        qp.assert_equal(final_decomp[2], qp.S("a"))
        qp.assert_equal(final_decomp[3], qp.adjoint(qp.PauliY("a")))
        qp.assert_equal(final_decomp[4], qp.adjoint(qp.PauliX("a")))


def test_no_sampling():
    """Tests for the no_sampling transform."""

    tape1 = qp.tape.QuantumScript(shots=None)
    batch, _ = no_sampling(tape1)
    assert batch[0] is tape1

    tape2 = qp.tape.QuantumScript(shots=2)
    with pytest.raises(DeviceError, match="Finite shots are not supported with abc"):
        no_sampling(tape2, name="abc")


def test_no_analytic():
    """Test for the no_anayltic transform"""

    tape1 = qp.tape.QuantumScript(shots=2)
    batch, _ = no_analytic(tape1)
    assert batch[0] is tape1

    tape2 = qp.tape.QuantumScript(shots=None)
    with pytest.raises(DeviceError, match="Analytic execution is not supported with abc"):
        no_analytic(tape2, name="abc")


def test_validate_adjoint_trainable_params_obs_warning():
    """Tests warning raised for validate_adjoint_trainable_params with trainable observables."""

    params = qp.numpy.array(0.123)
    tape = qp.tape.QuantumScript([], [qp.expval(2 * qp.RX(params, wires=0))])
    with pytest.warns(UserWarning, match="Differentiating with respect to the input "):
        validate_adjoint_trainable_params(tape)

    params_non_trainable = qp.numpy.array(0.123, requires_grad=False)
    tape = qp.tape.QuantumScript([], [qp.expval(2 * qp.RX(params_non_trainable, wires=0))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # assert no warning raised
        validate_adjoint_trainable_params(tape)


def test_validate_adjoint_trainable_params_state_prep_error():
    """Tests error raised for validate_adjoint_trainable_params with trainable state-preps."""
    tape = qp.tape.QuantumScript([qp.StatePrep(qp.numpy.array([1.0, 0.0]), wires=[0])])
    with pytest.raises(QuantumFunctionError, match="Differentiating with respect to"):
        validate_adjoint_trainable_params(tape)


class TestValidateDeviceWires:
    def test_error(self):
        """Tests for the error raised by validate_device_wires transform."""

        tape1 = qp.tape.QuantumScript([qp.S("a")])
        with pytest.raises(qp.wires.WireError, match="on abc as they contain wires"):
            validate_device_wires(tape1, wires=qp.wires.Wires((0,)), name="abc")

    def test_null_if_no_wires_provided(self):
        """Test that nothing happens if no wires are provided to the transform."""

        tape1 = qp.tape.QuantumScript([qp.S("b")], [qp.expval(qp.PauliZ(0))])
        batch, _ = validate_device_wires(tape1)
        assert batch[0] is tape1

    def test_fill_in_wires(self):
        """Tests that if the wires are provided, measurements without wires gain them."""
        tape1 = qp.tape.QuantumScript([qp.S("b")], [qp.state(), qp.probs()], shots=52)

        wires = qp.wires.Wires(["a", "b", "c"])
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
            tape_with_abstract_wires = QuantumScript([qp.CNOT(wires=qp.wires.Wires(wires))])
            validate_device_wires(tape_with_abstract_wires, name="fictional_device")

        with pytest.raises(
            qp.wires.WireError,
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
            qp.wires.WireError,
            match="on fictional_device as abstract wires are present in the device",
        ):
            jax.jit(jit_wires_dev)([0, 1])

    def test_fill_in_wires_on_snapshots(self):
        """Test that validate_device_wires also fills in the wires on snapshots."""

        tape = qp.tape.QuantumScript([qp.Snapshot(), qp.Snapshot(measurement=qp.probs())])

        (output,), _ = validate_device_wires(tape, wires=qp.wires.Wires((0, 1, 2)))
        mp0 = qp.measurements.StateMP(wires=qp.wires.Wires((0, 1, 2)))
        qp.assert_equal(output[0], qp.Snapshot(measurement=mp0))
        qp.assert_equal(output[1], qp.Snapshot(measurement=qp.probs(wires=(0, 1, 2))))


class TestDecomposeValidation:
    """Unit tests for helper functions in qp.devices.qubit.preprocess"""

    def test_error_if_invalid_op(self):
        """Test that expand_fn throws an error when an operation does not define a matrix or decomposition."""

        tape = QuantumScript(ops=[NoMatNoDecompOp(0)], measurements=[qp.expval(qp.Hadamard(0))])
        with pytest.raises(DeviceError, match="not supported with abc"):
            decompose(tape, lambda op: op.has_matrix, name="abc")

    def test_error_if_invalid_op_decomposer(self):
        """Test that expand_fn throws an error when an operation does not define a matrix or decomposition."""

        tape = QuantumScript(ops=[NoMatNoDecompOp(0)], measurements=[qp.expval(qp.Hadamard(0))])
        with pytest.raises(DeviceError, match="not supported with abc"):
            decompose(
                tape, lambda op: op.has_matrix, decomposer=lambda op: op.decomposition(), name="abc"
            )

    def test_decompose(self):
        """Test that expand_fn doesn't throw any errors for a valid circuit"""
        tape = QuantumScript(
            ops=[qp.PauliX(0), qp.RZ(0.123, wires=0)], measurements=[qp.state()]
        )
        decompose(tape, lambda obj: obj.has_matrix)

    def test_infinite_decomposition_loop(self):
        """Test that a device error is raised if decomposition enters an infinite loop."""

        qs = qp.tape.QuantumScript([InfiniteOp(1.23, 0)])
        with pytest.raises(DeviceError, match=r"Reached recursion limit trying to decompose"):
            decompose(qs, lambda obj: obj.has_matrix)

    @pytest.mark.parametrize(
        "error_type", [RuntimeError, qp.operation.DecompositionUndefinedError]
    )
    def test_error_type_can_be_set(self, error_type):
        """Test that passing a class of Error the ``decompose`` transform allows raising another type
        of error instead of the default ``DeviceError``."""

        decomp_error_tape = QuantumScript(
            ops=[NoMatNoDecompOp(0)], measurements=[qp.expval(qp.Hadamard(0))]
        )
        recursion_error_tape = qp.tape.QuantumScript([InfiniteOp(1.23, 0)])

        with pytest.raises(error_type, match="not supported with abc"):
            decompose(decomp_error_tape, lambda op: op.has_matrix, name="abc", error=error_type)

        with pytest.raises(error_type, match=r"Reached recursion limit trying to decompose"):
            decompose(recursion_error_tape, lambda obj: obj.has_matrix, error=error_type)


class TestValidateObservables:
    """Tests for the validate observables transform."""

    def test_invalid_observable(self):
        """Test that expand_fn throws an error when an observable is invalid."""
        tape = QuantumScript(
            ops=[qp.PauliX(0)], measurements=[qp.expval(qp.GellMann(wires=0, index=1))]
        )
        with pytest.raises(DeviceError, match=r"not supported on abc"):
            validate_observables(tape, lambda obs: obs.name == "PauliX", name="abc")

    def test_invalid_tensor_observable(self):
        """Test that expand_fn throws an error when a tensor includes invalid obserables"""
        tape = QuantumScript(
            ops=[qp.PauliX(0), qp.PauliY(1)],
            measurements=[qp.expval(qp.PauliX(0) @ qp.GellMann(wires=1, index=2))],
        )
        with pytest.raises(DeviceError, match="not supported on device"):
            validate_observables(tape, lambda obj: obj.name == "PauliX")


class TestValidateMeasurements:
    """Tests for the validate measurements transform."""

    @pytest.mark.parametrize(
        "measurements",
        [
            [qp.state()],
            [qp.expval(qp.PauliZ(0))],
            [qp.state(), qp.expval(qp.PauliZ(0)), qp.probs(0)],
            [qp.state(), qp.vn_entropy(0), qp.mutual_info(0, 1)],
        ],
    )
    def test_only_state_measurements(self, measurements):
        """Test that an analytic circuit containing only StateMeasurements works"""
        tape = QuantumScript([], measurements, shots=None)
        validate_measurements(tape, lambda obj: True)

    @pytest.mark.parametrize(
        "measurements",
        [
            [qp.sample(wires=0)],
            [qp.expval(qp.PauliZ(0))],
            [qp.sample(wires=0), qp.expval(qp.PauliZ(0)), qp.probs(0)],
            [qp.classical_shadow(wires=[0])],
            [qp.shadow_expval(qp.PauliZ(0))],
        ],
    )
    def test_only_sample_measurements(self, measurements):
        """Test that a circuit with finite shots containing only SampleMeasurements works"""
        tape = QuantumScript([], measurements, shots=100)
        validate_measurements(tape, sample_measurements=lambda obj: True)

    @pytest.mark.parametrize(
        "measurements",
        [
            [qp.sample(wires=0)],
            [qp.state(), qp.sample(wires=0)],
            [qp.sample(wires=0), qp.expval(qp.PauliZ(0))],
            [qp.classical_shadow(wires=[0])],
            [qp.shadow_expval(qp.PauliZ(0))],
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
            [qp.state()],
            [qp.sample(wires=0), qp.state()],
            [qp.expval(qp.PauliZ(0)), qp.state(), qp.sample(wires=0)],
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
        ops = [qp.Hadamard(0), NoMatOp(1), qp.RZ(0.123, wires=1)]
        measurements = [qp.expval(qp.PauliZ(0)), qp.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements, shots=shots)

        expanded_tapes, _ = decompose(tape, lambda obj: obj.has_matrix)
        expanded_tape = expanded_tapes[0]
        expected = [qp.Hadamard(0), qp.PauliX(1), qp.PauliY(1), qp.RZ(0.123, wires=1)]

        for op, exp in zip(expanded_tape.circuit, expected + measurements):
            qp.assert_equal(op, exp)

        assert tape.shots == expanded_tape.shots

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_decompose_no_expansion(self):
        """Test that expand_fn does nothing to a fully supported quantum script."""
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RZ(0.123, wires=1)]
        measurements = [qp.expval(qp.PauliZ(0)), qp.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)
        expanded_tapes, _ = decompose(tape, lambda obj: obj.has_matrix)
        expanded_tape = expanded_tapes[0]

        for op, exp in zip(expanded_tape.circuit, ops + measurements):
            qp.assert_equal(op, exp)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize("validation_transform", (validate_measurements, validate_observables))
    def test_valdiate_measurements_non_commuting_measurements(self, validation_transform):
        """Test that validate_measurements and validate_observables works when non commuting measurements exist in the circuit."""

        qs = QuantumScript([NoMatOp("a")], [qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliY(0))])

        new_qs, _ = validation_transform(qs, lambda obj: True)
        new_qs = new_qs[0]
        assert new_qs.measurements == qs.measurements

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        "prep_op",
        (
            qp.BasisState([1], wires=0),
            qp.StatePrep([0, 1], wires=1),
            qp.AmplitudeEmbedding([0, 1], wires=1),
        ),
    )
    def test_decompose_state_prep_skip_first(self, prep_op):
        """Test that the decompose only expands mid-circuit instances of StatePrepBase if requested."""
        ops = [
            prep_op,
            qp.Hadamard(wires=0),
            qp.StatePrep([0, 1], wires=1),
            qp.BasisState([1], wires=0),
            qp.RZ(0.123, wires=1),
            qp.AmplitudeEmbedding([0, 1, 0, 0], wires=[0, 1]),
        ]
        measurements = [qp.expval(qp.PauliZ(0)), qp.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tapes, _ = decompose(
            tape, lambda obj: obj.has_matrix, skip_initial_state_prep=True
        )
        expanded_tape = expanded_tapes[0]
        expected = [
            prep_op,
            qp.Hadamard(0),
            qp.RY(3.14159265, wires=1),  # decomposition of StatePrep
            qp.PauliX(wires=0),  # decomposition of BasisState
            qp.RZ(0.123, wires=1),
            qp.RY(1.57079633, wires=[1]),  # decomposition of AmplitudeEmbedding
            qp.CNOT(wires=[0, 1]),
            qp.RY(1.57079633, wires=[1]),
            qp.CNOT(wires=[0, 1]),
        ]

        assert expanded_tape.circuit == expected + measurements

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    @pytest.mark.parametrize(
        "prep_op",
        (
            qp.BasisState([1], wires=0),
            qp.StatePrep([0, 1], wires=1),
            qp.AmplitudeEmbedding([0, 1], wires=1),
        ),
    )
    def test_decompose_initial_state_prep_if_requested(self, prep_op):
        """Test that initial state prep operations are decomposed if skip_initial_state_prep is False."""

        tape = qp.tape.QuantumScript([prep_op])
        batch, _ = decompose(tape, lambda obj: obj.has_matrix, skip_initial_state_prep=False)
        new_tape = batch[0]

        assert new_tape[0] != prep_op

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_decompose_with_device_wires_and_target_gates(self):
        """Test that decompose works correctly with device_wires and target_gates parameters."""
        # Mock a simple target gate set
        target_gates = {"Hadamard", "CNOT", "RZ", "RX", "RY", "GlobalPhase"}
        device_wires = qp.wires.Wires([0, 1, 2, 3])

        # Create a tape with an operation that needs decomposition
        tape = qp.tape.QuantumScript([qp.QFT(wires=[0, 1])], [qp.expval(qp.Z(0))])

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

        ops = [qp.IsingXX(1.2, wires=tape_wires_list)]

        tape = qp.tape.QuantumScript(ops, [qp.expval(qp.Z(tape_wires_list[0]))])
        device_wires = qp.wires.Wires(device_wires_list) if device_wires_list else None

        # Directly test the work wire calculation logic
        if device_wires is None:
            computed_work_wires = None  # No constraint on work wires
        else:
            # This is the actual calculation from the decompose function
            computed_work_wires = len(set(device_wires) - set(tape.wires))

        # Verify our calculation matches expected
        assert computed_work_wires == expected_work_wires

        # Use a spy to verify the parameter is passed correctly to the decomposition logic
        spy = mocker.spy(qp.devices.preprocess, "_operator_decomposition_gen")

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
                if "num_work_wires" in call_kwargs:
                    if call_kwargs["num_work_wires"] == expected_work_wires:
                        found_correct_call = True
                        break

            assert (
                found_correct_call
            ), f"Expected num_work_wires={expected_work_wires} not found in calls"


class TestGraphModeExclusiveFeatures:
    """Tests that only work when graph mode is enabled.

    NOTE: All tests in this suite will auto-enable graph mode via fixture.
    """

    @pytest.fixture(autouse=True)
    def enable_graph_mode_only(self):
        """Auto-enable graph mode for all tests in this class."""
        qp.decomposition.enable_graph()
        yield
        qp.decomposition.disable_graph()

    def test_work_wire_unavailability_causes_fallback(self):
        """Test that decompositions requiring more work wires than available are discarded.

        This addresses the reviewer's question: if a device has 1 wire but a decomposition
        requires 5 burnable work wires, that decomposition should be discarded.
        """

        class MyOp(qp.operation.Operator):
            num_wires = 1

        # Fallback decomposition (no work wires needed)
        @qp.register_resources({qp.H: 2})
        def decomp_fallback(wires):
            qp.H(wires)
            qp.H(wires)

        # Work wire decomposition (needs 5 burnable wires)
        @qp.register_resources({qp.X: 1}, work_wires={"burnable": 5})
        def decomp_with_work_wire(wires):
            qp.X(wires)

        qp.add_decomps(MyOp, decomp_fallback, decomp_with_work_wire)

        tape = qp.tape.QuantumScript([MyOp(0)])
        device_wires = qp.wires.Wires(1)  # Only 1 wire, insufficient for 5 burnable
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


@pytest.fixture()
def check_deprecated():
    with pytest.warns(
        PennyLaneDeprecationWarning,
        match="The mid_circuit_measurements transform is deprecated",
    ):
        yield


@pytest.mark.usefixtures("check_deprecated")
class TestMidCircuitMeasurements:
    """Unit tests for the mid_circuit_measurements preprocessing transform"""

    @pytest.mark.parametrize(
        "mcm_method, shots, expected_transform",
        [
            ("deferred", 10, qp.defer_measurements),
            ("deferred", None, qp.defer_measurements),
            (None, None, qp.defer_measurements),
            (None, 10, qp.dynamic_one_shot),
            ("one-shot", 10, qp.dynamic_one_shot),
        ],
    )
    def test_mcm_method(self, mcm_method, shots, expected_transform, mocker):
        """Test that the preprocessing transform adheres to the specified transform"""
        dev = qp.device("default.qubit")
        mcm_config = {"postselect_mode": None, "mcm_method": mcm_method}
        tape = QuantumScript([qp.ops.MidMeasure(0)], [], shots=shots)
        spy = mocker.spy(expected_transform, "_tape_transform")

        _, _ = mid_circuit_measurements(tape, dev, mcm_config)
        spy.assert_called_once()

    @pytest.mark.parametrize("mcm_method", ["device", "tree-traversal"])
    @pytest.mark.parametrize("shots", [10, None])
    def test_device_mcm_method(self, mcm_method, shots):
        """Test that no transform is applied by mid_circuit_measurements when the
        mcm method is handled by the device"""
        dev = qp.device("default.qubit")
        mcm_config = {"postselect_mode": None, "mcm_method": mcm_method}
        tape = QuantumScript([qp.ops.MidMeasure(0)], [], shots=shots)

        (new_tape,), post_processing_fn = mid_circuit_measurements(tape, dev, mcm_config)

        assert qp.equal(tape, new_tape)
        assert post_processing_fn == null_postprocessing

    def test_error_incompatible_mcm_method(self):
        """Test that an error is raised if requesting the one-shot transform without shots"""
        dev = qp.device("default.qubit")
        shots = None
        mcm_config = {"postselect_mode": None, "mcm_method": "one-shot"}
        tape = QuantumScript([qp.ops.MidMeasure(0)], [], shots=shots)

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

        tape = qp.tape.QuantumScript([], measurements=[qp.probs()], shots=10)

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
                qp.expval(qp.PauliY(wires=0) @ qp.PauliY(wires=1)),
                lambda theta: np.sin(theta) * np.sin(theta / 2),
            ),
            (qp.var(qp.Y(wires=1)), lambda theta: 1 - np.sin(theta / 2) ** 2),
            (
                qp.probs(),
                lambda theta: np.outer(
                    np.outer(
                        [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2],
                        [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
                    ),
                    [1, 0, 0, 0],
                ).flatten(),
            ),
            (
                qp.probs(wires=[1]),
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
        dev = qp.device("default.qubit", wires=4)

        tape = qp.tape.QuantumScript(
            [qp.RX(theta, 0), qp.RX(theta / 2, 1)],
            measurements=[input_measurement],
            shots=shots,
        )
        (validated_tape,), _ = validate_device_wires(tape, dev.wires)
        tapes, fn = meas_transform(validated_tape)

        assert len(tapes) == 1
        assert len(tapes[0].measurements) == 1
        expected_type = SampleMP if meas_transform == measurements_from_samples else CountsMP
        assert isinstance(tapes[0].measurements[0], expected_type)

        output = qp.execute(tapes, device=dev)
        res = fn(output)

        if dev.shots.has_partitioned_shots:
            assert len(res) == dev.shots.num_copies

        assert np.allclose(res, expected_res(theta), atol=0.05)

    @pytest.mark.parametrize(
        "counts_kwargs",
        [
            {},
            {"wires": [2, 3]},
            {"op": qp.PauliX(wires=0) @ qp.PauliX(wires=1) @ qp.PauliX(wires=2)},
        ],
    )
    @pytest.mark.parametrize(
        "meas_transform", [measurements_from_counts, measurements_from_samples]
    )
    def test_with_counts_output(self, counts_kwargs, meas_transform):
        """Test that returning counts works as expected for all-wires, specific wires, or an observable,
        when using both the measurements_from_counts and measurements_from_samples transforms."""

        dev = qp.device("default.qubit", wires=4)

        @validate_device_wires(wires=dev.wires)
        @qp.set_shots(5000)
        @qp.qnode(dev)
        def basic_circuit(theta: float):
            qp.RY(theta, 0)
            qp.RY(theta / 2, 1)
            qp.RY(2 * theta, 2)
            qp.RY(theta, 3)
            return qp.counts(**counts_kwargs)

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
            {"op": qp.PauliX(wires=0) @ qp.PauliX(wires=1) @ qp.PauliX(wires=2)},
        ],
    )
    @pytest.mark.parametrize(
        "meas_transform", [measurements_from_counts, measurements_from_samples]
    )
    def test_with_sample_output(self, sample_kwargs, meas_transform, seed):
        """Test that returning sample works as expected for all-wires, specific wires, or an observable,
        when using both the measurements_from_counts and measurements_from_samples transforms."""

        dev = qp.device("default.qubit", wires=4, seed=seed)

        @validate_device_wires(wires=dev.wires)
        @qp.set_shots(5000)
        @qp.qnode(dev)
        def basic_circuit(theta: float):
            qp.RY(theta, 0)
            qp.RY(theta / 2, 1)
            qp.RY(2 * theta, 2)
            qp.RY(theta, 3)
            return qp.sample(**sample_kwargs)

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
            {"op": qp.Z(wires=0) @ qp.Z(wires=1) @ qp.Z(wires=2)},
        ],
    )
    @pytest.mark.parametrize("all_outcomes", [True, False])
    @pytest.mark.parametrize(
        "meas_transform", [measurements_from_counts, measurements_from_samples]
    )
    def test_counts_all_outcomes(self, counts_kwargs, all_outcomes, meas_transform):
        """Test that the measurements with counts when only some states have non-zero counts,
        and confirm that all_counts returns the expected entries"""

        dev = qp.device("default.qubit", wires=4)

        @validate_device_wires(wires=dev.wires)
        @qp.set_shots(5000)
        @qp.qnode(dev)
        def basic_circuit():
            return qp.counts(**counts_kwargs, all_outcomes=all_outcomes)

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

        dev = qp.device("default.qubit", wires=4, seed=seed)

        @qp.set_shots(5000)
        @qp.qnode(dev)
        def basic_circuit(theta: float):
            qp.RY(theta, 0)
            qp.RY(theta / 2, 1)
            qp.RY(2 * theta, 2)
            qp.RY(theta, 3)
            return (
                qp.expval(qp.PauliX(wires=0) @ qp.PauliX(wires=1)),
                qp.var(qp.PauliX(wires=2)),
                qp.counts(qp.PauliX(wires=0) @ qp.PauliX(wires=1) @ qp.PauliX(wires=2)),
                qp.sample(qp.PauliX(wires=0) @ qp.PauliX(wires=1) @ qp.PauliX(wires=2)),
                qp.probs(wires=[3]),
            )

        theta = 1.9
        tape = qp.workflow.construct_tape(basic_circuit)(theta)
        (validated_tape,), _ = validate_device_wires(tape, dev.wires)
        tapes, fn = meas_transform(validated_tape)

        sample_output = qp.execute(tapes, device=dev)
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

        tape = qp.tape.QuantumScript(
            [], measurements=[qp.expval(qp.Z(0)), qp.var(qp.X(1))], shots=shots
        )

        (new_tape,), fn = meas_transform(tape)

        if shots is None:
            assert qp.equal(new_tape, tape)
            assert fn == null_postprocessing
        else:
            assert len(new_tape.measurements) == 1
            assert isinstance(new_tape.measurements[0], (SampleMP, CountsMP))
            assert fn != null_postprocessing


def test_validate_multiprocessing_workers_None():
    """Test that validation does not fail when max_workers is None"""
    qs = QuantumScript(
        [qp.Rot(0.1, 0.2, 0.3, wires=0), qp.CNOT([0, 1])],
        [qp.expval(qp.PauliZ(1))],
    )
    device = qp.devices.DefaultQubit()
    validate_multiprocessing_workers(qs, None, device)


class TestDeviceResolveDynamicWires:
    def test_many_allocations_no_wires(self):
        """Test that min integer will keep incrementing to higher numbers."""

        allocations = [qp.allocation.Allocate.from_num_wires(1) for _ in range(10)]
        ops = [qp.X(op.wires) for op in allocations]
        tape = qp.tape.QuantumScript(allocations + ops)

        [new_tape], fn = device_resolve_dynamic_wires(tape, wires=None)

        assert fn(("a",)) == "a"
        for op, wire in zip(new_tape.operations, range(0, 10)):
            qp.assert_equal(op, qp.X(wire))

    def test_error_on_not_enough_available_wires(self):
        """Test that an error is raised if there are not enough available wires on the device."""

        allocation = qp.allocation.Allocate.from_num_wires(2)
        ops = [qp.X(w) for w in allocation.wires]
        tape = qp.tape.QuantumScript([allocation] + ops)

        with pytest.raises(
            qp.exceptions.AllocationError, match=r"Not enough available wires on device"
        ):
            device_resolve_dynamic_wires(tape, wires=qp.wires.Wires([0]))

    def test_many_allocations_device_wires(self):
        """Test that provided device wires are used properly."""

        allocations = [qp.allocation.Allocate.from_num_wires(1) for _ in range(10)]
        ops = [qp.X(op.wires) for op in allocations]
        tape = qp.tape.QuantumScript(allocations + ops)

        wires = qp.wires.Wires(list(range(10)))
        [new_tape], fn = device_resolve_dynamic_wires(tape, wires=wires)

        assert fn(("a",)) == "a"
        for op, wire in zip(new_tape.operations, wires):
            qp.assert_equal(op, qp.X(wire))

    def test_min_int_non_integer_algorithmic_wires(self):
        """Test that a min int can be calculated when the tape contains non-integer wires."""

        allocation = qp.allocation.Allocate.from_num_wires(1)
        ops = [qp.Z("a"), qp.Z("b"), qp.Z("my_wire"), qp.X(allocation.wires)]
        tape = qp.tape.QuantumScript([allocation] + ops)

        [new_tape], _ = device_resolve_dynamic_wires(tape, wires=None)
        qp.assert_equal(new_tape[-1], qp.X(0))

    def test_min_int_unsorted_wires(self):
        """Test that the min_int is the smallest integer larger than all integers in operation wires."""
        allocation = qp.allocation.Allocate.from_num_wires(1)
        ops = [qp.Z(5), qp.Z(11), qp.Z(4), qp.X(allocation.wires)]
        tape = qp.tape.QuantumScript([allocation] + ops)

        [new_tape], _ = device_resolve_dynamic_wires(tape, wires=None)
        qp.assert_equal(new_tape[-1], qp.X(12))

    def test_error_if_no_zeroed_wires_left_no_reset(self):
        """Test that an error is raised if all zeroed wires have been used and cant reset."""

        alloc1 = qp.allocation.Allocate.from_num_wires(1)
        dealloc1 = qp.allocation.Deallocate(alloc1.wires)
        alloc2 = qp.allocation.Allocate.from_num_wires(1)
        dealloc2 = qp.allocation.Deallocate(alloc2.wires)

        tape = qp.tape.QuantumScript(
            [qp.Z(0), alloc1, qp.X(alloc1.wires), dealloc1, alloc2, qp.Y(alloc2.wires), dealloc2]
        )

        with pytest.raises(
            qp.exceptions.AllocationError, match="Not enough available wires on device"
        ):
            device_resolve_dynamic_wires(tape, wires=(0, 1), allow_resets=False)

    def test_no_resets_min_int(self):
        """Test that if a min_int is specified along with allow_resets=False, then fresh wires keep getting added."""

        alloc1 = qp.allocation.Allocate.from_num_wires(1)
        dealloc1 = qp.allocation.Deallocate(alloc1.wires)
        alloc2 = qp.allocation.Allocate.from_num_wires(1)
        dealloc2 = qp.allocation.Deallocate(alloc2.wires)

        tape = qp.tape.QuantumScript(
            [qp.Z(0), alloc1, qp.X(alloc1.wires), dealloc1, alloc2, qp.Y(alloc2.wires), dealloc2]
        )

        [new_tape], _ = device_resolve_dynamic_wires(tape, wires=None, allow_resets=False)

        expected = qp.tape.QuantumScript([qp.Z(0), qp.X(1), qp.Y(2)])
        qp.assert_equal(expected, new_tape)
