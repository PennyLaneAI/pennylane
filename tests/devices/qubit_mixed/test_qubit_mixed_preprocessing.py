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
Tests for default mixed device preprocessing.
"""

import warnings

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.default_mixed import (
    DefaultMixed,
    observable_stopping_condition,
    stopping_condition,
)
from pennylane.exceptions import DeviceError


# pylint: disable=protected-access
def test_mid_circuit_measurement_preprocessing():
    """Test mid-circuit measurement preprocessing not supported with default.mixed device."""
    dev = DefaultMixed(wires=2)

    # Define operations and mid-circuit measurement
    m0 = qml.measure(0)
    ops = [*m0.measurements, qml.ops.Conditional(m0, qml.X(0))]

    # Construct the QuantumScript
    tape = qml.tape.QuantumScript(ops, [qml.expval(qml.Z(0))], shots=1000)

    # Process the tape with the device's preprocess method
    transform_program, _ = dev.preprocess()

    # Apply the transform program to the tape
    processed_tapes, _ = transform_program([tape])

    # There should be one processed tape
    assert len(processed_tapes) == 1, "Expected exactly one processed tape."
    processed_tape = processed_tapes[0]

    # Check that mid-circuit measurements have been deferred
    mid_measure_ops = [
        op for op in processed_tape.operations if isinstance(op, qml.measurements.MidMeasureMP)
    ]
    assert len(mid_measure_ops) == 0, "Mid-circuit measurements were not deferred properly."
    assert processed_tape.circuit == [qml.CNOT([0, 1]), qml.CNOT([1, 0]), qml.expval(qml.Z(0))]


class NoMatOp(qml.operation.Operation):
    """Dummy operation for expanding circuit in qubit devices."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


# pylint: disable=too-few-public-methods
class NoMatNoDecompOp(qml.operation.Operation):
    """Dummy operation for checking check_validity throws error when expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


# pylint: disable=too-few-public-methods
class TestPreprocessing:
    """Unit tests for the preprocessing method."""

    def test_error_if_device_option_not_available(self):
        """Test that an error is raised if a device option is requested but not a valid option."""
        dev = DefaultMixed()

        config = ExecutionConfig(device_options={"invalid_option": "val"})
        with pytest.raises(DeviceError, match="device option invalid_option"):
            dev.preprocess(config)

    def test_chooses_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        dev = DefaultMixed()

        config = ExecutionConfig(gradient_method="best")

        _, new_config = dev.preprocess(config)

        assert new_config.gradient_method == "backprop"
        assert new_config.use_device_gradient
        assert not new_config.grad_on_execution

    def test_circuit_wire_validation(self):
        """Test that preprocessing validates wires on the circuits being executed."""
        dev = DefaultMixed(wires=3)

        circuit_valid_0 = qml.tape.QuantumScript([qml.PauliX(0)])
        program, _ = dev.preprocess()
        circuits, _ = program([circuit_valid_0])
        assert circuits[0].circuit == circuit_valid_0.circuit

        circuit_valid_1 = qml.tape.QuantumScript([qml.PauliY(1)])
        program, _ = dev.preprocess()
        circuits, _ = program([circuit_valid_0, circuit_valid_1])
        assert circuits[0].circuit == circuit_valid_0.circuit
        assert circuits[1].circuit == circuit_valid_1.circuit

        invalid_circuit = qml.tape.QuantumScript([qml.PauliZ(4)])
        program, _ = dev.preprocess()

        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program([invalid_circuit])

        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program([circuit_valid_0, invalid_circuit])

    @pytest.mark.parametrize(
        "mp_fn,mp_cls,shots",
        [
            (qml.sample, qml.measurements.SampleMP, 10),
            (qml.state, qml.measurements.StateMP, None),
            (qml.probs, qml.measurements.ProbabilityMP, None),
        ],
    )
    def test_measurement_is_swapped_out(self, mp_fn, mp_cls, shots):
        """Test that preprocessing swaps out any MeasurementProcess with no wires or obs"""
        dev = DefaultMixed(wires=3)
        original_mp = mp_fn()
        exp_z = qml.expval(qml.PauliZ(0))
        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [original_mp, exp_z], shots=shots)
        program, _ = dev.preprocess()
        tapes, _ = program([qs])
        assert len(tapes) == 1
        tape = tapes[0]
        assert tape.operations == qs.operations
        assert tape.measurements != qs.measurements
        qml.assert_equal(tape.measurements[0], mp_cls(wires=[0, 1, 2]))
        assert tape.measurements[1] is exp_z

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.PauliX(0), True),
            (qml.Hermitian(np.eye(2), wires=0), False),
            (qml.Snapshot(), True),
            (qml.RX(1.1, 0), True),
            (qml.DepolarizingChannel(0.4, wires=0), True),
            (qml.AmplitudeDamping(0.1, wires=0), True),
            (NoMatOp(0), False),
        ],
    )
    def test_accepted_operator(self, op, expected):
        """Test that stopping_condition works correctly"""
        res = stopping_condition(op)
        assert res == expected

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(0), True),
            (qml.DepolarizingChannel(0.4, wires=0), False),
            (qml.Hermitian(np.eye(2), wires=0), True),
            (qml.Snapshot(), False),
            (qml.s_prod(1.2, qml.PauliX(0)), True),
            (qml.sum(qml.s_prod(1.2, qml.PauliX(0)), qml.PauliZ(1)), True),
            (qml.prod(qml.PauliX(0), qml.PauliZ(1)), True),
            # Simple LinearCombination with valid observables
            (qml.Hamiltonian([1.0, 0.5], [qml.PauliX(0), qml.PauliZ(1)]), True),
            # LinearCombination with mixed valid/invalid ops
            (
                qml.Hamiltonian([1.0, 0.5], [qml.PauliX(0), qml.DepolarizingChannel(0.4, wires=0)]),
                False,
            ),
            # LinearCombination with all invalid ops
            (
                qml.Hamiltonian(
                    [1.0, 0.5], [qml.Snapshot(), qml.DepolarizingChannel(0.4, wires=0)]
                ),
                False,
            ),
            # Complex LinearCombination
            (
                qml.Hamiltonian(
                    [0.3, 0.7], [qml.prod(qml.PauliX(0), qml.PauliZ(1)), qml.PauliY(2)]
                ),
                True,
            ),
        ],
    )
    def test_accepted_observable(self, obs, expected):
        """Test that observable_stopping_condition works correctly"""
        res = observable_stopping_condition(obs)
        assert res == expected

    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1]), qml.RX(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        device = DefaultMixed(wires=2)

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        assert tapes[0].circuit == ops + measurements

    def test_batch_transform_broadcast(self):
        """Test that batch_transform does nothing when batching is required but
        internal PennyLane broadcasting can be used (diff method != adjoint)"""
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        device = DefaultMixed(wires=2)

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        assert tapes[0].circuit == ops + measurements

    def test_preprocess_batch_transform(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program, _ = DefaultMixed(wires=2).preprocess()
        res_tapes, batch_fn = program(tapes)

        assert len(res_tapes) == 2
        for res_tape, measurement in zip(res_tapes, measurements):
            for op, expected_op in zip(res_tape.operations, ops):
                qml.assert_equal(op, expected_op)
            assert res_tape.measurements == [measurement]

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_expand(self):
        """Test that preprocess returns the correct tapes when expansion is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliZ(1))]]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qml.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program, _ = DefaultMixed(wires=2).preprocess()
        res_tapes, batch_fn = program(tapes)

        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, exp in zip(t.circuit, expected + measurements[i]):
                qml.assert_equal(op, exp)

        val = (("a", "b"), "c", "d")
        assert batch_fn(val) == (("a", "b"), "c")

    def test_preprocess_batch_and_expand(self):
        """Test that preprocess returns the correct tapes when batching and expanding
        is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program, _ = DefaultMixed(wires=2).preprocess()
        res_tapes, batch_fn = program(tapes)
        expected_ops = [
            qml.Hadamard(0),
            qml.PauliX(1),
            qml.PauliY(1),
            qml.RX([np.pi, np.pi / 2], wires=1),
        ]

        assert len(res_tapes) == 2
        for res_tape, measurement in zip(res_tapes, measurements):
            for op, expected_op in zip(res_tape.operations, expected_ops):
                qml.assert_equal(op, expected_op)
            assert res_tape.measurements == [measurement]

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_check_validity_fail(self):
        """Test that preprocess throws an error if the batched and expanded tapes have
        unsupported operators."""
        ops = [qml.Hadamard(0), NoMatNoDecompOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliZ(1))]]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qml.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program, _ = DefaultMixed(wires=2).preprocess()
        with pytest.raises(DeviceError, match="Operator NoMatNoDecompOp"):
            program(tapes)

    @pytest.mark.parametrize(
        "readout_err, req_warn",
        [
            (0.1, True),
            (None, False),
        ],
    )
    @pytest.mark.parametrize(
        "measurements",
        [
            [qml.state()],
            [qml.density_matrix(0)],
            [qml.state(), qml.density_matrix([1, 2])],
            [qml.state(), qml.expval(qml.PauliZ(1))],
        ],
    )
    def test_preprocess_warns_measurement_error_state(self, readout_err, req_warn, measurements):
        """Test that preprocess raises a warning if there is an analytic state measurement and
        measurement error."""
        tapes = [
            qml.tape.QuantumScript(ops=[], measurements=measurements),
            qml.tape.QuantumScript(
                ops=[qml.Hadamard(0), qml.RZ(0.123, wires=1)], measurements=measurements
            ),
        ]
        device = DefaultMixed(wires=3, readout_prob=readout_err)
        program, _ = device.preprocess()

        with warnings.catch_warnings(record=True) as warning:
            program(tapes)
            if req_warn:
                assert len(warning) != 0
                for warn in warning:
                    assert "is not affected by readout error" in str(warn.message)
            else:
                assert len(warning) == 0

    def test_preprocess_linear_combination_observable(self):
        """Test that the device's preprocessing handles linear combinations of observables correctly."""
        dev = DefaultMixed(wires=2)

        # Define the linear combination observable
        obs = qml.PauliX(0) + 2 * qml.PauliZ(1)

        # Define the circuit
        ops = [qml.Hadamard(0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(obs)]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        # Preprocess the tape
        program, _ = dev.preprocess()
        tapes, _ = program([tape])

        # Check that the measurement is handled correctly during preprocessing
        # The tape should remain unchanged as the device should accept the observable
        assert len(tapes) == 1
        processed_tape = tapes[0]

        # Verify that the operations and measurements are unchanged
        assert processed_tape.operations == tape.operations
        assert processed_tape.measurements == tape.measurements

        # Ensure that the linear combination observable is accepted
        measurement = processed_tape.measurements[0]
        assert isinstance(measurement.obs, qml.ops.Sum)

    @pytest.mark.jax
    def test_preprocess_jax_seed(self):
        """Test that the device's preprocessing correctly handles JAX PRNG keys as seeds."""
        jax = pytest.importorskip("jax")

        seed = jax.random.PRNGKey(42)

        dev = DefaultMixed(wires=1, seed=seed)

        # Preprocess the device
        _ = dev.preprocess()

        # Since preprocessing does not modify the seed, we check the device's attributes
        # Verify that the device's _prng_key is set correctly
        # pylint: disable=protected-access
        assert dev._prng_key is seed

        # Verify that the device's _rng is initialized appropriately
        # pylint: disable=protected-access
        assert dev._rng is not None
