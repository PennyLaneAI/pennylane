# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit preprocessing."""

import pytest

import numpy as np

import pennylane as qml
from pennylane.devices import DefaultQubit, ExecutionConfig

from pennylane.devices.default_qubit import stopping_condition


class NoMatOp(qml.operation.Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


# pylint: disable=too-few-public-methods
class NoMatNoDecompOp(qml.operation.Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


def test_snapshot_multiprocessing_execute():
    """DefaultQubit cannot execute tapes with Snapshot if `max_workers` is not `None`"""
    dev = qml.device("default.qubit", max_workers=2)

    tape = qml.tape.QuantumScript(
        [
            qml.Snapshot(),
            qml.Hadamard(wires=0),
            qml.Snapshot("very_important_state"),
            qml.CNOT(wires=[0, 1]),
            qml.Snapshot(),
        ],
        [qml.expval(qml.PauliX(0))],
    )
    with pytest.raises(RuntimeError, match="ProcessPoolExecutor cannot execute a QuantumScript"):
        program, _ = dev.preprocess()
        program([tape])


class TestConfigSetup:
    """Tests involving setting up the execution config."""

    def test_choose_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        config = qml.devices.ExecutionConfig(gradient_method="best")
        _, config = qml.device("default.qubit").preprocess(config)
        assert config.gradient_method == "backprop"
        assert config.use_device_gradient
        assert not config.grad_on_execution

    def test_config_choices_for_adjoint(self):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        config = qml.devices.ExecutionConfig(
            gradient_method="adjoint", use_device_gradient=None, grad_on_execution=None
        )
        _, new_config = qml.device("default.qubit").preprocess(config)

        assert new_config.use_device_gradient
        assert new_config.grad_on_execution


# pylint: disable=too-few-public-methods
class TestPreprocessing:
    """Unit tests for the preprocessing method."""

    def test_chooses_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        dev = DefaultQubit()

        config = ExecutionConfig(
            gradient_method="best", use_device_gradient=None, grad_on_execution=None
        )

        _, new_config = dev.preprocess(config)

        assert new_config.gradient_method == "backprop"
        assert new_config.use_device_gradient
        assert not new_config.grad_on_execution

    def test_config_choices_for_adjoint(self):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        dev = DefaultQubit()

        config = ExecutionConfig(
            gradient_method="adjoint", use_device_gradient=None, grad_on_execution=None
        )

        _, new_config = dev.preprocess(config)

        assert new_config.use_device_gradient
        assert new_config.grad_on_execution

    @pytest.mark.parametrize("max_workers", [None, 1, 2, 3])
    def test_config_choices_for_threading(self, max_workers):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        dev = DefaultQubit()

        config = ExecutionConfig(device_options={"max_workers": max_workers})
        _, new_config = dev.preprocess(config)

        assert new_config.device_options["max_workers"] == max_workers

    def test_circuit_wire_validation(self):
        """Test that preprocessing validates wires on the circuits being executed."""
        dev = DefaultQubit(wires=3)
        circuit_valid_0 = qml.tape.QuantumScript([qml.PauliX(0)])
        program, _ = dev.preprocess()
        circuits, _ = program([circuit_valid_0])
        assert circuits[0].circuit == circuit_valid_0.circuit

        circuit_valid_1 = qml.tape.QuantumScript([qml.PauliX(1)])
        program, _ = dev.preprocess()
        circuits, _ = program([circuit_valid_0, circuit_valid_1])
        assert circuits[0].circuit == circuit_valid_0.circuit
        assert circuits[1].circuit == circuit_valid_1.circuit

        invalid_circuit = qml.tape.QuantumScript([qml.PauliX(4)])
        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program, _ = dev.preprocess()
            program(
                [
                    invalid_circuit,
                ]
            )

        with pytest.raises(qml.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program, _ = dev.preprocess()
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
        """Test that preprocessing swaps out any MP with no wires or obs"""
        dev = DefaultQubit(wires=3)
        original_mp = mp_fn()
        exp_z = qml.expval(qml.PauliZ(0))
        qs = qml.tape.QuantumScript([qml.Hadamard(0)], [original_mp, exp_z], shots=shots)
        program, _ = dev.preprocess()
        tapes, _ = program([qs])
        assert len(tapes) == 1
        tape = tapes[0]
        assert tape.operations == qs.operations
        assert tape.measurements != qs.measurements
        assert qml.equal(tape.measurements[0], mp_cls(wires=[0, 1, 2]))
        assert tape.measurements[1] is exp_z

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.PauliX(0), True),
            (qml.CRX(0.1, wires=[0, 1]), True),
            (qml.Snapshot(), True),
            (qml.Barrier(), False),
            (qml.QFT(wires=range(5)), True),
            (qml.QFT(wires=range(10)), False),
            (qml.GroverOperator(wires=range(10)), True),
            (qml.GroverOperator(wires=range(14)), False),
            (qml.pow(qml.RX(1.1, 0), 3), True),
            (qml.pow(qml.RX(qml.numpy.array(1.1), 0), 3), False),
        ],
    )
    def test_accepted_operator(self, op, expected):
        """Test that _accepted_operator works correctly"""
        res = stopping_condition(op)
        assert res == expected

    def test_adjoint_only_one_wire(self):
        """Tests adjoint accepts operators with no parameters or a sinlge parameter and a generator."""

        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]

        class MatOp(qml.operation.Operation):
            """Dummy operation for expanding circuit."""

            # pylint: disable=arguments-renamed, invalid-overridden-method
            @property
            def has_matrix(self):
                return True

            def decomposition(self):
                return [qml.PauliX(self.wires), qml.PauliY(self.wires)]

        tape1 = qml.tape.QuantumScript([MatOp(wires=0)])
        batch, _ = program((tape1,))
        assert batch[0].circuit == tape1.circuit

        tape2 = qml.tape.QuantumScript([MatOp(1.2, wires=0)])
        batch, _ = program((tape2,))
        assert batch[0].circuit != tape2.circuit

        tape3 = qml.tape.QuantumScript([MatOp(1.2, 2.3, wires=0)])
        batch, _ = program((tape2,))
        assert batch[0].circuit != tape3.circuit

        class CustomOpWithGenerator(qml.operation.Operator):
            """A custom operator with a generator."""

            def generator(self):
                return qml.PauliX(0)

            # pylint: disable=arguments-renamed, invalid-overridden-method
            @property
            def has_matrix(self):
                return True

        tape4 = qml.tape.QuantumScript([CustomOpWithGenerator(1.2, wires=0)])
        batch, _ = program((tape4,))
        assert batch[0].circuit == tape4.circuit


class TestPreprocessingIntegration:
    """Test preprocess produces output that can be executed by the device."""

    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        device = qml.device("default.qubit")

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            assert qml.equal(op, expected)

    def test_batch_transform_broadcast_not_adjoint(self):
        """Test that batch_transform does nothing when batching is required but
        internal PennyLane broadcasting can be used (diff method != adjoint)"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)
        device = qml.devices.DefaultQubit()

        program, _ = device.preprocess()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        assert tapes[0].circuit == ops + measurements

    def test_batch_transform_broadcast_adjoint(self):
        """Test that batch_transform splits broadcasted tapes correctly when
        the diff method is adjoint"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        execution_config = ExecutionConfig(gradient_method="adjoint")

        device = qml.devices.DefaultQubit()

        program, _ = device.preprocess(execution_config=execution_config)
        tapes, _ = program([tape])
        expected_ops = [
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(tapes) == 2
        for i, t in enumerate(tapes):
            for op, expected in zip(t.circuit, expected_ops[i] + measurements):
                assert qml.equal(op, expected)

    def test_preprocess_batch_transform_not_adjoint(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program, _ = qml.device("default.qubit").preprocess()
        res_tapes, batch_fn = program(tapes)

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, ops):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i == 0:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_batch_transform_adjoint(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        execution_config = ExecutionConfig(gradient_method="adjoint")

        program, _ = qml.device("default.qubit").preprocess(execution_config=execution_config)
        res_tapes, batch_fn = program(tapes)

        expected_ops = [
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(res_tapes) == 4
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 2]):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 2:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        val = ([[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_expand(self):
        """Test that preprocess returns the correct tapes when expansion is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qml.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program, _ = qml.device("default.qubit").preprocess()
        res_tapes, batch_fn = program(tapes)

        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, exp in zip(t.circuit, expected + measurements[i]):
                assert qml.equal(op, exp)

        val = (("a", "b"), "c", "d")
        assert batch_fn(val) == (("a", "b"), "c")

    def test_preprocess_split_and_expand_not_adjoint(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program, _ = qml.device("default.qubit").preprocess()
        res_tapes, batch_fn = program(tapes)
        expected_ops = [
            qml.Hadamard(0),
            qml.PauliX(1),
            qml.PauliY(1),
            qml.RX([np.pi, np.pi / 2], wires=1),
        ]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i == 0:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_split_and_expand_adjoint(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qml.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        execution_config = ExecutionConfig(gradient_method="adjoint")

        program, _ = qml.device("default.qubit").preprocess(execution_config=execution_config)
        res_tapes, batch_fn = program(tapes)

        expected_ops = [
            [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(res_tapes) == 4
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 2]):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 2:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        val = ([[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_check_validity_fail(self):
        """Test that preprocess throws an error if the split and expanded tapes have
        unsupported operators."""
        ops = [qml.Hadamard(0), NoMatNoDecompOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]
        tapes = [
            qml.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qml.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program, _ = qml.device("default.qubit").preprocess()
        with pytest.raises(qml.DeviceError, match="Operator NoMatNoDecompOp"):
            program(tapes)

    @pytest.mark.parametrize(
        "ops, measurement, message",
        [
            (
                [qml.RX(0.1, wires=0)],
                [qml.probs(wires=[0, 1, 2])],
                "not accepted for analytic simulation on adjoint",
            ),
            (
                [qml.RX(0.1, wires=0)],
                [qml.expval(qml.Hamiltonian([1], [qml.PauliZ(0)]))],
                "not supported on adjoint",
            ),
        ],
    )
    @pytest.mark.filterwarnings("ignore:Differentiating with respect to")
    def test_preprocess_invalid_tape_adjoint(self, ops, measurement, message):
        """Test that preprocessing fails if adjoint differentiation is requested and an
        invalid tape is used"""
        qs = qml.tape.QuantumScript(ops, measurement)
        execution_config = qml.devices.ExecutionConfig(gradient_method="adjoint")

        program, _ = qml.device("default.qubit").preprocess(execution_config)
        with pytest.raises(qml.DeviceError, match=message):
            program([qs])

    def test_preprocess_tape_for_adjoint(self):
        """Test that a tape is expanded correctly if adjoint differentiation is requested"""
        qs = qml.tape.QuantumScript(
            [qml.Rot(0.1, 0.2, 0.3, wires=0), qml.CNOT([0, 1])],
            [qml.expval(qml.PauliZ(1))],
        )
        execution_config = qml.devices.ExecutionConfig(gradient_method="adjoint")

        program, _ = qml.device("default.qubit").preprocess(execution_config)
        expanded_tapes, _ = program([qs])

        assert len(expanded_tapes) == 1
        expanded_qs = expanded_tapes[0]

        expected_qs = qml.tape.QuantumScript(
            [qml.RZ(0.1, wires=0), qml.RY(0.2, wires=0), qml.RZ(0.3, wires=0), qml.CNOT([0, 1])],
            [qml.expval(qml.PauliZ(1))],
        )

        assert expanded_qs.operations == expected_qs.operations
        assert expanded_qs.measurements == expected_qs.measurements
        assert expanded_qs.trainable_params == expected_qs.trainable_params

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_preprocess_single_circuit(self, max_workers):
        """Test integration between preprocessing and execution with numpy parameters."""

        # pylint: disable=too-few-public-methods
        class MyTemplate(qml.operation.Operation):
            """Temp operator."""

            num_wires = 2

            # pylint: disable=missing-function-docstring
            def decomposition(self):
                return [
                    qml.RX(self.data[0], self.wires[0]),
                    qml.RY(self.data[1], self.wires[1]),
                    qml.CNOT(self.wires),
                ]

        x = 0.928
        y = -0.792
        qscript = qml.tape.QuantumScript(
            [MyTemplate(x, y, ("a", "b"))],
            [qml.expval(qml.PauliY("a")), qml.expval(qml.PauliZ("a")), qml.expval(qml.PauliX("b"))],
        )

        dev = DefaultQubit(max_workers=max_workers)
        tapes = tuple([qscript])
        program, config = dev.preprocess()
        tapes, pre_processing_fn = program(tapes)

        assert len(tapes) == 1
        execute_circuit = tapes[0]
        assert qml.equal(execute_circuit[0], qml.RX(x, "a"))
        assert qml.equal(execute_circuit[1], qml.RY(y, "b"))
        assert qml.equal(execute_circuit[2], qml.CNOT(("a", "b")))
        assert qml.equal(execute_circuit[3], qml.expval(qml.PauliY("a")))
        assert qml.equal(execute_circuit[4], qml.expval(qml.PauliZ("a")))
        assert qml.equal(execute_circuit[5], qml.expval(qml.PauliX("b")))

        results = dev.execute(tapes, config)
        assert len(results) == 1
        assert len(results[0]) == 3

        processed_results = pre_processing_fn(results)
        processed_result = processed_results[0]
        assert len(processed_result) == 3
        assert qml.math.allclose(processed_result[0], -np.sin(x) * np.sin(y))
        assert qml.math.allclose(processed_result[1], np.cos(x))
        assert qml.math.allclose(processed_result[2], np.sin(y))

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    def test_preprocess_batch_circuit(self, max_workers):
        """Test preprocess integrates with default qubit when we start with a batch of circuits."""

        # pylint: disable=too-few-public-methods
        class CustomIsingXX(qml.operation.Operation):
            """Temp operator."""

            num_wires = 2

            # pylint: disable=missing-function-docstring
            def decomposition(self):
                return [qml.IsingXX(self.data[0], self.wires)]

        x = 0.692

        measurements1 = [qml.density_matrix("a"), qml.vn_entropy("a")]
        qs1 = qml.tape.QuantumScript([CustomIsingXX(x, ("a", "b"))], measurements1)

        y = -0.923

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliX(wires=1)
            m_0 = qml.measure(1)
            qml.cond(m_0, qml.RY)(y, wires=0)
            qml.expval(qml.PauliZ(0))

        qs2 = qml.tape.QuantumScript.from_queue(q)

        initial_batch = [qs1, qs2]

        dev = DefaultQubit(max_workers=max_workers)

        program, config = dev.preprocess()
        batch, pre_processing_fn = program(initial_batch)

        results = dev.execute(batch, config)
        processed_results = pre_processing_fn(results)

        assert len(processed_results) == 2
        assert len(processed_results[0]) == 2

        expected_density_mat = np.array([[np.cos(x / 2) ** 2, 0], [0, np.sin(x / 2) ** 2]])
        assert qml.math.allclose(processed_results[0][0], expected_density_mat)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = -np.sum(eigs * np.log(eigs))
        assert qml.math.allclose(processed_results[0][1], expected_entropy)

        expected_expval = np.cos(y)
        assert qml.math.allclose(expected_expval, processed_results[1])

    def test_preprocess_defer_measurements(self):
        """Test preprocessing contains the defer measurement transform."""
        dev = DefaultQubit()

        program, _ = dev.preprocess()
        assert qml.defer_measurements.transform in [t.transform for t in program]


class TestAdjointDiffTapeValidation:
    """Unit tests for validate_and_expand_adjoint"""

    @pytest.mark.parametrize("diff_method", ["adjoint", "backprop"])
    def test_finite_shots_analytic_diff_method(self, diff_method):
        """Test that a circuit with finite shots executed with diff_method "adjoint"
        or "backprop" raises an error"""
        tape = qml.tape.QuantumScript([], [qml.expval(qml.PauliZ(0))], shots=100)

        execution_config = ExecutionConfig(gradient_method=diff_method)
        program, _ = qml.device("default.qubit").preprocess(execution_config)

        msg = "Finite shots are not supported with"
        with pytest.raises(qml.DeviceError, match=msg):
            program((tape,))

    def test_not_expval(self):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        measurements = [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(3))]
        qs = qml.tape.QuantumScript(ops=[], measurements=measurements)

        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]

        with pytest.raises(
            qml.DeviceError,
            match=r"not accepted for analytic simulation on adjoint",
        ):
            program((qs,))

    def test_unsupported_op_decomposed(self):
        """Test that an operation supported on the forward pass but not adjoint is decomposed when adjoint is requested."""

        qs = qml.tape.QuantumScript([qml.U2(0.1, 0.2, wires=[0])], [qml.expval(qml.PauliZ(2))])
        batch = (qs,)
        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]
        res, _ = program(batch)
        res = res[0]
        assert isinstance(res, qml.tape.QuantumScript)
        assert qml.equal(res[0], qml.RZ(0.2, wires=0))
        assert qml.equal(res[1], qml.RY(np.pi / 2, wires=0))
        assert qml.equal(res[2], qml.RZ(-0.2, wires=0))
        assert qml.equal(res[3], qml.PhaseShift(0.2, wires=0))
        assert qml.equal(res[4], qml.PhaseShift(0.1, wires=0))

    def test_trainable_params_decomposed(self):
        """Test that the trainable parameters of a tape are updated when it is expanded"""
        ops = [
            qml.QubitUnitary([[0, 1], [1, 0]], wires=0),
            qml.CNOT([0, 1]),
            qml.Rot(0.1, 0.2, 0.3, wires=0),
        ]
        qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))])

        qs.trainable_params = [0]
        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]
        res, _ = program((qs,))
        res = res[0]

        assert isinstance(res, qml.tape.QuantumScript)
        assert len(res.operations) == 7
        assert qml.equal(res[0], qml.RZ(np.pi / 2, 0))
        assert qml.equal(res[1], qml.RY(np.pi, 0))
        assert qml.equal(res[2], qml.RZ(7 * np.pi / 2, 0))
        assert qml.equal(res[3], qml.CNOT([0, 1]))
        assert qml.equal(res[4], qml.RZ(0.1, 0))
        assert qml.equal(res[5], qml.RY(0.2, 0))
        assert qml.equal(res[6], qml.RZ(0.3, 0))
        assert res.trainable_params == [0, 1, 2, 3, 4, 5]

        qs.trainable_params = [2, 3]
        res, _ = program((qs,))
        res = res[0]
        assert isinstance(res, qml.tape.QuantumScript)
        assert len(res.operations) == 7
        assert qml.equal(res[0], qml.RZ(np.pi / 2, 0))
        assert qml.equal(res[1], qml.RY(np.pi, 0))
        assert qml.equal(res[2], qml.RZ(7 * np.pi / 2, 0))
        assert qml.equal(res[3], qml.CNOT([0, 1]))
        assert qml.equal(res[4], qml.RZ(0.1, 0))
        assert qml.equal(res[5], qml.RY(0.2, 0))
        assert qml.equal(res[6], qml.RZ(0.3, 0))
        assert res.trainable_params == [0, 1, 2, 3, 4, 5]

    def test_u3_non_trainable_params(self):
        """Test that a warning is raised and all parameters are trainable in the expanded
        tape when not all parameters in U3 are trainable"""
        qs = qml.tape.QuantumScript([qml.U3(0.2, 0.4, 0.6, wires=0)], [qml.expval(qml.PauliZ(0))])
        qs.trainable_params = [0, 2]

        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]
        res, _ = program((qs,))
        res = res[0]
        assert isinstance(res, qml.tape.QuantumScript)

        # U3 decomposes into 5 operators
        assert len(res.operations) == 5
        assert res.trainable_params == [0, 1, 2, 3, 4]

    def test_unsupported_obs(self):
        """Test that the correct error is raised if a Hamiltonian measurement is differentiated"""
        obs = qml.Hamiltonian([2, 0.5], [qml.PauliZ(0), qml.PauliY(1)])
        qs = qml.tape.QuantumScript([qml.RX(0.5, wires=1)], [qml.expval(obs)])
        qs.trainable_params = {0}

        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]

        with pytest.raises(qml.DeviceError, match=r"Observable "):
            program((qs,))

    def test_trainable_hermitian_warns(self):
        """Test attempting to compute the gradient of a tape that obtains the
        expectation value of a Hermitian operator emits a warning if the
        parameters to Hermitian are trainable."""

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        qs = qml.tape.QuantumScript([], [qml.expval(qml.Hermitian(mx, wires=[0, 2]))])

        qs.trainable_params = {0}

        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]

        with pytest.warns(
            UserWarning, match="Differentiating with respect to the input parameters of Hermitian"
        ):
            _ = program((qs,))

    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_valid_tape_no_expand(self, G):
        """Test that a tape that is valid doesn't raise errors and is not expanded"""
        prep_op = qml.StatePrep(
            qml.numpy.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )
        qs = qml.tape.QuantumScript(
            ops=[prep_op, G(np.pi, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
        )

        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]

        qs.trainable_params = {1}
        qs_valid, _ = program((qs,))
        qs_valid = qs_valid[0]
        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.operations, qs_valid.operations))
        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.measurements, qs_valid.measurements))
        assert qs_valid.trainable_params == [0, 1]

    def test_valid_tape_with_expansion(self):
        """Test that a tape that is valid with operations that need to be expanded doesn't raise errors
        and is expanded"""
        prep_op = qml.StatePrep(
            qml.numpy.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )
        qs = qml.tape.QuantumScript(
            ops=[prep_op, qml.Rot(0.1, 0.2, 0.3, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
        )

        program = qml.device("default.qubit").preprocess(
            ExecutionConfig(gradient_method="adjoint")
        )[0]

        qs.trainable_params = {1, 2, 3}
        qs_valid, _ = program((qs,))
        qs_valid = qs_valid[0]

        expected_ops = [
            prep_op,
            qml.RZ(0.1, wires=[0]),
            qml.RY(0.2, wires=[0]),
            qml.RZ(0.3, wires=[0]),
        ]

        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs_valid.operations, expected_ops))
        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.measurements, qs_valid.measurements))
        assert qs_valid.trainable_params == [0, 1, 2, 3]
        assert qs.shots == qs_valid.shots
