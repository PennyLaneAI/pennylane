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

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.devices.qubit.preprocess import (
    _accepted_operator,
    _operator_decomposition_gen,
    expand_fn,
    batch_transform,
    preprocess,
)
from pennylane.measurements import MidMeasureMP, MeasurementValue
from pennylane.tape import QuantumScript
from pennylane import DeviceError

# pylint: disable=too-few-public-methods


class NoMatOp(Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=missing-function-docstring
    num_wires = 1

    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qml.PauliX(self.wires), qml.PauliY(self.wires)]


class NoMatNoDecompOp(Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=missing-function-docstring
    num_wires = 1

    @property
    def has_matrix(self):
        return False


class TestPrivateHelpers:
    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.PauliX(0), True),
            (qml.CRX(0.1, wires=[0, 1]), True),
            (qml.Snapshot(), False),
            (qml.Barrier(), False),
            (qml.QFT(wires=range(5)), True),
            (qml.QFT(wires=range(10)), False),
            (qml.GroverOperator(wires=range(10)), True),
            (qml.GroverOperator(wires=range(14)), False),
        ],
    )
    def test_accepted_operator(self, op, expected):
        """Test that _accepted_operator works correctly"""
        res = _accepted_operator(op)
        assert res == expected

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.RX(1.2, wires=0), qml.QFT(wires=range(3))))
    def test_operator_decomposition_gen_accepted_operator(self, op):
        casted_to_list = list(_operator_decomposition_gen(op))
        assert len(casted_to_list) == 1
        assert casted_to_list[0] is op

    def test_operator_decomposition_gen_decomposed_operators_single_nesting(self):
        """Assert _operator_decomposition_gen turns into a list with the operators decomposition
        when only a single layer of expansion is necessary."""
        op = NoMatOp("a")
        casted_to_list = list(_operator_decomposition_gen(op))
        assert len(casted_to_list) == 2
        assert qml.equal(casted_to_list[0], qml.PauliX("a"))
        assert qml.equal(casted_to_list[1], qml.PauliY("a"))

    def test_operator_decomposition_gen_decomposed_operator_ragged_nesting(self):
        """Test that _operator_decomposition_gen handles a decomposition that requires different depths of decomposition."""

        class RaggedDecompositionOp(Operation):
            num_wires = 1

            def decomposition(self):
                return [NoMatOp(self.wires), qml.S(self.wires), qml.adjoint(NoMatOp(self.wires))]

        op = RaggedDecompositionOp("a")
        final_decomp = list(_operator_decomposition_gen(op))
        assert len(final_decomp) == 5
        assert qml.equal(final_decomp[0], qml.PauliX("a"))
        assert qml.equal(final_decomp[1], qml.PauliY("a"))
        assert qml.equal(final_decomp[2], qml.S("a"))
        assert qml.equal(final_decomp[3], qml.adjoint(qml.PauliY("a")))
        assert qml.equal(final_decomp[4], qml.adjoint(qml.PauliX("a")))

    def test_error_from_unsupported_operation(self):
        """Test that a device error is raised if the operator cant be decomposed and doesn't have a matrix."""
        op = NoMatNoDecompOp("a")
        with pytest.raises(DeviceError, match=r"Operator NoMatNoDecompOp"):
            tuple(_operator_decomposition_gen(op))


class TestExpandFnValidation:
    """Unit tests for helper functions in qml.devices.qubit.preprocess"""

    def test_error_if_invalid_op(self):
        """Test that expand_fn throws an error when an operation is does not define a matrix or decomposition."""
        tape = QuantumScript(ops=[NoMatNoDecompOp(0)], measurements=[qml.expval(qml.Hadamard(0))])
        with pytest.raises(DeviceError, match="Operator NoMatNoDecompOp"):
            expand_fn(tape)

    def test_expand_fn_invalid_observable(self):
        """Test that expand_fn throws an error when an observable is invalid."""
        tape = QuantumScript(
            ops=[qml.PauliX(0)], measurements=[qml.expval(qml.GellMann(wires=0, index=1))]
        )
        with pytest.raises(DeviceError, match=r"Observable GellMann1"):
            expand_fn(tape)

    def test_expand_fn_invalid_tensor_observable(self):
        """Test that expand_fn throws an error when a tensor includes invalid obserables"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.PauliY(1)],
            measurements=[
                qml.expval(qml.GellMann(wires=0, index=1) @ qml.GellMann(wires=1, index=2))
            ],
        )
        with pytest.raises(DeviceError, match="Observable expval"):
            expand_fn(tape)

    def test_expand_fn_valid_tensor_observable(self):
        """Test that expand_fn passes when a tensor includes only valid obserables."""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.PauliY(1)],
            measurements=[qml.expval(qml.PauliZ(0) @ qml.PauliX(1))],
        )
        expand_fn(tape)

    def test_state_prep_only_one(self):
        """Test that a device error is raised if the script has multiple state prep ops."""
        qs = QuantumScript(prep=[qml.BasisState([0], wires=0), qml.BasisState([1], wires=1)])
        with pytest.raises(
            DeviceError, match=r"DefaultQubit2 accepts at most one state prep operation."
        ):
            expand_fn(qs)

    def test_only_state_based_measurements(self):
        """Test that a device error is raised if a measurement is not a state measurement."""
        qs = QuantumScript([], [qml.expval(qml.PauliZ(0)), qml.sample()])
        with pytest.raises(DeviceError, match=r"Measurement process sample"):
            expand_fn(qs)

    def test_expand_fn_passes(self):
        """Test that expand_fn doesn't throw any errors for a valid circuit"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.RZ(0.123, wires=0)], measurements=[qml.state()]
        )
        expand_fn(tape)

    def test_infinite_decomposition_loop(self):
        """Test that a device error is raised if decomposition enters an infinite loop."""

        class InfiniteOp(qml.operation.Operation):
            num_wires = 1

            def decomposition(self):
                return [InfiniteOp(self.wires)]

        qs = qml.tape.QuantumScript([InfiniteOp(0)])
        with pytest.raises(DeviceError, match=r"Reached recursion limit trying to decompose"):
            expand_fn(qs)


class TestExpandFnTransformations:
    def test_expand_fn_expand_unsupported_op(self):
        """Test that expand_fn expands the tape when unsupported operators are present"""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tape = expand_fn(tape)
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        for op, exp in zip(expanded_tape.circuit, expected + measurements):
            assert qml.equal(op, exp)

    def test_expand_fn_defer_measurement(self):
        """Test that expand_fn defers mid-circuit measurements."""
        mv = MeasurementValue(["test_id"], processing_fn=lambda v: v)
        ops = [
            qml.Hadamard(0),
            MidMeasureMP(wires=[0], id="test_id"),
            qml.transforms.Conditional(mv, qml.RX(0.123, wires=1)),
        ]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tape = expand_fn(tape)
        expected = [qml.Hadamard(0), qml.ops.Controlled(qml.RX(0.123, wires=1), 0)]

        for op, exp in zip(expanded_tape, expected + measurements):
            assert qml.equal(op, exp)

    def test_expand_fn_no_expansion(self):
        """Test that expand_fn does nothing to a fully supported quantum script."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)
        expanded_tape = expand_fn(tape)

        for op, exp in zip(expanded_tape.circuit, ops + measurements):
            assert qml.equal(op, exp)

    def test_expand_fn_non_commuting_measurements(self):
        """Test that expand function can decompose operations even when non commuting measurements exist in the circuit."""

        qs = QuantumScript([NoMatOp("a")], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0))])
        new_qs = expand_fn(qs)
        assert new_qs.measurements == qs.measurements


class TestBatchTransform:
    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        tapes, batch_fn = batch_transform(tape)

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            assert qml.equal(op, expected)

        input = (("a", "b"), "c")
        assert batch_fn(input) == input[0]

    def test_batch_transform_broadcast(self):
        """Test that batch_transform splits broadcasted tapes correctly."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        tapes, batch_fn = batch_transform(tape)
        expected_ops = [
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi, wires=1)],
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX(np.pi / 2, wires=1)],
        ]

        assert len(tapes) == 2
        for i, t in enumerate(tapes):
            for op, expected in zip(t.circuit, expected_ops[i] + measurements):
                assert qml.equal(op, expected)

        input = ([[1, 2]], [[3, 4]])
        assert np.array_equal(batch_fn(input), np.array([[1, 2], [3, 4]]))


class TestPreprocess:
    """Unit tests for ``qml.devices.qubit.preprocess``."""

    def test_preprocess_finite_shots_error(self):
        """Test that preprocess throws an error if finite shots are present."""
        config = qml.devices.ExecutionConfig(shots=100)
        tape = QuantumScript(ops=[], measurements=[])
        with pytest.raises(
            qml.DeviceError, match="The Python Device does not support finite shots."
        ):
            _ = preprocess([tape], execution_config=config)

    def test_preprocess_batch_transform(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        res_tapes, batch_fn = preprocess(tapes)
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

        input = ([[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]])
        assert np.array_equal(batch_fn(input), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_expand(self):
        """Test that preprocess returns the correct tapes when expansion is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]
        tapes = [
            QuantumScript(ops=ops, measurements=measurements[0]),
            QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        res_tapes, batch_fn = preprocess(tapes)
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, exp in zip(t.circuit, expected + measurements[i]):
                assert qml.equal(op, exp)

        input = (("a", "b"), "c", "d")
        assert batch_fn(input) == [("a", "b"), "c"]

    def test_preprocess_split_and_expand(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        res_tapes, batch_fn = preprocess(tapes)
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

        input = ([[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]])
        assert np.array_equal(batch_fn(input), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_check_validity_fail(self):
        """Test that preprocess throws an error if the split and expanded tapes have
        unsupported operators."""
        ops = [qml.Hadamard(0), NoMatNoDecompOp(1), qml.RZ(0.123, wires=1)]
        measurements = [[qml.expval(qml.PauliZ(0))], [qml.expval(qml.PauliX(1))]]
        tapes = [
            QuantumScript(ops=ops, measurements=measurements[0]),
            QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        with pytest.raises(qml.DeviceError, match="Operator NoMatNoDecompOp"):
            _ = preprocess(tapes)
