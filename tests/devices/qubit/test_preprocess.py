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
import pennylane.numpy as pnp
import pennylane as qml
from pennylane.operation import Operation
from pennylane.devices.qubit.preprocess import (
    _accepted_operator,
    _operator_decomposition_gen,
    expand_fn,
    batch_transform,
    preprocess,
    validate_and_expand_adjoint,
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


class TestAdjointDiffTapeValidation:
    """Unit tests for validate_and_expand_adjoint"""

    def test_not_expval(self):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        measurements = [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(3)), qml.sample()]
        qs = QuantumScript(ops=[], measurements=measurements)

        with pytest.raises(
            DeviceError,
            match="Adjoint differentiation method does not support measurement",
        ):
            validate_and_expand_adjoint(qs)

    def test_unsupported_op(self):
        """Test if a QuantumFunctionError is raised for an unsupported operation, i.e.,
        multi-parameter operations that are not qml.Rot"""

        qs = QuantumScript([qml.U2(0.1, 0.2, wires=[0])], [qml.expval(qml.PauliZ(2))])

        with pytest.raises(
            DeviceError,
            match='operation is not supported using the "adjoint" differentiation method',
        ):
            validate_and_expand_adjoint(qs)

    @pytest.mark.parametrize(
        "obs",
        [
            qml.Hamiltonian([2, 0.5], [qml.PauliZ(0), qml.PauliY(1)]),
        ],
    )
    def test_unsupported_obs(self, obs):
        """Test that the correct error is raised if a Hamiltonian or Sum measurement is differentiated"""
        qs = QuantumScript([qml.RX(0.5, wires=1)], [qml.expval(obs)])

        with pytest.raises(
            DeviceError,
            match="Adjoint differentiation method does not support observable",
        ):
            validate_and_expand_adjoint(qs)

    def test_trainable_hermitian_warns(self):
        """Test attempting to compute the gradient of a tape that obtains the
        expectation value of a Hermitian operator emits a warning if the
        parameters to Hermitian are trainable."""

        mx = qml.matrix(qml.PauliX(0) @ qml.PauliY(2))
        qs = QuantumScript([], [qml.expval(qml.Hermitian(mx, wires=[0, 2]))])

        qs.trainable_params = {0}

        with pytest.warns(
            UserWarning, match="Differentiating with respect to the input parameters of Hermitian"
        ):
            _ = validate_and_expand_adjoint(qs)

    @pytest.mark.parametrize("G", [qml.RX, qml.RY, qml.RZ])
    def test_valid_tape_no_expand(self, G):
        """Test that a tape that is valid doesn't raise errors and is not expanded"""
        prep_op = qml.QubitStateVector(
            pnp.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )
        qs = QuantumScript(
            ops=[G(np.pi, wires=[0])], measurements=[qml.expval(qml.PauliZ(0))], prep=[prep_op]
        )

        qs.trainable_params = {1}
        qs_valid = validate_and_expand_adjoint(qs)

        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.operations, qs_valid.operations))
        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.measurements, qs_valid.measurements))
        assert qs.trainable_params == qs_valid.trainable_params

    def test_valid_tape_with_expansion(self):
        """Test that a tape that is valid with operations that need to be expanded doesn't raise errors
        and is expanded"""
        prep_op = qml.QubitStateVector(
            pnp.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )
        qs = QuantumScript(
            ops=[qml.Rot(0.1, 0.2, 0.3, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
            prep=[prep_op],
        )

        qs.trainable_params = {1, 2, 3}
        qs_valid = validate_and_expand_adjoint(qs)

        expected_ops = [
            prep_op,
            qml.RZ(0.1, wires=[0]),
            qml.RY(0.2, wires=[0]),
            qml.RZ(0.3, wires=[0]),
        ]

        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs_valid.operations, expected_ops))
        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.measurements, qs_valid.measurements))
        assert qs.trainable_params == qs_valid.trainable_params


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

    @pytest.mark.parametrize(
        "ops, measurement, message",
        [
            (
                [qml.RX(0.1, wires=0)],
                [qml.probs(wires=[0, 1, 2])],
                "does not support measurement ProbabilityMP",
            ),
            (
                [qml.RX(0.1, wires=0)],
                [qml.expval(qml.Hamiltonian([1], [qml.PauliZ(0)]))],
                "does not support observable Hamiltonian",
            ),
            (
                [qml.U2(0.1, 0.2, wires=0)],
                [qml.expval(qml.PauliZ(0))],
                'operation is not supported using the "adjoint" differentiation method',
            ),
        ],
    )
    def test_preprocess_invalid_tape_adjoint(self, ops, measurement, message):
        """Test that preprocessing fails if adjoint differentiation is requested and an
        invalid tape is used"""
        qs = QuantumScript(ops, measurement)
        execution_config = qml.devices.experimental.ExecutionConfig(gradient_method="adjoint")

        with pytest.raises(DeviceError, match=message):
            _ = preprocess([qs], execution_config)

    def test_preprocess_tape_for_adjoint(self):
        """Test that a tape is expanded correctly if adjoint differentiation is requested"""
        qs = QuantumScript(
            [qml.Rot(0.1, 0.2, 0.3, wires=0), qml.CNOT([0, 1])], [qml.expval(qml.PauliZ(1))]
        )
        execution_config = qml.devices.experimental.ExecutionConfig(gradient_method="adjoint")

        expanded_tapes, _ = preprocess([qs], execution_config)

        assert len(expanded_tapes) == 1
        expanded_qs = expanded_tapes[0]

        expected_qs = QuantumScript(
            [qml.RZ(0.1, wires=0), qml.RY(0.2, wires=0), qml.RZ(0.3, wires=0), qml.CNOT([0, 1])],
            [qml.expval(qml.PauliZ(1))],
        )

        assert all(
            qml.equal(o1, o2) for o1, o2 in zip(expanded_qs.operations, expected_qs.operations)
        )
        assert all(
            qml.equal(o1, o2) for o1, o2 in zip(expanded_qs.measurements, expected_qs.measurements)
        )
        assert expanded_qs.trainable_params == expected_qs.trainable_params
