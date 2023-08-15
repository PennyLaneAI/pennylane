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
    _accepted_adjoint_operator,
    _operator_decomposition_gen,
    expand_fn,
    batch_transform,
    preprocess,
    validate_and_expand_adjoint,
    validate_measurements,
    validate_multiprocessing_workers,
)
from pennylane.devices.experimental import ExecutionConfig
from pennylane.measurements import MidMeasureMP, MeasurementValue
from pennylane.tape import QuantumScript
from pennylane import DeviceError

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


class TestPrivateHelpers:
    """Test the private helpers for preprocessing."""

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
        ],
    )
    def test_accepted_operator(self, op, expected):
        """Test that _accepted_operator works correctly"""
        res = _accepted_operator(op)
        assert res == expected

    def test_adjoint_accepted_operator_only_one_wire(self):
        """Tests adjoint accepts operators with no parameters or a sinlge parameter and a generator."""

        assert _accepted_adjoint_operator(NoMatOp(wires=0))
        assert not _accepted_adjoint_operator(NoMatOp(1.2, wires=0))
        assert not _accepted_adjoint_operator(NoMatOp(1.2, 2.3, wires=0))

        class CustomOpWithGenerator(qml.operation.Operator):
            """A custom operator with a generator."""

            def generator(self):
                return qml.PauliX(0)

        assert _accepted_adjoint_operator(CustomOpWithGenerator(1.2, wires=0))

    @pytest.mark.parametrize("op", (qml.PauliX(0), qml.RX(1.2, wires=0), qml.QFT(wires=range(3))))
    def test_operator_decomposition_gen_accepted_operator(self, op):
        """Test the _operator_decomposition_gen function on an operator that is accepted."""
        casted_to_list = list(_operator_decomposition_gen(op, _accepted_operator))
        assert len(casted_to_list) == 1
        assert casted_to_list[0] is op

    def test_operator_decomposition_gen_decomposed_operators_single_nesting(self):
        """Assert _operator_decomposition_gen turns into a list with the operators decomposition
        when only a single layer of expansion is necessary."""
        op = NoMatOp("a")
        casted_to_list = list(_operator_decomposition_gen(op, _accepted_operator))
        assert len(casted_to_list) == 2
        assert qml.equal(casted_to_list[0], qml.PauliX("a"))
        assert qml.equal(casted_to_list[1], qml.PauliY("a"))

    def test_operator_decomposition_gen_decomposed_operator_ragged_nesting(self):
        """Test that _operator_decomposition_gen handles a decomposition that requires different depths of decomposition."""

        class RaggedDecompositionOp(Operation):
            """class with a ragged decomposition."""

            num_wires = 1

            def decomposition(self):
                return [NoMatOp(self.wires), qml.S(self.wires), qml.adjoint(NoMatOp(self.wires))]

        op = RaggedDecompositionOp("a")
        final_decomp = list(_operator_decomposition_gen(op, _accepted_operator))
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
            tuple(_operator_decomposition_gen(op, _accepted_operator))


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
            measurements=[qml.expval(qml.PauliX(0) @ qml.GellMann(wires=1, index=2))],
        )
        with pytest.raises(DeviceError, match="Observable expval"):
            expand_fn(tape)

    def test_valid_tensor_observable(self):
        """Test that a valid tensor ovservable passes without error."""
        tape = QuantumScript([], [qml.expval(qml.PauliZ(0) @ qml.PauliY(1))])
        assert expand_fn(tape) is tape

    def test_expand_fn_passes(self):
        """Test that expand_fn doesn't throw any errors for a valid circuit"""
        tape = QuantumScript(
            ops=[qml.PauliX(0), qml.RZ(0.123, wires=0)], measurements=[qml.state()]
        )
        expand_fn(tape)

    def test_infinite_decomposition_loop(self):
        """Test that a device error is raised if decomposition enters an infinite loop."""

        class InfiniteOp(qml.operation.Operation):
            """An op with an infinite decomposition."""

            num_wires = 1

            def decomposition(self):
                return [InfiniteOp(*self.parameters, self.wires)]

        qs = qml.tape.QuantumScript([InfiniteOp(1.23, 0)])
        with pytest.raises(DeviceError, match=r"Reached recursion limit trying to decompose"):
            expand_fn(qs)

        with pytest.raises(DeviceError, match=r"Reached recursion limit trying to decompose"):
            validate_and_expand_adjoint(qs)


class TestExpandFnTransformations:
    """Tests for the behavior of the `expand_fn` helper."""

    @pytest.mark.parametrize("shots", [None, 100])
    def test_expand_fn_expand_unsupported_op(self, shots):
        """Test that expand_fn expands the tape when unsupported operators are present"""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RZ(0.123, wires=1)]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements, shots=shots)

        expanded_tape = expand_fn(tape)
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        for op, exp in zip(expanded_tape.circuit, expected + measurements):
            assert qml.equal(op, exp)

        assert tape.shots == expanded_tape.shots

    # pylint: disable=no-member
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
        expected = [
            qml.Hadamard(0),
            qml.ops.Controlled(qml.RX(0.123, wires=1), 0),
        ]

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

    @pytest.mark.parametrize(
        "prep_op", (qml.BasisState([1], wires=0), qml.StatePrep([0, 1], wires=1))
    )
    def test_expand_fn_state_prep(self, prep_op):
        """Test that the expand_fn only expands mid-circuit instances of StatePrepBase"""
        ops = [
            prep_op,
            qml.Hadamard(wires=0),
            qml.StatePrep([0, 1], wires=1),
            qml.BasisState([1], wires=0),
            qml.RZ(0.123, wires=1),
        ]
        measurements = [qml.expval(qml.PauliZ(0)), qml.probs()]
        tape = QuantumScript(ops=ops, measurements=measurements)

        expanded_tape = expand_fn(tape)
        expected = [
            prep_op,
            qml.Hadamard(0),
            qml.RY(3.14159265, wires=1),  # decomposition of StatePrep
            qml.PauliX(wires=0),  # decomposition of BasisState
            qml.RZ(0.123, wires=1),
        ]

        for op, exp in zip(expanded_tape.circuit, expected + measurements):
            assert qml.equal(op, exp)


class TestValidateMeasurements:
    """Unit tests for the validate_measurements function"""

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
        validate_measurements(tape)

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
        validate_measurements(tape)

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

        msg = "Analytic circuits must only contain StateMeasurements"
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

        msg = "Circuits with finite shots must only contain SampleMeasurements"
        with pytest.raises(DeviceError, match=msg):
            validate_measurements(tape)

    @pytest.mark.parametrize("diff_method", ["adjoint", "backprop"])
    def test_finite_shots_analytic_diff_method(self, diff_method):
        """Test that a circuit with finite shots executed with diff_method "adjoint"
        or "backprop" raises an error"""
        tape = QuantumScript([], [qml.expval(qml.PauliZ(0))], shots=100)

        execution_config = ExecutionConfig()
        execution_config.gradient_method = diff_method

        msg = "Circuits with finite shots must be executed with non-analytic gradient methods"
        with pytest.raises(DeviceError, match=msg):
            validate_measurements(tape, execution_config)


class TestBatchTransform:
    """Tests for the batch transformations."""

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

    def test_batch_transform_broadcast_not_adjoint(self):
        """Test that batch_transform does nothing when batching is required but
        internal PennyLane broadcasting can be used (diff method != adjoint)"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        tapes, batch_fn = batch_transform(tape)

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            assert qml.equal(op, expected)

        input = ([[1, 2], [3, 4]],)
        assert np.array_equal(batch_fn(input), np.array([[1, 2], [3, 4]]))

    def test_batch_transform_broadcast_adjoint(self):
        """Test that batch_transform splits broadcasted tapes correctly when
        the diff method is adjoint"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qml.expval(qml.PauliZ(1))]
        tape = QuantumScript(ops=ops, measurements=measurements)

        execution_config = ExecutionConfig()
        execution_config.gradient_method = "adjoint"

        tapes, batch_fn = batch_transform(tape, execution_config=execution_config)
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

    @staticmethod
    def assert_validate_fails_with(qs, err):
        """Check that an specific error is raised."""
        res = validate_and_expand_adjoint(qs)
        assert isinstance(res, DeviceError)
        assert res.args == (err,)

    def test_not_expval(self):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        measurements = [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(3)), qml.sample()]
        qs = QuantumScript(ops=[], measurements=measurements)
        self.assert_validate_fails_with(
            qs, "Adjoint differentiation method does not support measurement VarianceMP."
        )

    def test_unsupported_op_decomposed(self):
        """Test that an operation supported on the forward pass but not adjoint is decomposed when adjoint is requested."""

        qs = QuantumScript([qml.U2(0.1, 0.2, wires=[0])], [qml.expval(qml.PauliZ(2))])
        res = validate_and_expand_adjoint(qs)
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
        qs = QuantumScript(ops, [qml.expval(qml.PauliZ(0))])

        qs.trainable_params = [0]
        res = validate_and_expand_adjoint(qs)
        assert isinstance(res, QuantumScript)
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
        res = validate_and_expand_adjoint(qs)
        assert isinstance(res, QuantumScript)
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
        qs = QuantumScript([qml.U3(0.2, 0.4, 0.6, wires=0)], [qml.expval(qml.PauliZ(0))])
        qs.trainable_params = [0, 2]

        res = validate_and_expand_adjoint(qs)
        assert isinstance(res, QuantumScript)

        # U3 decomposes into 5 operators
        assert len(res.operations) == 5
        assert res.trainable_params == [0, 1, 2, 3, 4]

    def test_unsupported_obs(self):
        """Test that the correct error is raised if a Hamiltonian or Sum measurement is differentiated"""
        obs = qml.Hamiltonian([2, 0.5], [qml.PauliZ(0), qml.PauliY(1)])
        qs = QuantumScript([qml.RX(0.5, wires=1)], [qml.expval(obs)])
        qs.trainable_params = {0}
        self.assert_validate_fails_with(
            qs, "Adjoint differentiation method does not support observable Hamiltonian."
        )

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
        prep_op = qml.StatePrep(pnp.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
        qs = QuantumScript(
            ops=[G(np.pi, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
            prep=[prep_op],
        )

        qs.trainable_params = {1}
        qs_valid = validate_and_expand_adjoint(qs)

        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.operations, qs_valid.operations))
        assert all(qml.equal(o1, o2) for o1, o2 in zip(qs.measurements, qs_valid.measurements))
        assert qs_valid.trainable_params == [0, 1]

    @pytest.mark.parametrize("shots", [None, 100])
    def test_valid_tape_with_expansion(self, shots):
        """Test that a tape that is valid with operations that need to be expanded doesn't raise errors
        and is expanded"""
        prep_op = qml.StatePrep(pnp.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0)
        qs = QuantumScript(
            ops=[qml.Rot(0.1, 0.2, 0.3, wires=[0])],
            measurements=[qml.expval(qml.PauliZ(0))],
            prep=[prep_op],
            shots=shots,
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
        assert qs_valid.trainable_params == [0, 1, 2, 3]
        assert qs.shots == qs_valid.shots


class TestPreprocess:
    """Unit tests for ``qml.devices.qubit.preprocess``."""

    def test_choose_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        tape = QuantumScript(ops=[], measurements=[])
        config = qml.devices.ExecutionConfig(gradient_method="best")
        _, _, config = preprocess([tape], config)
        assert config.gradient_method == "backprop"
        assert config.use_device_gradient
        assert not config.grad_on_execution

    def test_config_choices_for_adjoint(self):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""

        tape = QuantumScript(ops=[], measurements=[])
        config = qml.devices.ExecutionConfig(
            gradient_method="adjoint", use_device_gradient=None, grad_on_execution=None
        )
        _, _, new_config = preprocess([tape], config)

        assert new_config.use_device_gradient
        assert new_config.grad_on_execution

    def test_preprocess_batch_transform_not_adjoint(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        res_tapes, batch_fn, _ = preprocess(tapes)

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, ops):
                assert qml.equal(op, expected_op)
            assert len(t.measurements) == 1
            if i == 0:
                assert qml.equal(t.measurements[0], measurements[0])
            else:
                assert qml.equal(t.measurements[0], measurements[1])

        input = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(input), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_batch_transform_adjoint(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1]), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        execution_config = ExecutionConfig()
        execution_config.gradient_method = "adjoint"

        res_tapes, batch_fn, _ = preprocess(tapes, execution_config=execution_config)
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

        res_tapes, batch_fn, _ = preprocess(tapes)
        expected = [qml.Hadamard(0), qml.PauliX(1), qml.PauliY(1), qml.RZ(0.123, wires=1)]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, exp in zip(t.circuit, expected + measurements[i]):
                assert qml.equal(op, exp)

        input = (("a", "b"), "c", "d")
        assert batch_fn(input) == [("a", "b"), "c"]

    def test_preprocess_split_and_expand_not_adjoint(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        res_tapes, batch_fn, _ = preprocess(tapes)
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

        input = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(input), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    def test_preprocess_split_and_expand_adjoint(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qml.Hadamard(0), NoMatOp(1), qml.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))]
        tapes = [
            QuantumScript(ops=ops, measurements=[measurements[0]]),
            QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        execution_config = ExecutionConfig()
        execution_config.gradient_method = "adjoint"

        res_tapes, batch_fn, _ = preprocess(tapes, execution_config=execution_config)
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
        ],
    )
    @pytest.mark.filterwarnings("ignore:Differentiating with respect to")
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
            [qml.Rot(0.1, 0.2, 0.3, wires=0), qml.CNOT([0, 1])],
            [qml.expval(qml.PauliZ(1))],
        )
        execution_config = qml.devices.experimental.ExecutionConfig(gradient_method="adjoint")

        expanded_tapes, _, _ = preprocess([qs], execution_config)

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


def test_validate_multiprocessing_workers_None():
    """Test that validation does not fail when max_workers is None"""
    validate_multiprocessing_workers(None)
