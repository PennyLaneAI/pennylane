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

import numpy as np
import pytest
import scipy as sp

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.devices import DefaultQubit, ExecutionConfig
from pennylane.devices.default_qubit import stopping_condition
from pennylane.devices.execution_config import MCMConfig
from pennylane.exceptions import DeviceError
from pennylane.operation import classproperty


class NoMatOp(qp.operation.Operation):
    """Dummy operation for expanding circuit."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False

    def decomposition(self):
        return [qp.PauliX(self.wires), qp.PauliY(self.wires)]


# pylint: disable=too-few-public-methods
class NoMatNoDecompOp(qp.operation.Operation):
    """Dummy operation for checking check_validity throws error when
    expected."""

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return False


# pylint: disable=too-few-public-methods
class HasDiagonalizingGatesOp(qp.operation.Operator):
    """Dummy observable that has diagonalizing gates."""

    # pylint: disable=arguments-renamed,invalid-overridden-method,no-self-argument
    @classproperty
    def has_diagonalizing_gates(cls):
        return True


# pylint: disable=too-few-public-methods
class CustomizedSparseOp(qp.operation.Operator):
    def __init__(self, wires):
        U = sp.sparse.eye(2 ** len(wires))
        super().__init__(U, wires)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self) -> bool:
        return False

    def compute_sparse_matrix(self, U):  # pylint:disable=unused-argument, arguments-differ
        return sp.sparse.eye(2 ** len(self.wires))


def test_snapshot_multiprocessing_execute():
    """DefaultQubit cannot execute tapes with Snapshot if `max_workers` is not `None`"""
    dev = qp.device("default.qubit", max_workers=2)

    tape = qp.tape.QuantumScript(
        [
            qp.Snapshot(),
            qp.Hadamard(wires=0),
            qp.Snapshot("very_important_state"),
            qp.CNOT(wires=[0, 1]),
            qp.Snapshot(),
        ],
        [qp.expval(qp.PauliX(0))],
    )
    with pytest.raises(RuntimeError, match="ProcessPoolExecutor cannot execute a QuantumScript"):
        program = dev.preprocess_transforms()
        program([tape])


class TestConfigSetup:
    """Tests involving setting up the execution config."""

    def test_error_if_device_option_not_available(self):
        """Test that an error is raised if a device option is requested but not a valid option."""
        config = qp.devices.ExecutionConfig(device_options={"bla": "val"})
        with pytest.raises(DeviceError, match="device option bla"):
            qp.device("default.qubit").preprocess(config)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_choose_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        config = qp.devices.ExecutionConfig(gradient_method="best")
        config = qp.device("default.qubit").setup_execution_config(config)
        assert config.gradient_method == "backprop"
        assert config.use_device_gradient
        assert not config.grad_on_execution

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_config_choices_for_adjoint(self):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        config = qp.devices.ExecutionConfig(
            gradient_method="adjoint", use_device_gradient=None, grad_on_execution=None
        )
        new_config = qp.device("default.qubit").setup_execution_config(config)

        assert new_config.use_device_gradient
        assert new_config.grad_on_execution

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_chose_adjoint_as_best_if_max_workers_on_device(self):
        """Test that adjoint is best if max_workers as present."""

        dev = qp.device("default.qubit", max_workers=2)
        config = qp.devices.ExecutionConfig(gradient_method="best")
        config = dev.setup_execution_config(config)
        assert config.gradient_method == "adjoint"
        assert config.use_device_gradient
        assert config.grad_on_execution
        assert config.use_device_jacobian_product

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_chose_adjoint_as_best_if_max_workers_on_config(self):
        """Test that adjoint is best if max_workers as present."""

        dev = qp.device("default.qubit")
        config = qp.devices.ExecutionConfig(
            gradient_method="best", device_options={"max_workers": 2}
        )
        config = dev.setup_execution_config(config)
        assert config.gradient_method == "adjoint"
        assert config.use_device_gradient
        assert config.grad_on_execution
        assert config.use_device_jacobian_product

    def test_integration_uses_adjoint_if_maxworkers(self):
        """Test that the workflow uses adjoint if the device has max_workers set."""

        dev = qp.device("default.qubit", max_workers=2)

        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliZ(0))

        with dev.tracker:
            qp.grad(circuit)(qp.numpy.array(0.1))

        assert dev.tracker.totals["execute_and_derivative_batches"] == 1

    @pytest.mark.jax
    @pytest.mark.parametrize("interface", ("jax", "jax-jit"))
    @pytest.mark.parametrize("use_key", (True, False))
    def test_convert_to_numpy_with_jax(self, interface, use_key):
        """Test that we will not convert to numpy when working with jax."""
        # separate test so we can easily update it once we solve the
        # compilation overhead issue
        # TODO: [sc-82874]
        import jax

        key = jax.random.PRNGKey(12354) if use_key else None

        dev = qp.device("default.qubit", seed=key)
        config = qp.devices.ExecutionConfig(
            gradient_method=qp.gradients.param_shift, interface=interface
        )
        processed = dev.setup_execution_config(config)
        assert processed.convert_to_numpy != use_key

    def test_convert_to_numpy_with_adjoint(self):
        """Test that we will convert to numpy with adjoint."""
        config = qp.devices.ExecutionConfig(gradient_method="adjoint", interface="jax-jit")
        dev = qp.device("default.qubit")
        processed = dev.setup_execution_config(config)
        assert processed.convert_to_numpy

    @pytest.mark.parametrize("interface", ("autograd", "torch"))
    def test_convert_to_numpy_non_jax(self, interface):
        """Test that other interfaces are still converted to numpy."""
        config = qp.devices.ExecutionConfig(gradient_method="adjoint", interface=interface)
        dev = qp.device("default.qubit")
        processed = dev.setup_execution_config(config)
        assert processed.convert_to_numpy

    def test_resolve_native_mcm_method(self):
        """Tests that mcm_method="device" resolves to tree-traversal"""
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="device"))
        dev = qp.device("default.qubit")
        processed = dev.setup_execution_config(config)
        assert processed.mcm_config.mcm_method == "tree-traversal"

    @pytest.mark.parametrize("shots, expected", [(None, "deferred"), (10, "one-shot")])
    def test_default_mcm_method_circuit(self, shots, expected):
        config = ExecutionConfig()
        dev = qp.device("default.qubit")
        processed = dev.setup_execution_config(config, circuit=qp.tape.QuantumScript(shots=shots))
        assert processed.mcm_config.mcm_method == expected

    def test_default_mcm_method_no_circuit(self):
        config = ExecutionConfig()
        dev = qp.device("default.qubit")
        processed = dev.setup_execution_config(config)
        assert processed.mcm_config.mcm_method == "deferred"

    def test_error_on_unsupported_mcm_method(self):
        """Test that an error is raised on unsupported mcm_method's."""

        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="single-branch-statistics"))

        dev = qp.device("default.qubit")

        with pytest.raises(DeviceError, match="not supported on default.qubit"):
            dev.setup_execution_config(config)

    @pytest.mark.parametrize("mcm_method", ["one-shot", "tree-traversal"])
    def test_error_on_unsupported_postselect_mode(self, mcm_method):
        """Tests that fill-shots is not supported on anything but deferred."""

        config = ExecutionConfig(
            mcm_config=MCMConfig(mcm_method=mcm_method, postselect_mode="fill-shots")
        )
        dev = qp.device("default.qubit")
        with pytest.raises(DeviceError, match="Using postselect_mode='fill-shots'"):
            dev.setup_execution_config(config)


# pylint: disable=too-few-public-methods
class TestPreprocessing:
    """Unit tests for the preprocessing method."""

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_chooses_best_gradient_method(self):
        """Test that preprocessing chooses backprop as the best gradient method."""
        dev = DefaultQubit()

        config = ExecutionConfig(
            gradient_method="best", use_device_gradient=None, grad_on_execution=None
        )

        new_config = dev.setup_execution_config(config)

        assert new_config.gradient_method == "backprop"
        assert new_config.use_device_gradient
        assert not new_config.grad_on_execution

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_config_choices_for_adjoint(self):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        dev = DefaultQubit()

        config = ExecutionConfig(
            gradient_method="adjoint", use_device_gradient=None, grad_on_execution=None
        )

        new_config = dev.setup_execution_config(config)

        assert new_config.use_device_gradient
        assert new_config.grad_on_execution

    @pytest.mark.parametrize("max_workers", [None, 1, 2, 3])
    def test_config_choices_for_threading(self, max_workers):
        """Test that preprocessing request grad on execution and says to use the device gradient if adjoint is requested."""
        dev = DefaultQubit()

        config = ExecutionConfig(device_options={"max_workers": max_workers})
        new_config = dev.setup_execution_config(config)

        assert new_config.device_options["max_workers"] == max_workers

    def test_circuit_wire_validation(self):
        """Test that preprocessing validates wires on the circuits being executed."""
        dev = DefaultQubit(wires=3)
        circuit_valid_0 = qp.tape.QuantumScript([qp.PauliX(0)])
        program = dev.preprocess_transforms()
        circuits, _ = program([circuit_valid_0])
        assert circuits[0].circuit == circuit_valid_0.circuit

        circuit_valid_1 = qp.tape.QuantumScript([qp.PauliX(1)])
        program = dev.preprocess_transforms()
        circuits, _ = program([circuit_valid_0, circuit_valid_1])
        assert circuits[0].circuit == circuit_valid_0.circuit
        assert circuits[1].circuit == circuit_valid_1.circuit

        invalid_circuit = qp.tape.QuantumScript([qp.PauliX(4)])
        with pytest.raises(qp.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program = dev.preprocess_transforms()
            program(
                [
                    invalid_circuit,
                ]
            )

        with pytest.raises(qp.wires.WireError, match=r"Cannot run circuit\(s\) on"):
            program = dev.preprocess_transforms()
            program([circuit_valid_0, invalid_circuit])

    @pytest.mark.parametrize(
        "mp_fn,mp_cls,shots",
        [
            (qp.sample, qp.measurements.SampleMP, 10),
            (qp.state, qp.measurements.StateMP, None),
            (qp.probs, qp.measurements.ProbabilityMP, None),
        ],
    )
    def test_measurement_is_swapped_out(self, mp_fn, mp_cls, shots):
        """Test that preprocessing swaps out any MP with no wires or obs"""
        dev = DefaultQubit(wires=3)
        original_mp = mp_fn()
        exp_z = qp.expval(qp.PauliZ(0))
        qs = qp.tape.QuantumScript([qp.Hadamard(0)], [original_mp, exp_z], shots=shots)
        program = dev.preprocess_transforms()
        tapes, _ = program([qs])
        assert len(tapes) == 1
        tape = tapes[0]
        assert tape.operations == qs.operations
        assert tape.measurements != qs.measurements
        qp.assert_equal(tape.measurements[0], mp_cls(wires=[0, 1, 2]))
        assert tape.measurements[1] is exp_z

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qp.PauliX(0), True),
            (qp.CRX(0.1, wires=[0, 1]), True),
            (qp.Snapshot(), True),
            (qp.Barrier(), False),
            (qp.QFT(wires=range(5)), True),
            (qp.QFT(wires=range(10)), False),
            (qp.GroverOperator(wires=range(10)), True),
            (qp.GroverOperator(wires=range(14)), False),
            (
                qp.IQP(
                    np.zeros(5, dtype=np.float64), num_wires=5, pattern=[[[i]] for i in range(5)]
                ),
                True,
            ),
            (
                qp.IQP(
                    np.zeros(6, dtype=np.float64), num_wires=6, pattern=[[[i]] for i in range(6)]
                ),
                False,
            ),
            (qp.pow(qp.RX(1.1, 0), 3), True),
            (qp.pow(qp.RX(qp.numpy.array(1.1), 0), 3), False),
            (qp.QubitUnitary(sp.sparse.csr_matrix(np.eye(8)), wires=range(3)), True),
            (qp.QubitUnitary(sp.sparse.eye(2), wires=0), True),
            (qp.adjoint(qp.QubitUnitary(sp.sparse.eye(2), wires=0)), True),
            (CustomizedSparseOp([0, 1, 2]), True),
        ],
    )
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_accepted_operator(self, op, expected):
        """Test that _accepted_operator works correctly"""
        res = stopping_condition(op)
        assert res == expected

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_adjoint_only_one_wire(self):
        """Tests adjoint accepts operators with no parameters or a single parameter and a generator."""

        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )

        class MatOp(qp.operation.Operation):
            """Dummy operation for expanding circuit."""

            # pylint: disable=arguments-renamed, invalid-overridden-method
            @property
            def has_matrix(self):
                return True

            def decomposition(self):
                return [qp.PauliX(self.wires), qp.PauliY(self.wires)]

        tape1 = qp.tape.QuantumScript([MatOp(wires=0)])
        batch, _ = program((tape1,))
        assert batch[0].circuit == tape1.circuit

        tape2 = qp.tape.QuantumScript([MatOp(qp.numpy.array(1.2), wires=0)])
        batch, _ = program((tape2,))
        assert batch[0].circuit != tape2.circuit

        tape3 = qp.tape.QuantumScript([MatOp(qp.numpy.array(1.2), qp.numpy.array(2.3), wires=0)])
        batch, _ = program((tape2,))
        assert batch[0].circuit != tape3.circuit

        class CustomOpWithGenerator(qp.operation.Operator):
            """A custom operator with a generator."""

            def generator(self):
                return qp.PauliX(0)

            # pylint: disable=arguments-renamed, invalid-overridden-method
            @property
            def has_matrix(self):
                return True

        tape4 = qp.tape.QuantumScript([CustomOpWithGenerator(qp.numpy.array(1.2), wires=0)])
        batch, _ = program((tape4,))
        assert batch[0].circuit == tape4.circuit

    @pytest.mark.parametrize(
        "shots, measurements, supported",
        [
            # Supported measurements in analytic mode
            (None, [qp.state()], True),
            (None, [qp.expval(qp.X(0))], True),
            (None, [qp.expval(qp.RX(0.123, 0))], False),
            (None, [qp.expval(qp.SparseHamiltonian(qp.X.compute_sparse_matrix(), 0))], True),
            (None, [qp.expval(qp.Hermitian(np.diag([1, 2]), wires=0))], True),
            (None, [qp.var(qp.SparseHamiltonian(qp.X.compute_sparse_matrix(), 0))], False),
            (None, [qp.expval(qp.X(0) @ qp.Hermitian(np.diag([1, 2]), wires=1))], True),
            (
                None,
                [
                    qp.expval(
                        qp.Hamiltonian(
                            [0.1, 0.2],
                            [qp.Z(0), qp.SparseHamiltonian(qp.X.compute_sparse_matrix(), 1)],
                        )
                    )
                ],
                True,
            ),
            (None, [qp.expval(qp.Hamiltonian([0.1, 0.2], [qp.RZ(0.234, 0), qp.X(0)]))], False),
            (
                None,
                [qp.expval(qp.Hamiltonian([1, 1], [qp.Z(0), HasDiagonalizingGatesOp(1)]))],
                True,
            ),
            # Supported measurements in finite shots mode
            (100, [qp.state()], False),
            (100, [qp.expval(qp.X(0))], True),
            (100, [qp.expval(qp.RX(0.123, 0))], False),
            (100, [qp.expval(qp.X(0) @ qp.RX(0.123, 1))], False),
            (100, [qp.expval(qp.X(0) @ qp.Y(1))], True),
            (100, [qp.expval(qp.Hamiltonian([0.1, 0.2], [qp.Z(0), qp.X(1)]))], True),
            (100, [qp.expval(qp.Hamiltonian([0.1, 0.2], [qp.RZ(0.123, 0), qp.X(1)]))], False),
        ],
    )
    def test_validate_measurements(self, shots, measurements, supported):
        """Tests that preprocess correctly validates measurements."""

        device = qp.device("default.qubit")
        tape = qp.tape.QuantumScript(measurements=measurements, shots=shots)
        program = device.preprocess_transforms()

        if not supported:
            with pytest.raises(DeviceError):
                program([tape])
        else:
            program([tape])


class TestPreprocessingIntegration:
    """Test preprocess produces output that can be executed by the device."""

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_batch_transform_no_batching(self):
        """Test that batch_transform does nothing when no batching is required."""
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX(0.123, wires=1)]
        measurements = [qp.expval(qp.PauliZ(1))]
        tape = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        device = qp.device("default.qubit")

        program = device.preprocess_transforms()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        for op, expected in zip(tapes[0].circuit, ops + measurements):
            qp.assert_equal(op, expected)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_batch_transform_broadcast_not_adjoint(self):
        """Test that batch_transform does nothing when batching is required but
        internal PennyLane broadcasting can be used (diff method != adjoint)"""
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qp.expval(qp.PauliZ(1))]
        tape = qp.tape.QuantumScript(ops=ops, measurements=measurements)
        device = qp.devices.DefaultQubit()

        program = device.preprocess_transforms()
        tapes, _ = program([tape])

        assert len(tapes) == 1
        assert tapes[0].circuit == ops + measurements

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_batch_transform_broadcast_adjoint(self):
        """Test that batch_transform splits broadcasted tapes correctly when
        the diff method is adjoint"""
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX([np.pi, np.pi / 2], wires=1)]
        measurements = [qp.expval(qp.PauliZ(1))]
        tape = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        execution_config = ExecutionConfig(gradient_method="adjoint")

        device = qp.devices.DefaultQubit()

        program = device.preprocess_transforms(execution_config=execution_config)
        tapes, _ = program([tape])
        expected_ops = [
            [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX(np.pi, wires=1)],
            [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX(np.pi / 2, wires=1)],
        ]

        assert len(tapes) == 2
        for i, t in enumerate(tapes):
            for op, expected in zip(t.circuit, expected_ops[i] + measurements):
                qp.assert_equal(op, expected)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_batch_transform_not_adjoint(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(1))]
        tapes = [
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program = qp.device("default.qubit").preprocess_transforms()
        res_tapes, batch_fn = program(tapes)

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, ops):
                qp.assert_equal(op, expected_op)
            assert len(t.measurements) == 1
            if i == 0:
                qp.assert_equal(t.measurements[0], measurements[0])
            else:
                qp.assert_equal(t.measurements[0], measurements[1])

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_batch_transform_adjoint(self):
        """Test that preprocess returns the correct tapes when a batch transform
        is needed."""
        ops = [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX([np.pi, np.pi / 2, 2.5], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(1))]
        tapes = [
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        execution_config = ExecutionConfig(gradient_method="adjoint")

        program = qp.device("default.qubit").preprocess_transforms(
            execution_config=execution_config
        )
        res_tapes, batch_fn = program(tapes)

        expected_ops = [
            [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX(np.pi, wires=1)],
            [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX(np.pi / 2, wires=1)],
            [qp.Hadamard(0), qp.CNOT([0, 1]), qp.RX(2.5, wires=1)],
        ]

        assert len(res_tapes) == 6
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 3]):
                qp.assert_equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 3:
                qp.assert_equal(t.measurements[0], measurements[0])
            else:
                qp.assert_equal(t.measurements[0], measurements[1])

        # outer dimension = tapes, each has one meausrement
        val = (1, 2, 3, 4, 5, 6)
        expected = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert np.array_equal(batch_fn(val), expected)

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_expand(self):
        """Test that preprocess returns the correct tapes when expansion is needed."""
        ops = [qp.Hadamard(0), NoMatOp(1), qp.RZ(0.123, wires=1)]
        measurements = [[qp.expval(qp.PauliZ(0))], [qp.expval(qp.PauliX(1))]]
        tapes = [
            qp.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qp.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program = qp.device("default.qubit").preprocess_transforms()
        res_tapes, batch_fn = program(tapes)

        expected = [qp.Hadamard(0), qp.PauliX(1), qp.PauliY(1), qp.RZ(0.123, wires=1)]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, exp in zip(t.circuit, expected + measurements[i]):
                qp.assert_equal(op, exp)

        val = (("a", "b"), "c", "d")
        assert batch_fn(val) == (("a", "b"), "c")

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_split_and_expand_not_adjoint(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qp.Hadamard(0), NoMatOp(1), qp.RX([np.pi, np.pi / 2], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(1))]
        tapes = [
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        program = qp.device("default.qubit").preprocess_transforms()
        res_tapes, batch_fn = program(tapes)
        expected_ops = [
            qp.Hadamard(0),
            qp.PauliX(1),
            qp.PauliY(1),
            qp.RX([np.pi, np.pi / 2], wires=1),
        ]

        assert len(res_tapes) == 2
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops):
                qp.assert_equal(op, expected_op)
            assert len(t.measurements) == 1
            if i == 0:
                qp.assert_equal(t.measurements[0], measurements[0])
            else:
                qp.assert_equal(t.measurements[0], measurements[1])

        val = ([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        assert np.array_equal(batch_fn(val), np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_split_and_expand_adjoint(self):
        """Test that preprocess returns the correct tapes when splitting and expanding
        is needed."""
        ops = [qp.Hadamard(0), NoMatOp(1), qp.RX([np.pi, np.pi / 2, 1.23], wires=1)]
        # Need to specify grouping type to transform tape
        measurements = [qp.expval(qp.PauliX(0)), qp.expval(qp.PauliZ(1))]
        tapes = [
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[0]]),
            qp.tape.QuantumScript(ops=ops, measurements=[measurements[1]]),
        ]

        execution_config = ExecutionConfig(gradient_method="adjoint")

        program = qp.device("default.qubit").preprocess_transforms(
            execution_config=execution_config
        )
        res_tapes, batch_fn = program(tapes)

        expected_ops = [
            [qp.Hadamard(0), qp.PauliX(1), qp.PauliY(1), qp.RX(np.pi, wires=1)],
            [qp.Hadamard(0), qp.PauliX(1), qp.PauliY(1), qp.RX(np.pi / 2, wires=1)],
            [qp.Hadamard(0), qp.PauliX(1), qp.PauliY(1), qp.RX(1.23, wires=1)],
        ]

        assert len(res_tapes) == 6
        for i, t in enumerate(res_tapes):
            for op, expected_op in zip(t.operations, expected_ops[i % 3]):
                qp.assert_equal(op, expected_op)
            assert len(t.measurements) == 1
            if i < 3:
                qp.assert_equal(t.measurements[0], measurements[0])
            else:
                qp.assert_equal(t.measurements[0], measurements[1])

        val = (1, 2, 3, 4, 5, 6)
        expected = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert np.array_equal(batch_fn(val), expected)

    def test_preprocess_check_validity_fail(self):
        """Test that preprocess throws an error if the split and expanded tapes have
        unsupported operators."""
        ops = [qp.Hadamard(0), NoMatNoDecompOp(1), qp.RZ(0.123, wires=1)]
        measurements = [[qp.expval(qp.PauliZ(0))], [qp.expval(qp.PauliX(1))]]
        tapes = [
            qp.tape.QuantumScript(ops=ops, measurements=measurements[0]),
            qp.tape.QuantumScript(ops=ops, measurements=measurements[1]),
        ]

        program = qp.device("default.qubit").preprocess_transforms()
        with pytest.raises(DeviceError, match="Operator NoMatNoDecompOp"):
            program(tapes)

    @pytest.mark.parametrize(
        "ops, measurement, message",
        [
            (
                [qp.RX(0.1, wires=0)],
                [qp.probs(op=qp.PauliX(0))],
                "adjoint diff supports either all expectation values or",
            )
        ],
    )
    @pytest.mark.filterwarnings("ignore:Differentiating with respect to")
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_invalid_tape_adjoint(self, ops, measurement, message):
        """Test that preprocessing fails if adjoint differentiation is requested and an
        invalid tape is used"""
        qs = qp.tape.QuantumScript(ops, measurement)
        execution_config = qp.devices.ExecutionConfig(gradient_method="adjoint")

        program = qp.device("default.qubit").preprocess_transforms(execution_config)
        with pytest.raises(DeviceError, match=message):
            program([qs])

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_tape_for_adjoint(self):
        """Test that a tape is expanded correctly if adjoint differentiation is requested"""
        qs = qp.tape.QuantumScript(
            [
                qp.Rot(qp.numpy.array(0.1), qp.numpy.array(0.2), qp.numpy.array(0.3), wires=0),
                qp.CNOT([0, 1]),
            ],
            [qp.expval(qp.PauliZ(1))],
        )
        execution_config = qp.devices.ExecutionConfig(gradient_method="adjoint")

        program = qp.device("default.qubit").preprocess_transforms(execution_config)
        expanded_tapes, _ = program([qs])

        assert len(expanded_tapes) == 1
        expanded_qs = expanded_tapes[0]

        expected_qs = qp.tape.QuantumScript(
            [
                qp.RZ(qp.numpy.array(0.1), wires=0),
                qp.RY(qp.numpy.array(0.2), wires=0),
                qp.RZ(qp.numpy.array(0.3), wires=0),
                qp.CNOT([0, 1]),
            ],
            [qp.expval(qp.PauliZ(1))],
        )

        assert expanded_qs.operations == expected_qs.operations
        assert expanded_qs.measurements == expected_qs.measurements
        assert expanded_qs.trainable_params == expected_qs.trainable_params

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_single_circuit(self, max_workers):
        """Test integration between preprocessing and execution with numpy parameters."""

        # pylint: disable=too-few-public-methods
        class MyTemplate(qp.operation.Operation):
            """Temp operator."""

            num_wires = 2

            # pylint: disable=missing-function-docstring
            def decomposition(self):
                return [
                    qp.RX(self.data[0], self.wires[0]),
                    qp.RY(self.data[1], self.wires[1]),
                    qp.CNOT(self.wires),
                ]

        x = 0.928
        y = -0.792
        qscript = qp.tape.QuantumScript(
            [MyTemplate(x, y, ("a", "b"))],
            [qp.expval(qp.PauliY("a")), qp.expval(qp.PauliZ("a")), qp.expval(qp.PauliX("b"))],
        )

        dev = DefaultQubit(max_workers=max_workers)
        tapes = tuple([qscript])
        program, config = dev.preprocess()
        tapes, pre_processing_fn = program(tapes)

        assert len(tapes) == 1
        execute_circuit = tapes[0]
        qp.assert_equal(execute_circuit[0], qp.RX(x, "a"))
        qp.assert_equal(execute_circuit[1], qp.RY(y, "b"))
        qp.assert_equal(execute_circuit[2], qp.CNOT(("a", "b")))
        qp.assert_equal(execute_circuit[3], qp.expval(qp.PauliY("a")))
        qp.assert_equal(execute_circuit[4], qp.expval(qp.PauliZ("a")))
        qp.assert_equal(execute_circuit[5], qp.expval(qp.PauliX("b")))

        results = dev.execute(tapes, config)
        assert len(results) == 1
        assert len(results[0]) == 3

        processed_results = pre_processing_fn(results)
        processed_result = processed_results[0]
        assert len(processed_result) == 3
        assert qp.math.allclose(processed_result[0], -np.sin(x) * np.sin(y))
        assert qp.math.allclose(processed_result[1], np.cos(x))
        assert qp.math.allclose(processed_result[2], np.sin(y))

    @pytest.mark.parametrize("max_workers", [None, 1, 2])
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_preprocess_batch_circuit(self, max_workers):
        """Test preprocess integrates with default qubit when we start with a batch of circuits."""

        # pylint: disable=too-few-public-methods
        class CustomIsingXX(qp.operation.Operation):
            """Temp operator."""

            num_wires = 2

            # pylint: disable=missing-function-docstring
            def decomposition(self):
                return [qp.IsingXX(self.data[0], self.wires)]

        x = 0.692

        measurements1 = [qp.density_matrix("a"), qp.vn_entropy("a")]
        qs1 = qp.tape.QuantumScript([CustomIsingXX(x, ("a", "b"))], measurements1)

        y = -0.923

        with qp.queuing.AnnotatedQueue() as q:
            qp.PauliX(wires=1)
            m_0 = qp.measure(1)
            qp.cond(m_0, qp.RY)(y, wires=0)
            qp.expval(qp.PauliZ(0))

        qs2 = qp.tape.QuantumScript.from_queue(q)

        initial_batch = [qs1, qs2]

        dev = DefaultQubit(max_workers=max_workers)

        program, config = dev.preprocess()
        batch, pre_processing_fn = program(initial_batch)

        results = dev.execute(batch, config)
        processed_results = pre_processing_fn(results)

        assert len(processed_results) == 2
        assert len(processed_results[0]) == 2

        expected_density_mat = np.array([[np.cos(x / 2) ** 2, 0], [0, np.sin(x / 2) ** 2]])
        assert qp.math.allclose(processed_results[0][0], expected_density_mat)

        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(x / 2) ** 2 * np.sin(x / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]

        expected_entropy = -np.sum(eigs * np.log(eigs))
        assert qp.math.allclose(processed_results[0][1], expected_entropy)

        expected_expval = np.cos(y)
        assert qp.math.allclose(expected_expval, processed_results[1])

    def test_decompose_conditionals(self):
        """Test that conditional templates are properly decomposed."""

        m0 = qp.measure(0)
        tape = qp.tape.QuantumScript(
            [m0.measurements[0], qp.ops.Conditional(m0, NoMatOp(wires=0))], [qp.probs(wires=0)]
        )
        config = ExecutionConfig(mcm_config=MCMConfig(mcm_method="deferred"))

        prog = qp.device("default.qubit").preprocess_transforms(config)
        [new_tape], _ = prog((tape,))

        expected = qp.tape.QuantumScript(
            [qp.CNOT((0, 1)), qp.CNOT((1, 0)), qp.CY((1, 0))], [qp.probs(wires=0)]
        )
        qp.assert_equal(new_tape, expected)

    def test_no_mcms_conditionals_defer_measurements(self):
        """Test that an error is raised if an mcm occurs in a decomposition after defer measurements has been applied."""

        m0 = qp.measure(0)

        class MyOp(qp.operation.Operator):
            def decomposition(self):
                return m0.measurements

        tape = qp.tape.QuantumScript([MyOp(0)])
        config = qp.devices.ExecutionConfig(
            mcm_config=qp.devices.MCMConfig(mcm_method="deferred")
        )

        prog = DefaultQubit().preprocess_transforms(config)

        with pytest.raises(DeviceError, match="not supported with default.qubit"):
            prog((tape,))


class TestAdjointDiffTapeValidation:
    """Unit tests for validate_and_expand_adjoint"""

    @pytest.mark.parametrize("diff_method", ["adjoint", "backprop"])
    def test_finite_shots_analytic_diff_method(self, diff_method):
        """Test that a circuit with finite shots executed with diff_method "adjoint"
        or "backprop" raises an error"""
        tape = qp.tape.QuantumScript([], [qp.expval(qp.PauliZ(0))], shots=100)

        execution_config = ExecutionConfig(gradient_method=diff_method)
        program = qp.device("default.qubit").preprocess_transforms(execution_config)

        msg = "Finite shots are not supported with"
        with pytest.raises(DeviceError, match=msg):
            program((tape,))

    def test_non_diagonal_non_expval(self):
        """Test if a QuantumFunctionError is raised for a tape with measurements that are not
        expectation values"""

        measurements = [qp.expval(qp.PauliZ(0)), qp.var(qp.PauliX(3))]
        qs = qp.tape.QuantumScript(ops=[], measurements=measurements)

        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )

        with pytest.raises(
            DeviceError,
            match=r"adjoint diff supports either all expectation values or only measurements without observables",
        ):
            program((qs,))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_unsupported_op_decomposed(self):
        """Test that an operation supported on the forward pass but
        not adjoint is decomposed when adjoint is requested."""

        qs = qp.tape.QuantumScript(
            [qp.U2(qp.numpy.array(0.1), qp.numpy.array(0.2), wires=[0])],
            [qp.expval(qp.PauliZ(2))],
        )
        batch = (qs,)
        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )
        res, _ = program(batch)
        res = res[0]
        assert isinstance(res, qp.tape.QuantumScript)
        qp.assert_equal(res[0], qp.RZ(qp.numpy.array(0.2), wires=0))
        qp.assert_equal(res[1], qp.RY(qp.numpy.array(np.pi / 2), wires=0))
        qp.assert_equal(res[2], qp.RZ(qp.numpy.array(-0.2), wires=0))
        qp.assert_equal(res[3], qp.PhaseShift(qp.numpy.array(0.2), wires=0))
        qp.assert_equal(res[4], qp.PhaseShift(qp.numpy.array(0.1), wires=0))

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_trainable_params_decomposed(self):
        """Test that the trainable parameters of a tape are updated when it is expanded"""

        ops = [
            qp.QubitUnitary(qp.numpy.array([[0, 1], [1, 0]]), wires=0),
            qp.CNOT([0, 1]),
            qp.Rot(qp.numpy.array(0.1), qp.numpy.array(0.2), qp.numpy.array(0.3), wires=0),
        ]
        qs = qp.tape.QuantumScript(ops, [qp.expval(qp.PauliZ(0))])

        qs.trainable_params = [0]
        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )
        res, _ = program((qs,))
        res = res[0]

        assert isinstance(res, qp.tape.QuantumScript)
        assert len(res.operations) == 8
        qp.assert_equal(res[0], qp.RZ(qp.numpy.array(np.pi / 2), 0))
        qp.assert_equal(res[1], qp.RY(qp.numpy.array(np.pi), 0))
        qp.assert_equal(res[2], qp.RZ(qp.numpy.array(7 * np.pi / 2), 0))
        qp.assert_equal(res[3], qp.GlobalPhase(-np.pi / 2))
        qp.assert_equal(res[4], qp.CNOT([0, 1]))
        qp.assert_equal(res[5], qp.RZ(qp.numpy.array(0.1), 0))
        qp.assert_equal(res[6], qp.RY(qp.numpy.array(0.2), 0))
        qp.assert_equal(res[7], qp.RZ(qp.numpy.array(0.3), 0))
        assert res.trainable_params == [0, 1, 2, 3, 4, 5, 6]

        qs.trainable_params = [2, 3]
        res, _ = program((qs,))
        res = res[0]
        assert isinstance(res, qp.tape.QuantumScript)
        assert len(res.operations) == 8
        qp.assert_equal(res[0], qp.RZ(qp.numpy.array(np.pi / 2), 0))
        qp.assert_equal(res[1], qp.RY(qp.numpy.array(np.pi), 0))
        qp.assert_equal(res[2], qp.RZ(qp.numpy.array(7 * np.pi / 2), 0))
        qp.assert_equal(res[3], qp.GlobalPhase(-np.pi / 2))
        qp.assert_equal(res[4], qp.CNOT([0, 1]))
        qp.assert_equal(res[5], qp.RZ(qp.numpy.array(0.1), 0))
        qp.assert_equal(res[6], qp.RY(qp.numpy.array(0.2), 0))
        qp.assert_equal(res[7], qp.RZ(qp.numpy.array(0.3), 0))
        assert res.trainable_params == [0, 1, 2, 3, 4, 5, 6]

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_u3_non_trainable_params(self):
        """Test that a warning is raised and all parameters are trainable in the expanded
        tape when not all parameters in U3 are trainable"""
        qs = qp.tape.QuantumScript(
            [qp.U3(qp.numpy.array(0.2), qp.numpy.array(0.4), qp.numpy.array(0.6), wires=0)],
            [qp.expval(qp.PauliZ(0))],
        )
        qs.trainable_params = [0, 2]

        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )
        res, _ = program((qs,))
        res = res[0]
        assert isinstance(res, qp.tape.QuantumScript)

        # U3 decomposes into 5 operators
        assert len(res.operations) == 5
        assert res.trainable_params == [0, 1, 2, 3, 4]

    def test_trainable_hermitian_warns(self):
        """Test attempting to compute the gradient of a tape that obtains the
        expectation value of a Hermitian operator emits a warning if the
        parameters to Hermitian are trainable."""

        mx = qp.numpy.array(qp.matrix(qp.PauliX(0) @ qp.PauliY(2)))
        qs = qp.tape.QuantumScript([], [qp.expval(qp.Hermitian(mx, wires=[0, 2]))])

        qs.trainable_params = {0}

        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )

        with pytest.warns(
            UserWarning, match="Differentiating with respect to the input parameters of Hermitian"
        ):
            _ = program((qs,))

    @pytest.mark.parametrize("G", [qp.RX, qp.RY, qp.RZ])
    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_valid_tape_no_expand(self, G):
        """Test that a tape that is valid doesn't raise errors and is not expanded"""
        prep_op = qp.StatePrep(
            qp.numpy.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )
        qs = qp.tape.QuantumScript(
            ops=[prep_op, G(np.pi, wires=[0])],
            measurements=[qp.expval(qp.PauliZ(0))],
        )

        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )

        qs.trainable_params = [1]
        qs_valid, _ = program((qs,))
        qs_valid = qs_valid[0]
        for o1, o2 in zip(qs.operations, qs_valid.operations):
            qp.assert_equal(o1, o2)
        for o1, o2 in zip(qs.measurements, qs_valid.measurements):
            qp.assert_equal(o1, o2)
        assert qs_valid.trainable_params == [1]  # same as input tape since no decomposition

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_valid_tape_with_expansion(self):
        """Test that a tape that is valid with operations that need to be expanded doesn't raise errors
        and is expanded"""
        prep_op = qp.StatePrep(
            qp.numpy.array([1.0, -1.0], requires_grad=False) / np.sqrt(2), wires=0
        )
        qs = qp.tape.QuantumScript(
            ops=[
                prep_op,
                qp.Rot(
                    qp.numpy.array(0.1), qp.numpy.array(0.2), qp.numpy.array(0.3), wires=[0]
                ),
            ],
            measurements=[qp.expval(qp.PauliZ(0))],
        )

        program = qp.device("default.qubit").preprocess_transforms(
            ExecutionConfig(gradient_method="adjoint")
        )

        qs.trainable_params = {1, 2, 3}
        qs_valid, _ = program((qs,))
        qs_valid = qs_valid[0]

        expected_ops = [
            prep_op,
            qp.RZ(qp.numpy.array(0.1), wires=[0]),
            qp.RY(qp.numpy.array(0.2), wires=[0]),
            qp.RZ(qp.numpy.array(0.3), wires=[0]),
        ]

        for o1, o2 in zip(qs_valid.operations, expected_ops):
            qp.assert_equal(o1, o2)
        for o1, o2 in zip(qs.measurements, qs_valid.measurements):
            qp.assert_equal(o1, o2)
        assert qs_valid.trainable_params == [0, 1, 2, 3]
        assert qs.shots == qs_valid.shots

    @pytest.mark.usefixtures("enable_and_disable_graph_decomp")
    def test_untrainable_operations(self):
        """Tests that a parametrized QubitUnitary that is not trainable is not expanded"""

        @qp.qnode(qp.device("default.qubit", wires=3), diff_method="adjoint")
        def circuit(x):
            qp.RX(x, 0)
            qp.QubitUnitary(np.eye(8), [0, 1, 2])
            return qp.expval(qp.PauliZ(2))

        x = pnp.array(1.1, requires_grad=True)
        assert qp.jacobian(circuit)(x) == 0


class TestDefaultQubitGraphModeExclusive:
    """Tests for DefaultQubit features that require graph mode enabled.
    The legacy decomposition mode should not be able to run these tests.

    NOTE: All tests in this suite will auto-enable graph mode via fixture.
    """

    @pytest.fixture(autouse=True)
    def enable_graph_mode_only(self):
        """Auto-enable graph mode for all tests in this class."""
        qp.decomposition.enable_graph()
        yield
        qp.decomposition.disable_graph()

    def test_insufficient_work_wires_causes_fallback(self):
        """Test that if a decomposition requires more work wires than available on default.qubit,
        that decomposition is discarded and fallback is used."""

        class MyDefaultQubitOp(qp.operation.Operator):
            num_wires = 1

        @qp.register_resources({qp.H: 2})
        def decomp_fallback(wires):
            qp.H(wires)
            qp.H(wires)

        @qp.register_resources({qp.X: 1}, work_wires={"burnable": 5})
        def decomp_with_work_wire(wires):
            qp.X(wires)

        qp.add_decomps(MyDefaultQubitOp, decomp_fallback, decomp_with_work_wire)

        tape = qp.tape.QuantumScript([MyDefaultQubitOp(0)])
        dev = qp.device("default.qubit", wires=1)  # Only 1 wire, but decomp needs 5 burnable
        program = dev.preprocess_transforms()
        (out_tape,), _ = program([tape])

        assert len(out_tape.operations) == 2
        assert out_tape.operations[0].name == "Hadamard"
        assert out_tape.operations[1].name == "Hadamard"
