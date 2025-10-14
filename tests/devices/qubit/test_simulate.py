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
"""Unit tests for simulate in devices/qubit."""

import mcm_utils
import numpy as np
import pytest
import scipy as sp
from dummy_debugger import Debugger
from stat_utils import fisher_exact_test

import pennylane as qml
from pennylane.devices.qubit import get_final_state, measure_final_state, simulate
from pennylane.devices.qubit.simulate import (
    TreeTraversalStack,
    _FlexShots,
    branch_state,
    combine_measurements_core,
    counts_to_probs,
    find_post_processed_mcms,
    samples_to_counts,
    simulate_one_shot_native_mcm,
    simulate_tree_mcm,
    split_circuit_at_mcms,
)

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self):
        """Test sample-only measurements raise a notimplementedError."""

        qs = qml.tape.QuantumScript(measurements=[qml.sample(wires=0)])
        with pytest.raises(NotImplementedError):
            simulate(qs)


def test_custom_operation():
    """Test execution works with a manually defined operator if it has a matrix."""

    # pylint: disable=too-few-public-methods
    class MyOperator(qml.operation.Operator):
        num_wires = 1

        @staticmethod
        def compute_matrix():
            return qml.PauliX.compute_matrix()

    qs = qml.tape.QuantumScript([MyOperator(0)], [qml.expval(qml.PauliZ(0))])

    result = simulate(qs)
    assert qml.math.allclose(result, -1.0)


# pylint: disable=too-few-public-methods
class TestSparsePipeline:
    """System tests for the sparse pipelines."""

    ground_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    cat_state = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)

    @pytest.mark.parametrize(
        "state",
        [
            ground_state,
            cat_state,
        ],
    )
    def test_sparse_op(self, state):
        """Test that a sparse QubitUnitary operation works on default.qubit with expval measurement."""
        mat = sp.sparse.csr_matrix([[0, 1], [1, 0]])
        op = qml.QubitUnitary(mat, wires=[0])
        qs = qml.tape.QuantumScript(
            ops=[qml.StatePrep(state, wires=range(8), pad_with=0), op],
            measurements=[qml.expval(qml.Z(0))],
        )

        result = simulate(qs)

        assert qml.math.allclose(result, -1)

    def test_sparse_state_prep(self):
        """Test a spares state prep can be acted upon by later operations."""
        state = sp.sparse.csr_matrix([0, 0, 0, 1])
        qml.StatePrep(state, wires=(0, 1))

        tape = qml.tape.QuantumScript(
            [qml.StatePrep(state, wires=(0, 1)), qml.X(0), qml.RX(0.0, 0)],
            [qml.probs(wires=(0, 1))],
        )

        res = simulate(tape)
        expected = np.array([0, 1, 0, 0])
        assert qml.math.allclose(res, expected)


# pylint: disable=too-few-public-methods
class TestStatePrepBase:
    """Tests integration with various state prep methods."""

    def test_basis_state(self):
        """Test that the BasisState operator prepares the desired state."""
        qs = qml.tape.QuantumScript(
            ops=[qml.BasisState([0, 1], wires=(0, 1))], measurements=[qml.probs(wires=(0, 1, 2))]
        )
        probs = simulate(qs)
        expected = np.zeros(8)
        expected[2] = 1.0
        assert qml.math.allclose(probs, expected)


class TestBasicCircuit:
    """Tests a basic circuit with one rx gate and two simple expectation values."""

    def test_analytic_mid_meas_raise(self):
        """Test measure_final_state raises an error when getting a mid-measurement dictionary."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )
        state, is_state_batched = get_final_state(qs)
        with pytest.raises(
            TypeError, match="Native mid-circuit measurements are only supported with finite shots."
        ):
            _ = measure_final_state(qs, state, is_state_batched, mid_measurements={})

    def test_basic_circuit_numpy(self):
        """Test execution with a basic circuit."""
        phi = np.array(0.397)
        qs = qml.tape.QuantumScript(
            [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

        state, is_state_batched = get_final_state(qs)
        result = measure_final_state(qs, state, is_state_batched)

        assert np.allclose(state, np.array([np.cos(phi / 2), -1j * np.sin(phi / 2)]))
        assert not is_state_batched

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

    @pytest.mark.autograd
    def test_autograd_results_and_backprop(self):
        """Tests execution and gradients with autograd"""
        phi = qml.numpy.array(-0.52)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return qml.numpy.array(simulate(qs))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        g = qml.jacobian(f)(phi)
        expected = np.array([-np.cos(phi), -np.sin(phi)])
        assert qml.math.allclose(g, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_results_and_backprop(self, use_jit):
        """Tests exeuction and gradients with jax."""
        import jax

        phi = jax.numpy.array(0.678)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        result = f(phi)
        assert qml.math.allclose(result[0], -np.sin(phi))
        assert qml.math.allclose(result[1], np.cos(phi))

        g = jax.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -np.sin(phi))

    @pytest.mark.torch
    def test_torch_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with torch."""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)

        def f(x):
            qs = qml.tape.QuantumScript(
                [qml.RX(x, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            return simulate(qs)

        result = f(phi)
        assert qml.math.allclose(result[0], -torch.sin(phi))
        assert qml.math.allclose(result[1], torch.cos(phi))

        g = torch.autograd.functional.jacobian(f, phi + 0j)
        assert qml.math.allclose(g[0], -torch.cos(phi))
        assert qml.math.allclose(g[1], -torch.sin(phi))

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_tf_results_and_backprop(self):
        """Tests execution and gradients of a simple circuit with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873, dtype="float64")

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript(
                [qml.RX(phi, wires=0)], [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            )
            result = simulate(qs)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        grad0 = grad_tape.jacobian(result[0], [phi])
        grad1 = grad_tape.jacobian(result[1], [phi])

        assert qml.math.allclose(grad0[0], -tf.cos(phi))
        assert qml.math.allclose(grad1[0], -tf.sin(phi))

    @pytest.mark.jax
    @pytest.mark.parametrize("op", [qml.RX(np.pi, 0), qml.BasisState([1], 0)])
    def test_result_has_correct_interface(self, op):
        """Test that even if no interface parameters are given, result is correct."""
        qs = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])
        res = simulate(qs, interface="jax")
        assert qml.math.get_interface(res) == "jax"
        assert qml.math.allclose(res, -1)

    def test_expand_state_keeps_autograd_interface(self):
        """Test that expand_state doesn't convert autograd to numpy."""

        @qml.qnode(qml.device("default.qubit", wires=2), interface="autograd")
        def circuit(x):
            qml.RX(x, 0)
            return qml.probs(wires=[0, 1])

        assert qml.math.get_interface(circuit(1.5)) == "autograd"


class TestBroadcasting:
    """Test that simulate works with broadcasted parameters"""

    def test_broadcasted_prep_state(self):
        """Test that simulate works for state measurements
        when the state prep has broadcasted parameters"""
        x = np.array(1.2)

        ops = [qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        prep = [qml.StatePrep(np.eye(4), wires=[0, 1])]

        qs = qml.tape.QuantumScript(prep + ops, measurements)
        res = simulate(qs)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]))
        assert np.allclose(res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]))

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched)
        expected_state = np.array(
            [
                [np.cos(x / 2), 0, 0, np.sin(x / 2)],
                [0, np.cos(x / 2), np.sin(x / 2), 0],
                [-np.sin(x / 2), 0, 0, np.cos(x / 2)],
                [0, -np.sin(x / 2), np.cos(x / 2), 0],
            ]
        ).reshape((4, 2, 2))

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]))
        assert np.allclose(res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]))

    def test_broadcasted_op_state(self):
        """Test that simulate works for state measurements
        when an operation has broadcasted parameters"""
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [qml.PauliX(wires=1), qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]

        qs = qml.tape.QuantumScript(ops, measurements)
        res = simulate(qs)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x))
        assert np.allclose(res[1], -np.cos(x))

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched)

        expected_state = np.zeros((4, 2, 2))
        expected_state[:, 0, 1] = np.cos(x / 2)
        expected_state[:, 1, 0] = np.sin(x / 2)

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x))
        assert np.allclose(res[1], -np.cos(x))

    def test_broadcasted_prep_sample(self, seed):
        """Test that simulate works for sample measurements
        when the state prep has broadcasted parameters"""
        x = np.array(1.2)

        ops = [qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]
        prep = [qml.StatePrep(np.eye(4), wires=[0, 1])]

        qs = qml.tape.QuantumScript(prep + ops, measurements, shots=qml.measurements.Shots(5000))
        res = simulate(qs, rng=seed)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(
            res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]), atol=0.05
        )
        assert np.allclose(
            res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]), atol=0.05
        )

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched, rng=seed)
        expected_state = np.array(
            [
                [np.cos(x / 2), 0, 0, np.sin(x / 2)],
                [0, np.cos(x / 2), np.sin(x / 2), 0],
                [-np.sin(x / 2), 0, 0, np.cos(x / 2)],
                [0, -np.sin(x / 2), np.cos(x / 2), 0],
            ]
        ).reshape((4, 2, 2))

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(
            res[0], np.array([np.cos(x), np.cos(x), -np.cos(x), -np.cos(x)]), atol=0.05
        )
        assert np.allclose(
            res[1], np.array([np.cos(x), -np.cos(x), -np.cos(x), np.cos(x)]), atol=0.05
        )

    def test_broadcasted_op_sample(self, seed):
        """Test that simulate works for sample measurements
        when an operation has broadcasted parameters"""
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [qml.PauliX(wires=1), qml.RY(x, wires=0), qml.CNOT(wires=[0, 1])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(2)]

        qs = qml.tape.QuantumScript(ops, measurements, shots=qml.measurements.Shots(5000))
        res = simulate(qs, rng=seed)

        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x), atol=0.05)
        assert np.allclose(res[1], -np.cos(x), atol=0.05)

        state, is_state_batched = get_final_state(qs)
        res = measure_final_state(qs, state, is_state_batched, rng=seed)

        expected_state = np.zeros((4, 2, 2))
        expected_state[:, 0, 1] = np.cos(x / 2)
        expected_state[:, 1, 0] = np.sin(x / 2)

        assert np.allclose(state, expected_state)
        assert is_state_batched
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert np.allclose(res[0], np.cos(x), atol=0.05)
        assert np.allclose(res[1], -np.cos(x), atol=0.05)

    def test_broadcasting_with_extra_measurement_wires(self, mocker):
        """Test that broadcasting works when the operations don't act on all wires."""
        # I can't mock anything in `simulate` because the module name is the function name
        spy = mocker.spy(qml, "map_wires")
        x = np.array([0.8, 1.0, 1.2, 1.4])

        ops = [qml.PauliX(wires=2), qml.RY(x, wires=1), qml.CNOT(wires=[1, 2])]
        measurements = [qml.expval(qml.PauliZ(i)) for i in range(3)]

        qs = qml.tape.QuantumScript(ops, measurements)
        res = simulate(qs)

        assert isinstance(res, tuple)
        assert len(res) == 3
        assert np.allclose(res[0], 1.0)
        assert np.allclose(res[1], np.cos(x))
        assert np.allclose(res[2], -np.cos(x))
        qml.assert_equal(spy.call_args_list[0].args[0], qs)
        assert spy.call_args_list[0].args[1] == {2: 0, 1: 1, 0: 2}


class TestPostselection:
    """Tests for applying projectors as operations."""

    def test_projector_norm(self):
        """Test that the norm of the state is maintained after applying a projector"""
        tape = qml.tape.QuantumScript(
            [qml.PauliX(0), qml.RX(0.123, 1), qml.Projector([0], wires=1)], [qml.state()]
        )
        res = simulate(tape)
        assert np.isclose(np.linalg.norm(res), 1.0)

    @pytest.mark.parametrize("shots", [None, 10, [10, 10]])
    def test_broadcasting_with_projector(self, shots):
        """Test that postselecting a broadcasted state raises an error"""
        tape = qml.tape.QuantumScript(
            [
                qml.RX([0.1, 0.2], 0),
                qml.Projector([0], wires=0),
            ],
            [qml.state()],
            shots=shots,
        )

        with pytest.raises(ValueError, match="Cannot postselect on circuits with broadcasting"):
            _ = simulate(tape)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "torch", "jax", "autograd"])
    def test_nan_state(self, interface):
        """Test that a state with nan values is returned if the probability of a postselection state
        is 0."""
        tape = qml.tape.QuantumScript([qml.PauliX(0), qml.Projector([0], 0)])

        res, _ = get_final_state(tape, interface=interface)
        assert qml.math.all(qml.math.isnan(res))


class Test_FlexShots:
    """Unit tests for _FlexShots"""

    @pytest.mark.parametrize(
        "shots, expected_shot_vector",
        [
            (0, (0,)),
            ((10, 0, 5, 0), (10, 0, 5, 0)),
            (((10, 3), (0, 5)), (10, 10, 10, 0, 0, 0, 0, 0)),
        ],
    )
    def test_init_with_zero_shots(self, shots, expected_shot_vector):
        """Test that _FlexShots is initialized correctly with zero shots"""
        flex_shots = _FlexShots(shots)
        shot_vector = tuple(s for s in flex_shots)
        assert shot_vector == expected_shot_vector

    def test_init_with_other_shots(self):
        """Test that a new _FlexShots object is not created if the input is a _FlexShots object."""
        shots = _FlexShots(10)
        new_shots = _FlexShots(shots)
        assert new_shots is shots


class TestDebugger:
    """Tests that the debugger works for a simple circuit"""

    def test_debugger_numpy(self):
        """Test debugger with numpy"""
        phi = np.array(0.397)
        ops = [qml.Snapshot(), qml.RX(phi, wires=0), qml.Snapshot("final_state")]
        qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])

        debugger = Debugger()
        result = simulate(qs, debugger=debugger)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert np.allclose(result[0], -np.sin(phi))
        assert np.allclose(result[1], np.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert np.allclose(debugger.snapshots[0], [1, 0])
        assert np.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )

    @pytest.mark.autograd
    def test_debugger_autograd(self):
        """Tests debugger with autograd"""
        phi = qml.numpy.array(-0.52)
        debugger = Debugger()

        def f(x):
            ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
            qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
            return qml.numpy.array(simulate(qs, debugger=debugger))

        result = f(phi)
        expected = np.array([-np.sin(phi), np.cos(phi)])
        assert qml.math.allclose(result, expected)

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        assert qml.math.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )

    @pytest.mark.jax
    def test_debugger_jax(self):
        """Tests debugger with JAX"""
        import jax

        phi = jax.numpy.array(0.678)
        debugger = Debugger()

        def f(x):
            ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
            qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
            return simulate(qs, debugger=debugger)

        result = f(phi)
        assert qml.math.allclose(result[0], -np.sin(phi))
        assert qml.math.allclose(result[1], np.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        assert qml.math.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )

    @pytest.mark.torch
    def test_debugger_torch(self):
        """Tests debugger with torch"""

        import torch

        phi = torch.tensor(-0.526, requires_grad=True)
        debugger = Debugger()

        def f(x):
            ops = [qml.Snapshot(), qml.RX(x, wires=0), qml.Snapshot("final_state")]
            qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
            return simulate(qs, debugger=debugger)

        result = f(phi)
        assert qml.math.allclose(result[0], -torch.sin(phi))
        assert qml.math.allclose(result[1], torch.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        print(debugger.snapshots["final_state"])
        assert qml.math.allclose(
            debugger.snapshots["final_state"],
            torch.tensor([torch.cos(phi / 2), -torch.sin(phi / 2) * 1j]),
        )

    # pylint: disable=invalid-unary-operand-type
    @pytest.mark.tf
    def test_debugger_tf(self):
        """Tests debugger with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(4.873, dtype="float64")
        debugger = Debugger()

        ops = [qml.Snapshot(), qml.RX(phi, wires=0), qml.Snapshot("final_state")]
        qs = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
        result = simulate(qs, debugger=debugger)

        assert qml.math.allclose(result[0], -tf.sin(phi))
        assert qml.math.allclose(result[1], tf.cos(phi))

        assert list(debugger.snapshots.keys()) == [0, "final_state"]
        assert qml.math.allclose(debugger.snapshots[0], [1, 0])
        assert qml.math.allclose(
            debugger.snapshots["final_state"], [np.cos(phi / 2), -np.sin(phi / 2) * 1j]
        )


class TestSampleMeasurements:
    """Tests circuits with sample-based measurements"""

    def test_single_expval(self):
        """Test a simple circuit with a single expval measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=10000)
        result = simulate(qs)

        assert isinstance(result, np.float64)
        assert result.shape == ()
        assert np.allclose(result, np.cos(x), atol=0.1)

    def test_single_probs(self):
        """Test a simple circuit with a single prob measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=10000)
        result = simulate(qs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert np.allclose(result, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1)

    def test_single_sample(self):
        """Test a simple circuit with a single sample measurement"""
        x = np.array(0.732)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=10000)
        result = simulate(qs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10000, 2)
        assert np.allclose(
            np.sum(result, axis=0).astype(np.float32) / 10000, [np.sin(x / 2) ** 2, 0], atol=0.1
        )

    def test_multi_measurements(self):
        """Test a simple circuit containing multiple measurements"""
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.PauliZ(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=10000,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.float64)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

        assert np.allclose(result[0], np.cos(x), atol=0.1)

        assert result[1].shape == (4,)
        assert np.allclose(
            result[1],
            [
                np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
                np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
            ],
            atol=0.1,
        )

        assert result[2].shape == (10000, 2)

    def test_shots_reuse(self, mocker):
        """Test that samples are reused when two measurements commute"""
        ops = [qml.Hadamard(0), qml.CNOT([0, 1])]
        mps = [
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliX(1)),
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliX(1)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[0]),
            qml.probs(wires=[0, 1]),
            qml.sample(wires=[0, 1]),
            qml.expval(
                qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(1)])
            ),
            qml.expval(qml.sum(qml.PauliX(0), qml.PauliZ(1), qml.PauliY(1))),
            qml.expval(qml.s_prod(2.0, qml.PauliX(0))),
            qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1))),
        ]

        qs = qml.tape.QuantumScript(ops, mps, shots=100)

        spy = mocker.spy(qml.devices.qubit.sampling, "sample_state")
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(mps)

        # check that samples are reused when possible
        assert spy.call_count == 8

    shots_data = [
        [10000, 10000],
        [(10000, 2)],
        [10000, 20000],
        [(10000, 2), 20000],
        [(10000, 3), 20000, (30000, 2)],
    ]

    @pytest.mark.parametrize("shots", shots_data)
    def test_expval_shot_vector(self, shots):
        """Test a simple circuit with a single expval measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.expval(qml.PauliZ(0))], shots=shots)
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.float64) for res in result)
        assert all(res.shape == () for res in result)
        assert all(np.allclose(res, np.cos(x), atol=0.1) for res in result)

    @pytest.mark.parametrize("shots", shots_data)
    def test_probs_shot_vector(self, shots):
        """Test a simple circuit with a single prob measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.probs(wires=0)], shots=shots)
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.ndarray) for res in result)
        assert all(res.shape == (2,) for res in result)
        assert all(
            np.allclose(res, [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2], atol=0.1) for res in result
        )

    @pytest.mark.parametrize("shots", shots_data)
    def test_sample_shot_vector(self, shots):
        """Test a simple circuit with a single sample measurement for shot vectors"""
        x = np.array(0.732)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript([qml.RY(x, wires=0)], [qml.sample(wires=range(2))], shots=shots)
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        assert all(isinstance(res, np.ndarray) for res in result)
        assert all(res.shape == (s, 2) for res, s in zip(result, shots))
        assert all(
            np.allclose(
                np.sum(res, axis=0).astype(np.float32) / s, [np.sin(x / 2) ** 2, 0], atol=0.1
            )
            for res, s in zip(result, shots)
        )

    @pytest.mark.parametrize("shots", shots_data)
    def test_multi_measurement_shot_vector(self, shots):
        """Test a simple circuit containing multiple measurements for shot vectors"""
        x, y = np.array(0.732), np.array(0.488)
        shots = qml.measurements.Shots(shots)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires=0), qml.CNOT(wires=[0, 1]), qml.RY(y, wires=1)],
            [qml.expval(qml.PauliZ(0)), qml.probs(wires=range(2)), qml.sample(wires=range(2))],
            shots=shots,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == len(list(shots))

        for shot_res, s in zip(result, shots):
            assert isinstance(shot_res, tuple)
            assert len(shot_res) == 3

            assert isinstance(shot_res[0], np.float64)
            assert isinstance(shot_res[1], np.ndarray)
            assert isinstance(shot_res[2], np.ndarray)

            assert np.allclose(shot_res[0], np.cos(x), atol=0.1)

            assert shot_res[1].shape == (4,)
            assert np.allclose(
                shot_res[1],
                [
                    np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
                    np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
                    np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
                    np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
                ],
                atol=0.1,
            )

            assert shot_res[2].shape == (s, 2)

    def test_custom_wire_labels(self):
        """Test that custom wire labels works as expected"""
        x, y = np.array(0.732), np.array(0.488)
        qs = qml.tape.QuantumScript(
            [qml.RX(x, wires="b"), qml.CNOT(wires=["b", "a"]), qml.RY(y, wires="a")],
            [
                qml.expval(qml.PauliZ("b")),
                qml.probs(wires=["a", "b"]),
                qml.sample(wires=["b", "a"]),
            ],
            shots=10000,
        )
        result = simulate(qs)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.float64)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

        assert np.allclose(result[0], np.cos(x), atol=0.1)

        assert result[1].shape == (4,)
        assert np.allclose(
            result[1],
            [
                np.cos(x / 2) ** 2 * np.cos(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.cos(x / 2) ** 2 * np.sin(y / 2) ** 2,
                np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2,
            ],
            atol=0.1,
        )

        assert result[2].shape == (10000, 2)


class TestOperatorArithmetic:
    def test_numpy_op_arithmetic(self):
        """Test an operator arithmetic circuit with non-integer wires with numpy."""
        phi = 1.2
        ops = [
            qml.PauliX("a"),
            qml.PauliX("b"),
            qml.ctrl(qml.RX(phi, "target") ** 2, ("a", "b", -3), control_values=[1, 1, 0]),
        ]

        qs = qml.tape.QuantumScript(
            ops,
            [
                qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
            ],
        )

        results = simulate(qs)
        assert qml.math.allclose(results[0], -np.sin(2 * phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(2 * phi))

    @pytest.mark.autograd
    def test_autograd_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with autograd."""

        phi = qml.numpy.array(1.2)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return qml.numpy.array(simulate(qs))

        results = f(phi)
        assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(phi))

        g = qml.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -3 * np.sin(phi))

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_jax_op_arithmetic(self, use_jit):
        """Test operator arithmetic circuit with non-integer wires works with jax."""
        import jax

        phi = jax.numpy.array(1.2)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        results = f(phi)
        assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(phi))

        g = jax.jacobian(f)(phi)
        assert qml.math.allclose(g[0], -np.cos(phi))
        assert qml.math.allclose(g[1], -3 * np.sin(phi))

    @pytest.mark.torch
    def test_torch_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with torch."""
        import torch

        phi = torch.tensor(-0.7290, requires_grad=True)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return simulate(qs)

        results = f(phi)
        assert qml.math.allclose(results[0], -torch.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * torch.cos(phi))

        g = torch.autograd.functional.jacobian(f, phi)
        assert qml.math.allclose(g[0], -torch.cos(phi))
        assert qml.math.allclose(g[1], -3 * torch.sin(phi))

    @pytest.mark.tf
    def test_tensorflow_op_arithmetic(self):
        """Test operator arithmetic circuit with non-integer wires works with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(0.4203)

        def f(x):
            ops = [
                qml.PauliX("a"),
                qml.PauliX("b"),
                qml.ctrl(qml.RX(x, "target"), ("a", "b", -3), control_values=[1, 1, 0]),
            ]

            qs = qml.tape.QuantumScript(
                ops,
                [
                    qml.expval(qml.sum(qml.PauliY("target"), qml.PauliZ("b"))),
                    qml.expval(qml.s_prod(3, qml.PauliZ("target"))),
                ],
            )

            return simulate(qs)

        with tf.GradientTape(persistent=True) as tape:
            results = f(phi)

        assert qml.math.allclose(results[0], -np.sin(phi) - 1)
        assert qml.math.allclose(results[1], 3 * np.cos(phi))

        g0 = tape.gradient(results[0], phi)
        assert qml.math.allclose(g0, -np.cos(phi))
        g1 = tape.gradient(results[1], phi)
        assert qml.math.allclose(g1, -3 * np.sin(phi))


class TestQInfoMeasurements:
    measurements = [
        qml.density_matrix(0),
        qml.density_matrix(1),
        qml.density_matrix((0, 1)),
        qml.vn_entropy(0),
        qml.vn_entropy(1),
        qml.mutual_info(0, 1),
    ]

    def expected_results(self, phi):
        density_i = np.array([[np.cos(phi / 2) ** 2, 0], [0, np.sin(phi / 2) ** 2]])
        density_both = np.array(
            [
                [np.cos(phi / 2) ** 2, 0, 0, 0.0 + np.sin(phi) * 0.5j],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.0 - np.sin(phi) * 0.5j, 0, 0, np.sin(phi / 2) ** 2],
            ]
        )
        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = [eig for eig in eigs if eig > 0]
        rho_log_rho = eigs * np.log(eigs)
        expected_entropy = -np.sum(rho_log_rho)
        mutual_info = 2 * expected_entropy

        return (density_i, density_i, density_both, expected_entropy, expected_entropy, mutual_info)

    def calculate_entropy_grad(self, phi):
        eig_1 = (1 + np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eig_2 = (1 - np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)) / 2
        eigs = [eig_1, eig_2]
        eigs = np.maximum(eigs, 1e-08)

        return -(
            (np.log(eigs[0]) + 1)
            * (np.sin(phi / 2) ** 3 * np.cos(phi / 2) - np.sin(phi / 2) * np.cos(phi / 2) ** 3)
            / np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)
        ) - (
            (np.log(eigs[1]) + 1)
            * (np.sin(phi / 2) * np.cos(phi / 2) * (np.cos(phi / 2) ** 2 - np.sin(phi / 2) ** 2))
            / np.sqrt(1 - 4 * np.cos(phi / 2) ** 2 * np.sin(phi / 2) ** 2)
        )

    def expected_grad(self, phi):
        p_2 = phi / 2
        g_density_i = np.array([[-np.cos(p_2) * np.sin(p_2), 0], [0, np.sin(p_2) * np.cos(p_2)]])
        g_density_both = np.array(
            [
                [-np.cos(p_2) * np.sin(p_2), 0, 0, 0.0 + 0.5j * np.cos(phi)],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.0 - 0.5j * np.cos(phi), 0, 0, np.sin(p_2) * np.cos(p_2)],
            ]
        )
        g_entropy = self.calculate_entropy_grad(phi)
        g_mutual_info = 2 * g_entropy
        return (g_density_i, g_density_i, g_density_both, g_entropy, g_entropy, g_mutual_info)

    def test_qinfo_numpy(self):
        """Test quantum info measurements with numpy"""
        phi = -0.623
        qs = qml.tape.QuantumScript([qml.IsingXX(phi, wires=(0, 1))], self.measurements)

        results = simulate(qs)
        for val1, val2 in zip(results, self.expected_results(phi)):
            assert qml.math.allclose(val1, val2)

    @pytest.mark.autograd
    def test_qinfo_autograd_execute(self):
        """Tests autograd execution with qinfo measurements."""
        phi = qml.numpy.array(6.920)

        def f(x):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], self.measurements)
            return simulate(qs)

        results = f(phi)
        expected = self.expected_results(phi)
        for val1, val2 in zip(results, expected):
            assert qml.math.allclose(val1, val2)

    @pytest.mark.autograd
    def test_qinfo_autograd_backprop(self):
        """Tests autograd derivatives with qinfo measurements.
        Only simulates one measurement at a time due to autograd differentiation limitations."""

        phi = qml.numpy.array(0.263)

        def f(x, m_ind, real=True):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], [self.measurements[m_ind]])
            out = simulate(qs)
            return qml.math.real(out) if real else qml.math.imag(out)

        expected_grads = self.expected_grad(phi)

        for i in (0, 1, 3, 4, 5):  # skip density matrix on both wires
            g = qml.jacobian(f)(phi, i, real=True)
            assert qml.math.allclose(expected_grads[i], g)

        # density matrix on both wires case
        g_real = qml.jacobian(f)(phi, 2, True)
        g_imag = qml.jacobian(f)(phi, 2, False)
        assert qml.math.allclose(g_real + 1j * g_imag, expected_grads[2])

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_qinfo_jax(self, use_jit):
        """Test qinfo meausrements work with jax and jitting."""

        import jax

        def f(x):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], self.measurements)
            return simulate(qs)

        if use_jit:
            f = jax.jit(f)

        phi = jax.numpy.array(-0.792)

        results = f(phi)
        for val1, val2 in zip(results, self.expected_results(phi)):
            assert qml.math.allclose(val1, val2)

        def real_out(phi):
            return tuple(jax.numpy.real(r) for r in f(phi))

        grad_real = jax.jacobian(real_out)(phi)

        def imag_out(phi):
            return tuple(jax.numpy.imag(r) for r in f(phi))

        grad_imag = jax.jacobian(imag_out)(phi)
        grads = tuple(r + 1j * i for r, i in zip(grad_real, grad_imag))
        expected_grads = self.expected_grad(phi)

        # Writing this way makes it easier to figure out which is failing
        # density 0
        assert qml.math.allclose(grads[0], expected_grads[0])
        # density 1
        assert qml.math.allclose(grads[1], expected_grads[1])
        # density both
        assert qml.math.allclose(grads[2], expected_grads[2])
        # entropy 0
        assert qml.math.allclose(grads[3], expected_grads[3])
        # entropy 1
        assert qml.math.allclose(grads[4], expected_grads[4])
        # mutual info
        assert qml.math.allclose(grads[5], expected_grads[5])

    @pytest.mark.torch
    def test_qinfo_torch(self):
        """Test qinfo measurements with torch."""
        import torch

        phi = torch.tensor(1.928, requires_grad=True)

        def f(x):
            qs = qml.tape.QuantumScript([qml.IsingXX(x, wires=(0, 1))], self.measurements)
            return simulate(qs)

        results = f(phi)
        for val1, val2 in zip(results, self.expected_results(phi.detach().numpy())):
            assert qml.math.allclose(val1, val2)

        def real_out(phi):
            return tuple(torch.real(r) for r in f(torch.real(phi)))

        grad_real = torch.autograd.functional.jacobian(real_out, phi)

        def imag_out(phi):
            return tuple(torch.imag(r) if torch.is_complex(r) else torch.tensor(0) for r in f(phi))

        grad_imag = torch.autograd.functional.jacobian(imag_out, phi)
        grads = tuple(r + 1j * i for r, i in zip(grad_real, grad_imag))

        expected_grads = self.expected_grad(phi.detach().numpy())

        # Writing this way makes it easier to figure out which is failing
        # density 0
        assert qml.math.allclose(grads[0], expected_grads[0])
        # density 1
        assert qml.math.allclose(grads[1], expected_grads[1])
        # density both
        assert qml.math.allclose(grads[2], expected_grads[2])
        # entropy 0
        assert qml.math.allclose(grads[3], expected_grads[3])
        # entropy 1
        assert qml.math.allclose(grads[4], expected_grads[4])
        # mutual info
        assert qml.math.allclose(grads[5], expected_grads[5])

    @pytest.mark.tf
    def test_qinfo_tf(self):
        """Test qinfo measurements with tensorflow."""
        import tensorflow as tf

        phi = tf.Variable(0.9276)

        with tf.GradientTape(persistent=True) as grad_tape:
            qs = qml.tape.QuantumScript([qml.IsingXX(phi, wires=(0, 1))], self.measurements)
            results = simulate(qs)

            result2_real = tf.math.real(results[2])
            result2_imag = tf.math.imag(results[2])

        expected = self.expected_results(phi)
        for val1, val2 in zip(results, expected):
            assert qml.math.allclose(val1, val2)

        expected_grads = self.expected_grad(phi)

        grad0 = grad_tape.jacobian(results[0], phi)
        assert qml.math.allclose(grad0, expected_grads[0])

        grad1 = grad_tape.jacobian(results[1], phi)
        assert qml.math.allclose(grad1, expected_grads[1])

        grad2_real = grad_tape.jacobian(result2_real, phi)
        grad2_imag = grad_tape.jacobian(result2_imag, phi)
        assert qml.math.allclose(
            np.array(grad2_real) + 1j * np.array(grad2_imag), expected_grads[2]
        )

        grad3 = grad_tape.jacobian(results[3], phi)
        assert qml.math.allclose(grad3, expected_grads[3])

        grad4 = grad_tape.jacobian(results[4], phi)
        assert qml.math.allclose(grad4, expected_grads[4])

        grad5 = grad_tape.jacobian(results[5], phi)
        assert qml.math.allclose(grad5, expected_grads[5])


@pytest.mark.unit
class TestTreeTraversalStack:
    """Unit tests for TreeTraversalStack"""

    @pytest.mark.parametrize(
        "max_depth",
        [0, 1, 10, 100],
    )
    def test_init_with_depth(self, max_depth):
        """Test that TreeTraversalStack is initialized correctly with given ``max_depth``"""
        tree_stack = TreeTraversalStack(max_depth)

        assert tree_stack.counts.count(None) == max_depth
        assert tree_stack.probs.count(None) == max_depth
        assert tree_stack.results_0.count(None) == max_depth
        assert tree_stack.results_1.count(None) == max_depth
        assert tree_stack.states.count(None) == max_depth

    def test_full_prune_empty_methods(self):
        """Test that TreeTraversalStack object's class methods work correctly."""

        max_depth = 10
        tree_stack = TreeTraversalStack(max_depth)

        np.random.shuffle(r_depths := list(range(max_depth)))
        for depth in r_depths:
            counts_0 = np.random.randint(1, 9)
            counts_1 = 10 - counts_0
            tree_stack.counts[depth] = [counts_0, counts_1]
            tree_stack.probs[depth] = [counts_0 / 10, counts_1 / 10]
            tree_stack.results_0[depth] = [0] * counts_0
            tree_stack.results_1[depth] = [1] * counts_1
            tree_stack.states[depth] = [np.sqrt(counts_0), np.sqrt(counts_1)]
            assert tree_stack.is_full(depth)

            assert tree_stack.counts[depth] == list(
                samples_to_counts(
                    np.array(tree_stack.results_0[depth] + tree_stack.results_1[depth])
                ).values()
            )
            assert tree_stack.probs[depth] == list(
                counts_to_probs(dict(zip([0, 1], tree_stack.counts[depth]))).values()
            )

            state_vec = np.array(tree_stack.states[depth]).T
            state_vec /= np.linalg.norm(tree_stack.states[depth])
            meas, meas_r = qml.measure(0), qml.measure(0, reset=True)
            assert np.allclose(branch_state(state_vec, 0, meas.measurements[0]), [1.0, 0.0])
            assert np.allclose(branch_state(state_vec, 1, meas_r.measurements[0]), [1.0, 0.0])

            tree_stack.prune(depth)
            assert tree_stack.any_is_empty(depth)


@pytest.mark.unit
class TestMidMeasurements:
    """Tests for simulating scripts with mid-circuit measurements using the ``simulate_tree_mcm``."""

    @pytest.mark.parametrize(
        "shots, postselect_mode, error",
        [
            (10, "fill-shots", True),
            (None, "fill-shots", False),
            (10, "hw-like", False),
            (None, "hw-like", False),
        ],
    )
    def test_defer_measurements_fill_shots_zero_prob_postselection_error(
        self, shots, postselect_mode, error
    ):
        """Test that an error is raised if `postselect_mode="fill-shots"` with finite shots
        and the postselection probability is zero when using defer_measurements."""

        # State is |0>, so postselection probability is zero
        qs = qml.tape.QuantumScript(
            [qml.measurements.MidMeasureMP(0, postselect=1)], [qml.expval(qml.Z(0))], shots=shots
        )
        [deferred_qs], _ = qml.defer_measurements(qs)

        if error:
            with pytest.raises(RuntimeError, match="The probability of the postselected"):
                simulate(deferred_qs, mcm_method="deferred", postselect_mode=postselect_mode)

        else:
            # No error should be raised
            simulate(deferred_qs, mcm_method="deferred", postselect_mode=postselect_mode)

    @pytest.mark.parametrize("val", [0, 1])
    def test_basic_mid_meas_circuit(self, val):
        """Test execution with a basic circuit with mid-circuit measurements."""
        qs = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1]), qml.measurements.MidMeasureMP(0, postselect=val)],
            [qml.expval(qml.X(0)), qml.expval(qml.Z(0))],
        )
        result = simulate_tree_mcm(qs)
        assert result == (0, (-1.0) ** val)

    def test_basic_mid_meas_circuit_with_reset(self):
        """Test execution with a basic circuit with mid-circuit measurements."""
        qs = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.CNOT([0, 1]),
                (m0 := qml.measure(0, reset=True)).measurements[0],
                qml.Hadamard(0),
                qml.CNOT([1, 0]),
            ],  # equivalent to a circuit that gives equiprobable basis states
            [qml.probs(op=m0), qml.probs(op=qml.Z(0)), qml.probs(op=qml.Z(1))],
        )
        result = simulate_tree_mcm(qs)
        assert qml.math.allclose(result, qml.math.array([0.5, 0.5]))

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize("shots", [None, 5500])
    @pytest.mark.parametrize("postselect", [None, 0])
    @pytest.mark.parametrize("reset", [False, True])
    @pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
    @pytest.mark.parametrize(
        "meas_obj", [qml.Y(0), [1], [1, 0], "mcm", "composite_mcm", "mcm_list"]
    )
    def test_simple_dynamic_circuit(self, *, shots, measure_f, postselect, reset, meas_obj, seed):
        """Tests that `simulate` can handles a simple dynamic circuit with the following measurements:

            * qml.counts with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
            * qml.expval with obs (comp basis or not), MCM, f(MCM), MCM list
            * qml.probs with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
            * qml.sample with obs (comp basis or not), single wire, multiple wires (ordered/unordered), MCM, f(MCM), MCM list
            * qml.var with obs (comp basis or not), MCM, f(MCM), MCM list

        The above combinations should work for finite shots, shot vectors and post-selecting of either the 0 or 1 branch.
        """

        if measure_f in (qml.expval, qml.var) and (
            isinstance(meas_obj, list) or meas_obj == "mcm_list"
        ):
            pytest.skip("Can't use wires/mcm lists with var or expval")

        if measure_f in (qml.counts, qml.sample) and shots is None:
            pytest.skip("Can't measure counts/sample in analytic mode (`shots=None`)")

        if measure_f in (qml.probs,) and meas_obj in ["composite_mcm"]:
            pytest.skip(
                "Cannot use qml.probs() when measuring multiple mid-circuit measurements collected using arithmetic operators."
            )

        qscript = qml.tape.QuantumScript(
            [
                qml.RX(np.pi / 2.5, 0),
                qml.RZ(np.pi / 4, 0),
                (m0 := qml.measure(0, reset=reset)).measurements[0],
                qml.ops.op_math.Conditional(m0 == 0, qml.RX(np.pi / 4, 0)),
                qml.ops.op_math.Conditional(m0 == 1, qml.RX(-np.pi / 4, 0)),
                qml.RX(np.pi / 3, 1),
                qml.RZ(np.pi / 4, 1),
                (m1 := qml.measure(1, postselect=postselect)).measurements[0],
                qml.ops.op_math.Conditional(m1 == 0, qml.RY(np.pi / 4, 1)),
                qml.ops.op_math.Conditional(m1 == 1, qml.RY(-np.pi / 4, 1)),
            ],
            [
                measure_f(
                    **{
                        "wires" if isinstance(meas_obj, list) else "op": (
                            (
                                m0
                                if meas_obj == "mcm"
                                else (0.5 * m0 + m1 if meas_obj == "composite_mcm" else [m0, m1])
                            )
                            if isinstance(meas_obj, str)
                            else meas_obj
                        )
                    }
                )
            ],
            shots=shots,
        )

        rng = np.random.default_rng(seed)
        results0 = simulate(qscript, mcm_method="tree-traversal", rng=rng)

        deferred_tapes, deferred_func = qml.defer_measurements(qscript)
        results1 = deferred_func(
            [simulate(tape, mcm_method="deferred", rng=rng) for tape in deferred_tapes]
        )
        mcm_utils.validate_measurements(measure_f, shots, results1, results0)

        if shots is not None:
            one_shot_tapes, one_shot_func = qml.dynamic_one_shot(qscript)
            results2 = one_shot_func(
                [simulate(tape, mcm_method="one-shot", rng=rng) for tape in one_shot_tapes]
            )
            mcm_utils.validate_measurements(measure_f, shots, results2, results0)

    @pytest.mark.parametrize("shots", [None, 5500, [5500, 5500]])
    @pytest.mark.parametrize("angles", [(0.123, 0.015), (0.543, 0.057)])
    @pytest.mark.parametrize("measure_f", [qml.probs, qml.sample])
    def test_approx_dynamic_mid_meas_circuit(self, shots, angles, measure_f, seed):
        """Test execution of a dynamic circuit with an equivalent static one."""

        if measure_f in (qml.sample,) and shots is None:
            pytest.skip("Can't measure samples in analytic mode (`shots=None`)")

        qs_with_mid_meas = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.CZ([0, 1]),
                qml.CNOT([0, 2]),
                qml.CNOT([2, 3]),
                qml.CZ([1, 3]),
                qml.Toffoli([3, 2, 0]),
                (m0 := qml.measure(0)).measurements[0],
                qml.ops.op_math.Conditional(m0, qml.RZ(angles[0], 1)),
                qml.Hadamard(1),
                qml.Z(1),
                (m1 := qml.measure(1)).measurements[0],
                qml.ops.op_math.Conditional(m1, qml.RX(angles[1], 3)),
            ],
            [measure_f(wires=[0, 1, 2, 3])],
            shots=shots,
        )
        qs_without_mid_meas = qml.tape.QuantumScript(
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.CZ([0, 1]),
                qml.CNOT([0, 2]),
                qml.CNOT([2, 3]),
                qml.CZ([1, 3]),
                qml.Toffoli([3, 2, 0]),
                qml.Hadamard(1),
                qml.Z(1),
                qml.RX(angles[1], 3),
            ],
            [measure_f(wires=[0, 1, 2, 3])],
            shots=shots,
        )  # approximate compiled circuit of the above
        res1 = simulate_tree_mcm(qs_with_mid_meas, rng=seed)
        res2 = simulate(qs_without_mid_meas, rng=seed)

        if not isinstance(shots, list):
            res1, res2 = (res1,), (res2,)

        for rs1, rs2 in zip(res1, res2):
            prob_dist1, prob_dist2 = rs1, rs2
            if measure_f in (qml.sample,):
                n_wires = rs1.shape[1]
                prob_dist1, prob_dist2 = np.zeros(2**n_wires), np.zeros(2**n_wires)
                for prob, rs in zip([prob_dist1, prob_dist2], [rs1, rs2]):
                    index, count = np.unique(
                        np.packbits(rs, axis=1, bitorder="little").squeeze(), return_counts=True
                    )
                    prob[index] = count

            assert qml.math.allclose(
                sp.stats.entropy(prob_dist1 + 1e-12, prob_dist2 + 1e-12), 0.0, atol=5e-2
            )

    @pytest.mark.parametrize("ml_framework", ml_frameworks_list)
    @pytest.mark.parametrize("postselect_mode", [None, "hw-like", "pad-invalid-samples"])
    def test_tree_traversal_interface_mcm(self, ml_framework, postselect_mode, seed):
        """Test that tree traversal works numerically with different interfaces"""
        # pylint:disable = singleton-comparison, import-outside-toplevel

        qscript = qml.tape.QuantumScript(
            [
                qml.RX(np.pi / 4, wires=0),
                (m0 := qml.measure(0, reset=True)).measurements[0],
                qml.RX(np.pi / 4, wires=0),
            ],
            [qml.sample(qml.Z(0)), qml.sample(m0)],
            shots=5500,
        )

        rng = np.random.default_rng(seed)
        res1, res2 = simulate_tree_mcm(qscript, interface=ml_framework, rng=rng)

        p1 = [qml.math.mean(res1 == -1), qml.math.mean(res1 == 1)]
        p2 = [qml.math.mean(res2 == True), qml.math.mean(res2 == False)]
        assert qml.math.allclose(qml.math.sum(sp.special.rel_entr(p1, p2)), 0.0, atol=0.05)

        qscript2 = qml.tape.QuantumScript(
            [
                qml.RX(np.pi / 4, wires=0),
                (m0 := qml.measure(0, postselect=0)).measurements[0],
                qml.RX(np.pi / 4, wires=0),
            ],
            [qml.sample(qml.Z(0))],
            shots=5500,
        )
        qscript3 = qml.tape.QuantumScript(
            [qml.RX(np.pi / 4, wires=0)], [qml.sample(qml.Z(0))], shots=5500
        )

        res3 = simulate_tree_mcm(qscript2, postselect_mode=postselect_mode, rng=rng)
        res4 = simulate(qscript3, rng=rng)

        p3 = [qml.math.mean(res3 == -1), qml.math.mean(res3 == 1)]
        p4 = [qml.math.mean(res4 == -1), qml.math.mean(res4 == 1)]
        assert qml.math.allclose(qml.math.sum(sp.special.rel_entr(p3, p4)), 0.0, atol=0.05)

    def test_tree_traversal_postselect_mode(self):
        """Test that invalid shots are discarded if requested"""

        shots = 100
        qscript = qml.tape.QuantumScript(
            [
                qml.RX(np.pi / 2, 0),
                (m0 := qml.measure(0, postselect=1)).measurements[0],
                qml.ops.op_math.Conditional(m0, qml.RZ(1.57, 1)),
            ],
            [qml.sample(wires=[0, 1])],
            shots=shots,
        )

        res = simulate_tree_mcm(qscript, postselect_mode="hw-like")

        assert len(res) < shots
        assert np.all(res != np.iinfo(np.int32).min)

    def test_tree_traversal_deep_circuit(self):
        """Test that `simulate_tree_mcm` works with circuits with many mid-circuit measurements"""

        n_circs = 500
        operations = []
        for _ in range(n_circs):
            operations.extend(
                [
                    qml.RX(1.234, 0),
                    (m0 := qml.measure(0, postselect=1)).measurements[0],
                    qml.CNOT([0, 1]),
                    qml.ops.op_math.Conditional(m0, qml.RZ(1.786, 1)),
                ]
            )

        qscript = qml.tape.QuantumScript(
            operations,
            [qml.sample(wires=[0, 1]), qml.counts(wires=[0, 1])],
            shots=20,
        )

        mcms = find_post_processed_mcms(qscript)
        assert len(mcms) == n_circs

        split_circs = split_circuit_at_mcms(qscript)
        assert len(split_circs) == n_circs + 1
        for circ in split_circs:
            assert not [o for o in circ.operations if isinstance(o, qml.measurements.MidMeasureMP)]

    @pytest.mark.parametrize(
        "measurements, expected",
        [
            [(qml.counts(0), {"a": (1, {0: 42}), "b": (2, {1: 58})}), {0: 42, 1: 58}],
            [
                (qml.expval(qml.Z(0)), {"a": (1, (0.42, -1)), "b": (2, (0.58, 1))}),
                [1.58 / 3, 1 / 3],
            ],
            [(qml.probs(wires=0), {"a": (1, (0.42, -1)), "b": (2, (0.58, 1))}), [1.58 / 3, 1 / 3]],
            [(qml.sample(wires=0), {"a": (1, (0, 1, 0)), "b": (2, (1, 0, 1))}), [0, 1, 0, 1, 0, 1]],
        ],
    )
    def test_tree_traversal_combine_measurements(self, measurements, expected):
        """Test that the measurement value of a given type can be combined"""
        combined_measurement = combine_measurements_core(*measurements)
        if isinstance(combined_measurement, dict):
            assert combined_measurement == expected
        else:
            assert qml.math.allclose(combined_measurement, expected)

    # Near 10% failure rate; need revise and fix soon
    # FIXME: [sc-95724]
    @pytest.mark.local_salt(9)
    @pytest.mark.parametrize("ml_framework", ml_frameworks_list)
    @pytest.mark.parametrize("postselect_mode", [None, "hw-like", "pad-invalid-samples"])
    def test_simulate_one_shot_native_mcm(self, ml_framework, postselect_mode, seed):
        """Unit tests for simulate_one_shot_native_mcm"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(np.pi / 4, wires=0)
            m = qml.measure(wires=0, postselect=0)
            qml.RX(np.pi / 4, wires=0)

        circuit = qml.tape.QuantumScript(q.queue, [qml.expval(qml.Z(0)), qml.sample(m)], shots=[1])

        rng = np.random.default_rng(seed)
        n_shots = 1000
        results = [
            simulate_one_shot_native_mcm(
                circuit,
                n_shots,
                interface=ml_framework,
                postselect_mode=postselect_mode,
                rng=rng,
            )
            for _ in range(n_shots)
        ]
        terminal_results, mcm_results = zip(*results)

        equivalent_tape = qml.tape.QuantumScript(
            [qml.RX(np.pi / 4, wires=0)], [qml.sample(wires=0)], shots=n_shots
        )
        expected_result = simulate(equivalent_tape, rng=rng)
        fisher_exact_test(mcm_results, expected_result)

        subset = [ts for ms, ts in zip(mcm_results, terminal_results) if ms == 0]
        equivalent_tape = qml.tape.QuantumScript(
            [qml.RX(np.pi / 4, wires=0)], [qml.expval(qml.Z(0))], shots=n_shots
        )
        expected_sample = simulate(equivalent_tape, rng=rng)
        fisher_exact_test(subset, expected_sample, outcomes=(-1, 1))

        subset = [ts for ms, ts in zip(mcm_results, terminal_results) if ms == 1]
        equivalent_tape = qml.tape.QuantumScript(
            [qml.X(0), qml.RX(np.pi / 4, wires=0)], [qml.expval(qml.Z(0))], shots=n_shots
        )
        expected_sample = simulate(equivalent_tape, rng=rng)
        fisher_exact_test(subset, expected_sample, outcomes=(-1, 1))

    def test_tree_traversal_non_standard_wire_order(self):
        """Test that tree-traversal still works with a non-standard wire order."""

        ops = [qml.H(0), qml.CNOT((0, 2)), qml.measurements.MidMeasureMP(wires=0), qml.S(1)]

        tape = qml.tape.QuantumScript(ops, [qml.expval(qml.Z(2))])
        res = simulate_tree_mcm(tape)
        assert qml.math.allclose(res, 0)

    def test_tree_traversal_sample_dtype(self):
        """Test that tree-traversal returns samples of the correct dtype (int)."""

        dev = qml.device("default.qubit")

        @qml.qnode(dev, mcm_method="tree-traversal", shots=10)
        def circuit(phi):
            qml.RX(phi, wires=0)
            m_0 = qml.measure(wires=0)
            return qml.sample([m_0])

        res = circuit(1.23)
        assert res.dtype == int
        assert res.shape == (10, 1)

    def test_measurement_on_non_op_wire(self):
        """Test that we can measure wires not present in the circuit."""

        ops = [qml.measurements.MidMeasureMP(wires=0)]
        tape = qml.tape.QuantumScript(ops, [qml.probs(wires=(0, 1, 2))])
        res = simulate_tree_mcm(tape)
        assert qml.math.allclose(res, np.array([1, 0, 0, 0, 0, 0, 0, 0]))

    def test_measurement_on_non_op_wire_with_nonstandard_order(self):
        """Test that we can measure wires not present in the circuit."""

        ops = [qml.measurements.MidMeasureMP(wires=1), qml.X(1)]
        tape = qml.tape.QuantumScript(ops, [qml.probs(wires=(0, 1, 2))])
        res = simulate_tree_mcm(tape)
        assert qml.math.allclose(res, np.array([0, 0, 1, 0, 0, 0, 0, 0]))
