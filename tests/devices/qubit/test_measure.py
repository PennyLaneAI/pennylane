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
"""Unit tests for measure in devices/qubit."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane.devices.qubit import simulate
from pennylane.devices.qubit.measure import (
    csr_dot_products,
    full_dot_products,
    get_measurement_function,
    measure,
    state_diagonalizing_gates,
    sum_of_terms_method,
)


class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    def test_sample_based_observable(self):
        """Test sample-only measurements raise a notimplementedError."""

        state = 0.5 * np.ones((2, 2))
        with pytest.raises(NotImplementedError):
            _ = measure(qml.sample(wires=0), state)


@pytest.mark.unit
class TestMeasurementDispatch:
    """Test that get_measurement_function dispatchs to the correct place."""

    def test_state_no_obs(self):
        """Test that the correct internal function is used for a measurement process with no observables."""
        # Test a case where state_measurement_process is used
        mp1 = qml.state()
        assert get_measurement_function(mp1, state=1) == state_diagonalizing_gates

    @pytest.mark.parametrize(
        "m",
        (
            qml.var(qml.PauliZ(0)),
            qml.expval(qml.sum(qml.PauliZ(0), qml.PauliX(0))),
            qml.expval(qml.sum(*(qml.PauliX(i) for i in range(15)))),
            qml.expval(qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliZ(10))),
        ),
    )
    def test_diagonalizing_gates(self, m):
        """Test that the state_diagonalizing gates are used when there's an observable has diagonalizing
        gates and allows the measurement to be efficiently computed with them."""
        assert get_measurement_function(m, state=1) is state_diagonalizing_gates

    def test_hermitian_full_dot_product(self):
        """Test that the expectation value of a hermitian uses the full dot products method."""
        mp = qml.expval(qml.Hermitian(np.eye(2), wires=0))
        assert get_measurement_function(mp, state=1) is full_dot_products

    def test_hamiltonian_sparse_method(self):
        """Check that the sparse expectation value method is used if the state is numpy."""
        H = qml.Hamiltonian([2], [qml.PauliX(0)])
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(H), state) is csr_dot_products

    def test_hamiltonian_sum_of_terms_when_backprop(self):
        """Check that the sum of terms method is used when the state is trainable."""
        H = qml.Hamiltonian([2], [qml.PauliX(0)])
        state = qml.numpy.zeros(2)
        assert get_measurement_function(qml.expval(H), state) is sum_of_terms_method

    def test_sum_sparse_method_when_large_and_nonoverlapping(self):
        """Check that the sparse expectation value method is used if the state is numpy and
        the Sum is large with overlapping wires."""
        S = qml.prod(*(qml.PauliX(i) for i in range(8))) + qml.prod(
            *(qml.PauliY(i) for i in range(8))
        )
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(S), state) is csr_dot_products

    def test_sum_sum_of_terms_when_backprop(self):
        """Check that the sum of terms method is used when"""
        S = qml.prod(*(qml.PauliX(i) for i in range(8))) + qml.prod(
            *(qml.PauliY(i) for i in range(8))
        )
        state = qml.numpy.zeros(2)
        assert get_measurement_function(qml.expval(S), state) is sum_of_terms_method

    def test_no_sparse_matrix(self):
        """Tests Hamiltonians/Sums containing observables that do not have a sparse matrix."""

        class DummyOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods
            num_wires = 1

        S1 = qml.Hamiltonian([0.5, 0.5], [qml.X(0), DummyOp(wires=1)])
        state = np.zeros(2)
        assert get_measurement_function(qml.expval(S1), state) is sum_of_terms_method

        S2 = qml.X(0) + DummyOp(wires=1)
        assert get_measurement_function(qml.expval(S2), state) is sum_of_terms_method

        S3 = 0.5 * qml.X(0) + 0.5 * DummyOp(wires=1)
        assert get_measurement_function(qml.expval(S3), state) is sum_of_terms_method

        S4 = qml.Y(0) + qml.X(0) @ DummyOp(wires=1)
        assert get_measurement_function(qml.expval(S4), state) is sum_of_terms_method


class TestMeasurements:
    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (qml.state(), -0.5j * np.ones(4)),
            (qml.density_matrix(wires=0), 0.5 * np.ones((2, 2))),
            (qml.probs(wires=[0]), np.array([0.5, 0.5])),
        ],
    )
    def test_state_measurement_no_obs(self, measurement, expected):
        """Test that state measurements with no observable work as expected."""
        state = -0.5j * np.ones((2, 2))
        res = measure(measurement, state)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (
                qml.Hamiltonian([-0.5, 2], [qml.PauliY(0), qml.PauliZ(0)]),
                0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            ),
            (
                qml.SparseHamiltonian(
                    csr_matrix(-0.5 * qml.PauliY(0).matrix() + 2 * qml.PauliZ(0).matrix()),
                    wires=[0],
                ),
                0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            ),
            (
                qml.Hermitian(-0.5 * qml.PauliY(0).matrix() + 2 * qml.PauliZ(0).matrix(), wires=0),
                0.5 * np.sin(0.123) + 2 * np.cos(0.123),
            ),
        ],
    )
    def test_hamiltonian_expval(self, obs, expected):
        """Test that measurements of hamiltonian/ sparse hamiltonian/ hermitians work correctly."""
        # Create RX(0.123)|0> state
        state = np.array([np.cos(0.123 / 2), -1j * np.sin(0.123 / 2)])
        res = measure(qml.expval(obs), state)
        assert np.allclose(res, expected)

    def test_sum_expval_tensor_contraction(self):
        """Test that `Sum` expectation values are correct when tensor contraction
        is used for computation."""
        summands = (qml.prod(qml.PauliY(i), qml.PauliZ(i + 1)) for i in range(7))
        obs = qml.sum(*summands)
        ops = [qml.RX(0.123, wires=i) for i in range(8)]
        meas = [qml.expval(obs)]
        qs = qml.tape.QuantumScript(ops, meas)

        res = simulate(qs)
        expected = 7 * (-np.sin(0.123) * np.cos(0.123))
        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.sum(qml.PauliY(0), qml.PauliZ(0)), -np.sin(0.123) + np.cos(0.123)),
            (
                qml.sum(*(qml.PauliZ(i) for i in range(8))),
                sum(np.sin(i * np.pi / 2 + 0.123) for i in range(8)),
            ),
        ],
    )
    def test_sum_expval_eigs(self, obs, expected):
        """Test that `Sum` expectation values are correct when eigenvalues are used
        for computation."""
        ops = [qml.RX(i * np.pi / 2 + 0.123, wires=i) for i in range(8)]
        meas = [qml.expval(obs)]
        qs = qml.tape.QuantumScript(ops, meas)

        res = simulate(qs)
        assert np.allclose(res, expected)

    @pytest.mark.jax
    def test_op_math_observable_jit_compatible(self):
        import jax

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="jax")
        def qnode(t1, t2):
            return qml.expval((t1 * qml.X(0)) @ (t2 * qml.Y(1)))

        t1, t2 = 0.5, 1.0
        assert qml.math.allclose(qnode(t1, t2), jax.jit(qnode)(t1, t2))

    def test_measure_identity_no_wires(self):
        """Test that measure can handle the expectation value of identity on no wires."""

        state = np.random.random([2, 2, 2])
        out = measure(qml.measurements.ExpectationMP(qml.I()), state)
        assert qml.math.allclose(out, 1.0)

        out2 = measure(qml.measurements.ExpectationMP(2 * qml.I()), state)
        assert qml.math.allclose(out2, 2)


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    @pytest.mark.parametrize(
        "measurement, expected",
        [
            (
                qml.state(),
                np.array(
                    [
                        [0, 0, 0, 1],
                        [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
                        [1 / 2, 1 / 2, 1 / 2, 1 / 2],
                    ]
                ),
            ),
            (
                qml.density_matrix(wires=[0, 1]),
                np.array(
                    [
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                        [[1 / 2, 0, 1 / 2, 0], [0, 0, 0, 0], [1 / 2, 0, 1 / 2, 0], [0, 0, 0, 0]],
                        [
                            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                            [1 / 4, 1 / 4, 1 / 4, 1 / 4],
                        ],
                    ]
                ),
            ),
            (
                qml.probs(wires=[0, 1]),
                np.array([[0, 0, 0, 1], [1 / 2, 0, 1 / 2, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]),
            ),
            (qml.expval(qml.PauliZ(1)), np.array([-1, 1, 0])),
            (qml.var(qml.PauliZ(1)), np.array([0, 0, 1])),
        ],
    )
    def test_state_measurement(self, measurement, expected):
        """Test that broadcasting works for regular state measurements"""
        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        res = measure(measurement, state, is_state_batched=True)
        assert np.allclose(res, expected)

    def test_sparse_hamiltonian(self):
        """Test that broadcasting works for expectation values of SparseHamiltonians"""
        H = qml.Hamiltonian([2], [qml.PauliZ(1)])
        measurement = qml.expval(H)

        state = [
            np.array([[0, 0], [0, 1]]),
            np.array([[1, 0], [1, 0]]) / np.sqrt(2),
            np.array([[1, 1], [1, 1]]) / 2,
        ]
        state = np.stack(state)

        measurement_fn = get_measurement_function(measurement, state)
        assert measurement_fn is csr_dot_products

        res = measure(measurement, state, is_state_batched=True)
        expected = np.array([-2, 2, 0])
        assert np.allclose(res, expected)


class TestNaNMeasurements:
    """Tests for state vectors containing nan values."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "mp",
        [
            qml.expval(qml.PauliZ(0)),
            qml.expval(
                qml.Hamiltonian(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
            qml.expval(
                qml.dot(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
            qml.var(qml.PauliZ(0)),
            qml.var(
                qml.dot(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
        ],
    )
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch"])
    def test_nan_float_result(self, mp, interface):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((2, 2), np.nan, like=interface)
        res = measure(mp, state, is_state_batched=False)

        assert qml.math.ndim(res) == 0
        assert qml.math.isnan(res)
        assert qml.math.get_interface(res) == interface

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "mp",
        [
            qml.expval(qml.PauliZ(0)),
            qml.expval(
                qml.Hamiltonian(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
            qml.expval(
                qml.dot(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
            qml.var(qml.PauliZ(0)),
            qml.var(
                qml.dot(
                    [1.0, 2.0, 3.0, 4.0],
                    [qml.PauliZ(0) @ qml.PauliX(1), qml.PauliX(1), qml.PauliZ(1), qml.PauliY(1)],
                )
            ),
        ],
    )
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_nan_float_result_jax(self, mp, use_jit):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((2, 2), np.nan, like="jax")
        if use_jit:
            import jax

            res = jax.jit(measure, static_argnums=[0, 2])(mp, state, is_state_batched=False)
        else:
            res = measure(mp, state, is_state_batched=False)

        assert qml.math.ndim(res) == 0

        assert qml.math.isnan(res)
        assert qml.math.get_interface(res) == "jax"

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "mp", [qml.probs(wires=0), qml.probs(op=qml.PauliZ(0)), qml.probs(wires=[0, 1])]
    )
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch"])
    def test_nan_probs(self, mp, interface):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((2, 2), np.nan, like=interface)
        res = measure(mp, state, is_state_batched=False)

        assert qml.math.shape(res) == (2 ** len(mp.wires),)
        assert qml.math.all(qml.math.isnan(res))
        assert qml.math.get_interface(res) == interface

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "mp", [qml.probs(wires=0), qml.probs(op=qml.PauliZ(0)), qml.probs(wires=[0, 1])]
    )
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_nan_probs_jax(self, mp, use_jit):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full((2, 2), np.nan, like="jax")
        if use_jit:
            import jax

            res = jax.jit(measure, static_argnums=[0, 2])(mp, state, is_state_batched=False)
        else:
            res = measure(mp, state, is_state_batched=False)

        assert qml.math.shape(res) == (2 ** len(mp.wires),)
        assert qml.math.all(qml.math.isnan(res))
        assert qml.math.get_interface(res) == "jax"


class TestSumOfTermsDifferentiability:
    @staticmethod
    def f(scale, coeffs, n_wires=10, offset=0.1, convert_to_hamiltonian=False):
        ops = [qml.RX(offset + scale * i, wires=i) for i in range(n_wires)]

        if convert_to_hamiltonian:
            H = qml.Hamiltonian(
                coeffs,
                [
                    qml.prod(*(qml.PauliZ(i) for i in range(n_wires))),
                    qml.prod(*(qml.PauliY(i) for i in range(n_wires))),
                ],
            )
        else:
            t1 = qml.s_prod(coeffs[0], qml.prod(*(qml.PauliZ(i) for i in range(n_wires))))
            t2 = qml.s_prod(coeffs[1], qml.prod(*(qml.PauliY(i) for i in range(n_wires))))
            H = t1 + t2
        qs = qml.tape.QuantumScript(ops, [qml.expval(H)])
        return simulate(qs)

    @staticmethod
    def expected(scale, coeffs, n_wires=10, offset=0.1, like="numpy"):
        phase = offset + scale * qml.math.asarray(range(n_wires), like=like)
        cosines = qml.math.cos(phase)
        sines = qml.math.sin(phase)
        return coeffs[0] * qml.math.prod(cosines) + coeffs[1] * qml.math.prod(sines)

    @pytest.mark.autograd
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    @pytest.mark.parametrize(
        "coeffs",
        [
            (qml.numpy.array(2.5), qml.numpy.array(6.2)),
            (qml.numpy.array(2.5, requires_grad=False), qml.numpy.array(6.2, requires_grad=False)),
        ],
    )
    def test_autograd_backprop(self, convert_to_hamiltonian, coeffs):
        """Test that backpropagation derivatives work in autograd with hamiltonians and large sums."""
        x = qml.numpy.array(0.52)
        out = self.f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x, coeffs)
        assert qml.math.allclose(out, expected_out)

        g = qml.grad(self.f)(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_g = qml.grad(self.expected)(x, coeffs)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_jax_backprop(self, convert_to_hamiltonian, use_jit):
        """Test that backpropagation derivatives work with jax with hamiltonians and large sums."""
        import jax

        jax.config.update("jax_enable_x64", True)

        x = jax.numpy.array(0.52, dtype=jax.numpy.float64)
        coeffs = (5.2, 6.7)
        f = jax.jit(self.f, static_argnums=(1, 2, 3, 4)) if use_jit else self.f

        out = f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x, coeffs)
        assert qml.math.allclose(out, expected_out)

        g = jax.grad(f)(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_g = jax.grad(self.expected)(x, coeffs)
        assert qml.math.allclose(g, expected_g)

    @pytest.mark.torch
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_torch_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work with torch with hamiltonians and large sums."""
        import torch

        coeffs = [
            torch.tensor(9.2, requires_grad=False, dtype=torch.float64),
            torch.tensor(6.2, requires_grad=False, dtype=torch.float64),
        ]

        x = torch.tensor(-0.289, requires_grad=True, dtype=torch.float64)
        x2 = torch.tensor(-0.289, requires_grad=True, dtype=torch.float64)
        out = self.f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)
        expected_out = self.expected(x2, coeffs, like="torch")
        assert qml.math.allclose(out, expected_out)

        out.backward()
        expected_out.backward()
        assert qml.math.allclose(x.grad, x2.grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("convert_to_hamiltonian", (True, False))
    def test_tf_backprop(self, convert_to_hamiltonian):
        """Test that backpropagation derivatives work with tensorflow with hamiltonians and large sums."""
        import tensorflow as tf

        x = tf.Variable(0.5, dtype="float64")
        coeffs = [8.3, 5.7]

        with tf.GradientTape() as tape1:
            out = self.f(x, coeffs, convert_to_hamiltonian=convert_to_hamiltonian)

        with tf.GradientTape() as tape2:
            expected_out = self.expected(x, coeffs)

        assert qml.math.allclose(out, expected_out)
        g1 = tape1.gradient(out, x)
        g2 = tape2.gradient(expected_out, x)
        assert qml.math.allclose(g1, g2)
