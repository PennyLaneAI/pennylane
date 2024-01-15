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
"""Unit tests for measuring states in devices/qutrit_mixed."""

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import math

# from pennylane.devices.qutrit_mixed. import measure # TODO add when added to __init__
from pennylane.devices.qutrit_mixed.measure import (
    measure,
    apply_observable_einsum,
    get_measurement_function,
    trace_method,
    sum_of_terms_method,
    reduce_density_matrix,
)

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]
BATCH_SIZE = 2


class TestCurrentlyUnsupportedCases:
    # pylint: disable=too-few-public-methods
    @pytest.mark.parametrize(
        "m",
        (
            qml.var(qml.GellMann(0, 3)),  # TODO add variance
            qml.sample(wires=0),
            qml.probs(op=qml.GellMann(0, 3)),
        ),
    )
    def test_sample_based_observable(self, m, two_qutrit_state):
        """Test sample-only measurements raise a notimplementedError."""
        with pytest.raises(NotImplementedError):
            _ = measure(m, two_qutrit_state)


class TestMeasurementDispatch:
    """Test that get_measurement_function dispatchs to the correct place."""

    def test_state_no_obs(self):
        """Test that the correct internal function is used for a measurement process with no observables."""
        # Test a case where state_measurement_process is used
        mp1 = qml.state()
        assert get_measurement_function(mp1, state=1) is reduce_density_matrix

    def test_prod_trace_method(self):
        """Test that the expectation value of a product uses the trace method."""
        prod = qml.prod(*(qml.GellMann(i, 1) for i in range(8)))
        assert get_measurement_function(qml.expval(prod), state=1) is trace_method

    def test_hermitian_trace(self):
        """Test that the expectation value of a hermitian uses the trace method."""
        mp = qml.expval(qml.THermitian(np.eye(3), wires=0))
        assert get_measurement_function(mp, state=1) is trace_method

    def test_hamiltonian_sum_of_terms(self):
        """Check that the sum of terms method is used when Hamiltonian."""
        H = qml.Hamiltonian([2], [qml.GellMann(0, 1)])
        state = qml.numpy.zeros((3, 3))
        assert get_measurement_function(qml.expval(H), state) is sum_of_terms_method

    def test_sum_sum_of_terms(self):
        """Check that the sum of terms method is used when sum of terms"""
        S = qml.prod(*(qml.GellMann(i, 1) for i in range(8))) + qml.prod(
            *(qml.GellMann(i, 2) for i in range(8))
        )
        state = qml.numpy.zeros((3, 3))
        assert get_measurement_function(qml.expval(S), state) is sum_of_terms_method


@pytest.mark.parametrize("obs", [qml.GellMann(2, 3)])  # TODO add more observables
@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestApplyObservableEinsum:
    """Tests that observables are applied correctly for trace method."""

    num_qutrits = 3
    dims = (3**num_qutrits, 3**num_qutrits)

    def test_apply_observable_einsum(self, obs, three_qutrit_state, ml_framework):
        """Tests that unbatched observables are applied correctly to a unbatched state."""
        res = apply_observable_einsum(
            obs, qml.math.asarray(three_qutrit_state, like=ml_framework), is_state_batched=False
        )
        missing_wires = self.num_qutrits - len(obs.wires)
        mat = obs.matrix()
        expanded_mat = np.kron(np.eye(3**missing_wires), mat) if missing_wires else mat

        flattened_state = three_qutrit_state.reshape(self.dims)
        expected = (expanded_mat @ flattened_state).reshape([3] * (self.num_qutrits * 2))

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_apply_observable_einsum_batched(self, obs, three_qutrit_batched_state, ml_framework):
        """Tests that unbatched observables are applied correctly to a batched state."""
        res = apply_observable_einsum(
            obs,
            qml.math.asarray(three_qutrit_batched_state, like=ml_framework),
            is_state_batched=True,
        )
        missing_wires = self.num_qutrits - len(obs.wires)
        mat = obs.matrix()
        expanded_mat = np.kron(np.eye(3**missing_wires), mat) if missing_wires else mat
        expected = []

        for i in range(BATCH_SIZE):
            flattened_state = three_qutrit_batched_state[i].reshape(self.dims)
            expected.append((expanded_mat @ flattened_state).reshape([3] * (self.num_qutrits * 2)))

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)


class TestMeasurements:
    """Test that measurements on unbatched states work as expected."""

    @pytest.mark.parametrize(
        "measurement, get_expected",
        [
            (qml.state(), lambda x: math.reshape(x, newshape=(9, 9))),
            (qml.density_matrix(wires=0), lambda x: math.trace(x, axis1=1, axis2=3)),
            (
                qml.probs(wires=[0]),
                lambda x: math.real(math.diag(math.trace(x, axis1=1, axis2=3))),
            ),
        ],
    )
    def test_state_measurement_no_obs(self, measurement, get_expected, two_qutrit_state):
        """Test that state measurements with no observable work as expected."""
        res = measure(measurement, two_qutrit_state)
        expected = get_expected(two_qutrit_state)

        assert np.allclose(res, expected)

    def test_hamiltonian_expval(self, two_qutrit_state):
        """Test that measurements of hamiltonian work correctly."""
        coeffs = [-0.5, 2]
        observables = [qml.GellMann(0, 7), qml.GellMann(0, 8)]

        obs = qml.Hamiltonian(coeffs, observables)
        res = measure(qml.expval(obs), two_qutrit_state)

        flattened_state = two_qutrit_state.reshape(9, 9)

        missing_wires = 2 - len(obs.wires)
        expected = 0
        for i, coeff in enumerate(coeffs):
            mat = observables[i].matrix()
            expanded_mat = np.kron(mat, np.eye(3**missing_wires)) if missing_wires else mat
            expected += coeff * math.trace(expanded_mat @ flattened_state)

        assert np.isclose(res, expected)

    def test_hermitian_expval(self, two_qutrit_state):
        """Test that measurements of ternary hermitian work correctly."""
        obs = qml.THermitian(
            -0.5 * qml.GellMann(0, 7).matrix() + 2 * qml.GellMann(0, 8).matrix(), wires=0
        )
        res = measure(qml.expval(obs), two_qutrit_state)
        flattened_state = two_qutrit_state.reshape(9, 9)

        missing_wires = 2 - len(obs.wires)
        mat = obs.matrix()
        expanded_mat = np.kron(mat, np.eye(3**missing_wires)) if missing_wires else mat

        expected = math.trace(expanded_mat @ flattened_state)

        assert np.isclose(res, expected)

    def test_sum_expval_tensor_contraction(self):
        """Test that `Sum` expectation values are correct when tensor contraction
        is used for computation."""
        summands = (qml.prod(qml.GellMann(i, 2), qml.GellMann(i + 1, 3)) for i in range(4))
        obs = qml.sum(*summands)

        @qml.qnode(qml.device("default.qutrit", wires=5))
        def find_state(x):
            for i in range(5):
                qml.TRX(x, wires=i)
            return qml.state()

        rots = [0.123, 0.321]
        schmidts = [0.7, 0.3]
        state = np.zeros([3] * (2 * 5), dtype=complex)

        for i in range(2):
            vec = find_state(rots[i])
            state += schmidts[i] * np.outer(vec, np.conj(vec)).reshape([3] * (2 * 5))

        res = measure(qml.expval(obs), state)
        expect_from_rot = lambda x: 4 * (-np.sin(x) * np.cos(x))
        expected = schmidts[0] * expect_from_rot(rots[0]) + schmidts[1] * expect_from_rot(rots[1])
        assert np.allclose(res, expected)


class TestBroadcasting:
    """Test that measurements work when the state has a batch dim"""

    num_qutrits = 2

    @pytest.mark.parametrize(
        "measurement, get_expected",
        [
            (qml.state(), lambda x: math.reshape(x, newshape=(BATCH_SIZE, 9, 9))),
            (
                qml.density_matrix(wires=[0, 1]),
                lambda x: math.reshape(x, newshape=(BATCH_SIZE, 9, 9)),
            ),
            (qml.density_matrix(wires=[1]), lambda x: math.trace(x, axis1=1, axis2=3)),
        ],
    )
    def test_state_measurement(self, measurement, get_expected, two_qutrit_batched_state):
        """Test that state measurements work on broadcasted state"""

        res = measure(measurement, two_qutrit_batched_state, is_state_batched=True)
        expected = get_expected(two_qutrit_batched_state)

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "measurement, matrix_transform",
        [
            (qml.probs(wires=[0, 1]), lambda x: math.reshape(x, (2, 9, 9))),
            (qml.probs(wires=[0]), lambda x: math.trace(x, axis1=2, axis2=4)),
        ],
    )
    def test_probs_measurement(self, measurement, matrix_transform, two_qutrit_batched_state):
        """Test that probability measurements work on broadcasted state"""
        res = measure(measurement, two_qutrit_batched_state, is_state_batched=True)

        transformed_state = matrix_transform(two_qutrit_batched_state)

        expected = []
        for i in range(BATCH_SIZE):
            expected.append(math.diag(transformed_state[i]))

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "observable",
        [
            qml.GellMann(1, 3),
            qml.prod(qml.GellMann(0, 2), qml.GellMann(1, 3)),
            qml.THermitian(
                np.array(
                    [
                        [1.37770247 + 0.0j, 0.60335894 - 0.10889947j, 0.98223403 - 0.94429544j],
                        [0.60335894 + 0.10889947j, 0.90178212 + 0.0j, 0.45529663 - 0.03054001j],
                        [0.98223403 + 0.94429544j, 0.45529663 + 0.03054001j, 0.37721683 + 0.0j],
                    ]
                ),
                wires=1,
            ),
        ],
    )
    def test_expval_measurement(self, observable, two_qutrit_batched_state):
        """Test that expval measurements work on broadcasted state"""
        res = measure(qml.expval(observable), two_qutrit_batched_state, is_state_batched=True)

        missing_wires = 2 - len(observable.wires)
        mat = observable.matrix()
        expanded_mat = np.kron(np.eye(3**missing_wires), mat) if missing_wires else mat

        expected = []
        for i in range(BATCH_SIZE):
            resquared_matrix = two_qutrit_batched_state[i].reshape(9, 9)
            obs_mult_state = expanded_mat @ resquared_matrix
            expected.append(math.trace(obs_mult_state))

        assert qml.math.allclose(res, expected)

    def test_expval_sum_measurement(self, two_qutrit_batched_state):
        """Test that expval Sum measurements work on broadcasted state"""
        observable = qml.sum((2 * qml.GellMann(1, 1)), (0.4 * qml.GellMann(0, 6)))
        res = measure(qml.expval(observable), two_qutrit_batched_state, is_state_batched=True)

        expanded_mat = np.zeros((9, 9), dtype=complex)
        for summand in observable:
            mat = summand.matrix()
            expanded_mat += (
                np.kron(np.eye(3), mat) if summand.wires[0] == 1 else np.kron(mat, np.eye(3))
            )

        expected = []
        for i in range(BATCH_SIZE):
            resquared_matrix = two_qutrit_batched_state[i].reshape(9, 9)
            obs_mult_state = expanded_mat @ resquared_matrix
            expected.append(math.trace(obs_mult_state))

        assert qml.math.allclose(res, expected)

    def test_expval_hamiltonian_measurement(self, two_qutrit_batched_state):
        """Test that expval Hamiltonian measurements work on broadcasted state"""
        observables = [qml.GellMann(1, 1), qml.GellMann(0, 6)]
        coeffs = [2, 0.4]
        observable = qml.Hamiltonian(coeffs, observables)
        res = measure(qml.expval(observable), two_qutrit_batched_state, is_state_batched=True)

        expanded_mat = np.zeros((9, 9), dtype=complex)
        for coeff, summand in zip(coeffs, observables):
            mat = summand.matrix()
            expanded_mat += coeff * (
                np.kron(np.eye(3), mat) if summand.wires[0] == 1 else np.kron(mat, np.eye(3))
            )

        expected = []
        for i in range(BATCH_SIZE):
            resquared_matrix = two_qutrit_batched_state[i].reshape(9, 9)
            obs_mult_state = expanded_mat @ resquared_matrix
            expected.append(math.trace(obs_mult_state))

        assert qml.math.allclose(res, expected)

    # TODO add test for var once implemented


obs_list = [
    qml.GellMann(0, 3) @ qml.GellMann(1, 1),
    qml.GellMann(1, 6),
    qml.GellMann(1, 8),
    qml.GellMann(1, 7),
]
measurement_processes = [
    qml.expval(qml.GellMann(0, 3)),
    qml.expval(
        qml.Hamiltonian(
            [1.0, 2.0, 3.0, 4.0],
            obs_list,
        )
    ),
    qml.expval(
        qml.dot(
            [1.0, 2.0, 3.0, 4.0],
            obs_list,
        )
    ),
    # qml.var(qml.GellMann(0, 3)),
    # qml.var(
    #     qml.dot(
    #         [1.0, 2.0, 3.0, 4.0],
    #         obs_list,
    #     )
    # ),
]
probs_processes = "mp", [qml.probs(wires=0), qml.probs(wires=[0, 1])]


class TestNaNMeasurements:
    """Tests for mixed state matrices containing nan values."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("mp", measurement_processes)
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "tensorflow"])
    def test_nan_float_result(self, mp, interface):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full([3] * 4, np.NaN, like=interface)
        res = measure(mp, state, is_state_batched=False)

        assert qml.math.ndim(res) == 0
        assert qml.math.isnan(res)
        assert qml.math.get_interface(res) == interface

    @pytest.mark.jax
    @pytest.mark.parametrize("mp", measurement_processes)
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_nan_float_result_jax(self, mp, use_jit):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full([3] * 4, np.NaN, like="jax")
        if use_jit:
            import jax

            res = jax.jit(measure, static_argnums=[0, 2])(mp, state, is_state_batched=False)
        else:
            res = measure(mp, state, is_state_batched=False)

        assert qml.math.ndim(res) == 0

        assert qml.math.isnan(res)
        assert qml.math.get_interface(res) == "jax"

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("mp", probs_processes)
    @pytest.mark.parametrize("interface", ["numpy", "autograd", "torch", "tensorflow"])
    def test_nan_probs(self, mp, interface):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full([3] * 4, np.NaN, like=interface)
        res = measure(mp, state, is_state_batched=False)

        assert qml.math.shape(res) == (3 ** len(mp.wires),)
        assert qml.math.all(qml.math.isnan(res))
        assert qml.math.get_interface(res) == interface

    @pytest.mark.jax
    @pytest.mark.parametrize("mp", probs_processes)
    @pytest.mark.parametrize("use_jit", [True, False])
    def test_nan_probs_jax(self, mp, use_jit):
        """Test that the result of circuits with 0 probability postselections is NaN with the
        expected shape."""
        state = qml.math.full([3] * 4, np.NaN, like="jax")
        if use_jit:
            import jax

            res = jax.jit(measure, static_argnums=[0, 2])(mp, state, is_state_batched=False)
        else:
            res = measure(mp, state, is_state_batched=False)

        assert qml.math.shape(res) == (3 ** len(mp.wires),)
        assert qml.math.all(qml.math.isnan(res))
        assert qml.math.get_interface(res) == "jax"


# TODO TestSumOfTermsDifferentiability in future PR (with other differentiabilty tests)
