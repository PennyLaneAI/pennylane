# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the expval module"""
import copy

import numpy as np
import pytest

import pennylane as qp
from pennylane.measurements.expval import ExpectationMP


def test_expval_identity_nowires_DQ():
    """Test that attempting to measure qp.Identity() with no wires raises an explicit error in default.qubit"""

    # temporary solution to merge https://github.com/PennyLaneAI/pennylane/pull/5106
    @qp.qnode(qp.device("default.qubit"))
    def qnode():
        return qp.expval(qp.Identity())

    with pytest.raises(NotImplementedError, match="Expectation values of qp.Identity()"):
        _ = qnode()


def test_expval_identity_nowires_LQ():
    """Test that attempting to measure qp.Identity() with no wires raises an explicit error in lightning.qubit"""

    # temporary solution to merge https://github.com/PennyLaneAI/pennylane/pull/5106
    @qp.qnode(qp.device("lightning.qubit", wires=[0]))
    def qnode():
        return qp.expval(qp.Identity())

    with pytest.raises(NotImplementedError, match="Expectation values of qp.Identity()"):
        _ = qnode()


# pylint: disable=too-many-public-methods
class TestExpval:
    """Tests for the expval function"""

    @pytest.mark.parametrize("coeffs", [1, 1j, 1 + 1j])
    def test_process_counts_dtype(self, coeffs):
        """Test that the return type of the process_counts function is correct"""
        counts = {"000": 100, "100": 100}
        wire_order = qp.wires.Wires((0, 1, 2))
        res = qp.expval(coeffs * qp.Z(1)).process_counts(counts=counts, wire_order=wire_order)
        assert np.allclose(res, coeffs)

    @pytest.mark.parametrize("coeffs", [1, 1j, 1 + 1j])
    def test_process_state_dtype(self, coeffs):
        """Test that the return type of the process_state function is correct"""
        res = qp.measurements.ExpectationMP(obs=coeffs * qp.Z(0)).process_state(
            state=[1, 0], wire_order=qp.wires.Wires(0)
        )
        assert np.allclose(res, coeffs)

    @pytest.mark.parametrize("coeffs", [1, 1j, 1 + 1j])
    @pytest.mark.parametrize(
        "state,expected",
        [
            ([[1.0, 0.0], [0.0, 0.0]], 1.0),  # Pure |0⟩ state
            ([[0.0, 0.0], [0.0, 1.0]], -1.0),  # Pure |1⟩ state
            ([[0.5, 0.5], [0.5, 0.5]], 0.0),  # Mixed state
            ([[0.75, 0.0], [0.0, 0.25]], 0.5),  # Another mixed state
        ],
    )
    def test_process_density_matrix_dtype(self, coeffs, state, expected):
        mp = ExpectationMP(obs=coeffs * qp.PauliZ(0))
        result = mp.process_density_matrix(state, wire_order=qp.wires.Wires([0]))
        assert qp.math.allclose(result, expected * coeffs)

    @pytest.mark.parametrize("coeffs", [1, 1j, 1 + 1j])
    def test_process_samples_dtype(self, coeffs, seed):
        """Test that the return type of the process_samples function is correct"""
        shots = 100
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)
        obs = coeffs * qp.PauliZ(0)
        expected = qp.expval(obs).process_samples(samples, [0, 1])
        result = ExpectationMP(obs=obs).process_samples(samples, [0, 1])
        assert qp.math.allclose(result, expected)

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    def test_value(self, tol, shots, seed):
        """Test that the expval interface works"""
        dev = qp.device("default.qubit", wires=2, seed=seed)

        @qp.set_shots(shots)
        @qp.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.expval(qp.PauliY(0))

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        atol = tol if shots is None else 0.05
        rtol = 0 if shots is None else 0.05
        assert np.allclose(res, expected, atol=atol, rtol=rtol)

        r_dtype = np.float64
        if isinstance(res, tuple):
            assert res[0].dtype == r_dtype
            assert res[1].dtype == r_dtype
        else:
            assert res.dtype == r_dtype

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    def test_observable_is_measurement_value(
        self, shots, phi, tol, tol_stochastic, seed
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for mid-circuit measurement values
        are correct for a single measurement value."""
        dev = qp.device("default.qubit", wires=2, seed=seed)

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit(phi):
            qp.RX(phi, 0)
            m0 = qp.measure(0)
            return qp.expval(m0)

        atol = tol if shots is None else tol_stochastic
        for func in [circuit, qp.defer_measurements(circuit)]:
            res = func(phi)
            assert np.allclose(np.array(res), np.sin(phi / 2) ** 2, atol=atol, rtol=0)

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    def test_observable_is_composite_measurement_value(
        self, shots, phi, tol, tol_stochastic, seed
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for mid-circuit measurement values
        are correct for a composite measurement value."""
        dev = qp.device("default.qubit", seed=seed)

        @qp.set_shots(shots=shots)
        @qp.qnode(dev)
        def circuit(phi):
            qp.RX(phi, 0)
            m0 = qp.measure(0)
            qp.RX(0.5 * phi, 1)
            m1 = qp.measure(1)
            qp.RX(2 * phi, 2)
            m2 = qp.measure(2)
            return qp.expval(m0 * m1 + m2)

        @qp.qnode(dev)
        def expected_circuit(phi):
            qp.RX(phi, 0)
            qp.RX(0.5 * phi, 1)
            qp.RX(2 * phi, 2)
            return (
                qp.expval(qp.Projector([1], 0)),
                qp.expval(qp.Projector([1], 1)),
                qp.expval(qp.Projector([1], 2)),
            )

        evals = expected_circuit(phi)
        expected = evals[0] * evals[1] + evals[2]

        atol = tol if shots is None else tol_stochastic
        for func in [circuit, qp.defer_measurements(circuit)]:
            res = func(phi)
            assert np.allclose(np.array(res), expected, atol=atol, rtol=0)

    def test_eigvals_instead_of_observable(self, seed):
        """Tests process samples with eigvals instead of observables"""

        shots = 100
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)
        expected = qp.expval(qp.PauliZ(0)).process_samples(samples, [0, 1])
        assert (
            ExpectationMP(eigvals=[1, -1], wires=[0]).process_samples(samples, [0, 1]) == expected
        )

    def test_measurement_value_list_not_allowed(self):
        """Test that measuring a list of measurement values raises an error."""
        m0 = qp.measure(0)
        m1 = qp.measure(1)

        with pytest.raises(
            ValueError, match="qp.expval does not support measuring sequences of measurements"
        ):
            _ = qp.expval([m0, m1])

    @pytest.mark.parametrize(
        "obs",
        [qp.PauliZ(0), qp.Hermitian(np.diag([1, 2]), 0), qp.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_numeric_type(self, obs):
        """Test that the numeric type is correct."""
        res = qp.expval(obs)
        assert res.numeric_type is float

    @pytest.mark.parametrize(
        "obs",
        [qp.PauliZ(0), qp.Hermitian(np.diag([1, 2]), 0), qp.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""

        res = qp.expval(obs)
        # pylint: disable=use-implicit-booleaness-not-comparison
        assert res.shape(None, 1) == ()
        assert res.shape(100, 1) == ()

    @pytest.mark.parametrize("state", [np.array([0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0])])
    @pytest.mark.parametrize("shots", [None, 1000, [1000, 1111]])
    def test_projector_expval_qnode(self, state, shots, seed):
        """Tests that the expectation of a ``Projector`` object is computed correctly for both of
        its subclasses when integrating with the ``QNode``."""
        dev = qp.device("default.qubit", wires=3, seed=seed)

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.Projector(state, wires=range(3)))

        res = circuit()
        expected = [0.5, 0.5] if isinstance(shots, list) else 0.5
        assert np.allclose(res, expected, atol=0.02, rtol=0.04)

    def test_permuted_wires(self):
        """Test that the expectation value of an operator with permuted wires is the same."""
        obs = qp.prod(qp.PauliZ(8), qp.s_prod(2, qp.PauliZ(10)), qp.s_prod(3, qp.PauliZ("h")))
        obs_2 = qp.prod(
            qp.s_prod(3, qp.PauliZ("h")), qp.PauliZ(8), qp.s_prod(2, qp.PauliZ(10))
        )

        dev = qp.device("default.qubit", wires=["h", 8, 10])

        @qp.qnode(dev)
        def circuit():
            qp.RX(1.23, wires=["h"])
            qp.RY(2.34, wires=[8])
            return qp.expval(obs)

        @qp.qnode(dev)
        def circuit2():
            qp.RX(1.23, wires=["h"])
            qp.RY(2.34, wires=[8])
            return qp.expval(obs_2)

        assert circuit() == circuit2()

    def test_copy_observable(self):
        """Test that the observable is copied if present."""
        m = qp.expval(qp.PauliX(0))
        copied_m = copy.copy(m)
        assert m.obs is not copied_m.obs
        qp.assert_equal(m.obs, copied_m.obs)

    def test_copy_eigvals(self):
        """Test that the eigvals value is just assigned to new mp without copying."""
        # pylint: disable=protected-access
        m = ExpectationMP(eigvals=[-0.5, 0.5], wires=qp.wires.Wires(0))
        copied_m = copy.copy(m)
        assert m._eigvals is copied_m._eigvals

    def test_standard_obs(self):
        """Check that the hash of an expectation value of an observable can distinguish different observables."""

        o1 = qp.prod(qp.PauliX(0), qp.PauliY(1))
        o2 = qp.prod(qp.PauliX(0), qp.PauliZ(1))

        assert qp.expval(o1).hash == qp.expval(o1).hash
        assert qp.expval(o2).hash == qp.expval(o2).hash
        assert qp.expval(o1).hash != qp.expval(o2).hash

        o3 = qp.sum(qp.PauliX("a"), qp.PauliY("b"))
        assert qp.expval(o1).hash != qp.expval(o3).hash

    def test_eigvals(self):
        """Test that the eigvals property controls the hash property."""
        m1 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qp.wires.Wires(0))
        m2 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qp.wires.Wires(0), id="something")

        assert m1.hash == m2.hash

        m3 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qp.wires.Wires(1))
        assert m1.hash != m3.hash

        m4 = ExpectationMP(eigvals=[-1, 1], wires=qp.wires.Wires(1))
        assert m1.hash != m4.hash
        assert m3.hash != m4.hash

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "state,expected",
        [
            ([1.0, 0.0], 1.0),
            ([[1.0, 0.0], [0.0, 1.0]], [1.0, -1.0]),
        ],
    )
    def test_tf_function(self, state, expected):
        """Test that tf.function does not break process_state"""
        import tensorflow as tf

        @tf.function
        def compute_expval(s):
            mp = ExpectationMP(obs=qp.PauliZ(0))
            return mp.process_state(s, wire_order=qp.wires.Wires([0]))

        state = tf.Variable(state, dtype=tf.float64)
        assert qp.math.allequal(compute_expval(state), expected)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "state,expected",
        [
            ([[1.0, 0.0], [0.0, 1.0]], [0.0]),
        ],
    )
    def test_tf_function_density_matrix(self, state, expected):
        """Test that tf.function does not break process_density_matrix"""
        import tensorflow as tf

        @tf.function
        def compute_expval(s):
            mp = ExpectationMP(obs=qp.PauliZ(0))
            return mp.process_density_matrix(s, wire_order=qp.wires.Wires([0]))

        state = tf.Variable(state, dtype=tf.float64)
        assert qp.math.allequal(compute_expval(state), expected)

    @pytest.mark.parametrize(
        "state,expected",
        [
            ([[1.0, 0.0], [0.0, 0.0]], 1.0),  # Pure |0⟩ state
            ([[0.0, 0.0], [0.0, 1.0]], -1.0),  # Pure |1⟩ state
            ([[0.5, 0.5], [0.5, 0.5]], 0.0),  # Mixed state
            ([[0.75, 0.0], [0.0, 0.25]], 0.5),  # Another mixed state
        ],
    )
    def test_process_density_matrix(self, state, expected):
        mp = ExpectationMP(obs=qp.PauliZ(0))
        result = mp.process_density_matrix(state, wire_order=qp.wires.Wires([0]))
        assert qp.math.allclose(result, expected)

    @pytest.mark.parametrize(
        "state,expected",
        [
            ([[1.0, 0.0], [0.0, 0.0]], 1.0),  # Pure |0⟩ state
            ([[0.0, 0.0], [0.0, 1.0]], 1.0),  # Pure |1⟩ state
            ([[0.5, 0.0], [0.0, 0.5]], 1.0),  # Mixed state
        ],
    )
    def test_expval_process_density_matrix_no_wires(self, state, expected):
        """Test process_density_matrix method with an identity operator in the observable."""

        mp = ExpectationMP(obs=qp.I())
        # Run the circuit
        result = mp.process_density_matrix(state, wire_order=qp.wires.Wires([0]))
        assert np.allclose(result, expected)

    def test_batched_hamiltonian(self, seed):
        """Test that the expval interface works"""
        dev = qp.device("default.qubit")
        ops = (qp.Hadamard(0), qp.PauliZ(0) @ qp.PauliY(1) @ qp.PauliY(2) @ qp.PauliX(3))
        H = qp.Hamiltonian([0.5, 1.0], ops)

        @qp.qnode(dev, interface="autograd", diff_method="parameter-shift")
        def cost_circuit(params):
            qp.RX(params, 0)
            qp.CNOT([0, 1])
            return qp.expval(H)

        rng = np.random.default_rng(seed)
        params = rng.normal(0, np.pi, 4)
        energy = [cost_circuit(p) for p in params]
        energy_batched = cost_circuit(params)

        assert qp.math.allequal(energy_batched, energy)

    @pytest.mark.parametrize(
        "wire, expected",
        [
            (0, 0.0),
            (1, 1.0),
        ],
    )
    def test_estimate_expectation_with_counts(self, wire, expected):
        """Test that the expectation value of an observable is estimated correctly using counts"""
        counts = {"000": 100, "100": 100}

        wire_order = qp.wires.Wires((0, 1, 2))

        res = qp.expval(qp.Z(wire)).process_counts(counts=counts, wire_order=wire_order)

        assert np.allclose(res, expected)


@pytest.mark.parametrize("coeffs", [1, 1j, 1 + 1j])
def test_qnode_expval_dtype(coeffs):
    """System level test to ensure dtype is correctly preserved."""

    @qp.qnode(qp.device("default.qubit"))
    def circuit(coeffs):
        return qp.expval(coeffs * qp.PauliZ(0))

    res = circuit(coeffs)
    assert np.allclose(res, coeffs)
