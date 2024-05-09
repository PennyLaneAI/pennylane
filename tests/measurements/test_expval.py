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

import pennylane as qml
from pennylane.measurements import Expectation, Shots
from pennylane.measurements.expval import ExpectationMP


def test_expval_identity_nowires_DQ():
    """Test that attempting to measure qml.Identity() with no wires raises an explicit error in default.qubit"""

    # temporary solution to merge https://github.com/PennyLaneAI/pennylane/pull/5106
    @qml.qnode(qml.device("default.qubit"))
    def qnode():
        return qml.expval(qml.Identity())

    with pytest.raises(NotImplementedError, match="Expectation values of qml.Identity()"):
        _ = qnode()


def test_expval_identity_nowires_LQ():
    """Test that attempting to measure qml.Identity() with no wires raises an explicit error in lightning.qubit"""

    # temporary solution to merge https://github.com/PennyLaneAI/pennylane/pull/5106
    @qml.qnode(qml.device("lightning.qubit", wires=[0]))
    def qnode():
        return qml.expval(qml.Identity())

    with pytest.raises(NotImplementedError, match="Expectation values of qml.Identity()"):
        _ = qnode()


class TestExpval:
    """Tests for the expval function"""

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    def test_value(self, tol, shots):
        """Test that the expval interface works"""
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

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

    def test_observable_return_type_is_expectation(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        circuit()

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    def test_observable_is_measurement_value(
        self, shots, phi, tol, tol_stochastic
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for mid-circuit measurement values
        are correct for a single measurement value."""
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            return qml.expval(m0)

        atol = tol if shots is None else tol_stochastic
        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi)
            assert np.allclose(np.array(res), np.sin(phi / 2) ** 2, atol=atol, rtol=0)

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    def test_observable_is_composite_measurement_value(
        self, shots, phi, tol, tol_stochastic
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for mid-circuit measurement values
        are correct for a composite measurement value."""
        np.random.seed(0)
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RX(0.5 * phi, 1)
            m1 = qml.measure(1)
            qml.RX(2 * phi, 2)
            m2 = qml.measure(2)
            return qml.expval(m0 * m1 + m2)

        @qml.qnode(dev)
        def expected_circuit(phi):
            qml.RX(phi, 0)
            qml.RX(0.5 * phi, 1)
            qml.RX(2 * phi, 2)
            return (
                qml.expval(qml.Projector([1], 0)),
                qml.expval(qml.Projector([1], 1)),
                qml.expval(qml.Projector([1], 2)),
            )

        evals = expected_circuit(phi)
        expected = evals[0] * evals[1] + evals[2]

        atol = tol if shots is None else tol_stochastic
        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi, shots=shots)
            assert np.allclose(np.array(res), expected, atol=atol, rtol=0)

    def test_eigvals_instead_of_observable(self):
        """Tests process samples with eigvals instead of observables"""

        shots = 100
        samples = np.random.choice([0, 1], size=(shots, 2)).astype(np.int64)
        expected = qml.expval(qml.PauliZ(0)).process_samples(samples, [0, 1])
        assert (
            ExpectationMP(eigvals=[1, -1], wires=[0]).process_samples(samples, [0, 1]) == expected
        )

    def test_measurement_value_list_not_allowed(self):
        """Test that measuring a list of measurement values raises an error."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)

        with pytest.raises(
            ValueError, match="qml.expval does not support measuring sequences of measurements"
        ):
            _ = qml.expval([m0, m1])

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_numeric_type(self, obs):
        """Test that the numeric type is correct."""
        res = qml.expval(obs)
        assert res.numeric_type is float

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""
        dev = qml.device("default.qubit", wires=1)

        res = qml.expval(obs)
        # pylint: disable=use-implicit-booleaness-not-comparison
        assert res.shape(dev, Shots(None)) == ()
        assert res.shape(dev, Shots(100)) == ()

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape_shot_vector(self, obs):
        """Test that the shape is correct with the shot vector too."""
        res = qml.expval(obs)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)
        assert res.shape(dev, Shots(shot_vector)) == ((), (), ())

    @pytest.mark.parametrize("state", [np.array([0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0])])
    @pytest.mark.parametrize("shots", [None, 1000, [1000, 1111]])
    def test_projector_expval(self, state, shots):
        """Tests that the expectation of a ``Projector`` object is computed correctly for both of
        its subclasses."""
        dev = qml.device("default.qubit", wires=3, shots=shots)
        np.random.seed(42)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Projector(state, wires=range(3)))

        res = circuit()
        expected = [0.5, 0.5] if isinstance(shots, list) else 0.5
        assert np.allclose(res, expected, atol=0.02, rtol=0.02)

    def test_permuted_wires(self):
        """Test that the expectation value of an operator with permuted wires is the same."""
        obs = qml.prod(qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10)), qml.s_prod(3, qml.PauliZ("h")))
        obs_2 = qml.prod(
            qml.s_prod(3, qml.PauliZ("h")), qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10))
        )

        dev = qml.device("default.qubit", wires=["h", 8, 10])

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.expval(obs)

        @qml.qnode(dev)
        def circuit2():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.expval(obs_2)

        assert circuit() == circuit2()

    def test_copy_observable(self):
        """Test that the observable is copied if present."""
        m = qml.expval(qml.PauliX(0))
        copied_m = copy.copy(m)
        assert m.obs is not copied_m.obs
        assert qml.equal(m.obs, copied_m.obs)

    def test_copy_eigvals(self):
        """Test that the eigvals value is just assigned to new mp without copying."""
        # pylint: disable=protected-access
        m = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(0))
        copied_m = copy.copy(m)
        assert m._eigvals is copied_m._eigvals

    def test_standard_obs(self):
        """Check that the hash of an expectation value of an observable can distinguish different observables."""

        o1 = qml.prod(qml.PauliX(0), qml.PauliY(1))
        o2 = qml.prod(qml.PauliX(0), qml.PauliZ(1))

        assert qml.expval(o1).hash == qml.expval(o1).hash
        assert qml.expval(o2).hash == qml.expval(o2).hash
        assert qml.expval(o1).hash != qml.expval(o2).hash

        o3 = qml.sum(qml.PauliX("a"), qml.PauliY("b"))
        assert qml.expval(o1).hash != qml.expval(o3).hash

    def test_eigvals(self):
        """Test that the eigvals property controls the hash property."""
        m1 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(0))
        m2 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(0), id="something")

        assert m1.hash == m2.hash

        m3 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(1))
        assert m1.hash != m3.hash

        m4 = ExpectationMP(eigvals=[-1, 1], wires=qml.wires.Wires(1))
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
            mp = ExpectationMP(obs=qml.PauliZ(0))
            return mp.process_state(s, wire_order=qml.wires.Wires([0]))

        state = tf.Variable(state, dtype=tf.float64)
        assert qml.math.allequal(compute_expval(state), expected)

    def test_batched_hamiltonian(self):
        """Test that the expval interface works"""
        dev = qml.device("default.qubit")
        ops = (qml.Hadamard(0), qml.PauliZ(0) @ qml.PauliY(1) @ qml.PauliY(2) @ qml.PauliX(3))
        H = qml.Hamiltonian([0.5, 1.0], ops)

        @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
        def cost_circuit(params):
            qml.RX(params, 0)
            qml.CNOT([0, 1])
            return qml.expval(H)

        rng = np.random.default_rng(42)
        params = rng.normal(0, np.pi, 4)
        energy = [cost_circuit(p) for p in params]
        energy_batched = cost_circuit(params)

        assert qml.math.allequal(energy_batched, energy)

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

        wire_order = qml.wires.Wires((0, 1, 2))

        res = qml.expval(qml.Z(wire)).process_counts(counts=counts, wire_order=wire_order)

        assert np.allclose(res, expected)
