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
"""Unit tests for the var module"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import VarianceMP


class TestVar:
    """Tests for the var function"""

    @pytest.mark.parametrize("shots", [None, 5000, [5000, 5000]])
    def test_value(self, tol, shots):
        """Test that the var function works"""

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(shots)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = [np.sin(x) ** 2, np.sin(x) ** 2] if isinstance(shots, list) else np.sin(x) ** 2
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
        self, shots, phi, tol, tol_stochastic
    ):  # pylint: disable=too-many-arguments
        """Test that variances for mid-circuit measurement values
        are correct for a single measurement value."""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            return qml.var(m0)

        atol = tol if shots is None else tol_stochastic
        expected = np.sin(phi / 2) ** 2 - np.sin(phi / 2) ** 4
        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi)
            assert np.allclose(np.array(res), expected, atol=atol, rtol=0)

    @pytest.mark.parametrize("shots", [None, 5555, [5555, 5555]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    def test_observable_is_composite_measurement_value(
        self, shots, phi, tol, tol_stochastic, seed
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for mid-circuit measurement values
        are correct for a composite measurement value."""
        dev = qml.device("default.qubit", seed=seed)

        @qml.set_shots(shots=shots)
        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RX(0.5 * phi, 1)
            m1 = qml.measure(1)
            qml.RX(2 * phi, 2)
            m2 = qml.measure(2)
            return qml.var(m0 - 2 * m1 + m2)

        @qml.qnode(dev)
        def expected_circuit(phi):
            qml.RX(phi, 0)
            qml.RX(0.5 * phi, 1)
            qml.RX(2 * phi, 2)
            return qml.probs()

        probs = expected_circuit(phi)
        # List of possible outcomes by applying the formula to the binary repr of the indices
        outcomes = np.array([0.0, 1.0, -2.0, -1.0, 1.0, 2.0, -1.0, 0.0])
        expected = (
            sum(probs[i] * outcomes[i] ** 2 for i in range(len(probs)))
            - sum(probs[i] * outcomes[i] for i in range(len(probs))) ** 2
        )

        atol = tol if shots is None else tol_stochastic
        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi)
            assert np.allclose(np.array(res), expected, atol=atol, rtol=0)

    def test_eigvals_instead_of_observable(self, seed):
        """Tests process samples with eigvals instead of observables"""

        shots = 100
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=(shots, 2)).astype(np.int64)
        expected = qml.var(qml.PauliZ(0)).process_samples(samples, [0, 1])
        assert VarianceMP(eigvals=[1, -1], wires=[0]).process_samples(samples, [0, 1]) == expected

    def test_measurement_value_list_not_allowed(self):
        """Test that measuring a list of measurement values raises an error."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)

        with pytest.raises(
            ValueError, match="qml.var does not support measuring sequences of measurements"
        ):
            _ = qml.var([m0, m1])

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_numeric_type(self, obs):
        """Test that the numeric type is correct."""
        res = qml.var(obs)
        assert res.numeric_type is float

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""
        res = qml.var(obs)
        # pylint: disable=use-implicit-booleaness-not-comparison
        assert res.shape(None, 1) == ()
        assert res.shape(100, 1) == ()

    @pytest.mark.parametrize("state", [np.array([0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0])])
    @pytest.mark.parametrize("shots", [None, 1000, [1000, 1111]])
    def test_projector_var(self, state, shots):
        """Tests that the variance of a ``Projector`` object is computed correctly."""
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.var(qml.Projector(state, wires=range(3)))

        res = circuit()
        expected = [0.25, 0.25] if isinstance(shots, list) else 0.25

        assert np.allclose(res, expected, atol=0.02, rtol=0.02)

    def test_permuted_wires(self):
        """Test that the variance of an operator with permuted wires is the same."""
        obs = qml.prod(qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10)), qml.s_prod(3, qml.PauliZ("h")))
        obs_2 = qml.prod(
            qml.s_prod(3, qml.PauliZ("h")), qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10))
        )

        dev = qml.device("default.qubit", wires=["h", 8, 10])

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.var(obs)

        @qml.qnode(dev)
        def circuit2():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.var(obs_2)

        assert circuit() == circuit2()

    @pytest.mark.parametrize(
        "wire, expected",
        [
            (0, 1.0),
            (1, 0.0),
        ],
    )
    def test_estimate_variance_with_counts(self, wire, expected):
        """Test that the variance of an observable is estimated correctly using counts."""
        counts = {"000": 100, "100": 100}

        wire_order = qml.wires.Wires((0, 1, 2))

        res = qml.var(qml.Z(wire)).process_counts(counts=counts, wire_order=wire_order)

        assert np.allclose(res, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "autograd"])
    def test_process_density_matrix_basic(self, interface):
        """Test that process_density_matrix returns correct probabilities from a maximum mixed density matrix."""
        dm = qml.math.array([[0.5, 0], [0, 0.5]], like=interface)
        dm = qml.math.cast(dm, "float64")  # Ensure dm is float64s
        wires = qml.wires.Wires(range(1))
        expected = qml.math.array([0.0], like=interface)
        expected = qml.math.cast(expected, "float64")
        var = qml.var(qml.I(0)).process_density_matrix(dm, wires)
        var = qml.math.cast(var, "float64")
        atol = 1.0e-7 if (interface in ("torch", "tensorflow")) else 1.0e-8
        assert qml.math.allclose(var, expected, atol=atol), f"Expected {expected}, got {var}"

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "autograd"])
    @pytest.mark.parametrize(
        "subset_wires, expected_var",
        [
            ([0], 1.0),  # Variance of Z on first qubit
            ([1], 0.75),  # Variance of Z on second qubit
            ([0, 1], 0.99),  # Variance of ZZ (should be zero for this state)
        ],
    )
    def test_process_density_matrix_var_subsets(self, interface, subset_wires, expected_var):
        """Test variance calculation of density matrix with subsets of wires."""
        # Define a non-trivial two-qubit density matrix
        dm = qml.math.array(
            [[0.15, 0, 0.1, 0], [0, 0.35, 0, 0.4], [0.1, 0, 0.1, 0], [0, 0.4, 0, 0.4]],
            like=interface,
        )
        dm = qml.math.cast(dm, "float64")  # Ensure dm is float64s
        expected = qml.math.array(expected_var, like=interface)
        expected = qml.math.cast(expected, "float64")
        wires = qml.wires.Wires(range(2))

        # Calculate variance for the subset of wires
        if len(subset_wires) == 1:
            var = qml.var(qml.PauliZ(subset_wires[0])).process_density_matrix(dm, wires)
        else:
            var = qml.var(
                qml.PauliZ(subset_wires[0]) @ qml.PauliZ(subset_wires[1])
            ).process_density_matrix(dm, wires)
        var = qml.math.cast(var, "float64")

        # Set tolerance based on interface
        atol = 1.0e-7 if interface == "torch" else 1.0e-8

        assert qml.math.allclose(var, expected, atol=atol), f"Expected {expected}, got {var}"
