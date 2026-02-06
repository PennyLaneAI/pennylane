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
"""
Tests for the QSVT Optax-based iterative angle solver.

These tests require JAX and Optax to be installed and are marked as external.
"""

import jax
import jax.numpy as jnp

# pylint: disable=too-many-arguments, import-outside-toplevel, no-self-use
import pytest
from numpy.polynomial.chebyshev import Chebyshev

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines.qsvt import (
    _cheby_pol_optax,
    _grid_pts_optax,
    _poly_func_optax,
    _qsp_iterate_broadcast_optax,
    _qsp_iterate_optax,
    _qsp_optimization_optax,
    _W_of_x_optax,
    _z_rotation_optax,
)


def generate_polynomial_coeffs(degree, parity=None):
    """Generate random polynomial coefficients with specified parity."""
    rng = np.random.default_rng(seed=123)
    if parity is None:
        polynomial_coeffs_in_canonical_basis = rng.normal(size=degree + 1)
        return polynomial_coeffs_in_canonical_basis / np.sum(
            np.abs(polynomial_coeffs_in_canonical_basis)
        )
    if parity == 0:
        assert degree % 2 == 0.0
        polynomial_coeffs_in_canonical_basis = np.zeros(degree + 1)
        polynomial_coeffs_in_canonical_basis[0::2] = rng.normal(size=degree // 2 + 1)
        return polynomial_coeffs_in_canonical_basis / np.sum(
            np.abs(polynomial_coeffs_in_canonical_basis)
        )

    if parity == 1:
        assert degree % 2 == 1.0
        polynomial_coeffs_in_canonical_basis = np.zeros(degree + 1)
        polynomial_coeffs_in_canonical_basis[1::2] = rng.uniform(size=degree // 2 + 1)
        return polynomial_coeffs_in_canonical_basis / np.sum(
            np.abs(polynomial_coeffs_in_canonical_basis)
        )

    raise ValueError(f"parity must be None, 0 or 1 but got {parity}")


# All tests in this file require JAX and Optax
pytestmark = pytest.mark.external


class TestOptaxAngleSolver:
    """Tests for the Optax-based QSP angle solver (iterative_optax)."""

    @pytest.mark.parametrize(
        "poly",
        [
            (generate_polynomial_coeffs(4, 0)),
            (generate_polynomial_coeffs(3, 1)),
            (generate_polynomial_coeffs(6, 0)),
        ],
    )
    def test_correctness_QSP_angles_finding_optax(self, poly):
        """Tests that angles generate desired poly with iterative_optax solver"""
        jax.config.update("jax_enable_x64", True)

        angles = qml.poly_to_angles(list(poly), "QSP", angle_solver="iterative_optax")
        rng = np.random.default_rng(123)
        x = rng.uniform(low=-1.0, high=1.0)

        @qml.qnode(qml.device("default.qubit"))
        def circuit_qsp():
            qml.RX(2 * angles[0], wires=0)
            for angle in angles[1:]:
                qml.RZ(-2 * np.arccos(x), wires=0)
                qml.RX(2 * angle, wires=0)

            return qml.state()

        output = qml.matrix(circuit_qsp, wire_order=[0])()[0, 0]
        expected = sum(coef * (x**i) for i, coef in enumerate(poly))
        assert np.isclose(output.real, expected.real)

    @pytest.mark.parametrize(
        "poly",
        [
            (generate_polynomial_coeffs(4, 0)),
            (generate_polynomial_coeffs(3, 1)),
            (generate_polynomial_coeffs(6, 0)),
        ],
    )
    def test_correctness_QSVT_angles_optax(self, poly):
        """Tests that angles generate desired poly with iterative_optax solver"""
        jax.config.update("jax_enable_x64", True)

        angles = qml.poly_to_angles(list(poly), "QSVT", angle_solver="iterative_optax")
        rng = np.random.default_rng(123)
        x = rng.uniform(low=-1.0, high=1.0)

        block_encoding = qml.RX(-2 * np.arccos(x), wires=0)
        projectors = [qml.PCPhase(angle, dim=1, wires=0) for angle in angles]

        @qml.qnode(qml.device("default.qubit"))
        def circuit_qsvt():
            qml.QSVT(block_encoding, projectors)
            return qml.state()

        output = qml.matrix(circuit_qsvt, wire_order=[0])()[0, 0]
        expected = sum(coef * (x**i) for i, coef in enumerate(poly))
        assert qml.math.isclose(output.real, expected.real)


class TestOptaxInternalFunctions:
    """Tests for the internal Optax-based functions."""

    @pytest.mark.parametrize(
        "polynomial_coeffs_in_cheby_basis",
        [
            (generate_polynomial_coeffs(10, 0)),
            (generate_polynomial_coeffs(7, 1)),
            (generate_polynomial_coeffs(12, 0)),
        ],
    )
    def test_qsp_optimization_optax(self, polynomial_coeffs_in_cheby_basis):
        """Test that _qsp_optimization_optax returns correct angles"""
        jax.config.update("jax_enable_x64", True)

        degree = len(polynomial_coeffs_in_cheby_basis) - 1
        target_polynomial_coeffs = polynomial_coeffs_in_cheby_basis
        phis, cost_func = _qsp_optimization_optax(degree, jnp.array(target_polynomial_coeffs))

        key = jax.random.key(123)
        x_point = jax.random.uniform(key=key, shape=(1,), minval=-1, maxval=1)
        x_point = x_point.item()

        # Theorem 4: |\alpha_i-\beta_i|\leq 2\sqrt(cost_func)
        tolerance = np.sum(
            np.array(
                [
                    2 * np.sqrt(cost_func) * abs(_cheby_pol_optax(degree=i, x=x_point))
                    for i in range(len(target_polynomial_coeffs))
                ]
            )
        )

        assert qml.math.isclose(
            _qsp_iterate_broadcast_optax(phis, x_point, "jax"),
            _poly_func_optax(coeffs=jnp.array(target_polynomial_coeffs), x=x_point),
            atol=tolerance,
        )

    @pytest.mark.parametrize(
        "coeffs, x",
        [
            (generate_polynomial_coeffs(100, 0), 0.1),
            (generate_polynomial_coeffs(7, 1), 0.2),
            (generate_polynomial_coeffs(12, 0), 0.3),
            (generate_polynomial_coeffs(12, None), 0.4),
        ],
    )
    def test_poly_func_optax(self, coeffs, x):
        """Test internal function _poly_func_optax"""
        val = _poly_func_optax(coeffs=jnp.array(coeffs), x=x)
        ref = Chebyshev(coeffs)(x)
        assert np.isclose(val, ref)

    @pytest.mark.parametrize("angle", list([0.1, 0.2, 0.3, 0.4]))
    @pytest.mark.parametrize("interface", ["jax"])
    def test_z_rotation_optax(self, angle, interface):
        """Test internal function _z_rotation_optax"""
        assert np.allclose(_z_rotation_optax(angle, interface), qml.RZ.compute_matrix(-2 * angle))

    @pytest.mark.parametrize("phi", [0.1, 0.2, 0.3, 0.4])
    @pytest.mark.parametrize("interface", ["jax"])
    def test_qsp_iterate_optax(self, phi, interface):
        """Test internal function _qsp_iterate_optax"""
        mtx = _qsp_iterate_optax(0.0, phi, interface)
        ref = qml.RX.compute_matrix(-2 * np.arccos(phi))
        assert np.allclose(mtx, ref)

    @pytest.mark.parametrize(
        "x",
        list([0.1, 0.2, 0.3, 0.4]),
    )
    @pytest.mark.parametrize("degree", range(2, 6))
    def test_qsp_iterate_broadcast_optax(self, x, degree):
        """Test internal function _qsp_iterate_broadcast_optax"""
        jax.config.update("jax_enable_x64", True)

        phis = jnp.array([np.pi / 4] + [0.0] * (degree - 1) + [-np.pi / 4])
        qsp_be = _qsp_iterate_broadcast_optax(phis, x, "jax")
        ref = qml.RX.compute_matrix(-2 * (degree) * np.arccos(x))[0, 0]
        assert jnp.isclose(qsp_be, ref)

    @pytest.mark.parametrize("x", [0.1, 0.2, 0.3, 0.4])
    @pytest.mark.parametrize("interface", ["jax"])
    def test_W_of_x_optax(self, x, interface):
        """Test internal function _W_of_x_optax"""
        mtx = _W_of_x_optax(x, interface)
        ref = qml.RX.compute_matrix(-2 * np.arccos(x))
        assert np.allclose(mtx, ref)

    @pytest.mark.parametrize("degree", [4, 5, 10])
    @pytest.mark.parametrize("interface", ["jax"])
    def test_grid_pts_optax(self, degree, interface):
        """Test internal function _grid_pts_optax"""
        grid = _grid_pts_optax(degree, interface)
        # Grid points should be in [-1, 1]
        assert all(-1 <= x <= 1 for x in grid)
        # Grid points should have correct length: (degree + 1) // 2 + (degree + 1) % 2
        d = (degree + 1) // 2 + (degree + 1) % 2
        assert len(grid) == d


class TestOptaxErrorHandling:
    """Tests for error handling in the Optax-based solver."""

    def test_iterative_optax_requires_x64_mode(self):
        """Test that iterative_optax raises RuntimeError without 64-bit mode"""
        # Ensure x64 is disabled for this test
        original_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", False)

        try:
            poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3]
            with pytest.raises(RuntimeError, match="JAX must be in 64-bit mode"):
                qml.poly_to_angles(poly, "QSVT", angle_solver="iterative_optax")
        finally:
            # Restore original setting
            jax.config.update("jax_enable_x64", original_x64)

    def test_iterative_optax_import_error(self, monkeypatch):
        """Test that iterative_optax raises ImportError without jax/optax installed"""
        # Monkeypatch sys.modules to make jax import fail
        import builtins
        import sys

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "jax" or name.startswith("jax."):
                raise ImportError("No module named 'jax'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Need to also remove from sys.modules if cached
        jax_modules = [k for k in sys.modules if k == "jax" or k.startswith("jax.")]
        for mod in jax_modules:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        poly = [0, 1.0, 0, -1 / 2, 0, 1 / 3]
        with pytest.raises(ImportError, match="iterative_optax.*requires JAX and Optax"):
            # We need to reload the function that does the import
            from pennylane.templates.subroutines.qsvt import (
                _compute_qsp_angles_iteratively_optax,
            )

            _compute_qsp_angles_iteratively_optax(poly)
