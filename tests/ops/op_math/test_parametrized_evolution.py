# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the ParametrizedEvolution class
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.operation import AnyWires
from pennylane.ops import ParametrizedEvolution, ParametrizedHamiltonian


class MyOp(qml.RX):
    """Variant of qml.RX that claims to not have `adjoint` or a matrix defined."""

    has_matrix = False
    has_adjoint = False
    has_decomposition = False
    has_diagonalizing_gates = False


def compare_and_expand_mat(mat1, mat2):
    """Helper function which takes two square matrices (of potentially different sizes)
    and expands the smaller matrix until their shapes match."""

    if mat1.size == mat2.size:
        return mat1, mat2

    (smaller_mat, larger_mat, flip_order) = (
        (mat1, mat2, 0) if mat1.size < mat2.size else (mat2, mat1, 1)
    )

    while smaller_mat.size < larger_mat.size:
        smaller_mat = qml.math.cast_like(qml.math.kron(smaller_mat, qml.math.eye(2)), smaller_mat)

    if flip_order:
        return larger_mat, smaller_mat

    return smaller_mat, larger_mat


@pytest.fixture()
def param_ham():
    """Returns a parametrized hamiltonian"""
    f1 = lambda params, t: params[0] * np.polyval(params[1:5], t) + params[5]
    f2 = lambda params, t: params[6] * np.polyval(params[7:10], t) + params[10]
    return ParametrizedHamiltonian(
        [1, 2, f1, f2], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2), qml.Hadamard(3)]
    )


class TestInitialization:
    """Unit tests for the ParametrizedEvolution class."""

    def test_init(self, param_ham: ParametrizedHamiltonian):
        """Test the initialization."""
        ev = ParametrizedEvolution(H=param_ham, params=[1, 2], t=2)

        assert ev.H is param_ham
        assert ev.dt == 0.1
        assert qml.math.allequal(ev.t, [0, 2])
        assert qml.math.allequal(ev.h_params, [1, 2])

        assert ev.wires == param_ham.wires
        assert ev.num_wires == AnyWires
        assert ev.name == "ParametrizedEvolution"
        assert ev.id is None
        assert ev.queue_idx is None

        assert ev.data == []
        assert ev.parameters == []
        assert ev.num_params == 0

    def test_has_matrix_true_via_op_have_matrix(self):
        """Test that a parametrized evolution of operators that have `has_matrix=True`
        has `has_matrix=True` as well."""

        ops = [qml.PauliX(wires=0), qml.RZ(0.23, wires="a")]
        coeffs = [1, 1]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=[], t=0)

        assert ev.has_matrix is True

    def test_has_matrix_true_via_factor_has_no_matrix_but_is_hamiltonian(self):
        """Test that a product of operators of which one does not have `has_matrix=True`
        but is a Hamiltonian has `has_matrix=True`."""

        ops = [qml.Hamiltonian([0.5], [qml.PauliX(wires=1)]), qml.RZ(0.23, wires=5)]
        coeffs = [1, 1]

        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=[], t=0)

        assert ev.has_matrix is True

    @pytest.mark.parametrize(
        "first_factor", [qml.PauliX(wires=0), qml.Hamiltonian([0.5], [qml.PauliX(wires=1)])]
    )
    def test_has_matrix_false_via_factor_has_no_matrix(self, first_factor):
        """Test that a product of operators of which one does not have `has_matrix=True`
        has `has_matrix=False`."""

        ops = [first_factor, MyOp(0.23, wires="a")]
        coeffs = [1, 1]

        H = ParametrizedHamiltonian(coeffs, ops)
        ev = ParametrizedEvolution(H=H, params=[], t=0)

        assert ev.has_matrix is False


class TestMatrix:
    """Test matrix method."""

    def test_time_independent_hamiltonian(self):
        """Test matrix method for a time independent hamiltonian."""

        ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

        def f1(params, t):
            return params[0]  # constant

        def f2(params, t):
            return params[1]  # constant

        coeffs = [f1, f2, 4, 9]
        H = ParametrizedHamiltonian(coeffs, ops)
        t = 4
        params = [1, 2]
        ev = ParametrizedEvolution(H=H, params=params, t=t, dt=0.001)
        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t)) * t)
        assert qml.math.allclose(ev.matrix(), true_mat, atol=1e-2)

    @pytest.mark.jax
    def test_time_dependent_hamiltonian(self):
        """Test matrix method for a time dependent hamiltonian."""
        from jax import numpy as jnp

        ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

        def f1(params, t):
            return params[0] * t

        def f1_integral(params, t):
            return params[0] * (t**2) / 2

        def f2(params, t):
            return params[1] * jnp.cos(t)

        def f2_integral(params, t):
            return params[1] * jnp.sin(t)

        def f3(params, t):
            return 4

        def f3_integral(params, t):
            return 4 * t

        def f4(params, t):
            return 9

        def f4_integral(params, t):
            return 9 * t

        coeffs = [f1, f2, f3, f4]
        coeffs_integral = [f1_integral, f2_integral, f3_integral, f4_integral]
        H = ParametrizedHamiltonian(coeffs, ops)
        H_integral = ParametrizedHamiltonian(coeffs_integral, ops)
        t = jnp.pi / 4
        params = [1, 2]
        ev = ParametrizedEvolution(H=H, params=params, t=t, dt=1e-6)

        true_mat = qml.math.expm(
            -1j * (qml.matrix(H_integral(params, t)) - qml.matrix(H_integral(params, 0)))
        )
        assert qml.math.allclose(ev.matrix(), true_mat, atol=1e-1)
