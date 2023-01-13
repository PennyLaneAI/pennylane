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
# pylint: disable=unused-argument
import pytest

import pennylane as qml
from pennylane.operation import AnyWires
from pennylane.ops import Evolve, ParametrizedHamiltonian, QubitUnitary


class MyOp(qml.RX):  # pylint: disable=too-few-public-methods
    """Variant of qml.RX that claims to not have `adjoint` or a matrix defined."""

    has_matrix = False
    has_adjoint = False
    has_decomposition = False
    has_diagonalizing_gates = False


def time_independent_hamiltonian():
    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params[0]  # constant

    def f2(params, t):
        return params[1]  # constant

    coeffs = [f1, f2, 4, 9]

    return ParametrizedHamiltonian(coeffs, ops)


def time_dependent_hamiltonian():
    from jax import numpy as jnp

    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params[0] * t

    def f2(params, t):
        return params[1] * jnp.cos(t)

    def f3(params, t):
        return 4

    def f4(params, t):
        return 9

    coeffs = [f1, f2, f3, f4]

    def f1_integral(params, t):
        return params[0] * (t**2) / 2

    def f2_integral(params, t):
        return params[1] * jnp.sin(t)

    def f3_integral(params, t):
        return 4 * t

    def f4_integral(params, t):
        return 9 * t

    coeffs_integral = [f1_integral, f2_integral, f3_integral, f4_integral]
    H = ParametrizedHamiltonian(coeffs, ops)
    H_integral = ParametrizedHamiltonian(coeffs_integral, ops)
    return H, H_integral


class TestInitialization:
    """Unit tests for the ParametrizedEvolution class."""

    def test_init(self):
        """Test the initialization."""
        ops = [qml.PauliX(0), qml.PauliY(1)]
        coeffs = [1, 2]
        H = ParametrizedHamiltonian(coeffs, ops)
        ev = Evolve(H=H, params=[1, 2], t=2)

        assert ev.H is H
        assert ev.dt is None
        assert qml.math.allequal(ev.t, [0, 2])
        assert qml.math.allequal(ev.h_params, [1, 2])

        assert ev.wires == H.wires
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
        ev = Evolve(H=H, params=[], t=0)

        assert ev.has_matrix is True

    def test_has_matrix_true_via_factor_has_no_matrix_but_is_hamiltonian(self):
        """Test that a product of operators of which one does not have `has_matrix=True`
        but is a Hamiltonian has `has_matrix=True`."""

        ops = [qml.Hamiltonian([0.5], [qml.PauliX(wires=1)]), qml.RZ(0.23, wires=5)]
        coeffs = [1, 1]

        H = ParametrizedHamiltonian(coeffs, ops)
        ev = Evolve(H=H, params=[], t=0)

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
        ev = Evolve(H=H, params=[], t=0)

        assert ev.has_matrix is False


class TestMatrix:
    """Test matrix method."""

    # pylint: disable=unused-argument
    @pytest.mark.jax
    def test_time_independent_hamiltonian(self):
        """Test matrix method for a time independent hamiltonian."""
        H = time_independent_hamiltonian()
        t = 4
        params = [1, 2]
        ev = Evolve(H=H, params=params, t=t)
        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t)) * t)
        assert qml.math.allclose(ev.matrix(), true_mat, atol=1e-2)

    @pytest.mark.jax
    def test_time_dependent_hamiltonian(self):
        """Test matrix method for a time dependent hamiltonian."""
        from jax import numpy as jnp

        H, H_integral = time_dependent_hamiltonian()

        t = jnp.pi / 4
        params = [1, 2]
        ev = Evolve(H=H, params=params, t=t)

        true_mat = qml.math.expm(
            -1j * (qml.matrix(H_integral(params, t)) - qml.matrix(H_integral(params, 0)))
        )
        assert qml.math.allclose(ev.matrix(), true_mat, atol=1e-1)


class TestIntegration:
    """Integration tests for the ParametrizedEvolution class."""

    # pylint: disable=unused-argument
    @pytest.mark.jax
    def test_time_independent_hamiltonian(self):
        """Test matrix method for a time independent hamiltonian."""
        import jax
        from jax import numpy as jnp

        H = time_independent_hamiltonian()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(params, t):
            Evolve(H=H, params=params, t=t)
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1))

        @qml.qnode(dev, interface="jax")
        def true_circuit(params, t):
            true_mat = qml.math.expm(-1j * qml.matrix(H(params, t)) * t)
            QubitUnitary(U=true_mat, wires=[0, 1])
            return qml.expval(qml.PauliX(0) @ qml.PauliX(1))

        t = 4
        params = jnp.array([1.0, 2.0])

        assert qml.math.allclose(circuit(params, t), true_circuit(params, t), atol=1e-2)
        assert qml.math.allclose(
            jax.grad(circuit)(params, t), jax.grad(true_circuit)(params, t), atol=1e-2
        )

    @pytest.mark.jax
    def test_time_dependent_hamiltonian(self):
        """Test matrix method for a time dependent hamiltonian."""
        import jax
        import jax.numpy as jnp

        H, H_integral = time_dependent_hamiltonian()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(params, t):
            Evolve(H=H, params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        @qml.qnode(dev, interface="jax")
        def true_circuit(params, t):
            true_mat = qml.math.expm(
                -1j * (qml.matrix(H_integral(params, t)) - qml.matrix(H_integral(params, 0)))
            )
            QubitUnitary(U=true_mat, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        t = jnp.pi / 4
        params = jnp.array([1.0, 2.0])

        assert qml.math.allclose(circuit(params, t), true_circuit(params, t), atol=1e-2)
        assert qml.math.allclose(
            jax.grad(circuit)(params, t), jax.grad(true_circuit)(params, t), atol=1e-2
        )
