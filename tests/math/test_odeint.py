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
Integration tests for odeint (ordinary differential equation integrator)
"""
import pytest
import pennylane as qml
import scipy

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def toscipy(f):
    """Convenience to go from scipy convention f(t, y) to jax/qml convention f(y, t)"""

    def wrapper(t, y):
        return f(y, t)

    return wrapper


def scipyodeint(fun, y0, t):
    """Convenience to solve ODEs with scipy with the same signature"""
    res = scipy.integrate.solve_ivp(toscipy(fun), [t[0], t[-1]], y0)
    return res.y[:, -1]


def jaxode(fun, y0, t, *args):
    """Convenience to solve ODEs with jax with the same signature"""
    from jax.experimental.ode import odeint as jaxodeint

    return jaxodeint(fun, y0, jnp.array([t[0], t[-1]]), *args)[-1]

# pylint: disable=too-few-public-methods
class TestUnitTest:
    """Unit tests for odeint"""

    # TODO catch warnings to avoid annoying outputs (or reduce tolerance)
    def testInputOutputShapes(self):
        # Unit tests
        t = jnp.linspace(0, 1, 3)

        def fun(y, _):
            return y

        y0 = jnp.ones((5))
        y1 = qml.math.odeint(fun, y0, t)
        assert qml.math.allequal(y0.shape, y1.shape)

        y0 = jnp.ones((5, 5))
        y1 = qml.math.odeint(fun, y0, t)
        assert qml.math.allequal(y0.shape, y1.shape)

        y0 = jnp.ones((5, 5, 5))
        y1 = qml.math.odeint(fun, y0, t)
        assert qml.math.allequal(y0.shape, y1.shape)


class TestAnalyticODE:
    """Test ODEs with analytic solutions"""

    def test_ODE_f_equal_y(self):
        def fun(y, _):
            return y

        t = jnp.linspace(0, 1, 20)
        y0 = jnp.array([1.0])
        res_qml = qml.math.odeint(fun, y0, t)

        # analytic solution: y(t) = exp(t)
        assert qml.math.allclose(res_qml, jnp.exp(1.0))

    def test_ODE_f_equal_y_squared(self):
        def fun(y, _):
            return y**2

        t = jnp.linspace(0, 0.5, 20)
        y0 = jnp.array([1.0])
        res_qml = qml.math.odeint(fun, y0, t)

        # analytic solution y(t) = 1/(1-t)
        print(qml.math.allclose(res_qml, 2))


# Preparing some constant Hamiltonians for testing
Hpaulis = [
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2),
    qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2),
]

H1 = qml.Hamiltonian(jnp.ones(5), [qml.PauliX(i) @ qml.PauliX((i + 1) % 10) for i in range(5)])
H2 = qml.Hamiltonian(jnp.ones(5), [qml.PauliY(i) @ qml.PauliY((i + 1) % 10) for i in range(5)])
H3 = qml.Hamiltonian(jnp.ones(5), [qml.PauliZ(i) @ qml.PauliZ((i + 1) % 10) for i in range(5)])
H4 = H1 + H2
H5 = H1 + H3
H6 = H2 + H3
H7 = H1 + H2 + H3

Hs = Hpaulis + [H1, H2, H3, H4, H5, H6, H7]
for i, op in enumerate(Hs):
    Hs[i] = jnp.array(qml.matrix(op))


class TestODESchrodingerEquation:
    """Test Schrodinger equation ODEs with constant time-dependence"""

    @pytest.mark.parametrize("H", Hs)
    def testConstantHamiltonian(self, H):
        """Test that constant Hamiltonians are correctly integrated"""
        t = jnp.linspace(0, 1, 200)  # corresponds to dt=0.005
        y0 = jnp.eye(len(H), dtype=complex)

        def fun(y, _):
            return -1j * H @ y

        res_expm = jax.scipy.linalg.expm(-1j * H)
        res_qml = qml.math.odeint(fun, y0, t)
        res_jax = jaxode(fun, y0, t)

        assert qml.math.allclose(res_expm, res_qml, rtol=1e-4)
        assert qml.math.allclose(res_qml, res_jax, rtol=1e-4)

    @pytest.mark.parametrize("H", Hs)
    def testTimedependentHamiltonian(self, H):
        """Test that time dependent Hamiltonians are correctly integrated"""

        t = jnp.linspace(0, 1, 500)  # corresponds to dt=0.005
        y0 = jnp.eye(len(H), dtype=complex)

        def fun(y, t, params):
            return -1j * (params[0] * jnp.sin(t) * H) @ y

        params = jnp.ones(1)

        res_qml = qml.math.odeint(fun, y0, t, params)
        res_jax = jaxode(fun, y0, t, params)

        assert qml.math.allclose(res_qml, res_jax, rtol=1e-4)
