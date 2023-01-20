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
Unit tests for the convenience functions used in pulsed programming.
"""
import inspect
from functools import reduce

import numpy as np
import pytest

import pennylane as qml
from pennylane.ops import ParametrizedHamiltonian


class TestConstant:
    """Unit tests for the ``constant`` function."""

    def test_constant_returns_callable(self):
        """Test that the ``constant`` convenience function returns a callable with two arguments
        corresponding to the trainable parameters and time."""
        c = qml.pulse.constant(windows=[0, 10])  # constant function from 0 to 10
        argspec = inspect.getfullargspec(c)

        assert callable(c)
        assert argspec.args == ["p", "t"]

    def test_constant_returns_correct_value_single_window(self):
        """Test that the ``constant`` function returns the input parameter only when t is inside
        the window."""
        c = qml.pulse.constant(windows=[(4, 8)])

        times = np.arange(0, 10, step=1e-2)
        for t in times:
            if 4 <= t <= 8:
                assert c(p=1, t=t) == 1
            else:
                assert c(p=1, t=t) == 0

    def test_constant_returns_correct_value_multiple_windows(self):
        """Test that the ``constant`` function returns the input parameter only when t is inside
        the window."""
        c = qml.pulse.constant(windows=[(4, 8), (0, 1), (9, 10)])

        times = np.arange(0, 10, step=1e-2)
        for t in times:
            if 4 <= t <= 8 or 0 <= t <= 1 or 9 <= t <= 10:
                assert c(p=1, t=t) == 1
            else:
                assert c(p=1, t=t) == 0

    @pytest.mark.jax
    def test_constant_is_jittable(self):
        """Test that the callable returned by the ``constant`` function is jittable."""
        import jax

        c = jax.jit(qml.pulse.constant(windows=[(4, 8), (0, 1), (9, 10)]))

        times = np.arange(0, 10, step=1e-2)
        for t in times:
            if 4 <= t <= 8 or 0 <= t <= 1 or 9 <= t <= 10:
                assert c(p=1, t=t) == 1
            else:
                assert c(p=1, t=t) == 0


class TestPiecewise:
    """Unit tests for the ``piecewise`` function."""

    def test_piecewise_returns_callable(self):
        """Test that the ``piecewise`` convenience function returns a callable with two arguments
        corresponding to the trainable parameters and time."""
        c = qml.pulse.piecewise(x=10, windows=[0, 10])  # return 10 when time is between 0 and 10
        argspec = inspect.getfullargspec(c)

        assert callable(c)
        assert argspec.args == ["p", "t"]

    def test_piecewise_returns_correct_value_single_window(self):
        """Test that the ``piecewise`` function returns the correct value only when t is inside
        the window."""
        c = qml.pulse.piecewise(x=10, windows=[(4, 8)])

        times = np.arange(0, 10, step=1e-2)
        for t in times:
            if 4 <= t <= 8:
                assert c(p=1, t=t) == 10  # p is ignored
            else:
                assert c(p=1, t=t) == 0

    def test_piecewise_returns_correct_value_multiple_windows(self):
        """Test that the ``piecewise`` function returns the correct value only when t is inside
        the window."""

        def f(p, t):
            return p * t

        c = qml.pulse.piecewise(x=f, windows=[(4, 8), (0, 1), (9, 10)])

        times = np.arange(0, 10, step=1e-2)
        param = 10
        for t in times:
            if 4 <= t <= 8 or 0 <= t <= 1 or 9 <= t <= 10:
                assert qml.math.allclose(c(p=param, t=t), f(param, t))
            else:
                assert c(p=param, t=t) == 0

    @pytest.mark.jax
    def test_piecewise_is_jittable(self):
        """Test that the callable returned by the ``piecewise`` function is jittable."""
        import jax

        def f(p, t):
            return p * t

        c = jax.jit(qml.pulse.piecewise(x=f, windows=[(4, 8), (0, 1), (9, 10)]))

        times = np.arange(0, 10, step=1e-2)
        param = 10
        for t in times:
            if 4 <= t <= 8 or 0 <= t <= 1 or 9 <= t <= 10:
                assert qml.math.allclose(c(p=param, t=t), f(param, t))
            else:
                assert c(p=param, t=t) == 0


def f1(p, t):
    return p * t


windows1 = [(0, 0.5), (1, 1.5)]
windows2 = [(0.5, 1)]

coeffs = [qml.pulse.piecewise(f1, windows1), qml.pulse.constant(windows2)]
ops = [qml.PauliX(0), qml.PauliY(1)]


@pytest.mark.jax
class TestIntegration:
    """Unit tests testing the integration of convenience functions with parametrized hamiltonians."""

    def test_parametrized_hamiltonian(self):
        """Test that convenience functions can be used to define parametrized hamiltonians."""
        H = qml.ops.dot(coeffs, ops)

        assert isinstance(H, ParametrizedHamiltonian)
        # assert that at t=4.5 both functions are 0
        assert qml.math.allequal(qml.matrix(H(params=[1, 2], t=4.5)), 0)
        # assert that at t=3 only the first coefficient is non-zero
        true_mat = qml.matrix(f1(1, 3) * qml.PauliX(0), wire_order=[0, 1])
        assert qml.math.allequal(qml.matrix(H(params=[1, 2], t=3)), true_mat)
        # assert that at t=6 only the second coefficient is non-zero
        true_mat = qml.matrix(2 * qml.PauliY(1), wire_order=[0, 1])
        assert qml.math.allequal(qml.matrix(H(params=[1, 2], t=6)), true_mat)

    @pytest.mark.slow
    @pytest.mark.jax
    def test_qnode(self):
        """Test that the evolution of a parametrized hamiltonian defined with convenience functions
        can be executed on a QNode."""
        import jax
        import jax.numpy as jnp

        H = qml.ops.dot(coeffs, ops)

        t = (1, 2)

        def generator(params):
            time_step = 1e-3
            times = jnp.arange(*t, step=time_step)
            for ti in times:
                yield jax.scipy.linalg.expm(-1j * time_step * qml.matrix(H(params, t=ti)))

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def jitted_circuit(params):
            qml.evolve(H)(params=params, t=t)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qml.qnode(dev, interface="jax")
        def true_circuit(params):
            true_mat = reduce(lambda x, y: y @ x, generator(params))
            qml.QubitUnitary(U=true_mat, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        params = jnp.array([1.0, 2.0])

        assert qml.math.allclose(circuit(params), true_circuit(params), atol=5e-3)
        assert qml.math.allclose(jitted_circuit(params), true_circuit(params), atol=5e-3)
        assert qml.math.allclose(
            jax.grad(circuit)(params), jax.grad(true_circuit)(params), atol=5e-3
        )
        assert qml.math.allclose(
            jax.grad(jitted_circuit)(params), jax.grad(true_circuit)(params), atol=5e-3
        )
