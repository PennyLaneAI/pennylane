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
# pylint: disable=import-outside-toplevel
import inspect
from functools import reduce

import numpy as np
import pytest

import pennylane as qml
from pennylane.pulse import ParametrizedHamiltonian


def test_error_raised_if_jax_not_installed():
    """Test that an error is raised if a convenience function is called without jax installed"""
    try:
        import jax  # pylint: disable=unused-import

        pytest.skip()
    except ImportError:
        with pytest.raises(ImportError, match="Module jax is required"):
            qml.pulse.rect(x=10, windows=[(2, 8)])


@pytest.mark.jax
class TestConstant:
    """Unit tests for the ``constant`` function."""

    def test_constant_signature(self):
        """Test that the ``constant`` convenience function returns a callable with two arguments
        corresponding to the trainable parameters and time."""
        argspec = inspect.getfullargspec(qml.pulse.constant)

        assert callable(qml.pulse.constant)
        assert argspec.args == ["scalar", "time"]

    def test_constant_returns_correct_value(self):
        """Test that the ``constant`` function returns the input parameter only when t is inside
        the window."""
        times = np.arange(0, 10, step=1e-2)
        scalar = 1.23
        for t in times:
            assert qml.pulse.constant(scalar, time=t) == scalar

    def test_constant_is_jittable(self):
        """Test that the callable returned by the ``constant`` function is jittable."""
        import jax

        c = jax.jit(qml.pulse.constant)

        scalar = 1.23
        times = np.arange(0, 10, step=1e-2)
        for t in times:
            assert c(scalar, time=t) == scalar


@pytest.mark.jax
class TestRect:
    """Unit tests for the ``rect`` function."""

    def test_rect_returns_callable(self):
        """Test that the ``rect`` convenience function returns a callable with two arguments
        corresponding to the trainable parameters and time."""
        c = qml.pulse.rect(x=10, windows=[0, 10])  # return 10 when time is between 0 and 10
        argspec = inspect.getfullargspec(c)

        assert callable(c)
        assert argspec.args == ["p", "t"]

    @pytest.mark.parametrize("windows", ([(4, 8)], (4, 8)))
    def test_rect_returns_correct_value_single_window(self, windows):
        """Test that the ``rect`` function returns the correct value only when t is inside
        the window."""
        c = qml.pulse.rect(x=10, windows=windows)

        times = np.arange(0, 10, step=1e-2)
        for t in times:
            if 4 <= t <= 8:
                assert c(p=1, t=t) == 10  # p is ignored
            else:
                assert c(p=1, t=t) == 0

    @pytest.mark.parametrize("windows", ([2, (4, 8)], (4, 8, 8), [(4, 9), 1], (4,), ([4],)))
    def test_rect_raises_invalid_windows(self, windows):
        """Test that the ``rect`` function raises a ValueError for ill-formatted windows."""
        with pytest.raises(ValueError, match="At least one provided window"):
            _ = qml.pulse.rect(x=10, windows=windows)

    def test_rect_returns_correct_value_multiple_windows(self):
        """Test that the ``rect`` function returns the correct value only when t is inside
        the window."""

        def f(p, t):
            return p * t

        c = qml.pulse.rect(x=f, windows=[(4, 8), (0, 1), (9, 10)])

        times = np.arange(0, 10, step=1e-2)
        param = 10
        for t in times:
            if 4 <= t <= 8 or 0 <= t <= 1 or 9 <= t <= 10:
                assert qml.math.allclose(c(p=param, t=t), f(param, t))
            else:
                assert c(p=param, t=t) == 0

    def test_rect_returns_correct_value_no_windows(self):
        """Test that the ``rect`` function always returns the correct value when no windows
        are specified"""

        def f(p, t):
            return p * t

        c = qml.pulse.rect(x=f)

        times = np.arange(0, 10, step=1e-2)
        for t in times:
            assert c(p=0.5, t=t) == f(0.5, t)

    @pytest.mark.jax
    def test_rect_is_jittable(self):
        """Test that the callable returned by the ``rect`` function is jittable."""
        import jax

        def f(p, t):
            return p * t

        c = jax.jit(qml.pulse.rect(x=f, windows=[(4, 8), (0, 1), (9, 10)]))

        times = np.arange(0, 10, step=1e-2)
        param = 10
        for t in times:
            if 4 <= t <= 8 or 0 <= t <= 1 or 9 <= t <= 10:
                assert qml.math.allclose(c(p=param, t=t), f(param, t))
            else:
                assert c(p=param, t=t) == 0


@pytest.mark.jax
class TestIntegration:
    """Unit tests testing the integration of convenience functions with parametrized hamiltonians."""

    def test_parametrized_hamiltonian(self):
        """Test that convenience functions can be used to define parametrized hamiltonians."""

        def f1(p, t):
            return p * t

        windows1 = [(0, 0.5), (1, 1.5)]

        coeffs = [qml.pulse.rect(f1, windows1), qml.pulse.constant]
        ops = [qml.PauliX(0), qml.PauliY(1)]
        H = qml.dot(coeffs, ops)

        assert isinstance(H, ParametrizedHamiltonian)
        # assert that at t=0.3 both coefficients are non-zero
        true_mat = qml.matrix(f1(1, 0.3) * qml.PauliX(0) + 2 * qml.PauliY(1), wire_order=[0, 1])
        assert qml.math.allclose(qml.matrix(H(params=[1, 2], t=0.3)), true_mat)
        # assert that at t=0.7 only the second coefficient is non-zero
        true_mat = qml.matrix(2 * qml.PauliY(1), wire_order=[0, 1])
        assert qml.math.allclose(qml.matrix(H(params=[1, 2], t=0.7)), true_mat)

    @pytest.mark.slow
    @pytest.mark.jax
    def test_qnode(self):
        """Test that the evolution of a parametrized hamiltonian defined with convenience functions
        can be executed on a QNode."""
        import jax
        import jax.numpy as jnp

        def f1(p, t):
            return p * t

        windows1 = [(0, 0.5), (1, 1.5)]

        coeffs = [qml.pulse.rect(f1, windows1), qml.pulse.constant]
        ops = [qml.PauliX(0), qml.PauliY(1)]
        H = qml.dot(coeffs, ops)

        t = (1, 1.1)

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
