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

import numpy as np
import pytest

import pennylane as qml


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
