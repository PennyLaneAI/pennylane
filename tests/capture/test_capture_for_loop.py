# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for capturing for loops into jaxpr.
"""

# pylint: disable=no-value-for-parameter
# pylint: disable=too-few-public-methods
# pylint: disable=no-self-use

import numpy as np
import pytest

import pennylane as qml

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """Enable and disable the PennyLane JAX capture context manager."""
    qml.capture.enable()
    yield
    qml.capture.disable()


class TestCaptureForLoop:
    """Tests for capturing for loops into jaxpr."""

    @pytest.mark.parametrize("arg", [0, 5])
    def test_for_loop_identity(self, arg):
        """Test simple for-loop primitive vs dynamic dimensions."""

        def fn(arg):

            a = jax.numpy.ones([arg])

            @qml.for_loop(0, 10, 2)
            def loop(_, a):
                return a

            a2 = loop(a)
            return a2

        expected = jax.numpy.ones(arg)
        result = fn(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        # Note that this cannot be transformed with `jax.make_jaxpr`
        # because the concrete value of 'jax.numpy.ones([arg])' is not known.

    @pytest.mark.parametrize("arg", [0, 5])
    def test_for_loop_shared_indbidx(self, arg):
        """Test for-loops with shared dynamic input dimensions."""

        def fn(arg):
            a = jax.numpy.ones([arg], dtype=float)
            b = jax.numpy.ones([arg], dtype=float)

            @qml.for_loop(0, 10, 2)
            def loop(_, a, b):
                return (a, b)

            a2, b2 = loop(a, b)
            return a2 + b2

        result = fn(arg)
        expected = 2 * jax.numpy.ones(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        # Note that this cannot be transformed with `jax.make_jaxpr`
        # because the concrete value of 'jax.numpy.ones([arg])' is not known.

    @pytest.mark.parametrize("lower_bound, upper_bound, arg, expected", [(0, 10, 0, 285)])
    def test_for_loop_dynamic_bounds(self, lower_bound, upper_bound, arg, expected):
        """Test for-loops with dynamic lower and upper bounds."""

        def fn(lower_bound, upper_bound, arg):

            @qml.for_loop(lower_bound, upper_bound, 1)
            def loop_body(i, arg):
                return arg + i**2

            return loop_body(arg)

        args = [lower_bound, upper_bound, arg]

        result = fn(*args)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"


class TestCaptureCircuitsForLoop:
    """Tests for capturing for loops into jaxpr in the context of quantum circuits."""

    @pytest.mark.parametrize("n, expected", [(3, -0.9899925)])
    def test_for_loop_capture(self, n, expected):
        """Test that a for loop is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(n):

            @qml.for_loop(0, n, 1)
            def loop_fn(i):
                qml.RX(i, wires=0)

            loop_fn()

            return qml.expval(qml.PauliZ(0))

        result = circuit(n)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(n)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, n)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"
