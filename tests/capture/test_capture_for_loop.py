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
# pylint: disable=too-many-arguments
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
        # because the concrete value of 'jax.numpy.ones([arg])' wouldn't be known.

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
        # because the concrete value of 'jax.numpy.ones([arg])' wouldn't be known.

    @pytest.mark.parametrize(
        "lower_bound, upper_bound, step, arg, expected",
        [(0, 5, 1, 0, 30), (0, 5, 2, 0, 20), (0, 10, 1, 0, 285), (10, 50, 5, 2, 7102)],
    )
    def test_for_loop_dynamic_bounds_step(self, lower_bound, upper_bound, step, arg, expected):
        """Test for-loops with dynamic bounds and step sizes."""

        def fn(lower_bound, upper_bound, step, arg):

            @qml.for_loop(lower_bound, upper_bound, step)
            def loop_body(i, arg):
                return arg + i**2

            return loop_body(arg)

        args = [lower_bound, upper_bound, step, arg]
        result = fn(*args)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize(
        "array, expected", [(jax.numpy.array([0]), 0), (jax.numpy.array([0.1, 0.2, 0.3, 0.4]), 1.0)]
    )
    def test_for_loop_dynamic_array(self, array, expected):
        """Test for-loops with dynamic array inputs."""

        def fn(array):

            @qml.for_loop(0, 4, 1)
            def loop_body(i, array, sum_val):
                return array, sum_val + array[i]

            sum_val = 0
            _, res = loop_body(array, sum_val)
            return res

        result = fn(array)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(array)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, array)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"


class TestCaptureCircuitsForLoop:
    """Tests for capturing for loops into jaxpr in the context of quantum circuits."""

    def test_for_loop_capture(self):
        """Test that a for loop is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():

            @qml.for_loop(0, 3, 1)
            def loop_fn(i):
                qml.RX(i, wires=0)

            loop_fn()

            return qml.expval(qml.PauliZ(0))

        result = circuit()
        expected = -0.9899925
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)()
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize("arg, expected", [(2, 0.18239626), (10.5, -0.77942717)])
    def test_circuit_args(self, arg, expected):
        """Test that a for loop with arguments is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(arg):

            qml.Hadamard(wires=0)

            @qml.for_loop(0, 10, 1)
            def loop_body(i, x):
                qml.RX(x, wires=0)
                qml.RY(jax.numpy.sin(x), wires=0)
                return x + i**2

            loop_body(arg)

            return qml.expval(qml.PauliZ(0))

        result = circuit(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(arg)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize("arg, expected", [(2, -0.49999517), (0, -0.03277611)])
    def test_circuit_consts(self, arg, expected):
        """Test that a for loop with jaxpr constants is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(arg):

            # these are captured as consts
            arg1 = arg + 0.1
            arg2 = arg + 0.2

            qml.Hadamard(wires=0)

            @qml.for_loop(0, 10, 1)
            def loop_body(i, x):
                qml.RX(arg1, wires=0)
                qml.RX(arg2, wires=0)
                qml.RY(jax.numpy.sin(x), wires=0)
                return x + i**2

            loop_body(arg)

            return qml.expval(qml.PauliZ(0))

        result = circuit(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(arg)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize(
        "lower_bound, upper_bound, step, arg, expected",
        [(0, 10, 1, 10.5, -0.77942717), (10, 20, 2, 0, 0.35913655)],
    )
    def test_dynamic_circuit_arg(self, lower_bound, upper_bound, step, arg, expected):
        """Test that a for loop with dynamic bounds and argument is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(lower_bound, upper_bound, step, arg):

            qml.Hadamard(wires=0)

            @qml.for_loop(lower_bound, upper_bound, step)
            def loop_body(i, x):
                qml.RX(x, wires=0)
                qml.RY(jax.numpy.sin(x), wires=0)
                return x + i**2

            loop_body(arg)

            return qml.expval(qml.PauliZ(0))

        args = [lower_bound, upper_bound, step, arg]
        result = circuit(*args)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"
