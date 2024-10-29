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

# must be below jax importorskip
from pennylane.capture.primitives import for_loop_prim  # pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """Enable and disable the PennyLane JAX capture context manager."""
    qml.capture.enable()
    yield
    qml.capture.disable()


class TestCaptureForLoop:
    """Tests for capturing for loops into jaxpr."""

    @pytest.mark.parametrize("array", [jax.numpy.zeros(0), jax.numpy.zeros(5)])
    def test_for_loop_identity(self, array):
        """Test simple for-loop primitive vs dynamic dimensions."""

        def fn(arg):

            a = jax.numpy.ones(arg.shape)

            @qml.for_loop(0, 10, 2)
            def loop(_, a):
                return a

            a2 = loop(a)
            return a2

        expected = jax.numpy.ones(array.shape)
        result = fn(array)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(array)
        assert jaxpr.eqns[1].primitive == for_loop_prim
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, array)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize("array", [jax.numpy.zeros(0), jax.numpy.zeros(5)])
    def test_for_loop_defaults(self, array):
        """Test simple for-loop primitive using default values."""

        def fn(arg):

            a = jax.numpy.ones(arg.shape)

            @qml.for_loop(0, 10, 1)
            def loop1(_, a):
                return a

            @qml.for_loop(10, 1)
            def loop2(_, a):
                return a

            @qml.for_loop(10)
            def loop3(_, a):
                return a

            r1, r2, r3 = loop1(a), loop2(a), loop3(a)
            return r1, r2, r3

        expected = jax.numpy.ones(array.shape)
        result = fn(array)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(array)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, array)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize(
        "array, expected",
        [
            (jax.numpy.zeros(5), jax.numpy.array([0, 1, 4, 9, 16])),
            (jax.numpy.zeros(10), jax.numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])),
        ],
    )
    def test_for_loop_default(self, array, expected):
        """Test simple for-loop primitive using default values."""

        def fn(arg):

            stop = arg.shape[0]
            a = jax.numpy.ones(stop)

            @qml.for_loop(0, stop, 1)
            def loop1(i, a):
                return a.at[i].set(i**2)

            @qml.for_loop(0, stop)
            def loop2(i, a):
                return a.at[i].set(i**2)

            @qml.for_loop(stop)
            def loop3(i, a):
                return a.at[i].set(i**2)

            r1, r2, r3 = loop1(a), loop2(a), loop3(a)
            return r1, r2, r3

        result = fn(array)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(array)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, array)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    def test_for_loop_grad(self):
        """Test simple for-loop primitive with gradient."""
        from pennylane.capture.primitives import grad_prim

        @qml.qnode(qml.device("default.qubit", wires=2))
        def inner_func(x):

            @qml.for_loop(0, 2)
            def loop(w):
                qml.RX(x * w, w)

            loop()
            return qml.expval(qml.Z(0) @ qml.Z(1))

        def func_qml(x):
            return qml.grad(inner_func)(x)

        def func_jax(x):
            return jax.grad(inner_func)(x)

        x = 0.7
        jax_out = func_jax(x)
        assert qml.math.allclose(func_qml(x), jax_out)

        # Check overall jaxpr properties
        jaxpr = jax.make_jaxpr(func_qml)(x)
        assert len(jaxpr.eqns) == 1  # a single grad equation

        grad_eqn = jaxpr.eqns[0]
        assert grad_eqn.primitive == grad_prim
        assert set(grad_eqn.params.keys()) == {"argnum", "n_consts", "jaxpr", "method", "h"}
        assert grad_eqn.params["argnum"] == [0]
        assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
        assert len(grad_eqn.params["jaxpr"].eqns) == 1  # a single QNode equation

        manual_eval = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert qml.math.allclose(manual_eval, jax_out)

    @pytest.mark.parametrize("array", [jax.numpy.zeros(0), jax.numpy.zeros(5)])
    def test_for_loop_shared_indbidx(self, array):
        """Test for-loops with shared dynamic input dimensions."""

        def fn(arg):

            a = jax.numpy.ones(arg.shape, dtype=float)
            b = jax.numpy.ones(arg.shape, dtype=float)

            @qml.for_loop(0, 10, 2)
            def loop(_, a, b):
                return (a, b)

            a2, b2 = loop(a, b)
            return a2 + b2

        result = fn(array)
        expected = 2 * jax.numpy.ones(array.shape)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(array)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, array)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

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

            return qml.expval(qml.Z(0))

        result = circuit()
        expected = np.cos(0 + 1 + 2)
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

            return qml.expval(qml.Z(0))

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

            return qml.expval(qml.Z(0))

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

            return qml.expval(qml.Z(0))

        args = [lower_bound, upper_bound, step, arg]
        result = circuit(*args)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize(
        "upper_bound, arg, expected", [(3, 0.5, 0.00223126), (2, 12, 0.2653001)]
    )
    def test_for_loop_nested(self, upper_bound, arg, expected):
        """Test that a nested for loop is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(upper_bound, arg):

            # for loop with dynamic bounds
            @qml.for_loop(0, upper_bound, 1)
            def loop_fn(i):
                qml.Hadamard(wires=i)

            # nested for loops.
            # outer for loop updates x
            @qml.for_loop(0, upper_bound, 1)
            def loop_fn_returns(i, x):
                qml.RX(x, wires=i)

                # inner for loop
                @qml.for_loop(i + 1, upper_bound, 1)
                def inner(j):
                    qml.RZ(j, wires=0)
                    qml.RY(x**2, wires=0)

                inner()

                return x + 0.1

            loop_fn()
            loop_fn_returns(arg)

            return qml.expval(qml.Z(0))

        args = [upper_bound, arg]
        result = circuit(*args)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize(
        "upper_bound, arg, expected", [(3, 0.5, 0.00223126), (2, 12, 0.2653001)]
    )
    def test_nested_for_and_while_loop(self, upper_bound, arg, expected):
        """Test that a nested for loop and while loop is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(upper_bound, arg):

            # for loop with dynamic bounds
            @qml.for_loop(0, upper_bound, 1)
            def loop_fn(i):
                qml.Hadamard(wires=i)

            # nested for-while loops.
            @qml.for_loop(0, upper_bound, 1)
            def loop_fn_returns(i, x):
                qml.RX(x, wires=i)

                # inner while loop
                @qml.while_loop(lambda j: j < upper_bound)
                def inner(j):
                    qml.RZ(j, wires=0)
                    qml.RY(x**2, wires=0)
                    return j + 1

                inner(i + 1)

                return x + 0.1

            loop_fn()
            loop_fn_returns(arg)

            return qml.expval(qml.Z(0))

        args = [upper_bound, arg]
        result = circuit(*args)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"


def test_pytree_inputs():
    """Test that for_loop works with pytree inputs and outputs."""

    @qml.for_loop(1, 7, 2)
    def f(i, x):
        return {"x": i + x["x"]}

    x = {"x": 0}
    out = f(x)
    assert list(out.keys()) == ["x"]
    assert qml.math.allclose(out["x"], 9)  # 1 + 3 + 5 = 9
