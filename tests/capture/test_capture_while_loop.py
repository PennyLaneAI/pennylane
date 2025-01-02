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
Tests for capturing for while loops into jaxpr.
"""

import numpy as np
import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import while_loop_prim  # pylint: disable=wrong-import-position


class TestCaptureWhileLoop:
    """Tests for capturing for while loops into jaxpr."""

    @pytest.mark.parametrize("x", [1.6, 2.4])
    def test_while_loop_simple(self, x):
        """Test simple while-loop primitive"""

        def fn(x):

            @qml.while_loop(lambda x: x < 2)
            def loop(x):
                return x**2

            x2 = loop(x)
            return x2

        expected = x**2 if x < 2 else x
        result = fn(x)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(fn)(x)
        assert jaxpr.eqns[0].primitive == while_loop_prim
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize("array", [jax.numpy.zeros(0), jax.numpy.zeros(5)])
    def test_while_loop_dynamic_array(self, array):
        """Test while loops with dynamic array inputs."""

        def fn(arg):

            a, b = jax.numpy.ones(arg.shape, dtype=float), jax.numpy.ones(arg.shape, dtype=float)

            # Note: lambda *_, idx: idx < 10 doesn't work - necessary keyword argument not provided
            @qml.while_loop(lambda *args: args[-1] < 10)
            def loop(a, b, idx):
                return a + b, b + a, idx + 2

            return loop(a, b, 0)

        res_arr1, res_arr2, res_idx = fn(array)
        expected = 2**5 * jax.numpy.ones(*array.shape)
        assert jax.numpy.allclose(res_arr1, res_arr2)
        assert jax.numpy.allclose(res_arr1, expected), f"Expected {expected}, but got {res_arr1}"

        jaxpr = jax.make_jaxpr(fn)(array)
        res_arr1_jxpr, res_arr2_jxpr, res_idx_jxpr = jax.core.eval_jaxpr(
            jaxpr.jaxpr, jaxpr.consts, array
        )

        assert np.allclose(res_arr1_jxpr, res_arr2_jxpr)
        assert np.allclose(res_arr1_jxpr, expected), f"Expected {expected}, but got {res_arr1_jxpr}"
        assert np.allclose(res_idx, res_idx_jxpr) and res_idx_jxpr == 10


class TestCaptureCircuitsWhileLoop:
    """Tests for capturing for while loops into jaxpr in the context of quantum circuits."""

    def test_while_loop_capture(self):
        """Test that a while loop is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():

            @qml.while_loop(lambda i: i < 3)
            def loop_fn(i):
                qml.RX(i, wires=0)
                return i + 1

            _ = loop_fn(0)

            return qml.expval(qml.Z(0))

        result = circuit()
        expected = np.cos(0 + 1 + 2)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)()
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize("arg, expected", [(1.2, -0.16852022), (1.6, 0.598211352)])
    def test_circuit_args(self, arg, expected):
        """Test that a while loop with arguments is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(arg):

            qml.Hadamard(wires=0)
            arg1, arg2 = arg + 0.1, arg + 0.2

            @qml.while_loop(lambda x: x < 2.0)
            def loop_body(x):
                qml.RZ(arg1, wires=0)
                qml.RZ(arg2, wires=0)
                qml.RX(x, wires=0)
                qml.RY(jax.numpy.sin(x), wires=0)
                return x**2

            loop_body(arg)

            return qml.expval(qml.Z(0))

        result = circuit(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(arg)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize("arg, expected", [(3, 5), (11, 21)])
    def test_circuit_closure_vars(self, arg, expected):
        """Test that closure variables within a while loop are correctly captured via jaxpr."""

        def circuit(x):
            y = x + 1

            def while_f(i):
                return i < y

            @qml.while_loop(while_f)
            def f(i):
                return 4 * i + 1

            return f(0)

        result = circuit(arg)
        assert qml.math.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(arg)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize(
        "upper_bound, arg, expected", [(3, 0.5, 0.00223126), (2, 12, 0.2653001)]
    )
    def test_while_loop_nested(self, upper_bound, arg, expected):
        """Test that a nested while loop is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(upper_bound, arg):

            # while loop with dynamic bounds
            @qml.while_loop(lambda i: i < upper_bound)
            def loop_fn(i):
                qml.Hadamard(wires=i)
                return i + 1

            # nested while loops.
            # outer while loop updates x
            @qml.while_loop(lambda _, i: i < upper_bound)
            def loop_fn_returns(x, i):
                qml.RX(x, wires=i)

                # inner while loop
                @qml.while_loop(lambda j: j < upper_bound)
                def inner(j):
                    qml.RZ(j, wires=0)
                    qml.RY(x**2, wires=0)
                    return j + 1

                inner(i + 1)

                return x + 0.1, i + 1

            loop_fn(0)
            loop_fn_returns(arg, 0)

            return qml.expval(qml.Z(0))

        args = [upper_bound, arg]
        result = circuit(*args)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(circuit)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.xfail(strict=False)  # mcms only sometimes give the right answer
    @pytest.mark.parametrize("upper_bound, arg", [(3, 0.5), (2, 12)])
    def test_while_and_for_loop_nested(self, upper_bound, arg):
        """Test that a nested while and for loop is correctly captured into a jaxpr."""

        dev = qml.device("default.qubit", wires=3)

        def ry_fn(arg):
            qml.RY(arg, wires=1)

        @qml.qnode(dev)
        def circuit(upper_bound, arg):

            # while loop with dynamic bounds
            @qml.while_loop(lambda i: i < upper_bound)
            def loop_fn(i):
                qml.Hadamard(wires=i)

                @qml.for_loop(0, i, 1)
                def loop_fn_returns(i, x):
                    qml.RX(x, wires=i)
                    m_0 = qml.measure(0)
                    qml.cond(m_0, ry_fn)(x)
                    return i + 1

                loop_fn_returns(arg)
                return i + 1

            loop_fn(0)

            return qml.expval(qml.Z(0))

        args = [upper_bound, arg]
        result = circuit(*args)
        jaxpr = jax.make_jaxpr(circuit)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(result, res_ev_jxpr), f"Expected {result}, but got {res_ev_jxpr}"

    def test_while_loop_grad(self):
        """Test simple while-loop primitive with gradient."""
        from pennylane.capture.primitives import grad_prim

        @qml.qnode(qml.device("default.qubit", wires=2))
        def inner_func(x):

            @qml.while_loop(lambda i: i < 3)
            def loop_fn(i):
                qml.RX(i * x, wires=0)
                return i + 1

            _ = loop_fn(0)

            return qml.expval(qml.Z(0))

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


def test_pytree_input_output():
    """Test that the while loop supports pytree input and output."""

    @qml.while_loop(lambda x: x["x"] < 10)
    def f(x):
        return {"x": x["x"] + 1}

    x0 = {"x": 0}
    out = f(x0)
    assert list(out.keys()) == ["x"]
    assert qml.math.allclose(out["x"], 10)
