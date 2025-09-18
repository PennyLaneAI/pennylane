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

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

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

    def test_error_during_body_fn(self):
        """Test that an error in the body function is reraised."""

        @qml.while_loop(lambda i: i < 5)
        def w(i):
            raise ValueError("my random error")

        with pytest.raises(ValueError, match="my random error"):
            _ = jax.make_jaxpr(w)(0)


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


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestCaptureWhileLoopDynamicShapes:

    def test_while_loop_dynamic_shape_array(self):
        """Test while loop can accept arrays with dynamic shapes."""

        def f(x):
            @qml.while_loop(lambda res: jnp.sum(res) < 10)
            def g(res):
                return res + res

            return g(x)

        jaxpr = jax.make_jaxpr(f, abstracted_axes=("a",))(jnp.arange(2))

        [dynamic_shape, output] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3, jnp.arange(3))
        expected = jnp.array([0, 4, 8])
        assert qml.math.allclose(dynamic_shape, 3)
        assert jnp.allclose(output, expected)

    def test_while_loop_dynamic_array_creation(self):
        """Test that while loop can handle creating dynamic arrays."""

        @qml.while_loop(lambda s: s < 9)
        def f(s):
            a = jnp.ones(s + 1, dtype=int)
            return jnp.sum(a)

        def w():
            return f(3)

        jaxpr = jax.make_jaxpr(w)()
        [r] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert qml.math.allclose(r, 9)  # value that stops iteration

    def test_error_if_resizing_when_forbidden(self):
        """Test that a useful error is raised if the shape pattern changes with
        allow_array_resizing=False"""

        @qml.while_loop(lambda a, b: jnp.sum(a) < 10, allow_array_resizing=False)
        def f(a, b):
            return jnp.ones(a.shape[0] + b.shape[0]), 2 * b

        def w(i0):
            a0, b0 = jnp.ones(i0), jnp.ones(i0)
            return f(a0, b0)

        with pytest.raises(ValueError, match="Detected dynamically shaped arrays being resized"):
            jax.make_jaxpr(w)(1)

    def test_error_if_combining_independent_shapes(self):
        """Test that a useful error is raised if two arrays with dynamic shapes are combined."""

        @qml.while_loop(lambda a, b: jnp.sum(a) < 10, allow_array_resizing=True)
        def f(a, b):
            return a * b, 2 * b

        def w(i0):
            a0, b0 = jnp.ones(i0), jnp.ones(i0)
            return f(a0, b0)

        with pytest.raises(
            ValueError, match="attempt to combine arrays with two different dynamic shapes."
        ):
            jax.make_jaxpr(w)(2)

    def test_array_initialized_with_size_of_other_arg(self):
        """Test that one argument can have a shape that matches another argument, but
        can be resized independently of that arg."""

        @qml.while_loop(lambda i, a: i < 5)
        def f(i, a):
            return i + 1, 2 * a

        def w(i0):
            return f(i0, jnp.ones(i0))

        jaxpr = jax.make_jaxpr(w)(2)
        [a_size, final_i, final_a] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)
        assert qml.math.allclose(a_size, 2)  # what it was initialized with
        assert qml.math.allclose(final_i, 5)  # loop condition
        assert qml.math.allclose(final_a, jnp.ones(2) * 2**3)  # 2**(5-2)

    @pytest.mark.parametrize("allow_array_resizing", (True, False, "auto"))
    def test_error_if_combine_with_dynamic_closure_var(self, allow_array_resizing):
        """Test that if a broadcasting error is raised when a dynamically shaped closure variable
        is present, the error mentions it may be due to the closure variable with a dynamic shape.
        """

        def w(i0):
            c = jnp.arange(i0)

            @qml.while_loop(lambda a: jnp.sum(a) < 10, allow_array_resizing=allow_array_resizing)
            def f(a):
                return c * a

            return f(jnp.arange(i0))

        with pytest.raises(ValueError, match="due to a closure variable with a dynamic shape"):
            jax.make_jaxpr(w)(3)

    @pytest.mark.parametrize("allow_array_resizing", ("auto", False))
    def test_loop_with_argument_combining(self, allow_array_resizing):
        """Test that arguments with dynamic shapes can be combined if allow_array_resizing=auto or False."""

        @qml.while_loop(lambda a, b: jnp.sum(a) < 20, allow_array_resizing=allow_array_resizing)
        def f(a, b):
            return a + b, b + 1

        def w(i0):
            a0, b0 = jnp.ones(i0), jnp.ones(i0)
            return f(a0, b0)

        jaxpr = jax.make_jaxpr(w)(2)
        [dynamic_shape, a, b] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)
        assert qml.math.allclose(dynamic_shape, 2)  # the initial size
        assert qml.math.allclose(a, jnp.array([11, 11]))  # 11 + 11 > 20 , 11 = 1 + 1+ 2 + 3+ 4
        assert qml.math.allclose(b, jnp.array([5, 5]))

    @pytest.mark.parametrize("allow_array_resizing", ("auto", False))
    def test_loop_args_resized_together(self, allow_array_resizing):
        """Test that arrays can be resized as long as they are resized together."""

        @qml.while_loop(lambda a, b: jnp.sum(a) < 10, allow_array_resizing=allow_array_resizing)
        def f(x, y):
            x = jnp.ones(x.shape[0] + y.shape[0])
            return x, 2 * x

        def workflow(i0):
            x0 = jnp.ones(i0)
            y0 = jnp.ones(i0)
            return f(x0, y0)

        jaxpr = jax.make_jaxpr(workflow)(2)
        [dynamic_shape, x, y] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1)
        assert qml.math.allclose(dynamic_shape, 16)
        x_expected = jnp.ones(16)
        assert qml.math.allclose(x, x_expected)
        assert qml.math.allclose(y, 2 * x_expected)

    @pytest.mark.parametrize("allow_array_resizing", ("auto", True))
    def test_independent_resizing(self, allow_array_resizing):
        """Test that two arrays can be resized independently of each other."""

        @qml.while_loop(
            lambda a, b: jax.numpy.sum(a) < 10, allow_array_resizing=allow_array_resizing
        )
        def f(a, b):
            return jnp.ones(a.shape[0] + b.shape[0]), b + 1

        def w(i0):
            return f(jnp.zeros(i0), jnp.zeros(i0))

        jaxpr = jax.make_jaxpr(w)(2)
        [shape1, shape2, a, b] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)
        assert jnp.allclose(shape1, 12)
        assert jnp.allclose(shape2, 3)
        expected = jnp.ones(12)
        assert jnp.allclose(a, expected)
        assert jnp.allclose(b, jnp.array([3, 3, 3]))
