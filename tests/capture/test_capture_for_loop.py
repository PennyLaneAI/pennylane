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

# pylint: disable=no-value-for-parameter, too-few-public-methods, no-self-use
# pylint: disable=too-many-positional-arguments, too-many-arguments

import numpy as np
import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# must be below jax importorskip
from pennylane.capture.primitives import for_loop_prim  # pylint: disable=wrong-import-position


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


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestDynamicShapes:

    # pylint: disable=unused-argument
    def test_dynamic_shape_input(self):
        """Test that the for loop can accept inputs with dynamic shapes."""

        def f(x):
            n = jax.numpy.shape(x)[0]

            @qml.for_loop(n)
            def g(_, y):
                return y + y

            return g(x)

        jaxpr = jax.make_jaxpr(f, abstracted_axes=("a",))(jax.numpy.arange(5))

        [shape, output] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3, jax.numpy.arange(3))
        expected = jax.numpy.array([0, 8, 16])  # [0, 1, 2] * 2**3
        assert jax.numpy.allclose(output, expected)
        assert qml.math.allclose(shape, 3)

    # pylint: disable=unused-argument
    def test_dynamic_array_creation(self):
        """Test that for_loops can create dynamically shaped arrays."""

        def f(i, x):
            y = jax.numpy.arange(i)
            return jax.numpy.sum(y)

        def w():
            return qml.for_loop(4)(f)(0)

        jaxpr = jax.make_jaxpr(w)()
        [r] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)
        assert qml.math.allclose(r, 3)  # sum([0,1,2]) from final loop iteration

    def test_error_if_resizing_when_forbidden(self):
        """Test that a useful error is raised if the shape pattern changes with
        allow_array_resizing=False"""

        @qml.for_loop(3, allow_array_resizing=False)
        def f(i, a, b):
            a_size = a.shape[0]
            b_size = b.shape[0]
            return jax.numpy.ones(a_size + b_size), 2 * b

        def w(i0):
            a0, b0 = jnp.ones(i0), jnp.ones(i0)
            return f(a0, b0)

        with pytest.raises(ValueError, match="Detected dynamically shaped arrays being resized"):
            jax.make_jaxpr(w)(1)

    def test_error_is_combining_independent_shapes(self):
        """Test that a useful error is raised if two arrays with dynamic shapes are combined."""

        @qml.for_loop(3, allow_array_resizing=True)
        def f(i, a, b):
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

        @qml.for_loop(3)
        def f(i, j, a):
            return j + i, 2 * a

        def w(i0):
            return f(i0, jnp.ones(i0))

        jaxpr = jax.make_jaxpr(w)(2)
        [a_size, final_j, final_a] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)
        assert qml.math.allclose(a_size, 2)  # what it was initialized with
        assert qml.math.allclose(final_j, 5)  # 2 +3
        assert qml.math.allclose(final_a, jnp.ones(2) * 2**3)  # 2**3

    @pytest.mark.parametrize("allow_array_resizing", (True, False, "auto"))
    def test_error_if_combine_with_dynamic_closure_var(self, allow_array_resizing):
        """Test that if a broadcasting error is raised when a dynamically shaped closure variable
        is present, the error mentions it may be due to the closure variable with a dynamic shape.
        """

        def w(i0):
            c = jnp.arange(i0)

            @qml.for_loop(3, allow_array_resizing=allow_array_resizing)
            def f(i, a):
                return c * a

            return f(jnp.arange(i0))

        with pytest.raises(ValueError, match="due to a closure variable with a dynamic shape"):
            jax.make_jaxpr(w)(3)

    @pytest.mark.parametrize("allow_array_resizing", ("auto", False))
    def test_loop_with_argument_combining(self, allow_array_resizing):
        """Test that arguments with dynamic shapes can be combined if allow_array_resizing=auto or False."""

        @qml.for_loop(4, allow_array_resizing=allow_array_resizing)
        def f(i, a, b):
            return a + i, a + b

        def w(i0):
            a0, b0 = jnp.ones(i0), jnp.ones(i0)
            return f(a0, b0)

        jaxpr = jax.make_jaxpr(w)(2)
        [dynamic_shape, a, b] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)
        assert qml.math.allclose(dynamic_shape, 2)  # the initial size
        assert qml.math.allclose(a, jnp.array([7, 7]))  # 1 + 0 + 1 + 2 + 3 = 7
        assert qml.math.allclose(b, jnp.array([9, 9]))  # 1 + 1 + 1 + 2 + 4

    @pytest.mark.parametrize("allow_array_resizing", ("auto", False))
    def test_loop_args_resized_together(self, allow_array_resizing):
        """Test that arrays can be resized as long as they are resized together."""

        @qml.for_loop(2, allow_array_resizing=allow_array_resizing)
        def f(i, x, y):
            x = jnp.ones(x.shape[0] + y.shape[0])
            return x, (i + 2) * x

        def workflow(i0):
            x0 = jnp.ones(i0)
            y0 = jnp.ones(i0)
            return f(x0, y0)

        jaxpr = jax.make_jaxpr(workflow)(2)
        [s, x, y] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)
        assert qml.math.allclose(s, 8)
        x_expected = jnp.ones(8)
        assert qml.math.allclose(x, x_expected)
        assert qml.math.allclose(y, 3 * x_expected)

    @pytest.mark.parametrize("allow_array_resizing", ("auto", True))
    def test_independent_resizing(self, allow_array_resizing):
        """Test that two arrays can be resized independently of each other."""

        @qml.for_loop(4, allow_array_resizing=allow_array_resizing)
        def f(i, a, b):
            return jnp.ones(a.shape[0] + b.shape[0]), b + 1

        def w(i0):
            return f(jnp.zeros(i0), jnp.zeros(i0))

        jaxpr = jax.make_jaxpr(w)(2)
        [shape1, shape2, a, b] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)
        assert jnp.allclose(shape1, 15)
        assert jnp.allclose(shape2, 3)
        expected = jnp.ones(15)
        assert jnp.allclose(a, expected)
        assert jnp.allclose(b, jnp.array([4, 4, 4]))


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
