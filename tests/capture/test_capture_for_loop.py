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

    def test_error_if_wrong_number_of_outputs(self):
        """Test that a helpful error is raised if the function has the wrong number of outputs."""

        @qml.for_loop(3)
        def f(i):  # pylint: disable=unused-argument
            return 2

        with pytest.raises(
            ValueError, match="number of inputs must be one greater than the number of outputs"
        ):
            f()

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

        assert jaxpr.eqns[1].invars[0].val == 0
        assert jaxpr.eqns[1].invars[1].val == 10
        assert jaxpr.eqns[1].invars[2].val == 2

        assert len(jaxpr.eqns[1].params["jaxpr_body_fn"].eqns) == 0

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

        assert jaxpr.eqns[1].invars[0].val == 0
        assert jaxpr.eqns[1].invars[1].val == 10
        assert jaxpr.eqns[1].invars[2].val == 1

        assert jaxpr.eqns[2].invars[0].val == 10
        assert jaxpr.eqns[2].invars[1].val == 1
        assert jaxpr.eqns[2].invars[2].val == 1

        assert jaxpr.eqns[3].invars[0].val == 0
        assert jaxpr.eqns[3].invars[1].val == 10
        assert jaxpr.eqns[3].invars[2].val == 1

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

        assert jaxpr.eqns[1].invars[0].val == 0
        assert jaxpr.eqns[1].invars[2].val == 1

        assert jaxpr.eqns[2].invars[0].val == 0
        assert jaxpr.eqns[2].invars[2].val == 1

        assert jaxpr.eqns[3].invars[0].val == 0
        assert jaxpr.eqns[3].invars[2].val == 1

        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, array)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    def test_for_loop_grad(self):
        """Test simple for-loop primitive with gradient."""
        from pennylane.capture.primitives import jacobian_prim

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
        assert grad_eqn.primitive == jacobian_prim
        assert set(grad_eqn.params.keys()) == {
            "argnums",
            "n_consts",
            "jaxpr",
            "method",
            "h",
            "fn",
            "scalar_out",
        }
        assert grad_eqn.params["argnums"] == (0,)
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

    def test_reverse_iteration(self):
        """Test that a requested reverse iteration is converted to a positive iteration."""

        @qml.for_loop(7, 0, -3)
        def f(i, j, x):
            x = x.at[j].set(i)
            return j + 1, x

        jaxpr = jax.make_jaxpr(f)(0, jnp.ones(5))

        # will hit 7, 4, 1
        assert jaxpr.eqns[0].invars[0].val == 0
        assert jaxpr.eqns[0].invars[1].val == 3
        assert jaxpr.eqns[0].invars[2].val == 1

        body_fn = jaxpr.eqns[0].params["jaxpr_body_fn"]
        assert body_fn.eqns[0].primitive.name == "mul"
        assert body_fn.eqns[0].invars[0].val == -3  # the step
        assert body_fn.eqns[1].primitive.name == "add"
        assert body_fn.eqns[1].invars[0].val == 7  # the initial start

        final_j, x = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0, jnp.zeros(5))
        assert qml.math.allclose(final_j, 3)
        assert qml.math.allclose(x, jnp.array([7, 4, 1, 0, 0]))

    def test_reverse_iteration_abstract_step(self):
        """Test that reverse iteration can be detected if the step is abstract
        but the start and stop are not."""

        def w(step):

            @qml.for_loop(8, 0, step)
            def f(i, j, x):
                # keep track of the order i occured in
                x = x.at[j].set(i)
                return j + 1, x

            return f(0, jnp.zeros(6))

        jaxpr = jax.make_jaxpr(w)(-1)

        assert jaxpr.eqns[-1].invars[0].val == 0
        assert jaxpr.eqns[-1].invars[2].val == 1

        assert jaxpr.eqns[-1].invars[1].aval.dtype == jnp.int64

        body_fn = jaxpr.eqns[-1].params["jaxpr_body_fn"]
        assert body_fn.eqns[0].primitive.name == "mul"
        assert body_fn.eqns[0].invars[0] == body_fn.constvars[0]  # the step
        assert body_fn.eqns[1].primitive.name == "add"
        assert body_fn.eqns[1].invars[0].val == 8  # the initial start

        final_j, x = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, -2)
        assert qml.math.allclose(final_j, 4)
        assert qml.math.allclose(x, jnp.array([8, 6, 4, 2, 0, 0]))

    def test_array_step(self):
        """Test that a jnp.array as a step works.  Checks that _reverse_iteration
        check still works and doesn't internally produce a tracer because of the array."""

        step = jnp.array(2)
        stop = jnp.array(5)

        def w():

            @qml.for_loop(1, stop, step)
            def l(i, x):
                return x + i

            l(0)

        jaxpr = jax.make_jaxpr(w)()
        assert jaxpr.eqns[0].invars[0].val == 1
        assert jaxpr.eqns[0].invars[1] == jaxpr.jaxpr.constvars[0]
        assert jaxpr.eqns[0].invars[2] == jaxpr.jaxpr.constvars[1]

        assert qml.math.allclose(jaxpr.consts[0], 5)
        assert qml.math.allclose(jaxpr.consts[1], 2)

    def test_array_step_reverse_iteration(self):
        """Test that a jnp.array as a step works.  Checks that _reverse_iteration
        check still works and doesn't internally produce a tracer because of the array."""

        step = jnp.array(-1)
        stop = jnp.array(-5)

        def w():

            @qml.for_loop(1, stop, step)
            def l(i, x):
                return x + i

            l(0)

        jaxpr = jax.make_jaxpr(w)()
        # check that it detected a reverse iteration
        # step is one instead of negative one
        assert jaxpr.eqns[-1].invars[0].val == 0
        assert jaxpr.eqns[-1].invars[2].val == 1

        # includes all the conversions to calculate num_iterations
        # so not just the for_loop eqn
        assert len(jaxpr.eqns) > 1
        # includes the index conversion to reversed
        assert jaxpr.eqns[-1].params["jaxpr_body_fn"].eqns[0].primitive.name == "mul"


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

        [output] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3, jax.numpy.arange(3))
        expected = jax.numpy.array([0, 8, 16])  # [0, 1, 2] * 2**3
        assert jax.numpy.allclose(output, expected)

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

        with pytest.warns(
            qml.exceptions.CaptureWarning, match="Structured capture of qml.for_loop failed"
        ):
            jaxpr = jax.make_jaxpr(w)(1)

        assert for_loop_prim not in {eqn.primitive for eqn in jaxpr.eqns}

    def test_error_is_combining_independent_shapes(self):
        """Test that a useful error is raised if two arrays with dynamic shapes are combined."""

        @qml.for_loop(3, allow_array_resizing=True)
        def f(i, a, b):
            return a * b, 2 * b

        def w(i0):
            a0, b0 = jnp.ones(i0), jnp.ones(i0)
            return f(a0, b0)

        with pytest.warns(
            qml.exceptions.CaptureWarning, match="Structured capture of qml.for_loop failed"
        ):
            jaxpr = jax.make_jaxpr(w)(2)

        assert for_loop_prim not in {eqn.primitive for eqn in jaxpr.eqns}

    def test_array_initialized_with_size_of_other_arg(self):
        """Test that one argument can have a shape that matches another argument, but
        can be resized independently of that arg."""

        @qml.for_loop(3)
        def f(i, j, a):
            return j + i, 2 * a

        def w(i0):
            return f(i0, jnp.ones(i0))

        jaxpr = jax.make_jaxpr(w)(2)
        [final_j, final_a] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)
        assert qml.math.allclose(final_j, 5)  # 2 +3
        assert qml.math.allclose(final_a, jnp.ones(2) * 2**3)  # 2**3

    @pytest.mark.parametrize("allow_array_resizing", (False, "auto"))
    def test_combine_with_dynamic_closure_var(self, allow_array_resizing):
        """Test that if the closure variable has a dynamic shape that matches an input dynamic shape, they
        can be combined."""

        def w(i0):
            c = jnp.arange(i0)

            @qml.for_loop(3, allow_array_resizing=allow_array_resizing)
            def f(i, a):
                return c * a

            return f(jnp.arange(i0))

        jaxpr = jax.make_jaxpr(w)(3)
        assert jaxpr.eqns[-1].primitive == for_loop_prim
        _, return_array, c = jaxpr.eqns[-1].outvars

        assert c.aval.shape[0] == jaxpr.jaxpr.invars[0]
        assert isinstance(c, jax.core.DropVar)

        assert return_array == jaxpr.jaxpr.outvars[0]

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
        [a, b] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)
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
        # shape of b present in inputs, shape of a is not
        [shape1, a, b] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)
        assert jnp.allclose(shape1, 15)
        expected = jnp.ones(15)
        assert jnp.allclose(a, expected)
        assert jnp.allclose(b, jnp.array([4, 4, 4]))

    def test_recombine_after_loop(self):
        """Test that arrays with the same dynamic shape can be recombined after a loop."""

        @qml.for_loop(2)
        def f(i, a, b):
            return 2 * a, b

        def w(i0):
            a = jnp.ones(i0)
            a_new, b_new = f(a, jnp.ones(i0))
            assert a_new.shape[0] is i0
            assert b_new.shape[0] is i0
            c = a_new + b_new
            d = a_new + a
            return c, d

        jaxpr = jax.make_jaxpr(w)(2)
        [c, d] = qml.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)
        assert jnp.allclose(c, jnp.array([5, 5, 5]))  # 2*2 + 1
        assert jnp.allclose(d, jnp.array([5, 5, 5]))

    def test_closure_var_as_dynamic_shape(self):
        """Test that a closure var can be used to produce something with a dynamic shape."""

        def f(sz):

            @qml.for_loop(0, 3, 1)
            def loop(i, a):
                return jnp.ones([sz])

            a2 = loop(jnp.ones(sz))
            return a2

        jaxpr = jax.make_jaxpr(f)(3)
        _, a2 = jaxpr.eqns[-1].outvars
        assert a2.aval.shape[0] is jaxpr.jaxpr.invars[0]  # sz

    def test_abstract_shapes_with_const(self):
        """Test that abstract dimensions can be used when consts are used."""

        def w(a):

            b = jnp.array([1.0, 2.0])
            c = jnp.ones(a)

            @qml.for_loop(3)
            def f(i, x, y):
                return x + b, 2 * y

            return f(b, c)

        jaxpr = jax.make_jaxpr(w)(2)
        shape, static_array, dynamic_array = jaxpr.eqns[-1].outvars
        assert isinstance(shape, jax.core.DropVar)
        assert static_array.aval.shape == (2,)
        assert dynamic_array.aval.shape[0] == jaxpr.jaxpr.invars[0]  # the input a

    def test_same_closure_variable_multiple_loops(self):
        """Test that if the same variable is used as a closure var multiple times, we don't get leaked tracers.
        When _loop_abstract_axes.promote_consts_to_inputs copied the function, it made it so that we ended
        up with consts with tracer values, leading to leaked tracers when integrated with catalyst.
        This just tests that doesn't happen again.
        """

        def w(x):
            @qml.for_loop(x.shape[0])
            def f(i):
                2 * x  # pylint: disable=pointless-statement

            f()

            @qml.for_loop(x.shape[0])
            def g(i):
                3 * x  # pylint: disable=pointless-statement

            g()

        jaxpr = jax.make_jaxpr(w, abstracted_axes={0: "a"})(jnp.array([0, 1, 2]))
        assert len(jaxpr.consts) == 0


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

    def test_closure_var_has_shape_property_that_isnt_a_shape(self):
        """Test an edge case that a closure variable can have an attribute shape that isn't a tuple of ints.

        Encountered in benchmarking suite.
        """

        class ThingWithShape:

            def __init__(self):
                pass

            def shape(self):  # method not property
                return 2

        def w():

            thing = ThingWithShape()

            # pylint: disable=unused-argument
            @qml.for_loop(3)
            def f(i, x):
                return x + thing.shape()

            return f(2)

        jaxpr = jax.make_jaxpr(w)()
        assert jaxpr.eqns[0].primitive == for_loop_prim
        assert jaxpr.eqns[0].params["jaxpr_body_fn"].eqns[0].primitive.name == "add"


def test_pytree_inputs():
    """Test that for_loop works with pytree inputs and outputs."""

    @qml.for_loop(1, 7, 2)
    def f(i, x):
        return {"x": i + x["x"]}

    x = {"x": 0}
    out = f(x)
    assert list(out.keys()) == ["x"]
    assert qml.math.allclose(out["x"], 9)  # 1 + 3 + 5 = 9
