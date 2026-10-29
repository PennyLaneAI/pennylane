# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for capturing value_and_grad into jaxpr.

Note some tests on the errors exist in test_capture_diff.py
"""

from functools import partial

import pytest

import pennylane as qp

pytestmark = pytest.mark.capture

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# pylint: disable=wrong-import-position
from pennylane.capture.autograph import autograph_source, disable_autograph, run_autograph
from pennylane.capture.primitives import cond_prim, qnode_prim, value_and_grad_prim


def test_value_and_grad_error_with_non_scalar_function():
    """Test that an error is raised if the differentiated function has non-scalar outputs."""
    with pytest.raises(
        TypeError, match="value_and_grad only defined for scalar-output functions. "
    ):
        jax.make_jaxpr(qp.value_and_grad(jnp.sin))(jnp.array([0.5, 0.2]))

    def f(x):
        return (x, x)

    with pytest.raises(
        TypeError, match="value_and_grad only defined for scalar-output functions. "
    ):
        jax.make_jaxpr(qp.value_and_grad(f))(0.5)


def diff_eqn_assertions(eqn, argnums=None, fn=None):
    """Perform basic checks on a value_and_grad equation."""
    argnums = (0,) if argnums is None else argnums
    assert eqn.primitive == value_and_grad_prim
    assert set(eqn.params.keys()) == {
        "argnums",
        "jaxpr",
        "method",
        "h",
        "fn",
    }
    assert eqn.params["argnums"] == tuple(argnums)
    assert eqn.params["method"] == "auto"
    assert eqn.params["h"] == 1e-6
    assert eqn.params["fn"] == fn


@pytest.mark.parametrize("argnums", ([0, 1], [0], [1], 0, 1))
def test_classical_func(argnums):
    """Test taking the value_and_grad of a classical function."""

    def inner_func(x, y):
        return jnp.prod(jnp.sin(x) * jnp.cos(y) ** 2)

    def workflow(x):
        return qp.value_and_grad(inner_func, argnums=argnums)(x, 0.5 * jnp.sqrt(x))

    def workflow_jax(x):
        return jax.value_and_grad(inner_func, argnums=argnums)(x, 0.5 * jnp.sqrt(x))

    x = 0.4
    jax_res, jax_grad = workflow_jax(x)
    res, g = workflow(x)
    assert qp.math.allclose(res, jax_res)
    assert qp.math.allclose(g, jax_grad)

    jaxpr = jax.make_jaxpr(workflow)(x)
    assert jaxpr.in_avals == [jax.core.ShapedArray((), float, weak_type=True)]
    assert len(jaxpr.eqns) == 3

    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)
    assert jaxpr.out_avals[0] == jax.core.ShapedArray((), float)
    assert jaxpr.out_avals[1:] == [jax.core.ShapedArray((), float, weak_type=True)] * len(argnums)

    grad_eqn = jaxpr.eqns[2]
    diff_eqn_assertions(grad_eqn, argnums=argnums, fn=inner_func)
    assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
    assert len(grad_eqn.params["jaxpr"].eqns) == 6  # 5 numeric eqns, 1 conversion eqn

    manual_eval = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
    assert qp.math.allclose(manual_eval[0], jax_res)
    assert qp.math.allclose(manual_eval[1:], jax_grad)


def test_nested_value_and_grad():
    """Test that value_and_grad can be nested."""
    fdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    def func(x):
        return jnp.sin(x) ** 3

    x = 0.654

    qp_func_1 = qp.value_and_grad(func)
    expected_1 = 3 * jnp.sin(x) ** 2 * jnp.cos(x)
    r, g = qp_func_1(x)
    assert qp.math.allclose(r, func(x))
    assert qp.math.allclose(g, expected_1)

    jaxpr_1 = jax.make_jaxpr(qp_func_1)(x)
    assert jaxpr_1.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
    assert len(jaxpr_1.eqns) == 1
    assert jaxpr_1.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)] * 2

    def just_grad_part(*args):
        return qp_func_1(*args)[1]

    qp_func_2 = qp.value_and_grad(just_grad_part)

    jaxpr_2 = jax.make_jaxpr(qp_func_2)(x)
    assert jaxpr_2.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
    assert len(jaxpr_2.eqns) == 1
    assert jaxpr_2.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)] * 2

    grad_eqn = jaxpr_2.eqns[0]
    assert [var.aval for var in grad_eqn.outvars] == jaxpr_2.out_avals
    diff_eqn_assertions(grad_eqn, fn=just_grad_part)
    assert len(grad_eqn.params["jaxpr"].eqns) == 1  # inner grad equation
    assert grad_eqn.params["jaxpr"].eqns[0].primitive == value_and_grad_prim


@pytest.mark.parametrize("argnums", ([0, 1], [0], [1]))
def test_pytree_input(argnums):
    """Test that the value_and_grad primitive can be captured with pytree inputs."""

    fdtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32

    def inner_func(x, y):
        return jnp.prod(jnp.sin(x["a"]) * jnp.cos(y[0]["b"][1]) ** 2)

    def func_qp(x):
        return qp.value_and_grad(inner_func, argnums=argnums)(
            {"a": x}, ({"b": [None, 0.4 * jnp.sqrt(x)]},)
        )

    def func_jax(x):
        return jax.value_and_grad(inner_func, argnums=argnums)(
            {"a": x}, ({"b": [None, 0.4 * jnp.sqrt(x)]},)
        )

    x = 0.7
    jax_res, jax_grad = func_jax(x)
    jax_out_flat, jax_out_tree = jax.tree_util.tree_flatten(jax_grad)

    res, grad = func_qp(x)
    qp_out_flat, qp_out_tree = jax.tree_util.tree_flatten(grad)
    assert jax_out_tree == qp_out_tree
    assert qp.math.allclose(jax_out_flat, qp_out_flat)
    assert qp.math.allclose(jax_res, res)

    # Check overall jaxpr properties
    jaxpr = jax.make_jaxpr(func_qp)(x)
    assert jaxpr.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
    assert len(jaxpr.eqns) == 3
    argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    assert jaxpr.out_avals[0] == jax.core.ShapedArray((), fdtype)
    assert jaxpr.out_avals[1:] == [jax.core.ShapedArray((), fdtype, weak_type=True)] * len(argnums)

    grad_eqn = jaxpr.eqns[2]
    diff_eqn_assertions(grad_eqn, argnums=argnums, fn=inner_func)
    assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
    assert len(grad_eqn.params["jaxpr"].eqns) == 6  # 5 numeric eqns, 1 conversion eqn

    manual_out = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
    grad_out_flat, grad_out_tree = jax.tree_util.tree_flatten(manual_out[1:])
    # Assert that the output from the manual evaluation is flat
    assert grad_out_tree == jax.tree_util.tree_flatten(grad_out_flat)[1]
    assert qp.math.allclose(jax_out_flat, grad_out_flat)


class TestAutograph:
    """Tests for AutoGraph conversion of value_and_grad wrappers."""

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("condition", [True, False])
    def test_existing_instance(self, nested, condition):
        """Existing wrappers convert dynamic control flow and preserve differentiation settings."""

        def function(predicate, data):
            if predicate:
                result = data**2
            else:
                result = data**3
            return result

        original = qp.value_and_grad(function, argnums=1, method="fd", h=0.125)

        if nested:

            def caller(predicate, data):
                return original(predicate, data)

            converted = run_autograph(caller)
        else:
            converted = run_autograph(original)
            assert isinstance(converted, qp.value_and_grad)

        jaxpr = jax.make_jaxpr(converted)(condition, 2.0)
        equation = jaxpr.eqns[0]
        assert equation.primitive == value_and_grad_prim
        assert equation.params["argnums"] == (1,)
        assert equation.params["method"] == "fd"
        assert equation.params["h"] == 0.125
        assert len(equation.outvars) == 2
        assert cond_prim in [eqn.primitive for eqn in equation.params["jaxpr"].eqns]
        assert original.func is function
        assert not hasattr(function, "ag_unconverted")
        assert "ag__.if_stmt" in autograph_source(original)
        if not nested:
            assert "ag__.if_stmt" in autograph_source(converted)

    @pytest.mark.parametrize("condition", [True, False])
    def test_multiple_arguments(self, condition):
        """Capture preserves the conditional value and gradients for multiple arguments."""

        def function(predicate, data, scale):
            if predicate:
                result = scale * data**2
            else:
                result = scale * data**3
            return result

        converted = run_autograph(qp.value_and_grad(function, argnums=(1, 2)))
        jaxpr = jax.make_jaxpr(converted)(condition, 2.0, 3.0)
        equation = jaxpr.eqns[0]
        assert equation.primitive == value_and_grad_prim
        assert equation.params["argnums"] == (1, 2)
        assert len(equation.outvars) == 3
        assert all(var.aval.shape == () for var in equation.outvars)
        assert cond_prim in [eqn.primitive for eqn in equation.params["jaxpr"].eqns]
        value = jax.core.eval_jaxpr(equation.params["jaxpr"], [], condition, 2.0, 3.0)
        assert qp.math.allclose(value, [12.0 if condition else 24.0])

    @pytest.mark.parametrize("nested", [False, True])
    def test_qnode(self, nested):
        """Conversion follows a value_and_grad wrapper into its QNode's quantum function."""

        @qp.qnode(qp.device("lightning.qubit", wires=1), diff_method="parameter-shift")
        def circuit(condition, data):
            if condition:
                qp.RY(data, wires=0)
            return qp.expval(qp.Z(0))

        original = qp.value_and_grad(circuit, argnums=1)
        original_function = circuit.func

        if nested:

            def caller(condition, data):
                return original(condition, data)

            converted = run_autograph(caller)
        else:
            converted = run_autograph(original)
            assert converted.func is not circuit
            assert converted.func.device is circuit.device
            assert converted.func.diff_method == circuit.diff_method

        jaxpr = jax.make_jaxpr(converted)(True, 0.5)
        equation = jaxpr.eqns[0]
        assert equation.primitive == value_and_grad_prim
        assert len(equation.outvars) == 2
        qnode_eqn = equation.params["jaxpr"].eqns[0]
        assert qnode_eqn.primitive == qnode_prim
        assert cond_prim in [eqn.primitive for eqn in qnode_eqn.params["qfunc_jaxpr"].eqns]
        assert circuit.func is original_function
        assert original.func is circuit
        assert not hasattr(original_function, "ag_unconverted")
        assert "ag__.if_stmt" in autograph_source(original)

    def test_decorator_stack(self):
        """AutoGraph outside the value_and_grad and QNode decorators converts conditionals."""

        @run_autograph
        @partial(qp.value_and_grad, argnums=1)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(condition, data):
            if condition:
                qp.RY(data, wires=0)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(True, 0.5)
        equation = jaxpr.eqns[0]
        assert equation.primitive == value_and_grad_prim
        qnode_eqn = equation.params["jaxpr"].eqns[0]
        assert qnode_eqn.primitive == qnode_prim
        assert cond_prim in [eqn.primitive for eqn in qnode_eqn.params["qfunc_jaxpr"].eqns]
        assert "ag__.if_stmt" in autograph_source(circuit)

    def test_constructor_in_converted_function(self):
        """A wrapper constructed by converted code converts its target when called afterwards."""

        def function(condition, data):
            if condition:
                result = data**2
            else:
                result = data**3
            return result

        @run_autograph
        def factory():
            return qp.value_and_grad(function, argnums=1)

        derivative = factory()
        jaxpr = jax.make_jaxpr(derivative)(True, 2.0)
        equation = jaxpr.eqns[0]
        assert equation.primitive == value_and_grad_prim
        assert equation.params["argnums"] == (1,)
        assert cond_prim in [eqn.primitive for eqn in equation.params["jaxpr"].eqns]
        assert "ag__.if_stmt" in autograph_source(function)

    def test_disabled_conversion(self):
        """Existing wrapper calls respect the disabled AutoGraph context."""

        def function(condition, data):
            if condition:
                result = data**2
            else:
                result = data**3
            return result

        derivative = qp.value_and_grad(function, argnums=1)

        @run_autograph
        def caller(condition, data):
            with disable_autograph:
                return derivative(condition, data)

        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.make_jaxpr(caller)(True, 2.0)
