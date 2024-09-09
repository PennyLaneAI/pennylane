# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Tests for capturing differentiation into jaxpr.
"""
import pytest

import pennylane as qml
from pennylane.capture import qnode_prim

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import grad_prim  # pylint: disable=wrong-import-position

jnp = jax.numpy


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


@pytest.mark.parametrize("kwargs", [{"method": "fd"}, {"h": 0.3}, {"h": 0.2, "method": "fd"}])
def test_error_with_method_or_h(kwargs):
    """Test that an error is raised if kwargs for QJIT's grad are passed to PLxPRs grad."""

    def func(x):
        return qml.grad(jnp.sin, **kwargs)(x)

    method = kwargs.get("method", None)
    h = kwargs.get("h", None)
    jaxpr = jax.make_jaxpr(func)(0.6)
    with pytest.raises(ValueError, match=f"'{method=}' and '{h=}' without QJIT"):
        func(0.6)
    with pytest.raises(ValueError, match=f"'{method=}' and '{h=}' without QJIT"):
        jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.6)


def test_error_with_non_scalar_function():
    """Test that an error is raised if the differentiated function has non-scalar outputs."""
    with pytest.raises(TypeError, match="Grad only applies to scalar-output functions."):
        jax.make_jaxpr(qml.grad(jnp.sin))(jnp.array([0.5, 0.2]))


def grad_eqn_assertions(eqn, argnum=None, n_consts=0):
    argnum = [0] if argnum is None else argnum
    assert eqn.primitive == grad_prim
    assert set(eqn.params.keys()) == {"argnum", "n_consts", "jaxpr", "method", "h"}
    assert eqn.params["argnum"] == argnum
    assert eqn.params["n_consts"] == n_consts
    assert eqn.params["method"] is None
    assert eqn.params["h"] is None


@pytest.mark.parametrize("x64_mode", (True, False))
@pytest.mark.parametrize("argnum", ([0, 1], [0], [1], 0, 1))
def test_classical_grad(x64_mode, argnum):
    """Test that the qml.grad primitive can be captured with classical nodes."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)
    fdtype = jnp.float64 if x64_mode else jnp.float32

    def inner_func(x, y):
        return jnp.prod(jnp.sin(x) * jnp.cos(y) ** 2)

    def func_qml(x):
        return qml.grad(inner_func, argnum=argnum)(x, 0.4 * jnp.sqrt(x))

    def func_jax(x):
        return jax.grad(inner_func, argnums=argnum)(x, 0.4 * jnp.sqrt(x))

    x = 0.7
    jax_out = func_jax(x)
    assert qml.math.allclose(func_qml(x), jax_out)

    # Check overall jaxpr properties
    if isinstance(argnum, int):
        argnum = [argnum]
    jaxpr = jax.make_jaxpr(func_qml)(x)
    assert jaxpr.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
    assert len(jaxpr.eqns) == 3
    assert jaxpr.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)] * len(argnum)

    grad_eqn = jaxpr.eqns[2]
    grad_eqn_assertions(grad_eqn, argnum=argnum)
    assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
    assert len(grad_eqn.params["jaxpr"].eqns) == 6  # 5 numeric eqns, 1 conversion eqn

    manual_eval = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
    assert qml.math.allclose(manual_eval, jax_out)

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
def test_nested_grad(x64_mode):
    """Test that nested qml.grad primitives can be captured.
    We use the function
    f(x) = sin(x)^3
    f'(x) = 3 sin(x)^2 cos(x)
    f''(x) = 6 sin(x) cos(x)^2 - 3 sin(x)^3
    f'''(x) = 6 cos(x)^3 - 12 sin(x)^2 cos(x) - 9 sin(x)^2 cos(x)
    """
    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)
    fdtype = jnp.float64 if x64_mode else jnp.float32

    def func(x):
        return jnp.sin(x) ** 3

    x = 0.7

    # 1st order
    qml_func_1 = qml.grad(func)
    expected_1 = 3 * jnp.sin(x) ** 2 * jnp.cos(x)
    assert qml.math.allclose(qml_func_1(x), expected_1)

    jaxpr_1 = jax.make_jaxpr(qml_func_1)(x)
    assert jaxpr_1.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
    assert len(jaxpr_1.eqns) == 1
    assert jaxpr_1.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

    grad_eqn = jaxpr_1.eqns[0]
    assert [var.aval for var in grad_eqn.outvars] == jaxpr_1.out_avals
    grad_eqn_assertions(grad_eqn)
    assert len(grad_eqn.params["jaxpr"].eqns) == 2

    manual_eval_1 = jax.core.eval_jaxpr(jaxpr_1.jaxpr, jaxpr_1.consts, x)
    assert qml.math.allclose(manual_eval_1, expected_1)

    # 2nd order
    qml_func_2 = qml.grad(qml_func_1)
    expected_2 = 6 * jnp.sin(x) * jnp.cos(x) ** 2 - 3 * jnp.sin(x) ** 3
    assert qml.math.allclose(qml_func_2(x), expected_2)

    jaxpr_2 = jax.make_jaxpr(qml_func_2)(x)
    assert jaxpr_2.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
    assert len(jaxpr_2.eqns) == 1
    assert jaxpr_2.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

    grad_eqn = jaxpr_2.eqns[0]
    assert [var.aval for var in grad_eqn.outvars] == jaxpr_2.out_avals
    grad_eqn_assertions(grad_eqn)
    assert len(grad_eqn.params["jaxpr"].eqns) == 1  # inner grad equation
    assert grad_eqn.params["jaxpr"].eqns[0].primitive == grad_prim

    manual_eval_2 = jax.core.eval_jaxpr(jaxpr_2.jaxpr, jaxpr_2.consts, x)
    assert qml.math.allclose(manual_eval_2, expected_2)

    # 3rd order
    qml_func_3 = qml.grad(qml_func_2)
    expected_3 = (
        6 * jnp.cos(x) ** 3 - 12 * jnp.sin(x) ** 2 * jnp.cos(x) - 9 * jnp.sin(x) ** 2 * jnp.cos(x)
    )

    assert qml.math.allclose(qml_func_3(x), expected_3)

    jaxpr_3 = jax.make_jaxpr(qml_func_3)(x)
    assert jaxpr_3.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
    assert len(jaxpr_3.eqns) == 1
    assert jaxpr_3.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

    grad_eqn = jaxpr_3.eqns[0]
    assert [var.aval for var in grad_eqn.outvars] == jaxpr_3.out_avals
    grad_eqn_assertions(grad_eqn)
    assert len(grad_eqn.params["jaxpr"].eqns) == 1  # inner grad equation
    assert grad_eqn.params["jaxpr"].eqns[0].primitive == grad_prim

    manual_eval_3 = jax.core.eval_jaxpr(jaxpr_3.jaxpr, jaxpr_3.consts, x)
    assert qml.math.allclose(manual_eval_3, expected_3)

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
@pytest.mark.parametrize("diff_method", ("backprop", "parameter-shift"))
def test_grad_of_simple_qnode(x64_mode, diff_method, mocker):
    """Test capturing the gradient of a simple qnode."""
    # pylint: disable=protected-access
    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)
    fdtype = jax.numpy.float64 if x64_mode else jax.numpy.float32

    dev = qml.device("default.qubit", wires=4)

    @qml.grad
    @qml.qnode(dev, diff_method=diff_method)
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1] ** 2, wires=0)
        return qml.expval(qml.Z(0))

    x = jnp.array([0.5, 0.9])
    res = circuit(x)
    expected_res = (
        -jnp.sin(x[0]) * jnp.cos(x[1] ** 2),
        -2 * x[1] * jnp.sin(x[1] ** 2) * jnp.cos(x[0]),
    )
    assert qml.math.allclose(res, expected_res)

    jaxpr = jax.make_jaxpr(circuit)(x)

    assert len(jaxpr.eqns) == 1  # grad equation
    assert jaxpr.in_avals == [jax.core.ShapedArray((2,), fdtype)]
    assert jaxpr.out_avals == [jax.core.ShapedArray((2,), fdtype)]

    grad_eqn = jaxpr.eqns[0]
    assert grad_eqn.invars[0].aval == jaxpr.in_avals[0]
    grad_eqn_assertions(grad_eqn)
    grad_jaxpr = grad_eqn.params["jaxpr"]
    assert len(grad_jaxpr.eqns) == 1  # qnode equation

    qnode_eqn = grad_jaxpr.eqns[0]
    assert qnode_eqn.primitive == qnode_prim
    assert qnode_eqn.invars[0].aval == jaxpr.in_avals[0]

    qfunc_jaxpr = qnode_eqn.params["qfunc_jaxpr"]
    # Skipping a few equations related to indexing and preprocessing
    assert qfunc_jaxpr.eqns[2].primitive == qml.RX._primitive
    assert qfunc_jaxpr.eqns[6].primitive == qml.RY._primitive
    assert qfunc_jaxpr.eqns[7].primitive == qml.Z._primitive
    assert qfunc_jaxpr.eqns[8].primitive == qml.measurements.ExpectationMP._obs_primitive

    assert len(qnode_eqn.outvars) == 1
    assert qnode_eqn.outvars[0].aval == jax.core.ShapedArray((), fdtype)

    assert len(grad_eqn.outvars) == 1
    assert grad_eqn.outvars[0].aval == jax.core.ShapedArray((2,), fdtype)

    spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")
    manual_res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
    if diff_method == "parameter-shift":
        spy.assert_called_once()
    else:
        spy.assert_not_called()
    assert qml.math.allclose(manual_res, expected_res)

    jax.config.update("jax_enable_x64", initial_mode)
