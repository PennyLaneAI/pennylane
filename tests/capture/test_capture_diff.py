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
from pennylane._grad import _get_grad_prim
from pennylane.capture import qnode_prim

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jnp = jax.numpy

grad_prim = _get_grad_prim()


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
    with pytest.raises(ValueError, match=f"'{method=}' and '{h=}' without QJIT"):
        jax.make_jaxpr(func)(0.6)


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
    assert grad_eqn.primitive == grad_prim
    assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
    assert set(grad_eqn.params.keys()) == {"argnum", "n_consts", "jaxpr"}
    assert grad_eqn.params["argnum"] == argnum
    assert grad_eqn.params["n_consts"] == 0
    assert len(grad_eqn.params["jaxpr"].eqns) == 6  # 5 numeric eqns, 1 conversion eqn

    manual_eval = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
    assert qml.math.allclose(manual_eval, jax_out)

    jax.config.update("jax_enable_x64", initial_mode)


@pytest.mark.parametrize("x64_mode", (True, False))
def test_grad_of_simple_qnode(x64_mode):
    """Test capturing the gradient of a simple qnode."""

    initial_mode = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", x64_mode)
    fdtype = jax.numpy.float64 if x64_mode else jax.numpy.float32

    dev = qml.device("default.qubit", wires=4)

    @qml.grad
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=0)
        return qml.expval(qml.Z(0))

    x = jnp.array([0.5, 0.9])
    res = circuit(x)
    expected_res = (-jnp.sin(x[0]) * jnp.cos(x[1]), -jnp.sin(x[1]) * jnp.cos(x[0]))
    assert qml.math.allclose(res, expected_res)

    jaxpr = jax.make_jaxpr(circuit)(x)

    assert len(jaxpr.eqns) == 1  # grad equation
    assert jaxpr.in_avals == [jax.core.ShapedArray((2,), fdtype)]
    assert jaxpr.out_avals == [jax.core.ShapedArray((2,), fdtype)]

    grad_eqn = jaxpr.eqns[0]
    assert grad_eqn.primitive == grad_prim
    assert grad_eqn.invars[0].aval == jaxpr.in_avals[0]
    assert set(grad_eqn.params.keys()) == {"argnum", "n_consts", "jaxpr"}
    assert grad_eqn.params["argnum"] == [0]
    assert grad_eqn.params["n_consts"] == 0

    grad_jaxpr = grad_eqn.params["jaxpr"]
    assert len(grad_jaxpr.eqns) == 1  # qnode equation

    qnode_eqn = grad_jaxpr.eqns[0]
    assert qnode_eqn.primitive == qnode_prim
    assert qnode_eqn.invars[0].aval == jaxpr.in_avals[0]

    qfunc_jaxpr = qnode_eqn.params["qfunc_jaxpr"]
    # Skipping a few equations related to indexing
    assert qfunc_jaxpr.eqns[2].primitive == qml.RX._primitive
    assert qfunc_jaxpr.eqns[5].primitive == qml.RY._primitive
    assert qfunc_jaxpr.eqns[6].primitive == qml.Z._primitive
    assert qfunc_jaxpr.eqns[7].primitive == qml.measurements.ExpectationMP._obs_primitive

    assert len(qnode_eqn.outvars) == 1
    assert qnode_eqn.outvars[0].aval == jax.core.ShapedArray((), fdtype)

    assert len(grad_eqn.outvars) == 1
    assert grad_eqn.outvars[0].aval == jax.core.ShapedArray((2,), fdtype)

    manual_res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
    assert qml.math.allclose(manual_res, expected_res)

    jax.config.update("jax_enable_x64", initial_mode)
