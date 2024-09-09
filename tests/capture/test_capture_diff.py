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

from pennylane.capture.primitives import (  # pylint: disable=wrong-import-position
    grad_prim,
    jacobian_prim,
)

jnp = jax.numpy


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


class TestExceptions:
    """Test that expected exceptions are correctly raised."""

    @pytest.mark.parametrize("kwargs", [{"method": "fd"}, {"h": 0.3}, {"h": 0.2, "method": "fd"}])
    @pytest.mark.parametrize("diff", [qml.grad, qml.jacobian])
    def test_error_with_method_or_h(self, kwargs, diff):
        """Test that an error is raised if kwargs for QJIT's grad are passed to PLxPRs grad."""

        def func(x):
            return diff(jnp.sin, **kwargs)(x)

        method = kwargs.get("method", None)
        h = kwargs.get("h", None)
        jaxpr = jax.make_jaxpr(func)(0.6)
        with pytest.raises(ValueError, match=f"'{method=}' and '{h=}' without QJIT"):
            func(0.6)
        with pytest.raises(ValueError, match=f"'{method=}' and '{h=}' without QJIT"):
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.6)

    def test_error_with_non_scalar_function(self):
        """Test that an error is raised if the differentiated function has non-scalar outputs."""
        with pytest.raises(TypeError, match="Grad only applies to scalar-output functions."):
            jax.make_jaxpr(qml.grad(jnp.sin))(jnp.array([0.5, 0.2]))


def diff_eqn_assertions(eqn, primitive, argnum=None, n_consts=0):
    argnum = [0] if argnum is None else argnum
    assert eqn.primitive == primitive
    assert set(eqn.params.keys()) == {"argnum", "n_consts", "jaxpr", "method", "h"}
    assert eqn.params["argnum"] == argnum
    assert eqn.params["n_consts"] == n_consts
    assert eqn.params["method"] is None
    assert eqn.params["h"] is None


@pytest.mark.parametrize("x64_mode", (True, False))
class TestGrad:
    """Tests for capturing `qml.grad`."""

    @pytest.mark.parametrize("argnum", ([0, 1], [0], [1], 0, 1))
    def test_classical_grad(self, x64_mode, argnum):
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
        diff_eqn_assertions(grad_eqn, grad_prim, argnum=argnum)
        assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
        assert len(grad_eqn.params["jaxpr"].eqns) == 6  # 5 numeric eqns, 1 conversion eqn

        manual_eval = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert qml.math.allclose(manual_eval, jax_out)

        jax.config.update("jax_enable_x64", initial_mode)

    def test_nested_grad(self, x64_mode):
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
        diff_eqn_assertions(grad_eqn, grad_prim)
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
        diff_eqn_assertions(grad_eqn, grad_prim)
        assert len(grad_eqn.params["jaxpr"].eqns) == 1  # inner grad equation
        assert grad_eqn.params["jaxpr"].eqns[0].primitive == grad_prim

        manual_eval_2 = jax.core.eval_jaxpr(jaxpr_2.jaxpr, jaxpr_2.consts, x)
        assert qml.math.allclose(manual_eval_2, expected_2)

        # 3rd order
        qml_func_3 = qml.grad(qml_func_2)
        expected_3 = (
            6 * jnp.cos(x) ** 3
            - 12 * jnp.sin(x) ** 2 * jnp.cos(x)
            - 9 * jnp.sin(x) ** 2 * jnp.cos(x)
        )

        assert qml.math.allclose(qml_func_3(x), expected_3)

        jaxpr_3 = jax.make_jaxpr(qml_func_3)(x)
        assert jaxpr_3.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
        assert len(jaxpr_3.eqns) == 1
        assert jaxpr_3.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

        grad_eqn = jaxpr_3.eqns[0]
        assert [var.aval for var in grad_eqn.outvars] == jaxpr_3.out_avals
        diff_eqn_assertions(grad_eqn, grad_prim)
        assert len(grad_eqn.params["jaxpr"].eqns) == 1  # inner grad equation
        assert grad_eqn.params["jaxpr"].eqns[0].primitive == grad_prim

        manual_eval_3 = jax.core.eval_jaxpr(jaxpr_3.jaxpr, jaxpr_3.consts, x)
        assert qml.math.allclose(manual_eval_3, expected_3)

        jax.config.update("jax_enable_x64", initial_mode)

    @pytest.mark.parametrize("diff_method", ("backprop", "parameter-shift"))
    def test_grad_of_simple_qnode(self, x64_mode, diff_method, mocker):
        """Test capturing the gradient of a simple qnode."""
        # pylint: disable=protected-access
        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)
        fdtype = jax.numpy.float64 if x64_mode else jax.numpy.float32

        dev = qml.device("default.qubit", wires=2)

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
        diff_eqn_assertions(grad_eqn, grad_prim)
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


def _jac_allclose(jac1, jac2, num_axes, atol=1e-8):
    """Test that two Jacobians, given as nested sequences of arrays, are equal."""
    if num_axes == 0:
        return qml.math.allclose(jac1, jac2, atol=atol)
    if len(jac1) != len(jac2):
        return False
    return all(
        _jac_allclose(_jac1, _jac2, num_axes - 1, atol=atol) for _jac1, _jac2 in zip(jac1, jac2)
    )


@pytest.mark.parametrize("x64_mode", (True, False))
class TestJacobian:
    """Tests for capturing `qml.jacobian`."""

    @pytest.mark.parametrize("argnum", ([0, 1], [0], [1], 0, 1))
    def test_classical_jacobian(self, x64_mode, argnum):
        """Test that the qml.jacobian primitive can be captured with classical nodes."""
        if isinstance(argnum, list) and len(argnum) > 1:
            # These cases will only be unlocked with Pytree support
            pytest.xfail()

        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)
        fdtype = jnp.float64 if x64_mode else jnp.float32

        def shaped_array(shape):
            """Make a ShapedArray with a given shape."""
            return jax.core.ShapedArray(shape, fdtype)

        def inner_func(x, y):
            """A function with output signature
            (4,), (2, 3) -> (2,), (4, 3), ()
            """
            return (
                x[0:2] * y[:, 1],
                jnp.outer(x, y[0]).astype(jnp.float32),
                jnp.prod(y) - jnp.sum(x),
            )

        x = jnp.array([0.3, 0.2, 0.1, 0.6])
        y = jnp.array([[0.4, -0.7, 0.2], [1.2, -7.2, 0.2]])
        func_qml = qml.jacobian(inner_func, argnum=argnum)
        func_jax = jax.jacobian(inner_func, argnums=argnum)

        jax_out = func_jax(x, y)
        num_axes = 1 if isinstance(argnum, int) else 2
        assert _jac_allclose(func_qml(x, y), jax_out, num_axes)

        # Check overall jaxpr properties
        jaxpr = jax.make_jaxpr(func_jax)(x, y)
        jaxpr = jax.make_jaxpr(func_qml)(x, y)

        if isinstance(argnum, int):
            argnum = [argnum]

        exp_in_avals = [shaped_array(shape) for shape in [(4,), (2, 3)]]
        # Expected Jacobian shapes for argnum=[0, 1]
        exp_out_shapes = [[(2, 4), (2, 2, 3)], [(4, 3, 4), (4, 3, 2, 3)], [(4,), (2, 3)]]
        # Slice out shapes corresponding to the actual argnum
        exp_out_avals = [shaped_array(shapes[i]) for shapes in exp_out_shapes for i in argnum]

        assert jaxpr.in_avals == exp_in_avals
        assert len(jaxpr.eqns) == 1
        assert jaxpr.out_avals == exp_out_avals

        jac_eqn = jaxpr.eqns[0]
        diff_eqn_assertions(jac_eqn, jacobian_prim, argnum=argnum)

        manual_eval = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, y)
        assert _jac_allclose(manual_eval, jax_out, num_axes)

        jax.config.update("jax_enable_x64", initial_mode)

    def test_nested_jacobian(self, x64_mode):
        r"""Test that nested qml.jacobian primitives can be captured.
        We use the function
        f(x) = (prod(x) * sin(x), sum(x**2))
        f'(x) = (prod(x)/x_i * sin(x) + prod(x) cos(x) e_i, 2 x_i)
        f''(x) = | (prod(x)/x_i x_j * sin(x) + prod(x)cos(x) (e_j/x_i + e_i/x_j)
                 | - prod(x) sin(x) e_i e_j, 0)                              for i != j
                 |
                 | (2 prod(x)/x_i * cos(x) e_i - prod(x) sin(x) e_i e_i, 2)  for i = j
        """
        # pylint: disable=too-many-statements
        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)
        fdtype = jnp.float64 if x64_mode else jnp.float32

        def func(x):
            return jnp.prod(x) * jnp.sin(x), jnp.sum(x**2)

        x = jnp.array([0.7, -0.9, 0.6, 0.3])
        x = x[:1]
        dim = len(x)
        eye = jnp.eye(dim)

        # 1st order
        qml_func_1 = qml.jacobian(func)
        prod_sin = jnp.prod(x) * jnp.sin(x)
        prod_cos_e_i = jnp.prod(x) * jnp.cos(x) * eye
        expected_1 = (prod_sin[:, None] / x[None, :] + prod_cos_e_i, 2 * x)
        assert _jac_allclose(qml_func_1(x), expected_1, 1)

        jaxpr_1 = jax.make_jaxpr(qml_func_1)(x)
        assert jaxpr_1.in_avals == [jax.core.ShapedArray((dim,), fdtype)]
        assert len(jaxpr_1.eqns) == 1
        assert jaxpr_1.out_avals == [
            jax.core.ShapedArray(sh, fdtype) for sh in [(dim, dim), (dim,)]
        ]

        jac_eqn = jaxpr_1.eqns[0]
        assert [var.aval for var in jac_eqn.outvars] == jaxpr_1.out_avals
        diff_eqn_assertions(jac_eqn, jacobian_prim)
        assert len(jac_eqn.params["jaxpr"].eqns) == 5

        manual_eval_1 = jax.core.eval_jaxpr(jaxpr_1.jaxpr, jaxpr_1.consts, x)
        assert _jac_allclose(manual_eval_1, expected_1, 1)

        # 2nd order
        qml_func_2 = qml.jacobian(qml_func_1)
        expected_2 = (
            prod_sin[:, None, None] / x[None, :, None] / x[None, None, :]
            + prod_cos_e_i[:, :, None] / x[None, None, :]
            + prod_cos_e_i[:, None, :] / x[None, :, None]
            - jnp.tensordot(prod_sin, eye + eye / x**2, axes=0),
            jnp.tensordot(jnp.ones(dim), eye * 2, axes=0),
        )
        # Output only has one tuple axis
        assert _jac_allclose(qml_func_2(x), expected_2, 1)

        jaxpr_2 = jax.make_jaxpr(qml_func_2)(x)
        assert jaxpr_2.in_avals == [jax.core.ShapedArray((dim,), fdtype)]
        assert len(jaxpr_2.eqns) == 1
        assert jaxpr_2.out_avals == [
            jax.core.ShapedArray(sh, fdtype) for sh in [(dim, dim, dim), (dim, dim)]
        ]

        jac_eqn = jaxpr_2.eqns[0]
        assert [var.aval for var in jac_eqn.outvars] == jaxpr_2.out_avals
        diff_eqn_assertions(jac_eqn, jacobian_prim)
        assert len(jac_eqn.params["jaxpr"].eqns) == 1  # inner jacobian equation
        assert jac_eqn.params["jaxpr"].eqns[0].primitive == jacobian_prim

        manual_eval_2 = jax.core.eval_jaxpr(jaxpr_2.jaxpr, jaxpr_2.consts, x)
        assert _jac_allclose(manual_eval_2, expected_2, 1)

        jax.config.update("jax_enable_x64", initial_mode)

    @pytest.mark.parametrize("diff_method", ("backprop", "parameter-shift"))
    def test_jacobian_of_simple_qnode(self, x64_mode, diff_method, mocker):
        """Test capturing the gradient of a simple qnode."""
        # pylint: disable=protected-access
        initial_mode = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", x64_mode)
        fdtype = jax.numpy.float64 if x64_mode else jax.numpy.float32

        dev = qml.device("default.qubit", wires=2)

        # Note the decorator
        @qml.jacobian
        @qml.qnode(dev, diff_method=diff_method)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            return qml.expval(qml.Z(0)), qml.probs(0)

        x = jnp.array([0.5, 0.9])
        res = circuit(x)
        expval_diff = -jnp.sin(x) * jnp.cos(x[::-1])
        expected_res = (expval_diff, jnp.stack([expval_diff / 2, -expval_diff / 2]))

        assert _jac_allclose(res, expected_res, 1)

        jaxpr = jax.make_jaxpr(circuit)(x)

        assert len(jaxpr.eqns) == 1  # Jacobian equation
        assert jaxpr.in_avals == [jax.core.ShapedArray((2,), fdtype)]
        assert jaxpr.out_avals == [jax.core.ShapedArray(sh, fdtype) for sh in [(2,), (2, 2)]]

        jac_eqn = jaxpr.eqns[0]
        assert jac_eqn.invars[0].aval == jaxpr.in_avals[0]
        diff_eqn_assertions(jac_eqn, jacobian_prim)
        jac_jaxpr = jac_eqn.params["jaxpr"]
        assert len(jac_jaxpr.eqns) == 1  # qnode equation

        qnode_eqn = jac_jaxpr.eqns[0]
        assert qnode_eqn.primitive == qnode_prim
        assert qnode_eqn.invars[0].aval == jaxpr.in_avals[0]

        qfunc_jaxpr = qnode_eqn.params["qfunc_jaxpr"]
        # Skipping a few equations related to indexing
        assert qfunc_jaxpr.eqns[2].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[5].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[6].primitive == qml.Z._primitive
        assert qfunc_jaxpr.eqns[7].primitive == qml.measurements.ExpectationMP._obs_primitive

        assert len(qnode_eqn.outvars) == 2
        assert qnode_eqn.outvars[0].aval == jax.core.ShapedArray((), fdtype)
        assert qnode_eqn.outvars[1].aval == jax.core.ShapedArray((2,), fdtype)

        assert [outvar.aval for outvar in jac_eqn.outvars] == jaxpr.out_avals

        spy = mocker.spy(qml.gradients.parameter_shift, "expval_param_shift")
        manual_res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        if diff_method == "parameter-shift":
            spy.assert_called_once()
        else:
            spy.assert_not_called()
        assert _jac_allclose(manual_res, expected_res, 1)

        jax.config.update("jax_enable_x64", initial_mode)
