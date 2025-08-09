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

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (  # pylint: disable=wrong-import-position
    grad_prim,
    jacobian_prim,
    qnode_prim,
)

jnp = jax.numpy


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


class TestGrad:
    """Tests for capturing `qml.grad`."""

    @pytest.mark.parametrize("argnum", ([0, 1], [0], [1], 0, 1))
    def test_classical_grad(self, argnum):
        """Test that the qml.grad primitive can be captured with classical nodes."""

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
        jaxpr = jax.make_jaxpr(func_qml)(x)
        assert jaxpr.in_avals == [jax.core.ShapedArray((), float, weak_type=True)]
        assert len(jaxpr.eqns) == 3
        if isinstance(argnum, int):
            argnum = [argnum]
        assert jaxpr.out_avals == [jax.core.ShapedArray((), float, weak_type=True)] * len(argnum)

        grad_eqn = jaxpr.eqns[2]
        diff_eqn_assertions(grad_eqn, grad_prim, argnum=argnum)
        assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
        assert len(grad_eqn.params["jaxpr"].eqns) == 6  # 5 numeric eqns, 1 conversion eqn

        manual_eval = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert qml.math.allclose(manual_eval, jax_out)

    def test_nested_grad(self):
        """Test that nested qml.grad primitives can be captured.
        We use the function
        f(x) = sin(x)^3
        f'(x) = 3 sin(x)^2 cos(x)
        f''(x) = 6 sin(x) cos(x)^2 - 3 sin(x)^3
        f'''(x) = 6 cos(x)^3 - 12 sin(x)^2 cos(x) - 9 sin(x)^2 cos(x)
        """
        fdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

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

        jaxpr_2 = jax.make_jaxpr(qml_func_2)(x)
        assert jaxpr_2.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
        assert len(jaxpr_2.eqns) == 1
        assert jaxpr_2.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

        grad_eqn = jaxpr_2.eqns[0]
        assert [var.aval for var in grad_eqn.outvars] == jaxpr_2.out_avals
        diff_eqn_assertions(grad_eqn, grad_prim)
        assert len(grad_eqn.params["jaxpr"].eqns) == 1  # inner grad equation
        assert grad_eqn.params["jaxpr"].eqns[0].primitive == grad_prim

        # 3rd order
        qml_func_3 = qml.grad(qml_func_2)

        jaxpr_3 = jax.make_jaxpr(qml_func_3)(x)
        assert jaxpr_3.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
        assert len(jaxpr_3.eqns) == 1
        assert jaxpr_3.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]

        grad_eqn = jaxpr_3.eqns[0]
        assert [var.aval for var in grad_eqn.outvars] == jaxpr_3.out_avals
        diff_eqn_assertions(grad_eqn, grad_prim)
        assert len(grad_eqn.params["jaxpr"].eqns) == 1  # inner grad equation
        assert grad_eqn.params["jaxpr"].eqns[0].primitive == grad_prim

        # jax v0.5.3 broke this
        # expected_2 = 6 * jnp.sin(x) * jnp.cos(x) ** 2 - 3 * jnp.sin(x) ** 3
        # assert qml.math.allclose(qml_func_2(x), expected_2)

        # manual_eval_2 = jax.core.eval_jaxpr(jaxpr_2.jaxpr, jaxpr_2.consts, x)
        # assert qml.math.allclose(manual_eval_2, expected_2)

        # expected_3 = (
        #    6 * jnp.cos(x) ** 3
        #   - 12 * jnp.sin(x) ** 2 * jnp.cos(x)
        #    - 9 * jnp.sin(x) ** 2 * jnp.cos(x)
        # )

        # assert qml.math.allclose(qml_func_3(x), expected_3)
        # manual_eval_3 = jax.core.eval_jaxpr(jaxpr_3.jaxpr, jaxpr_3.consts, x)
        # assert qml.math.allclose(manual_eval_3, expected_3)

    @pytest.mark.parametrize(
        "diff_method", ("backprop", pytest.param("parameter-shift", marks=pytest.mark.xfail))
    )
    def test_grad_of_simple_qnode(self, diff_method):
        """Test capturing the gradient of a simple qnode."""
        # pylint: disable=protected-access
        fdtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32

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

        manual_res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert qml.math.allclose(manual_res, expected_res)

    @pytest.mark.parametrize("argnum", ([0, 1], [0], [1]))
    def test_grad_pytree_input(self, argnum):
        """Test that the qml.grad primitive can be captured with pytree inputs."""

        fdtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32

        def inner_func(x, y):
            return jnp.prod(jnp.sin(x["a"]) * jnp.cos(y[0]["b"][1]) ** 2)

        def func_qml(x):
            return qml.grad(inner_func, argnum=argnum)(
                {"a": x}, ({"b": [None, 0.4 * jnp.sqrt(x)]},)
            )

        def func_jax(x):
            return jax.grad(inner_func, argnums=argnum)(
                {"a": x}, ({"b": [None, 0.4 * jnp.sqrt(x)]},)
            )

        x = 0.7
        jax_out = func_jax(x)
        jax_out_flat, jax_out_tree = jax.tree_util.tree_flatten(jax_out)
        qml_out_flat, qml_out_tree = jax.tree_util.tree_flatten(func_qml(x))
        assert jax_out_tree == qml_out_tree
        assert qml.math.allclose(jax_out_flat, qml_out_flat)

        # Check overall jaxpr properties
        jaxpr = jax.make_jaxpr(func_qml)(x)
        assert jaxpr.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
        assert len(jaxpr.eqns) == 3
        argnum = [argnum] if isinstance(argnum, int) else argnum
        assert jaxpr.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)] * len(argnum)

        grad_eqn = jaxpr.eqns[2]
        diff_eqn_assertions(grad_eqn, grad_prim, argnum=argnum)
        assert [var.aval for var in grad_eqn.outvars] == jaxpr.out_avals
        assert len(grad_eqn.params["jaxpr"].eqns) == 6  # 5 numeric eqns, 1 conversion eqn

        manual_out = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        manual_out_flat, manual_out_tree = jax.tree_util.tree_flatten(manual_out)
        # Assert that the output from the manual evaluation is flat
        assert manual_out_tree == jax.tree_util.tree_flatten(manual_out_flat)[1]
        assert qml.math.allclose(jax_out_flat, manual_out_flat)

    @pytest.mark.parametrize("argnum", ([0, 1, 2], [0, 2], [1], 0))
    def test_grad_qnode_with_pytrees(self, argnum):
        """Test capturing the gradient of a qnode that uses Pytrees."""
        # pylint: disable=protected-access
        fdtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="backprop")
        def circuit(x, y, z):
            qml.RX(x["a"], wires=0)
            qml.RY(y, wires=0)
            qml.RZ(z[1][0], wires=0)
            return qml.expval(qml.X(0))

        dcircuit = qml.grad(circuit, argnum=argnum)
        x = {"a": 0.6, "b": 0.9}
        y = 0.6
        z = ({"c": 0.5}, [0.2, 0.3])
        qml_out = dcircuit(x, y, z)
        qml_out_flat, qml_out_tree = jax.tree_util.tree_flatten(qml_out)
        jax_out = jax.grad(circuit, argnums=argnum)(x, y, z)
        jax_out_flat, jax_out_tree = jax.tree_util.tree_flatten(jax_out)
        assert jax_out_tree == qml_out_tree
        assert qml.math.allclose(jax_out_flat, qml_out_flat)

        jaxpr = jax.make_jaxpr(dcircuit)(x, y, z)

        assert len(jaxpr.eqns) == 1  # grad equation
        assert jaxpr.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)] * 6
        argnum = [argnum] if isinstance(argnum, int) else argnum
        num_out_avals = 2 * (0 in argnum) + (1 in argnum) + 3 * (2 in argnum)
        assert jaxpr.out_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)] * num_out_avals

        grad_eqn = jaxpr.eqns[0]
        assert all(invar.aval == in_aval for invar, in_aval in zip(grad_eqn.invars, jaxpr.in_avals))
        flat_argnum = [0, 1] * (0 in argnum) + [2] * (1 in argnum) + [3, 4, 5] * (2 in argnum)
        diff_eqn_assertions(grad_eqn, grad_prim, argnum=flat_argnum)
        grad_jaxpr = grad_eqn.params["jaxpr"]
        assert len(grad_jaxpr.eqns) == 1  # qnode equation

        flat_args = jax.tree_util.tree_leaves((x, y, z))
        manual_out = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *flat_args)
        manual_out_flat, manual_out_tree = jax.tree_util.tree_flatten(manual_out)
        # Assert that the output from the manual evaluation is flat
        assert manual_out_tree == jax.tree_util.tree_flatten(manual_out_flat)[1]
        assert qml.math.allclose(jax_out_flat, manual_out_flat)

    @pytest.mark.usefixtures("enable_disable_dynamic_shapes")
    @pytest.mark.parametrize("same_dynamic_shape", (True, False))
    def test_grad_dynamic_shape_inputs(self, same_dynamic_shape):
        """Test that qml.grad can handle dynamic shapes"""

        @qml.qnode(qml.device("default.qubit", wires=4))
        def c(x, y):
            qml.RX(x, 0)
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

        def w(n):
            x = jnp.arange(n)
            if same_dynamic_shape:
                y = jnp.arange(n)
            else:
                y = jnp.arange(n + 1)
            return qml.grad(c, argnum=(0, 1))(x, y)

        jaxpr = jax.make_jaxpr(w)(2)
        grad_eqn = jaxpr.eqns[2] if same_dynamic_shape else jaxpr.eqns[3]
        assert grad_eqn.primitive == grad_prim

        shift = 1 if same_dynamic_shape else 2
        assert grad_eqn.params["argnum"] == [shift, shift + 1]
        assert len(grad_eqn.outvars) == 2
        assert grad_eqn.outvars[0].aval.shape == grad_eqn.invars[shift].aval.shape
        assert grad_eqn.outvars[1].aval.shape == grad_eqn.invars[shift + 1].aval.shape


def _jac_allclose(jac1, jac2, num_axes, atol=1e-8):
    """Test that two Jacobians, given as nested sequences of arrays, are equal."""
    if num_axes == 0:
        return qml.math.allclose(jac1, jac2, atol=atol)
    if len(jac1) != len(jac2):
        return False
    return all(
        _jac_allclose(_jac1, _jac2, num_axes - 1, atol=atol) for _jac1, _jac2 in zip(jac1, jac2)
    )


class TestJacobian:
    """Tests for capturing `qml.jacobian`."""

    @pytest.mark.parametrize("argnum", ([0, 1], [0], [1], 0, 1))
    def test_classical_jacobian(self, argnum):
        """Test that the qml.jacobian primitive can be captured with classical nodes."""
        fdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

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
        qml_out = func_qml(x, y)
        num_axes = 1 if (int_argnum := isinstance(argnum, int)) else 2
        assert _jac_allclose(qml_out, jax_out, num_axes)

        # Check overall jaxpr properties
        jaxpr = jax.make_jaxpr(func_qml)(x, y)

        if int_argnum:
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
        # Evaluating jaxpr gives flat list results. Need to adapt the JAX output to that
        if not int_argnum:
            jax_out = sum(jax_out, start=())
        assert _jac_allclose(manual_eval, jax_out, num_axes)

    def test_nested_jacobian(self):
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
        fdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        def func(x):
            return jnp.prod(x) * jnp.sin(x), jnp.sum(x**2)

        x = jnp.array([0.7, -0.9, 0.6, 0.3])
        dim = len(x)
        eye = jnp.eye(dim)

        # 1st order
        qml_func_1 = qml.jacobian(func)
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

        prod_sin = jnp.prod(x) * jnp.sin(x)
        prod_cos_e_i = jnp.prod(x) * jnp.cos(x) * eye
        expected_1 = (prod_sin[:, None] / x[None, :] + prod_cos_e_i, 2 * x)
        assert _jac_allclose(qml_func_1(x), expected_1, 1)

        # 2nd order
        qml_func_2 = qml.jacobian(qml_func_1)

        manual_eval_1 = jax.core.eval_jaxpr(jaxpr_1.jaxpr, jaxpr_1.consts, x)
        assert _jac_allclose(manual_eval_1, expected_1, 1)

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

        # broken by jax 0.5.3
        # hyperdiag = qml.numpy.zeros((4, 4, 4))
        # for i in range(4):
        #    hyperdiag[i, i, i] = 1
        # expected_2 = (
        #    prod_sin[:, None, None] / x[None, :, None] / x[None, None, :]
        #    - jnp.tensordot(prod_sin, eye / x**2, axes=0)  # Correct diagonal entries
        #    + prod_cos_e_i[:, :, None] / x[None, None, :]
        #    + prod_cos_e_i[:, None, :] / x[None, :, None]
        #    - prod_sin * hyperdiag,
        #    eye * 2,
        # )
        # Output only has one tuple axis
        # atol = 1e-8 if jax.config.jax_enable_x64 else 2e-7
        # assert _jac_allclose(qml_func_2(x), expected_2, 1, atol=atol)

        # manual_eval_2 = jax.core.eval_jaxpr(jaxpr_2.jaxpr, jaxpr_2.consts, x)
        # assert _jac_allclose(manual_eval_2, expected_2, 1, atol=atol)

    @pytest.mark.parametrize(
        "diff_method", ("backprop", pytest.param("parameter-shift", marks=pytest.mark.xfail))
    )
    def test_jacobian_of_simple_qnode(self, diff_method):
        """Test capturing the gradient of a simple qnode."""
        # pylint: disable=protected-access
        fdtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32

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

        manual_res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        assert _jac_allclose(manual_res, expected_res, 1)

    @pytest.mark.parametrize("argnum", ([0, 1], [0], [1]))
    def test_jacobian_pytrees(self, argnum):
        """Test that the qml.jacobian primitive can be captured with
        pytree inputs and outputs."""

        fdtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32

        def inner_func(x, y):
            return {
                "prod_cos": jnp.prod(jnp.sin(x["a"]) * jnp.cos(y[0]["b"][1]) ** 2),
                "sum_sin": jnp.sum(jnp.sin(x["a"]) * jnp.sin(y[1]["c"]) ** 2),
            }

        def func_qml(x):
            return qml.jacobian(inner_func, argnum=argnum)(
                {"a": x}, ({"b": [None, 0.4 * jnp.sqrt(x)]}, {"c": 0.5})
            )

        def func_jax(x):
            return jax.jacobian(inner_func, argnums=argnum)(
                {"a": x}, ({"b": [None, 0.4 * jnp.sqrt(x)]}, {"c": 0.5})
            )

        x = 0.7
        jax_out = func_jax(x)
        jax_out_flat, jax_out_tree = jax.tree_util.tree_flatten(jax_out)
        qml_out_flat, qml_out_tree = jax.tree_util.tree_flatten(func_qml(x))
        assert jax_out_tree == qml_out_tree
        assert qml.math.allclose(jax_out_flat, qml_out_flat)

        # Check overall jaxpr properties
        jaxpr = jax.make_jaxpr(func_qml)(x)
        assert jaxpr.in_avals == [jax.core.ShapedArray((), fdtype, weak_type=True)]
        assert len(jaxpr.eqns) == 3

        argnum = [argnum] if isinstance(argnum, int) else argnum
        # Compute the flat argnum in order to determine the expected number of out tracers
        flat_argnum = [0] * (0 in argnum) + [1, 2] * (1 in argnum)
        assert jaxpr.out_avals == [jax.core.ShapedArray((), fdtype)] * (2 * len(flat_argnum))

        jac_eqn = jaxpr.eqns[2]

        diff_eqn_assertions(jac_eqn, jacobian_prim, argnum=flat_argnum)
        assert [var.aval for var in jac_eqn.outvars] == jaxpr.out_avals

        manual_out = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)
        manual_out_flat, manual_out_tree = jax.tree_util.tree_flatten(manual_out)
        # Assert that the output from the manual evaluation is flat
        assert manual_out_tree == jax.tree_util.tree_flatten(manual_out_flat)[1]
        assert qml.math.allclose(jax_out_flat, manual_out_flat)

    @pytest.mark.usefixtures("enable_disable_dynamic_shapes")
    @pytest.mark.parametrize("same_dynamic_shape", (True, False))
    def test_jacobian_dynamic_shape_inputs(self, same_dynamic_shape):
        """Test that qml.jacobian can handle dynamic shapes"""

        @qml.qnode(qml.device("default.qubit", wires=4))
        def c(x, y):
            qml.RX(x, 0)
            qml.RY(y, 0)
            return qml.probs(wires=(0, 1))

        def w(n):
            x = jnp.arange(n)
            if same_dynamic_shape:
                y = jnp.arange(n)
            else:
                y = jnp.arange(n + 1)
            return qml.jacobian(c, argnum=(0, 1))(x, y)

        jaxpr = jax.make_jaxpr(w)(2)
        grad_eqn = jaxpr.eqns[2] if same_dynamic_shape else jaxpr.eqns[3]
        assert grad_eqn.primitive == jacobian_prim

        shift = 1 if same_dynamic_shape else 2
        assert grad_eqn.params["argnum"] == [shift, shift + 1]
        assert len(grad_eqn.outvars) == 2

        assert grad_eqn.outvars[0].aval.shape == (4, *grad_eqn.invars[shift].aval.shape)
        assert grad_eqn.outvars[1].aval.shape == (4, *grad_eqn.invars[shift + 1].aval.shape)
