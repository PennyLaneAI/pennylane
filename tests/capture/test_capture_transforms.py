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
This submodule tests transforms with program capture
"""

import pytest

import pennylane as qml
from pennylane.transforms.core import transform

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


@transform
def dummy_transform1(tape, arg1, arg2, kwarg1=None, kwarg2=None):  # pylint: disable=unused-argument
    """Dummy transform for testing."""
    return [tape], lambda res: res[0]


@transform
def dummy_transform2(tape, arg3, arg4, kwarg3=None, kwarg4=None):  # pylint: disable=unused-argument
    """Dummy transform for testing."""
    return [tape], lambda res: res[0]


class TestCaptureTransforms:
    """Tests to verify that transforms are captured correctly."""

    def test_transform_primitive_capture(self):
        """Test that a transform's primitive is captured correctly."""

        def func(x):
            y = x * 5
            return y**0.5

        args = (1.5,)
        targs = [0, 1]
        tkwargs = {"kwarg1": "foo", "kwarg2": "bar"}

        transformed_func = dummy_transform1(func, *targs, **tkwargs)
        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn := jaxpr.eqns[0]).primitive == dummy_transform1._primitive

        params = transform_eqn.params
        assert params["args_slice"] == slice(0, 1)
        assert params["consts_slice"] == slice(1, None)
        assert params["targs"] == targs
        assert params["tkwargs"] == tkwargs

        inner_jaxpr = params["inner_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).jaxpr
        for eqn1, eqn2 in zip(inner_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_transform_qnode_capture(self):
        """Test that a transformed QNode is captured correctly."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (1.5,)
        targs = [0, 1]
        tkwargs = {"kwarg1": "foo", "kwarg2": "bar"}

        transformed_func = dummy_transform1(func, *targs, **tkwargs)

        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn := jaxpr.eqns[0]).primitive == dummy_transform1._primitive

        params = transform_eqn.params
        qnode_jaxpr = params["inner_jaxpr"]
        assert qnode_jaxpr.eqns[0].primitive == qml.capture.qnode_prim

        qfunc_jaxpr = qnode_jaxpr.eqns[0].params["qfunc_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).eqns[0].params["qfunc_jaxpr"]
        for eqn1, eqn2 in zip(qfunc_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive

    def test_transform_primitive_eval(self):
        """Test that JAXPR containing a transform primitive can be evaluated correctly."""

        def func(x):
            y = x * 5
            return y**0.5

        args = (1.5,)
        targs = (0, 1)
        tkwargs = {"kwarg1": "foo", "kwarg2": "bar"}

        transformed_func = dummy_transform1(func, *targs, **tkwargs)
        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        inner_jaxpr = jax.make_jaxpr(func)(*args)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)[0]
        expected = jax.core.eval_jaxpr(inner_jaxpr.jaxpr, inner_jaxpr.consts, *args)[0]
        assert res == expected == func(*args)

    def test_transform_qnode_eval(self):
        """Test that JAXPR containing a transformed qnode primitive is evaluated correctly."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        args = (1.5,)
        targs = (0, 1)
        tkwargs = {"kwarg1": "foo", "kwarg2": "bar"}

        transformed_func = dummy_transform1(func, *targs, **tkwargs)
        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        inner_jaxpr = jax.make_jaxpr(func)(*args)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)[0]
        expected = jax.core.eval_jaxpr(inner_jaxpr.jaxpr, inner_jaxpr.consts, *args)[0]
        assert res == expected == func(*args)

    def test_multiple_transforms(self):
        """Test that JAXPR containing a transformed qnode primitive is evaluated correctly."""

        def func(x):
            y = x * 5
            return y**0.5

        args = (1.5,)
        targs1 = (0, 1)
        tkwargs1 = {"kwarg1": "foo", "kwarg2": "bar"}
        targs2 = (2, 3)
        tkwargs2 = {"kwarg3": "hello", "kwarg4": "world"}

        transformed_func = dummy_transform1(
            dummy_transform2(func, *targs2, **tkwargs2), *targs1, **tkwargs1
        )
        jaxpr = jax.make_jaxpr(transformed_func)(*args)
        assert (transform_eqn1 := jaxpr.eqns[0]).primitive == dummy_transform1._primitive

        params1 = transform_eqn1.params
        assert params1["args_slice"] == slice(0, 1)
        assert params1["consts_slice"] == slice(1, None)
        assert params1["targs"] == list(targs1)
        assert params1["tkwargs"] == tkwargs1

        inner_jaxpr = params1["inner_jaxpr"]
        assert (transform_eqn2 := inner_jaxpr.eqns[0]).primitive == dummy_transform2._primitive

        params2 = transform_eqn2.params
        assert params2["args_slice"] == slice(0, 1)
        assert params2["consts_slice"] == slice(1, None)
        assert params2["targs"] == list(targs2)
        assert params2["tkwargs"] == tkwargs2

        inner_inner_jaxpr = params2["inner_jaxpr"]
        expected_jaxpr = jax.make_jaxpr(func)(*args).jaxpr
        for eqn1, eqn2 in zip(inner_inner_jaxpr.eqns, expected_jaxpr.eqns, strict=True):
            assert eqn1.primitive == eqn2.primitive
