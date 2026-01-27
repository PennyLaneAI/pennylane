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
Tests for capturing vjp into jaxpr.
"""
from functools import partial

import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

jnp = pytest.importorskip("jax.numpy")

from pennylane.capture.primitives import jvp_prim  # pylint: disable=wrong-import-position


class TestErrors:

    def test_error_on_big_argnum(self):
        """Test that an error is raised if the max argnums is bigger than the number of args."""

        def f(x, y):
            return x + y

        with pytest.raises(ValueError, match="Differentiating with respect to argnums"):
            qml.jvp(f, (0.5, 1.2), (1.0, 1.0), argnums=2)

    def test_error_on_bad_h(self):
        """Test that an error is raised on a bad h value."""

        with pytest.raises(ValueError, match="Invalid h value"):
            qml.jvp(lambda x: x * 2, (0.5,), (1.0,), h="something")

    def test_error_on_bad_method(self):
        """Test that an error is raised on a bad method."""

        with pytest.raises(ValueError, match="Got unrecognized method"):
            qml.jvp(lambda x: x**2, (0.5,), (1.0,), method="param-shift")

    def test_error_wrong_number_tangents(self):
        """Test that an error is raised for the wrong number of tangents."""

        with pytest.raises(
            TypeError, match="number of tangent and number of differentiable parameters"
        ):
            qml.jvp(lambda x: x**2, (0.5,), (1.0, 1.0))

    def test_error_wrong_dtype_tangents(self):
        """Test that an error is raised if the tangent is of the wrong dtype."""

        with pytest.raises(TypeError, match="dtypes must be equal"):
            qml.jvp(lambda x: x**2, (0.5,), (1,))

    def test_error_wrong_shape_tangents(self):
        """Test that an error is raised if the tangent has the wrong shape."""

        with pytest.raises(ValueError, match="params and tangent shapes"):
            qml.jvp(lambda x: x**2, (jnp.array(0.5),), (jnp.array([1.0, 1.0]),))


class TestCapturingJVP:

    def test_const(self):
        """Test capturing the jvp with a constant."""

        def f(x):
            return jnp.array([2, 1]) * x

        def w(x):
            return qml.jvp(f, (x,), (1.0,))

        jaxpr = jax.make_jaxpr(w)(0.5)

        jvp_eqn = jaxpr.eqns[0]
        assert jvp_eqn.primitive == jvp_prim
        assert jvp_eqn.params["argnums"] == (1,)  # shifted by one const

        inner_j = jvp_eqn.params["jaxpr"]
        assert len(inner_j.constvars) == 0
        assert len(inner_j.invars) == 2  # const + input
        assert inner_j.invars[0].aval.shape == (2,)  # the const

        assert jaxpr.out_avals[0].shape == (2,)  # the result
        assert jaxpr.out_avals[1].shape == (2,)  # d_result

    def test_argnum_pytree_input(self):
        """Test that the argnums are expanded for a pytree input."""

        def f(x, y):
            return x["a"] * y[0] + x["b"] * y[1]

        x = {"a": 0.5, "b": 1.2}
        y = [2.0, 3.0]

        dx = {"a": 1.0, "b": 1.0}
        dy = [1.0, 1.0]

        def w(x, y, argnums):
            dinputs = (dx, dy)
            return qml.jvp(f, (x, y), (dinputs[argnums],), argnums=argnums)

        for argnums in (0, 1):

            jaxpr = jax.make_jaxpr(partial(w, argnums=argnums))(x, y)
            jvp_eqn = jaxpr.eqns[0]
            assert jvp_eqn.primitive == jvp_prim
            assert jvp_eqn.params["argnums"] == (2 * argnums, 2 * argnums + 1)
            assert len(jvp_eqn.outvars) == 2  # result, d_result

    def test_setting_h(self):
        """Test that an h can be set and captured."""

        def f(x):
            return 2 * x

        def w(x):
            return qml.jvp(f, (x,), (1.0,), h=1e-4)

        jaxpr = jax.make_jaxpr(w)(0.5)

        jaxpr_eqn = jaxpr.eqns[0]

        assert jaxpr_eqn.params["h"] == 1e-4
        assert jaxpr_eqn.params["method"] == "auto"

    def test_setting_method(self):
        """Test that method=fd can be captured."""

        def f(x):
            return 2 * x

        def w(x):
            return qml.jvp(f, (x,), (1.0,), method="fd")

        jaxpr = jax.make_jaxpr(w)(0.5)

        jaxpr_eqn = jaxpr.eqns[0]

        assert jaxpr_eqn.params["method"] == "fd"
        assert jaxpr_eqn.params["h"] == 1e-6

    def test_multiple_outputs(self):
        """Test capturing the jvp of a function with multiple outputs."""

        def f(x):
            y = jnp.stack([x, x])
            z = jnp.stack([y, y])
            return y**2, z**3

        def w(x, dx):
            return qml.jvp(f, (x,), (dx,))

        x = jnp.array(0.5)
        dx = jnp.array(2.0)
        jaxpr = jax.make_jaxpr(w)(x, dx).jaxpr
        vjp_eqn = jaxpr.eqns[0]

        assert len(vjp_eqn.invars) == 2
        assert vjp_eqn.invars[0].aval.shape == ()  # input
        assert vjp_eqn.invars[1].aval.shape == ()  # d_input

        assert len(vjp_eqn.outvars) == 4
        assert vjp_eqn.outvars[0].aval.shape == (2,)  # y1
        assert vjp_eqn.outvars[1].aval.shape == (2, 2)  # y2
        assert vjp_eqn.outvars[2].aval.shape == (2,)  # d_y1
        assert vjp_eqn.outvars[3].aval.shape == (2, 2)  # d_y2

    def test_sequence_argnums(self):
        """Test that multiple argnums can be provided in a sequence."""

        def f(x, y, z):
            return jnp.sum(x) + jnp.sum(y) + jnp.sum(z)

        def w(x, y, z, dx, dz):
            return qml.jvp(f, (x, y, z), (dx, dz), argnums=[0, 2])

        x = jnp.arange(2, dtype=float)
        y = jnp.arange(3, dtype=float)
        z = jnp.arange(4, dtype=float)
        dx = jnp.array([2.0, 2])
        dz = jnp.array([3.0, 3, 3, 3])
        jaxpr = jax.make_jaxpr(w)(x, y, z, dx, dz)
        vjp_eqn = jaxpr.eqns[0]

        assert vjp_eqn.params["argnums"] == (0, 2)

        assert len(vjp_eqn.invars) == 5  # three inputs, two tangents
        assert len(vjp_eqn.outvars) == 2  # one result, one derivative

        assert vjp_eqn.outvars[0].aval.shape == ()  # result
        assert vjp_eqn.outvars[1].aval.shape == ()  # d_result


def test_pytrees_in_and_out():
    """Test that pytrees can be handled with both the inputs and the outputs."""

    def f(x, y):
        return {"result": x["a"] * y[0] + x["b"] * y[1]}

    x = {"a": 0.5, "b": 1.2}
    y = [2.0, 3.0]

    dx = {"a": 5.0, "b": 10.0}

    results, d_results = qml.jvp(f, (x, y), (dx,))

    assert isinstance(results, dict)
    assert jnp.allclose(results["result"], 0.5 * 2 + 3.0 * 1.2)

    assert isinstance(d_results, dict)
    assert jnp.allclose(d_results["result"], y[0] * dx["a"] + dx["b"] * y[1])
