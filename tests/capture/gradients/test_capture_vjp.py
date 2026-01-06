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

from pennylane.capture.primitives import vjp_prim  # pylint: disable=wrong-import-position


class TestErrors:

    def test_error_on_big_argnum(self):
        """Test that an error is raised if the max argnums is bigger than the number of args."""

        def f(x, y):
            return x + y

        with pytest.raises(ValueError, match="Differentiating with respect to argnums"):
            qml.vjp(f, (0.5, 1.2), (1.0,), argnums=2)

    def test_error_on_bad_h(self):
        """Test that an error is raised on a bad h value."""

        with pytest.raises(ValueError, match="Invalid h value"):
            qml.vjp(lambda x: x * 2, (0.5,), (1.0,), h="something")

    def test_error_on_bad_method(self):
        """Test that an error is raised on a bad method."""

        with pytest.raises(ValueError, match="Got unrecognized method"):
            qml.vjp(lambda x: x**2, (0.5,), (1.0,), method="param-shift")

    @pytest.mark.parametrize("cotangents", ((0.5,), (1.2, 2.3, 3.4)))
    def test_error_on_wrong_number_of_cotangents(self, cotangents):
        """Test an error is raised on the wrong number of cotangents."""

        def f(x):
            return 2 * x, 3 * x

        with pytest.raises(
            ValueError, match=r"length of cotangents must match the number of outputs"
        ):
            qml.vjp(f, (0.5,), cotangents)

    def test_error_on_wrong_cotangent_dtype(self):
        """Test an error is raised on the wrong cotangent dtype."""

        def f(x):
            return 2 * x

        with pytest.raises(TypeError, match="dtypes must be equal."):
            qml.vjp(f, (jnp.array(0.5),), jnp.array(1))

    def test_error_on_wrong_cotangent_shape(self):
        """Test an error is raised on the wrong cotangent shape."""

        def f(x, y):
            return 2 * x, 3 * y

        cotangents = (jnp.array(0.5), jnp.array([1.0, 1.0, 1.0]))
        params = (jnp.array(1.0), jnp.array([1.0, 1.0]))

        with pytest.raises(ValueError, match=r"got function output params shape"):
            qml.vjp(f, params, cotangents)


class TestCapturingVJP:

    def test_const(self):
        """Test capturing the vjp with a constant."""

        def f(x):
            return jnp.array([2, 1]) * x

        def w(x):
            return qml.vjp(f, (x,), (jnp.array([1.0, 1.0]),))

        jaxpr = jax.make_jaxpr(w)(0.5)

        vjp_eqn = jaxpr.eqns[0]
        assert vjp_eqn.primitive == vjp_prim
        assert vjp_eqn.params["argnums"] == (1,)  # shifted by one const

        inner_j = vjp_eqn.params["jaxpr"]
        assert len(inner_j.constvars) == 0
        assert len(inner_j.invars) == 2  # const + input
        assert inner_j.invars[0].aval.shape == (2,)  # the const

        assert jaxpr.out_avals[0].shape == (2,)  # the result
        assert jaxpr.out_avals[1].shape == ()  # dparams

    def test_argnum_pytree_input(self):
        """Test that the argnums are expanded for a pytree input."""

        def f(x, y):
            return x["a"] * y[0] + x["b"] * y[1]

        x = {"a": 0.5, "b": 1.2}
        y = [2.0, 3.0]

        def w(x, y, argnums):
            return qml.vjp(f, (x, y), (-1.0,), argnums=argnums)

        for argnums in (0, 1):

            jaxpr = jax.make_jaxpr(partial(w, argnums=argnums))(x, y)
            vjp_eqn = jaxpr.eqns[0]
            assert vjp_eqn.primitive == vjp_prim
            assert vjp_eqn.params["argnums"] == (2 * argnums, 2 * argnums + 1)
            assert len(vjp_eqn.outvars) == 3  # one result, two dparams

    def test_setting_h(self):
        """Test that an h can be set and captured."""

        def f(x):
            return 2 * x

        def w(x):
            return qml.vjp(f, (x,), (1.0,), h=1e-4)

        jaxpr = jax.make_jaxpr(w)(0.5)

        jaxpr_eqn = jaxpr.eqns[0]

        assert jaxpr_eqn.params["h"] == 1e-4
        assert jaxpr_eqn.params["method"] == "auto"

    def test_setting_method(self):
        """Test that method=fd can be captured."""

        def f(x):
            return 2 * x

        def w(x):
            return qml.vjp(f, (x,), (1.0,), method="fd")

        jaxpr = jax.make_jaxpr(w)(0.5)

        jaxpr_eqn = jaxpr.eqns[0]

        assert jaxpr_eqn.params["method"] == "fd"
        assert jaxpr_eqn.params["h"] == 1e-6


def test_pytrees_in_and_out():
    """Test that pytrees can be handled with both the inputs and the outputs."""

    def f(x, y):
        return {"result": x["a"] * y[0] + x["b"] * y[1]}

    x = {"a": 0.5, "b": 1.2}
    y = [2.0, 3.0]

    results, dparams = qml.vjp(f, (x, y), {"result": -1.0})

    assert isinstance(results, dict)
    assert jnp.allclose(results["result"], 0.5 * 2 + 3.0 * 1.2)

    assert isinstance(dparams, tuple)
    assert len(dparams) == 1
    assert jnp.allclose(dparams[0]["a"], -y[0])
    assert jnp.allclose(dparams[0]["b"], -y[1])
