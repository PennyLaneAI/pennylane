# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Tests a function for determining abstracted axes and extracting the abstract shapes.
"""
# pylint: disable=redefined-outer-name, unused-argument

import pytest

from pennylane.capture import determine_abstracted_axes, register_custom_staging_rule

pytestmark = pytest.mark.capture

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_null_if_not_enabled():
    """Test None and an empty tuple are returned if dynamic shapes is not enabled."""

    def f(*args):
        abstracted_axes, abstract_shapes = determine_abstracted_axes(args)

        assert abstracted_axes is None
        assert abstract_shapes == ()

    _ = jax.make_jaxpr(f)(jnp.eye(4))


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestDyanmicShapes:

    def test_null_if_no_abstract_shapes(self):
        """Test the None and an empty tuple are returned if no dynamic shapes exist."""

        def f(*args):
            abstracted_axes, abstract_shapes = determine_abstracted_axes(args)

            assert abstracted_axes is None
            assert abstract_shapes == ()

        _ = jax.make_jaxpr(f)(jnp.eye(4))

    def test_single_abstract_shape(self):
        """Test we get the correct answer for a single abstract shape."""

        initial_abstracted_axes = ({0: 0},)

        def f(*args):
            abstracted_axes, abstract_shapes = determine_abstracted_axes(args)

            assert abstracted_axes == initial_abstracted_axes
            assert len(abstract_shapes) == 1

            # test we can make jaxpr with these abstracted axes
            jaxpr = jax.make_jaxpr(lambda *args: 0, abstracted_axes=abstracted_axes)(*args)
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *abstract_shapes, *args)

        _ = jax.make_jaxpr(f, abstracted_axes=initial_abstracted_axes)(jnp.arange(4))

    @pytest.mark.parametrize(
        "initial_abstracted_axes, num_shapes",
        [
            (({0: 0, 1: 1},), 2),
            (({0: 0, 1: 0},), 1),
            (({1: 0},), 1),
        ],
    )
    def test_single_abstract_shape_multiple_abstract_axes(
        self, initial_abstracted_axes, num_shapes
    ):
        """Test we get the correct answer for a single input with two abstract axes."""

        def f(*args):
            abstracted_axes, abstract_shapes = determine_abstracted_axes(args)

            assert abstracted_axes == initial_abstracted_axes
            assert len(abstract_shapes) == num_shapes

            # test we can make jaxpr with these abstracted axes
            jaxpr = jax.make_jaxpr(lambda *args: 0, abstracted_axes=abstracted_axes)(*args)
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *abstract_shapes, *args)

        _ = jax.make_jaxpr(f, abstracted_axes=initial_abstracted_axes)(jnp.eye(4))

    def test_pytree_input(self):
        """Test a pytree input with dynamic shapes."""

        initial_abstracted_axes = (
            {"input0": {}, "input1": {0: 0}, "input2": {0: 0}, "input3": {1: 1}},
        )
        arg = {
            "input0": jnp.arange(5),
            "input1": jnp.arange(3),
            "input2": jnp.arange(3),
            "input3": jnp.eye(4),
        }

        def f(*args):
            abstracted_axes, abstract_shapes = determine_abstracted_axes(args)
            assert abstracted_axes == initial_abstracted_axes
            assert len(abstract_shapes) == 2

            # test we can make jaxpr with these abstracted axes
            jaxpr = jax.make_jaxpr(lambda *args: 0, abstracted_axes=abstracted_axes)(*args)
            flat_args = jax.tree_util.tree_leaves(args)
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *abstract_shapes, *flat_args)

        _ = jax.make_jaxpr(f, abstracted_axes=initial_abstracted_axes)(arg)

    def test_input_created_with_jnp_ones(self):
        """Test that determine_abstracted_axes works with manually created dynamic arrays."""

        def f(n):
            m = n + 1
            ones = jax.numpy.ones((m, 3))
            zeros = jax.numpy.zeros((4, n))

            abstracted_axes, abstract_shapes = determine_abstracted_axes((ones, zeros))
            assert abstracted_axes == ({0: 0}, {1: 1})
            assert len(abstract_shapes) == 2
            assert abstract_shapes[0] is m
            assert abstract_shapes[1] is n

        _ = jax.make_jaxpr(f)(3)

    def test_large_number_of_abstract_axes(self):
        """Test that determine_abstracted_axes can handle over 26 abstract axes."""

        def f(shapes):
            ones = jax.numpy.zeros(shapes)
            abstracted_axes, abstract_shapes = determine_abstracted_axes((ones,))

            assert abstracted_axes
            assert len(set(abstracted_axes[0].keys())) == 30  # unique keys for each axis
            assert len(abstract_shapes) == 30

        _ = jax.make_jaxpr(f)(list(range(30)))


def test_custom_staging_rule(enable_disable_dynamic_shapes):
    """Test regsitering a custom staging rule for a new primitive."""
    my_prim = jax.extend.core.Primitive("my_prim")
    register_custom_staging_rule(my_prim, lambda params: params["jaxpr"].outvars)

    def f(i):
        return i, jax.numpy.ones(i)

    jaxpr = jax.make_jaxpr(f)(2)

    def workflow():
        return my_prim.bind(jaxpr=jaxpr.jaxpr)

    jaxpr = jax.make_jaxpr(workflow)()
    assert jaxpr.eqns[0].primitive == my_prim
    assert len(jaxpr.eqns[0].outvars) == 2
    assert jaxpr.eqns[0].outvars[0] is jaxpr.eqns[0].outvars[1].aval.shape[0]

    # doesn't return a dynamic shape unless it needs to
    assert isinstance(jaxpr.jaxpr.outvars[0].aval, jax.core.ShapedArray)
    assert isinstance(jaxpr.jaxpr.outvars[1].aval, jax.core.DShapedArray)
