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

from pennylane.capture import determine_abstracted_axes

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture
def enable_disable():
    jax.config.update("jax_dynamic_shapes", True)
    try:
        yield
    finally:
        jax.config.update("jax_dynamic_shapes", False)


def test_null_if_not_enabled():
    """Test None and an empty tuple are returned if dynamic shapes is not enabled."""

    def f(*args):
        abstracted_axes, abstract_shapes = determine_abstracted_axes(args)

        assert abstracted_axes is None
        assert abstract_shapes == ()

    _ = jax.make_jaxpr(f)(jnp.eye(4))


def test_null_if_no_abstract_shapes(enable_disable):
    """Test the None and an empty tuple are returned if no dynamic shapes exist."""

    def f(*args):
        abstracted_axes, abstract_shapes = determine_abstracted_axes(args)

        assert abstracted_axes is None
        assert abstract_shapes == ()

    _ = jax.make_jaxpr(f)(jnp.eye(4))


def test_single_abstract_shape(enable_disable):
    """Test we get the correct answer for a single abstract shape."""

    initial_abstracted_axes = ({0: "a"},)

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
        (({0: "a", 1: "b"},), 2),
        (({0: "a", 1: "a"},), 1),
        (({1: "a"},), 1),
    ],
)
def test_single_abstract_shape_multiple_abstract_axes(
    enable_disable, initial_abstracted_axes, num_shapes
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


def test_pytree_input(enable_disable):
    """Test a pytree input with dynamic shapes."""

    initial_abstracted_axes = (
        {"input0": {}, "input1": {0: "a"}, "input2": {0: "a"}, "input3": {1: "b"}},
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
