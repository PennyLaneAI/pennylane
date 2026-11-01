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
"""Tests for registered AutoGraph function wrappers."""

import copy
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import pennylane as qp
from pennylane.capture.autograph import (
    autograph_source,
    disable_autograph,
    register_autograph_wrapper,
    run_autograph,
)
from pennylane.capture.autograph.transformer import TRANSFORMER
from pennylane.capture.primitives import cond_prim, jacobian_prim
from pennylane.testing import extract_all_primitives

pytestmark = [pytest.mark.capture, pytest.mark.jax]

jax = pytest.importorskip("jax")


@pytest.fixture(name="wrapper_type")
def registered_wrapper_type():
    """Register a library wrapper that also preserves its own configuration."""

    @dataclass(eq=False)
    class FunctionWrapper:
        """Wrap a function with an additional offset."""

        __module__ = "pennylane"

        function: Callable
        offset: float = 1.0

        def __call__(self, *args, **kwargs):
            return self.function(*args, **kwargs) + self.offset

    @register_autograph_wrapper(FunctionWrapper)
    def adapter(wrapper):
        def rebuild(function):
            new_wrapper = copy.copy(wrapper)
            new_wrapper.function = function
            return new_wrapper

        return wrapper.function, rebuild

    return FunctionWrapper


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("subclass", [False, True])
def test_custom_wrapper(wrapper_type, nested, subclass):
    """Registered library wrappers and their subclasses convert dynamic control flow."""

    def function(condition, data):
        if condition:
            result = data**2
        else:
            result = data**3
        return result

    selected_type = type("Subclass", (wrapper_type,), {}) if subclass else wrapper_type
    wrapper = selected_type(function, offset=3.0)

    if nested:

        def caller(condition, data):
            return wrapper(condition, data=data)

        converted = run_autograph(caller)
    else:
        converted = run_autograph(wrapper)
        assert isinstance(converted, type(wrapper))
        assert converted.offset == 3.0
        assert converted is not wrapper

    jaxpr = jax.make_jaxpr(converted)(True, 2.0)
    assert cond_prim in [eqn.primitive for eqn in jaxpr.eqns]
    assert jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, True, 2.0) == [7.0]
    assert jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, False, 2.0) == [11.0]
    assert wrapper.function is function
    assert not hasattr(function, "ag_unconverted")
    assert "ag__.if_stmt" in autograph_source(wrapper)
    if not nested:
        assert "ag__.if_stmt" in autograph_source(converted)


@pytest.mark.parametrize("derivative", [qp.grad, qp.jacobian])
@pytest.mark.parametrize("custom_outermost", [True, False])
@pytest.mark.parametrize("condition", [True, False])
def test_custom_wrapper_with_derivative(wrapper_type, derivative, custom_outermost, condition):
    """A registered custom wrapper composes with derivative wrappers in either order."""

    def function(predicate, data):
        if predicate:
            result = data**2
        else:
            result = data**3
        return result

    if custom_outermost:
        original = wrapper_type(derivative(function, argnums=1), offset=3.0)
    else:
        original = derivative(wrapper_type(function, offset=3.0), argnums=1)
    converted = run_autograph(original)

    jaxpr = jax.make_jaxpr(converted)(condition, 2.0)
    primitives = extract_all_primitives(jaxpr.jaxpr)
    assert jacobian_prim in primitives
    assert cond_prim in primitives

    with qp.capture.pause():
        result = converted(condition, qp.numpy.array(2.0, requires_grad=True))

    expected = 4.0 if condition else 12.0
    if custom_outermost:
        expected += 3.0
        assert original.function.func is function
        assert converted.offset == original.offset
    else:
        assert original.func.function is function
        assert converted.func.offset == original.func.offset
    assert qp.math.allclose(result, expected)
    assert not hasattr(function, "ag_unconverted")


def test_disabled_autograph(wrapper_type):
    """Registered wrapper calls respect the disabled AutoGraph context."""

    def function(data):
        if data > 0:
            result = data**2
        else:
            result = data**3
        return result

    wrapper = wrapper_type(function)

    @run_autograph
    def caller():
        with disable_autograph:
            return wrapper(2.0)

    jaxpr = jax.make_jaxpr(caller)()
    assert not jaxpr.eqns
    assert jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts) == [5.0]
    assert not TRANSFORMER.has_cache(function)
