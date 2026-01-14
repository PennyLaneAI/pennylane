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
"""Tests for capturing Pauli product measurements."""

# pylint: disable=wrong-import-order,wrong-import-position,ungrouped-imports
from functools import partial

import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

import jax.numpy as jnp

pytestmark = [pytest.mark.capture, pytest.mark.jax]


def test_no_capture():
    """Test that if capture is turned off, no special primitive is captured."""

    @qml.capture.subroutine
    def f(x):
        return x + 1

    qml.capture.disable()

    jaxpr = jax.make_jaxpr(f)(2)
    assert jaxpr.eqns[0].primitive.name == "add"  # not a quantum subroutine


def test_repeated_call_same_shapes():
    """Test that if the subroutine is called with the same shapes, you get the same jaxpr."""

    @qml.capture.subroutine
    def f(x):
        qml.RX(x, 0)

    def w(x):
        f(x)
        f(2 * x)

    jaxpr = jax.make_jaxpr(w)(0.5)

    for i in [0, 2]:
        eqn = jaxpr.eqns[i]
        assert eqn.primitive == qml.capture.primitives.quantum_subroutine_prim
        assert eqn.params["name"] == "f"

    assert jaxpr.eqns[0].params["jaxpr"] is jaxpr.eqns[2].params["jaxpr"]


@pytest.mark.parametrize(("x", "y"), [(0.5, 1), (jnp.array([0.5]), jnp.array([0.5, 0.6, 0.7]))])
def test_repeated_call_different_inputs(x, y):
    """Test that if different inputs shapes or dtypes are passed in, we get different jaxprs."""

    @qml.capture.subroutine
    def add_func(x):
        return x + 1

    def c(x, y):
        add_func(x)
        add_func(y)

    jaxpr = jax.make_jaxpr(c)(x, y)

    assert jaxpr.eqns[0].params["name"] == "add_func"
    assert jaxpr.eqns[1].params["name"] == "add_func"

    assert jaxpr.eqns[0].params["jaxpr"] != jaxpr.eqns[1].params["jaxpr"]


@pytest.mark.parametrize("static_kwargs", ({"static_argnums": 1}, {"static_argnames": "op_type"}))
def test_static_arguments(static_kwargs):
    """Test that static arguments effect the captured jaxpr."""

    @partial(qml.capture.subroutine, **static_kwargs)
    def some_func(x, op_type):
        if op_type == "RX":
            qml.RX(x, 0)
        else:
            qml.RY(x, 0)

    def c(x):
        some_func(x, "RX")
        some_func(x, "RX")
        some_func(x, "RY")

    jaxpr = jax.make_jaxpr(c)(0.5)

    jaxpr0 = jaxpr.eqns[0].params["jaxpr"]
    jaxpr2 = jaxpr.eqns[2].params["jaxpr"]
    assert jaxpr0 == jaxpr.eqns[1].params["jaxpr"]
    assert jaxpr0 != jaxpr2

    assert len(jaxpr0.eqns) == 1
    assert jaxpr0.eqns[0].primitive == qml.RX._primitive  # pylint: disable=protected-access

    assert len(jaxpr2.eqns) == 1
    assert jaxpr2.eqns[0].primitive == qml.RY._primitive  # pylint: disable=protected-access
