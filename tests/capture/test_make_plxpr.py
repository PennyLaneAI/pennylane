# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the make_plxpr function for capturing qnodes as jaxpr.
"""

from unittest.mock import call

import numpy as np
import pytest

import pennylane as qml

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

# must be below jax importorskip
from pennylane.capture import make_plxpr  # pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


def test_make_plxpr(mocker):
    """Test that make_plxpr uses make_jaxpr, and returns a callable that will
    create a jaxpr representation of the qnode"""

    dev = qml.device("default.qubit", wires=1)

    spy = mocker.spy(jax, "make_jaxpr")

    @qml.qnode(dev)
    def circ(x):
        qml.RX(x, 0)
        qml.Hadamard(0)
        return qml.expval(qml.X(0))

    plxpr = make_plxpr(circ)(1.2)

    spy.assert_called()
    assert hasattr(plxpr, "jaxpr")


@pytest.mark.parametrize("static_argnums", [[0], [1], [0, 1], []])
def test_static_argnums(static_argnums, mocker):
    """Test that passing static_argnums works as expected"""

    dev = qml.device("default.qubit", wires=1)

    spy = mocker.spy(jax, "make_jaxpr")

    @qml.qnode(dev)
    def circ(x, y):
        qml.RX(x, 0)
        qml.RY(y, 0)
        qml.Hadamard(0)
        return qml.expval(qml.X(0))

    params = [1.2, 2.3]
    non_static_params = [params[i] for i in (0, 1) if i not in static_argnums]

    plxpr = make_plxpr(circ, static_argnums=static_argnums)(*params)

    # most recent call is to make a jaxpr of something else, so we can't use assert_called_with
    spy.assert_has_calls([call(circ, static_argnums=static_argnums)])

    # plxpr behaves as expected wrt static argnums
    res = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, *non_static_params)
    assert np.allclose(res, circ(*params))


def test_kwargs(mocker):
    """Test additional kwargs are passed through to make_jaxpr"""

    dev = qml.device("default.qubit", wires=1)

    spy = mocker.spy(jax, "make_jaxpr")

    @qml.qnode(dev)
    def circ():
        qml.Hadamard(0)
        return qml.expval(qml.X(0))

    # assert new value for return_shape is passed to make_jaxpr
    _ = make_plxpr(circ, return_shape=True)()
    spy.assert_has_calls([call(circ, static_argnums=None, return_shape=True)])


# ToDo: is this error fine, and if not, how would we modify it?
def test_dynamically_shaped_with_static_argnums_raises_error():
    """Test that the expected error is raised when passing a dynamically
    shaped array to an argument marked as static"""

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def f(x, y):
        for angle in x:
            qml.RX(angle, 0)

        qml.RY(y, 0)
        return qml.expval(qml.X(0))

    plxpr_fn = make_plxpr(f, static_argnums=0)
    with pytest.raises(ValueError, match="Non-hashable static arguments are not supported."):
        plxpr_fn([0.1, 0.2, 0.3], 0.1967)

    plxpr_fn((0.1, 0.2, 0.3), 0.1967)
