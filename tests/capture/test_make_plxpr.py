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
Tests for the make_plxpr function for capturing functions as jaxpr.
"""

from unittest.mock import call

import numpy as np
import pytest

import pennylane as qml
from pennylane.capture import make_plxpr

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

# must be below jax importorskip
from jax import numpy as jnp  # pylint: disable=wrong-import-position, wrong-import-order


def test_error_is_raised_with_capture_disabled():
    """Test that an error is raised."""

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circ(x):
        qml.RX(x, 0)
        qml.Hadamard(0)
        return qml.expval(qml.X(0))

    with pytest.raises(RuntimeError, match="requires PennyLane capture to be enabled"):
        _ = make_plxpr(circ)(1.2)


@pytest.mark.capture
class TestMakePLxPR:
    """Tests the basic make_plxpr functionality"""

    def test_make_plxpr(self, mocker):
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
        isinstance(plxpr, jax.extend.core.ClosedJaxpr)  # pylint: disable=protected-access

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("static_argnums", [[0], [1], [0, 1], []])
    def test_static_argnums(self, static_argnums, autograph, mocker):
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
        plxpr = make_plxpr(circ, autograph=autograph, static_argnums=static_argnums)(*params)

        if not autograph:
            # when using autograph, we don't have the function make_jaxpr was called with
            # (it was produced by run_autograph), so we can't check for the function call.
            spy.assert_has_calls([call(circ, static_argnums=static_argnums)])

        # plxpr behaves as expected wrt static argnums
        res = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, *non_static_params)
        assert np.allclose(res, circ(*params))

    @pytest.mark.parametrize("autograph", [True, False])
    def test_kwargs(self, mocker, autograph):
        """Test additional kwargs are passed through to make_jaxpr"""

        dev = qml.device("default.qubit", wires=1)

        spy = mocker.spy(jax, "make_jaxpr")

        @qml.qnode(dev)
        def circ():
            qml.Hadamard(0)
            return qml.expval(qml.X(0))

        output = make_plxpr(circ, autograph=autograph, return_shape=True)()

        # assert new value for return_shape is passed to make_jaxpr
        if not autograph:
            # when using autograph, we don't have the function make_jaxpr was called with
            # (it was produced by run_autograph), so we can't check for the function call.
            spy.assert_has_calls([call(circ, static_argnums=(), return_shape=True)], any_order=True)

        # output is as expected for return_shape=True
        assert len(output) == 2
        isinstance(output[0], jax.extend.core.ClosedJaxpr)  # pylint: disable=protected-access
        isinstance(output[0], jax.ShapeDtypeStruct)  # pylint: disable=protected-access


@pytest.mark.capture
class TestAutoGraphIntegration:
    """Test autograph integration for converting Python control flow into native PennyLane
    `cond`, `for_loop` and `while_loop`. Note that autograph defaults to True in make_plxpr."""

    def test_if_stmt(self):
        """Test that an if statement is converted to a jaxpr with a ``cond`` function, and
        that in the case of a QNode, the resulting plxpr can be evaluated as expected"""

        def func(x):
            if x > 1.967:
                qml.Hadamard(0)
            else:
                qml.Y(0)
            return qml.state()

        dev = qml.device("default.qubit", wires=1)
        qnode = qml.QNode(func, dev)

        plxpr1 = qml.capture.make_plxpr(func)(2)
        plxpr2 = qml.capture.make_plxpr(qnode)(2)

        # the plxpr includes a representation of a `cond` function
        assert "cond[" in str(plxpr1)
        assert "cond[" in str(plxpr2)

        def eval(x):
            return jax.core.eval_jaxpr(plxpr2.jaxpr, plxpr2.consts, x)

        assert np.allclose(eval(2), [0.70710678, 0.70710678])
        assert np.allclose(eval(1), [0, 1j])

    def test_while_loop(self):
        """Test that a while loop is converted to a jaxpr with a ``while_loop`` function, and
        that in the case of a QNode, the resulting plxpr can be evaluated as expected"""

        def func(counter):
            while counter < 10:
                qml.RX(np.pi * 0.1, wires=0)
                counter += 1

            return qml.expval(qml.Z(0))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(func, dev)

        plxpr1 = qml.capture.make_plxpr(func)(0)
        plxpr2 = qml.capture.make_plxpr(qnode)(0)

        # the plxpr includes a representation of a `while_loop` function
        assert "while_loop[" in str(plxpr1)
        assert "while_loop[" in str(plxpr2)

        def eval(x):
            return jax.core.eval_jaxpr(plxpr2.jaxpr, plxpr2.consts, x)

        assert np.allclose(eval(0), [-1])
        assert np.allclose(eval(5), [0])

    def test_for_loop(self):
        """Test that a for loop is converted to a jaxpr with a ``for_loop`` function, and
        that in the case of a QNode, the resulting plxpr can be evaluated as expected"""

        def func(angles):
            for i, x in enumerate(angles):
                qml.RX(x, wires=i)

            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(func, dev)

        plxpr1 = qml.capture.make_plxpr(func)(jnp.array([0.0, 0.0]))
        plxpr2 = qml.capture.make_plxpr(qnode)(jnp.array([0.0, 0.0]))

        # the plxpr includes a representation of a `for_loop` function
        assert "for_loop[" in str(plxpr1)
        assert "for_loop[" in str(plxpr2)

        def eval(x):
            x = jnp.array(x)
            return jax.core.eval_jaxpr(plxpr2.jaxpr, plxpr2.consts, x)

        assert np.allclose(eval([np.pi, np.pi / 2]), [-1, 0])
