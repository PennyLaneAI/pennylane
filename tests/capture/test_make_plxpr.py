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

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

# must be below jax importorskip
from pennylane.capture import make_plxpr  # pylint: disable=wrong-import-position


def test_error_is_raised_with_capture_disabled():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circ(x):
        qml.RX(x, 0)
        qml.Hadamard(0)
        return qml.expval(qml.X(0))

    with pytest.raises(RuntimeError, match="requires PennyLane capture to be enabled"):
        _ = make_plxpr(circ)(1.2)


@pytest.mark.usefixtures("enable_disable_plxpr")
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
        isinstance(plxpr, jax._src.core.ClosedJaxpr)  # pylint: disable=protected-access

    @pytest.mark.parametrize("static_argnums", [[0], [1], [0, 1], []])
    def test_static_argnums(self, static_argnums, mocker):
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

    def test_kwargs(self, mocker):
        """Test additional kwargs are passed through to make_jaxpr"""

        dev = qml.device("default.qubit", wires=1)

        spy = mocker.spy(jax, "make_jaxpr")

        @qml.qnode(dev)
        def circ():
            qml.Hadamard(0)
            return qml.expval(qml.X(0))

        # assert new value for return_shape is passed to make_jaxpr
        _ = make_plxpr(circ, return_shape=True)()
        spy.assert_has_calls([call(circ, static_argnums=(), return_shape=True)])
