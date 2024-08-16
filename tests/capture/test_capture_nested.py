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
Tests for capturing of nested controlled flows into jaxpr.
"""

# pylint: disable=no-value-for-parameter
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=no-self-use

import numpy as np
import pytest

import pennylane as qml

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """Enable and disable the PennyLane JAX capture context manager."""
    qml.capture.enable()
    yield
    qml.capture.disable()


@pytest.mark.parametrize("upper_bound, arg, expected", [(3, 0.5, 0.00223126), (2, 12, 0.2653001)])
def test_nested_for_and_while_loop(upper_bound, arg, expected):
    """Test that a nested for loop and while loop is correctly captured into a jaxpr."""

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(upper_bound, arg):

        # for loop with dynamic bounds
        @qml.for_loop(0, upper_bound, 1)
        def loop_fn(i):
            qml.Hadamard(wires=i)

        # nested for loops.
        # outer for loop updates x
        @qml.for_loop(0, upper_bound, 1)
        def loop_fn_returns(i, x):
            qml.RX(x, wires=i)

            # inner while loop
            @qml.while_loop(lambda j: j < upper_bound)
            def inner(j):
                qml.RZ(j, wires=0)
                qml.RY(x**2, wires=0)
                return j + 1

            inner(i + 1)

            return x + 0.1

        loop_fn()
        loop_fn_returns(arg)

        return qml.expval(qml.Z(0))

    args = [upper_bound, arg]
    result = circuit(*args)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    jaxpr = jax.make_jaxpr(circuit)(*args)
    res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
    assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"


@pytest.mark.parametrize("upper_bound, arg", [(3, [0.1, 0.3, 0.5]), (2, [2, 7, 12])])
def test_nested_cond_for_while_loop(upper_bound, arg):
    """Test that a nested control flows are correctly captured into a jaxpr."""

    dev = qml.device("default.qubit", wires=3)

    # Control flow for qml.conds
    def true_fn(_):
        @qml.for_loop(0, upper_bound, 1)
        def loop_fn(i):
            qml.Hadamard(wires=i)

        loop_fn()

    def elif_fn(arg):
        qml.RY(arg**2, wires=[2])

    def false_fn(arg):
        qml.RY(-arg, wires=[2])

    @qml.qnode(dev)
    def circuit(upper_bound, arg):
        qml.RY(-np.pi / 2, wires=[2])
        m_0 = qml.measure(2)

        # NOTE: qml.cond(m_0, qml.RX)(arg[1], wires=1) doesn't work
        def rx_fn():
            qml.RX(arg[1], wires=1)

        qml.cond(m_0, rx_fn)()

        def ry_fn():
            qml.RY(arg[1] ** 3, wires=1)

        # nested for loops.
        # outer for loop updates x
        @qml.for_loop(0, upper_bound, 1)
        def loop_fn_returns(i, x):
            qml.RX(x, wires=i)
            m_1 = qml.measure(0)
            # NOTE: qml.cond(m_0, qml.RY)(arg[1], wires=1) doesn't work
            qml.cond(m_1, ry_fn)()

            # inner while loop
            @qml.while_loop(lambda j: j < upper_bound)
            def inner(j):
                qml.RZ(j, wires=0)
                qml.RY(x**2, wires=0)
                m_2 = qml.measure(0)
                qml.cond(m_2, true_fn=true_fn, false_fn=false_fn, elifs=((m_1, elif_fn)))(arg[0])
                return j + 1

            inner(i + 1)
            return x + 0.1

        loop_fn_returns(arg[2])

        return qml.expval(qml.Z(0))

    args = [upper_bound, arg]
    result = circuit(*args)
    jaxpr = jax.make_jaxpr(circuit)(*args)
    res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, upper_bound, *arg)
    assert np.allclose(result, res_ev_jxpr), f"Expected {result}, but got {res_ev_jxpr}"
