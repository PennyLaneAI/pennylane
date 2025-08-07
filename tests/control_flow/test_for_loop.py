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
Tests for the for_loop
"""
import pytest

import pennylane as qml


@pytest.mark.capture
@pytest.mark.jax
def test_early_exit():
    """Test we exit early when start==stop."""
    import jax

    @qml.for_loop(0)
    def inner_loop(i, x):  # pylint: disable=unused-argument
        x += 1
        return x

    jaxpr = jax.make_jaxpr(inner_loop)(0)
    assert len(jaxpr.eqns) == 0
    assert inner_loop(4) == 4


def test_for_loop_python_fallback():
    """Test that qml.for_loop fallsback to Python
    interpretation if Catalyst is not available"""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(x, n):

        # for loop with dynamic bounds
        @qml.for_loop(0, n, 1)
        def loop_fn(i):
            qml.Hadamard(wires=i)

        # nested for loops.
        # outer for loop updates x
        @qml.for_loop(0, n, 1)
        def loop_fn_returns(i, x):
            qml.RX(x, wires=i)

            # inner for loop
            @qml.for_loop(i + 1, n, 1)
            def inner(j):
                qml.CRY(x**2, [i, j])

            inner()

            return x + 0.1

        loop_fn()
        loop_fn_returns(x)

        return qml.expval(qml.PauliZ(0))

    x = 0.5

    res = qml.workflow.construct_tape(circuit)(x, 3).operations
    expected = [
        qml.Hadamard(wires=[0]),
        qml.Hadamard(wires=[1]),
        qml.Hadamard(wires=[2]),
        qml.RX(0.5, wires=[0]),
        qml.CRY(0.25, wires=[0, 1]),
        qml.CRY(0.25, wires=[0, 2]),
        qml.RX(0.6, wires=[1]),
        qml.CRY(0.36, wires=[1, 2]),
        qml.RX(0.7, wires=[2]),
    ]

    _ = [qml.assert_equal(i, j) for i, j in zip(res, expected)]
