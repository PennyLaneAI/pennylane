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

import pennylane as qp
from pennylane.control_flow.for_loop import ForLoopCallable


@pytest.mark.capture
@pytest.mark.jax
def test_early_exit():
    """Test we exit early when start==stop."""
    import jax

    @qp.for_loop(0)
    def inner_loop(i, x):  # pylint: disable=unused-argument
        x += 1
        return x

    jaxpr = jax.make_jaxpr(inner_loop)(0)
    assert len(jaxpr.eqns) == 0
    assert inner_loop(4) == 4


def test_error_if_outputs_when_no_inputs():
    """Test an error is raised if there is an output when there is no additional input."""

    @qp.for_loop(3)
    def f(i):  # pylint: disable=unused-argument
        return 2

    with pytest.raises(ValueError, match="should not return anything "):
        f()


def test_for_loop_python_fallback():
    """Test that qp.for_loop fallsback to Python
    interpretation if Catalyst is not available"""
    dev = qp.device("default.qubit", wires=3)

    @qp.qnode(dev)
    def circuit(x, n):

        # for loop with dynamic bounds
        @qp.for_loop(0, n, 1)
        def loop_fn(i):
            qp.Hadamard(wires=i)

        # nested for loops.
        # outer for loop updates x
        @qp.for_loop(0, n, 1)
        def loop_fn_returns(i, x):
            qp.RX(x, wires=i)

            # inner for loop
            @qp.for_loop(i + 1, n, 1)
            def inner(j):
                qp.CRY(x**2, [i, j])

            inner()

            return x + 0.1

        loop_fn()
        loop_fn_returns(x)

        return qp.expval(qp.PauliZ(0))

    x = 0.5

    res = qp.workflow.construct_tape(circuit)(x, 3).operations
    expected = [
        qp.Hadamard(wires=[0]),
        qp.Hadamard(wires=[1]),
        qp.Hadamard(wires=[2]),
        qp.RX(0.5, wires=[0]),
        qp.CRY(0.25, wires=[0, 1]),
        qp.CRY(0.25, wires=[0, 2]),
        qp.RX(0.6, wires=[1]),
        qp.CRY(0.36, wires=[1, 2]),
        qp.RX(0.7, wires=[2]),
    ]

    _ = [qp.assert_equal(i, j) for i, j in zip(res, expected)]


class TestForLoopHints:
    """Tests for ``num-iters`` compiler hints on :func:`~.for_loop`."""

    def test_typo_on_hinted_body_is_canonicalized(self):
        """A typo'd key on a HintedCallable body should still set the hint."""

        @qp.hint({"num_iters": 10})
        def body(i, x):  # pylint: disable=unused-argument
            return x + 1

        loop = qp.for_loop(3)(body)
        assert loop.num_iters_hint == 10
        assert loop(0) == 3

    def test_typo_on_apply_hint_is_canonicalized(self):
        """Applying a typo'd hint to a for-loop callable should canonicalize it."""

        def body(i, x):  # pylint: disable=unused-argument
            return x + 1

        loop = qp.hint({"num_iters": 10})(qp.for_loop(3)(body))
        assert loop.num_iters_hint == 10
        assert loop(0) == 3

    def test_unknown_hint_on_body_is_ignored(self):
        """Unrecognized hint keys on the body should be ignored."""

        @qp.hint({"identity": True})
        def body(i, x):  # pylint: disable=unused-argument
            return x + 1

        loop = qp.for_loop(3)(body)
        assert loop.num_iters_hint is None
        assert loop(0) == 3

    def test_unknown_hint_on_apply_is_ignored(self):
        """Unrecognized keys applied to a for-loop callable should be ignored."""

        def body(i, x):  # pylint: disable=unused-argument
            return x + 1

        loop = qp.hint({"identity": True})(qp.for_loop(3)(body))
        assert loop.num_iters_hint is None
        assert loop(0) == 3

    def test_valid_hint_on_body(self):
        """A correctly spelled hint on the body should be accepted."""

        @qp.hint({"num-iters": 10})
        def body(i, x):  # pylint: disable=unused-argument
            return x + 1

        loop = qp.for_loop(3)(body)
        assert loop.num_iters_hint == 10
        assert loop(0) == 3

    def test_valid_apply_hint(self):
        """Applying a correctly spelled hint should set ``num_iters_hint``."""

        def body(i, x):  # pylint: disable=unused-argument
            return x + 1

        loop = qp.hint({"num-iters": 7})(qp.for_loop(3)(body))
        assert loop.num_iters_hint == 7
        assert loop(0) == 3

    def test_direct_form_preserves_for_loop_callable(self):
        """``qp.hint(loop, hints)`` should return a ``ForLoopCallable`` with the new hint."""

        @qp.for_loop(3)
        def loop(i, x):  # pylint: disable=unused-argument
            return x + 1

        new_loop = qp.hint(loop, {"num-iters": 4})
        assert isinstance(new_loop, ForLoopCallable)
        assert new_loop.num_iters_hint == 4
        assert new_loop(0) == 3
