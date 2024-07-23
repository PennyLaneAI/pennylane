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
Tests for capturing conditionals into jaxpr.
"""
import numpy as np

# pylint: disable=protected-access
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


def cond_true_elifs_false(selector, arg):
    """A function with conditional containing true, elifs, and false branches."""

    def true_fn(arg):
        return 2 * arg

    def elif_fn1(arg):
        return arg - 1

    def elif_fn2(arg):
        return arg - 2

    def elif_fn3(arg):
        return arg - 3

    def elif_fn4(arg):
        return arg - 4

    def false_fn(arg):
        return 3 * arg

    return qml.cond(
        selector > 0,
        true_fn,
        false_fn,
        elifs=(
            (selector == -1, elif_fn1),
            (selector == -2, elif_fn2),
            (selector == -3, elif_fn3),
            (selector == -4, elif_fn4),
        ),
    )(arg)


def cond_true_elifs(selector, arg):
    """A function with conditional containing true and elifs branches."""

    def true_fn(arg):
        return 2 * arg

    def elif_fn1(arg):
        return arg - 1

    def elif_fn2(arg):
        return arg - 2

    return qml.cond(
        selector > 0,
        true_fn,
        elifs=(
            (selector == -1, elif_fn1),
            (selector == -2, elif_fn2),
        ),
    )(arg)


def cond_true_false(selector, arg):
    """A function with conditional containing true and false branches."""

    def true_fn(arg):
        return 2 * arg

    def false_fn(arg):
        return 3 * arg

    return qml.cond(
        selector > 0,
        true_fn,
        false_fn,
    )(arg)


def cond_true(selector, arg):
    """A function with conditional containing only the true branch."""

    def true_fn(arg):
        return 2 * arg

    return qml.cond(
        selector > 0,
        true_fn,
    )(arg)


# pylint: disable=no-self-use
class TestCond:
    """Tests for capturing conditional statements."""

    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, 10, 20),  # True condition
            (-1, 10, 9),  # Elif condition 1
            (-2, 10, 8),  # Elif condition 2
            (-3, 10, 7),  # Elif condition 3
            (-4, 10, 6),  # Elif condition 4
            (0, 10, 30),  # False condition
        ],
    )
    def test_cond_true_elifs_false(self, selector, arg, expected):
        """Test the conditional with true, elifs, and false branches."""

        result = cond_true_elifs_false(selector, arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, 10, 20),  # True condition
            (-1, 10, 9),  # Elif condition 1
            (-2, 10, 8),  # Elif condition 2
            (-3, 10, ()),  # No condition met
        ],
    )
    def test_cond_true_elifs(self, selector, arg, expected):
        """Test the conditional with true and elifs branches."""

        result = cond_true_elifs(selector, arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, 10, 20),  # True condition
            (0, 10, 30),  # False condition
        ],
    )
    def test_cond_true_false(self, selector, arg, expected):
        """Test the conditional with true and false branches."""

        result = cond_true_false(selector, arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, 10, 20),  # True condition
            (0, 10, ()),  # No condition met
        ],
    )
    def test_cond_true(self, selector, arg, expected):
        """Test the conditional with only the true branch."""

        result = cond_true(selector, arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
