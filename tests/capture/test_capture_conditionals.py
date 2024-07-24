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

# pylint: disable=redefined-outer-name

import jax.numpy as jnp
import numpy as np
import pytest

import pennylane as qml
from pennylane.ops.op_math.condition import _capture_cond

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """Enable and disable the PennyLane JAX capture context manager."""
    qml.capture.enable()
    yield
    qml.capture.disable()


@pytest.fixture
def testing_functions():
    """Returns a set of functions for testing."""

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

    return true_fn, false_fn, elif_fn1, elif_fn2, elif_fn3, elif_fn4


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
def test_cond_true_elifs_false(testing_functions, selector, arg, expected):
    """Test the conditional with true, elifs, and false branches."""

    true_fn, false_fn, elif_fn1, elif_fn2, elif_fn3, elif_fn4 = testing_functions

    result = qml.cond(
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
def test_cond_true_elifs(testing_functions, selector, arg, expected):
    """Test the conditional with true and elifs branches."""

    true_fn, _, elif_fn1, elif_fn2, _, _ = testing_functions

    result = qml.cond(
        selector > 0,
        true_fn,
        elifs=(
            (selector == -1, elif_fn1),
            (selector == -2, elif_fn2),
        ),
    )(arg)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "selector, arg, expected",
    [
        (1, 10, 20),  # True condition
        (0, 10, 30),  # False condition
    ],
)
def test_cond_true_false(testing_functions, selector, arg, expected):
    """Test the conditional with true and false branches."""

    true_fn, false_fn, _, _, _, _ = testing_functions

    result = qml.cond(
        selector > 0,
        true_fn,
        false_fn,
    )(arg)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


@pytest.mark.parametrize(
    "selector, arg, expected",
    [
        (1, 10, 20),  # True condition
        (0, 10, ()),  # No condition met
    ],
)
def test_cond_true(testing_functions, selector, arg, expected):
    """Test the conditional with only the true branch."""

    true_fn, _, _, _, _, _ = testing_functions

    result = qml.cond(
        selector > 0,
        true_fn,
    )(arg)
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_validate_number_of_output_variables():
    """Test mismatch in number of output variables."""

    def true_fn(x):
        return x + 1, x + 2

    def false_fn(x):
        return x + 1

    with pytest.raises(AssertionError, match=r"Mismatch in number of output variables"):
        jax.make_jaxpr(_capture_cond(True, true_fn, false_fn))(jnp.array(1))


def test_validate_output_variable_types():
    """Test mismatch in output variable types."""

    def true_fn(x):
        return x + 1, x + 2

    def false_fn(x):
        return x + 1, x + 2.0

    with pytest.raises(AssertionError, match=r"Mismatch in output abstract values"):
        jax.make_jaxpr(_capture_cond(True, true_fn, false_fn))(jnp.array(1))


def test_validate_elif_branches():
    """Test elif branch mismatches."""

    def true_fn(x):
        return x + 1, x + 2

    def false_fn(x):
        return x + 1, x + 2

    def elif_fn1(x):
        return x + 1, x + 2

    def elif_fn2(x):
        return x + 1, x + 2.0  # Type mismatch

    def elif_fn3(x):
        return x + 1  # Length mismatch

    with pytest.raises(
        AssertionError, match=r"Mismatch in output abstract values in elif branch #1"
    ):
        jax.make_jaxpr(
            _capture_cond(False, true_fn, false_fn, [(True, elif_fn1), (False, elif_fn2)])
        )(jnp.array(1))

    with pytest.raises(
        AssertionError, match=r"Mismatch in number of output variables in elif branch #0"
    ):
        jax.make_jaxpr(_capture_cond(False, true_fn, false_fn, [(True, elif_fn3)]))(jnp.array(1))


@pytest.mark.parametrize(
    "true_fn, false_fn, expected_error, match",
    [
        (
            lambda x: (x + 1, x + 2),
            lambda x: (x + 1),
            AssertionError,
            r"Mismatch in number of output variables",
        ),
        (
            lambda x: (x + 1, x + 2),
            lambda x: (x + 1, x + 2.0),
            AssertionError,
            r"Mismatch in output abstract values",
        ),
    ],
)
def test_validate_mismatches(true_fn, false_fn, expected_error, match):
    """Test mismatch in number and type of output variables."""
    with pytest.raises(expected_error, match=match):
        jax.make_jaxpr(_capture_cond(True, true_fn, false_fn))(jnp.array(1))
