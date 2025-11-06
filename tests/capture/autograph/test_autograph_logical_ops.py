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
"""The tests for logical operations in AutoGraph"""

# pylint: disable = unnecessary-lambda-assignment, wrong-import-position

import pytest

pytestmark = pytest.mark.capture

jax = pytest.importorskip("jax")
import jax.numpy as jnp

# pylint: disable=wrong-import-position
from jax import make_jaxpr
from jax.core import eval_jaxpr

import pennylane as qml
from pennylane.capture.autograph import run_autograph


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestAnd:
    """Tests the logical AND operation."""

    @pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
    def test_truth_table(self, a, b):
        """Test the truth table for the logical AND operation."""

        fn = lambda x, y: x and y
        ag_fn = run_autograph(fn)

        args = (a, b)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        assert ag_fn_jaxpr.jaxpr.eqns[0].primitive == jax.lax.and_p
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (a and b)

    @pytest.mark.parametrize("val", (0, 0.0, 0.0j, 1, 1.0, 1 + 0.0j))
    @pytest.mark.parametrize("x", [True, False])
    def test_truthy_and_falsy_values(self, x, val):
        """Test that truthy and falsy values also work."""

        fn = lambda x, y: x and y
        ag_fn = run_autograph(fn)

        args = (x, val)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (x and val)

    @pytest.mark.parametrize(
        "python_object",
        [
            "",
            "string",
            [],
            [1, 2],
            {},
            {1: "a"},
        ],
    )
    @pytest.mark.parametrize("arg", [True, False])
    def test_python_object_interaction(self, python_object, arg):
        """Test that logical AND works with Python objects."""

        fn = lambda: 1 if arg and python_object else 0

        ag_fn = run_autograph(fn)

        args = ()
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (1 if arg and python_object else 0)

    @pytest.mark.parametrize("x", [True, False])
    def test_static_dynamic_arg_mixed(self, x):
        """Tests that the logical operation works if provided with static and dynamic arguments."""

        def fn(dynamic_arg):
            x = 1
            static_arg = x >= 0
            if static_arg and dynamic_arg:
                return 1
            return 0

        ag_fn = run_autograph(fn)

        args = (x,)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (x and True)

    @pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
    def test_qnode_integration(self, a, b):
        """Test that the logical AND operation can be used in a PennyLane circuit."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x: bool, y: bool):
            if x and y:
                qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        result = run_autograph(circuit)(x=a, y=b)
        assert result == -1.0 if (a and b) else 1.0

    def test_while_loop_integration(self):
        """Test while loop integration with logical operations and AutoGraph."""

        def fn(a, b):
            counter = 0
            while a < b and counter < 5:
                a += 1
                counter += 1
            return a

        ag_fn = run_autograph(fn)
        args = (0, 10)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        # Counter makes loop bail earlier.
        assert result[0] == 5, f"Expected 5, got {result[0]}"

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (1, 2, (2, 2)),  # first branch: a > 0 and b < 5
            (5, 5, (5, 6)),  # second branch: a == 5 and b == 5
            (5, 6, (4, 5)),  # else branch
        ],
    )
    def test_cond_integration(self, a, b, expected):
        """Test conditional integration with logical AND and AutoGraph, covering all branches."""

        def fn(a, b):
            if a > 0 and b < 5:
                a += 1
            elif a == 5 and b == 5:
                b += 1
            else:
                a -= 1
                b -= 1
            return a, b

        ag_fn = run_autograph(fn)
        args = (a, b)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert jnp.array_equal(
            result,
            (expected),
        ), f"Expected {expected}, got {result}"


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestOr:
    """Tests the logical OR operation."""

    @pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
    def test_truth_table(self, a, b):
        """Test the truth table for the logical OR operation."""

        fn = lambda x, y: x or y
        ag_fn = run_autograph(fn)

        args = (a, b)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        assert ag_fn_jaxpr.jaxpr.eqns[0].primitive == jax.lax.or_p
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (a or b)

    @pytest.mark.parametrize("val", (0, 0.0, 0.0j, 1, 1.0, 1 + 0.0j))
    @pytest.mark.parametrize("x", [True, False])
    def test_truthy_and_falsy_values(self, x, val):
        """Test that truthy and falsy values also work."""

        fn = lambda x, y: x or y
        ag_fn = run_autograph(fn)

        args = (x, val)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (x or val)

    @pytest.mark.parametrize(
        "python_object",
        [
            "",
            "string",
            [],
            [1, 2],
            {},
            {1: "a"},
        ],
    )
    @pytest.mark.parametrize("arg", [True, False])
    def test_python_object_interaction(self, python_object, arg):
        """Test that logical OR works with Python objects."""

        fn = lambda: 1 if arg or python_object else 0

        ag_fn = run_autograph(fn)

        args = ()
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (1 if arg or python_object else 0)

    @pytest.mark.parametrize("x", [True, False])
    def test_static_dynamic_arg_mixed(self, x):
        """Tests that the logical operation works if provided with static and dynamic arguments."""

        def fn(dynamic_arg):
            x = 1
            static_arg = x >= 0
            if static_arg or dynamic_arg:
                return 1
            return 0

        ag_fn = run_autograph(fn)

        args = (x,)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (x or True)

    @pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
    def test_qnode_integration(self, a, b):
        """Test that the logical OR operation can be used in a PennyLane circuit."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x: bool, y: bool):
            if x or y:
                qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        result = run_autograph(circuit)(x=a, y=b)
        assert result == -1.0 if (a or b) else 1.0

    def test_while_loop_integration(self):
        """Test while loop integration with logical operations and AutoGraph."""

        def fn(a, b):
            counter = 0
            while a < b or counter < 5:
                a += 1
                counter += 1
            return a

        ag_fn = run_autograph(fn)
        args = (0, 10)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        # Despite the counter constraint, a will continue to increment.
        assert result[0] == 10, f"Expected 10, got {result[0]}"

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (10, 4, (11, 4)),  # first branch: a > 0
            (0, 6, (1, 6)),  # first branch: b > 5
            (-5, 5, (-5, 6)),  # second branch: a == -5
            (0, -5, (0, -4)),  # second branch: b == -5
            (0, 5, (-1, 4)),  # else branch
        ],
    )
    def test_cond_integration(self, a, b, expected):
        """Test conditional integration with logical OR and AutoGraph, covering all branches."""

        def fn(a, b):
            if a > 0 or b > 5:
                a += 1
            elif a == -5 or b == -5:
                b += 1
            else:
                a -= 1
                b -= 1
            return a, b

        ag_fn = run_autograph(fn)
        args = (a, b)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert jnp.array_equal(
            result, (expected)
        ), f"For input ({a}, {b}), expected {expected}, got {result}"


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestNot:
    """Tests the logical NOT operation."""

    @pytest.mark.parametrize("x", [True, False])
    def test_truth_table(self, x):
        """Test the truth table for the logical NOT operation."""

        fn = lambda x: not x
        ag_fn = run_autograph(fn)

        args = (x,)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        assert ag_fn_jaxpr.jaxpr.eqns[0].primitive == jax.lax.not_p
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (not x)

    @pytest.mark.parametrize("val", (0, 0.0, 0.0j, 1, 1.0, 1 + 0.0j))
    def test_truthy_and_falsy_values(self, val):
        """Test that truthy and falsy values also work."""

        fn = lambda x: not x
        ag_fn = run_autograph(fn)

        args = (val,)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (not val)

    @pytest.mark.parametrize(
        "python_object",
        [
            "",
            "string",
            [],
            [1, 2],
            {},
            {1: "a"},
        ],
    )
    def test_python_object_interaction(self, python_object):
        """Test that logic al NOT works with Python objects."""

        fn = lambda: 1 if not python_object else 0

        ag_fn = run_autograph(fn)

        args = ()
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (1 if not python_object else 0)

    def test_static_args(self):
        """Test that static arguments can be used."""

        def fn_static():
            static_arg = False
            if not static_arg:
                return 1
            return 0

        ag_fn_static = run_autograph(fn_static)
        ag_fn_static_jaxpr = make_jaxpr(ag_fn_static)()
        result = eval_jaxpr(ag_fn_static_jaxpr.jaxpr, ag_fn_static_jaxpr.consts)

        assert result[0] == 1

    @pytest.mark.parametrize("a", [True, False])
    def test_qnode_integration(self, a):
        """Test that the logical NOT operation can be used in a PennyLane circuit."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x: bool):
            if not x:
                qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        result = run_autograph(circuit)(x=a)
        assert result == -1.0 if (not a) else 1.0

    def test_while_loop_integration(self):
        """Test while loop integration with logical operations and AutoGraph."""

        def fn(a, b):
            while not a >= b:
                a += 1
            return a

        ag_fn = run_autograph(fn)
        args = (0, 10)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == 10, f"Expected 10, got {result[0]}"

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (-1, 0, (0, 0)),  # first branch: a < 0
            (1, 1, (1, 2)),  # first branch: b > 0
            (1, -1, (0, -2)),  # else branch
        ],
    )
    def test_cond_integration(self, a, b, expected):
        """Test conditional integration with logical OR and AutoGraph, covering all branches."""

        def fn(a, b):
            if not a > 0:
                a += 1
            elif not b < 0:
                b += 1
            else:
                a -= 1
                b -= 1
            return a, b

        ag_fn = run_autograph(fn)
        args = (a, b)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert jnp.array_equal(
            result, (expected)
        ), f"For input ({a}, {b}), expected {expected}, got {result}"


@pytest.mark.usefixtures("enable_disable_plxpr")
@pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
def test_combined_operations(a, b):
    """Test how all of the logical operations coexist together."""

    fn = lambda x, y: (x and y) or (not x and not y)
    ag_fn = run_autograph(fn)

    args = (a, b)
    ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
    result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

    expected_result = (a and b) or (not a and not b)
    assert result[0] == expected_result
