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

import pytest

pytestmark = pytest.mark.capture

jax = pytest.importorskip("jax")

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
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (a and b)

    @pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
    def test_pennylane_circuit(self, a, b):
        """Test that the logical AND operation can be used in a PennyLane circuit."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x: bool, y: bool):
            if x and y:
                qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        result = circuit(x=a, y=b)
        assert result == 1.0 if (a and b) else -1.0
        # Expect PauliX to flip the state to |1> if both are True, else |0>


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
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (a or b)

    @pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
    def test_pennylane_circuit(self, a, b):
        """Test that the logical OR operation can be used in a PennyLane circuit."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x: bool, y: bool):
            if x or y:
                qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        result = circuit(x=a, y=b)
        assert (
            result == 1.0 if (a or b) else -1.0
        )  # Expect PauliX to flip the state to |1> if either is True, else |0>


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
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        assert result[0] == (not x)

    @pytest.mark.parametrize("a", [True, False])
    def test_pennylane_circuit(self, a):
        """Test that the logical NOT operation can be used in a PennyLane circuit."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x: bool):
            if not x:
                qml.PauliX(0)
            return qml.expval(qml.PauliZ(0))

        result = circuit(x=a)
        assert (
            result == -1.0 if a else 1.0
        )  # Expect PauliX to flip the state to |1> if x is False, else |0>


# pylint: disable=too-few-public-methods
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestIntegration:
    """Tests the integration of logical operations with other functions."""

    @pytest.mark.parametrize("a, b", [(True, True), (True, False), (False, True), (False, False)])
    def test_combined_operations(self, a, b):
        """Test combining logical operations with arithmetic operations."""

        fn = lambda x, y: (x and y) or (not x and not y)
        ag_fn = run_autograph(fn)

        args = (a, b)
        ag_fn_jaxpr = make_jaxpr(ag_fn)(*args)
        result = eval_jaxpr(ag_fn_jaxpr.jaxpr, ag_fn_jaxpr.consts, *args)

        # The result should be True if both are True or both are False
        expected_result = (a and b) or (not a and not b)
        assert result[0] == expected_result
