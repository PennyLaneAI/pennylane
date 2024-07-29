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
# pylint: disable=no-self-use

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


class TestCond:
    """Tests for conditional functions using qml.cond."""

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
    def test_cond_true_elifs_false(self, testing_functions, selector, arg, expected):
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
    def test_cond_true_elifs(self, testing_functions, selector, arg, expected):
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
            (1, 10, 20),
            (0, 10, 30),
        ],
    )
    def test_cond_true_false(self, testing_functions, selector, arg, expected):
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
            (1, 10, 20),
            (0, 10, ()),
        ],
    )
    def test_cond_true(self, testing_functions, selector, arg, expected):
        """Test the conditional with only the true branch."""
        true_fn, _, _, _, _, _ = testing_functions

        result = qml.cond(
            selector > 0,
            true_fn,
        )(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, jax.numpy.array([2, 3]), 12),
            (0, jax.numpy.array([2, 3]), 15),
        ],
    )
    def test_cond_with_jax_array(self, selector, arg, expected):
        """Test the conditional with array arguments."""

        def true_fn(jax_array):
            return jax_array[0] * jax_array[1] * 2

        def false_fn(jax_array):
            return jax_array[0] * jax_array[1] * 2.5

        result = qml.cond(selector > 0, true_fn, false_fn)(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"


class TestCondReturns:
    """Tests for validating the number and types of output variables in conditional functions."""

    @pytest.mark.parametrize(
        "true_fn, false_fn, expected_error, match",
        [
            (
                lambda x: (x + 1, x + 2),
                lambda x: None,
                ValueError,
                r"Mismatch in number of output variables",
            ),
            (
                lambda x: (x + 1, x + 2),
                lambda x: (x + 1,),
                ValueError,
                r"Mismatch in number of output variables",
            ),
            (
                lambda x: (x + 1, x + 2),
                lambda x: (x + 1, x + 2.0),
                ValueError,
                r"Mismatch in output abstract values",
            ),
        ],
    )
    def test_validate_mismatches(self, true_fn, false_fn, expected_error, match):
        """Test mismatch in number and type of output variables."""
        with pytest.raises(expected_error, match=match):
            jax.make_jaxpr(_capture_cond(True, true_fn, false_fn))(jax.numpy.array(1))

    def test_validate_number_of_output_variables(self):
        """Test mismatch in number of output variables."""

        def true_fn(x):
            return x + 1, x + 2

        def false_fn(x):
            return x + 1

        with pytest.raises(ValueError, match=r"Mismatch in number of output variables"):
            jax.make_jaxpr(_capture_cond(True, true_fn, false_fn))(jax.numpy.array(1))

    def test_validate_output_variable_types(self):
        """Test mismatch in output variable types."""

        def true_fn(x):
            return x + 1, x + 2

        def false_fn(x):
            return x + 1, x + 2.0

        with pytest.raises(ValueError, match=r"Mismatch in output abstract values"):
            jax.make_jaxpr(_capture_cond(True, true_fn, false_fn))(jax.numpy.array(1))

    def test_validate_no_false_branch_with_return(self):
        """Test no false branch provided with return variables."""

        def true_fn(x):
            return x + 1, x + 2

        with pytest.raises(
            ValueError,
            match=r"The false branch must be provided if the true branch returns any variables",
        ):
            jax.make_jaxpr(_capture_cond(True, true_fn))(jax.numpy.array(1))

    def test_validate_no_false_branch_with_return_2(self):
        """Test no false branch provided with return variables."""

        def true_fn(x):
            return x + 1, x + 2

        def elif_fn(x):
            return x + 1, x + 2

        with pytest.raises(
            ValueError,
            match=r"The false branch must be provided if the true branch returns any variables",
        ):
            jax.make_jaxpr(_capture_cond(True, true_fn, false_fn=None, elifs=(False, elif_fn)))(
                jax.numpy.array(1)
            )

    def test_validate_elif_branches(self):
        """Test elif branch mismatches."""

        def true_fn(x):
            return x + 1, x + 2

        def false_fn(x):
            return x + 1, x + 2

        def elif_fn1(x):
            return x + 1, x + 2

        def elif_fn2(x):
            return x + 1, x + 2.0

        def elif_fn3(x):
            return x + 1

        with pytest.raises(
            ValueError, match=r"Mismatch in output abstract values in elif branch #1"
        ):
            jax.make_jaxpr(
                _capture_cond(False, true_fn, false_fn, [(True, elif_fn1), (False, elif_fn2)])
            )(jax.numpy.array(1))

        with pytest.raises(
            ValueError, match=r"Mismatch in number of output variables in elif branch #0"
        ):
            jax.make_jaxpr(_capture_cond(False, true_fn, false_fn, [(True, elif_fn3)]))(
                jax.numpy.array(1)
            )


dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit(pred):
    """Quantum circuit with only a true branch."""

    def true_fn():
        qml.RX(0.1, wires=0)

    qml.cond(pred > 0, true_fn)()
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev)
def circuit_branches(pred, arg1, arg2):
    """Quantum circuit with conditional branches."""

    qml.RX(0.10, wires=0)

    def true_fn(arg1, arg2):
        qml.RY(arg1, wires=0)
        qml.RX(arg2, wires=0)
        qml.RZ(arg1, wires=0)

    def false_fn(arg1, arg2):
        qml.RX(arg1, wires=0)
        qml.RX(arg2, wires=0)

    def elif_fn1(arg1, arg2):
        qml.RZ(arg2, wires=0)
        qml.RX(arg1, wires=0)

    qml.cond(pred > 0, true_fn, false_fn, elifs=(pred == -1, elif_fn1))(arg1, arg2)
    qml.RX(0.10, wires=0)
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev)
def circuit_with_returned_operator(pred, arg1, arg2):
    """Quantum circuit with conditional branches that return operators."""

    qml.RX(0.10, wires=0)

    def true_fn(arg1, arg2):
        qml.RY(arg1, wires=0)
        return 7, 4.6, qml.RY(arg2, wires=0), True

    def false_fn(arg1, arg2):
        qml.RZ(arg2, wires=0)
        return 2, 2.2, qml.RZ(arg1, wires=0), False

    qml.cond(pred > 0, true_fn, false_fn)(arg1, arg2)
    qml.RX(0.10, wires=0)
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(dev)
def circuit_multiple_cond(tmp_pred, tmp_arg):
    """Quantum circuit with multiple dynamic conditional branches."""

    dyn_pred_1 = tmp_pred > 0
    arg = tmp_arg

    def true_fn_1(arg):
        return True, qml.RX(arg, wires=0)

    # pylint: disable=unused-argument
    def false_fn_1(arg):
        return False, qml.RY(0.1, wires=0)

    def true_fn_2(arg):
        return qml.RX(arg, wires=0)

    # pylint: disable=unused-argument
    def false_fn_2(arg):
        return qml.RY(0.1, wires=0)

    [dyn_pred_2, _] = qml.cond(dyn_pred_1, true_fn_1, false_fn_1, elifs=())(arg)
    qml.cond(dyn_pred_2, true_fn_2, false_fn_2, elifs=())(arg)
    return qml.expval(qml.Z(0))


@qml.qnode(dev)
def circuit_with_consts(pred, arg):
    """Quantum circuit with jaxpr constants."""

    # these are captured as consts
    arg1 = arg
    arg2 = arg + 0.2
    arg3 = arg + 0.3
    arg4 = arg + 0.4
    arg5 = arg + 0.5
    arg6 = arg + 0.6

    def true_fn():
        qml.RX(arg1, 0)

    def false_fn():
        qml.RX(arg2, 0)
        qml.RX(arg3, 0)

    def elif_fn1():
        qml.RX(arg4, 0)
        qml.RX(arg5, 0)
        qml.RX(arg6, 0)

    qml.cond(pred > 0, true_fn, false_fn, elifs=((pred == 0, elif_fn1),))()

    return qml.expval(qml.Z(0))


class TestCondCircuits:
    """Tests for conditional quantum circuits."""

    @pytest.mark.parametrize(
        "pred, expected",
        [
            (1, 0.99500417),  # RX(0.1)
            (0, 1.0),  # No operation
        ],
    )
    def test_circuit(self, pred, expected):
        """Test circuit with only a true branch."""
        result = circuit(pred)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "pred, arg1, arg2, expected",
        [
            (1, 0.5, 0.6, 0.63340907),  # RX(0.10) -> RY(0.5) -> RX(0.6) -> RZ(0.5) -> RX(0.10)
            (0, 0.5, 0.6, 0.26749883),  # RX(0.10) -> RX(0.5) -> RX(0.6) -> RX(0.10)
            (-1, 0.5, 0.6, 0.77468805),  # RX(0.10) -> RZ(0.6) -> RX(0.5) -> RX(0.10)
        ],
    )
    def test_circuit_branches(self, pred, arg1, arg2, expected):
        """Test circuit with true, false, and elif branches."""
        result = circuit_branches(pred, arg1, arg2)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "pred, arg1, arg2, expected",
        [
            (1, 0.5, 0.6, 0.43910855),  # RX(0.10) -> RY(0.5) -> RY(0.6) -> RX(0.10)
            (0, 0.5, 0.6, 0.98551243),  # RX(0.10) -> RZ(0.6) -> RX(0.5) -> RX(0.10)
        ],
    )
    def test_circuit_with_returned_operator(self, pred, arg1, arg2, expected):
        """Test circuit with returned operators in the branches."""
        result = circuit_with_returned_operator(pred, arg1, arg2)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "tmp_pred, tmp_arg, expected",
        [
            (1, 0.5, 0.54030231),  # RX(0.5) -> RX(0.5)
            (-1, 0.5, 0.98006658),  # RY(0.1) -> RY(0.1)
        ],
    )
    def test_circuit_multiple_cond(self, tmp_pred, tmp_arg, expected):
        """Test circuit with returned operators in the branches."""
        result = circuit_multiple_cond(tmp_pred, tmp_arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "pred, arg, expected",
        [
            (1, 0.5, 0.87758256),  # RX(0.5)
            (-1, 0.5, 0.0707372),  # RX(0.7) -> RX(0.8)
            (0, 0.5, -0.9899925),  # RX(0.9) -> RX(1.0) -> RX(1.1)
        ],
    )
    def test_circuit_consts(self, pred, arg, expected):
        """Test circuit with jaxpr constants."""
        result = circuit_with_consts(pred, arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"
