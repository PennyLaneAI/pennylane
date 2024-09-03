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

# pylint: disable=redefined-outer-name, too-many-arguments
# pylint: disable=no-self-use

import numpy as np
import pytest

import pennylane as qml
from pennylane.ops.op_math.condition import CondCallable, ConditionalTransformError

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")

# must be below jax importorskip
from pennylane.capture.primitives import cond_prim  # pylint: disable=wrong-import-position


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


@pytest.mark.parametrize("decorator", [True, False])
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
    def test_cond_true_elifs_false(self, testing_functions, selector, arg, expected, decorator):
        """Test the conditional with true, elifs, and false branches."""
        true_fn, false_fn, elif_fn1, elif_fn2, elif_fn3, elif_fn4 = testing_functions

        def test_func(pred):
            if decorator:
                conditional = qml.cond(pred > 0)(true_fn)
                conditional.else_if(pred == -1)(elif_fn1)
                conditional.else_if(pred == -2)(elif_fn2)
                conditional.else_if(pred == -3)(elif_fn3)
                conditional.else_if(pred == -4)(elif_fn4)
                conditional.otherwise(false_fn)
                return conditional

            return qml.cond(
                pred > 0,
                true_fn,
                false_fn,
                elifs=(
                    (pred == -1, elif_fn1),
                    (pred == -2, elif_fn2),
                    (pred == -3, elif_fn3),
                    (pred == -4, elif_fn4),
                ),
            )

        result = test_func(selector)(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(test_func(selector))(arg)
        assert jaxpr.eqns[0].primitive == cond_prim
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    # Note that this would fail by running the abstract evaluation of the jaxpr
    # because the false branch must be provided if the true branch returns any variables.
    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, 10, 20),  # True condition
            (-1, 10, 9),  # Elif condition 1
            (-2, 10, 8),  # Elif condition 2
            (-3, 10, ()),  # No condition met
        ],
    )
    def test_cond_true_elifs(self, testing_functions, selector, arg, expected, decorator):
        """Test the conditional with true and elifs branches."""
        true_fn, _, elif_fn1, elif_fn2, _, _ = testing_functions

        def test_func(pred):
            if decorator:
                conditional = qml.cond(pred > 0)(true_fn)
                conditional.else_if(pred == -1)(elif_fn1)
                conditional.else_if(pred == -2)(elif_fn2)
                return conditional

            return qml.cond(
                pred > 0,
                true_fn,
                elifs=(
                    (pred == -1, elif_fn1),
                    (pred == -2, elif_fn2),
                ),
            )

        result = test_func(selector)(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, 10, 20),
            (0, 10, 30),
        ],
    )
    def test_cond_true_false(self, testing_functions, selector, arg, expected, decorator):
        """Test the conditional with true and false branches."""
        true_fn, false_fn, _, _, _, _ = testing_functions

        def test_func(pred):
            if decorator:
                conditional = qml.cond(pred > 0)(true_fn)
                conditional.otherwise(false_fn)
                return conditional

            return qml.cond(
                pred > 0,
                true_fn,
                false_fn,
            )

        result = test_func(selector)(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(test_func(selector))(arg)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    # Note that this would fail by running the abstract evaluation of the jaxpr
    # because the false branch must be provided if the true branch returns any variables.
    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, 10, 20),
            (0, 10, ()),
        ],
    )
    def test_cond_true(self, testing_functions, selector, arg, expected, decorator):
        """Test the conditional with only the true branch."""
        true_fn, _, _, _, _, _ = testing_functions

        def test_func(pred):
            if decorator:
                conditional = qml.cond(pred > 0)(true_fn)
                return conditional

            return qml.cond(
                pred > 0,
                true_fn,
            )

        result = test_func(selector)(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    @pytest.mark.parametrize(
        "selector, arg, expected",
        [
            (1, jax.numpy.array([2, 3]), 12),
            (0, jax.numpy.array([2, 3]), 15),
        ],
    )
    def test_cond_with_jax_array(self, selector, arg, expected, decorator):
        """Test the conditional with array arguments."""

        def true_fn(jax_array):
            return jax_array[0] * jax_array[1] * 2.0

        def false_fn(jax_array):
            return jax_array[0] * jax_array[1] * 2.5

        def test_func(pred):
            if decorator:
                conditional = qml.cond(pred > 0)(true_fn)
                conditional.otherwise(false_fn)
                return conditional

            return qml.cond(
                pred > 0,
                true_fn,
                false_fn,
            )

        result = test_func(selector)(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(test_func(selector))(arg)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    def test_mcm_return_error(self, decorator):
        """Test that an error is raised if executing a quantum function that uses qml.cond with
        mid-circuit measurement predicates where the conditional functions return something."""

        def true_fn(arg):
            qml.RX(arg, 0)
            return 0

        def false_fn(arg):
            qml.RY(arg, 0)
            return 1

        def f(x):
            m1 = qml.measure(0)
            if decorator:
                conditional = qml.cond(m1)(true_fn)
                conditional.otherwise(false_fn)
                conditional(x)
            else:
                qml.cond(m1, true_fn, false_fn)(x)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.23)
        with pytest.raises(
            ConditionalTransformError, match="Only quantum functions without return values"
        ):
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.23)

    def test_mcm_mixed_conds_error(self, decorator):
        """Test that an error is raised if executing a quantum function that uses qml.cond with
        a combination of mid-circuit measurement and other predicates."""

        def true_fn(arg):
            qml.RX(arg, 0)

        def elif_fn(arg):
            qml.RZ(arg, 0)

        def false_fn(arg):
            qml.RY(arg, 0)

        def f(x):
            m1 = qml.measure(0)
            if decorator:
                conditional = qml.cond(m1)(true_fn)
                conditional.else_if(x > 1.5)(elif_fn)
                conditional.otherwise(false_fn)
                conditional(x)
            else:
                qml.cond(m1, true_fn, false_fn, elifs=(x > 1.5, elif_fn))(x)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.23)
        with pytest.raises(
            ConditionalTransformError,
            match="Cannot use qml.cond with a combination of mid-circuit measurements",
        ):
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.23)


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
            jax.make_jaxpr(CondCallable(True, true_fn, false_fn))(jax.numpy.array(1))

    def test_validate_number_of_output_variables(self):
        """Test mismatch in number of output variables."""

        def true_fn(x):
            return x + 1, x + 2

        def false_fn(x):
            return x + 1

        with pytest.raises(ValueError, match=r"Mismatch in number of output variables"):
            jax.make_jaxpr(CondCallable(True, true_fn, false_fn))(jax.numpy.array(1))

    def test_validate_output_variable_types(self):
        """Test mismatch in output variable types."""

        def true_fn(x):
            return x + 1, x + 2

        def false_fn(x):
            return x + 1, x + 2.0

        with pytest.raises(ValueError, match=r"Mismatch in output abstract values"):
            jax.make_jaxpr(CondCallable(True, true_fn, false_fn))(jax.numpy.array(1))

    def test_validate_no_false_branch_with_return(self):
        """Test no false branch provided with return variables."""

        def true_fn(x):
            return x + 1, x + 2

        with pytest.raises(
            ValueError,
            match=r"The false branch must be provided if the true branch returns any variables",
        ):
            jax.make_jaxpr(CondCallable(True, true_fn))(jax.numpy.array(1))

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
            jax.make_jaxpr(CondCallable(True, true_fn, elifs=[(True, elif_fn)]))(jax.numpy.array(1))

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
                CondCallable(True, true_fn, false_fn, [(True, elif_fn1), (False, elif_fn2)])
            )(jax.numpy.array(1))

        with pytest.raises(
            ValueError, match=r"Mismatch in number of output variables in elif branch #0"
        ):
            jax.make_jaxpr(CondCallable(True, true_fn, false_fn, elifs=[(True, elif_fn3)]))(
                jax.numpy.array(1)
            )


dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit(pred):
    """Quantum circuit with only a true branch."""

    def true_fn():
        qml.RX(0.1, wires=0)

    qml.cond(pred > 0, true_fn)()
    return qml.expval(qml.Z(wires=0))


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
    return qml.expval(qml.Z(wires=0))


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
    return qml.expval(qml.Z(wires=0))


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

        args = [pred]
        jaxpr = jax.make_jaxpr(circuit)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

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

        args = [pred, arg1, arg2]
        jaxpr = jax.make_jaxpr(circuit_branches)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

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

        args = [pred, arg1, arg2]
        jaxpr = jax.make_jaxpr(circuit_with_returned_operator)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

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

        args = [tmp_pred, tmp_arg]
        jaxpr = jax.make_jaxpr(circuit_multiple_cond)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

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

        args = [pred, arg]
        jaxpr = jax.make_jaxpr(circuit_with_consts)(*args)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("shots", [None, 20])
    def test_mcm_predicate_execution(self, reset, postselect, shots):
        """Test that QNodes executed with mid-circuit measurement predicates for
        qml.cond give correct results."""
        device = qml.device("default.qubit", wires=3, shots=shots, seed=jax.random.PRNGKey(1234))

        def true_fn(arg):
            qml.RX(arg, 0)

        def false_fn(arg):
            qml.RY(3 * arg, 0)

        @qml.qnode(device)
        def f(x, y):
            qml.RX(x, 0)
            m = qml.measure(0, reset=reset, postselect=postselect)

            qml.cond(m, true_fn, false_fn)(y)
            return qml.expval(qml.Z(0))

        params = [2.5, 4.0]
        res = f(*params)
        qml.capture.disable()
        expected = f(*params)

        assert np.allclose(res, expected), f"Expected {expected}, but got {res}"

    @pytest.mark.parametrize("shots", [None, 100])
    @pytest.mark.parametrize(
        "params, expected",
        # The parameters used here will essentially apply a PauliX just before mid-circuit
        # measurements, each of which will trigger a different conditional block. Each
        # conditional block prepares basis states in different bases, so the expectation value
        # for the measured observables will vary accordingly.
        [
            ([np.pi, 0, 0], (1, 1 / np.sqrt(2), 0, 1 / np.sqrt(2))),  # true_fn, Hadamard basis
            ([0, np.pi, 0], (1 / np.sqrt(2), 1, 0, 0)),  # elif_fn1, PauliX basis
            ([0, 0, np.pi], (0, 0, 1, 0)),  # elif_fn2, PauliY basis
            ([0, 0, 0, 0], (1 / np.sqrt(2), 0, 0, 1)),  # false_fn, PauliZ basis
        ],
    )
    def test_mcm_predicate_execution_with_elifs(self, params, expected, shots, tol):
        """Test that QNodes executed with mid-circuit measurement predicates for
        qml.cond give correct results when there are also elifs present."""
        # pylint: disable=expression-not-assigned
        device = qml.device("default.qubit", wires=5, shots=shots, seed=jax.random.PRNGKey(10))

        def true_fn():
            # Adjoint Hadamard diagonalizing gates to get Hadamard basis state
            [qml.adjoint(op) for op in qml.Hadamard.compute_diagonalizing_gates(0)[::-1]]

        def elif_fn1():
            # Adjoint PauliX diagonalizing gates to get X basis state
            [qml.adjoint(op) for op in qml.X.compute_diagonalizing_gates(0)[::-1]]

        def elif_fn2():
            # Adjoint PauliY diagonalizing gates to get Y basis state
            [qml.adjoint(op) for op in qml.Y.compute_diagonalizing_gates(0)[::-1]]

        def false_fn():
            # Adjoint PauliZ diagonalizing gates to get Z basis state
            return

        @qml.qnode(device)
        def f(*x):
            qml.RX(x[0], 0)
            m1 = qml.measure(0, reset=True)
            qml.RX(x[1], 0)
            m2 = qml.measure(0, reset=True)
            qml.RX(x[2], 0)
            m3 = qml.measure(0, reset=True)

            qml.cond(m1, true_fn, false_fn, elifs=((m2, elif_fn1), (m3, elif_fn2)))()
            return (
                qml.expval(qml.Hadamard(0)),
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(0)),
                qml.expval(qml.Z(0)),
            )

        res = f(*params)
        atol = tol if shots is None else 0.1

        assert np.allclose(res, expected, atol=atol, rtol=0), f"Expected {expected}, but got {res}"

    @pytest.mark.parametrize("upper_bound, arg", [(3, [0.1, 0.3, 0.5]), (2, [2, 7, 12])])
    def test_nested_cond_for_while_loop(self, upper_bound, arg):
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
                    qml.cond(m_2, true_fn=true_fn, false_fn=false_fn, elifs=((m_1, elif_fn)))(
                        arg[0]
                    )
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


class TestPytree:
    """Test pytree support for cond."""

    def test_pytree_input_output(self):
        """Test that cond can handle pytree inputs and outputs."""

        def f(x):
            return {"val": x["1"]}

        def g(x):
            return {"val": x["2"]}

        def h(x):
            return {"val": x["h"]}

        res_true = qml.cond(True, f, false_fn=g, elifs=(False, h))({"1": 1, "2": 2, "h": 3})
        assert res_true == {"val": 1}

        res_elif = qml.cond(False, f, false_fn=g, elifs=(True, h))({"1": 1, "2": 2, "h": 3})
        assert res_elif == {"val": 3}

        res_false = qml.cond(False, f, false_fn=g, elifs=(False, h))({"1": 1, "2": 2, "h": 3})
        assert res_false == {"val": 2}

    def test_pytree_measurment_value(self):
        """Test that pytree args can be used when the condition is on a measurement value."""

        def g(x):
            qml.RX(x["x"], x["wire"])

        def f(x):
            m0 = qml.measure(0)
            qml.cond(m0, g)(x)

        with qml.queuing.AnnotatedQueue() as q:
            f({"x": 0.5, "wire": 0})

        assert len(q) == 2
        assert isinstance(q.queue[0], qml.measurements.MidMeasureMP)
        assert isinstance(q.queue[1], qml.ops.Conditional)
        qml.assert_equal(q.queue[1].base, qml.RX(0.5, 0))
