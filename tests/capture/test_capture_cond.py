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

# pylint: disable=redefined-outer-name, too-many-arguments, too-many-positional-arguments
# pylint: disable=no-self-use

from functools import partial

import numpy as np
import pytest

import pennylane as qp
from pennylane.exceptions import ConditionalTransformError
from pennylane.ops.op_math.condition import CondCallable

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

# must be below jax importorskip
from pennylane.capture.primitives import cond_prim  # pylint: disable=wrong-import-position


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


def test_bad_predicate_shape():
    """Test that an error is raised if the predicate is not a scalar."""

    def f(pred):
        qp.cond(pred, qp.X, qp.Z)(0)

    with pytest.raises(ValueError, match="predicate must be a scalar"):
        jax.make_jaxpr(f)(np.array([0, 0]))


@pytest.mark.parametrize("decorator", [True, False])
class TestCond:
    """Tests for conditional functions using qp.cond."""

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
                conditional = qp.cond(pred > 0)(true_fn)
                conditional.else_if(pred == -1)(elif_fn1)
                conditional.else_if(pred == -2)(elif_fn2)
                conditional.else_if(pred == -3)(elif_fn3)
                conditional.else_if(pred == -4)(elif_fn4)
                conditional.otherwise(false_fn)
                return conditional

            return qp.cond(
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

        def f(pred, arg):
            return test_func(pred)(arg)

        jaxpr = jax.make_jaxpr(f)(selector, arg)
        # 0-4 greater than and equality.
        assert jaxpr.eqns[5].primitive == cond_prim
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, selector, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

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
                conditional = qp.cond(pred > 0)(true_fn)
                conditional.otherwise(false_fn)
                return conditional

            return qp.cond(
                pred > 0,
                true_fn,
                false_fn,
            )

        result = test_func(selector)(arg)
        assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

        jaxpr = jax.make_jaxpr(test_func(selector))(arg)
        res_ev_jxpr = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        assert np.allclose(res_ev_jxpr, expected), f"Expected {expected}, but got {res_ev_jxpr}"

    @pytest.mark.parametrize(
        "selector, arg",
        [
            (1, 10.0),
            (0, 10.0),
        ],
    )
    def test_gradient(self, testing_functions, selector, arg, decorator):
        """Test the gradient of the conditional."""
        from pennylane.capture.primitives import jacobian_prim

        true_fn, false_fn, _, _, _, _ = testing_functions

        def func(pred):
            if decorator:
                conditional = qp.cond(pred > 0)(true_fn)
                conditional.otherwise(false_fn)
                return conditional

            return qp.cond(
                pred > 0,
                true_fn,
                false_fn,
            )

        test_func = qp.grad(func(selector))

        jaxpr = jax.make_jaxpr(test_func)(arg)
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == jacobian_prim
        # broken on jax0.5.3
        # correct_func = jax.grad(func(selector))
        # assert np.allclose(correct_func(arg), expected)
        # assert np.allclose(test_func(arg), correct_func(arg))

        # manual_res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, arg)
        # assert np.allclose(manual_res, correct_func(arg))

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
                conditional = qp.cond(pred > 0)(true_fn)
                conditional.otherwise(false_fn)
                return conditional

            return qp.cond(
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
        """Test that an error is raised if executing a quantum function that uses qp.cond with
        mid-circuit measurement predicates where the conditional functions return something."""

        def true_fn(arg):
            qp.RX(arg, 0)
            return 0

        def false_fn(arg):
            qp.RY(arg, 0)
            return 1

        def f(x):
            m1 = qp.measure(0)
            if decorator:
                conditional = qp.cond(m1)(true_fn)
                conditional.otherwise(false_fn)
                conditional(x)
            else:
                qp.cond(m1, true_fn, false_fn)(x)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.23)
        with pytest.raises(
            ConditionalTransformError, match="Only quantum functions without return values"
        ):
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.23)

    def test_mcm_mixed_conds_error(self, decorator):
        """Test that an error is raised if executing a quantum function that uses qp.cond with
        a combination of mid-circuit measurement and other predicates."""

        def true_fn(arg):
            qp.RX(arg, 0)

        def elif_fn(arg):
            qp.RZ(arg, 0)

        def false_fn(arg):
            qp.RY(arg, 0)

        def f(x):
            m1 = qp.measure(0)
            if decorator:
                conditional = qp.cond(m1)(true_fn)
                conditional.else_if(x > 1.5)(elif_fn)
                conditional.otherwise(false_fn)
                conditional(x)
            else:
                qp.cond(m1, true_fn, false_fn, elifs=(x > 1.5, elif_fn))(x)
            return qp.expval(qp.Z(0))

        jaxpr = jax.make_jaxpr(f)(1.23)
        with pytest.raises(
            ConditionalTransformError,
            match="Cannot use qp.cond with a combination of mid-circuit measurements",
        ):
            _ = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 1.23)


def test_keyword_argument():
    """Test that keyword arguments are treated as traceable inputs."""

    def f(pred, x):

        @qp.cond(pred)
        def b(*, x):
            qp.RX(x, 0)

        @b.otherwise
        def _(*, x):
            qp.RZ(x, 0)

        return b(x=x)

    jaxpr = jax.make_jaxpr(f)(True, 0.5)
    assert jaxpr.eqns[0].primitive == cond_prim

    for j in jaxpr.eqns[0].params["jaxpr_branches"]:
        # x is an input, not a constvar
        assert len(j.constvars) == 0
        assert len(j.invars) == 1


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

        def f(pred, input):
            return CondCallable(pred, true_fn, false_fn)(input)

        with pytest.raises(expected_error, match=match):
            jax.make_jaxpr(f)(True, jax.numpy.array(1))

    def test_validate_number_of_output_variables(self):
        """Test mismatch in number of output variables."""

        def true_fn(x):
            return x + 1, x + 2

        def false_fn(x):
            return x + 1

        def f(pred, input):
            return CondCallable(pred, true_fn, false_fn)(input)

        with pytest.raises(ValueError, match=r"Mismatch in number of output variables"):
            jax.make_jaxpr(f)(True, jax.numpy.array(1))

    def test_validate_output_variable_types(self):
        """Test mismatch in output variable types."""

        def true_fn(x):
            return x + 1, x + 2

        def false_fn(x):
            return x + 1, x + 2.0

        def f(pred, input):
            return CondCallable(pred, true_fn, false_fn)(input)

        with pytest.raises(ValueError, match=r"Mismatch in output abstract values"):
            jax.make_jaxpr(f)(True, jax.numpy.array(1))

    def test_validate_no_false_branch_with_return(self):
        """Test no false branch provided with return variables."""

        def true_fn(x):
            return x + 1, x + 2

        def f(pred, input):
            return CondCallable(pred, true_fn)(input)

        with pytest.raises(
            ValueError,
            match=r"The false branch must be provided if the true branch returns any variables",
        ):
            jax.make_jaxpr(f)(True, jax.numpy.array(1))

    def test_validate_no_false_branch_with_return_2(self):
        """Test no false branch provided with return variables."""

        def true_fn(x):
            return x + 1, x + 2

        def elif_fn(x):
            return x + 1, x + 2

        def f(pred, input):
            return CondCallable(pred, true_fn, elifs=[(True, elif_fn)])(input)

        with pytest.raises(
            ValueError,
            match=r"The false branch must be provided if the true branch returns any variables",
        ):
            jax.make_jaxpr(f)(True, jax.numpy.array(1))

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

        def f(pred1, pred2, pred3, input):
            return CondCallable(pred1, true_fn, false_fn, [(pred2, elif_fn1), (pred3, elif_fn2)])(
                input
            )

        with pytest.raises(
            ValueError, match=r"Mismatch in output abstract values in elif branch #1"
        ):
            jax.make_jaxpr(f)(True, True, False, jax.numpy.array(1))

        def g(pred1, pred2, input):
            return CondCallable(pred1, true_fn, false_fn, elifs=[(pred2, elif_fn3)])(input)

        with pytest.raises(
            ValueError, match=r"Mismatch in number of output variables in elif branch #0"
        ):
            jax.make_jaxpr(g)(True, True, jax.numpy.array(1))

    @pytest.mark.parametrize("partial_operator", (True, False))
    def test_true_fn_operator_type_no_false_fn(self, partial_operator):
        """Test that the true_fn can be an operator type when there is no false function. Instead,
        the cond simply has no output."""

        def f(pred):
            fn = partial(qp.X) if partial_operator else qp.X
            qp.cond(pred, fn)(0)

        jaxpr = jax.make_jaxpr(f)(True)
        assert jaxpr.eqns[0].primitive == cond_prim
        assert len(jaxpr.eqns[0].outvars) == 0

        true_fn = jaxpr.eqns[0].params["jaxpr_branches"][0]
        assert len(true_fn.outvars) == 0
        assert true_fn.eqns[0].primitive == qp.X._primitive  # pylint: disable=protected-access

        false_fn = jaxpr.eqns[0].params["jaxpr_branches"][-1]
        assert len(false_fn.eqns) == 0
        assert len(false_fn.outvars) == 0


dev = qp.device("default.qubit", wires=3)


@qp.qnode(dev)
def circuit(pred):
    """Quantum circuit with only a true branch."""

    def true_fn():
        qp.RX(0.1, wires=0)

    qp.cond(pred > 0, true_fn)()
    return qp.expval(qp.Z(wires=0))


@qp.qnode(dev)
def circuit_branches(pred, arg1, arg2):
    """Quantum circuit with conditional branches."""

    qp.RX(0.10, wires=0)

    def true_fn(arg1, arg2):
        qp.RY(arg1, wires=0)
        qp.RX(arg2, wires=0)
        qp.RZ(arg1, wires=0)

    def false_fn(arg1, arg2):
        qp.RX(arg1, wires=0)
        qp.RX(arg2, wires=0)

    def elif_fn1(arg1, arg2):
        qp.RZ(arg2, wires=0)
        qp.RX(arg1, wires=0)

    qp.cond(pred > 0, true_fn, false_fn, elifs=(pred == -1, elif_fn1))(arg1, arg2)
    qp.RX(0.10, wires=0)
    return qp.expval(qp.Z(wires=0))


@qp.qnode(dev)
def circuit_with_returned_operator(pred, arg1, arg2):
    """Quantum circuit with conditional branches that return operators."""

    qp.RX(0.10, wires=0)

    def true_fn(arg1, arg2):
        qp.RY(arg1, wires=0)
        return 7, 4.6, qp.RY(arg2, wires=0), True

    def false_fn(arg1, arg2):
        qp.RZ(arg2, wires=0)
        return 2, 2.2, qp.RZ(arg1, wires=0), False

    qp.cond(pred > 0, true_fn, false_fn)(arg1, arg2)
    qp.RX(0.10, wires=0)
    return qp.expval(qp.Z(wires=0))


@qp.qnode(dev)
def circuit_multiple_cond(tmp_pred, tmp_arg):
    """Quantum circuit with multiple dynamic conditional branches."""

    dyn_pred_1 = tmp_pred > 0
    arg = tmp_arg

    def true_fn_1(arg):
        return True, qp.RX(arg, wires=0)

    # pylint: disable=unused-argument
    def false_fn_1(arg):
        return False, qp.RY(0.1, wires=0)

    def true_fn_2(arg):
        return qp.RX(arg, wires=0)

    # pylint: disable=unused-argument
    def false_fn_2(arg):
        return qp.RY(0.1, wires=0)

    dyn_pred_2, _ = qp.cond(dyn_pred_1, true_fn_1, false_fn_1, elifs=())(arg)
    qp.cond(dyn_pred_2, true_fn_2, false_fn_2, elifs=())(arg)
    return qp.expval(qp.Z(0))


@qp.qnode(dev)
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
        qp.RX(arg1, 0)

    def false_fn():
        qp.RX(arg2, 0)
        qp.RX(arg3, 0)

    def elif_fn1():
        qp.RX(arg4, 0)
        qp.RX(arg5, 0)
        qp.RX(arg6, 0)

    qp.cond(pred > 0, true_fn, false_fn, elifs=((pred == 0, elif_fn1),))()

    return qp.expval(qp.Z(0))


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

    @pytest.mark.xfail(strict=False)  # might pass if postselection equal to measurement
    @pytest.mark.local_salt(1)
    @pytest.mark.parametrize("reset", [True, False])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("shots", [None, 20])
    def test_mcm_predicate_execution(self, reset, postselect, shots, seed):
        """Test that QNodes executed with mid-circuit measurement predicates for
        qp.cond give correct results."""
        device = qp.device("default.qubit", wires=3, seed=jax.random.PRNGKey(seed))

        def true_fn(arg):
            qp.RX(arg, 0)

        def false_fn(arg):
            qp.RY(3 * arg, 0)

        @qp.set_shots(shots)
        @qp.qnode(device)
        def f(x, y):
            qp.RX(x, 0)
            m = qp.measure(0, reset=reset, postselect=postselect)

            qp.cond(m, true_fn, false_fn)(y)
            return qp.expval(qp.Z(0))

        params = [2.5, 4.0]
        res = f(*params)
        qp.capture.disable()
        expected = f(*params)

        assert np.allclose(res, expected), f"Expected {expected}, but got {res}"

    @pytest.mark.xfail(
        strict=False
    )  # currently using single branch statistics, sometimes gives good results
    @pytest.mark.parametrize("shots", [None, 300])
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
    def test_mcm_predicate_execution_with_elifs(self, params, expected, shots, tol, seed):
        """Test that QNodes executed with mid-circuit measurement predicates for
        qp.cond give correct results when there are also elifs present."""
        # pylint: disable=expression-not-assigned
        device = qp.device("default.qubit", wires=5, seed=jax.random.PRNGKey(seed))

        def true_fn():
            # Adjoint Hadamard diagonalizing gates to get Hadamard basis state
            [qp.adjoint(op) for op in qp.Hadamard.compute_diagonalizing_gates(0)[::-1]]

        def elif_fn1():
            # Adjoint PauliX diagonalizing gates to get X basis state
            [qp.adjoint(op) for op in qp.X.compute_diagonalizing_gates(0)[::-1]]

        def elif_fn2():
            # Adjoint PauliY diagonalizing gates to get Y basis state
            [qp.adjoint(op) for op in qp.Y.compute_diagonalizing_gates(0)[::-1]]

        def false_fn():
            # Adjoint PauliZ diagonalizing gates to get Z basis state
            return

        @qp.set_shots(shots)
        @qp.qnode(device)
        def f(*x):
            qp.RX(x[0], 0)
            m1 = qp.measure(0, reset=True)
            qp.RX(x[1], 0)
            m2 = qp.measure(0, reset=True)
            qp.RX(x[2], 0)
            m3 = qp.measure(0, reset=True)

            qp.cond(m1, true_fn, false_fn, elifs=((m2, elif_fn1), (m3, elif_fn2)))()
            return (
                qp.expval(qp.Hadamard(0)),
                qp.expval(qp.X(0)),
                qp.expval(qp.Y(0)),
                qp.expval(qp.Z(0)),
            )

        res = f(*params)
        atol = tol if shots is None else 0.1

        assert np.allclose(res, expected, atol=atol, rtol=0), f"Expected {expected}, but got {res}"

    @pytest.mark.xfail(strict=False)  # single-branch-statistics only sometimes gives good results
    @pytest.mark.parametrize("upper_bound, arg", [(3, [0.1, 0.3, 0.5]), (2, [2, 7, 12])])
    def test_nested_cond_for_while_loop(self, upper_bound, arg):
        """Test that a nested control flows are correctly captured into a jaxpr."""

        dev = qp.device("default.qubit", wires=3)

        # Control flow for qp.conds
        def true_fn(_):
            @qp.for_loop(0, upper_bound, 1)
            def loop_fn(i):
                qp.Hadamard(wires=i)

            loop_fn()

        def elif_fn(arg):
            qp.RY(arg**2, wires=[2])

        def false_fn(arg):
            qp.RY(-arg, wires=[2])

        @qp.qnode(dev)
        def circuit(upper_bound, arg):
            qp.RY(-np.pi / 2, wires=[2])
            m_0 = qp.measure(2)

            # NOTE: qp.cond(m_0, qp.RX)(arg[1], wires=1) doesn't work
            def rx_fn():
                qp.RX(arg[1], wires=1)

            qp.cond(m_0, rx_fn)()

            def ry_fn():
                qp.RY(arg[1] ** 3, wires=1)

            # nested for loops.
            # outer for loop updates x
            @qp.for_loop(0, upper_bound, 1)
            def loop_fn_returns(i, x):
                qp.RX(x, wires=i)
                m_1 = qp.measure(0)
                # NOTE: qp.cond(m_0, qp.RY)(arg[1], wires=1) doesn't work
                qp.cond(m_1, ry_fn)()

                # inner while loop
                @qp.while_loop(lambda j: j < upper_bound)
                def inner(j):
                    qp.RZ(j, wires=0)
                    qp.RY(x**2, wires=0)
                    m_2 = qp.measure(0)
                    qp.cond(m_2, true_fn=true_fn, false_fn=false_fn, elifs=((m_1, elif_fn)))(
                        arg[0]
                    )
                    return j + 1

                inner(i + 1)
                return x + 0.1

            loop_fn_returns(arg[2])

            return qp.expval(qp.Z(0))

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

        res_true = qp.cond(True, f, false_fn=g, elifs=(False, h))({"1": 1, "2": 2, "h": 3})
        assert res_true == {"val": 1}

        res_elif = qp.cond(False, f, false_fn=g, elifs=(True, h))({"1": 1, "2": 2, "h": 3})
        assert res_elif == {"val": 3}

        res_false = qp.cond(False, f, false_fn=g, elifs=(False, h))({"1": 1, "2": 2, "h": 3})
        assert res_false == {"val": 2}

    def test_pytree_measurment_value(self):
        """Test that pytree args can be used when the condition is on a measurement value."""

        def g(x):
            qp.RX(x["x"], x["wire"])

        def f(x):
            m0 = qp.measure(0)
            qp.cond(m0, g)(x)

        with qp.queuing.AnnotatedQueue() as q:
            f({"x": 0.5, "wire": 0})

        assert len(q) == 2
        assert isinstance(q.queue[0], qp.ops.MidMeasure)
        assert isinstance(q.queue[1], qp.ops.Conditional)
        qp.assert_equal(q.queue[1].base, qp.RX(0.5, 0))


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestDynamicShapeValidation:

    def test_different_outval_types(self):
        """Test an error is raised if the outvals have different types."""

        def true_fn():
            return qp.X(0)

        def false_fn():
            return jax.numpy.array(3)

        def f(val):
            return qp.cond(val, true_fn, false_fn=false_fn)()

        with pytest.raises(ValueError, match="Mismatch in output abstract values"):
            jax.make_jaxpr(f)(True)

    def test_different_dtype(self):
        """Test an error is raised in the outputs have different dtypes."""

        def true_fn(n):
            return jax.numpy.arange(n, dtype=int)

        def false_fn(n):
            return jax.numpy.arange(n, dtype=float)

        def f(val, n):
            return qp.cond(val, true_fn, false_fn=false_fn)(n)

        with pytest.raises(ValueError, match="Mismatch in output abstract values"):
            jax.make_jaxpr(f)(True, 3)

    def test_one_dynamic_shape_other_not(self):
        """Test that an error is raised if one dimension in abstract on one branch, but not on another."""

        def true_fn(n):  # pylint: disable=unused-argument
            return jax.numpy.ones((2, n))

        def false_fn(n):
            return jax.numpy.ones((n, 2))

        def f(val, n):
            return qp.cond(val, true_fn, false_fn=false_fn)(n)

        with pytest.raises(ValueError, match="Mismatch in output abstract values"):
            jax.make_jaxpr(f)(True, 3)

    def test_different_concrete_shapes(self):
        """Test that errors are still raised if they have different concrete shapes."""

        def true_fn():
            return jax.numpy.ones(3)

        def false_fn():
            return jax.numpy.ones(4)

        def f(val):
            return qp.cond(val, true_fn, false_fn=false_fn)()

        with pytest.raises(ValueError, match="Mismatch in output abstract values"):
            jax.make_jaxpr(f)(True)

    def test_different_sized_shapes(self):
        """Test an error is raised with different sized shapes."""

        def true_fn(n):
            return jax.numpy.ones((n, n))

        def false_fn(n):
            return jax.numpy.ones(n)

        def f(val, n):
            return qp.cond(val, true_fn, false_fn=false_fn)(n)

        with pytest.raises(ValueError, match="may be due to different sized shapes"):
            jax.make_jaxpr(f)(True, 4)


@pytest.mark.usefixtures("enable_disable_dynamic_shapes")
class TestDynamicShapes:

    def test_cond_no_returns(self):
        """Test that cond can have empty returns when dynamic shapes are enabled."""

        def rx(x, w):
            qp.RX(x, w)

        def ry(x, w):
            qp.RY(x, w)

        def f(condition):
            qp.cond(condition == 2, rx, ry)(0.5, 1)

        jaxpr = jax.make_jaxpr(f)(0)
        [op] = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts, 1).operations
        qp.assert_equal(op, qp.RY(0.5, 1))

    def test_cond_abstracted_axes(self):
        """Test cond can accept inputs with dynamic shapes."""

        def workflow(x, predicate):
            return qp.cond(predicate, jax.numpy.sum, false_fn=jax.numpy.prod)(x)

        jaxpr = jax.make_jaxpr(workflow, abstracted_axes=({0: "a"}, {}))(jax.numpy.arange(3), True)

        output_true = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 4, jax.numpy.arange(4), True)
        assert qp.math.allclose(output_true[0], 6)  # 0 + 1 + 2 + 3

        output_false = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2, jax.numpy.arange(2), False)
        assert qp.math.allclose(output_false[0], 0)  # 0 * 1

    def test_cond_dynamic_shape_output(self):
        """test that cond can return dynamic shapes."""

        def true_fn(n):
            return jax.numpy.arange(n, dtype=int)

        def false_fn(n):
            return jax.numpy.zeros(n**2, dtype=int)

        def f(val, n):
            return {"result": qp.cond(val, true_fn, false_fn=false_fn)(n)}

        jaxpr = jax.make_jaxpr(f)(True, 3)

        assert len(jaxpr.jaxpr.outvars) == 2
        assert jaxpr.jaxpr.outvars[1].aval.shape[0] is jaxpr.jaxpr.outvars[0]

        [a, b] = qp.capture.PlxprInterpreter().eval(jaxpr.jaxpr, jaxpr.consts, True, 4)

        assert a == 4
        assert qp.math.allclose(b, true_fn(4))

        [c, d] = qp.capture.PlxprInterpreter().eval(jaxpr.jaxpr, jaxpr.consts, False, 7)
        assert c == 49
        assert qp.math.allclose(d, false_fn(7))

        res = f(True, 6)  # slicing out the shape variable
        assert qp.math.allclose(res["result"], jax.numpy.arange(6))

    def test_return_operators_with_dynamic_enabled(self):
        """Test that we can return operators when dynamic shapes are enabled."""

        def f(val, w):
            return qp.cond(val, qp.X, false_fn=qp.Y)(w)

        x_op = f(True, 0)
        qp.assert_equal(x_op, qp.X(0))
        y_op = f(False, 3)
        qp.assert_equal(y_op, qp.Y(3))

        jaxpr = jax.make_jaxpr(f)(False, 3)
        [x_op2] = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, True, 3)
        qp.assert_equal(x_op2, qp.X(3))

    def test_cond_dynamic_array_creation(self):
        """Test that arrays with dynamic shapes can be created within branches."""

        def true_fn(i):
            return jax.numpy.sum(jax.numpy.ones(i), dtype=int)

        def false_fn(i):
            return jax.numpy.sum(jax.numpy.arange(i), dtype=int)

        def f(condition, i):
            return qp.cond(condition, true_fn, false_fn)(i)

        jaxpr = jax.make_jaxpr(f)(True, 2)
        [res_true] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, True, 4)
        assert qp.math.allclose(res_true, 4)

        [res_false] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, False, 5)
        assert qp.math.allclose(res_false, 10)  # 0 + 1 + 2 + 3 + 4

    def test_dynamic_shape_matches_arg(self):
        """Test that cond can handle dynamic shapes where the dimension matches an earlier arg."""

        def t(i, x):
            return qp.RX(x, i)

        def f(i, x):
            return qp.RY(x, i)

        def w(val, i):
            return qp.cond(val, t, f)(i, jax.numpy.arange(i))

        jaxpr = jax.make_jaxpr(w)(True, 3)

        [res_true] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, True, 2)

        expected = qp.RX(jax.numpy.arange(2), 2)
        qp.assert_equal(res_true, expected)

        [res_false] = qp.capture.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, False, 3)
        expected_false = qp.RY(jax.numpy.arange(3), 3)
        qp.assert_equal(res_false, expected_false)
