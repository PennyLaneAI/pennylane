# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTests for the AutoGraph source-to-source transformation feature for
converting if/else statements to qml.cond."""

# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports

import numpy as np
import pytest

import pennylane as qml
from pennylane import cond, measure

pytestmark = pytest.mark.capture

jax = pytest.importorskip("jax")

# must be below jax importorskip
from jax.core import eval_jaxpr

from pennylane.capture.autograph.transformer import TRANSFORMER, run_autograph
from pennylane.exceptions import AutoGraphError

check_cache = TRANSFORMER.has_cache


class TestConditionals:
    """Test that the autograph transformations produce correct results on conditionals."""

    def test_simple_cond(self):
        """Test basic function with conditional."""

        def circuit(n):
            if n > 4:
                res = n**2
            else:
                res = n

            return res

        # can't convert to jaxpr without autorgraph
        with pytest.raises(
            jax.errors.TracerBoolConversionError,
            match="Attempted boolean conversion of traced array",
        ):
            jax.make_jaxpr(circuit)(1)

        # with autograph we can convert to jaxpr
        circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(circuit)(0)
        assert "cond" in str(jaxpr)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)

        # evaluating the jaxpr gives expected results
        assert res(0) == [0]
        assert res(1) == [1]
        assert res(2) == [2]
        assert res(3) == [3]
        assert res(4) == [4]
        assert res(5) == [25]
        assert res(6) == [36]

    def test_cond_one_else_if(self):
        """Test a cond with one else_if branch"""

        def circuit(x):
            if x > 2.7:
                res = x * 4
            elif x > 1.4:
                res = x * 2
            else:
                res = x

            return res

        ag_circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert "cond" in str(jaxpr)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        assert res(4) == 16
        assert res(2) == 4
        assert res(1) == 1

    def test_cond_many_else_if(self):
        """Test a cond with multiple else_if branches"""

        def circuit(x):
            if x > 4.8:
                res = x * 8
            elif x > 2.7:
                res = x * 4
            elif x > 1.4:
                res = x * 2
            else:
                res = x

            return res

        ag_circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert "cond" in str(jaxpr)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        assert res(5) == 40
        assert res(3) == 12
        assert res(2) == 4
        assert res(-3) == -3

    def test_qubit_manipulation_cond(self):
        """Test conditional with quantum operation."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circuit(x):
            if x > 4:
                qml.PauliX(wires=0)

            m = measure(wires=0)

            return qml.expval(m)

        ag_circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert "cond" in str(jaxpr)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        # pylint: disable=singleton-comparison
        assert res(3) == 0
        assert res(6) == 1

    def test_nested_cond(self):
        """Test that a nested conditional is converted as expected"""

        def inner(x):
            if x > 3:
                return x**2
            return x**3

        def fn(x: int):
            return inner(x)

        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.make_jaxpr(fn)(4)

        ag_fn = run_autograph(fn)
        jaxpr = jax.make_jaxpr(ag_fn)(0)
        assert "cond" in str(jaxpr)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        # pylint: disable=singleton-comparison
        assert res(5) == 25
        assert res(2) == 8

    def test_multiple_return(self):
        """Test return statements from different branches of an if/else statement
        with autograph."""

        # pylint: disable=no-else-return

        def f(x: int):
            if x > 0:
                return 25  # converted to cond_fn
            else:
                return 60  # converted to else_fn

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert "cond" in str(jaxpr)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        assert res(1) == 25
        assert res(0) == 60

    def test_multiple_return_early(self):
        """Test that returning early is possible, and that the final return outside
        the conditional works as expected."""

        def f(x: float):

            if x:
                return x  # converted to cond_fn with no else_fn

            x = x + 2
            return x

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        # returning early is an option, and code after the return is not executed
        assert res(1) == 1

        # if an early return isn't hit, the code between the early return and the
        # final return is executed
        assert res(0) == 2

    def test_cond_decorator(self):
        """Test if Autograph works when applied to a function explicitly decorated with cond"""

        def f(n):

            @cond(n > 4)
            def cond_fn():
                return n**2

            # pylint: disable=unused-variable
            @cond_fn.otherwise
            def else_fn():
                return n

            return cond_fn()

        ag_fn = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_fn)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 6)[0] == 36
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)[0] == 2

    def test_branch_return_mismatch(self):
        """Test that an exception is raised when the true branch defines a value without an else
        branch.
        """
        # pylint: disable=using-constant-test

        def circuit():
            if True:
                res = measure(wires=0)

            return qml.expval(res)  # pylint: disable=possibly-used-before-assignment

        with pytest.raises(
            AutoGraphError, match="Some branches did not define a value for variable 'res'"
        ):
            qml.capture.autograph.run_autograph(circuit)()

    def test_branch_multi_return_type_mismatch(self):
        """Test that an exception is raised when the return types of all branches do not match."""
        # pylint: disable=using-constant-test

        def circuit():
            if True:
                res = 1
            elif False:
                res = 0.0
            else:
                res = 2

            return res

        with pytest.raises(ValueError, match="Mismatch in output abstract values"):
            run_autograph(circuit)()

    def test_multiple_return_different_measurements(self):
        """Test that different measurements be used in the return in different branches, as
        they are all represented by the AbstractMeasurement class."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(switch: bool):
            if switch:
                return qml.expval(qml.PauliY(0))

            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        assert np.allclose(res(True), 0)
        assert np.allclose(res(False), 1)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
