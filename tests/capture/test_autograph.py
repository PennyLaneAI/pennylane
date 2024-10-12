# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTests for the AutoGraph source-to-source transformation feature."""

import traceback
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.core import eval_jaxpr

# from catalyst import debug, qjit, vmap
from jax.errors import TracerBoolConversionError
from numpy.testing import assert_allclose

import pennylane as qml
from pennylane import cond, for_loop, grad, jacobian, jvp, measure, vjp, while_loop
from pennylane.capture.autograph.ag_primitives import PRange
from pennylane.capture.autograph.transformer import (
    TRANSFORMER,
    autograph_source,
    disable_autograph,
    run_autograph,
)
from pennylane.capture.autograph.utils import AutoGraphError, CompileError, dummy_func

check_cache = TRANSFORMER.has_cache

# pylint: disable=import-outside-toplevel
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-lines


vmap = lambda x: None


@pytest.fixture
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


@pytest.fixture
def autograph_strict_conversion():
    qml.capture.autograph.autograph_strict_conversion = True
    yield
    qml.capture.autograph.autograph_strict_conversion = False


@pytest.fixture
def ignore_fallbacks():
    qml.capture.autograph.autograph_ignore_fallbacks = True
    yield
    qml.capture.autograph.autograph_strict_conversion = False


class Failing:
    """Test class that emulates failures in user-code"""

    triggered = defaultdict(bool)

    def __init__(self, ref, label: str = "default"):
        self.label = label
        self.ref = ref

    @property
    def val(self):
        """Get a reference to a variable or fail if programmed so."""
        # pylint: disable=broad-exception-raised
        if not Failing.triggered[self.label]:
            Failing.triggered[self.label] = True
            raise Exception(f"Emulated failure with label {self.label}")
        return self.ref


class TestSourceCodeInfo:
    """Unit tests for exception utilities that retrieves traceback information for the original
    source code."""

    def test_non_converted_function(self):
        """Test the robustness of traceback conversion on a non-converted function."""
        # from catalyst.autograph.ag_primitives import get_source_code_info
        from pennylane.capture.autograph.ag_primitives import (  # why not top-level?
            get_source_code_info,
        )

        try:
            result = ""
            raise RuntimeError("Test failure")
        except RuntimeError as e:
            result = get_source_code_info(traceback.extract_tb(e.__traceback__, limit=1)[0])

        assert result.split("\n")[1] == '    raise RuntimeError("Test failure")'

    def test_qjit(self):
        """Test source info retrieval for a qjit function."""

        def main():
            for _ in range(5):
                raise RuntimeError("Test failure")
            return 0

        with pytest.warns(
            UserWarning,
            match=(
                f'  File "{__file__}", line [0-9]+, in {main.__name__}\n'
                r"    for _ in range\(5\):"
            ),
        ):
            try:
                qjit(autograph=True)(main)
            except RuntimeError as e:
                assert e.args == ("Test failure",)

    def test_qnode(self):
        """Test source info retrieval for a qnode function."""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def main():
            for _ in range(5):
                raise RuntimeError("Test failure")
            return 0

        with pytest.warns(
            UserWarning,
            match=(
                f'  File "{__file__}", line [0-9]+, in {main.__name__}\n'
                r"    for _ in range\(5\):"
            ),
        ):
            try:
                qjit(autograph=True)(main)
            except RuntimeError as e:
                assert e.args == ("Test failure",)

    def test_func(self):
        """Test source info retrieval for a nested function."""

        def inner():
            for _ in range(5):
                raise RuntimeError("Test failure")

        def main():
            inner()
            return 0

        with pytest.warns(
            UserWarning,
            match=(
                f'  File "{__file__}", line [0-9]+, in {inner.__name__}\n'
                r"    for _ in range\(5\):"
            ),
        ):
            try:
                qjit(autograph=True)(main)
            except RuntimeError as e:
                assert e.args == ("Test failure",)


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestIntegration:
    """Test that the autograph transformations trigger correctly in different settings."""

    def test_unsupported_object(self):
        """Check the error produced when attempting to convert an unsupported object (neither of
        QNode, function, method or callable)."""

        class FN:
            """Test object."""

            __name__ = "unknown"

        fn = FN()

        with pytest.raises(AutoGraphError, match="Unsupported object for transformation"):
            run_autograph(fn)

    def test_callable_object(self):
        """Test qjit applied to a callable object."""

        class FN:
            """Test object."""

            __name__ = "unknown"

            def __call__(self, x):
                return x**2

        fn = FN()

        assert run_autograph(fn)(3) == 9

    def test_lambda(self):
        """Test autograph on a lambda function."""

        fn = lambda x: x**2
        ag_fn = run_autograph(fn)

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert fn(4) == 16

    def test_classical_function(self):
        """Test autograph on a purely classical function."""

        def fn(x):
            return x**2

        ag_fn = run_autograph(fn)
        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert fn(4) == 16

    def test_nested_function(self):
        """Test autograph on nested classical functions."""

        def inner(x):
            return x**2

        def fn(x: int):
            return inner(x)

        ag_fn = run_autograph(fn)
        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert check_cache(inner)
        assert fn(4) == 16

    def test_qnode(self):
        """Test autograph on a QNode."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def fn(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_fn = run_autograph(fn)
        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert fn(np.pi) == -1

    def test_indirect_qnode(self):
        """Test autograph on a QNode called from within a classical function."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        def fn(x: float):
            return inner(x)

        ag_fn = run_autograph(fn)
        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert check_cache(inner.func)
        assert fn(np.pi) == -1

    def test_multiple_qnode(self):
        """Test autograph on multiple QNodes called from different classical functions."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner1(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner1(x) + inner2(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert check_cache(inner1.func)
        assert check_cache(inner2.func)
        assert fn(np.pi) == -2

    def test_nested_qjit(self):
        """Test autograph on a QJIT function called from within the compilation entry point."""

        @qjit
        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(fn.original_function)
        assert check_cache(inner.user_function.func)
        assert fn(np.pi) == -1

    def test_adjoint_wrapper(self):
        """Test conversion is happening succesfully on functions wrapped with 'adjoint'."""

        def inner(x):
            qml.RY(x, wires=0)

        @qjit(autograph=True)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def fn(x: float):
            qml.adjoint(inner)(x)
            return qml.probs()

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner)
        assert np.allclose(fn(np.pi), [0.0, 1.0])

    def test_ctrl_wrapper(self):
        """Test conversion is happening succesfully on functions wrapped with 'ctrl'."""

        def inner(x):
            qml.RY(x, wires=0)

        @qjit(autograph=True)
        @qml.qnode(qml.device("default.qubit", wires=2))
        def fn(x: float):
            qml.ctrl(inner, control=1)(x)
            return qml.probs()

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner)
        assert np.allclose(fn(np.pi), [1.0, 0.0, 0.0, 0.0])

    def test_grad_wrapper(self):
        """Test conversion is happening succesfully on functions wrapped with 'grad'."""

        def inner(x):
            return 2 * x

        @qjit(autograph=True)
        def fn(x: float):
            return grad(inner)(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner)
        assert fn(3) == 2.0

    def test_jacobian_wrapper(self):
        """Test conversion is happening succesfully on functions wrapped with 'jacobian'."""

        def inner(x):
            return 2 * x, x**2

        @qjit(autograph=True)
        def fn(x: float):
            return jacobian(inner)(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner)
        assert fn(3) == tuple([jax.numpy.array(2.0), jax.numpy.array(6.0)])

    def test_vjp_wrapper(self):
        """Test conversion is happening succesfully on functions wrapped with 'vjp'."""

        def inner(x):
            return 2 * x, x**2

        @qjit(autograph=True)
        def fn(x: float):
            return vjp(inner, (x,), (1.0, 1.0))

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner)
        assert np.allclose(fn(3)[0], tuple([jnp.array(6.0), jnp.array(9.0)]))
        assert np.allclose(fn(3)[1], jnp.array(8.0))

    def test_jvp_wrapper(self):
        """Test conversion is happening succesfully on functions wrapped with 'jvp'."""

        def inner(x):
            return 2 * x, x**2

        @qjit(autograph=True)
        def fn(x: float):
            return jvp(inner, (x,), (1.0,))

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner)

        assert np.allclose(fn(3)[0], tuple([jnp.array(6.0), jnp.array(9.0)]))
        assert np.allclose(fn(3)[1], tuple([jnp.array(2.0), jnp.array(6.0)]))

    def test_tape_transform(self):
        """Test if tape transform is applied when autograph is on."""

        dev = dev = qml.device("default.qubit", wires=1)

        @qml.transform
        def my_quantum_transform(tape):
            raise NotImplementedError

        @qml.qjit(autograph=True)
        def f(x):
            @my_quantum_transform
            @qml.qnode(dev)
            def circuit(x):
                qml.RY(x, wires=0)
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

            return circuit(x)

        with pytest.raises(NotImplementedError):
            f(0.5)

    def test_mcm_one_shot(self):
        """Test if mcm one-shot miss transforms."""
        dev = qml.device("default.qubit", wires=5, shots=20)

        @qml.qjit(autograph=True)
        @qml.qnode(dev, mcm_method="one-shot", postselect_mode="hw-like")
        def func(x):
            qml.RX(x, wires=0)
            measure(0, postselect=1)
            return qml.sample(wires=0)

        # If transforms are missed, the output will be all ones.
        assert not np.all(func(0.9) == 1)


class TestCodePrinting:
    """Test that the transformed source code can be printed in different settings."""

    def test_unconverted(self):
        """Test printing on an unconverted function."""

        @qjit(autograph=False)
        def fn(x):
            return x**2

        with pytest.raises(AutoGraphError, match="function was not converted by AutoGraph"):
            autograph_source(fn)

    def test_lambda(self):
        """Test printing on a lambda function."""

        fn = lambda x: x**2
        qjit(autograph=True)(fn)

        assert autograph_source(fn)

    def test_classical_function(self):
        """Test printing on a purely classical function."""

        @qjit(autograph=True)
        def fn(x):
            return x**2

        assert autograph_source(fn)

    def test_nested_function(self):
        """Test printing on nested classical functions."""

        def inner(x):
            return x**2

        @qjit(autograph=True)
        def fn(x: int):
            return inner(x)

        assert autograph_source(fn)
        assert autograph_source(inner)

    def test_qnode(self):
        """Test printing on a QNode."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("default.qubit", wires=1))
        def fn(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert autograph_source(fn)

    def test_indirect_qnode(self):
        """Test printing on a QNode called from within a classical function."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert autograph_source(fn)
        assert autograph_source(inner)

    def test_multiple_qnode(self):
        """Test printing on multiple QNodes called from different classical functions."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner1(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner1(x) + inner2(x)

        assert autograph_source(fn)
        assert autograph_source(inner1)
        assert autograph_source(inner2)

    def test_nested_qjit(self):
        """Test printing on a QJIT function called from within the compilation entry point."""

        @qjit
        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert autograph_source(fn)
        assert autograph_source(inner)


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestConditionals:
    """Test that the autograph transformations produce correct results on conditionals.
    These tests are adapted from the test_conditionals.TestCond class of tests."""

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

        @qml.qnode(qml.device("default.qubit", wires=1))
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

    def test_branch_return_mismatch(self):
        """Test that an exception is raised when the true branch returns a value without an else
        branch.
        """
        # pylint: disable=using-constant-test

        def circuit():
            if True:
                res = measure(wires=0)

            return qml.expval(res)

        with pytest.raises(
            AutoGraphError, match="Some branches did not define a value for variable 'res'"
        ):
            qml.capture.autograph.run_autograph(circuit)()

    def test_branch_multi_return_mismatch(self):
        """Test that an exception is raised when the return types of all branches do not match."""
        # pylint: disable=using-constant-test

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circuit():
            if True:
                res = measure(wires=0)
            elif False:
                res = 0.0
            else:
                res = measure(wires=0)

            return res

        with pytest.raises(
            TypeError, match="Conditional requires consistent return types across all branches"
        ):
            run_autograph(circuit)

    def test_multiple_return(self):
        """Test return statements from different branches of an if/else statement
        with autograph."""

        def f(x: int):
            if x > 0:
                return 25
            else:
                return 60

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert "cond" in str(jaxpr)

        def res(x):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        assert res(1) == 25
        assert res(0) == 60

    def test_multiple_return_early(self, capfd):
        """Test that returning early is possible, and that the final return outside
        if the conditional works as expected."""

        def f(x: float):

            if x:
                return x

            x = x+2
            return x

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        def res(x): return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

        # returning early is an option, and code after the return is not executed
        assert res(1) == 1

        # if an early return isn't hit, the code between the early return and the
        # final return is executed
        assert res(0) == 2

    def test_multiple_return_mismatched_type(self):
        """Test that different observables cannot be used in the return in different branches."""

        # ToDo: I don't see this contraint - should I?

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(switch: bool):
            if switch:
                return qml.expval(qml.PauliY(0))

            return qml.expval(qml.PauliZ(0))

        with pytest.raises(TypeError, match="requires a consistent return structure"):
            ag_circuit = run_autograph(f)
            jaxpr = jax.make_jaxpr(ag_circuit)(0)

            def res(x):
                return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)[0]

            res(1)


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestForLoops:
    """Test that the autograph transformations produce correct results on for loops."""

    def test_python_range_fallback(self):
        """Test that the custom CRange wrapper correctly falls back to Python."""

        # pylint: disable=protected-access

        pl_range = PRange(0, 5, 1)
        assert pl_range._py_range is None

        assert isinstance(pl_range.py_range, range)  # automatically instantiates the Python range
        assert isinstance(pl_range._py_range, range)
        assert pl_range[2] == 2

    def test_for_in_array(self):
        """Test for loop over JAX array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(params):
            for x in params:
                qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)([1.0, 2.0, 3.0])

        def res(params):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params)

        result = f(jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]))
        print(result)
        assert np.allclose(result, -jnp.sqrt(2) / 2)

    def test_for_in_array_unpack(self):
        """Test for loop over a 2D JAX array unpacking the inner dimension."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(params):
            for x1, x2 in params:
                qml.RY(x1, wires=0)
                qml.RY(x2, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(jnp.array([[0.0, 0.0], [0.0, 0.0]]))

        params = jnp.array([[0.0, 1 / 4 * jnp.pi], [2 / 4 * jnp.pi, jnp.pi]])
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params)

        assert np.allclose(result, jnp.sqrt(2) / 2)

    def test_for_in_numeric_list(self):
        """Test for loop over a Python list that is convertible to an array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for x in params:
                qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()

        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, -jnp.sqrt(2) / 2)

    def test_for_in_numeric_list_of_list(self):
        """Test for loop over a nested Python list that is convertible to an array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = [[0.0, 1 / 4 * jnp.pi], [2 / 4 * jnp.pi, jnp.pi]]
            for xx in params:
                for x in xx:
                    qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, jnp.sqrt(2) / 2)

    @pytest.mark.xfail(reason="relies on unimplemented fallback behaviour")
    def test_for_in_object_list(self):
        """Test for loop over a Python list that is *not* convertible to an array.
        The behaviour should fall back to standard Python."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = ["0", "1", "2"]
            for x in params:
                qml.RY(int(x) / 4 * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, -jnp.sqrt(2) / 2)

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_for_in_object_list_strict(self):
        """Check the error raised in strict mode when a for loop iterates over a Python list that
        is *not* convertible to an array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = ["0", "1", "2"]
            for x in params:
                qml.RY(int(x) / 4 * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(AutoGraphError, match="Could not convert the iteration target"):
            run_autograph(f)()

    def test_for_in_static_range(self):
        """Test for loop over a Python range with static bounds."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def f():
            for i in range(3):
                qml.Hadamard(i)
            return qml.probs()

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, [1 / 8] * 8)

    def test_for_in_static_range_indexing_array(self):
        """Test for loop over a Python range with static bounds that is used to index an array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
            for i in range(3):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, -jnp.sqrt(2) / 2)

    # With conversion always taking place, the user needs to be careful to manually wrap
    # objects accessed via loop iteration indices into arrays (see test case above).
    # The warning here is actionable.
    def test_for_in_static_range_indexing_numeric_list(self):
        """Test for loop over a Python range with static bounds that is used to index an
        array-compatible Python list. This should fall back to Python with a warning."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i in range(3):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            run_autograph(f)()

    # This case is slightly problematic because there is no way for the user to compile this for
    # loop correctly. Fallback to a Python loop is always necessary, and will result in a warning.
    # The warning here is not actionable.
    def test_for_in_static_range_indexing_object_list(self):
        """Test for loop over a Python range with static bounds that is used to index an
        array-incompatible Python list. This should fall back to Python with a warning."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = ["0", "1", "2"]
            for i in range(3):
                qml.RY(int(params[i]) / 4 * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            run_autograph(f)()

    def test_for_in_dynamic_range(self):
        """Test for loop over a Python range with dynamic bounds."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def f(n: int):
            for i in range(n):
                qml.Hadamard(i)
            return qml.probs()

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)

        assert np.allclose(result, [1 / 8] * 8)

    def test_for_in_dynamic_range_indexing_array(self):
        """Test for loop over a Python range with dynamic bounds that is used to index an array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(n: int):
            params = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
            for i in range(n):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)

        assert np.allclose(result, -jnp.sqrt(2) / 2)

    # This case will fail even without autograph conversion, since dynamic iteration bounds are not
    # allowed in Python ranges. Here, AutoGraph improves the situation by allowing this test case
    # with a slight modification of the user code (see test case above).
    # Raising the warning is vital here to notify the user that this use case is actually supported,
    # but requires a modification. Without it, the user may simply conclude it is unsupported.
    def test_for_in_dynamic_range_indexing_numeric_list(self):
        """Test for loop over a Python range with dynamic bounds that is used to index an
        array-compatible Python list. The fallback to Python will first raise a warning,
        then an error."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(n: int):
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i in range(n):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            with pytest.raises(jax.errors.TracerIntegerConversionError, match="__index__"):
                run_autograph(f)()

    # This use case is never possible, regardless of whether AutoGraph is used or not.
    def test_for_in_dynamic_range_indexing_object_list(self):
        """Test for loop over a Python range with dynamic bounds that is used to index an
        array-incompatible Python list. The fallback to Python will first raise a warning,
        then an error."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(n: int):
            params = ["0", "1", "2"]
            for i in range(n):
                qml.RY(int(params[i]) * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.warns(
            match=r"TracerIntegerConversionError:    The __index__\(\) method was called"
        ):
            with pytest.raises(jax.errors.TracerIntegerConversionError, match="__index__"):
                run_autograph(f)()

    def test_for_in_enumerate_array(self):
        """Test for loop over a Python enumeration on an array."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def f(params):
            for i, x in enumerate(params):
                qml.RY(x, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(jnp.array([0.0, 0.0, 0.0]))

        params = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params)

        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_enumerate_array_no_unpack(self):
        """Test for loop over a Python enumeration with delayed unpacking."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def f(params):
            for v in enumerate(params):
                qml.RY(v[1], wires=v[0])
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(jnp.array([0.0, 0.0, 0.0]))

        params = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params)

        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_enumerate_nested_unpack(self):
        """Test for loop over a Python enumeration with nested unpacking."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def f(params):
            for i, (x1, x2) in enumerate(params):
                qml.RY(x1, wires=i)
                qml.RY(x2, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))

        params = jnp.array(
            [[0.0, 1 / 4 * jnp.pi], [2 / 4 * jnp.pi, 3 / 4 * jnp.pi], [jnp.pi, 2 * jnp.pi]]
        )
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params)

        assert np.allclose(result, [jnp.sqrt(2) / 2, -jnp.sqrt(2) / 2, -1.0])

    def test_for_in_enumerate_start(self):
        """Test for loop over a Python enumeration with offset indices."""

        @qml.qnode(qml.device("default.qubit", wires=5))
        def f(params):
            for i, x in enumerate(params, start=2):
                qml.RY(x, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(5)]

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(jnp.array([0.0, 0.0, 0.0]))

        params = jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi])
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params)

        assert np.allclose(result, [1.0, 1.0, 1.0, jnp.sqrt(2) / 2, 0.0])

    def test_for_in_enumerate_numeric_list(self):
        """Test for loop over a Python enumeration on a list that is convertible to an array."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def f():
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i, x in enumerate(params):
                qml.RY(x, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    @pytest.mark.xfail(reason="relies on unimplemented fallback behaviour")
    def test_for_in_enumerate_object_list(self):
        """Test for loop over a Python enumeration on a list that is *not* convertible to an array.
        The behaviour should fall back to standard Python."""

        @qml.qnode(qml.device("default.qubit", wires=3))
        def f():
            params = ["0", "1", "2"]
            for i, x in enumerate(params):
                qml.RY(int(x) / 4 * jnp.pi, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, [1.0, jnp.sqrt(2) / 2, 0.0])

    @pytest.mark.xfail(reason="relies on unimplemented fallback behaviour")
    def test_for_in_other_iterable_object(self):
        """Test for loop over arbitrary iterable Python objects.
        The behaviour should fall back to standard Python."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = {"a": 0.0, "b": 1 / 4 * jnp.pi, "c": 2 / 4 * jnp.pi}
            for k, v in params.items():
                print(k)
                qml.RY(v, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

        assert np.allclose(result, -jnp.sqrt(2) / 2)

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_loop_carried_value(self):
        """Test a loop which updates a value each iteration."""

        def f1():
            acc = 0
            for x in [0, 4, 5]:
                acc = acc + x

            return acc

        ag_circuit = run_autograph(f1)
        jaxpr1 = jax.make_jaxpr(ag_circuit)()
        assert eval_jaxpr(jaxpr1.jaxpr, jaxpr1.consts)[0] == 9

        def f2(acc):
            for x in [0, 4, 5]:
                acc = acc + x

            return acc

        ag_circuit = run_autograph(f2)
        jaxpr2 = jax.make_jaxpr(ag_circuit)(0)
        assert eval_jaxpr(jaxpr2.jaxpr, jaxpr2.consts, 2)[0] == 11

        def f3():
            acc = 0
            for x in [0, 4, 5]:
                acc += x

            return acc

        ag_circuit = run_autograph(f3)
        jaxpr3 = jax.make_jaxpr(ag_circuit)()
        assert eval_jaxpr(jaxpr3.jaxpr, jaxpr3.consts)[0] == 9

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_iteration_element_access(self):
        """Test that access to the iteration index/elements is possible after the loop executed
        (assuming initialization)."""

        def f1(acc):
            x = 0
            for x in [0, 4, 5]:
                acc = acc + x
            ...  # use acc

            return x

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 5

        def f2(acc):
            i = 0
            l = jnp.array([0, 4, 5])
            for i in range(3):
                acc = acc + l[i]
            ...  # use acc

            return i

        ag_circuit = run_autograph(f2)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 2

        def f3(acc):
            i, x = 0, 0
            for i, x in enumerate([0, 4, 5]):
                acc = acc + x
            ...  # use acc

            return i, x

        ag_circuit = run_autograph(f3)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)

        assert np.allclose(result, [2, 5])

    @pytest.mark.usefixtures("autograph_strict_conversion")
    @pytest.mark.xfail(reason="currently unsupported, but we may find a way to do so in the future")
    def test_iteration_element_access_no_init(self):
        """Test that access to the iteration index/elements is possible after the loop executed
        even without prior initialization."""

        def f1(acc):
            for x in [0, 4, 5]:
                acc = acc + x
            ...  # use acc

            return x

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 5

        def f2(acc):
            l = jnp.array([0, 4, 5])
            for i in range(3):
                acc = acc + l[i]
            ...  # use acc

            return i

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 2

        def f3(acc):
            for i, x in enumerate([0, 4, 5]):
                acc = acc + x
            ...  # use acc

            return i, x

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == (2, 5)

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_temporary_loop_variable(self):
        """Test that temporary (local) variables can be initialized inside a loop."""

        def f1():
            acc = 0
            for x in [0, 4, 5]:
                c = 2
                acc = acc + c * x

            return acc

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 18

        def f2():
            acc = 2
            for x in [0, 4, 5]:
                c = x * 2
                acc = acc + c

            return acc

        ag_circuit = run_autograph(f2)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 20

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_uninitialized_variables(self):
        """Verify errors for (potentially) uninitialized loop variables."""

        def f1():
            for x in [0, 4, 5]:
                acc = acc + x

            return acc

        with pytest.raises(AutoGraphError, match="'acc' is potentially uninitialized"):
            run_autograph(f1)()

        def f2():
            acc = 0
            for x in [0, 4, 5]:
                acc = acc + x

            return x

        with pytest.raises(AutoGraphError, match="'x' is potentially uninitialized"):
            run_autograph(f2)()

        def f3():
            acc = 0
            for x in [0, 4, 5]:
                c = 2
                acc = acc + c * x

            return c

        with pytest.raises(AutoGraphError, match="'c' is potentially uninitialized"):
            run_autograph(f3)()

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_init_with_invalid_jax_type(self):
        """Test loop carried values initialized with an invalid JAX type."""

        def f():
            acc = 0
            x = ""
            for x in [0, 4, 5]:
                acc = acc + x

            return x

        with pytest.raises(AutoGraphError, match="'x' was initialized with type <class 'str'>"):
            run_autograph(f)()

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_init_with_mismatched_type(self):
        """Test loop carried values initialized with a mismatched type compared to the values used
        inside the loop."""

        def f():
            acc = 0
            x = 0.0
            for x in [0, 4, 5]:
                acc = acc + x

            return x

        with pytest.raises(AutoGraphError, match="'x' was initialized with the wrong type"):
            run_autograph(f)()

    # @pytest.mark.filterwarnings("error::UserWarning")
    # @pytest.mark.usefixtures("ignore_fallbacks")
    # def test_ignore_warnings(self):
    #     """Test the AutoGraph config flag properly silences warnings."""
    #
    #     def f():
    #         acc = 0
    #         data = [0, 4, 5]
    #         for i in range(3):
    #             acc = acc + data[i]
    #
    #         return acc
    #
    #     assert f() == 9


@pytest.mark.usefixtures("enable_disable_plxpr")
class TestWhileLoops:
    """Test that the autograph transformations produce correct results on while loops."""

    @pytest.mark.usefixtures("autograph_strict_conversion")
    @pytest.mark.parametrize(
        "init,inc,expected", [(0, 1, 3), (0.0, 1.0, 3.0), (0.0 + 0j, 1.0 + 0j, 3.0 + 0j)]
    )
    def test_whileloop_basic(self, init, inc, expected):
        """Test basic while-loop functionality"""

        def f(limit):
            i = init
            while i < limit:
                i += inc
            return i

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert "while_loop" in str(jaxpr)

        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, expected)[0]
        assert result == expected

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_whileloop_multiple_variables(self):
        """Test while-loop with a multiple state variables"""

        def f(param):
            a = 0
            b = 0
            while a < param:
                a += 1
                b += 1
            return b

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert "while_loop" in str(jaxpr)

        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)[0]
        assert result == 3

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_whileloop_qnode(self):
        """Test while-loop used with a qnode"""

        @qml.qnode(qml.device("default.qubit", wires=4))
        def f(p):
            w = int(0)
            while w < 4:
                qml.RY(p, wires=w)
                p *= 0.5
                w += 1
            return qml.probs()

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(0.0)
        assert "while_loop" in str(jaxpr)

        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.0**4)[0]
        expected = jnp.array(
            # fmt:off
            [
                0.00045727, 0.00110912, 0.0021832, 0.0052954,
                0.000613, 0.00148684, 0.00292669, 0.00709874,
                0.02114249, 0.0512815, 0.10094267, 0.24483834,
                0.02834256, 0.06874542, 0.13531871, 0.32821807,
            ]
            # fmt:on
        )
        assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_whileloop_temporary_variable(self, monkeypatch):
        """Test that temporary (local) variables can be initialized inside a while loop."""

        def f1():
            acc = 0
            while acc < 3:
                c = 2
                acc = acc + c

            return acc

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        assert "while" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 4

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_whileloop_forloop_interop(self):
        """Test for-loop co-existing with while loop."""

        def f1():
            acc = 0
            while acc < 5:
                acc = acc + 1
                for x in [1, 2, 3]:
                    acc += x
            return acc

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        assert "while" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 0 + 1 + sum([1, 2, 3])

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_whileloop_cond_interop(self):
        """Test for-loop co-existing with cond."""

        def f1():
            acc = 0
            while acc < 5:
                if acc < 2:
                    acc += 1
                else:
                    acc += 2
            return acc

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        assert "while" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == sum([1, 1, 2, 2])

    # @pytest.mark.xfail(reason="this won't run warning-free until we fix the resource warning issue")
    # @pytest.mark.filterwarnings("error")
    # def test_whileloop_no_warning(self, monkeypatch):
    #     """Test the absence of warnings if fallbacks are ignored."""
    #     monkeypatch.setattr("catalyst.autograph_ignore_fallbacks", True)
    #
    #     @qjit(autograph=True)
    #     def f():
    #         acc = 0
    #         while Failing(acc).val < 5:
    #             acc = acc + 1
    #         return acc
    #
    #     assert f() == 5

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_whileloop_exception(self):
        """Test for-loop error if strict-conversion is enabled."""

        def f1():
            acc = 0
            while acc < 5:
                raise RuntimeError("Test failure")
            return acc

        with pytest.raises(RuntimeError):
            run_autograph(f1)()

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_uninitialized_variables(self):
        """Verify errors for (potentially) uninitialized loop variables."""

        def f(pred: bool):
            while pred:
                x = 3

            return x

        with pytest.raises(AutoGraphError, match="'x' is potentially uninitialized"):
            run_autograph(f)(True)

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_init_with_invalid_jax_type(self):
        """Test loop carried values initialized with an invalid JAX type."""

        def f(pred: bool):
            x = ""

            while pred:
                x = 3

            return x

        with pytest.raises(AutoGraphError, match="'x' was initialized with type <class 'str'>"):
            run_autograph(f)(True)

    @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_init_with_mismatched_type(self):
        """Test loop carried values initialized with a mismatched type compared to the values used
        inside the loop."""

        def f(pred: bool):
            x = 0.0

            while pred:
                x = 3

            return x

        with pytest.raises(AutoGraphError, match="'x' was initialized with the wrong type"):
            run_autograph(f)(False)


class TestMixed:
    """Test a mix of supported autograph conversions and Catalyst control flow."""

    def test_no_python_loops(self):
        """Test AutoGraph behaviour on function with PennyLane loops."""

        def f():
            @for_loop(0, 3, 1)
            def loop(i, acc):
                return acc + i

            return loop(0)

        ag_fn = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_fn)()

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 3

    # @pytest.mark.usefixtures("autograph_strict_conversion")
    def test_cond_if_for_loop_for(self):
        """Test Python conditionals and loops together with their Catalyst counterparts."""

        # pylint: disable=cell-var-from-loop

        def f(x):
            acc = 0
            if x < 3:

                @for_loop(0, 3, 1)
                def loop(_, acc):
                    # Oddly enough, AutoGraph treats 'i' as an iter_arg even though it's not
                    # accessed after the for loop. Maybe because it is captured in the nested
                    # function's closure?
                    # TODO: remove the need for initializing 'i'
                    i = 0
                    for i in range(5):

                        @cond(i % 2 == 0)
                        def even():
                            return i

                        @even.otherwise
                        def even():
                            return 0

                        acc += even()

                    return acc

                acc = loop(acc)

            return acc

        ag_fn = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_fn)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2) == 18
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3) == 0


#
#
#
# class TestDisableAutograph:
#     """Test ways of disabling autograph conversion"""
#
#     def test_disable_autograph_decorator(self):
#         """Test disabling autograph with decorator."""
#
#         @disable_autograph
#         def f():
#             x = 6
#             if x > 5:
#                 y = x**2
#             else:
#                 y = x**3
#             return y
#
#         @qjit(autograph=True)
#         def g(x: float, n: int):
#             for _ in range(n):
#                 x = x + f()
#             return x
#
#         assert g(0.4, 6) == 216.4
#
#     def test_disable_autograph_context_manager(self):
#         """Test disabling autograph with context manager."""
#
#         def f():
#             x = 6
#             if x > 5:
#                 y = x**2
#             else:
#                 y = x**3
#             return y
#
#         @qjit(autograph=True)
#         def g():
#             x = 0.4
#             with disable_autograph:
#                 x += f()
#             return x
#
#         assert g() == 36.4
#


class TestAutographInclude:
    """Test include modules to autograph conversion"""

    def test_dummy_func(self):
        """Test dummy function branches."""

        assert dummy_func(6) == 36
        assert dummy_func(4) == 64

    def test_autograph_included_module(self):
        """Test autograph included module."""

        @qjit(autograph=True)
        def excluded_by_default(x: float, n: int):
            for _ in range(n):
                x = x + dummy_func(6)
            return x

        @qjit(autograph=True, autograph_include=["catalyst.utils.dummy"])
        def included(x: float, n: int):
            for _ in range(n):
                x = x + dummy_func(6)
            return x

        result_excluded_by_default = excluded_by_default(0.4, 6)
        assert result_excluded_by_default == 216.4 and result_excluded_by_default == included(
            0.4, 6
        )

    def test_invalid_autograph_include_with_no_autograph(self):
        """Test including modules when autograph is disabled as invalid input."""

        def fn(x: float, n: int):
            for _ in range(n):
                x = x + dummy_func(6)
            return x

        with pytest.raises(
            CompileError,
            match="In order for 'autograph_include' to work, 'autograph' must be set to True",
        ):
            qjit(autograph_include=["catalyst.utils.dummy"])(fn)


class TestDecorators:
    """Test if Autograph works when applied to a decorated function"""

    #     def test_vmap(self):
    #         """Test if Autograph works when applied to a decorated function with vmap"""
    #
    #         def workflow(axes_dct):
    #             return axes_dct["x"] + axes_dct["y"]
    #
    #         expected = jnp.array([1, 2, 3, 4, 5])
    #
    #         result = qjit(vmap(workflow, in_axes=({"x": None, "y": 0},)), autograph=True)(
    #             {"x": 1, "y": jnp.arange(5)}
    #         )
    #         assert jnp.allclose(result, expected)
    #
    def test_cond(self):
        """Test if Autograph works when applied to a decorated function with cond"""

        n = 6

        @cond(n > 4)
        def cond_fn():
            return n**2

        @cond_fn.otherwise
        def else_fn():
            return n

        ag_fn = run_autograph(cond_fn)
        jaxpr = jax.make_jaxpr(ag_fn)()

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 36

    def test_for_loop(self):
        """Test if Autograph works when applied to a decorated function with for_loop"""

        x = 5
        n = 6

        @for_loop(0, n, 1)
        def loop(_, agg):
            return agg + x

        ag_fn = run_autograph(loop)
        jaxpr = jax.make_jaxpr(ag_fn)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 30

    def test_while_loop(self):
        """Test if Autograph works when applied to a decorated function with while_loop"""

        n = 6

        @while_loop(lambda i: i < n)
        def loop(i):
            return i + 1

        ag_fn = run_autograph(loop)
        jaxpr = jax.make_jaxpr(ag_fn)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == n


if __name__ == "__main__":
    pytest.main(["-x", __file__])
