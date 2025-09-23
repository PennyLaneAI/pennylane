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

"""PyTests for the integration between AutoGraph and PennyLane for the
source-to-source transformation feature."""

# pylint: disable = wrong-import-position, wrong-import-order, ungrouped-imports

from functools import partial

import numpy as np
import pytest
from malt.core import converter

import pennylane as qml
from pennylane import grad, jacobian, measure
from pennylane.capture.autograph.transformer import (
    NESTED_OPTIONS,
    STANDARD_OPTIONS,
    TOPLEVEL_OPTIONS,
    TRANSFORMER,
    PennyLaneTransformer,
    autograph_source,
    disable_autograph,
    run_autograph,
)

pytestmark = pytest.mark.capture

jax = pytest.importorskip("jax")
from jax import make_jaxpr

# must be below jax importorskip
# pylint: disable=wrong-import-position
from pennylane.exceptions import AutoGraphError

check_cache = TRANSFORMER.has_cache

# pylint: disable=too-few-public-methods, unnecessary-lambda-assignment


class TestPennyLaneTransformer:
    """Tests for the PennyLane child class of the diastatic-malt PytoPy transformer"""

    def test_lambda_function_error(self):
        """Test that a lambda function raises an error when passed to the transformer"""

        transformer = PennyLaneTransformer()
        user_context = converter.ProgramContext(TOPLEVEL_OPTIONS)

        def bad_circuit():

            @qml.while_loop(lambda x: x < 10)
            def loop(i):
                return i + 1

            return loop(0)

        def good_circuit():

            condition = lambda x: x < 10

            @qml.while_loop(condition)
            def loop(i):
                return i + 1

            return loop(0)

        with pytest.raises(
            AutoGraphError,
            match="AutoGraph currently does not support lambda functions as a loop condition for `qml.while_loop`.",
        ):
            transformer.transform(bad_circuit, user_context)

        # Check that the good circuit does not raise an error
        new_fn, _, _ = transformer.transform(good_circuit, user_context)
        assert new_fn() == 10

    def test_transform_on_function(self):
        """Test the transform method on a function works as expected"""

        transformer = PennyLaneTransformer()
        user_context = converter.ProgramContext(TOPLEVEL_OPTIONS)

        def fn(x):
            return 2 * x

        new_fn, _, _ = transformer.transform(fn, user_context)

        assert fn(1.23) == new_fn(1.23)
        assert "inner_factory.<locals>.ag__fn" in str(new_fn)

    def test_transform_on_lambda(self):
        """Test the transform method on a lambda function works as expected"""

        transformer = PennyLaneTransformer()
        user_context = converter.ProgramContext(TOPLEVEL_OPTIONS)

        new_fn, _, _ = transformer.transform(lambda x: 2 * x, user_context)

        assert new_fn(1.23) == 2.46
        assert "inner_factory.<locals>.<lambda>" in str(new_fn)

    def test_transform_on_qnode(self):
        """Test the transform method on a QNode updates the qnode.func"""
        transformer = PennyLaneTransformer()
        user_context = converter.ProgramContext(TOPLEVEL_OPTIONS)

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circ(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        new_circ, _, _ = transformer.transform(circ, user_context)

        assert circ(1.23) == new_circ(1.23)
        assert "inner_factory.<locals>.ag__circ" in str(new_circ.func)

    def test_get_extra_locals(self):
        """Test that the extra_locals for autograph are updated to replace the relevant
        functions with our custom ag_primtives"""

        transformer = PennyLaneTransformer()

        assert transformer._extra_locals is None  # pylint:disable = protected-access

        locals = transformer.get_extra_locals()
        ag_fn_dict = locals["ag__"].__dict__

        assert ag_fn_dict["if_stmt"].__module__ == "pennylane.capture.autograph.ag_primitives"
        assert ag_fn_dict["while_stmt"].__module__ == "pennylane.capture.autograph.ag_primitives"
        assert ag_fn_dict["for_stmt"].__module__ == "pennylane.capture.autograph.ag_primitives"
        assert (
            ag_fn_dict["converted_call"].__module__ == "pennylane.capture.autograph.ag_primitives"
        )

    @pytest.mark.parametrize("options", [TOPLEVEL_OPTIONS, NESTED_OPTIONS, STANDARD_OPTIONS])
    def test_function_caching(self, options):
        """Test that retrieving the cached function from the transformer works
        as expected for all options sets"""

        transformer = PennyLaneTransformer()
        user_context = converter.ProgramContext(options)

        def fn(x):
            """test"""
            return 2 * x

        assert transformer.has_cache(fn) is False

        transformer.transform(fn, user_context)

        assert transformer.has_cache(fn) is True
        assert transformer.get_cached_function(fn)(1.7) == fn(1.7)


# pylint: disable=too-many-public-methods
class TestIntegration:
    """Test that the autograph transformations trigger correctly in different settings."""

    def test_run_autograph_on_converted_function(self):
        """Test that running run_autograph on a function that has already been converted
        does not trigger the transformation again."""

        def fn(x):
            return x**2

        ag_fn = run_autograph(fn)
        ag_ag_fn = run_autograph(ag_fn)
        assert ag_ag_fn is ag_fn
        assert ag_ag_fn(4) == 16

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
        """Test run_autograph applied to a callable object."""

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
        assert ag_fn(4) == 16

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert check_cache(inner)

    def test_qnode(self):
        """Test autograph on a QNode."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circ(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_fn = run_autograph(circ)
        assert ag_fn(np.pi) == -1

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(circ.func)

    def test_indirect_qnode(self):
        """Test autograph on a QNode called from within a classical function."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        def fn(x: float):
            return inner(x)

        ag_fn = run_autograph(fn)
        assert ag_fn(np.pi) == -1

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert check_cache(inner.func)

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

        def fn(x: float):
            return inner1(x) + inner2(x)

        ag_fn = run_autograph(fn)
        assert ag_fn(np.pi) == -2

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert check_cache(inner1.func)
        assert check_cache(inner2.func)

    def test_adjoint_of_operator_instance(self):
        """Test that the adjoint of an operator successfully passes through autograph"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circ():
            qml.adjoint(qml.X(0))
            return qml.expval(qml.Z(0))

        plxpr = qml.capture.make_plxpr(circ, autograph=True)()
        assert jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts)[0] == -1

    def test_adjoint_of_operator_type(self):
        """Test that the adjoint of an operator successfully passes through autograph"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circ():
            qml.adjoint(qml.X)(0)
            return qml.expval(qml.Z(0))

        plxpr = qml.capture.make_plxpr(circ, autograph=True)()
        assert jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts)[0] == -1

    def test_adjoint_no_argument(self):
        """Test that passing no argument to qml.adjoint raises an error."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            qml.adjoint()
            return qml.probs(wires=0)

        with pytest.raises(ValueError, match="adjoint requires at least one argument"):
            _ = qml.capture.make_plxpr(circ, autograph=True)()

    def test_adjoint_wrong_argument(self):
        """Test that passing an invalid argument to qml.adjoint raises an error."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            qml.adjoint(3)
            return qml.probs(wires=0)

        with pytest.raises(
            ValueError, match="First argument to adjoint must be callable or an Operation"
        ):
            _ = qml.capture.make_plxpr(circ, autograph=True)()

    @pytest.mark.parametrize(
        "func1, func2, prim1, prim2",
        [
            (qml.adjoint, partial(qml.ctrl, control=0), "adjoint_transform", "ctrl_transform"),
            (partial(qml.ctrl, control=0), qml.adjoint, "ctrl_transform", "adjoint_transform"),
            (qml.adjoint, qml.adjoint, "adjoint_transform", "adjoint_transform"),
            (
                partial(qml.ctrl, control=0),
                partial(qml.ctrl, control=1),
                "ctrl_transform",
                "ctrl_transform",
            ),
        ],
    )
    def test_nested_adjoint_ctrl(self, func1, func2, prim1, prim2):
        """Test that nested adjoint and ctrl successfully pass through autograph"""

        # Build the nested operator
        op = func2(qml.X)
        final_op = func1(op)

        @qml.qnode(qml.device("default.qubit", wires=3))
        def circ():
            final_op(wires=2)
            return qml.state()

        plxpr = qml.capture.make_plxpr(circ, autograph=True)()
        qfunc_jaxpr = plxpr.eqns[0].params["qfunc_jaxpr"]
        hop_outer = qfunc_jaxpr.eqns[0].primitive
        hop_inner = qfunc_jaxpr.eqns[0].params["jaxpr"].eqns[0].primitive
        assert str(hop_outer) == prim1
        assert str(hop_inner) == prim2

    def test_ctrl_of_operator_instance(self):
        """Test that controlled operators successfully pass through autograph"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circ():
            qml.H(0)
            qml.ctrl(qml.X(1), control=0)
            return qml.state()

        plxpr = qml.capture.make_plxpr(circ, autograph=True)()
        expected_state = 1 / np.sqrt(2) * jax.numpy.array([1, 0, 0, 1])
        result = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts)[0]
        assert jax.numpy.allclose(result, expected_state)

    def test_ctrl_of_operator_type(self):
        """Test that controlled operators successfully pass through autograph"""

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circ():
            qml.H(0)
            qml.ctrl(qml.X, control=0)(1)
            return qml.state()

        plxpr = qml.capture.make_plxpr(circ, autograph=True)()
        expected_state = 1 / np.sqrt(2) * jax.numpy.array([1, 0, 0, 1])
        result = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts)[0]
        assert jax.numpy.allclose(result, expected_state)

    def test_ctrl_no_argument(self):
        """Test that passing no argument to qml.ctrl raises an error."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            qml.ctrl()
            return qml.probs(wires=0)

        with pytest.raises(ValueError, match="ctrl requires at least one argument"):
            _ = qml.capture.make_plxpr(circ, autograph=True)()

    def test_ctrl_wrong_argument(self):
        """Test that passing an invalid argument to qml.ctrl raises an error."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            qml.ctrl(3)
            return qml.probs(wires=0)

        with pytest.raises(
            ValueError, match="First argument to ctrl must be callable or an Operation"
        ):
            _ = qml.capture.make_plxpr(circ, autograph=True)()

    def test_adjoint_wrapper(self):
        """Test conversion is happening successfully on functions wrapped with 'adjoint'."""

        def inner(x):
            if x > 0:
                qml.I(wires=0)  # Test will only pass if this is hit
            else:
                qml.RY(x, wires=0)
            qml.RY(x, wires=0)

        @qml.qnode(qml.device("default.qubit", wires=1))
        def circ(x: float):
            inner(x * 2)
            qml.adjoint(inner)(x)
            return qml.probs()

        ag_fn = run_autograph(circ)
        phi = np.pi / 2
        assert np.allclose(ag_fn(phi), [np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2])
        assert not np.allclose(ag_fn(-phi), [np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2])

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(circ.func)
        assert check_cache(inner)

    def test_ctrl_wrapper(self):
        """Test conversion is happening successfully on functions wrapped with 'ctrl'."""

        def inner(x):
            if x == np.pi:
                qml.I(wires=0)  # Test will only pass if this is hit
            else:
                qml.RX(x, wires=0)
            qml.RY(x, wires=0)

        @qml.qnode(qml.device("default.qubit", wires=2))
        def circ(x: float):
            qml.PauliX(1)
            qml.ctrl(inner, control=1)(x)
            return qml.probs()

        ag_fn = run_autograph(circ)
        assert np.allclose(ag_fn(np.pi), [0.0, 0.0, 0.0, 1.0])
        assert not np.allclose(ag_fn(np.pi + 0.1), [0.0, 0.0, 0.0, 1.0])
        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(circ.func)
        assert check_cache(inner)

    def test_grad_wrapper(self):
        """Test conversion is happening successfully on functions wrapped with 'grad'."""

        def inner(x):
            return 2 * x

        def fn(x: float):
            return grad(inner)(x)

        ag_fn = run_autograph(fn)
        assert ag_fn(3.0) == 2.0

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert check_cache(inner)

    def test_jacobian_wrapper(self):
        """Test conversion is happening successfully on functions wrapped with 'jacobian'."""

        def inner(x):
            return 2 * x, x**2

        def fn(x: float):
            return jacobian(inner)(x)

        ag_fn = run_autograph(fn)
        assert ag_fn(3.0) == tuple([jax.numpy.array(2.0), jax.numpy.array(6.0)])

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(fn)
        assert check_cache(inner)

    def test_tape_transform(self):
        """Test if tape transform is applied when autograph is on."""

        dev = qml.device("default.qubit", wires=1)

        @qml.transform
        def my_quantum_transform(tape):
            raise NotImplementedError

        @my_quantum_transform
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_fn = run_autograph(circuit)

        with pytest.raises(NotImplementedError):
            ag_fn(0.5)

    @pytest.mark.xfail(
        strict=False,
        reason="MCM one shot not currently supported with program capture. See sc-83580.",
    )
    def test_mcm_one_shot(self, seed):
        """Test if mcm one-shot miss transforms."""
        dev = qml.device("default.qubit", wires=5, seed=seed)

        @qml.set_shots(20)
        @qml.qnode(dev, mcm_method="one-shot", postselect_mode="hw-like")
        def circ(x):
            qml.RX(x, wires=0)
            measure(0, postselect=1)
            return qml.sample(wires=0)

        ag_fn = run_autograph(circ)
        # If transforms are missed, the output will be all ones.
        assert not np.all(ag_fn(0.9) == 1)

        assert hasattr(ag_fn, "ag_unconverted")
        assert check_cache(circ.func)

    def test_custom_operation(self):
        """Test that autograph can be applied to circuits with custom operations."""

        class MyOperation(qml.operation.Operation):
            pass

        def f(x):
            MyOperation(x, wires=0)

        ag_fn = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_fn)(0.5)
        # pylint: disable=protected-access
        assert jaxpr.jaxpr.eqns[0].primitive == MyOperation._primitive
        assert len(jaxpr.jaxpr.eqns) == 1


class TestCodePrinting:
    """Test that the transformed source code can be printed in different settings."""

    def test_unconverted(self):
        """Test printing on an unconverted function."""

        def fn(x):
            return x**2

        with pytest.raises(AutoGraphError, match="function was not converted by AutoGraph"):
            autograph_source(fn)

    def test_lambda(self):
        """Test printing on a lambda function."""

        fn = lambda x: x**2
        fn = run_autograph(fn)

        assert "ag__lam" in autograph_source(fn)

    def test_classical_function(self):
        """Test printing on a purely classical function."""

        def fn(x):
            return x**2

        fn = run_autograph(fn)
        assert "def ag__fn(x" in autograph_source(fn)

    def test_nested_function(self):
        """Test printing on nested classical functions."""

        def inner(x):
            return x**2

        def fn(x: int):
            return inner(x)

        fn = run_autograph(fn)

        # if we don't call the function, the inner function isn't found in the TRANSFORMER cache,
        # and we can't get the source for the inner function
        _ = fn(2)

        assert "def ag__fn(x" in autograph_source(fn)
        assert "def ag__inner(x" in autograph_source(inner)

    def test_qnode(self):
        """Test printing on a QNode."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def fn(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        fn = run_autograph(fn)

        assert autograph_source(fn)

    def test_indirect_qnode(self):
        """Test printing on a QNode called from within a classical function."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        def fn(x: float):
            return inner(x)

        fn = run_autograph(fn)
        # if we don't call the function, the inner function isn't found in the TRANSFORMER cache,
        # and we can't get the source for the inner function
        _ = fn(2)

        assert "def ag__fn(x" in autograph_source(fn)
        assert "def ag__inner(x" in autograph_source(inner)

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

        def fn(x: float):
            return inner1(x) + inner2(x)

        fn = run_autograph(fn)
        # if we don't call the function, the inner function isn't found in the TRANSFORMER cache,
        # and we can't get the source for the inner function
        _ = fn(2)

        assert "def ag__fn(x" in autograph_source(fn)
        assert "def ag__inner1(x" in autograph_source(inner1)
        assert "def ag__inner2(x" in autograph_source(inner2)


if __name__ == "__main__":
    pytest.main(["-x", __file__])


class TestDisableAutograph:
    """Test ways of disabling autograph conversion"""

    def test_disable_autograph_decorator(self):
        """Test disabling autograph with decorator."""

        @disable_autograph
        def f():
            x = 2
            if x > 1:
                y = x**2
            else:
                y = x**3
            return y

        def g(x: int, n: int):
            for _ in range(n):
                x = x + f()
            return x

        g_ag = run_autograph(g)
        g_ag_jaxpr = make_jaxpr(g_ag)(1, 3)
        assert "for_loop" in str(g_ag_jaxpr.jaxpr)
        # If autograph was disabled, the cond primitive will not be captured.
        assert "cond" not in str(g_ag_jaxpr.jaxpr)
        assert g_ag(1, 3) == 13  # 1 + 4 * 3

    def test_disable_autograph_context_manager(self):
        """Test disabling autograph with context manager."""

        def f():
            x = 2
            if x > 1:
                y = x**2
            else:
                y = x**3
            return y

        def g(x: int, n: int):
            for _ in range(n):
                with disable_autograph:
                    x += f()
            return x

        g_ag = run_autograph(g)
        g_ag_jaxpr = make_jaxpr(g_ag)(1, 3)
        assert "for_loop" in str(g_ag_jaxpr.jaxpr)
        # If autograph was disabled, the cond primitive will not be captured.
        assert "cond" not in str(g_ag_jaxpr.jaxpr)

        assert g_ag(1, 3) == 13  # 1 + 4 * 3
