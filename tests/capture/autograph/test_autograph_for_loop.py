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

# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports, too-many-public-methods

from unittest import mock

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import AutoGraphError

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

from jax import numpy as jnp

# must be below jax importorskip
from jax.core import eval_jaxpr
from malt.operators import py_builtins as ag_py_builtins

from pennylane.capture.autograph.ag_primitives import PEnumerate, PRange
from pennylane.capture.autograph.transformer import TRANSFORMER, run_autograph

check_cache = TRANSFORMER.has_cache


class TestCustomRangeAndEnumeration:
    """Test the custom PennyLane range and enumeration objects, PRange and
    PEnumeration"""

    def test_python_range_fallback(self):
        """Test that the custom PRange wrapper correctly falls back to Python."""

        # pylint: disable=protected-access

        pl_range = PRange(0, 5, 1)
        assert pl_range._py_range is None

        assert isinstance(pl_range.py_range, range)  # automatically instantiates the Python range
        assert isinstance(pl_range._py_range, range)
        assert pl_range[2] == 2

    def test_get_raw_range(self):
        """Test that the get_raw_range function accesses the intial inputs of the range"""

        pl_range = PRange(0, 5, 1)
        assert pl_range.get_raw_range() == (0, 5, 1)

    @mock.patch.dict(
        "pennylane.capture.autograph.ag_primitives.py_builtins_map",
        {**ag_py_builtins.BUILTIN_FUNCTIONS_MAP},
    )
    def test_prange_vs_range(self):
        """Test that using PRange fixes the TracerIntegerConversionError raised by JAX
        when initializing range"""

        def f1(n):
            _ = range(n)
            return n

        def f2(n):
            _ = PRange(n)
            return n

        # autograph runs, but it's not compatible with conversion to JAXPR because of indexing
        ag_f1 = run_autograph(f1)
        with pytest.raises(
            jax.errors.TracerIntegerConversionError,
            match=r"The __index__\(\) method was called on traced array",
        ):
            _ = jax.make_jaxpr(ag_f1)(3)

        # using PRange fixes it
        _ = jax.make_jaxpr(run_autograph(f2))(3)

    @pytest.mark.parametrize("start", [None, 0, 1, 2])
    def test_penumerate(self, start):
        """Test that PEnumerate is an instance of enumerate with additional attributes start_idx
        and iteration_target"""

        iterable = [qml.X(0), qml.Y(1), qml.Z(0)]
        expected_start = 0 if start is None else start

        enum = PEnumerate(iterable) if start is None else PEnumerate(iterable, start)

        assert enum.iteration_target == iterable
        assert enum.start_idx == expected_start

        assert isinstance(enum, enumerate)


class TestForLoops:
    """Test that the autograph transformations produce correct results on for loops."""

    def test_for_in_array(self):
        """Test for loop over JAX array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(params):
            for x in params:
                qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)(jnp.array([1.0, 2.0, 3.0]))

        def res(params):
            return eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, params)

        result = res(jnp.array([0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]))
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

    @pytest.mark.xfail(
        reason="relies on unimplemented fallback behaviour (implemented in catalyst)"
    )
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

    def test_iterating_over_wires(self):
        """Test that a wires obejct is a valid iteration target for a for loop,
        and can be converted to jaxpr and evaluated without issue"""

        def f():
            total = 0
            for w in qml.wires.Wires([0, 1, 2]):
                total += w

            return total

        ag_circuit = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 3

    @pytest.mark.xfail(
        reason="relies on unimplemented fallback behaviour (implemented in catalyst)"
    )
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

    @pytest.mark.xfail(
        reason="relies on unimplemented fallback behaviour (implemented in catalyst)"
    )
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

    def test_iteration_element_access(self):
        """Test that access to the iteration index/elements is possible after the loop executed
        (assuming initialization)."""

        def f1(acc):
            x = 0
            for x in [0, 4, 5]:
                acc = acc + x

            return x

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 5

        def f2(acc):
            i = 0
            l = jnp.array([0, 4, 5])
            for i in range(3):
                acc = acc + l[i]

            return i

        ag_circuit = run_autograph(f2)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 2

        def f3(acc):
            i, x = 0, 0
            for i, x in enumerate([0, 4, 5]):
                acc = acc + x

            return i, x

        ag_circuit = run_autograph(f3)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)

        assert np.allclose(result, [2, 5])

    # pylint: disable=undefined-loop-variable
    @pytest.mark.xfail(reason="currently unsupported, but we may find a way to do so in the future")
    def test_iteration_element_access_no_init(self):
        """Test that access to the iteration index/elements is possible after the loop executed
        even without prior initialization."""

        def f1(acc):
            for x in [0, 4, 5]:
                acc = acc + x

            return x

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 5

        def f2(acc):
            l = jnp.array([0, 4, 5])
            for i in range(3):
                acc = acc + l[i]

            return i

        ag_circuit = run_autograph(f2)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 2

        def f3(acc):
            for i, x in enumerate([0, 4, 5]):
                acc = acc + x

            return i, x

        ag_circuit = run_autograph(f3)
        jaxpr = jax.make_jaxpr(ag_circuit)(0)
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == (2, 5)

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


class TestErrors:
    """Test that informative errors are raised where expected"""

    def test_for_in_object_list(self):
        """Check the error raised when a for loop iterates over a Python list that
        is *not* convertible to an array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = ["0", "1", "2"]
            for x in params:
                qml.RY(int(x) / 4 * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(AutoGraphError, match="Could not convert the iteration target"):
            run_autograph(f)()

    def test_for_in_static_range_indexing_numeric_list(self):
        """Test an informative error is raised when using a for loop with a static range
        to index through an array-compatible Python list. This can be fixed by wrapping the
        list in a jax array, so the error raised here is actionable."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f():
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i in range(3):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            AutoGraphError,
            match="Make sure that loop variables are not used in tracing-incompatible ways",
        ):
            run_autograph(f)()

    def test_for_in_dynamic_range_indexing_numeric_list(self):
        """Test an informative error is raised when using a for loop with a dynamic range
        to index through an array-compatible Python list. This can be fixed by wrapping the
        list in a jax array, so the error raised here is actionable."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(n: int):
            params = [0.0, 1 / 4 * jnp.pi, 2 / 4 * jnp.pi]
            for i in range(n):
                qml.RY(params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            AutoGraphError,
            match="Make sure that loop variables are not used in tracing-incompatible ways",
        ):
            _ = run_autograph(f)(2)

    def test_for_in_dynamic_range_indexing_object_list(self):
        """Test that an error is raised for a for loop over a Python range with dynamic bounds
        that is used to index an array-incompatible Python list. This use-case is never possible,
        even with AutoGraph, because the list can't be wrapped in a jax array."""

        @qml.qnode(qml.device("default.qubit", wires=1))
        def f(n: int):
            params = ["0", "1", "2"]
            for i in range(n):
                qml.RY(int(params[i]) * jnp.pi, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            AutoGraphError,
            match="Make sure that loop variables are not used in tracing-incompatible ways",
        ):
            run_autograph(f)(2)

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

    def test_init_with_invalid_jax_type(self):
        """Test an error is raised if a loop carried values initialized with an invalid JAX type."""

        def f():
            acc = 0
            x = ""
            for x in [0, 4, 5]:
                acc = acc + x

            return x

        with pytest.raises(AutoGraphError, match="'x' was initialized with type <class 'str'>"):
            run_autograph(f)()

    def test_init_with_mismatched_type(self):
        """Test that an error is raised if a loop carried values initialized with a mismatched
        type compared to the values used inside the loop."""

        def f():
            acc = 0
            x = 0.0
            for x in [0, 4, 5]:
                acc = acc + x

            return x

        with pytest.raises(AutoGraphError, match="'x' was initialized with the wrong type"):
            run_autograph(f)()


class TestPennyLaneForLoops:
    """Test that the Autograph behaviour works as expected on functions that
    contain an explicit PennyLane for_loop"""

    def test_no_python_loops(self):
        """Test AutoGraph behaviour on function that contains a PennyLane loops."""

        def f():
            @qml.for_loop(0, 3, 1)
            def loop(i, acc):
                return acc + i

            return loop(0)

        ag_fn = run_autograph(f)
        jaxpr = jax.make_jaxpr(ag_fn)()

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 3

    def test_for_loop(self):
        """Test if Autograph works when applied directly to a for_loop"""

        x = 5
        n = 6

        @qml.for_loop(0, n, 1)
        def loop(_, agg):
            return agg + x

        ag_fn = run_autograph(loop)
        jaxpr = jax.make_jaxpr(ag_fn)(0)
        assert "for_loop[" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == 30

    def test_cond_if_for_loop_for(self):
        """Test Python conditionals and loops together with their PennyLane counterparts."""

        # pylint: disable=cell-var-from-loop

        def f(x):
            acc = 0
            if x < 3:

                @qml.for_loop(0, 3, 1)
                def loop(_, acc):
                    # Oddly enough, AutoGraph treats 'i' as an iter_arg even though it's not
                    # accessed after the for loop. Maybe because it is captured in the nested
                    # function's closure?
                    # TODO: remove the need for initializing 'i'
                    i = 0
                    for i in range(5):

                        @qml.cond(i % 2 == 0)
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
        assert "for_loop[" in str(jaxpr)
        assert "cond[" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2)[0] == 18
        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)[0] == 0


if __name__ == "__main__":
    pytest.main(["-x", __file__])
