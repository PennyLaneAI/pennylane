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

"""PyTests for the AutoGraph source-to-source transformation feature."""

# pylint: disable=wrong-import-order, wrong-import-position, ungrouped-imports

import pytest
from numpy.testing import assert_allclose

import pennylane as qml
from pennylane import while_loop

pytestmark = [pytest.mark.capture]

jax = pytest.importorskip("jax")

from jax import numpy as jnp

# must be below jax importorskip
from jax.core import eval_jaxpr

from pennylane.capture.autograph.transformer import TRANSFORMER, run_autograph
from pennylane.exceptions import AutoGraphError

check_cache = TRANSFORMER.has_cache


class TestWhileLoops:
    """Test that the autograph transformations produce correct results on while loops."""

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
        assert "while_loop[" in str(jaxpr)

        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, expected)[0]
        assert result == expected

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
        assert "while_loop[" in str(jaxpr)

        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3)[0]
        assert result == 3

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
        assert "while_loop[" in str(jaxpr)

        result = eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.0**4)[0]
        expected = jnp.array(
            [
                0.00045727,
                0.00110912,
                0.0021832,
                0.0052954,
                0.000613,
                0.00148684,
                0.00292669,
                0.00709874,
                0.02114249,
                0.0512815,
                0.10094267,
                0.24483834,
                0.02834256,
                0.06874542,
                0.13531871,
                0.32821807,
            ]
        )
        assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    def test_whileloop_temporary_variable(self):
        """Test that temporary (local) variables can be initialized inside a while loop."""

        def f1():
            acc = 0
            while acc < 3:
                c = 2
                acc = acc + c

            return acc

        ag_circuit = run_autograph(f1)
        jaxpr = jax.make_jaxpr(ag_circuit)()
        assert "while_loop[" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 4

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
        assert "while_loop[" in str(jaxpr)
        assert "for_loop[" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == 0 + 1 + sum([1, 2, 3])

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
        assert "while_loop[" in str(jaxpr)
        assert "cond[" in str(jaxpr)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)[0] == sum([1, 1, 2, 2])

    def test_whileloop_exception(self):
        """Test for-loop can raise an error if strict-conversion is enabled."""

        def f1():
            acc = 0
            while acc < 5:
                raise RuntimeError("Test failure")
            return acc

        with pytest.raises(RuntimeError):
            run_autograph(f1)()

    def test_uninitialized_variables(self):
        """Verify errors for (potentially) uninitialized loop variables."""

        def f(pred: bool):
            while pred:
                x = 3

            return x

        with pytest.raises(AutoGraphError, match="'x' is potentially uninitialized"):
            ag_fn = run_autograph(f)
            jax.make_jaxpr(ag_fn)(False)

    def test_init_with_invalid_jax_type(self):
        """Test loop carried values initialized with an invalid JAX type."""

        def f(pred: bool):
            x = ""

            while pred:
                x = 3

            return x

        with pytest.raises(AutoGraphError, match="'x' was initialized with type <class 'str'>"):
            ag_fn = run_autograph(f)
            jax.make_jaxpr(ag_fn)(False)

    def test_while_loop(self):
        """Test if Autograph works when applied directly to a decorated function with while_loop"""

        n = 6

        @while_loop(lambda i: i < n)
        def loop(i):
            return i + 1

        ag_fn = run_autograph(loop)
        jaxpr = jax.make_jaxpr(ag_fn)(0)

        assert eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0)[0] == n
