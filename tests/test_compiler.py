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
"""
Unit tests for the compiler subpackage.
"""

# pylint: disable=import-outside-toplevel
from unittest.mock import patch

import mcm_utils
import numpy as np
import pytest

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import CompileError
from pennylane.transforms.dynamic_one_shot import fill_in_value

catalyst = pytest.importorskip("catalyst")
jax = pytest.importorskip("jax")

pytestmark = pytest.mark.external

from jax import numpy as jnp  # pylint:disable=wrong-import-order, wrong-import-position
from jax.core import ShapedArray  # pylint:disable=wrong-import-order, wrong-import-position

# pylint: disable=too-few-public-methods, too-many-public-methods


@pytest.fixture
def catalyst_incompatible_version():
    """An incompatible (low) version for Catalyst"""
    with patch("importlib.metadata.version") as mock_version:
        mock_version.return_value = "0.0.1"
        yield


@pytest.mark.usefixtures("catalyst_incompatible_version")
def test_catalyst_incompatible():
    """Test qjit with an incompatible Catalyst version that's lower than required."""

    dev = qp.device("lightning.qubit", wires=1)

    @qp.qnode(dev)
    def circuit():
        qp.PauliX(0)
        return qp.state()

    with pytest.raises(
        CompileError,
        match="PennyLane-Catalyst 0.[0-9]+.0 or greater is required, but installed 0.0.1",
    ):
        qp.qjit(circuit)()


class TestCatalyst:
    """Test ``qp.qjit`` with Catalyst"""

    def test_compiler(self):
        """Test compiler active and available methods"""

        assert not qp.compiler.active()
        assert not qp.compiler.available("SomeRandomCompiler")

        assert qp.compiler.available("catalyst")
        assert qp.compiler.available_compilers() == ["catalyst", "cuda_quantum"]

    def test_active_compiler(self):
        """Test `qp.compiler.active_compiler` inside a simple circuit"""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(phi, theta):
            if qp.compiler.active_compiler() == "catalyst":
                qp.RX(phi, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.PhaseShift(theta, wires=0)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(jnp.pi, jnp.pi / 2), 1.0)
        assert jnp.allclose(qp.qjit(circuit)(jnp.pi, jnp.pi / 2), -1.0)

    def test_active(self):
        """Test `qp.compiler.active` inside a simple circuit"""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(phi, theta):
            if qp.compiler.active():
                qp.RX(phi, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.PhaseShift(theta, wires=0)
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(jnp.pi, jnp.pi / 2), 1.0)
        assert jnp.allclose(qp.qjit(circuit)(jnp.pi, jnp.pi / 2), -1.0)

    @pytest.mark.parametrize("jax_enable_x64", [False, True])
    def test_jax_enable_x64(self, jax_enable_x64):
        """Test whether `qp.compiler.active` changes `jax_enable_x64`."""
        jax.config.update("jax_enable_x64", jax_enable_x64)
        assert jax.config.jax_enable_x64 is jax_enable_x64
        qp.compiler.active()
        assert jax.config.jax_enable_x64 is jax_enable_x64

    def test_qjit_circuit(self):
        """Test JIT compilation of a circuit with 2-qubit"""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(theta):
            qp.Hadamard(wires=0)
            qp.RX(theta, wires=1)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(wires=1))

        assert jnp.allclose(circuit(0.5), 0.0)

    def test_qjit_aot(self):
        """Test AOT compilation of a circuit with 2-qubit"""

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qp.RY(theta, wires=0)
            qp.Rot(z[0], z[1], z[2], wires=0)
            return qp.state()

        # Check that the compilation happens at definition
        assert circuit.compiled_function

        result = circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))
        expected = jnp.array(
            [0.75634905 - 0.52801002j, 0.0 + 0.0j, 0.35962678 + 0.14074839j, 0.0 + 0.0j]
        )
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize(
        "_in,_out",
        [
            (0, False),
            (1, True),
        ],
    )
    def test_variable_capture_multiple_devices(self, _in, _out):
        """Test variable capture using multiple backend devices."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit
        def workflow(n: int):
            @qp.qnode(dev)
            def f(x: float):
                qp.RX(n * x, wires=n)
                return qp.expval(qp.PauliZ(wires=n))

            @qp.qnode(dev)
            def g(x: float):
                qp.RX(x, wires=1)
                return qp.expval(qp.PauliZ(wires=1))

            return jnp.array_equal(f(jnp.pi), g(jnp.pi))

        assert workflow(_in) == _out

    def test_args_workflow(self):
        """Test arguments with workflows."""

        @qp.qjit
        def workflow1(params1, params2):
            """A classical workflow"""
            res1 = params1["a"][0][0] + params2[1]
            return jnp.sin(res1)

        params1 = {
            "a": [[0.1], 0.2],
        }
        params2 = (0.6, 0.8)
        expected = 0.78332691
        result = workflow1(params1, params2)
        assert jnp.allclose(result, expected)

    def test_return_value_dict(self):
        """Test pytree return values."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit1(params):
            qp.RX(params[0], wires=0)
            qp.RX(params[1], wires=1)
            return {
                "w0": qp.expval(qp.PauliZ(0)),
                "w1": qp.expval(qp.PauliZ(1)),
            }

        jitted_fn = qp.qjit(circuit1)

        params = [0.2, 0.6]
        expected = {"w0": 0.98006658, "w1": 0.82533561}
        result = jitted_fn(params)
        assert isinstance(result, dict)
        assert jnp.allclose(result["w0"], expected["w0"])
        assert jnp.allclose(result["w1"], expected["w1"])

    def test_qjit_python_if(self):
        """Test JIT compilation with the autograph support"""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit(autograph=True)
        @qp.qnode(dev)
        def circuit(x: int):
            if x < 5:
                qp.Hadamard(wires=0)
            else:
                qp.T(wires=0)

            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(3), 0.0)
        assert jnp.allclose(circuit(5), 1.0)

    def test_compilation_opt(self):
        """Test user-configurable compilation options"""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit(target="mlir", keep_intermediate=True)
        @qp.qnode(dev)
        def circuit(x: float):
            qp.RX(x, wires=0)
            qp.RX(x**2, wires=1)
            return qp.expval(qp.PauliZ(0))

        mlir_str = str(circuit.mlir)
        result_header = "func.func public @circuit(%arg0: tensor<f64>) -> tensor<f64>"
        assert result_header in mlir_str
        circuit.workspace.cleanup()

    def test_qjit_adjoint(self):
        """Test JIT compilation with adjoint support"""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit
        @qp.qnode(device=dev)
        def workflow_cl(theta, wires):
            def func():
                qp.RX(theta, wires=wires)

            qp.adjoint(func)()
            return qp.probs()

        @qp.qnode(device=dev)
        def workflow_pl(theta, wires):
            def func():
                qp.RX(theta, wires=wires)

            qp.adjoint(func)()
            return qp.probs()

        assert jnp.allclose(workflow_cl(0.1, [1]), workflow_pl(0.1, [1]))

    def test_qjit_adjoint_lazy(self):
        """Test that the lazy kwarg is supported."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(device=dev)
        def workflow_pl(theta, wires):
            qp.Hadamard(wires)
            qp.adjoint(qp.RX(theta, wires=wires), lazy=False)
            return qp.probs()

        workflow_cl = qp.qjit(workflow_pl)

        assert jnp.allclose(workflow_cl(0.1, [1]), workflow_pl(0.1, [1]))

    def test_control(self):
        """Test that control works with qjit."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit
        @qp.qnode(dev)
        def workflow(theta, w, cw):
            qp.Hadamard(wires=[0])
            qp.Hadamard(wires=[1])

            def func(arg):
                qp.RX(theta, wires=arg)

            def cond_fn():
                qp.RY(theta, wires=w)

            qp.ctrl(func, control=[cw])(w)
            qp.ctrl(qp.cond(theta > 0.0, cond_fn), control=[cw])()
            qp.ctrl(qp.RZ, control=[cw])(theta, wires=w)
            qp.ctrl(qp.RY(theta, wires=w), control=[cw])
            return qp.probs()

        assert jnp.allclose(
            workflow(jnp.pi / 4, 1, 0), jnp.array([0.25, 0.25, 0.03661165, 0.46338835])
        )


class TestCatalystControlFlow:
    """Test ``qp.qjit`` with Catalyst's control-flow operations"""

    def test_while_loop_defined_outside_qjit(self):
        """Test that the while loop can be defined outside the qjit."""

        @qp.while_loop(lambda n: n < 4)
        def w(n):
            return n + 1

        res = qp.qjit(w)(0)
        assert qp.math.allclose(res, 4)

    def test_for_loop_defined_outside_qjit(self):
        """Test that a for_loop can be defined outside the qjit."""

        @qp.for_loop(5)
        def f(i, x):
            return x + i

        res = qp.qjit(f)(0)
        assert qp.math.allclose(res, 10)

    def test_alternating_while_loop(self):
        """Test simple while loop."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(n):
            @qp.while_loop(lambda v: v[0] < v[1])
            def loop(v):
                qp.PauliX(wires=0)
                return v[0] + 1, v[1]

            loop((0, n))
            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(1), -1.0)

    def test_nested_while_loops(self):
        """Test nested while loops."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(n, m):
            @qp.while_loop(lambda i, _: i < n)
            def outer(i, sm):
                @qp.while_loop(lambda j: j < m)
                def inner(j):
                    return j + 1

                return i + 1, sm + inner(0)

            return outer(0, 0)[1]

        assert circuit(5, 6) == 30  # 5 * 6
        assert circuit(4, 7) == 28  # 4 * 7

    def test_dynamic_wires_for_loops(self):
        """Test for loops with iteration index-dependant wires."""
        dev = qp.device("lightning.qubit", wires=6)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(n: int):
            qp.Hadamard(wires=0)

            @qp.for_loop(0, n - 1, 1)
            def loop_fn(i):
                qp.CNOT(wires=[i, i + 1])

            loop_fn()
            return qp.state()

        expected = np.zeros(2**6)
        expected[[0, 2**6 - 1]] = 1 / np.sqrt(2)

        assert jnp.allclose(circuit(6), expected)

    def test_nested_for_loops(self):
        """Test nested for loops."""
        dev = qp.device("lightning.qubit", wires=4)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(n):
            # Input state: equal superposition
            @qp.for_loop(0, n, 1)
            def init(i):
                qp.Hadamard(wires=i)

            # QFT
            @qp.for_loop(0, n, 1)
            def qft(i):
                qp.Hadamard(wires=i)

                @qp.for_loop(i + 1, n, 1)
                def inner(j):
                    qp.ControlledPhaseShift(np.pi / 2 ** (n - j + 1), [i, j])

                inner()

            init()
            qft()

            # Expected output: |100...>
            return qp.state()

        assert jnp.allclose(circuit(4), jnp.eye(2**4)[0])

    def test_cond(self):
        """Test condition with simple true_fn"""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x: float):
            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            qp.cond(x > 1.4, ansatz_true)()

            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(1.4), 1.0)
        assert jnp.allclose(circuit(1.6), 0.0)

    def test_cond_with_else(self):
        """Test condition with simple true_fn and false_fn"""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x: float):
            def ansatz_true():
                qp.RX(x, wires=0)
                qp.Hadamard(wires=0)

            def ansatz_false():
                qp.RY(x, wires=0)

            qp.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(1.4), 0.16996714)
        assert jnp.allclose(circuit(1.6), 0.0)

    def test_cond_with_elif(self):
        """Test condition with a simple elif branch"""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x):
            def true_fn():
                qp.RX(x, wires=0)

            def elif_fn():
                qp.RY(x, wires=0)

            def false_fn():
                qp.RX(x**2, wires=0)

            qp.cond(x > 2.7, true_fn, false_fn, ((x > 1.4, elif_fn),))()

            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(1.2), 0.13042371)
        assert jnp.allclose(circuit(jnp.pi), -1.0)

    def test_cond_with_elifs(self):
        """Test condition with multiple elif branches"""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def circuit(x):
            def true_fn():
                qp.RX(x, wires=0)

            def elif1_fn():
                qp.RY(x, wires=0)

            def elif2_fn():
                qp.RZ(x, wires=0)

            def false_fn():
                qp.RX(x**2, wires=0)

            qp.cond(x > 2.7, true_fn, false_fn, ((x > 2.4, elif1_fn), (x > 1.4, elif2_fn)))()

            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(1.5), 1.0)
        assert jnp.allclose(circuit(jnp.pi), -1.0)

    def test_cond_with_elif_interpreted(self):
        """Test condition with an elif branch in interpreted mode"""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def circuit(x):
            def true_fn():
                qp.RX(x, wires=0)

            def elif_fn():
                qp.RX(x**2, wires=0)

            qp.cond(x > 2.7, true_fn, None, ((x > 1.4, elif_fn),))()

            return qp.expval(qp.PauliZ(0))

        assert jnp.allclose(circuit(1.2), 1.0)
        assert jnp.allclose(circuit(jnp.pi), -1.0)

    def test_cond_with_decorator_syntax(self):
        """Test condition using the decorator syntax"""

        @qp.qjit
        def f(x):
            @qp.cond(x > 0)
            def conditional():
                return (x + 1) ** 2

            @conditional.else_if(x < -2)
            def conditional_elif():  # pylint: disable=unused-variable
                return x + 1

            @conditional.otherwise
            def conditional_false_fn():  # pylint: disable=unused-variable
                return -(x + 1)

            return conditional()

        assert np.allclose(f(0.5), (0.5 + 1) ** 2)
        assert np.allclose(f(-0.5), -(-0.5 + 1))
        assert np.allclose(f(-2.5), (-2.5 + 1))


class TestCatalystGrad:
    """Test ``qp.qjit`` with Catalyst's grad operations"""

    @pytest.mark.parametrize("argnums", (None, 0))
    @pytest.mark.parametrize("g_fn", (qp.grad, qp.jacobian))
    def test_lazy_dispatch_grad(self, g_fn, argnums):
        """Test that grad is lazily dispatched to the catalyst version at runtime."""

        def f(x):
            return x**2

        g = qp.qjit(g_fn(f, argnums=argnums))(0.5)
        assert qp.math.allclose(g, 1.0)
        assert qp.math.get_interface(g) == "jax"

    def test_grad_classical_preprocessing(self):
        """Test the grad transformation with classical preprocessing."""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        def workflow(x):
            @qp.qnode(dev)
            def circuit(x):
                qp.RX(jnp.pi * x, wires=0)
                return qp.expval(qp.PauliY(0))

            g = qp.grad(circuit)
            return g(x)

        assert jnp.allclose(workflow(2.0), -jnp.pi)

    def test_grad_with_postprocessing(self):
        """Test the grad transformation with classical preprocessing and postprocessing."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        def workflow(theta):
            @qp.qnode(dev, diff_method="adjoint")
            def circuit(theta):
                qp.RX(jnp.exp(theta**2) / jnp.cos(theta / 4), wires=0)
                return qp.expval(qp.PauliZ(wires=0))

            def loss(theta):
                return jnp.pi / jnp.tanh(circuit(theta))

            return qp.grad(loss, method="auto")(theta)

        assert jnp.allclose(workflow(1.0), 5.04324559)

    def test_grad_with_multiple_qnodes(self):
        """Test the grad transformation with multiple QNodes with their own differentiation methods."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        def workflow(theta):
            @qp.qnode(dev, diff_method="parameter-shift")
            def circuit_A(params):
                qp.RX(jnp.exp(params[0] ** 2) / jnp.cos(params[1] / 4), wires=0)
                return qp.probs()

            @qp.qnode(dev, diff_method="adjoint")
            def circuit_B(params):
                qp.RX(jnp.exp(params[1] ** 2) / jnp.cos(params[0] / 4), wires=0)
                return qp.expval(qp.PauliZ(wires=0))

            def loss(params):
                return jnp.prod(circuit_A(params)) + circuit_B(params)

            return qp.grad(loss)(theta)

        result = workflow(jnp.array([1.0, 2.0]))
        reference = jnp.array([0.57367285, 44.4911605])

        assert jnp.allclose(result, reference)

    def test_grad_with_pure_classical(self):
        """Test the grad transformation with purely classical functions."""

        def square(x: float):
            return x**2

        @qp.qjit
        def dsquare(x: float):
            return qp.grad(square)(x)

        assert jnp.allclose(dsquare(2.3), 4.6)

    def test_jacobian_diff_method(self):
        """Test the Jacobian transformation with the device diff_method."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev, diff_method="parameter-shift")
        def func(p):
            qp.RY(p, wires=0)
            return qp.probs(wires=0)

        @qp.qjit
        def workflow(p: float):
            return qp.jacobian(func, method="auto")(p)

        result = workflow(0.5)
        reference = qp.jacobian(func, argnums=0)(0.5)

        assert jnp.allclose(result, reference)

    def test_jacobian_auto(self):
        """Test the Jacobian transformation with 'auto'."""
        dev = qp.device("lightning.qubit", wires=1)

        def workflow(x):
            @qp.qnode(dev)
            def circuit(x):
                qp.RX(jnp.pi * x[0], wires=0)
                qp.RY(x[1], wires=0)
                return qp.probs()

            g = qp.jacobian(circuit)
            return g(x)

        reference = workflow(np.array([2.0, 1.0]))
        result = qp.qjit(workflow)(jnp.array([2.0, 1.0]))

        assert jnp.allclose(result, reference)

    def test_jacobian_fd(self):
        """Test the Jacobian transformation with 'fd'."""
        dev = qp.device("lightning.qubit", wires=1)

        def workflow(x):
            @qp.qnode(dev)
            def circuit(x):
                qp.RX(np.pi * x[0], wires=0)
                qp.RY(x[1], wires=0)
                return qp.probs()

            g = qp.jacobian(circuit, method="fd", h=0.3)
            return g(x)

        result = qp.qjit(workflow)(np.array([2.0, 1.0]))
        reference = np.array([[-0.37120096, -0.45467246], [0.37120096, 0.45467246]])
        assert jnp.allclose(result, reference)

    def test_jvp(self):
        """Test that the correct JVP is returned with QJIT."""

        @qp.qjit
        def jvp(params, tangent):
            def f(x):
                y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
                return jnp.stack(y)

            return qp.jvp(f, [params], [tangent])

        x = jnp.array([0.1, 0.2])
        tangent = jnp.array([0.3, 0.6])
        res = jvp(x, tangent)
        assert len(res) == 2
        assert jnp.allclose(res[0], jnp.array([0.09983342, 0.04, 0.02]))
        assert jnp.allclose(res[1], jnp.array([0.29850125, 0.24000006, 0.12]))

    @pytest.mark.parametrize("argnum_name", ("argnum", "argnums"))
    def test_jvp_argnums(self, argnum_name):
        """Test that res."""

        def f(x, y):
            return y * x**2

        @qp.qjit
        def w(x, y):
            return qp.jvp(f, [x, y], [1.0], **{argnum_name: [1]})

        x = jnp.array(0.5)
        y = jnp.array(3.0)

        if argnum_name == "argnum":
            with pytest.warns(
                qp.exceptions.PennyLaneDeprecationWarning, match="argnum in qp.jvp"
            ):
                res, dres = w(x, y)
        else:
            res, dres = w(x, y)

        assert qp.math.allclose(res, f(x, y))
        assert qp.math.allclose(dres, x**2)

    @pytest.mark.parametrize("argnum_name", ("argnum", "argnums"))
    def test_vjp_argnums(self, argnum_name):
        """Test that res."""

        def f(x, y):
            return y * x**2

        @qp.qjit
        def w(x, y):
            return qp.vjp(f, [x, y], [1.0], **{argnum_name: [1]})

        x = jnp.array(0.5)
        y = jnp.array(3.0)

        if argnum_name == "argnum":
            with pytest.warns(
                qp.exceptions.PennyLaneDeprecationWarning, match="argnum in qp.vjp"
            ):
                res, dres = w(x, y)
        else:
            res, dres = w(x, y)

        assert qp.math.allclose(res, f(x, y))
        assert qp.math.allclose(dres, x**2)

    def test_jvp_without_qjit(self):
        """Test that an error is raised when using JVP without QJIT."""

        def jvp(params, tangent):
            def f(x):
                y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
                return jnp.stack(y)

            return qp.jvp(f, [params], [tangent])

        x = jnp.array([0.1, 0.2])
        tangent = jnp.array([0.3, 0.6])

        with pytest.raises(
            CompileError, match="Pennylane does not support the JVP function without QJIT."
        ):
            jvp(x, tangent)

    def test_vjp(self):
        """Test that the correct VJP is returned with QJIT."""

        @qp.qjit
        def vjp(params, cotangent):
            def f(x):
                y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
                return jnp.stack(y)

            return qp.vjp(f, [params], [cotangent])

        x = jnp.array([0.1, 0.2])
        dy = jnp.array([-0.5, 0.1, 0.3])

        res = vjp(x, dy)
        assert len(res) == 2
        assert jnp.allclose(res[0], jnp.array([0.09983342, 0.04, 0.02]))
        assert jnp.allclose(res[1][0], jnp.array([-0.43750208, 0.07000001]))

    def test_vjp_without_qjit(self):
        """Test that an error is raised when using VJP without QJIT."""

        def vjp(params, cotangent):
            def f(x):
                y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
                return jnp.stack(y)

            return qp.vjp(f, [params], [cotangent])

        x = jnp.array([0.1, 0.2])
        dy = jnp.array([-0.5, 0.1, 0.3])

        with pytest.raises(
            CompileError, match="Pennylane does not support the VJP function without QJIT."
        ):
            vjp(x, dy)


class TestCatalystSample:
    """Test qp.sample with Catalyst."""

    def test_sample_measure(self):
        """Test that qp.sample can be used with catalyst.measure."""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.set_shots(1)
        @qp.qnode(dev)
        def circuit(x):
            qp.RY(x, wires=0)
            m = catalyst.measure(0)
            qp.PauliX(0)
            return qp.sample(m)

        assert circuit(0.0) == 0
        assert circuit(jnp.pi) == 1


class TestCatalystMCMs:
    """Test dynamic_one_shot with Catalyst."""

    @pytest.mark.parametrize(
        "measure_f",
        [
            qp.counts,
            qp.expval,
            qp.probs,
        ],
    )
    @pytest.mark.parametrize("meas_obj", [qp.PauliZ(0), [0], "mcm"])
    def test_dynamic_one_shot_simple(self, measure_f, meas_obj, seed):
        """Tests that Catalyst yields the same results as PennyLane's DefaultQubit for a simple
        circuit with a mid-circuit measurement."""

        if measure_f in (qp.counts, qp.probs, qp.sample) and isinstance(meas_obj, qp.PauliZ):
            pytest.skip("Can't use observables with counts, probs or sample")

        if measure_f in (qp.var, qp.expval) and (isinstance(meas_obj, list)):
            pytest.skip("Can't use wires/mcm lists with var or expval")

        if measure_f == qp.var and (not isinstance(meas_obj, list) and not meas_obj == "mcm"):
            pytest.xfail("isa<UnrealizedConversionCastOp>")

        shots = 8000

        dq = qp.device("default.qubit", seed=seed)

        @qp.defer_measurements
        @qp.set_shots(shots)
        @qp.qnode(dq)
        def ref_func(x, y):
            qp.RX(x, wires=0)
            m0 = qp.measure(0)
            qp.cond(m0, qp.RY)(y, wires=1)

            meas_key = "wires" if isinstance(meas_obj, list) else "op"
            meas_value = m0 if isinstance(meas_obj, str) else meas_obj
            kwargs = {meas_key: meas_value}
            if measure_f is qp.counts:
                kwargs["all_outcomes"] = True
            return measure_f(**kwargs)

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qjit
        @qp.set_shots(shots)
        @qp.qnode(dev, mcm_method="one-shot")
        def func(x, y):
            qp.RX(x, wires=0)
            m0 = catalyst.measure(0)

            @catalyst.cond(m0 == 1)
            def ansatz():
                qp.RY(y, wires=1)

            ansatz()

            meas_key = "wires" if isinstance(meas_obj, list) else "op"
            meas_value = m0 if isinstance(meas_obj, str) else meas_obj
            kwargs = {meas_key: meas_value}
            if measure_f is qp.counts:
                kwargs["all_outcomes"] = True
            return measure_f(**kwargs)

        params = jnp.pi / 4 * jnp.ones(2)
        results0 = ref_func(*params)
        results1 = func(*params)
        if measure_f is qp.counts:

            def fname(x):
                return format(x, f"0{len(meas_obj)}b") if isinstance(meas_obj, list) else x

            results1 = {fname(int(state)): count for state, count in zip(*results1)}
        if measure_f is qp.sample:
            results0 = results0[results0 != fill_in_value]
            results1 = results1[results1 != fill_in_value]
        mcm_utils.validate_measurements(measure_f, shots, results1, results0)
