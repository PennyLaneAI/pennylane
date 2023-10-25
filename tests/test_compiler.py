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
TODO: Uncomment 'pytest.mark.external' to check these tests in GitHub actions with
    the 'pennylane-catalyst' v0.3.2 release. These tests require the installation
    of Catalyst from the main branch at the moment.
"""
# pylint: disable=import-outside-toplevel
import pytest
import pennylane as qml

from pennylane import numpy as np

catalyst = pytest.importorskip("catalyst")
jax = pytest.importorskip("jax")

# pytestmark = pytest.mark.external

from jax import numpy as jnp  # pylint:disable=wrong-import-order, wrong-import-position
from jax.core import ShapedArray  # pylint:disable=wrong-import-order, wrong-import-position

# pylint: disable=too-few-public-methods, too-many-public-methods


class TestCatalyst:
    """Test ``qml.qjit`` with Catalyst"""

    def test_compiler(self):
        """Test compiler active and available methods"""

        assert not qml.compiler.active()

        assert qml.compiler.available("catalyst")
        assert qml.compiler.available_compilers() == ["catalyst"]

        assert qml.compiler.available("catalyst")
        assert qml.compiler.available_compilers() == ["catalyst"]

    def test_active_compiler(self):
        """Test `qml.compiler.active_compiler` inside a simple circuit"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            if qml.compiler.active_compiler() == "catalyst":
                qml.RX(phi, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(jnp.pi, jnp.pi / 2), 1.0)
        assert jnp.allclose(qml.qjit(circuit)(jnp.pi, jnp.pi / 2), -1.0)

    def test_active(self):
        """Test `qml.compiler.active` inside a simple circuit"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            if qml.compiler.active():
                qml.RX(phi, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(jnp.pi, jnp.pi / 2), 1.0)
        assert jnp.allclose(qml.qjit(circuit)(jnp.pi, jnp.pi / 2), -1.0)

    def test_qjit_circuit(self):
        """Test JIT compilation of a circuit with 2-qubit"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=1))

        assert jnp.allclose(circuit(0.5), 0.0)

    def test_qjit_aot(self):
        """Test AOT compilation of a circuit with 2-qubit"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit  # compilation happens at definition
        @qml.qnode(dev)
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

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
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit()
        def workflow(n: int):
            @qml.qnode(dev)
            def f(x: float):
                qml.RX(n * x, wires=n)
                return qml.expval(qml.PauliZ(wires=n))

            @qml.qnode(dev)
            def g(x: float):
                qml.RX(x, wires=1)
                return qml.expval(qml.PauliZ(wires=1))

            return jnp.array_equal(f(jnp.pi), g(jnp.pi))

        assert workflow(_in) == _out

    def test_args_workflow(self):
        """Test arguments with workflows."""

        @qml.qjit
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
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit1(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=1)
            return {
                "w0": qml.expval(qml.PauliZ(0)),
                "w1": qml.expval(qml.PauliZ(1)),
            }

        jitted_fn = qml.qjit(circuit1)

        params = [0.2, 0.6]
        expected = {"w0": 0.98006658, "w1": 0.82533561}
        result = jitted_fn(params)
        assert isinstance(result, dict)
        assert jnp.allclose(result["w0"], expected["w0"])
        assert jnp.allclose(result["w1"], expected["w1"])

    def test_qjit_python_if(self):
        """Test JIT compilation with the autograph support"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(autograph=True)
        @qml.qnode(dev)
        def circuit(x: int):
            if x < 5:
                qml.Hadamard(wires=0)
            else:
                qml.T(wires=0)

            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(3), 0.0)
        assert jnp.allclose(circuit(5), 1.0)

    def test_compilation_opt(self):
        """Test user-configurable compilation options"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir")
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x**2, wires=1)
            return qml.expval(qml.PauliZ(0))

        mlir_str = str(circuit.mlir)
        result_header = "func.func private @circuit(%arg0: tensor<f64>) -> tensor<f64>"
        assert result_header in mlir_str

    def test_grad_classical_preprocessing(self):
        """Test the grad transformation with classical preprocessing."""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(jnp.pi * x, wires=0)
                return qml.expval(qml.PauliY(0))

            g = qml.grad(circuit)
            return g(x)

        assert jnp.allclose(workflow(2.0), -jnp.pi)

    def test_grad_with_postprocessing(self):
        """Test the grad transformation with classical preprocessing and postprocessing."""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        def workflow(theta):
            @qml.qnode(dev, diff_method="adjoint")
            def circuit(theta):
                qml.RX(jnp.exp(theta**2) / jnp.cos(theta / 4), wires=0)
                return qml.expval(qml.PauliZ(wires=0))

            def loss(theta):
                return jnp.pi / jnp.tanh(circuit(theta))

            return qml.grad(loss, method="auto")(theta)

        assert jnp.allclose(workflow(1.0), 5.04324559)

    def test_grad_with_multiple_qnodes(self):
        """Test the grad transformation with multiple QNodes with their own differentiation methods."""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        def workflow(theta):
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit_A(params):
                qml.RX(jnp.exp(params[0] ** 2) / jnp.cos(params[1] / 4), wires=0)
                return qml.probs()

            @qml.qnode(dev, diff_method="adjoint")
            def circuit_B(params):
                qml.RX(jnp.exp(params[1] ** 2) / jnp.cos(params[0] / 4), wires=0)
                return qml.expval(qml.PauliZ(wires=0))

            def loss(params):
                return jnp.prod(circuit_A(params)) + circuit_B(params)

            return qml.grad(loss)(theta)

        result = workflow(jnp.array([1.0, 2.0]))
        reference = jnp.array([0.57367285, 44.4911605])

        assert jnp.allclose(result, reference)

    def test_grad_with_pure_classical(self):
        """Test the grad transformation with purely classical functions."""

        def square(x: float):
            return x**2

        @qml.qjit
        def dsquare(x: float):
            return catalyst.grad(square)(x)

        assert jnp.allclose(dsquare(2.3), 4.6)

    def test_jacobian_diff_method(self):
        """Test the Jacobian transformation with the device diff_method."""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev, diff_method="parameter-shift")
        def func(p):
            qml.RY(p, wires=0)
            return qml.probs(wires=0)

        @qml.qjit
        def workflow(p: float):
            return qml.jacobian(func, method="auto")(p)

        result = workflow(0.5)
        reference = qml.jacobian(func, argnum=0)(0.5)

        assert jnp.allclose(result, reference)

    def test_jacobian_auto(self):
        """Test the Jacobian transformation with 'auto'."""
        dev = qml.device("lightning.qubit", wires=1)

        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1], wires=0)
                return qml.probs()

            g = qml.jacobian(circuit)
            return g(x)

        reference = workflow(np.array([2.0, 1.0]))
        result = qml.qjit(workflow)(jnp.array([2.0, 1.0]))

        assert jnp.allclose(result, reference)

    def test_jacobian_fd(self):
        """Test the Jacobian transformation with 'fd'."""
        dev = qml.device("lightning.qubit", wires=1)

        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(np.pi * x[0], wires=0)
                qml.RY(x[1], wires=0)
                return qml.probs()

            g = qml.jacobian(circuit, method="fd", step_size=0.3)
            return g(x)

        result = qml.qjit(workflow)(np.array([2.0, 1.0]))
        print(result)

        reference = np.array([[-0.37120096, -0.45467246], [0.37120096, 0.45467246]])
        print(jnp.allclose(result, reference))

        with pytest.raises(
            ValueError,
            match="invalid values for 'method' and 'step_size' arguments in interpreter mode",
        ):
            workflow(np.array([2.0, 1.0]))
