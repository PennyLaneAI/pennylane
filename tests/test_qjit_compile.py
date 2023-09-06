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
Unit tests for the qjit-compile support.
"""

import pytest

import pennylane as qml
import numpy as np

try:
    import catalyst
except ImportError:
    pytest.skip(
        "skipping qjit-compile tests because ``pennylane-catalyst`` is not installed",
        allow_module_level=True,
    )

# pylint: disable=too-few-public-methods, too-many-public-methods

class TestCatalystJAXFrontend:
    """Test ``catalyst.qjit`` with JAX interface"""

    def test_qjit_doc(self):
        """Test qjit docstring"""
        qml_qjit_doc = str(qml.qjit.__doc__)
        cat_qjit_doc_header = "just-in-time decorator for PennyLane and JAX programs using Catalyst"
        assert cat_qjit_doc_header in qml_qjit_doc

    def test_compilation_opt(self):
        """Test user-configurable compilation options"""
        dev = qml.device('lightning.qubit', wires=2)

        @qml.qjit(target="mlir")
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            qml.RX(x**2, wires=1)
            return qml.expval(qml.PauliZ(0))

        mlir_str = str(circuit.mlir)
        result_header = "func.func private @circuit(%arg0: tensor<f64>) -> tensor<f64>"
        assert result_header in mlir_str

    def test_for_loop(self):
        """Test Catalyst control-flow statement (``qml.for_loop``)"""
        dev = qml.device('lightning.qubit', wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(n: int):
            @qml.for_loop(0, n, 1)
            def loop_fn(_, x):
                qml.RY(x, wires=0)
                return x + np.pi / 4

            loop_fn(0.0)
            return qml.expval(qml.PauliZ(0))

        expected = np.array(-0.7071067811865489, dtype=float)
        assert np.allclose(expected, circuit(10))

    def test_for_loop(self):
        """Test Catalyst control-flow statement (``qml.if_cond``)"""
        dev = qml.device('lightning.qubit', wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x):
            @qml.if_cond(x > 4.8)
            def cond_fn():
                return x * 8

            @cond_fn.else_if(x > 2.7)
            def cond_elif():
                return x * 4

            @cond_fn.else_if(x > 1.4)
            def cond_elif2():
                return x * 2

            @cond_fn.otherwise
            def cond_else():
                return x

            return cond_fn()

        assert circuit(5) == 40

    def test_measure_with_reset(self):
        """Test mid-circuit measurement with ``reset=True``"""
        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def result():
            qml.PauliX(1)
            m_0 = qml.measure(1, reset=True)
            return qml.probs(wires=[1])

        with pytest.raises(AssertionError, match="reset option is not supported in Catalyst"):
            qml.qjit(result)()

    def test_measure_1(self):
        """Test mid-circuit measurement"""
        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def result():
            qml.PauliX(1)
            m_0 = qml.measure(1)
            return qml.probs(wires=[1])

        py_result = result()
        jit_result = qml.qjit(result)()
        assert np.allclose(py_result, jit_result)

    def test_measure_2(self):
        """Test mid-circuit measurement in a circuit with Catalyst support"""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x: float):
            qml.RX(x, wires=0)
            m1 = qml.measure(wires=0)
            maybe_pi = m1 * np.pi
            qml.RX(maybe_pi, wires=1)
            m2 = qml.measure(wires=1)
            return m2

        assert circuit(np.pi)
        assert not circuit(0.0)

    def test_adjoint_lazy(self):
        """Test adjoint with none default value for ``lazy``"""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(theta):
            qml.adjoint(qml.RZ, lazy=False)(theta, wires=0)

        with pytest.raises(AssertionError, match="Lazy Evaluation is not supported in Catalyst"):
            qml.qjit(circuit)(np.pi/2)

    def test_adjoint(self):
        """Test adjoint"""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def workflow(theta):
            qml.adjoint(qml.RZ)(theta, wires=0)
            qml.adjoint(qml.RZ(theta, wires=0))
            def func():
                qml.RX(theta, wires=0)
                qml.RY(theta, wires=0)
            qml.adjoint(func)()
            return qml.probs()

        workflow(np.pi/2)

    def test_grad_1(self):
        """Test gradient"""
        dev = qml.device("lightning.qubit", wires=1)

        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        @qml.qjit
        def grad_fn(x: float):
            g = qml.qnode(dev)(f)
            h = qml.grad(g, argnum=0)
            return h(x)

        expected = np.array(-0.86482227)
        assert np.allclose(grad_fn(0.526), expected)

    def test_grad_2(self):
        """Test gradient with custom ``method`` and ``h``"""
        dev = qml.device("lightning.qubit", wires=1)

        def f(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        @qml.qjit
        def grad_fn(x: float):
            g = qml.qnode(dev)(f)
            h = qml.grad(g, argnum=0, method="fd", h=0.5)
            return h(x)

        expected = np.array(-0.70630957)
        assert np.allclose(grad_fn(0.526), expected)

    def test_jacobian_1(self):
        """Test jacobian"""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def func(p):
            qml.RY(p, wires=0)
            return qml.probs(wires=0)

        @qml.qjit
        def grad_fn(p: float):
            return qml.jacobian(func)(p)

        expected = np.array([-0.25103906,  0.25103906])
        assert np.allclose(grad_fn(0.526), expected)

    def test_jacobian_2(self):
        """Test jacobian with custom ``method`` and ``h``"""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def func(p):
            qml.RY(p, wires=0)
            return qml.probs(wires=0)

        @qml.qjit
        def grad_fn(p: float):
            return qml.jacobian(func, method="fd", h=0.5)(p)

        expected = np.array([-0.34657838,  0.34657838])
        assert np.allclose(grad_fn(0.526), expected)
