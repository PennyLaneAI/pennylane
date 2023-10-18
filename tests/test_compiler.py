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

import pytest

import pennylane as qml
from jax.core import ShapedArray
from jax import numpy as jnp
import numpy as np


pytest.importorskip("catalyst")

# pylint: disable=too-few-public-methods, too-many-public-methods


class TestCatalyst:
    """Test ``qml.qjit`` with Catalyst"""

    def test_qjit_doc(self):
        """Test qjit docstring"""
        qml_qjit_doc = str(qml.qjit.__doc__)
        cat_qjit_doc_header = "just-in-time decorator for PennyLane and JAX programs using Catalyst"
        assert cat_qjit_doc_header in qml_qjit_doc

    def test_compiler(self):
        """Test compiler active and available methods"""

        with pytest.raises(RuntimeError, match="There is no available compiler package"):
            qml.Compiler.active()

        assert not qml.Compiler.available_backends()
        assert qml.Compiler.available()
        assert qml.Compiler.available_backends() == ["pennylane-catalyst"]

        assert qml.Compiler.available()
        assert qml.Compiler.available_backends() == ["pennylane-catalyst"]

    def test_qjit_cost_fn(self):
        """Test JIT compilation of a simple function"""
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qjit
        def cost_fn(x: float):
            res = circuit(x)
            return res

        assert np.allclose(cost_fn(2.0), -0.41614684)

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

        assert np.allclose(circuit(0.5), 0.0)

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
        assert np.allclose(result, expected)

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

        assert np.allclose(circuit(3), 0.0)
        assert np.allclose(circuit(5), 1.0)

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
