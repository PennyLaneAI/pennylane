# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test module for the split non-commuting transform"""
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    SplitNonCommutingPass,
    split_non_commuting_pass,
)


class TestSplitNonCommutingPass:
    """Unit tests for SplitNonCommutingPass."""

    def test_func_w_qnode_attr(self, run_filecheck):
        """Test split non-commuting pass would be applied to a func with a qnode attribute."""
        program = """
            module @module_circuit {
                func.func public @circuit() -> tensor<f64> attributes {qnode} {
                    // CHECK-NOT: arith.constant
                    // CHECK: [[value:%.+]] = func.call [[dup_func:@[a-zA-Z0-9_.]+]]
                    // CHECK: func.return [[value]]
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc( 50) : !quantum.reg
                    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
                // CHECK: func.func [[dup_func]]
            }
        """

        pipeline = (SplitNonCommutingPass(),)
        run_filecheck(program, pipeline)

    def test_multiple_func_w_qnode_attr(self, run_filecheck):
        """Test split non-commuting pass would be applied to a func with a qnode attribute."""
        program = """
            module @module_circuit {
                func.func public @circuit1() -> tensor<f64> attributes {qnode} {
                    // CHECK-NOT: arith.constant
                    // CHECK: [[value:%.+]] = func.call [[dup_func1:@[a-zA-Z0-9_.]+]]
                    // CHECK: func.return [[value]]
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc( 50) : !quantum.reg
                    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
                func.func public @circuit2() -> tensor<f64> attributes {qnode} {
                    // CHECK-NOT: arith.constant
                    // CHECK: [[value:%.+]] = func.call [[dup_func2:@[a-zA-Z0-9_.]+]]
                    // CHECK: func.return [[value]]
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc( 50) : !quantum.reg
                    %2 = quantum.extract %1[ 0] : !quantum.reg -> !quantum.bit
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
                // CHECK: func.func [[dup_func1]]
                // CHECK: func.func [[dup_func2]]
            }
        """

        pipeline = (SplitNonCommutingPass(),)
        run_filecheck(program, pipeline)

    def test_func_w_commuting_measurements(self, run_filecheck):
        """Test split non-commuting pass would be applied to a func with commuting ops."""
        program = """
            module @module_circuit {
                func.func public @circuit() -> (tensor<f64>, tensor<f64>) attributes {qnode} {
                    // CHECK-NOT: arith.constant
                    // CHECK: [[v0:%.+]], [[v1:%.+]] = func.call [[dup_func:@[a-zA-Z0-9_.]+]]
                    // CHECK: func.return [[v0]], [[v1]]
                    %c0 = arith.constant 0 : i64
                    quantum.device shots(%c0) ["", "", ""]
                    %0 = quantum.alloc( 5) : !quantum.reg
                    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
                    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
                    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
                    %out_qubits_0 = quantum.custom "Hadamard"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.namedobs %out_qubits_0[ PauliZ] : !quantum.obs
                    %6 = quantum.expval %5 : f64
                    %from_elements_1 = tensor.from_elements %6 : tensor<f64>
                    %7 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    %8 = quantum.insert %7[ 1], %out_qubits_0 : !quantum.reg, !quantum.bit
                    quantum.dealloc %8 : !quantum.reg
                    quantum.device_release
                    return %from_elements, %from_elements_1 : tensor<f64>, tensor<f64>
                }
                // CHECK: func.func [[dup_func]]
            }
        """

        pipeline = (SplitNonCommutingPass(),)
        run_filecheck(program, pipeline)

    def test_func_w_non_commuting_measurements(self, run_filecheck):
        """Test split non-commuting pass would be applied to a func with non-commuting ops."""
        program = """
            module @module_circuit {
                func.func public @circuit() -> (tensor<f64>, tensor<f64>) attributes {qnode} {
                    // CHECK-NOT: arith.constant
                    // CHECK: [[v0:%.+]] = func.call [[dup_func0:@[a-zA-Z0-9_.]+]]
                    // CHECK: [[v1:%.+]] = func.call [[dup_func1:@[a-zA-Z0-9_.]+]]
                    // CHECK: func.return [[v0]], [[v1]]
                    %c0 = arith.constant 0 : i64
                    quantum.device shots(%c0) ["", "", ""]
                    %0 = quantum.alloc( 5) : !quantum.reg
                    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
                    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
                    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
                    %out_qubits_0 = quantum.custom "Hadamard"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
                    %6 = quantum.expval %5 : f64
                    %from_elements_1 = tensor.from_elements %6 : tensor<f64>
                    %7 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    %8 = quantum.insert %7[ 1], %out_qubits_0 : !quantum.reg, !quantum.bit
                    quantum.dealloc %8 : !quantum.reg
                    quantum.device_release
                    return %from_elements, %from_elements_1 : tensor<f64>, tensor<f64>
                }
                // CHECK: func.func [[dup_func0]]
                // CHECK: PauliX
                // CHECK: func.func [[dup_func1]]
                // CHECK: PauliZ
            }
        """

        pipeline = (SplitNonCommutingPass(),)
        run_filecheck(program, pipeline)

    def test_func_w_mixed_measurements(self, run_filecheck):
        """Test split non-commuting pass would be applied to a func with mixed ops."""
        program = """
            module @module_circuit {
                func.func public @circuit() -> (tensor<f64>, tensor<f64>) attributes {qnode} {
                    // CHECK-NOT: arith.constant
                    // CHECK: [[v0:%.+]], [[v1:%.+]] = func.call [[dup_func0:@[a-zA-Z0-9_.]+]]
                    // CHECK: [[v2:%.+]] = func.call [[dup_func1:@[a-zA-Z0-9_.]+]]
                    // CHECK: func.return [[v0]], [[v2]], [[v1]]
                    %c0 = arith.constant 0 : i64
                    quantum.device shots(%c0) ["", "", ""]
                    %0 = quantum.alloc( 5) : !quantum.reg
                    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
                    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
                    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
                    %out_qubits_0 = quantum.custom "Hadamard"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[ PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
                    %6 = quantum.expval %5 : f64
                    %from_elements_1 = tensor.from_elements %6 : tensor<f64>
                    %7 = quantum.namedobs %out_qubits_0[ PauliY] : !quantum.obs
                    %8 = quantum.expval %7 : f64
                    %from_elements_2 = tensor.from_elements %8 : tensor<f64>
                    %9 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
                    %10 = quantum.insert %9[ 1], %out_qubits_0 : !quantum.reg, !quantum.bit
                    quantum.dealloc %10 : !quantum.reg
                    quantum.device_release
                    return %from_elements, %from_elements_1, %from_elements_2 : tensor<f64>, tensor<f64>, tensor<f64>
                }
                // CHECK: func.func [[dup_func0]]
                // CHECK: PauliX
                // CHECK: PauliY
                // CHECK: func.func [[dup_func1]]
                // CHECK: PauliZ
            }
        """

        pipeline = (SplitNonCommutingPass(),)
        run_filecheck(program, pipeline)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_split_non_commuting_pass_only(self, run_filecheck_qjit):
        """Test the split non-commuting pass only."""
        dev = qml.device("lightning.qubit", wires=5)

        @qml.while_loop(lambda i: i < 5)
        def _while_for(i):
            qml.H(i)
            i = i + 1
            return i

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @split_non_commuting_pass
        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            # CHECK-LABEL: func.func public @circuit()
            # CHECK: [[v0:%.+]], [[v1:%.+]] = func.call [[dup_func:@[a-zA-Z0-9_.]+]]
            # CHECK: [[v2:%.+]] = func.call [[dup_func1:@[a-zA-Z0-9_.]+]]
            # CHECK: func.return [[v0]], [[v1]], [[v2]]
            _while_for(0)
            qml.CNOT(wires=[0, 1])
            return (
                qml.expval(qml.Z(wires=0)),
                qml.expval(qml.Y(wires=1)),
                qml.expval(qml.X(wires=0)),
            )
            # CHECK: func.func [[dup_func]]
            # CHECK: PauliZ
            # CHECK: PauliY
            # CHECK: func.func [[dup_func1]]
            # CHECK: PauliX

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_lightning_execution_with_structure(self):
        """Test that the split non-commuting pass on lightning.qubit for a circuit with program structure is executable and returns results as expected."""
        dev = qml.device("lightning.qubit", wires=10)

        @qml.for_loop(0, 10, 1)
        def for_fn(i):
            qml.H(i)
            qml.S(i)
            qml.RZ(phi=0.1, wires=[i])

        @qml.while_loop(lambda i: i < 10)
        def while_fn(i):
            qml.H(i)
            qml.S(i)
            qml.RZ(phi=0.1, wires=[i])
            i = i + 1
            return i

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @split_non_commuting_pass
        @qml.qnode(dev)
        def circuit():
            for_fn()
            while_fn(0)
            qml.CNOT(wires=[0, 1])
            return (
                qml.expval(qml.Z(wires=0)),
                qml.expval(qml.Y(wires=1)),
                qml.expval(qml.X(wires=0)),
            )

        res = circuit()

        @qml.qjit(
            target="mlir",
        )
        @qml.qnode(dev)
        def circuit_ref():
            for_fn()
            while_fn(0)
            qml.CNOT(wires=[0, 1])
            return (
                qml.expval(qml.Z(wires=0)),
                qml.expval(qml.Y(wires=1)),
                qml.expval(qml.X(wires=0)),
            )

        res_ref = circuit_ref()
        assert res == res_ref
