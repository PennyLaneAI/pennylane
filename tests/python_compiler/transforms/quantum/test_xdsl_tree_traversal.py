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
"""Unit test module for the tree-traversal transform"""
import mcm_utils
import numpy as np
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import TreeTraversalPass, tree_traversal_pass

"""
Not supported features:
    - mcm postselection
    - mcm reset
    - return qml.state()
    - return qml.probs()
    - return qml.sample()
    - return multiple measurements, e.g.
        return qml.expval(Z(0)), qml.expval(X(1))
    - qml.expval(mcm_result)
"""


class TestTreeTraversalPassBase:
    """Unit tests for TreeTraversalPass. without control flow operation."""

    def test_func_no_mcm(self, run_filecheck):
        """Test tree traversal pass would not be applied to a func without MCMs."""

        """
        Circuit Test
        def c():
            qml.X(0)
            return qml.expval(qml.Z(0))
        """

        program = """
            builtin.module @module_circuit {
                func.func public @circuit() -> tensor<f64> attributes {qnode} {
                    // CHECK: quantum.device shots
                    // CHECK: quantum.alloc(50) : !quantum.reg
                    // CHECK: quantum.custom "PauliX"
                    // CHECK: quantum.expval
                    // CHECK: func.return
                    %0 = arith.constant 0 : i64
                    quantum.device shots(%0) ["", "", ""]
                    %1 = quantum.alloc(50) : !quantum.reg
                    %2 = quantum.extract %1[0] : !quantum.reg -> !quantum.bit
                    %out_qubits = quantum.custom "PauliX"() %2 : !quantum.bit
                    %3 = quantum.namedobs %out_qubits[PauliX] : !quantum.obs
                    %4 = quantum.expval %3 : f64
                    %from_elements = tensor.from_elements %4 : tensor<f64>
                    %5 = quantum.insert %1[0], %out_qubits : !quantum.reg, !quantum.bit
                    quantum.dealloc %5 : !quantum.reg
                    quantum.device_release
                    return %from_elements : tensor<f64>
                }
            // CHECK-NOT: func.func public @circuit.simple_io.tree_traversal()
            // CHECK-NOT: func.func @state_transition

            }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_single_mcm(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with a single MCM."""

        """
        Circuit Test
        def c():
            qml.X(0)
            qml.measure(1)
            qml.Y(1)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> tensor<f64> attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = tensor.extract %0[] : tensor<i64>
                %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                %7, %8 = quantum.measure %6 : i1, !quantum.bit
                %9 = tensor.from_elements %7 : tensor<i1>
                %10 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %11 = tensor.extract %10[] : tensor<i64>
                %12 = quantum.extract %3[%11] : !quantum.reg -> !quantum.bit
                %13 = quantum.custom "PauliY"() %12 : !quantum.bit
                %14 = quantum.namedobs %8[PauliZ] : !quantum.obs
                %15 = quantum.expval %14 : f64
                %16 = tensor.from_elements %15 : tensor<f64>
                %17 = tensor.extract %0[] : tensor<i64>
                %18 = quantum.insert %3[%17], %8 : !quantum.reg, !quantum.bit
                %19 = tensor.extract %10[] : tensor<i64>
                %20 = quantum.insert %18[%19], %13 : !quantum.reg, !quantum.bit
                quantum.dealloc %20 : !quantum.reg
                quantum.device_release
                func.return %16 : tensor<f64>
            }
            // CHECK: func.func public @circuit.simple_io.tree_traversal()
            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: func.func @quantum_segment_1
            // CHECK-NOT: func.func @quantum_segment_2
            // CHECK-NOT: func.func @quantum_segment_3
            // CHECK: func.func @segment_table
            // CHECK: case 0 {
            // CHECK: case 1 {
            // CHECK-NOT: case 2 {
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)

    def test_two_mcm(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with two MCMs."""

        """
        Circuit Test
        def c():
            qml.X(0)
            qml.measure(0)
            qml.Y(1)
            qml.measure(1)
            qml.Z(0)
            return qml.expval(qml.Z(0))
        """
        program = """

        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = tensor.extract %0[] : tensor<i64>
                %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                %7, %8 = quantum.measure %6 : i1, !quantum.bit
                %9 = tensor.from_elements %7 : tensor<i1>
                %10 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %11 = tensor.extract %10[] : tensor<i64>
                %12 = quantum.extract %3[%11] : !quantum.reg -> !quantum.bit
                %13 = quantum.custom "PauliY"() %12 : !quantum.bit
                %14, %15 = quantum.measure %13 : i1, !quantum.bit
                %16 = tensor.from_elements %14 : tensor<i1>
                %17 = quantum.custom "PauliZ"() %8 : !quantum.bit
                %18 = quantum.namedobs %17[PauliZ] : !quantum.obs
                %19 = quantum.expval %18 : f64
                %20 = tensor.from_elements %19 : tensor<f64>
                %21 = tensor.extract %0[] : tensor<i64>
                %22 = quantum.insert %3[%21], %17 : !quantum.reg, !quantum.bit
                %23 = tensor.extract %10[] : tensor<i64>
                %24 = quantum.insert %22[%23], %15 : !quantum.reg, !quantum.bit
                quantum.dealloc %24 : !quantum.reg
                quantum.device_release
                func.return %20 : tensor<f64>
            }

            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: func.func @quantum_segment_1
            // CHECK: func.func @quantum_segment_2
            // CHECK-NOT: func.func @quantum_segment_3
            // CHECK-NOT: func.func @quantum_segment_4
            // CHECK: func.func @segment_table
            // CHECK: case 0 {
            // CHECK: case 1 {
            // CHECK: case 2 {
            // CHECK-NOT: case 3 {
            // CHECK-NOT: case 4 {
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)

    def test_many_mcm(self, run_filecheck):
        """Test tree traversal pass with a function containing many MCM (mid-circuit measurement) operations."""

        """
        Circuit Test
        def c():
            qml.X(0)
            qml.measure(0)
            qml.measure(1)
            qml.measure(0)
            qml.measure(1)
            qml.measure(0)
            qml.Z(0)
            return qml.expval(qml.Z(0))
        """
        program = """

        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = tensor.extract %0[] : tensor<i64>
                %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                %7, %8 = quantum.measure %6 : i1, !quantum.bit
                %9 = tensor.from_elements %7 : tensor<i1>
                %10 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %11 = tensor.extract %10[] : tensor<i64>
                %12 = quantum.extract %3[%11] : !quantum.reg -> !quantum.bit
                %13, %14 = quantum.measure %12 : i1, !quantum.bit
                %15 = tensor.from_elements %13 : tensor<i1>
                %16, %17 = quantum.measure %8 : i1, !quantum.bit
                %18 = tensor.from_elements %16 : tensor<i1>
                %19, %20 = quantum.measure %14 : i1, !quantum.bit
                %21 = tensor.from_elements %19 : tensor<i1>
                %22, %23 = quantum.measure %17 : i1, !quantum.bit
                %24 = tensor.from_elements %22 : tensor<i1>
                %25 = quantum.custom "PauliZ"() %23 : !quantum.bit
                %26 = quantum.namedobs %25[PauliZ] : !quantum.obs
                %27 = quantum.expval %26 : f64
                %28 = tensor.from_elements %27 : tensor<f64>
                %29 = tensor.extract %0[] : tensor<i64>
                %30 = quantum.insert %3[%29], %25 : !quantum.reg, !quantum.bit
                %31 = tensor.extract %10[] : tensor<i64>
                %32 = quantum.insert %30[%31], %20 : !quantum.reg, !quantum.bit
                quantum.dealloc %32 : !quantum.reg
                quantum.device_release
                func.return %28 : tensor<f64>
            }

            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: func.func @quantum_segment_1
            // CHECK: func.func @quantum_segment_2
            // CHECK: func.func @quantum_segment_3
            // CHECK: func.func @quantum_segment_4
            // CHECK: func.func @quantum_segment_5
            // CHECK-NOT: func.func @quantum_segment_6
            // CHECK-NOT: func.func @quantum_segment_7
            // CHECK: func.func @segment_table
            // CHECK: case 0 {
            // CHECK: case 1 {
            // CHECK: case 2 {
            // CHECK: case 3 {
            // CHECK: case 4 {
            // CHECK: case 5 {
            // CHECK-NOT: case 6 {
            // CHECK-NOT: case 7 {

        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)

    def test_gates_on_mcm_segments(self, run_filecheck):
        """Test tree traversal pass to check if the gates are applied on the correct segments."""

        """
        Circuit Test
        def c():
            qml.X(0)
            qml.measure(0)
            qml.S(0)
            qml.measure(0)
            qml.T(0)

            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = tensor.extract %0[] : tensor<i64>
                %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                %7, %8 = quantum.measure %6 : i1, !quantum.bit
                %9 = tensor.from_elements %7 : tensor<i1>
                %10 = quantum.custom "S"() %8 : !quantum.bit
                %11, %12 = quantum.measure %10 : i1, !quantum.bit
                %13 = tensor.from_elements %11 : tensor<i1>
                %14 = quantum.custom "T"() %12 : !quantum.bit
                %15 = quantum.namedobs %14[PauliZ] : !quantum.obs
                %16 = quantum.expval %15 : f64
                %17 = tensor.from_elements %16 : tensor<f64>
                %18 = tensor.extract %0[] : tensor<i64>
                %19 = quantum.insert %3[%18], %14 : !quantum.reg, !quantum.bit
                quantum.dealloc %19 : !quantum.reg
                quantum.device_release
                func.return %17 : tensor<f64>
            }

            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: quantum.custom "PauliX"()
            // CHECK: func.func @quantum_segment_1
            // CHECK: quantum.custom "S"()
            // CHECK: func.func @quantum_segment_2
            // CHECK: quantum.custom "T"()
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)

    def test_simple_io_function(self, run_filecheck):
        """Test tree traversal pass with simple io structure."""

        """
        Circuit Test
        def c():
            qml.X(0)
            qml.measure(0)
            qml.Y(1)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = tensor.extract %0[] : tensor<i64>
                %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                %7, %8 = quantum.measure %6 : i1, !quantum.bit
                %9 = tensor.from_elements %7 : tensor<i1>
                %10 = quantum.custom "PauliY"() %8 : !quantum.bit
                %11 = quantum.namedobs %10[PauliZ] : !quantum.obs
                %12 = quantum.expval %11 : f64
                %13 = tensor.from_elements %12 : tensor<f64>
                %14 = tensor.extract %0[] : tensor<i64>
                %15 = quantum.insert %3[%14], %10 : !quantum.reg, !quantum.bit
                quantum.dealloc %15 : !quantum.reg
                quantum.device_release
                func.return %13 : tensor<f64>
            }

            // CHECK: func.func public @circuit.simple_io.tree_traversal()

            // CHECK: quantum.alloc(2) : !quantum.reg

            // Allocate memory for middle state-vectors
            // CHECK: memref.alloc({{%.+}}, {{%.+}}) : memref<?x?xcomplex<f64>
            // Allocate memory for visited array
            // CHECK: memref.alloc({{%.+}}) : memref<?xi8>

            // While loop for visiting all the segments
            // CHECK: = scf.while ({{.*}}) : (index, !quantum.reg, tensor<i64>, i64, tensor<f64>)
            // If statement for traversing
            // CHECK: = scf.if {{%.+}} -> (index, !quantum.reg,
            // Call to segment table
            // CHECK: = func.call @segment_table(

            // CHECK: quantum.dealloc {{%.+}} : !quantum.reg

            // Just single qubit [de]allocation
            // CHECK-NOT: quantum.alloc(2) : !quantum.reg
            // CHECK-NOT: quantum.dealloc {{%.+}} : !quantum.reg
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)

    def test_state_transition_function(self, run_filecheck):
        """Test tree traversal pass with simple io structure."""

        """
        Circuit Test
        def c():
            qml.X(0)
            qml.measure(0)
            qml.Y(1)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = tensor.extract %0[] : tensor<i64>
                %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                %7, %8 = quantum.measure %6 : i1, !quantum.bit
                %9 = tensor.from_elements %7 : tensor<i1>
                %10 = quantum.custom "PauliY"() %8 : !quantum.bit
                %11 = quantum.namedobs %10[PauliZ] : !quantum.obs
                %12 = quantum.expval %11 : f64
                %13 = tensor.from_elements %12 : tensor<f64>
                %14 = tensor.extract %0[] : tensor<i64>
                %15 = quantum.insert %3[%14], %10 : !quantum.reg, !quantum.bit
                quantum.dealloc %15 : !quantum.reg
                quantum.device_release
                func.return %13 : tensor<f64>
            }

            // CHECK: func.func @state_transition({{.*}}) -> (!quantum.reg, i8)
            // Fixed if statement structure for tree traversal
            // CHECK: = scf.if {{%.+}} -> (!quantum.reg, i8, i8) {
            // CHECK: = scf.if {{%.+}} -> (!quantum.reg, i8, i8) {
            // CHECK: = scf.if {{%.+}} -> (!quantum.reg, i8, i8) {

            // Update state-vector memory
            // CHECK: [[sv0:%.+]] = quantum.state {{%.+}} shape {{%.+}} : tensor<?xcomplex<f64>>
            // CHECK: [[sv_ptr0:%.+]] = tensor.extract [[sv0]][{{%.+}}] : tensor<?xcomplex<f64>>
            // CHECK: memref.store [[sv_ptr0]], {{%.+}}[{{%.+}}, {{%.+}}] : memref<?x?xcomplex<f64>>

            // Extract state-vector pointer
            // CHECK: [[q0:%.+]] = memref.subview {{%.+}}[{{%.+}}, 0] [1, {{%.+}}] [1, 1] : memref<?x?xcomplex<f64>> to memref<?xcomplex<f64>
            // Set the state vector
            // CHECK: quantum.set_state([[q0]]) {{%.+}}, {{%.+}} : (memref<?xcomplex<f64>, strided<[1], offset: ?>>, !quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            // Update visited array
            // CHECK: memref.store {{%.+}}, {{%.+}}[{{%.+}}] : memref<?xi8>
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)

    def test_segment_table_function(self, run_filecheck):
        """Test tree traversal pass with simple io structure."""

        """
        Circuit Test
        def c():
            qml.X(0)
            qml.measure(0)
            qml.Y(1)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = tensor.extract %0[] : tensor<i64>
                %5 = quantum.extract %3[%4] : !quantum.reg -> !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                %7, %8 = quantum.measure %6 : i1, !quantum.bit
                %9 = tensor.from_elements %7 : tensor<i1>
                %10 = quantum.custom "PauliY"() %8 : !quantum.bit
                %11 = quantum.namedobs %10[PauliZ] : !quantum.obs
                %12 = quantum.expval %11 : f64
                %13 = tensor.from_elements %12 : tensor<f64>
                %14 = tensor.extract %0[] : tensor<i64>
                %15 = quantum.insert %3[%14], %10 : !quantum.reg, !quantum.bit
                quantum.dealloc %15 : !quantum.reg
                quantum.device_release
                func.return %13 : tensor<f64>
            }

            // CHECK: func.func @segment_table({{.*}}) -> (!quantum.reg, tensor<i64>, i64, tensor<f64>)

            // Swtich Op
            // CHECK: scf.index_switch %0 -> !quantum.reg, tensor<i64>, i64, tensor<f64>

            // Check cases
            // CHECK: case 0 {
            // CHECK: = func.call @quantum_segment_0(
            // CHECK: case 1 {
            // CHECK: = func.call @quantum_segment_1(

        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_tree_traversal_pass_only(self, run_filecheck_qjit):
        """Test the tree traversal pass only."""
        dev = qml.device("lightning.qubit", wires=5)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @tree_traversal_pass
        @qml.qnode(dev)
        def circuit():
            # CHECK: func.func public @circuit() -> (tensor<f64>)
            # CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            # CHECK: func.return %0 : tensor<f64>
            qml.X(0)
            qml.measure(0)
            qml.Y(1)
            return qml.expval(qml.Z(0))
            # CHECK: func.func public @circuit.simple_io.tree_traversal()
            # CHECK: func.func @state_transition
            # CHECK: func.func @quantum_segment_0
            # CHECK: func.func @quantum_segment_1
            # CHECK-NOT: func.func @quantum_segment_2
            # CHECK-NOT: func.func @quantum_segment_3
            # CHECK: func.func @segment_table
            # CHECK: case 0 {
            # CHECK: case 1 {
            # CHECK-NOT: case 2 {

        run_filecheck_qjit(circuit)

    @pytest.mark.parametrize("shots", [None, 10000])
    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_result_validation(self, shots):

        dev = qml.device("lightning.qubit", wires=4, seed=42)

        def test_circuit():
            qml.RX(1.3, 0)
            qml.RX(1.5, 1)
            qml.measure(0)
            qml.RX(1.3, 0)
            qml.RX(1.5, 1)
            qml.measure(0)
            return qml.expval(qml.Z(0))

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @qml.set_shots(shots)
        @tree_traversal_pass
        @qml.qnode(dev)
        def circuit():
            return test_circuit()

        res = circuit()

        qml.capture.disable()

        dev = qml.device("lightning.qubit", wires=4, seed=42)

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="deferred")
        def circuit_ref():
            return test_circuit()

        res_ref = circuit_ref()

        mcm_utils.validate_measurements(qml.expval, shots, res_ref, res, batch_size=None)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [None, 10000])
    @pytest.mark.parametrize("postselect", [None])
    @pytest.mark.parametrize("reset", [False])
    @pytest.mark.parametrize(
        "measure_f",
        [
            lambda: qml.expval(qml.Z(0)),
            lambda: qml.expval(qml.Y(0)),
            lambda: qml.expval(qml.Z(1)),
            lambda: qml.expval(qml.Y(1)),
        ],
    )
    def test_result_validation_multiple_measurements(self, shots, postselect, reset, measure_f):
        """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
        and a conditional gate. Multiple measurements of the mid-circuit measurement value are
        performed. This function also tests `reset` parametrizing over the parameter."""

        dev = qml.device("lightning.qubit", wires=3)
        params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]

        def obs_tape(x, y, z, reset=False, postselect=None):

            for i in range(3):
                qml.measure(i, reset=reset)

            qml.RX(x, 0)
            qml.RZ(np.pi / 8, 0)

            qml.measure(0, reset=reset)

            qml.RX(np.pi / 4, 0)
            qml.RZ(np.pi / 4, 0)

            qml.measure(1, postselect=postselect)

            qml.RX(y, 1)
            qml.RZ(np.pi / 4, 1)
            qml.measure(1, postselect=postselect)
            qml.RX(-np.pi / 8, 1)
            qml.RZ(-np.pi / 8, 1)

        # Measures:
        # Without shots
        # qml.expval, qml.probs, qml.var
        # With shots
        # qml.expval, qml.count, qml.var, qml.sample

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @qml.set_shots(shots)
        @tree_traversal_pass
        @qml.qnode(dev)
        def qjit_func(x, y, z):
            obs_tape(x, y, z, reset=reset, postselect=postselect)
            return measure_f()

        results0 = qjit_func(*params)

        dev = qml.device("default.qubit")

        qml.capture.disable()

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="deferred")
        def ref_func(x, y, z):
            obs_tape(x, y, z, reset=reset, postselect=postselect)
            return measure_f()

        results1 = ref_func(*params)

        mcm_utils.validate_measurements(qml.expval, shots, results1, results0, batch_size=None)


class TestTreeTraversalPassIfStatement:
    """Unit tests for TreeTraversalPass. with if statement control flow operation."""

    def test_if_statement_with_mcm_true(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with an if statement."""

        """
        Circuit Test
        def circuit():
            def ansatz_true():
                m = qml.measure(1)
                qml.Z(1)
            def ansatz_false():
                qml.Y(1)
            qml.cond(True, ansatz_true, ansatz_false)()
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
                %5 = tensor.extract %4[] : tensor<i1>
                %6 = scf.if %5 -> (!quantum.reg) {
                    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %8 = tensor.extract %7[] : tensor<i64>
                    %9 = quantum.extract %3[%8] : !quantum.reg -> !quantum.bit
                    %10, %11 = quantum.measure %9 : i1, !quantum.bit
                    %12 = tensor.from_elements %10 : tensor<i1>
                    %13 = quantum.custom "PauliZ"() %11 : !quantum.bit
                    %14 = tensor.extract %7[] : tensor<i64>
                    %15 = quantum.insert %3[%14], %13 : !quantum.reg, !quantum.bit
                    scf.yield %15 : !quantum.reg
                } else {
                    %16 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %17 = tensor.extract %16[] : tensor<i64>
                    %18 = quantum.extract %3[%17] : !quantum.reg -> !quantum.bit
                    %19 = quantum.custom "PauliY"() %18 : !quantum.bit
                    %20 = tensor.extract %16[] : tensor<i64>
                    %21 = quantum.insert %3[%20], %19 : !quantum.reg, !quantum.bit
                    scf.yield %21 : !quantum.reg
                }
                %22 = tensor.extract %0[] : tensor<i64>
                %23 = quantum.extract %6[%22] : !quantum.reg -> !quantum.bit
                %24 = quantum.namedobs %23[PauliZ] : !quantum.obs
                %25 = quantum.expval %24 : f64
                %26 = tensor.from_elements %25 : tensor<f64>
                %27 = tensor.extract %0[] : tensor<i64>
                %28 = quantum.insert %6[%27], %23 : !quantum.reg, !quantum.bit
                quantum.dealloc %28 : !quantum.reg
                quantum.device_release
                func.return %26 : tensor<f64>
            }
            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0

            // CHECK: func.func @quantum_segment_1
                // Main if statement
                // CHECK: {{%.+}} = scf.if {{%.+}} -> (!quantum.reg) {
                    // If statement for mcm
                    // CHECK: {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (i1, !quantum.bit) {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit
                    // CHECK: } else {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit

                    // Check that mcm results are not leaked outside the mcm if statement
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: } else {
                    // CHECK-NEXT: scf.yield {{%.+}} : !quantum.reg

                // CHECK: }{contain_mcm = "true", partition = "true_branch"}
                // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                // CHECK: {{%.+}} = arith.xori {{%.+}}, {{%.+}} : i1
                // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit

            // CHECK: func.func @quantum_segment_2
                // CHECK: {{%.+}} = scf.if {{%.+}} -> (!quantum.reg) {
                    // CHECK-NOT: {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (i1, !quantum.bit) {
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: } else {
                    // CHECK: scf.yield {{%.+}} : !quantum.reg

                // CHECK: }{contain_mcm = "true", partition = "false_branch"}
                // CHECK-NOT: {{%.+}} = arith.xori {{%.+}}, {{%.+}} : i1
            // CHECK-NOT: func.func @quantum_segment_3
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_if_statement_with_mcm_false(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with an if statement."""

        """
        Circuit Test
        def circuit():
            def ansatz_true():
                qml.Z(1)
            def ansatz_false():
                m = qml.measure(1)
                qml.Y(1)
            qml.cond(True, ansatz_true, ansatz_false)()
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
                %5 = tensor.extract %4[] : tensor<i1>
                %6 = scf.if %5 -> (!quantum.reg) {
                    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %8 = tensor.extract %7[] : tensor<i64>
                    %9 = quantum.extract %3[%8] : !quantum.reg -> !quantum.bit
                    %10 = quantum.custom "PauliZ"() %9 : !quantum.bit
                    %11 = tensor.extract %7[] : tensor<i64>
                    %12 = quantum.insert %3[%11], %10 : !quantum.reg, !quantum.bit
                    scf.yield %12 : !quantum.reg
                } else {
                    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %14 = tensor.extract %13[] : tensor<i64>
                    %15 = quantum.extract %3[%14] : !quantum.reg -> !quantum.bit
                    %16, %17 = quantum.measure %15 : i1, !quantum.bit
                    %18 = tensor.from_elements %16 : tensor<i1>
                    %19 = quantum.custom "PauliY"() %17 : !quantum.bit
                    %20 = tensor.extract %13[] : tensor<i64>
                    %21 = quantum.insert %3[%20], %19 : !quantum.reg, !quantum.bit
                    scf.yield %21 : !quantum.reg
                }
                %22 = tensor.extract %0[] : tensor<i64>
                %23 = quantum.extract %6[%22] : !quantum.reg -> !quantum.bit
                %24 = quantum.namedobs %23[PauliZ] : !quantum.obs
                %25 = quantum.expval %24 : f64
                %26 = tensor.from_elements %25 : tensor<f64>
                %27 = tensor.extract %0[] : tensor<i64>
                %28 = quantum.insert %6[%27], %23 : !quantum.reg, !quantum.bit
                quantum.dealloc %28 : !quantum.reg
                quantum.device_release
                func.return %26 : tensor<f64>
            }
            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: func.func @quantum_segment_1
            // CHECK: {{%.+}} = scf.if {{%.+}} -> (!quantum.reg) {
            // CHECK-NOT: {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (i1, !quantum.bit) {
            // CHECK: scf.yield {{%.+}} : !quantum.reg
            // CHECK: } else {
            // CHECK: scf.yield {{%.+}} : !quantum.reg
            // CHECK: }{contain_mcm = "true", partition = "true_branch"}
            // CHECK: {{%.+}} = arith.xori {{%.+}}, {{%.+}} : i1

            // CHECK: func.func @quantum_segment_2
                // Main if statement
                // CHECK: {{%.+}} = scf.if {{%.+}} -> (!quantum.reg) {
                    // If statement for mcm
                    // CHECK: {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (i1, !quantum.bit) {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit
                    // CHECK: } else {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit

                    // Check that mcm results are not leaked outside the mcm if statement
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: } else {
                    // CHECK-NEXT: scf.yield {{%.+}} : !quantum.reg

                // CHECK: }{contain_mcm = "true", partition = "false_branch"}
                // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit

            // CHECK-NOT: func.func @quantum_segment_3
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_if_statement_with_mcm_both(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with an if statement."""

        """
        Circuit Test
        def circuit():
            def ansatz_true():
                m = qml.measure(1)
                qml.Z(1)
            def ansatz_false():
                m = qml.measure(1)
                qml.Y(1)
            qml.cond(True, ansatz_true, ansatz_false)()
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
                %5 = tensor.extract %4[] : tensor<i1>
                %6 = scf.if %5 -> (!quantum.reg) {
                    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %8 = tensor.extract %7[] : tensor<i64>
                    %9 = quantum.extract %3[%8] : !quantum.reg -> !quantum.bit
                    %10, %11 = quantum.measure %9 : i1, !quantum.bit
                    %12 = tensor.from_elements %10 : tensor<i1>
                    %13 = quantum.custom "PauliZ"() %11 : !quantum.bit
                    %14 = tensor.extract %7[] : tensor<i64>
                    %15 = quantum.insert %3[%14], %13 : !quantum.reg, !quantum.bit
                    scf.yield %15 : !quantum.reg
                } else {
                    %16 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %17 = tensor.extract %16[] : tensor<i64>
                    %18 = quantum.extract %3[%17] : !quantum.reg -> !quantum.bit
                    %19, %20 = quantum.measure %18 : i1, !quantum.bit
                    %21 = tensor.from_elements %19 : tensor<i1>
                    %22 = quantum.custom "PauliY"() %20 : !quantum.bit
                    %23 = tensor.extract %16[] : tensor<i64>
                    %24 = quantum.insert %3[%23], %22 : !quantum.reg, !quantum.bit
                    scf.yield %24 : !quantum.reg
                }
                %25 = tensor.extract %0[] : tensor<i64>
                %26 = quantum.extract %6[%25] : !quantum.reg -> !quantum.bit
                %27 = quantum.namedobs %26[PauliZ] : !quantum.obs
                %28 = quantum.expval %27 : f64
                %29 = tensor.from_elements %28 : tensor<f64>
                %30 = tensor.extract %0[] : tensor<i64>
                %31 = quantum.insert %6[%30], %26 : !quantum.reg, !quantum.bit
                quantum.dealloc %31 : !quantum.reg
                quantum.device_release
                func.return %29 : tensor<f64>
            }

            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: func.func @quantum_segment_1
                // Main if statement
                // CHECK: {{%.+}} = scf.if {{%.+}} -> (!quantum.reg) {
                    // If statement for mcm
                    // CHECK: {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (i1, !quantum.bit) {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit
                    // CHECK: } else {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit

                    // Check that mcm results are not leaked outside the mcm if statement
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: } else {
                    // CHECK-NEXT: scf.yield {{%.+}} : !quantum.reg

                // CHECK: }{contain_mcm = "true", partition = "true_branch"}
                // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                // CHECK: {{%.+}} = arith.xori {{%.+}}, {{%.+}} : i1
                // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit

            // CHECK: func.func @quantum_segment_2

                // Main if statement
                // CHECK: {{%.+}} = scf.if {{%.+}} -> (!quantum.reg) {
                    // If statement for mcm
                    // CHECK: {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (i1, !quantum.bit) {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit
                    // CHECK: } else {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}}, {{%.+}} : i1, !quantum.bit

                    // Check that mcm results are not leaked outside the mcm if statement
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: } else {
                    // CHECK-NEXT: scf.yield {{%.+}} : !quantum.reg

                // CHECK: }{contain_mcm = "true", partition = "false_branch"}
                // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit

            // CHECK-NOT: func.func @quantum_segment_3
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_if_statement_with_nested_mcm(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with an if statement."""

        """
        Circuit Test
        def circuit():
            def ansatz_true():
                qml.Z(0)

                def nested_ansatz_true():
                    m1 = qml.measure(1)
                    qml.X( 0)
                def nested_ansatz_false():
                    qml.Y( 0)
                qml.cond(True, nested_ansatz_true, nested_ansatz_false)()

            def ansatz_false():
                qml.T(0)

            qml.cond(True, ansatz_true, ansatz_false)()
            qml.S(0)

            return qml.expval(qml.Z(0))

        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(2) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
                %5 = tensor.extract %4[] : tensor<i1>
                %6 = scf.if %5 -> (!quantum.reg) {
                    %7 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                    %8 = tensor.extract %7[] : tensor<i64>
                    %9 = quantum.extract %3[%8] : !quantum.reg -> !quantum.bit
                    %10 = quantum.custom "PauliZ"() %9 : !quantum.bit
                    %11 = tensor.extract %7[] : tensor<i64>
                    %12 = quantum.insert %3[%11], %10 : !quantum.reg, !quantum.bit
                    %13 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
                    %14 = tensor.extract %13[] : tensor<i1>
                    %15 = scf.if %14 -> (!quantum.reg) {
                        %16 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                        %17 = tensor.extract %16[] : tensor<i64>
                        %18 = quantum.extract %12[%17] : !quantum.reg -> !quantum.bit
                        %19, %20 = quantum.measure %18 : i1, !quantum.bit
                        %21 = tensor.from_elements %19 : tensor<i1>
                        %22 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                        %23 = tensor.extract %22[] : tensor<i64>
                        %24 = quantum.extract %12[%23] : !quantum.reg -> !quantum.bit
                        %25 = quantum.custom "PauliX"() %24 : !quantum.bit
                        %26 = tensor.extract %16[] : tensor<i64>
                        %27 = quantum.insert %12[%26], %20 : !quantum.reg, !quantum.bit
                        %28 = tensor.extract %22[] : tensor<i64>
                        %29 = quantum.insert %27[%28], %25 : !quantum.reg, !quantum.bit
                        scf.yield %29 : !quantum.reg
                    } else {
                        %30 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                        %31 = tensor.extract %30[] : tensor<i64>
                        %32 = quantum.extract %12[%31] : !quantum.reg -> !quantum.bit
                        %33 = quantum.custom "PauliY"() %32 : !quantum.bit
                        %34 = tensor.extract %30[] : tensor<i64>
                        %35 = quantum.insert %12[%34], %33 : !quantum.reg, !quantum.bit
                        scf.yield %35 : !quantum.reg
                    }
                    scf.yield %15 : !quantum.reg
                } else {
                    %36 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                    %37 = tensor.extract %36[] : tensor<i64>
                    %38 = quantum.extract %3[%37] : !quantum.reg -> !quantum.bit
                    %39 = quantum.custom "T"() %38 : !quantum.bit
                    %40 = tensor.extract %36[] : tensor<i64>
                    %41 = quantum.insert %3[%40], %39 : !quantum.reg, !quantum.bit
                    scf.yield %41 : !quantum.reg
                }
                %42 = tensor.extract %0[] : tensor<i64>
                %43 = quantum.extract %6[%42] : !quantum.reg -> !quantum.bit
                %44 = quantum.custom "S"() %43 : !quantum.bit
                %45 = quantum.namedobs %44[PauliZ] : !quantum.obs
                %46 = quantum.expval %45 : f64
                %47 = tensor.from_elements %46 : tensor<f64>
                %48 = tensor.extract %0[] : tensor<i64>
                %49 = quantum.insert %6[%48], %44 : !quantum.reg, !quantum.bit
                quantum.dealloc %49 : !quantum.reg
                quantum.device_release
                func.return %47 : tensor<f64>
            }

            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0

            // Outer if, true branch
            // CHECK: func.func @quantum_segment_1
                // CHECK: {{%.+}}, {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (!quantum.reg, i1, i1) {

                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                    // CHECK: quantum.custom "PauliZ"()
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit

                    // CHECK: [[false_branch:%.+]] = arith.xori {{%.+}}, {{%.+}} : i1
                    // CHECK: scf.yield  {{%.+}}, {{%.+}}, [[false_branch]] : !quantum.reg, i1, i1
                // CHECK: } else {
                    // CHECK: [[true_branch_t:%.+]] = arith.constant false
                    // CHECK: [[false_branch_t:%.+]] = arith.constant false
                    // CHECK: scf.yield  {{%.+}}, [[false_branch_t]], [[true_branch_t]] : !quantum.reg, i1, i1
                // CHECK: }{contain_mcm = "true", partition = "true_branch"}

                // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                // CHECK: func.return

            // Inner if, inside  outer if statement in the true branch
            // CHECK: func.func @quantum_segment_2({{.*}}, [[if_cond_t:%.+]] : i1{{.*}}) -> (!quantum.reg, i64) {
                // CHECK: scf.if [[if_cond_t]] -> (!quantum.reg) {
                    // CHECK: {{%.+}}, {{%.+}} = scf.if {{%.+}} -> (i1, !quantum.bit) {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}} {{%.+}} : i1, !quantum.bit
                    // CHECK } else {
                        // CHECK: quantum.measure {{%.+}} : i1, !quantum.bit
                        // CHECK: scf.yield {{%.+}} {{%.+}} : i1, !quantum.bit

                    // CHECK: quantum.custom "PauliX"()
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: } else {
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: }{contain_mcm = "true", partition = "true_branch", flattened = "true"}

            // Inner if, inside  outer if statement in the false branch
            // CHECK: func.func @quantum_segment_3({{.*}}, [[if_cond_tt:%.+]] : i1, [[if_cond_ttt:%.+]] : i1{{.*}}) -> (!quantum.reg
                // CHECK scf.if [[if_cond_tt]] -> (!quantum.reg) {

                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                    // CHECK: quantum.custom "PauliY"()
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit

                // CHECK: } else {
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: }{contain_mcm = "true", partition = "false_branch", flattened = "true"}

            // Outer if, false branch
            // CHECK: func.func @quantum_segment_4({{.*}}, [[if_cond_f:%.+]] : i1{{.*}}) -> (!quantum.reg, tensor<f64>) {
                // CHECK scf.if [[if_cond_f]] -> (!quantum.reg) {
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                    // CHECK: quantum.custom "T"()
                    // CHECK-NOT: quantum.measure {{%.+}} : i1, !quantum.bit
                // CHECK: } else {
                    // CHECK: scf.yield {{%.+}} : !quantum.reg
                // CHECK: }{contain_mcm = "true", partition = "false_branch"}

            // CHECK-NOT: func.func @quantum_segment_5
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [None, 50000])
    @pytest.mark.parametrize("branch", [False, True])
    @pytest.mark.parametrize(
        "measure_f",
        [
            lambda: qml.expval(qml.Z(0)),
            lambda: qml.expval(qml.Y(0)),
            lambda: qml.expval(qml.Z(1)),
            lambda: qml.expval(qml.Y(1)),
        ],
    )
    def test_execution_validation(self, shots, branch, measure_f):

        dev = qml.device("lightning.qubit", wires=4, seed=33)

        def test_circuit(branch_select):

            qml.H(0)
            qml.H(1)

            def ansatz_true():
                qml.measure(1)
                qml.RY(0.3, 0)
                qml.RY(0.3, 1)

            def ansatz_false():
                qml.RY(0.5, 0)

            qml.cond(branch_select, ansatz_true, ansatz_false)()

            return measure_f()

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @qml.set_shots(shots)
        @tree_traversal_pass
        @qml.qnode(dev)
        def circuit(branch=branch):
            return test_circuit(branch)

        res = circuit()

        qml.capture.disable()

        dev = qml.device("lightning.qubit", wires=4, seed=33)

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="tree-traversal")
        def circuit_ref(branch=branch):
            return test_circuit(branch)

        res_ref = circuit_ref()
        mcm_utils.validate_measurements(qml.expval, shots, res_ref, res, batch_size=None)


class TestTreeTraversalPassStaticForLoop:
    """Unit tests for TreeTraversalPass. without control flow operation."""

    def test_static_for_loop_without_mcm(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with a static for loop."""

        """
        Circuit Test
        def circuit():
            for i in range(3):
                qml.X(i)
            qml.measure(0)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(3) : !quantum.reg

                 %4 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %5 = tensor.extract %0[] : tensor<i64>
                %6 = arith.index_cast %5 : i64 to index
                %7 = tensor.extract %2[] : tensor<i64>
                %8 = arith.index_cast %7 : i64 to index
                %9 = tensor.extract %4[] : tensor<i64>
                %10 = arith.index_cast %9 : i64 to index
                %11 = scf.for %arg1 = %6 to %8 step %10 iter_args(%arg2 = %3) -> (!quantum.reg) {
                    %12 = arith.index_cast %arg1 : index to i64
                    %13 = tensor.from_elements %12 : tensor<i64>
                    %14 = tensor.extract %13[] : tensor<i64>
                    %15 = quantum.extract %arg2[%14] : !quantum.reg -> !quantum.bit
                    %16 = quantum.custom "PauliX"() %15 : !quantum.bit
                    %17 = tensor.extract %13[] : tensor<i64>
                    %18 = quantum.insert %arg2[%17], %16 : !quantum.reg, !quantum.bit
                    scf.yield %18 : !quantum.reg
                }
                %19 = tensor.extract %0[] : tensor<i64>
                %20 = quantum.extract %11[%19] : !quantum.reg -> !quantum.bit
                %21, %22 = quantum.measure %20 : i1, !quantum.bit
                %23 = tensor.from_elements %21 : tensor<i1>
                %24 = quantum.namedobs %22[PauliZ] : !quantum.obs
                %25 = quantum.expval %24 : f64
                %26 = tensor.from_elements %25 : tensor<f64>
                %27 = tensor.extract %0[] : tensor<i64>
                %28 = quantum.insert %11[%27], %22 : !quantum.reg, !quantum.bit
                quantum.dealloc %28 : !quantum.reg
                quantum.device_release
                func.return %26 : tensor<f64>
            }
            // CHECK: func.func public @circuit.simple_io.tree_traversal()
            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliX"()
            // CHECK: func.func @quantum_segment_1
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK-NOT: quantum.custom "PauliX"()
        }
        """
        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_static_for_loop_remove_loop(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with a static for loop."""

        """
        Circuit Test
        def circuit():
            for i in range(3):
                qml.X(i)
                qml.measure(i)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(3) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %5 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %6 = tensor.extract %0[] : tensor<i64>
                %7 = arith.index_cast %6 : i64 to index
                %8 = tensor.extract %4[] : tensor<i64>
                %9 = arith.index_cast %8 : i64 to index
                %10 = tensor.extract %5[] : tensor<i64>
                %11 = arith.index_cast %10 : i64 to index
                %12, %13 = scf.for %arg1 = %7 to %9 step %11 iter_args(%arg2 = %0, %arg3 = %3) -> (tensor<i64>, !quantum.reg) {
                    %14 = arith.index_cast %arg1 : index to i64
                    %15 = tensor.from_elements %14 : tensor<i64>
                    %16 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %17 = "stablehlo.add"(%arg2, %16) : (tensor<i64>, tensor<i64>) -> tensor<i64>
                    %18 = tensor.extract %15[] : tensor<i64>
                    %19 = quantum.extract %arg3[%18] : !quantum.reg -> !quantum.bit
                    %20 = quantum.custom "PauliX"() %19 : !quantum.bit
                    %21, %22 = quantum.measure %20 : i1, !quantum.bit
                    %23 = tensor.from_elements %21 : tensor<i1>
                    %24 = tensor.extract %15[] : tensor<i64>
                    %25 = quantum.insert %arg3[%24], %22 : !quantum.reg, !quantum.bit
                    scf.yield %17, %25 : tensor<i64>, !quantum.reg
                }
                %26 = tensor.extract %0[] : tensor<i64>
                %27 = quantum.extract %13[%26] : !quantum.reg -> !quantum.bit
                %28 = quantum.namedobs %27[PauliZ] : !quantum.obs
                %29 = quantum.expval %28 : f64
                %30 = tensor.from_elements %29 : tensor<f64>
                %31 = tensor.extract %0[] : tensor<i64>
                %32 = quantum.insert %13[%31], %27 : !quantum.reg, !quantum.bit
                quantum.dealloc %32 : !quantum.reg
                quantum.device_release
                func.return %30 : tensor<f64>
            }
            // CHECK: func.func public @circuit.simple_io.tree_traversal()
            // CHECK: func.func @state_transition
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)

        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_static_for_loop_count_ops(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with a static for loop."""

        """
        Circuit Test
        def circuit():
            for i in range(3):
                qml.X(i)
                qml.measure(i)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(3) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %5 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %6 = tensor.extract %0[] : tensor<i64>
                %7 = arith.index_cast %6 : i64 to index
                %8 = tensor.extract %4[] : tensor<i64>
                %9 = arith.index_cast %8 : i64 to index
                %10 = tensor.extract %5[] : tensor<i64>
                %11 = arith.index_cast %10 : i64 to index
                %12 = scf.for %arg1 = %7 to %9 step %11 iter_args(%arg2 = %3) -> (!quantum.reg) {
                    %13 = arith.index_cast %arg1 : index to i64
                    %14 = tensor.from_elements %13 : tensor<i64>
                    %15 = tensor.extract %14[] : tensor<i64>
                    %16 = quantum.extract %arg2[%15] : !quantum.reg -> !quantum.bit
                    %17 = quantum.custom "PauliX"() %16 : !quantum.bit
                    %18, %19 = quantum.measure %17 : i1, !quantum.bit
                    %20 = tensor.from_elements %18 : tensor<i1>
                    %21 = tensor.extract %14[] : tensor<i64>
                    %22 = quantum.insert %arg2[%21], %19 : !quantum.reg, !quantum.bit
                    scf.yield %22 : !quantum.reg
                }
                %23 = tensor.extract %0[] : tensor<i64>
                %24 = quantum.extract %12[%23] : !quantum.reg -> !quantum.bit
                %25 = quantum.namedobs %24[PauliZ] : !quantum.obs
                %26 = quantum.expval %25 : f64
                %27 = tensor.from_elements %26 : tensor<f64>
                %28 = tensor.extract %0[] : tensor<i64>
                %29 = quantum.insert %12[%28], %24 : !quantum.reg, !quantum.bit
                quantum.dealloc %29 : !quantum.reg
                quantum.device_release
                func.return %27 : tensor<f64>
            }
            // CHECK: func.func public @circuit.simple_io.tree_traversal()
            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: quantum.custom "PauliX"()
            // CHECK: func.func @quantum_segment_1
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliX"()
            // CHECK: func.func @quantum_segment_2
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliX"()
            // CHECK: func.func @quantum_segment_3
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK-NOT: quantum.custom "PauliX"()
            // CHECK-NOT: quantum.measure {{%.*}} postselect 0
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_2_static_for_loop_with_without_mcm(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with a static for loop."""

        """
        Circuit Test
        def circuit():
            for i in range(3):
                qml.X(i)
            for i in range(3):
                qml.measure(i)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(3) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %5 = tensor.extract %0[] : tensor<i64>
                %6 = arith.index_cast %5 : i64 to index
                %7 = tensor.extract %2[] : tensor<i64>
                %8 = arith.index_cast %7 : i64 to index
                %9 = tensor.extract %4[] : tensor<i64>
                %10 = arith.index_cast %9 : i64 to index
                %11 = scf.for %arg3 = %6 to %8 step %10 iter_args(%arg4 = %3) -> (!quantum.reg) {
                    %12 = arith.index_cast %arg3 : index to i64
                    %13 = tensor.from_elements %12 : tensor<i64>
                    %14 = tensor.extract %13[] : tensor<i64>
                    %15 = quantum.extract %arg4[%14] : !quantum.reg -> !quantum.bit
                    %16 = quantum.custom "PauliX"() %15 : !quantum.bit
                    %17 = tensor.extract %13[] : tensor<i64>
                    %18 = quantum.insert %arg4[%17], %16 : !quantum.reg, !quantum.bit
                    scf.yield %18 : !quantum.reg
                }
                %19 = tensor.extract %0[] : tensor<i64>
                %20 = arith.index_cast %19 : i64 to index
                %21 = tensor.extract %2[] : tensor<i64>
                %22 = arith.index_cast %21 : i64 to index
                %23 = tensor.extract %4[] : tensor<i64>
                %24 = arith.index_cast %23 : i64 to index
                %25 = scf.for %arg1 = %20 to %22 step %24 iter_args(%arg2 = %11) -> (!quantum.reg) {
                    %26 = arith.index_cast %arg1 : index to i64
                    %27 = tensor.from_elements %26 : tensor<i64>
                    %28 = tensor.extract %27[] : tensor<i64>
                    %29 = quantum.extract %arg2[%28] : !quantum.reg -> !quantum.bit
                    %30, %31 = quantum.measure %29 : i1, !quantum.bit
                    %32 = tensor.from_elements %30 : tensor<i1>
                    %33 = tensor.extract %27[] : tensor<i64>
                    %34 = quantum.insert %arg2[%33], %31 : !quantum.reg, !quantum.bit
                    scf.yield %34 : !quantum.reg
                }
                %35 = tensor.extract %0[] : tensor<i64>
                %36 = quantum.extract %25[%35] : !quantum.reg -> !quantum.bit
                %37 = quantum.namedobs %36[PauliZ] : !quantum.obs
                %38 = quantum.expval %37 : f64
                %39 = tensor.from_elements %38 : tensor<f64>
                %40 = tensor.extract %0[] : tensor<i64>
                %41 = quantum.insert %25[%40], %36 : !quantum.reg, !quantum.bit
                quantum.dealloc %41 : !quantum.reg
                quantum.device_release
                func.return %39 : tensor<f64>
            }
            // CHECK: func.func public @circuit.simple_io.tree_traversal()
            // CHECK: func.func @state_transition
            // CHECK: func.func @quantum_segment_0
            // CHECK: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliX"()
            // CHECK: func.func @quantum_segment_1
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: func.func @quantum_segment_2
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: func.func @quantum_segment_3
            // CHECK: quantum.measure {{%.*}} postselect 0
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_nested_static_for_loop_with_mcm(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with a static for loop."""

        """
        Circuit Test
        def circuit():
            for i in range(3):
                qml.X(i)
                for j in range(3):
                    qml.Y(j)
                    qml.measure(j)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(3) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %5 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %6 = tensor.extract %0[] : tensor<i64>
                %7 = arith.index_cast %6 : i64 to index
                %8 = tensor.extract %4[] : tensor<i64>
                %9 = arith.index_cast %8 : i64 to index
                %10 = tensor.extract %5[] : tensor<i64>
                %11 = arith.index_cast %10 : i64 to index
                %12 = scf.for %arg1 = %7 to %9 step %11 iter_args(%arg2 = %3) -> (!quantum.reg) {
                    %13 = arith.index_cast %arg1 : index to i64
                    %14 = tensor.from_elements %13 : tensor<i64>
                    %15 = tensor.extract %14[] : tensor<i64>
                    %16 = quantum.extract %arg2[%15] : !quantum.reg -> !quantum.bit
                    %17 = quantum.custom "PauliX"() %16 : !quantum.bit
                    %18 = tensor.extract %14[] : tensor<i64>
                    %19 = quantum.insert %arg2[%18], %17 : !quantum.reg, !quantum.bit
                    %20 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                    %21 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                    %22 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %23 = tensor.extract %20[] : tensor<i64>
                    %24 = arith.index_cast %23 : i64 to index
                    %25 = tensor.extract %21[] : tensor<i64>
                    %26 = arith.index_cast %25 : i64 to index
                    %27 = tensor.extract %22[] : tensor<i64>
                    %28 = arith.index_cast %27 : i64 to index
                    %29 = scf.for %arg3 = %24 to %26 step %28 iter_args(%arg4 = %19) -> (!quantum.reg) {
                        %30 = arith.index_cast %arg3 : index to i64
                        %31 = tensor.from_elements %30 : tensor<i64>
                        %32 = tensor.extract %31[] : tensor<i64>
                        %33 = quantum.extract %arg4[%32] : !quantum.reg -> !quantum.bit
                        %34 = quantum.custom "PauliY"() %33 : !quantum.bit
                        %35, %36 = quantum.measure %34 : i1, !quantum.bit
                        %37 = tensor.from_elements %35 : tensor<i1>
                        %38 = tensor.extract %31[] : tensor<i64>
                        %39 = quantum.insert %arg4[%38], %36 : !quantum.reg, !quantum.bit
                        scf.yield %39 : !quantum.reg
                    }
                    scf.yield %29 : !quantum.reg
                }
                %40 = tensor.extract %0[] : tensor<i64>
                %41 = quantum.extract %12[%40] : !quantum.reg -> !quantum.bit
                %42 = quantum.namedobs %41[PauliZ] : !quantum.obs
                %43 = quantum.expval %42 : f64
                %44 = tensor.from_elements %43 : tensor<f64>
                %45 = tensor.extract %0[] : tensor<i64>
                %46 = quantum.insert %12[%45], %41 : !quantum.reg, !quantum.bit
                quantum.dealloc %46 : !quantum.reg
                quantum.device_release
                func.return %44 : tensor<f64>
            }
            // CHECK: func.func public @circuit.simple_io.tree_traversal()
            // CHECK: func.func @state_transition
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)

            // CHECK: quantum.custom "PauliX"()
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0

            // CHECK: quantum.custom "PauliX"()
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0

            // CHECK: quantum.custom "PauliX"()
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: quantum.custom "PauliY"()
            // CHECK: quantum.measure {{%.*}} postselect 0

        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    def test_nested_static_for_loop_with_mcm_early(self, run_filecheck):
        """Test tree traversal pass would be applied to a func with a static for loop."""

        """
        Circuit Test
        def circuit():
            for i in range(3):
                qml.X(i)
                qml.measure(i)
                for j in range(3):
                    qml.Y(j)
            return qml.expval(qml.Z(0))
        """

        program = """
        builtin.module @module_circuit {

            // CHECK: func.func public @circuit() -> tensor<f64> attributes {qnode}
            // CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            // CHECK: func.return %0 : tensor<f64>

            func.func public @circuit() -> (tensor<f64>) attributes {qnode} {
                %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                %1 = tensor.extract %0[] : tensor<i64>
                quantum.device shots(%1) ["", "", ""]
                %2 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %3 = quantum.alloc(3) : !quantum.reg
                %4 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                %5 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                %6 = tensor.extract %0[] : tensor<i64>
                %7 = arith.index_cast %6 : i64 to index
                %8 = tensor.extract %4[] : tensor<i64>
                %9 = arith.index_cast %8 : i64 to index
                %10 = tensor.extract %5[] : tensor<i64>
                %11 = arith.index_cast %10 : i64 to index
                %12 = scf.for %arg1 = %7 to %9 step %11 iter_args(%arg2 = %3) -> (!quantum.reg) {
                    %13 = arith.index_cast %arg1 : index to i64
                    %14 = tensor.from_elements %13 : tensor<i64>
                    %15 = tensor.extract %14[] : tensor<i64>
                    %16 = quantum.extract %arg2[%15] : !quantum.reg -> !quantum.bit
                    %17 = quantum.custom "PauliX"() %16 : !quantum.bit
                    %18, %19 = quantum.measure %17 : i1, !quantum.bit
                    %20 = tensor.from_elements %18 : tensor<i1>
                    %21 = tensor.extract %14[] : tensor<i64>
                    %22 = quantum.insert %arg2[%21], %19 : !quantum.reg, !quantum.bit
                    %23 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
                    %24 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
                    %25 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
                    %26 = tensor.extract %23[] : tensor<i64>
                    %27 = arith.index_cast %26 : i64 to index
                    %28 = tensor.extract %24[] : tensor<i64>
                    %29 = arith.index_cast %28 : i64 to index
                    %30 = tensor.extract %25[] : tensor<i64>
                    %31 = arith.index_cast %30 : i64 to index
                    %32 = scf.for %arg3 = %27 to %29 step %31 iter_args(%arg4 = %22) -> (!quantum.reg) {
                        %33 = arith.index_cast %arg3 : index to i64
                        %34 = tensor.from_elements %33 : tensor<i64>
                        %35 = tensor.extract %34[] : tensor<i64>
                        %36 = quantum.extract %arg4[%35] : !quantum.reg -> !quantum.bit
                        %37 = quantum.custom "PauliY"() %36 : !quantum.bit
                        %38 = tensor.extract %34[] : tensor<i64>
                        %39 = quantum.insert %arg4[%38], %37 : !quantum.reg, !quantum.bit
                        scf.yield %39 : !quantum.reg
                    }
                    scf.yield %32 : !quantum.reg
                }
                %40 = tensor.extract %0[] : tensor<i64>
                %41 = quantum.extract %12[%40] : !quantum.reg -> !quantum.bit
                %42 = quantum.namedobs %41[PauliZ] : !quantum.obs
                %43 = quantum.expval %42 : f64
                %44 = tensor.from_elements %43 : tensor<f64>
                %45 = tensor.extract %0[] : tensor<i64>
                %46 = quantum.insert %12[%45], %41 : !quantum.reg, !quantum.bit
                quantum.dealloc %46 : !quantum.reg
                quantum.device_release
                func.return %44 : tensor<f64>
            }
            // CHECK: func.func public @circuit.simple_io.tree_traversal()
            // CHECK: func.func @state_transition
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)

            // CHECK: quantum.custom "PauliX"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliY"()

            // CHECK: quantum.custom "PauliX"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliY"()

            // CHECK: quantum.custom "PauliX"()
            // CHECK: quantum.measure {{%.*}} postselect 0
            // CHECK: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK: quantum.custom "PauliY"()

            // CHECK-NOT: quantum.custom "PauliX"()
            // CHECK-NOT: quantum.measure {{%.*}} postselect 0
            // CHECK-NOT: scf.for {{%.+}} = {{%.+}} to {{%.+}} step {{%.+}} iter_args({{%.+}} = {{%.+}}) -> (!quantum.reg)
            // CHECK-NOT: quantum.custom "PauliY"()
        }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline, roundtrip=True)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_tree_traversal_pass_only(self, run_filecheck_qjit):
        """Test the tree traversal pass only."""
        dev = qml.device("lightning.qubit", wires=5)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @tree_traversal_pass
        @qml.qnode(dev)
        def circuit():
            # CHECK: func.func public @circuit() -> (tensor<f64>)
            # CHECK: func.call @circuit.simple_io.tree_traversal() : ()
            # CHECK: func.return %0 : tensor<f64>
            for i in range(3):
                qml.X(0)
                qml.measure(i)
            return qml.expval(qml.Z(0))
            # CHECK: func.func public @circuit.simple_io.tree_traversal()
            # CHECK: func.func @state_transition
            # CHECK: func.func @quantum_segment_0
            # CHECK: quantum.custom "PauliX"()
            # CHECK: func.func @quantum_segment_1
            # CHECK: quantum.measure {{%.*}} postselect 0
            # CHECK: quantum.custom "PauliX"()
            # CHECK: func.func @quantum_segment_2
            # CHECK: quantum.measure {{%.*}} postselect 0
            # CHECK: quantum.custom "PauliX"()
            # CHECK: func.func @quantum_segment_3
            # CHECK: quantum.measure {{%.*}} postselect 0
            # CHECK-NOT: quantum.custom "PauliX"()
            # CHECK-NOT: quantum.measure {{%.*}} postselect 0

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [None, 10000])
    def test_execution_validation(self, shots):

        dev = qml.device("lightning.qubit", wires=4)

        def test_circuit():
            for i in range(3):
                qml.H(i)

            for i in range(3):
                qml.measure(0)
                qml.RY(np.pi / 8, i)
            return qml.expval(qml.Z(1))

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @qml.set_shots(shots)
        @tree_traversal_pass
        @qml.qnode(dev)
        def circuit():
            return test_circuit()

        res = circuit()

        qml.capture.disable()

        dev = qml.device("lightning.qubit", wires=4)

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="tree-traversal")
        def circuit_ref():
            return test_circuit()

        res_ref = circuit_ref()
        mcm_utils.validate_measurements(qml.expval, shots, res_ref, res, batch_size=None)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [None])
    @pytest.mark.parametrize("postselect", [None])
    @pytest.mark.parametrize("reset", [False])
    @pytest.mark.parametrize(
        "measure_f",
        [
            lambda: qml.expval(qml.Z(0)),
            lambda: qml.expval(qml.Y(0)),
            lambda: qml.expval(qml.Z(1)),
            lambda: qml.expval(qml.Y(1)),
        ],
    )
    def test_multiple_measurements(self, shots, postselect, reset, measure_f):
        """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
        and a conditional gate. Multiple measurements of the mid-circuit measurement value are
        performed. This function also tests `reset` parametrizing over the parameter."""

        dev = qml.device("lightning.qubit", wires=3, seed=42)
        params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]

        def obs_tape(x, y, z, reset=False, postselect=None):
            for i in range(3):
                qml.RX(np.pi / (i + 1), i)
                # qml.measure(i)
                for j in range(3):
                    qml.RY(np.pi / (i + j + 1), j)
                    qml.measure(1)

        # Measures:
        # Without shots
        # qml.expval, qml.probs, qml.var
        # With shots
        # qml.expval, qml.count, qml.var, qml.sample

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
        )
        @qml.set_shots(shots)
        @tree_traversal_pass
        @qml.qnode(dev)
        def qjit_func(x, y, z):
            obs_tape(x, y, z, reset=reset, postselect=postselect)
            return measure_f()

        results0 = qjit_func(*params)

        dev = qml.device("default.qubit", seed=42)

        qml.capture.disable()

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="deferred")
        def ref_func(x, y, z):
            obs_tape(x, y, z, reset=reset, postselect=postselect)
            return measure_f()

        results1 = ref_func(*params)

        mcm_utils.validate_measurements(qml.expval, shots, results1, results0, batch_size=None)
