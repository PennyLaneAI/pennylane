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
import jax.numpy as jnp
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
    - return multiple measurements,e.g.
        return qml.expval(Z(0)), qml.expval(X(1))
    - qml.expval(mcm_result)
"""


class TestTreeTraversalPass:
    """Unit tests for TreeTraversalPass."""

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
        """Test tree traversal pass would not be applied to a func with only qubit gates."""

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
        """Test tree traversal pass would not be applied to a func with only qubit gates."""

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

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_execution_validation(self):

        dev = qml.device("lightning.qubit", wires=4)

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
        @tree_traversal_pass
        @qml.qnode(dev)
        def circuit():
            return test_circuit()

        res = circuit()

        qml.capture.disable()

        @qml.qnode(dev, mcm_method="deferred")
        def circuit_ref():
            return test_circuit()

        res_ref = circuit_ref()
        np.testing.assert_allclose(res, res_ref)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    @pytest.mark.parametrize("shots", [None, 10000])
    @pytest.mark.parametrize("postselect", [None])
    # @pytest.mark.parametrize("postselect", [None,0, 1])
    @pytest.mark.parametrize("reset", [False])
    # @pytest.mark.parametrize("reset", [False, True])
    @pytest.mark.parametrize(
        "measure_f",
        [
            lambda : qml.expval(qml.Z(0)),
            lambda : qml.expval(qml.Y(0)),
            lambda : qml.expval(qml.Z(1)),
            lambda : qml.expval(qml.Y(1)),
        ],
    )
    def test_multiple_measurements_and_reset(self, shots, postselect, reset, measure_f, seed):
        """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
        and a conditional gate. Multiple measurements of the mid-circuit measurement value are
        performed. This function also tests `reset` parametrizing over the parameter."""

        dev = qml.device("lightning.qubit", wires=3, seed=seed)
        params = [np.pi / 2.5, np.pi / 3, -np.pi / 3.5]

        def obs_tape(x, y, z, reset=False, postselect=None):

            for i in range(3):
                qml.measure(i, reset=reset)

            qml.RX(x, 0)
            qml.RZ(np.pi / 8, 0)

            m0 = qml.measure(0, reset=reset)

            def ansatz_m0_0_true():
                qml.RX(np.pi / 4, 0)
                qml.RZ(np.pi / 4, 0)

            def ansatz_m0_0_false():
                qml.RX(-np.pi / 4, 0)
                qml.RZ(-np.pi / 4, 0)

            qml.cond(m0, ansatz_m0_0_true, ansatz_m0_0_false)()

            m1 = qml.measure(1, postselect=postselect)

            qml.RX(y, 1)
            qml.RZ(np.pi / 4, 1)
            m1 = qml.measure(1, postselect=postselect)

            def ansatz_m1_0_true():
                qml.RX(np.pi / 8, 1)
                qml.RZ(np.pi / 8, 1)

            def ansatz_m1_0_false():
                qml.RX(-np.pi / 8, 1)
                qml.RZ(-np.pi / 8, 1)

            qml.cond(m1, ansatz_m1_0_true, ansatz_m1_0_false)()

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

        dev = qml.device("default.qubit", seed=seed)

        qml.capture.disable()

        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="deferred")
        def ref_func(x, y, z):
            obs_tape(x, y, z, reset=reset, postselect=postselect)
            return measure_f()

        results1 = ref_func(*params)

        mcm_utils.validate_measurements(qml.expval, shots, results1, results0, batch_size=None)
