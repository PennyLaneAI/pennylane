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
"""Unit test module for the convert to MBQC formalism transform"""
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.ftqc import mbqc_pipeline

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    ConvertToMBQCFormalismPass,
    convert_to_mbqc_formalism_pass,
    decompose_graph_state_pass,
    measurements_from_samples_pass,
)
from pennylane.ftqc import RotXZX


class TestConvertToMBQCFormalismPass:
    """Unit tests for ConvertToMBQCFormalismPass."""

    def test_unsupported_gate(self, run_filecheck):
        """Test for error threw for unsupported gate"""
        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.custom "IsingXX"() %0 : !quantum.bit
                return
            }
        """
        with pytest.raises(NotImplementedError):
            pipeline = (ConvertToMBQCFormalismPass(),)
            run_filecheck(program, pipeline)

    def test_unconverted_gate_set(self, run_filecheck):
        """Test for supported gates that are not converted in the pass"""
        program = """
            func.func @test_func(%arg0 :f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = quantum.custom "PauliX"() [[q0:%.+]] : !quantum.bit
                %1 = quantum.custom "PauliX"() %0 : !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = quantum.custom "PauliY"() [[q0:%.+]] : !quantum.bit
                %2 = quantum.custom "PauliY"() %1 : !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = quantum.custom "PauliZ"() [[q0:%.+]] : !quantum.bit
                %3 = quantum.custom "PauliZ"() %2 : !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = quantum.custom "Identity"() [[q0:%.+]] : !quantum.bit
                %4 = quantum.custom "Identity"() %3 : !quantum.bit
                // CHECK-NEXT: quantum.gphase
                quantum.gphase %arg0
                return
            }
        """
        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_hadamard_gate(self, run_filecheck):
        """Test for lowering a Hadamard gate to a MBQC formalism."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = func.call @hadamard_in_mbqc([[q0:%.+]]) : (!quantum.bit) -> !quantum.bit
                %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
                return
            }

            // CHECK: func.func private @hadamard_in_mbqc(%0 : !quantum.bit) -> !quantum.bit attributes {mbqc_transform = none} {
            // CHECK-NEXT: %1 = arith.constant dense<[true, false, true, false, false, true]> : tensor<6xi1>
            // CHECK-NEXT: %2 = mbqc.graph_state_prep(%1 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            // CHECK-NEXT: %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: %4 = quantum.extract %2[1] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: %5 = quantum.extract %2[2] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: %6 = quantum.extract %2[3] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: %7, %8 = quantum.custom "CZ"() %0, %3 : !quantum.bit, !quantum.bit
            // CHECK-NEXT: %9 = arith.constant 0.000000e+00 : f64
            // CHECK-NEXT: %10 = arith.constant 1.5707963267948966 : f64
            // CHECK-NEXT: %11, %12 = mbqc.measure_in_basis[XY, %9] %7 : i1, !quantum.bit
            // CHECK-NEXT: %13, %14 = mbqc.measure_in_basis[XY, %10] %8 : i1, !quantum.bit
            // CHECK-NEXT: %15, %16 = mbqc.measure_in_basis[XY, %10] %4 : i1, !quantum.bit
            // CHECK-NEXT: %17, %18 = mbqc.measure_in_basis[XY, %10] %5 : i1, !quantum.bit
            // CHECK-NEXT: %19 = arith.xori %11, %15 : i1
            // CHECK-NEXT: %20 = arith.xori %19, %17 : i1
            // CHECK-NEXT: %21 = arith.constant true
            // CHECK-NEXT: %22 = arith.cmpi eq, %20, %21 : i1
            // CHECK-NEXT: %23 = scf.if %22 -> (!quantum.bit) {
            // CHECK-NEXT:   %24 = quantum.custom "PauliX"() %6 : !quantum.bit
            // CHECK-NEXT:   scf.yield %24 : !quantum.bit
            // CHECK-NEXT: } else {
            // CHECK-NEXT:   scf.yield %6 : !quantum.bit
            // CHECK-NEXT: }
            // CHECK-NEXT: %25 = arith.xori %13, %15 : i1
            // CHECK-NEXT: %26 = arith.constant true
            // CHECK-NEXT: %27 = arith.cmpi eq, %25, %26 : i1
            // CHECK-NEXT: %28 = scf.if %27 -> (!quantum.bit) {
            // CHECK-NEXT:   %29 = quantum.custom "PauliZ"() %23 : !quantum.bit
            // CHECK-NEXT:   scf.yield %29 : !quantum.bit
            // CHECK-NEXT: } else {
            // CHECK-NEXT:   scf.yield %23 : !quantum.bit
            // CHECK-NEXT: }
            // CHECK-NEXT: quantum.dealloc_qb %14 : !quantum.bit
            // CHECK-NEXT: quantum.dealloc_qb %16 : !quantum.bit
            // CHECK-NEXT: quantum.dealloc_qb %18 : !quantum.bit
            // CHECK-NEXT: quantum.dealloc_qb %12 : !quantum.bit
            // CHECK-NEXT: func.return %28 : !quantum.bit
            // CHECK-NEXT: }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_s_gate(self, run_filecheck):
        """Test for lowering a S gate to a MBQC formalism."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = func.call @s_in_mbqc([[q0:%.+]]) : (!quantum.bit) -> !quantum.bit
                %1 = quantum.custom "S"() %0 : !quantum.bit
                return
            }

            // CHECK: func.func private @s_in_mbqc(%0 : !quantum.bit) -> !quantum.bit attributes {mbqc_transform = none} {
            // CHECK-NEXT:   %1 = arith.constant dense<[true, false, true, false, false, true]> : tensor<6xi1>
            // CHECK-NEXT:   %2 = mbqc.graph_state_prep(%1 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            // CHECK-NEXT:   %3 = quantum.extract %2[0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %4 = quantum.extract %2[1] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %5 = quantum.extract %2[2] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %6 = quantum.extract %2[3] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %7, %8 = quantum.custom "CZ"() %0, %3 : !quantum.bit, !quantum.bit
            // CHECK-NEXT:   %9 = arith.constant 0.000000e+00 : f64
            // CHECK-NEXT:   %10 = arith.constant 1.5707963267948966 : f64
            // CHECK-NEXT:   %11, %12 = mbqc.measure_in_basis[XY, %9] %7 : i1, !quantum.bit
            // CHECK-NEXT:   %13, %14 = mbqc.measure_in_basis[XY, %9] %8 : i1, !quantum.bit
            // CHECK-NEXT:   %15, %16 = mbqc.measure_in_basis[XY, %10] %4 : i1, !quantum.bit
            // CHECK-NEXT:   %17, %18 = mbqc.measure_in_basis[XY, %9] %5 : i1, !quantum.bit
            // CHECK-NEXT:   %19 = arith.xori %13, %17 : i1
            // CHECK-NEXT:   %20 = arith.constant true
            // CHECK-NEXT:   %21 = arith.cmpi eq, %19, %20 : i1
            // CHECK-NEXT:   %22 = scf.if %21 -> (!quantum.bit) {
            // CHECK-NEXT:     %23 = quantum.custom "PauliX"() %6 : !quantum.bit
            // CHECK-NEXT:     scf.yield %23 : !quantum.bit
            // CHECK-NEXT:   } else {
            // CHECK-NEXT:     scf.yield %6 : !quantum.bit
            // CHECK-NEXT:   }
            // CHECK-NEXT:   %24 = arith.xori %11, %13 : i1
            // CHECK-NEXT:   %25 = arith.xori %24, %15 : i1
            // CHECK-NEXT:   %26 = arith.constant true
            // CHECK-NEXT:   %27 = arith.xori %25, %26 : i1
            // CHECK-NEXT:   %28 = arith.constant true
            // CHECK-NEXT:   %29 = arith.cmpi eq, %27, %28 : i1
            // CHECK-NEXT:   %30 = scf.if %29 -> (!quantum.bit) {
            // CHECK-NEXT:     %31 = quantum.custom "PauliZ"() %22 : !quantum.bit
            // CHECK-NEXT:     scf.yield %31 : !quantum.bit
            // CHECK-NEXT:   } else {
            // CHECK-NEXT:     scf.yield %22 : !quantum.bit
            // CHECK-NEXT:   }
            // CHECK-NEXT:   quantum.dealloc_qb %14 : !quantum.bit
            // CHECK-NEXT:   quantum.dealloc_qb %16 : !quantum.bit
            // CHECK-NEXT:   quantum.dealloc_qb %18 : !quantum.bit
            // CHECK-NEXT:   quantum.dealloc_qb %12 : !quantum.bit
            // CHECK-NEXT:   func.return %30 : !quantum.bit
            // CHECK-NEXT: }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_rz_gate(self, run_filecheck):
        """Test for lowering a RZ gate to a MBQC formalism."""
        program = """
            func.func @test_func(%param0: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = func.call @rz_in_mbqc(%param0, %0) : (f64, !quantum.bit) -> !quantum.bit
                %1 = quantum.custom "RZ"(%param0) %0 : !quantum.bit
                return
            }
            // CHECK: func.func private @rz_in_mbqc(%0 : f64, %1 : !quantum.bit) -> !quantum.bit attributes {mbqc_transform = none} {
            // CHECK-NEXT:   %2 = arith.constant dense<[true, false, true, false, false, true]> : tensor<6xi1>
            // CHECK-NEXT:   %3 = mbqc.graph_state_prep(%2 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            // CHECK-NEXT:   %4 = quantum.extract %3[0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %5 = quantum.extract %3[1] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %6 = quantum.extract %3[2] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %7 = quantum.extract %3[3] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:   %8, %9 = quantum.custom "CZ"() %1, %4 : !quantum.bit, !quantum.bit
            // CHECK-NEXT:   %10 = arith.constant 0.000000e+00 : f64
            // CHECK-NEXT:   %11, %12 = mbqc.measure_in_basis[XY, %10] %8 : i1, !quantum.bit
            // CHECK-NEXT:   %13, %14 = mbqc.measure_in_basis[XY, %10] %9 : i1, !quantum.bit
            // CHECK-NEXT:   %15 = arith.constant true
            // CHECK-NEXT:   %16 = arith.cmpi eq, %13, %15 : i1
            // CHECK-NEXT:   %17, %18 = scf.if %16 -> (i1, !quantum.bit) {
            // CHECK-NEXT:     %19, %20 = mbqc.measure_in_basis[XY, %0] %5 : i1, !quantum.bit
            // CHECK-NEXT:     scf.yield %19, %20 : i1, !quantum.bit
            // CHECK-NEXT:   } else {
            // CHECK-NEXT:     %21 = arith.negf %0 : f64
            // CHECK-NEXT:     %22, %23 = mbqc.measure_in_basis[XY, %21] %5 : i1, !quantum.bit
            // CHECK-NEXT:     scf.yield %22, %23 : i1, !quantum.bit
            // CHECK-NEXT:   }
            // CHECK-NEXT:   %24, %25 = mbqc.measure_in_basis[XY, %10] %6 : i1, !quantum.bit
            // CHECK-NEXT:   %26 = arith.xori %13, %24 : i1
            // CHECK-NEXT:   %27 = arith.constant true
            // CHECK-NEXT:   %28 = arith.cmpi eq, %26, %27 : i1
            // CHECK-NEXT:   %29 = scf.if %28 -> (!quantum.bit) {
            // CHECK-NEXT:     %30 = quantum.custom "PauliX"() %7 : !quantum.bit
            // CHECK-NEXT:     scf.yield %30 : !quantum.bit
            // CHECK-NEXT:   } else {
            // CHECK-NEXT:     scf.yield %7 : !quantum.bit
            // CHECK-NEXT:   }
            // CHECK-NEXT:   %31 = arith.xori %11, %17 : i1
            // CHECK-NEXT:   %32 = arith.constant true
            // CHECK-NEXT:   %33 = arith.cmpi eq, %31, %32 : i1
            // CHECK-NEXT:   %34 = scf.if %33 -> (!quantum.bit) {
            // CHECK-NEXT:     %35 = quantum.custom "PauliZ"() %29 : !quantum.bit
            // CHECK-NEXT:     scf.yield %35 : !quantum.bit
            // CHECK-NEXT:   } else {
            // CHECK-NEXT:     scf.yield %29 : !quantum.bit
            // CHECK-NEXT:   }
            // CHECK-NEXT:   quantum.dealloc_qb %14 : !quantum.bit
            // CHECK-NEXT:   quantum.dealloc_qb %18 : !quantum.bit
            // CHECK-NEXT:   quantum.dealloc_qb %25 : !quantum.bit
            // CHECK-NEXT:   quantum.dealloc_qb %12 : !quantum.bit
            // CHECK-NEXT:   func.return %34 : !quantum.bit
            // CHECK-NEXT: }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_rotxzx_gate(self, run_filecheck):
        """Test for lowering a RotXZX gate to a MBQC formalism."""
        program = """
            func.func @test_func(%param0: f64, %param1: f64, %param2: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q0:%.+]] = func.call @rotxzx_in_mbqc(%param0, %param1, %param2, %0) : (f64, f64, f64, !quantum.bit) -> !quantum.bit
                %1 = quantum.custom "RotXZX"(%param0, %param1, %param2) %0 : !quantum.bit
                return
            }
            // CHECK: func.func private @rotxzx_in_mbqc(%0 : f64, %1 : f64, %2 : f64, %3 : !quantum.bit) -> !quantum.bit attributes {mbqc_transform = none} {
            // CHECK-NEXT:  %4 = arith.constant dense<[true, false, true, false, false, true]> : tensor<6xi1>
            // CHECK-NEXT:  %5 = mbqc.graph_state_prep(%4 : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            // CHECK-NEXT:  %6 = quantum.extract %5[0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:  %7 = quantum.extract %5[1] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:  %8 = quantum.extract %5[2] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:  %9 = quantum.extract %5[3] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT:  %10, %11 = quantum.custom "CZ"() %3, %6 : !quantum.bit, !quantum.bit
            // CHECK-NEXT:  %12 = arith.constant 0.000000e+00 : f64
            // CHECK-NEXT:  %13, %14 = mbqc.measure_in_basis[XY, %12] %10 : i1, !quantum.bit
            // CHECK-NEXT:  %15 = arith.constant true
            // CHECK-NEXT:  %16 = arith.cmpi eq, %13, %15 : i1
            // CHECK-NEXT:  %17, %18 = scf.if %16 -> (i1, !quantum.bit) {
            // CHECK-NEXT:    %19, %20 = mbqc.measure_in_basis[XY, %0] %11 : i1, !quantum.bit
            // CHECK-NEXT:    scf.yield %19, %20 : i1, !quantum.bit
            // CHECK-NEXT:  } else {
            // CHECK-NEXT:    %21 = arith.negf %0 : f64
            // CHECK-NEXT:    %22, %23 = mbqc.measure_in_basis[XY, %21] %11 : i1, !quantum.bit
            // CHECK-NEXT:    scf.yield %22, %23 : i1, !quantum.bit
            // CHECK-NEXT:  }
            // CHECK-NEXT:  %24 = arith.constant true
            // CHECK-NEXT:  %25 = arith.cmpi eq, %17, %24 : i1
            // CHECK-NEXT:  %26, %27 = scf.if %25 -> (i1, !quantum.bit) {
            // CHECK-NEXT:    %28, %29 = mbqc.measure_in_basis[XY, %1] %7 : i1, !quantum.bit
            // CHECK-NEXT:    scf.yield %28, %29 : i1, !quantum.bit
            // CHECK-NEXT:  } else {
            // CHECK-NEXT:    %30 = arith.negf %1 : f64
            // CHECK-NEXT:    %31, %32 = mbqc.measure_in_basis[XY, %30] %7 : i1, !quantum.bit
            // CHECK-NEXT:    scf.yield %31, %32 : i1, !quantum.bit
            // CHECK-NEXT:  }
            // CHECK-NEXT:  %33 = arith.xori %13, %26 : i1
            // CHECK-NEXT:  %34 = arith.constant true
            // CHECK-NEXT:  %35 = arith.cmpi eq, %33, %34 : i1
            // CHECK-NEXT:  %36, %37 = scf.if %35 -> (i1, !quantum.bit) {
            // CHECK-NEXT:    %38, %39 = mbqc.measure_in_basis[XY, %2] %8 : i1, !quantum.bit
            // CHECK-NEXT:    scf.yield %38, %39 : i1, !quantum.bit
            // CHECK-NEXT:  } else {
            // CHECK-NEXT:    %40 = arith.negf %2 : f64
            // CHECK-NEXT:    %41, %42 = mbqc.measure_in_basis[XY, %40] %8 : i1, !quantum.bit
            // CHECK-NEXT:    scf.yield %41, %42 : i1, !quantum.bit
            // CHECK-NEXT:  }
            // CHECK-NEXT:  %43 = arith.xori %17, %36 : i1
            // CHECK-NEXT:  %44 = arith.constant true
            // CHECK-NEXT:  %45 = arith.cmpi eq, %43, %44 : i1
            // CHECK-NEXT:  %46 = scf.if %45 -> (!quantum.bit) {
            // CHECK-NEXT:    %47 = quantum.custom "PauliX"() %9 : !quantum.bit
            // CHECK-NEXT:    scf.yield %47 : !quantum.bit
            // CHECK-NEXT:  } else {
            // CHECK-NEXT:    scf.yield %9 : !quantum.bit
            // CHECK-NEXT:  }
            // CHECK-NEXT:  %48 = arith.xori %13, %26 : i1
            // CHECK-NEXT:  %49 = arith.constant true
            // CHECK-NEXT:  %50 = arith.cmpi eq, %48, %49 : i1
            // CHECK-NEXT:  %51 = scf.if %50 -> (!quantum.bit) {
            // CHECK-NEXT:    %52 = quantum.custom "PauliZ"() %46 : !quantum.bit
            // CHECK-NEXT:    scf.yield %52 : !quantum.bit
            // CHECK-NEXT:  } else {
            // CHECK-NEXT:    scf.yield %46 : !quantum.bit
            // CHECK-NEXT:  }
            // CHECK-NEXT:  quantum.dealloc_qb %18 : !quantum.bit
            // CHECK-NEXT:  quantum.dealloc_qb %27 : !quantum.bit
            // CHECK-NEXT:  quantum.dealloc_qb %37 : !quantum.bit
            // CHECK-NEXT:  quantum.dealloc_qb %14 : !quantum.bit
            // CHECK-NEXT:  func.return %51 : !quantum.bit
            // CHECK-NEXT:}
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_cnot_gate(self, run_filecheck):
        """Test for lowering a CNOT gate to a MBQC formalism."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK-NEXT: [[q0:%.+]],  [[q1:%.+]]= func.call @cnot_in_mbqc([[q0:%.+]], [[q1:%.+]]) : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
                %2, %3 = quantum.custom "CNOT"() %0, %1 : !quantum.bit, !quantum.bit
                return
            }
            // CHECK: func.func private @cnot_in_mbqc(%0 : !quantum.bit, %1 : !quantum.bit) -> (!quantum.bit, !quantum.bit) attributes {mbqc_transform = none} {
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_switch_statement(self, run_filecheck):
        """Test that the convert_to_mbqc_formalism_pass works correctly with a switch statement."""
        program = """
            func.func @test_func(%qubits: !quantum.bit, %l : index) {
                %0 = scf.index_switch %l -> !quantum.bit 
                case 0 {
                    // CHECK-NOT: quantum.custom "Hadamard"()
                    %q1 = quantum.custom "Hadamard"() %qubits : !quantum.bit
                    scf.yield %q1 : !quantum.bit
                }
                default {
                    // CHECK-NOT: quantum.custom "S"()
                    %q2 = quantum.custom "S"() %qubits : !quantum.bit
                    scf.yield %q2 : !quantum.bit
                }
                
                return
            }
        """
        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_function_no_body(self, run_filecheck):
        """Test that the convert_to_mbqc_formalism_pass works correctly with a function that has no body."""
        program = """
            func.func @test_func() {
                // CHECK: func.func private @func_1(f64, f64, i1) -> f64
                func.func private @func_1(f64, f64, i1) -> f64
                // CHECK: func.func private @func_2(memref<?xindex>, f64, f64, i1)
                func.func private @func_2(memref<?xindex>, f64, f64, i1)
                return
            }
        """
        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_lowering(self, run_filecheck_qjit):
        """Test that the convert_to_mbqc_formalism_pass works correctly with qjit and unrolled loops."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.qjit(
            target="mlir",
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @convert_to_mbqc_formalism_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-LABEL: circuit
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RZ"()
            # CHECK-NOT: quantum.custom "RotXZX"()
            # CHECK-NOT: quantum.custom "Hadamard"()
            # CHECK-NOT: scf.if
            # CHECK: scf.for
            # CHECK: func.call @hadamard_in_mbqc
            # CHECK: func.call @s_in_mbqc
            # CHECK: func.call @rotxzx_in_mbqc
            # CHECK: func.call @rz_in_mbqc
            # CHECK: func.call @cnot_in_mbqc
            # CHECK: mbqc.graph_state_prep
            # CHECK: quantum.custom "CZ"()
            # CHECK: mbqc.measure_in_basis
            # CHECK: scf.if
            # CHECK: quantum.custom "PauliX"()
            # CHECK: quantum.custom "PauliZ"()
            # CHECK: quantum.dealloc_qb
            for i in range(50):
                qml.H(i)
                qml.S(i)
                RotXZX(0.1, 0.2, 0.3, wires=[i])
                qml.RZ(phi=0.1, wires=[i])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_lowering_for(self, run_filecheck_qjit):
        """Test that the convert_to_mbqc_formalism_pass works correctly with qjit and for-loop structure."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.for_loop(1, 1000, 1)
        def loop_for(i):
            qml.H(i)
            qml.S(i)
            RotXZX(0.1, 0.2, 0.3, wires=[i])
            qml.RZ(phi=0.1, wires=[i])

        @qml.qjit(
            target="mlir",
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @convert_to_mbqc_formalism_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-LABEL: circuit
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RZ"()
            # CHECK-NOT: quantum.custom "RotXZX"()
            # CHECK-NOT: quantum.custom "Hadamard"()
            # CHECK: scf.for
            # CHECK: func.call @hadamard_in_mbqc
            # CHECK: func.call @s_in_mbqc
            # CHECK: func.call @rotxzx_in_mbqc
            # CHECK: func.call @rz_in_mbqc
            # CHECK: func.call @cnot_in_mbqc
            # CHECK: mbqc.graph_state_prep
            # CHECK: quantum.custom "CZ"()
            # CHECK: mbqc.measure_in_basis
            # CHECK: scf.if
            # CHECK: quantum.custom "PauliX"()
            # CHECK: quantum.custom "PauliZ"()
            # CHECK: quantum.dealloc_qb
            loop_for()
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_lowering_graph_state_decomp(self, run_filecheck_qjit):
        """Test that the convert_to_mbqc_formalism_pass works correctly with qjit and for-loop structure."""
        dev = qml.device("null.qubit", wires=1000)

        def loop_for(i):
            qml.H(i)
            qml.S(i)
            RotXZX(0.1, 0.2, 0.3, wires=[i])
            qml.RZ(phi=0.1, wires=[i])

        @qml.qjit(
            target="mlir",
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RZ"()
            # CHECK-NOT: quantum.custom "RotXZX"()
            # CHECK-NOT: mbqc.graph_state_prep
            # CHECK: scf.for
            # CHECK: quantum.custom "Hadamard"()
            # CHECK: quantum.custom "CZ"()
            # CHECK: mbqc.measure_in_basis
            # CHECK: scf.if
            # CHECK: quantum.custom "PauliX"()
            # CHECK: quantum.custom "PauliZ"()
            # CHECK: quantum.dealloc_qb
            for i in range(1000):
                loop_for(i)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_lowering_while(self, run_filecheck_qjit):
        """Test that the convert_to_mbqc_formalism_pass works correctly with qjit and while-loop structure."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.while_loop(lambda i: i > 1000)
        def while_for(i):
            qml.H(i)
            qml.S(i)
            RotXZX(0.1, 0.2, 0.3, wires=[i])
            qml.RZ(phi=0.1, wires=[i])
            i = i + 1
            return i

        @qml.qjit(
            target="mlir",
            autograph=True,
        )
        @convert_to_mbqc_formalism_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RZ"()
            # CHECK-NOT: quantum.custom "RotXZX"()
            # CHECK-NOT: quantum.custom "Hadamard"()
            # CHECK: scf.while
            # CHECK: quantum.custom "CZ"()
            # CHECK: mbqc.measure_in_basis
            # CHECK: scf.if
            # CHECK: quantum.custom "PauliX"()
            # CHECK: quantum.custom "PauliZ"()
            # CHECK: quantum.dealloc_qb
            while_for(0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_e2e(self):
        """Test that the convert_to_mbqc_formalism_pass end to end on null.qubit."""
        dev = qml.device("null.qubit", wires=1000)

        @qml.qjit(
            target="mlir",
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @decompose_graph_state_pass
        @convert_to_mbqc_formalism_pass
        @measurements_from_samples_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            for i in range(1000):
                qml.H(i)
                qml.S(i)
                RotXZX(0.1, 0.2, 0.3, wires=[i])
                qml.RZ(phi=0.1, wires=[i])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        res = circuit()
        assert res == 1.0
