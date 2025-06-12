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
"""Unit test module for the ``decompose`` compilation pass in the PennyLane Python compiler."""


import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
filecheck = pytest.importorskip("filecheck")

# pylint: disable=wrong-import-position
from xdsl.dialects.test import Test

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect as Quantum
from pennylane.compiler.python_compiler.transforms.decompose import DecompositionTransformPass

gate_set_cnot_rotations = {"CNOT", "RX", "RY", "RZ"}


class TestDecompositionPass:
    """Unit tests for the DecompositionPass."""

    def test_single_rot_decomposition(self, run_filecheck):
        """Test that the Rot gate is decomposed into RZ and RY gates."""

        # gate_set = {"CNOT", "RX", "RY", "RZ"}
        #
        # Rot(0.5, 0.5, 0.5, wires=0)
        # ---->
        # RZ(0.5, wires=0)
        # RY(0.5, wires=0)
        # RZ(0.5, wires=0)

        program = """
    func.func @test_func() {
        %0 = arith.constant 5.000000e-01 : f64
        %1 = arith.constant 0 : i64
        %2 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        %3 = "quantum.extract"(%2) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit

        // CHECK: [[VALUE:%.*]] = arith.constant 5.000000e-01 : f64
        // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}>
        // CHECK-NEXT: [[QUBIT1:%.*]] = "quantum.extract"(%2) <{idx_attr = 1 : i64}>
        // CHECK-NEXT: [[QUBIT2:%.*]] = quantum.custom "RZ"([[VALUE]]) [[QUBIT1]]
        // CHECK-NEXT: [[QUBIT3:%.*]] = quantum.custom "RY"([[VALUE]]) [[QUBIT2]]
        // CHECK-NEXT: [[LASTQUBIT:%.*]] = quantum.custom "RZ"([[VALUE]]) [[QUBIT3]]
        %4 = "quantum.custom"(%0, %0, %0, %3) <{gate_name = "Rot", operandSegmentSizes = array<i32: 3, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, f64, f64, !quantum.bit) -> !quantum.bit
        }
    
    """

        ctx = xdsl.context.Context()
        ctx.allow_unregistered_dialects = True
        ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        ctx.load_dialect(xdsl.dialects.func.Func)
        ctx.load_dialect(xdsl.dialects.transform.Transform)
        ctx.load_dialect(xdsl.dialects.arith.Arith)
        ctx.load_dialect(Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass(
            (DecompositionTransformPass(gate_set=gate_set_cnot_rotations),)
        )
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_multiple_rot_decomposition(self, run_filecheck):
        """Test that multiple Rot gates are decomposed correctly into RZ and RY gates."""

        # gate_set = {"CNOT", "RX", "RY", "RZ"}
        #
        # Rot(0.1, 0.2, 0.3, wires=0)
        # Rot(0.4, 0.5, 0.6, wires=0)
        # Rot(0.1, 0.1, 0.1, wires=1)
        # ---->
        # RZ(0.1, wires=0)
        # RY(0.2, wires=0)
        # RZ(0.3, wires=0)
        # RZ(0.4, wires=0)
        # RY(0.5, wires=0)
        # RZ(0.6, wires=0)
        # RZ(0.1, wires=1)
        # RY(0.1, wires=1)
        # RZ(0.1, wires=1)

        program = """
    func.func @test_func() {
            %0 = arith.constant 6.000000e-01 : f64
            %1 = arith.constant 5.000000e-01 : f64
            %2 = arith.constant 4.000000e-01 : f64
            %3 = arith.constant 3.000000e-01 : f64
            %4 = arith.constant 2.000000e-01 : f64
            %5 = arith.constant 1.000000e-01 : f64
            %6 = arith.constant 0 : i64

            %7 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
            %8 = "quantum.extract"(%7) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit

            // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}>
            // CHECK-NEXT: [[QUBIT1:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 0 : i64}>
            // CHECK-NEXT: [[QUBIT2:%.*]] = quantum.custom "RZ"(%5)  [[QUBIT1]]
            // CHECK-NEXT: [[QUBIT3:%.*]] = quantum.custom "RY"(%4) [[QUBIT2]]
            // CHECK-NEXT: [[QUBIT4:%.*]] = quantum.custom "RZ"(%3) [[QUBIT3]]
            // CHECK-NEXT: [[QUBIT5:%.*]] = quantum.custom "RZ"(%2) [[QUBIT4]]
            // CHECK-NEXT: [[QUBIT6:%.*]] = quantum.custom "RY"(%1) [[QUBIT5]]
            // CHECK-NEXT: [[QUBIT7:%.*]] = quantum.custom "RZ"(%0) [[QUBIT6]]
            // CHECK-NEXT: [[QUBIT8:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 1 : i64}>
            // CHECK-NEXT: [[QUBIT9:%.*]] = quantum.custom "RZ"(%5) [[QUBIT8]]
            // CHECK-NEXT: [[QUBIT10:%.*]] = quantum.custom "RY"(%5) [[QUBIT9]]
            // CHECK-NEXT: [[QUBIT11:%.*]] = quantum.custom "RZ"(%5) [[QUBIT10]]
            // CHECK-NEXT: [[QUBIT12:%.*]] = "quantum.insert"([[QUBITREGISTER]], [[QUBIT7]]) <{idx_attr = 0 : i64}>
            // CHECK-NEXT: [[___:%.*]] = "quantum.insert"([[QUBIT12]], [[QUBIT11]]) <{idx_attr = 1 : i64}>

            %9 = "quantum.custom"(%5, %8) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %10 = "quantum.custom"(%4, %9) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %11 = "quantum.custom"(%3, %10) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %12 = "quantum.custom"(%2, %11) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %13 = "quantum.custom"(%1, %12) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %14 = "quantum.custom"(%0, %13) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %15 = "quantum.extract"(%7) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %16 = "quantum.custom"(%5, %15) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %17 = "quantum.custom"(%5, %16) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %18 = "quantum.custom"(%5, %17) <{gate_name = "RZ", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
            %19 = "quantum.insert"(%7, %14) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
            %20 = "quantum.insert"(%19, %18) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

        }
    """

        ctx = xdsl.context.Context()
        ctx.allow_unregistered_dialects = True
        ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        ctx.load_dialect(xdsl.dialects.func.Func)
        ctx.load_dialect(xdsl.dialects.transform.Transform)
        ctx.load_dialect(xdsl.dialects.arith.Arith)
        ctx.load_dialect(Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass(
            (DecompositionTransformPass(gate_set=gate_set_cnot_rotations),)
        )
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_rot_cnot_decomposition(self, run_filecheck):
        """Test that a circuit with Rot and CNOT gates is decomposed correctly."""

        # gate_set = {"CNOT", "RX", "RY", "RZ"}
        #
        # CNOT(wires=[0, 1])
        # CNOT(wires=[0, 2])
        # Rot(0.1, 0.2, 0.3, wires=0)
        # Rot(0.4, 0.5, 0.6, wires=1)
        # CNOT(wires=[1, 0])
        # CNOT(wires=[1, 2])
        # ---->
        # CNOT(wires=[0, 1])
        # CNOT(wires=[0, 2])
        # RZ(0.1, wires=0)
        # RY(0.2, wires=0)
        # RZ(0.3, wires=0)
        # RZ(0.4, wires=1)
        # RY(0.5, wires=1)
        # RZ(0.6, wires=1)
        # CNOT(wires=[1, 0])
        # CNOT(wires=[1, 2])

        program = """
    func.func @test_func() {

            %0 = arith.constant 6.000000e-01 : f64
            %1 = arith.constant 5.000000e-01 : f64
            %2 = arith.constant 4.000000e-01 : f64
            %3 = arith.constant 3.000000e-01 : f64
            %4 = arith.constant 2.000000e-01 : f64
            %5 = arith.constant 1.000000e-01 : f64
            %6 = arith.constant 0 : i64

            %7 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
            %8 = "quantum.extract"(%7) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
            %9 = "quantum.extract"(%7) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit

            // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
            // CHECK-NEXT: [[QUBIT1:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 0 : i64}>
            // CHECK-NEXT: [[QUBIT2:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 1 : i64}>
            // CHECK-NEXT: [[CNOT1:%.*]], [[CNOT2:%.*]] = quantum.custom "CNOT"() [[QUBIT1]], [[QUBIT2]]
            // CHECK-NEXT: [[QUBIT3:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 2 : i64}>
            // CHECK-NEXT: [[CNOT3:%.*]], [[CNOT4:%.*]] = quantum.custom "CNOT"() [[CNOT1]], [[QUBIT3]]
            // CHECK-NEXT: [[QUBIT4:%.*]] = quantum.custom "RZ"(%5) [[CNOT3]]
            // CHECK-NEXT: [[QUBIT5:%.*]] = quantum.custom "RY"(%4) [[QUBIT4]]
            // CHECK-NEXT: [[QUBIT6:%.*]] = quantum.custom "RZ"(%3) [[QUBIT5]]
            // CHECK-NEXT: [[QUBIT7:%.*]] = quantum.custom "RZ"(%2) [[CNOT2]]
            // CHECK-NEXT: [[QUBIT8:%.*]] = quantum.custom "RY"(%1) [[QUBIT7]]
            // CHECK-NEXT: [[QUBIT9:%.*]] = quantum.custom "RZ"(%0) [[QUBIT8]]
            // CHECK-NEXT: [[CNOT5:%.*]], [[CNOT6:%.*]] = quantum.custom "CNOT"() [[QUBIT9]], [[QUBIT6]]
            // CHECK-NEXT: [[CNOT7:%.*]], [[CNOT8:%.*]] = quantum.custom "CNOT"() [[CNOT5]], [[CNOT4]]
            // CHECK-NEXT: [[QUBIT10:%.*]] = "quantum.insert"([[QUBITREGISTER]], [[CNOT6]]) <{idx_attr = 0 : i64}> 
            // CHECK-NEXT: [[QUBIT11:%.*]] = "quantum.insert"([[QUBIT10]], [[CNOT7]]) <{idx_attr = 1 : i64}>
            // CHECK-NEXT: [[___:%.*]] = "quantum.insert"([[QUBIT11]], [[CNOT8]]) <{idx_attr = 2 : i64}>
            
            %10, %11 = "quantum.custom"(%8, %9) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            %12 = "quantum.extract"(%7) <{idx_attr = 2 : i64}> : (!quantum.reg) -> !quantum.bit
            %13, %14 = "quantum.custom"(%10, %12) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            %15 = "quantum.custom"(%5, %4, %3, %13) <{gate_name = "Rot", operandSegmentSizes = array<i32: 3, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, f64, f64, !quantum.bit) -> !quantum.bit
            %16 = "quantum.custom"(%2, %1, %0, %11) <{gate_name = "Rot", operandSegmentSizes = array<i32: 3, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, f64, f64, !quantum.bit) -> !quantum.bit
            %17, %18 = "quantum.custom"(%16, %15) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            %19, %20 = "quantum.custom"(%17, %14) <{gate_name = "CNOT", operandSegmentSizes = array<i32: 0, 2, 0, 0>, resultSegmentSizes = array<i32: 2, 0>}> : (!quantum.bit, !quantum.bit) -> (!quantum.bit, !quantum.bit)
            %21 = "quantum.insert"(%7, %18) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
            %22 = "quantum.insert"(%21, %19) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
            %23 = "quantum.insert"(%22, %20) <{idx_attr = 2 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

        }
    """

        ctx = xdsl.context.Context()
        ctx.allow_unregistered_dialects = True
        ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        ctx.load_dialect(xdsl.dialects.func.Func)
        ctx.load_dialect(xdsl.dialects.transform.Transform)
        ctx.load_dialect(xdsl.dialects.arith.Arith)
        ctx.load_dialect(Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass(
            (DecompositionTransformPass(gate_set=gate_set_cnot_rotations),)
        )
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_hadamard_decomposition(self, run_filecheck):
        """Test that the Hadamard gate is decomposed into RZ and RY gates."""

        # gate_set = {"CNOT", "RX", "RY", "RZ"}
        #
        # Hadamard(wires=0)
        # Hadamard(wires=0)
        # Hadamard(wires=1)
        # ---->
        # RZ(0.5 * pi, wires=0)
        # RX(0.5 * pi, wires=0)
        # RZ(0.5 * pi, wires=0)
        # RZ(0.5 * pi, wires=0)
        # RX(0.5 * pi, wires=0)
        # RZ(0.5 * pi, wires=0)
        # RZ(0.5 * pi, wires=1)
        # RX(0.5 * pi, wires=1)
        # RZ(0.5 * pi, wires=1)

        program = """
    func.func @test_func() {

            %0 = arith.constant 0 : i64

            %1 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
            %2 = "quantum.extract"(%1) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit

            // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
            // CHECK-NEXT: [[QUBIT1:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 0 : i64}>
            // CHECK-NEXT: [[HALFPI:%.*]] = arith.constant 1.5707963267948966 : f64
            // CHECK-NEXT: [[QUBIT2:%.*]] = quantum.custom "RZ"([[HALFPI]]) [[QUBIT1]]
            // CHECK-NEXT: [[QUBIT3:%.*]] = quantum.custom "RX"([[HALFPI]]) [[QUBIT2]]
            // CHECK-NEXT: [[QUBIT4:%.*]] = quantum.custom "RZ"([[HALFPI]]) [[QUBIT3]]
            // CHECK-NEXT: [[QUBIT5:%.*]] = quantum.custom "RZ"([[HALFPI]]) [[QUBIT4]]
            // CHECK-NEXT: [[QUBIT6:%.*]] = quantum.custom "RX"([[HALFPI]]) [[QUBIT5]]
            // CHECK-NEXT: [[QUBIT7:%.*]] = quantum.custom "RZ"([[HALFPI]]) [[QUBIT6]]
            // CHECK-NEXT: [[QUBIT8:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 1 : i64}>
            // CHECK-NEXT: [[QUBIT9:%.*]] = quantum.custom "RZ"([[HALFPI]]) [[QUBIT8]]
            // CHECK-NEXT: [[QUBIT10:%.*]] = quantum.custom "RX"([[HALFPI]]) [[QUBIT9]]
            // CHECK-NEXT: [[QUBIT11:%.*]] = quantum.custom "RZ"([[HALFPI]]) [[QUBIT10]]
            // CHECK-NEXT: [[QUBIT12:%.*]] = "quantum.insert"([[QUBITREGISTER]], [[QUBIT7]]) <{idx_attr = 0 : i64}>
            // CHECK-NEXT: [[___:%.*]] = "quantum.insert"([[QUBIT12]], [[QUBIT11]]) <{idx_attr = 1 : i64}>
            
            
            %3 = quantum.custom "Hadamard"() %2 : !quantum.bit
            %4 = quantum.custom "Hadamard"() %3 : !quantum.bit
            %5 = "quantum.extract"(%1) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
            %6 = quantum.custom "Hadamard"() %5 : !quantum.bit
            %7 = "quantum.insert"(%1, %4) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
            %8 = "quantum.insert"(%7, %6) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

        }
    """

        ctx = xdsl.context.Context()
        ctx.allow_unregistered_dialects = True
        ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        ctx.load_dialect(xdsl.dialects.func.Func)
        ctx.load_dialect(xdsl.dialects.transform.Transform)
        ctx.load_dialect(xdsl.dialects.arith.Arith)
        ctx.load_dialect(Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass(
            (DecompositionTransformPass(gate_set=gate_set_cnot_rotations),)
        )
        pipeline.apply(ctx, module)

        run_filecheck(program, module)


class TestDecompositionPassAdjoint:
    """Test the decomposition of adjoint gates."""

    def test_adjoint_S_decomposition(self, run_filecheck):
        """Test that the adjoint of the S gate is decomposed correctly."""

        # gate_set = {"CNOT", "RX", "RY", "RZ"}, max_expansion = 1
        #
        # qml.adjoint(qml.S)(wires=0)
        # ---->
        # Adjoint(PhaseShift(0.5 * pi, wires=0))

        program = """
    func.func @test_func() {

        %0 = arith.constant 0 : i64
      
        %1 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        %2 = "quantum.extract"(%1) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit

        // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        // CHECK-NEXT: [[QUBIT1:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 0 : i64}>
        // CHECK-NEXT: [[HALFPI:%.*]] = arith.constant 1.5707963267948966 : f64
        // CHECK-NEXT: [[QUBIT2:%.*]] = quantum.custom "PhaseShift"([[HALFPI]]) [[QUBIT1]] adj
        // CHECK-NEXT: [[QUBIT3:%.*]] = "quantum.insert"([[QUBITREGISTER]], [[QUBIT2]]) <{idx_attr = 0 : i64}>

        %3 = quantum.custom "S"() %2 adj : !quantum.bit
        %4 = "quantum.insert"(%1, %3) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

        }
    """

        ctx = xdsl.context.Context()
        ctx.allow_unregistered_dialects = True
        ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        ctx.load_dialect(xdsl.dialects.func.Func)
        ctx.load_dialect(xdsl.dialects.transform.Transform)
        ctx.load_dialect(xdsl.dialects.arith.Arith)
        ctx.load_dialect(Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass(
            (DecompositionTransformPass(gate_set=gate_set_cnot_rotations, max_expansion=1),)
        )
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_toffoli_decomposition(self, run_filecheck):
        """Test that the Toffoli gate is decomposed correctly if decomposition contains adjoint gates."""

        # gate_set = {"Hadamard", "CNOT", "T"}, max_expansion = 1
        #
        # Toffoli(wires=[0, 1, 2])
        # ---->
        # Hadamard(2)
        # CNOT(wires=[1, 2])
        # Adjoint(T(2))
        # CNOT(wires=[0, 2])
        # T(2)
        # CNOT(wires=[1, 2])
        # Adjoint(T(2))
        # CNOT(wires=[0, 2])
        # T(2)
        # T(1)
        # CNOT(wires=[0, 1])
        # H(2)
        # T(0)
        # Adjoint(T(1))
        # CNOT(wires=[0, 1])

        program = """
    func.func @test_func() {

        %0 = arith.constant 0 : i64
      
        %1 = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        %2 = "quantum.extract"(%1) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
        %3 = "quantum.extract"(%1) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
        %4 = "quantum.extract"(%1) <{idx_attr = 2 : i64}> : (!quantum.reg) -> !quantum.bit

        // CHECK: [[QUBITREGISTER:%.*]] = "quantum.alloc"() <{nqubits_attr = 3 : i64}> : () -> !quantum.reg
        // CHECK-NEXT: [[QUBIT1:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 0 : i64}>
        // CHECK-NEXT: [[QUBIT2:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 1 : i64}>
        // CHECK-NEXT: [[QUBIT3:%.*]] = "quantum.extract"([[QUBITREGISTER]]) <{idx_attr = 2 : i64}>
        // CHECK-NEXT: [[QUBIT4:%.*]] = quantum.custom "Hadamard"() [[QUBIT3]]
        // CHECK-NEXT: [[QUBIT5:%.*]], [[QUBIT6:%.*]] = quantum.custom "CNOT"() [[QUBIT2]], [[QUBIT4]]
        // CHECK-NEXT: [[QUBIT7:%.*]] = quantum.custom "T"() [[QUBIT6]] adj
        // CHECK-NEXT: [[QUBIT8:%.*]], [[QUBIT9:%.*]] = quantum.custom "CNOT"() [[QUBIT1]], [[QUBIT7]]
        // CHECK-NEXT: [[QUBIT10:%.*]] = quantum.custom "T"() [[QUBIT9]]
        // CHECK-NEXT: [[QUBIT11:%.*]], [[QUBIT12:%.*]] = quantum.custom "CNOT"() [[QUBIT5]], [[QUBIT10]]
        // CHECK-NEXT: [[QUBIT13:%.*]] = quantum.custom "T"() [[QUBIT12]] adj
        // CHECK-NEXT: [[QUBIT14:%.*]], [[QUBIT15:%.*]] = quantum.custom "CNOT"() [[QUBIT8]], [[QUBIT13]]
        // CHECK-NEXT: [[QUBIT16:%.*]] = quantum.custom "T"() [[QUBIT15]]
        // CHECK-NEXT: [[QUBIT17:%.*]] = quantum.custom "T"() [[QUBIT11]]
        // CHECK-NEXT: [[QUBIT18:%.*]], [[QUBIT19:%.*]] = quantum.custom "CNOT"() [[QUBIT14]], [[QUBIT17]]
        // CHECK-NEXT: [[QUBIT20:%.*]] = quantum.custom "Hadamard"() [[QUBIT16]]
        // CHECK-NEXT: [[QUBIT21:%.*]] = quantum.custom "T"() [[QUBIT18]]
        // CHECK-NEXT: [[QUBIT22:%.*]] = quantum.custom "T"() [[QUBIT19]] adj
        // CHECK-NEXT: [[QUBIT23:%.*]], [[QUBIT24:%.*]] = quantum.custom "CNOT"() [[QUBIT21]], [[QUBIT22]]
        // CHECK-NEXT: [[QUBIT25:%.*]] = "quantum.insert"([[QUBITREGISTER]], [[QUBIT23]]) <{idx_attr = 0 : i64}>
        // CHECK-NEXT: [[QUBIT26:%.*]] = "quantum.insert"([[QUBIT25]], [[QUBIT24]]) <{idx_attr = 1 : i64}>
        // CHECK-NEXT: [[___:%.*]] = "quantum.insert"([[QUBIT26]], [[QUBIT20]]) <{idx_attr = 2 : i64}>
        

        %5, %6, %7 = quantum.custom "Toffoli"() %2, %3, %4 : !quantum.bit, !quantum.bit, !quantum.bit
        %8 = "quantum.insert"(%1, %5) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
        %9 = "quantum.insert"(%8, %6) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
        %10 = "quantum.insert"(%9, %7) <{idx_attr = 2 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

        }
    """

        ctx = xdsl.context.Context()
        ctx.allow_unregistered_dialects = True
        ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        ctx.load_dialect(xdsl.dialects.func.Func)
        ctx.load_dialect(xdsl.dialects.transform.Transform)
        ctx.load_dialect(xdsl.dialects.arith.Arith)
        ctx.load_dialect(Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()

        pipeline = xdsl.passes.PipelinePass(
            (DecompositionTransformPass(gate_set=gate_set_cnot_rotations, max_expansion=1),)
        )
        pipeline.apply(ctx, module)

        run_filecheck(program, module)
