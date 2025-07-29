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

import pennylane as qml

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath
from pennylane.compiler.python_compiler.transforms import (
    ConvertToMBQCFormalismPass,
    convert_to_mbqc_formalism_pass,
)

class TestConvertToMBQCFormalismPass:
    """Unit tests for ConvertToMBQCFormalismPass."""

    def test_hadamard_gate(self, run_filecheck):
        """Test for lowering a Hadamard gate to a MBQC formalism."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[qb1:%.+]] = quantum.alloc_qb : !quantum.bit
                // CHECK: [[qb2:%.+]] = quantum.alloc_qb : !quantum.bit
                // CHECK: [[qb3:%.+]] = quantum.alloc_qb : !quantum.bit
                // CHECK: [[qb4:%.+]] = quantum.alloc_qb : !quantum.bit
                // CHECK: [[qb1:%.+]] = quantum.custom "Hadamard"() [[qb1:%.+]] : !quantum.bit
                // CHECK: [[qb2:%.+]] = quantum.custom "Hadamard"() [[qb2:%.+]] : !quantum.bit
                // CHECK: [[qb3:%.+]] = quantum.custom "Hadamard"() [[qb3:%.+]] : !quantum.bit
                // CHECK: [[qb4:%.+]] = quantum.custom "Hadamard"() [[qb4:%.+]] : !quantum.bit
                // CHECK: [[qb1:%.+]], [[qb2:%.+]] = quantum.custom "CZ"() [[qb1:%.+]], [[qb2:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: [[qb2:%.+]], [[qb3:%.+]] = quantum.custom "CZ"() [[qb2:%.+]], [[qb3:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: [[qb3:%.+]], [[qb4:%.+]] = quantum.custom "CZ"() [[qb3:%.+]], [[qb4:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: [[q0:%.+]], [[qb1:%.+]] = quantum.custom "CZ"() [[q0:%.+]], [[qb1:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[cst_zero:%+]]] [[q0:%.+]] : i1, !quantum.bit
                // CHECK: [[half_pi:%.+]] = arith.constant {{1.57+}} : f64
                // CHECK: %27, %28 = mbqc.measure_in_basis[XY, %26] %22 : i1, !quantum.bit
                // CHECK: %29 = arith.constant 1.5707963267948966 : f64
                // CHECK: %30, %31 = mbqc.measure_in_basis[XY, %29] %17 : i1, !quantum.bit
                // CHECK: %32 = arith.constant 1.5707963267948966 : f64
                // CHECK: %33, %34 = mbqc.measure_in_basis[XY, %32] %19 : i1, !quantum.bit
                // CHECK: %35 = arith.addi %24, %30 : i1
                // CHECK: %36 = arith.addi %35, %33 : i1
                // CHECK: %37 = arith.constant true
                // CHECK: %38 = arith.xori %36, %37 : i1
                // CHECK: %39 = arith.constant true
                // CHECK: %40 = arith.cmpi eq, %38, %39 : i1
                // CHECK: %41 = scf.if %40 -> (!quantum.bit) {
                // CHECK:     %42 = quantum.custom "PauliX"() %20 : !quantum.bit
                // CHECK:     scf.yield %42 : !quantum.bit
                // CHECK: } else {
                // CHECK:     %43 = quantum.custom "Identity"() %20 : !quantum.bit
                // CHECK:     scf.yield %43 : !quantum.bit
                // CHECK: }
                // CHECK: %44 = arith.addi %27, %30 : i1
                // CHECK: %45 = arith.constant true
                // CHECK: %46 = arith.xori %44, %45 : i1
                // CHECK: %47 = arith.constant true
                // CHECK: %48 = arith.cmpi eq, %46, %47 : i1
                // CHECK: %49 = scf.if %48 -> (!quantum.bit) {
                // CHECK:     %50 = quantum.custom "PauliZ"() %41 : !quantum.bit
                // CHECK:     scf.yield %50 : !quantum.bit
                // CHECK: } else {
                // CHECK:     %51 = quantum.custom "Identity"() %41 : !quantum.bit
                // CHECK:     scf.yield %51 : !quantum.bit
                // CHECK: }
                // CHECK: %52 = quantum.custom "Identity"() %49 : !quantum.bit
                // CHECK: %53, %54 = quantum.custom "SWAP"() %25, %52 : !quantum.bit, !quantum.bit
                // CHECK: quantum.dealloc_qb %28 : !quantum.bit
                // CHECK: quantum.dealloc_qb %31 : !quantum.bit
                // CHECK: quantum.dealloc_qb %34 : !quantum.bit
                // CHECK: quantum.dealloc_qb %54 : !quantum.bit
                1% = quantum.custom "Hadamard" () %0 : !quantum.bit 
                return
            }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    # def test_allocop_aux_wires(self, run_filecheck):
    #     """Test that ConvertToMBQCFormalismPass can pre-allocate 13 aux wires. Note that
    #     this is a temporal solution.
    #     """
    #     program = """
    #         func.func @test_func() {
    #           // CHECK: %0 = "quantum.alloc"() <{nqubits_attr = 15 : i64}> : () -> !quantum.reg
    #           %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
    #           return
    #         }
    #     """

    #     ctx = Context()
    #     ctx.load_dialect(func.Func)
    #     ctx.load_dialect(test.Test)
    #     ctx.load_dialect(Quantum)

    #     module = xdsl.parser.Parser(ctx, program).parse_module()
    #     pipeline = xdsl.passes.PipelinePass((ConvertToMBQCFormalismPass(),))
    #     pipeline.apply(ctx, module)

    #     run_filecheck(program, module)

    # def test_qubit_mgr_1_qubit_gate(self, run_filecheck):
    #     """Test that ConvertToMBQCFormalismPass can extract and swap the target qubit and
    #     the result qubit in the auxiliary wires.
    #     """
    #     program = """
    #         func.func @test_func() {
    #           // CHECK: %0 = "quantum.alloc"() <{nqubits_attr = 15 : i64}> : () -> !quantum.reg
    #           %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
    #           // CHECK: %1 = "quantum.extract"() <{idx_attr = 0 : i64}> : () -> !quantum.bit
    #           %1 = "quantum.extract"() <{idx_attr = 0 : i64}> : () -> !quantum.bit
    #           // CHECK: %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    #           %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    #           // CHECK: %2 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %3 = "quantum.extract"(%0) <{idx_attr = 5 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %4 = "quantum.insert"(%0, %2) <{idx_attr = 5 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
    #           // CHECK: %5 = "quantum.insert"(%4, %3) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

    #           return
    #         }
    #     """

    #     ctx = Context()
    #     ctx.load_dialect(func.Func)
    #     ctx.load_dialect(test.Test)
    #     ctx.load_dialect(Quantum)

    #     module = xdsl.parser.Parser(ctx, program).parse_module()
    #     pipeline = xdsl.passes.PipelinePass((ConvertToMBQCFormalismPass(),))
    #     pipeline.apply(ctx, module)

    #     run_filecheck(program, module)

    # def test_qubit_mgr_cnot(self, run_filecheck):
    #     """Test that ConvertToMBQCFormalismPass can extract and swap the target qubit and
    #     the result qubit in the auxiliary wires.
    #     """
    #     program = """
    #         func.func @test_func() {
    #           // CHECK: %0 = "quantum.alloc"() <{nqubits_attr = 15 : i64}> : () -> !quantum.reg
    #           %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
    #           // CHECK: %1 = "quantum.extract"() <{idx_attr = 0 : i64}> : () -> !quantum.bit
    #           %1 = "quantum.extract"() <{idx_attr = 0 : i64}> : () -> !quantum.bit
    #           %2 = "quantum.extract"() <{idx_attr = 1 : i64}> : () -> !quantum.bit
    #           %out_qubits0, %out_qubits1 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    #           // CHECK: %3 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %4 = "quantum.extract"(%0) <{idx_attr = 13 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %5 = "quantum.insert"(%0, %3) <{idx_attr = 13 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
    #           // CHECK: %6 = "quantum.insert"(%5, %4) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
    #           // CHECK: %7 = "quantum.extract"(%6) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %8 = "quantum.extract"(%6) <{idx_attr = 14 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %9 = "quantum.insert"(%6, %7) <{idx_attr = 14 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
    #           // CHECK: %10 = "quantum.insert"(%9, %8) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

    #           return
    #         }
    #     """

    #     ctx = Context()
    #     ctx.load_dialect(func.Func)
    #     ctx.load_dialect(test.Test)
    #     ctx.load_dialect(Quantum)

    #     module = xdsl.parser.Parser(ctx, program).parse_module()
    #     pipeline = xdsl.passes.PipelinePass((ConvertToMBQCFormalismPass(),))
    #     pipeline.apply(ctx, module)

    #     run_filecheck(program, module)

    # @pytest.xfail(
    #     reason="This unit test is not complete and will be updated once PR#7888 is merged."
    # )
    # @pytest.mark.usefixtures("enable_disable_plxpr")
    # def test_qubit_mgr_integrate(self, run_filecheck):
    #     """Test that ConvertToMBQCFormalismPass can extract and swap the target qubit and
    #     the result qubit in the auxiliary wires.
    #     """
    #     program = """
    #         func.func @test_func() {
    #           // CHECK: %0 = "quantum.alloc"() <{nqubits_attr = 15 : i64}> : () -> !quantum.reg
    #           %0 = "quantum.alloc"() <{nqubits_attr = 2 : i64}> : () -> !quantum.reg
    #           // CHECK: %1 = "quantum.extract"() <{idx_attr = 0 : i64}> : () -> !quantum.bit
    #           %1 = "quantum.extract"() <{idx_attr = 0 : i64}> : () -> !quantum.bit
    #           %2 = "quantum.extract"() <{idx_attr = 1 : i64}> : () -> !quantum.bit
    #           %out_qubits0, %out_qubits1 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    #           // CHECK: %3 = "quantum.extract"(%0) <{idx_attr = 0 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %4 = "quantum.extract"(%0) <{idx_attr = 13 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %5 = "quantum.insert"(%0, %3) <{idx_attr = 13 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
    #           // CHECK: %6 = "quantum.insert"(%5, %4) <{idx_attr = 0 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
    #           // CHECK: %7 = "quantum.extract"(%6) <{idx_attr = 1 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %8 = "quantum.extract"(%6) <{idx_attr = 14 : i64}> : (!quantum.reg) -> !quantum.bit
    #           // CHECK: %9 = "quantum.insert"(%6, %7) <{idx_attr = 14 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg
    #           // CHECK: %10 = "quantum.insert"(%9, %8) <{idx_attr = 1 : i64}> : (!quantum.reg, !quantum.bit) -> !quantum.reg

    #           return
    #         }
    #     """

    #     dev = qml.device("lightning.qubit", wires=2)

    #     @qml.qjit(target="mlir", pass_plugins=[xdsl_plugin.getXDSLPluginAbsolutePath()])
    #     @convert_to_mbqc_formalism_pass
    #     @qml.qnode(dev)
    #     def circuit_ref():
    #         qml.H(0)
    #         qml.H(0)
    #         return qml.expval(qml.X(0))

    #     compiler = Compiler()
    #     mlir_module = compiler.run(circuit_ref.mlir_module)
    #     mod_str = mlir_module.operation.get_asm(
    #         binary=False, print_generic_op_form=True, assume_verified=True
    #     )
    #     xdsl_module = parse_generic_to_xdsl_module(mod_str)

    #     run_filecheck(program, xdsl_module)
