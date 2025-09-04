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
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    ConvertToMBQCFormalismPass,
    convert_to_mbqc_formalism_pass,
    measurements_from_samples_pass,
)
from pennylane.compiler.python_compiler.transforms.convert_to_mbqc_formalism import _generate_graph
from pennylane.ftqc import RotXZX


class TestConvertToMBQCFormalismPass:
    """Unit tests for ConvertToMBQCFormalismPass."""

    def test_generate_graph_unsupported_gate(self):
        """Test that error raised for unsupported gates."""
        with pytest.raises(NotImplementedError):
            _generate_graph("IsingXY")

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
                // CHECK-NEXT: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK-NEXT: [[m2:%.+]], [[qb1:%.+]] = mbqc.measure_in_basis[XY, [[qb1:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK-NEXT: [[m3:%.+]], [[qb2:%.+]] = mbqc.measure_in_basis[XY, [[qb2:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK-NEXT: [[m4:%.+]], [[qb3:%.+]] = mbqc.measure_in_basis[XY, [[qb3:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[m13:%.+]] = arith.xori [[m1:%.+]], [[m3:%.+]] : i1
                // CHECK-NEXT: [[m134:%.+]] = arith.xori [[m13:%.+]], [[m4:%.+]] : i1
                // CHECK-NEXT: [[qb4_x_res:%.+]] = scf.if [[m134:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK-NEXT: [[m23:%.+]] = arith.xori [[m2:%.+]], [[m3:%.+]] : i1
                // CHECK-NEXT: [[qb4_res:%.+]] = scf.if [[m23:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
                return
            }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_s_gate(self, run_filecheck):
        """Test for lowering a S gate to a MBQC formalism."""
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
                // CHECK-NEXT: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK-NEXT: [[m2:%.+]], [[qb1:%.+]] = mbqc.measure_in_basis[XY, [[qb1:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK-NEXT: [[m3:%.+]], [[qb2:%.+]] = mbqc.measure_in_basis[XY, [[qb2:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK-NEXT: [[m4:%.+]], [[qb3:%.+]] = mbqc.measure_in_basis[XY, [[qb3:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[m24:%.+]] = arith.xori [[m2:%.+]], [[m4:%.+]] : i1
                // CHECK-NEXT: [[qb4_x_res:%.+]] = scf.if [[m24:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK-NEXT: [[m12:%.+]] = arith.xori [[m1:%.+]], [[m2:%.+]] : i1
                // CHECK-NEXT: [[m123:%.+]] = arith.xori [[m12:%.+]], [[m3:%.+]] : i1
                // CHECK-NEXT: [[cst_ture:%.+]] = arith.constant true
                // CHECK-NEXT: [[m1231:%.+]] = arith.xori [[m123:%.+]], [[cst_ture:%.+]] : i1
                // CHECK-NEXT: [[qb4_res:%.+]] = scf.if [[m1231:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                %1 = quantum.custom "S"() %0 : !quantum.bit
                return
            }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_rz_gate(self, run_filecheck):
        """Test for lowering a RZ gate to a MBQC formalism."""
        program = """
            func.func @test_func(%param0: f64) {
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
                // CHECK-NEXT: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK-NEXT: [[m2:%.+]], [[qb1:%.+]] = mbqc.measure_in_basis[XY, [[qb1:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[m3:%.+]], [[qb2:%.+]] = scf.if [[m2:%.+]] -> (i1, !quantum.bit) {
                // CHECK-NEXT: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[param0:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: [[neg_param0:%.+]] = arith.negf [[param0:%.+]] : f64
                // CHECK-NEXT: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param0:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: }
                // CHECK-NEXT: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK-NEXT: [[m4:%.+]], [[qb3:%.+]] = mbqc.measure_in_basis[XY, [[qb3:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: [[m24:%.+]] = arith.xori [[m2:%.+]], [[m4:%.+]] : i1
                // CHECK-NEXT: [[qb4_x_res:%.+]] = scf.if [[m24:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK-NEXT: [[m13:%.+]] = arith.xori [[m1:%.+]], [[m3:%.+]] : i1
                // CHECK-NEXT: [[qb4_res:%.+]] = scf.if [[m13:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                %1 = quantum.custom "RZ"(%param0) %0 : !quantum.bit
                return
            }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_rotxzx_gate(self, run_filecheck):
        """Test for lowering a RotXZX gate to a MBQC formalism."""
        program = """
            func.func @test_func(%param0: f64, %param1: f64, %param2: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
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
                // CHECK-NEXT: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                
                // CHECK-NEXT: [[m2:%.+]], [[qb1:%.+]] = scf.if [[m1:%.+]] -> (i1, !quantum.bit) {
                // CHECK-NEXT: [[m2_res:%.+]], [[qb1_res:%.+]] = mbqc.measure_in_basis[XY, [[param0:%.+]]] [[qb1:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m2_res:%.+]], [[qb1_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: [[neg_param0:%.+]] = arith.negf [[param0:%.+]] : f64
                // CHECK-NEXT: [[m2_res:%.+]], [[qb1_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param0:%.+]]] [[qb1:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m2_res:%.+]], [[qb1_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: }
                
                // CHECK-NEXT: [[m3:%.+]], [[qb2:%.+]] = scf.if [[m2:%.+]] -> (i1, !quantum.bit) {
                // CHECK-NEXT: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[param1:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: [[neg_param1:%.+]] = arith.negf [[param1:%.+]] : f64
                // CHECK-NEXT: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param1:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: }
                
                // CHECK-NEXT: [[m1_xor_m3:%.+]] = arith.xori [[m1:%.+]], [[m3:%.+]] : i1
                // CHECK-NEXT: [[m4:%.+]], [[qb3:%.+]] = scf.if [[m1_xor_m3:%.+]] -> (i1, !quantum.bit) {
                // CHECK-NEXT: [[m4_res:%.+]], [[qb3_res:%.+]] = mbqc.measure_in_basis[XY, [[param2:%.+]]] [[qb3:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m4_res:%.+]], [[qb3_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: [[neg_param2:%.+]] = arith.negf [[param2:%.+]] : f64
                // CHECK-NEXT: [[m4_res:%.+]], [[qb3_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param2:%.+]]] [[qb3:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: scf.yield [[m4_res:%.+]], [[qb3_res:%.+]] : i1, !quantum.bit
                // CHECK-NEXT: }
                
                
                // CHECK-NEXT: [[m24:%.+]] = arith.xori [[m2:%.+]], [[m4:%.+]] : i1
                // CHECK-NEXT: [[qb4_x_res:%.+]] = scf.if [[m24:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK-NEXT: [[m13:%.+]] = arith.xori [[m1:%.+]], [[m3:%.+]] : i1
                // CHECK-NEXT: [[qb4_res:%.+]] = scf.if [[m13:%.+]] -> (!quantum.bit) {
                // CHECK-NEXT: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: } else {
                // CHECK-NEXT: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK-NEXT: }
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.custom "RotXZX"(%param0, %param1, %param2) %0 : !quantum.bit
                return
            }
        """

        pipeline = (ConvertToMBQCFormalismPass(),)
        run_filecheck(program, pipeline)

    def test_cnot_gate(self, run_filecheck):
        """Test for lowering a CNOT gate to a MBQC formalism."""
        program = """
            func.func @test_func() {
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q9:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: quantum.custom "CNOT"()
                %2, %3 = quantum.custom "CNOT"() %0, %1 : !quantum.bit, !quantum.bit
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
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
            autograph=False,
        )
        @convert_to_mbqc_formalism_pass
        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RZ"()
            # CHECK-NOT: quantum.custom "RotXZX"()
            # CHECK-NOT: scf.for
            # CHECK: quantum.custom "Hadamard"()
            # CHECK: quantum.custom "CZ"()
            # CHECK: mbqc.measure_in_basis
            # CHECK: scf.if
            # CHECK: quantum.custom "PauliX"()
            # CHECK: quantum.custom "PauliZ"()
            # CHECK: quantum.dealloc_qb
            for i in range(1000):
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
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
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
            # CHECK: scf.for
            # CHECK: quantum.custom "Hadamard"()
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
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
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
            # CHECK: scf.while
            # CHECK: quantum.custom "Hadamard"()
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

    @pytest.mark.xfail(
        reason="Failure due to the deallocation of qubits in a qreg and insertion of qubit into a qreg is not supported yet."
    )
    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_e2e(self):
        """Test that the convert_to_mbqc_formalism_pass end to end on null.qubit."""
        dev = qml.device("null.qubit", wires=1000, shots=100000)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @convert_to_mbqc_formalism_pass
        @measurements_from_samples_pass
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
