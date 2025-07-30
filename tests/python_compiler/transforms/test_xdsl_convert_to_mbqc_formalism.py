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

from catalyst.ftqc import mbqc_pipeline

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    ConvertToMBQCFormalismPass,
    convert_to_mbqc_formalism_pass,
    measurements_from_samples_pass,
)
from pennylane.ftqc import RotXZX


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
                // CHECK: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK: [[m2:%.+]], [[qb1:%.+]] = mbqc.measure_in_basis[XY, [[qb1:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK: [[m3:%.+]], [[qb2:%.+]] = mbqc.measure_in_basis[XY, [[qb2:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK: [[m4:%.+]], [[qb3:%.+]] = mbqc.measure_in_basis[XY, [[qb3:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK: [[m13:%.+]] = arith.addi [[m1:%.+]], [[m3:%.+]] : i1
                // CHECK: [[m134:%.+]] = arith.addi [[m13:%.+]], [[m4:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[x_idx:%.+]] = arith.xori [[m134:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[x_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_x_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4:%.+]] = quantum.custom "Identity"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[m23:%.+]] = arith.addi [[m2:%.+]], [[m3:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[z_idx:%.+]] = arith.xori [[m23:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[z_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "Identity"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[qb4_res:%.+]] = quantum.custom "Identity"() [[qb4_res:%.+]] : !quantum.bit
                // CHECK: [[q0_res:%.+]], [[qb4:%.+]] = quantum.custom "SWAP"() [[q0:%.+]], [[qb4_res:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb4:%.+]] : !quantum.bit
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
                // CHECK: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK: [[m2:%.+]], [[qb1:%.+]] = mbqc.measure_in_basis[XY, [[qb1:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK: [[half_pi:%.+]] = arith.constant {{1.+}} : f64
                // CHECK: [[m3:%.+]], [[qb2:%.+]] = mbqc.measure_in_basis[XY, [[qb2:%.+]]] [[half_pi:%.+]] : i1, !quantum.bit
                // CHECK: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK: [[m4:%.+]], [[qb3:%.+]] = mbqc.measure_in_basis[XY, [[qb3:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK: [[m24:%.+]] = arith.addi [[m2:%.+]], [[m4:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[x_idx:%.+]] = arith.xori [[m24:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[x_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_x_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4:%.+]] = quantum.custom "Identity"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[m12:%.+]] = arith.addi [[m1:%.+]], [[m2:%.+]] : i1
                // CHECK: [[m123:%.+]] = arith.addi [[m12:%.+]], [[m3:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[m1231:%.+]] = arith.addi [[m123:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[z_idx:%.+]] = arith.xori [[m1231:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[z_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "Identity"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[qb4_res:%.+]] = quantum.custom "Identity"() [[qb4_res:%.+]] : !quantum.bit
                // CHECK: [[q0_res:%.+]], [[qb4:%.+]] = quantum.custom "SWAP"() [[q0:%.+]], [[qb4_res:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb4:%.+]] : !quantum.bit
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
                // CHECK: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK: [[m2:%.+]], [[qb1:%.+]] = mbqc.measure_in_basis[XY, [[qb1:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[comp_op:%.+]] = arith.cmpi eq, [[m2:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[m3:%.+]], [[qb2:%.+]] = scf.if [[comp_op:%.+]] -> (i1, !quantum.bit) {
                // CHECK: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[param0:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK: } else {
                // CHECK: [[neg_param0:%.+]] = arith.negf [[param0:%.+]] : f64
                // CHECK: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param0:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK: }
                // CHECK: [[cst_zero:%.+]] = arith.constant {{0.+}} : f64
                // CHECK: [[m4:%.+]], [[qb3:%.+]] = mbqc.measure_in_basis[XY, [[qb3:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                // CHECK: [[m24:%.+]] = arith.addi [[m2:%.+]], [[m4:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[x_idx:%.+]] = arith.xori [[m24:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[x_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_x_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4:%.+]] = quantum.custom "Identity"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[m13:%.+]] = arith.addi [[m1:%.+]], [[m3:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[z_idx:%.+]] = arith.xori [[m13:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[z_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "Identity"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[qb4_res:%.+]] = quantum.custom "Identity"() [[qb4_res:%.+]] : !quantum.bit
                // CHECK: [[q0_res:%.+]], [[qb4:%.+]] = quantum.custom "SWAP"() [[q0:%.+]], [[qb4_res:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb4:%.+]] : !quantum.bit
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
                // CHECK: [[m1:%.+]], [[q0:%.+]] = mbqc.measure_in_basis[XY, [[q0:%.+]]] [[cst_zero:%.+]] : i1, !quantum.bit
                
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[comp_op:%.+]] = arith.cmpi eq, [[m1:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[m2:%.+]], [[qb1:%.+]] = scf.if [[comp_op:%.+]] -> (i1, !quantum.bit) {
                // CHECK: [[m2_res:%.+]], [[qb1_res:%.+]] = mbqc.measure_in_basis[XY, [[param0:%.+]]] [[qb1:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m2_res:%.+]], [[qb1_res:%.+]] : i1, !quantum.bit
                // CHECK: } else {
                // CHECK: [[neg_param0:%.+]] = arith.negf [[param0:%.+]] : f64
                // CHECK: [[m2_res:%.+]], [[qb1_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param0:%.+]]] [[qb1:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m2_res:%.+]], [[qb1_res:%.+]] : i1, !quantum.bit
                // CHECK: }
                
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[comp_op:%.+]] = arith.cmpi eq, [[m2:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[m3:%.+]], [[qb2:%.+]] = scf.if [[comp_op:%.+]] -> (i1, !quantum.bit) {
                // CHECK: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[param1:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK: } else {
                // CHECK: [[neg_param1:%.+]] = arith.negf [[param1:%.+]] : f64
                // CHECK: [[m3_res:%.+]], [[qb2_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param1:%.+]]] [[qb2:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m3_res:%.+]], [[qb2_res:%.+]] : i1, !quantum.bit
                // CHECK: }
                
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[comp_op:%.+]] = arith.cmpi eq, [[m3:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[m4:%.+]], [[qb3:%.+]] = scf.if [[comp_op:%.+]] -> (i1, !quantum.bit) {
                // CHECK: [[m4_res:%.+]], [[qb3_res:%.+]] = mbqc.measure_in_basis[XY, [[param2:%.+]]] [[qb3:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m4_res:%.+]], [[qb3_res:%.+]] : i1, !quantum.bit
                // CHECK: } else {
                // CHECK: [[neg_param2:%.+]] = arith.negf [[param2:%.+]] : f64
                // CHECK: [[m4_res:%.+]], [[qb3_res:%.+]] = mbqc.measure_in_basis[XY, [[neg_param2:%.+]]] [[qb3:%.+]] : i1, !quantum.bit
                // CHECK: scf.yield [[m4_res:%.+]], [[qb3_res:%.+]] : i1, !quantum.bit
                // CHECK: }
                
                
                // CHECK: [[m24:%.+]] = arith.addi [[m2:%.+]], [[m4:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[x_idx:%.+]] = arith.xori [[m24:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[x_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_x_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4:%.+]] = quantum.custom "PauliX"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4:%.+]] = quantum.custom "Identity"() [[qb4:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[m13:%.+]] = arith.addi [[m1:%.+]], [[m3:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[z_idx:%.+]] = arith.xori [[m13:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[cst_ture:%.+]] = arith.constant true
                // CHECK: [[cmp_res:%.+]] = arith.cmpi eq, [[z_idx:%.+]], [[cst_ture:%.+]] : i1
                // CHECK: [[qb4_res:%.+]] = scf.if [[cmp_res:%.+]] -> (!quantum.bit) {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "PauliZ"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: } else {
                // CHECK: [[qb4_x_res:%.+]] = quantum.custom "Identity"() [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: scf.yield [[qb4_x_res:%.+]] : !quantum.bit
                // CHECK: }
                // CHECK: [[qb4_res:%.+]] = quantum.custom "Identity"() [[qb4_res:%.+]] : !quantum.bit
                // CHECK: [[q0_res:%.+]], [[qb4:%.+]] = quantum.custom "SWAP"() [[q0:%.+]], [[qb4_res:%.+]] : !quantum.bit, !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb1:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb2:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb3:%.+]] : !quantum.bit
                // CHECK: quantum.dealloc_qb [[qb4:%.+]] : !quantum.bit
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
        """Test that the convert_to_mbqc_formalism_pass works correctly with qjit."""
        dev = qml.device("null.qubit", wires=1000, shots=100000)

        @qml.qjit(
            target="mlir",
            pass_plugins=[getXDSLPluginAbsolutePath()],
            pipelines=mbqc_pipeline(),
            autograph=True,
        )
        @convert_to_mbqc_formalism_pass
        @qml.qnode(dev)
        def circuit():
            # CHECK-NOT: quantum.custom "CNOT"()
            # CHECK-NOT: quantum.custom "S"()
            # CHECK-NOT: quantum.custom "RZ"()
            # CHECK-NOT: quantum.custom "RotXZX"()
            for i in range(1000):
                qml.H(i)
                qml.S(i)
                RotXZX(0.1, 0.2, 0.3, wires=[i])
                qml.RZ(phi=0.1, wires=[i])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.Z(wires=0))

        run_filecheck_qjit(circuit)

    @pytest.mark.usefixtures("enable_disable_plxpr")
    def test_gates_in_mbqc_gate_set_e2e(self, run_filecheck_qjit):
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
