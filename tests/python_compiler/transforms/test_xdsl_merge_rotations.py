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
"""Unit test module for the merge rotations transform"""
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position
from xdsl.context import Context
from xdsl.dialects import arith, func, test

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect as Quantum
from pennylane.compiler.python_compiler.transforms import MergeRotationsPass


class TestMergeRotationsPass:
    """Unit tests for MergeRotationsPass."""

    def test_no_composable_ops(self, run_filecheck):
        """Test that nothing changes when there are no composable gates."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[Q2:%.*]] = "quantum.custom"(%arg0, [[Q1:%.*]]) <{gate_name = "RX"
                // CHECK: "quantum.custom"(%arg1, [[Q2:%.*]]) <{gate_name = "RY"
                %1 = "quantum.custom"(%arg0, %0) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%arg1, %1) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_composable_ops(self, run_filecheck):
        """Test that composable gates are merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[VAL1:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK: "quantum.custom"([[VAL1:%.*]], [[Q1:%.*]]) <{gate_name = "RX"
                // CHECK-NOT: "quantum.custom"
                %1 = "quantum.custom"(%arg0, %0) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%arg1, %1) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_many_composable_ops(self, run_filecheck):
        """Test that more than 2 composable ops are merged correctly."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) {
                // CHECK: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[VAL1:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK: [[VAL2:%.*]] = arith.addf [[VAL1:%.*]], %arg2 : f64
                // CHECK: [[VAL3:%.*]] = arith.addf [[VAL2:%.*]], %arg3 : f64
                // CHECK: "quantum.custom"([[VAL3:%.*]], [[Q1:%.*]]) <{gate_name = "RX"
                // CHECK-NOT: "quantum.custom"
                %1 = "quantum.custom"(%arg0, %0) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%arg1, %1) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %3 = "quantum.custom"(%arg2, %2) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %4 = "quantum.custom"(%arg3, %3) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_non_consecutive_composable_ops(self, run_filecheck):
        """Test that non-consecutive composable gates are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[Q2:%.*]] = "quantum.custom"(%arg0, [[Q1:%.*]]) <{gate_name = "RX"
                // CHECK: [[Q3:%.*]] = "quantum.custom"(%arg0, [[Q2:%.*]]) <{gate_name = "RY"
                // CHECK: "quantum.custom"(%arg1, [[Q3:%.*]]) <{gate_name = "RX"
                %1 = "quantum.custom"(%arg0, %0) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%arg0, %1) <{gate_name = "RY", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %3 = "quantum.custom"(%arg1, %2) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_composable_ops_different_qubits(self, run_filecheck):
        """Test that composable gates on different qubits are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[Q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK: "quantum.custom"(%arg0, [[Q1:%.*]]) <{gate_name = "RX"
                // CHECK: "quantum.custom"(%arg1, [[Q2:%.*]]) <{gate_name = "RX"
                %2 = "quantum.custom"(%arg0, %0) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                %3 = "quantum.custom"(%arg1, %1) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (f64, !quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_controlled_composable_ops(self, run_filecheck):
        """Test that controlled composable ops can be merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst = "arith.constant"() <{value = true}> : () -> i1
                // CHECK-DAG: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[Q2:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[Q3:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK: [[VAL1:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK-DAG: "quantum.custom"([[VAL1:%.*]], [[Q1:%.*]], [[Q2:%.*]], [[Q3:%.*]], %cst, %cst) <{gate_name = "RX"
                // CHECK-NOT: "quantum.custom"
                %3, %4, %5 = "quantum.custom"(%arg0, %0, %1, %2, %cst, %cst) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (f64, !quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                %6, %7, %8 = "quantum.custom"(%arg1, %3, %4, %5, %cst, %cst) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (f64, !quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_controlled_composable_ops_same_control_values(self, run_filecheck):
        """Test that controlled composable ops with the same control values
        can be merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst0 = "arith.constant"() <{value = true}> : () -> i1
                %cst1 = "arith.constant"() <{value = false}> : () -> i1
                // CHECK-DAG: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[Q2:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[Q3:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK: [[VAL1:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK-DAG: "quantum.custom"([[VAL1:%.*]], [[Q1:%.*]], [[Q2:%.*]], [[Q3:%.*]], %cst0, %cst1) <{gate_name = "RX"
                // CHECK-NOT: "quantum.custom"
                %3, %4, %5 = "quantum.custom"(%arg0, %0, %1, %2, %cst0, %cst1) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (f64, !quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                %6, %7, %8 = "quantum.custom"(%arg1, %3, %4, %5, %cst0, %cst1) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (f64, !quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_controlled_composable_ops_different_control_values(self, run_filecheck):
        """Test that controlled composable ops with different control values
        are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst0 = "arith.constant"() <{value = true}> : () -> i1
                %cst1 = "arith.constant"() <{value = false}> : () -> i1
                // CHECK-DAG: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[Q2:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[Q3:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[Q4:%.*]], [[Q5:%.*]], [[Q6:%.*]] = "quantum.custom"(%arg0, [[Q1:%.*]], [[Q2:%.*]], [[Q3:%.*]], %cst0, %cst1) <{gate_name = "RX"
                // CHECK-DAG: "quantum.custom"(%arg1, [[Q4:%.*]], [[Q5:%.*]], [[Q6:%.*]], %cst1, %cst0) <{gate_name = "RX"
                %3, %4, %5 = "quantum.custom"(%arg0, %0, %1, %2, %cst0, %cst1) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (f64, !quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                %6, %7, %8 = "quantum.custom"(%arg1, %3, %4, %5, %cst1, %cst0) <{gate_name = "RX", operandSegmentSizes = array<i32: 1, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (f64, !quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((MergeRotationsPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
