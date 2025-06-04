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
"""Unit test module for the iterative cancel inverses transform"""

import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position

from xdsl.context import Context
from xdsl.dialects import arith, func, test

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect as Quantum
from pennylane.compiler.python_compiler.transforms import IterativeCancelInversesPass


class TestIterativeCancelInversesPass:
    """Unit tests for IterativeCancelInversesPass."""

    def test_no_inverses_same_qubit(self, run_filecheck):
        """Test that nothing changes when there are no inverses."""
        program = """
            func.func @test_func() {
                // CHECK: [[VAL1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[VAL2:%.*]] = "quantum.custom"([[VAL1:%.*]]) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                // CHECK: "quantum.custom"([[VAL2:%.*]]) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %1 = "quantum.custom"(%0) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%1) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_inverses_different_qubits(self, run_filecheck):
        """Test that nothing changes when there are no inverses."""
        program = """
            func.func @test_func() {
                // CHECK: [[VAL1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[VAL2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK: "quantum.custom"([[VAL1:%.*]]) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                // CHECK: "quantum.custom"([[VAL2:%.*]]) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%0) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %3 = "quantum.custom"(%1) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_simple_self_inverses(self, run_filecheck):
        """Test that inverses are cancelled."""
        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: quantum.custom
                %1 = "quantum.custom"(%0) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%1) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_nested_self_inverses(self, run_filecheck):
        """Test that nested self-inverses are cancelled."""
        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: quantum.custom
                %1 = "quantum.custom"(%0) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%1) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %3 = "quantum.custom"(%2) <{gate_name = "PauliZ", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %4 = "quantum.custom"(%3) <{gate_name = "PauliZ", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %5 = "quantum.custom"(%4) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %6 = "quantum.custom"(%5) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_cancel_ops_with_control_qubits(self, run_filecheck):
        """Test that ops with control qubits can be cancelled."""
        program = """
            func.func @test_func() {
                %0 = "arith.constant"() <{value = true}> : () -> i1
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: "quantum.custom"
                %4:3 = "quantum.custom"(%1, %2, %3, %0, %0) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (!quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                %5:3 = "quantum.custom"(%4#0, %4#1, %4#2, %0, %0) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (!quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_cancel_ops_with_same_control_qubits_and_values(self, run_filecheck):
        """Test that ops with control qubits and control values can be
        cancelled."""
        program = """
            func.func @test_func() {
                %0 = arith.constant false
                %1 = arith.constant true
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit
                %4 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: "quantum.custom"
                %5:3 = "quantum.custom"(%2, %3, %4, %1, %0) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (!quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                %6:3 = "quantum.custom"(%5#0, %5#1, %5#2, %1, %0) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (!quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_ops_with_control_qubits_different_control_values(self, run_filecheck):
        """Test that ops with the same control qubits but different
        control values don't cancel."""
        program = """
            func.func @test_func() {
                // CHECK-DAG: [[VAL1:%.*]] = arith.constant false
                // CHECK-DAG: [[VAL2:%.*]] = arith.constant true
                %0 = arith.constant false
                %1 = arith.constant true
                // CHECK: [[Q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[Q2:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[Q3:%.*]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit
                %4 = "test.op"() : () -> !quantum.bit
                // CHECK: [[Q4:%.*]], [[Q5:%.*]], [[Q6:%.*]] = "quantum.custom"([[Q1:%.*]], [[Q2:%.*]], [[Q3:%.*]], [[VAL2:%.*]], [[VAL1:%.*]]) <{gate_name = "PauliY"
                // CHECK: "quantum.custom"([[Q4:%.*]], [[Q5:%.*]], [[Q6:%.*]], [[VAL1:%.*]], [[VAL2:%.*]]) <{gate_name = "PauliY"
                %5:3 = "quantum.custom"(%2, %3, %4, %1, %0) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (!quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                %6:3 = "quantum.custom"(%5#0, %5#1, %5#2, %0, %1) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 2, 2>, resultSegmentSizes = array<i32: 1, 2>}> : (!quantum.bit, !quantum.bit, !quantum.bit, i1, i1) -> (!quantum.bit, !quantum.bit, !quantum.bit)
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_non_consecutive_self_inverse_ops(self, run_filecheck):
        """Test that self-inverse gates on the same qubit that are not
        consecutive are not cancelled."""
        program = """
            func.func @test_func() {
                // CHECK: [[VAL1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[VAL2:%.*]] = "quantum.custom"([[VAL1:%.*]]) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                // CHECK: [[VAL3:%.*]] = "quantum.custom"([[VAL2:%.*]]) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                // CHECK: "quantum.custom"([[VAL3:%.*]]) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %1 = "quantum.custom"(%0) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %2 = "quantum.custom"(%1) <{gate_name = "PauliY", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                %3 = "quantum.custom"(%2) <{gate_name = "PauliX", operandSegmentSizes = array<i32: 0, 1, 0, 0>, resultSegmentSizes = array<i32: 1, 0>}> : (!quantum.bit) -> !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((IterativeCancelInversesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
