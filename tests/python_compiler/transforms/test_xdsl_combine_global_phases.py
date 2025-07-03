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
"""Unit test module for the combine global phases transform"""
import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

# pylint: disable=wrong-import-position
from xdsl.context import Context
from xdsl.dialects import arith, func, scf, test

from pennylane.compiler.python_compiler.quantum_dialect import QuantumDialect as Quantum
from pennylane.compiler.python_compiler.transforms import CombineGlobalPhasesPass


class TestCombineGlobalPhasesPass:
    """Unit tests for CombineGlobalPhasesPass."""

    def test_combinable_ops_without_control_flow(self, run_filecheck):
        """Test that combine global phases in a func without control flows."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[phi_sum:%.*]] = arith.addf [[arg1:%.*]], [[arg0:%.*]] : f64
                // CHECK: quantum.gphase([[phi_sum:%.*]])
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                %2 = quantum.custom "RX"() %0 : !quantum.bit
                return
            }
        """

        ctx = Context()
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((CombineGlobalPhasesPass(),))
        pipeline.apply(ctx, module)

        run_filecheck(program, module)

    def test_combinable_ops_with_control_flow(self, run_filecheck):
        """Test that combine global phases in a func without control flows."""
        program = """
            func.func @test_func(%cond: i32, %arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                quantum.gphase %arg0
                %ret = scf.if %cond -> (f64) {
                    // CHECK: [[two:%.*]] = arith.constant [[two:2.*]] : f64
                    %two = arith.constant 2 : f64
                    // CHECK: [[arg02:%.*]] = arith.mulf [[arg0:%.*]], [[two:%.*]] : f64
                    %arg0x2 = arith.mulf %arg0, %two : f64
                    // CHECK: scf.yield [[arg02:%.*]] : f64
                    scf.yield %arg0x2 : f64
                } else {
                    // CHECK: [[two_1:%.*]] = arith.constant [[two_1:2.*]] : f64
                    %two_1 = arith.constant 2 : f64
                    // CHECK: [[arg12:%.*]] = arith.mulf [[arg1:%.*]], [[two_1:%.*]] : f64
                    %arg1x2 = arith.mulf %arg1, %two_1 : f64
                    // CHECK: scf.yield [[arg12:%.*]] : f64
                    scf.yield %arg1x2 : f64
                }
                // CHECK: [[phi_sum:%.*]] = arith.addf [[ret:%.*]], [[arg0:%.*]] : f64
                // CHECK: quantum.gphase([[phi_sum:%.*]]) :
                quantum.gphase %ret
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                %2 = quantum.custom "RX"() %0 : !quantum.bit
                return
            }
        """

        ctx = Context()
        # Load scf.Scf dialects to ensure operations (scf.if) in the program str are registered.
        ctx.load_dialect(scf.Scf)
        # Load arith.Arith dialects to ensure operations (arith.constant, f64) in the program str are registered.
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((CombineGlobalPhasesPass(),))

        pipeline.apply(ctx, module)
        run_filecheck(program, module)

    def test_combinable_ops_in_control_flow_if(self, run_filecheck):
        """Test that combine global phases in a func without control flows."""
        program = """
            func.func @test_func(%cond: i32, %arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[ret:%.*]] = scf.if [[cond:%.*]] -> (f64) {
                %ret = scf.if %cond -> (f64) {
                    // CHECK: [[two0:%.*]] = arith.constant [[two0:2.*]] : f64
                    %two0 = arith.constant 2 : f64
                    // CHECK: [[arg02:%.*]] = arith.mulf [[arg0:%.*]], [[two0:%.*]] : f64
                    %arg02 = arith.mulf %arg0, %two0 : f64
                    // CHECK: [[t0:%.*]] = "test.op"() : () -> f64
                    %t0 = "test.op"() : () -> f64
                    // CHECK: [[phi_sum:%.*]] = arith.addf [[t0:%.*]], [[t0:%.*]] : f64
                    // CHECK: quantum.gphase([[phi_sum:%.*]])
                    quantum.gphase %t0
                    quantum.gphase %t0
                    // CHECK: scf.yield [[arg02:%.*]] : f64
                    scf.yield %arg02 : f64
                    // CHECK: } else {
                } else {
                    // CHECK: [[two1:%.*]] = arith.constant [[two1:2.*]] : f64
                    %two1 = arith.constant 2 : f64
                    // CHECK: [[arg12:%.*]] = arith.mulf [[arg1:%.*]], [[two1:%.*]] : f64
                    %arg1x2 = arith.mulf %arg1, %two1 : f64
                    // CHECK: [[phi_sum:%.*]] = arith.addf [[arg1x2:%.*]], [[arg1x2:%.*]] : f64
                    // CHECK: quantum.gphase([[phi_sum:%.*]])
                    quantum.gphase %arg1x2
                    quantum.gphase %arg1x2
                    // CHECK: scf.yield [[arg12:%.*]] : f64
                    scf.yield %arg1x2 : f64
                    // CHECK: }
                }
                // CHECK: [[phi_sum:%.*]] = arith.addf [[arg0:%.*]], [[arg1:%.*]] : f64
                // CHECK: quantum.gphase([[phi_sum:%.*]])
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                %2 = quantum.custom "RX"() %0 : !quantum.bit
                return
            }
        """

        ctx = Context()
        # Load scf.Scf dialects to ensure operations (scf.if) in the program str are registered.
        ctx.load_dialect(scf.Scf)
        # Load arith.Arith dialects to ensure operations (arith.constant, f64) in the program str are registered.
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((CombineGlobalPhasesPass(),))

        pipeline.apply(ctx, module)
        run_filecheck(program, module)

    def test_combinable_ops_in_control_flow_for(self, run_filecheck):
        """Test that combine global phases in a func without control flows."""
        program = """
            func.func @test_func(%n: i32, %arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[c0:%*]] = arith.constant 0 : i32
                // CHECK: [[c1:%*]] = arith.constant 1 : i32
                %0 = "test.op"() : () -> !quantum.bit
                %c0 = arith.constant 0 : i32
                %c1 = arith.constant 1 : i32
                scf.for %i = %c0 to %n step %c1 {
                    // CHECK: [[two0:%.*]] = arith.constant [[two0:2.*]] : f64
                    %two0 = arith.constant 2 : f64
                    // CHECK: [[arg02:%.*]] = arith.mulf [[arg0:%.*]], [[two0:%.*]] : f64
                    %arg02 = arith.mulf %arg0, %two0 : f64
                    // CHECK: [[t0:%.*]] = "test.op"() : () -> f64
                    %t0 = "test.op"() : () -> f64
                    // CHECK: [[phi_sum:%.*]] = arith.addf [[t0:%.*]], [[t0:%.*]] : f64
                    // CHECK: quantum.gphase([[phi_sum:%.*]])
                    quantum.gphase %t0
                    quantum.gphase %t0
                }
                // CHECK: [[phi_sum:%.*]] = arith.addf [[arg0:%.*]], [[arg1:%.*]] : f64
                // CHECK: quantum.gphase([[phi_sum:%.*]])
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                %2 = quantum.custom "RX"() %0 : !quantum.bit
                return
            }
        """

        ctx = Context()
        # Load scf.Scf dialects to ensure operations (scf.if) in the program str are registered.
        ctx.load_dialect(scf.Scf)
        # Load arith.Arith dialects to ensure operations (arith.constant, f64) in the program str are registered.
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((CombineGlobalPhasesPass(),))

        pipeline.apply(ctx, module)
        run_filecheck(program, module)

    def test_combinable_ops_in_control_flow_while(self, run_filecheck):
        """Test that combine global phases in a func without control flows."""
        program = """
            func.func @test_func(%n: i32, %arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %c0 = arith.constant 0 : i32
                %c1 = arith.constant 1 : i32
                scf.while (%current_i = %n) : (i32) -> () {
                    %cond = arith.cmpi slt, %current_i, %n : i32
                    scf.condition(%cond) %current_i : i32
                } do {
                ^bb0(%i : i32):
                    %next_i = arith.addi %i, %c1 : i32
                    // CHECK: [[two0:%.*]] = arith.constant [[two0:2.*]] : f64
                    %two0 = arith.constant 2 : f64
                    // CHECK: [[arg02:%.*]] = arith.mulf [[arg0:%.*]], [[two0:%.*]] : f64
                    %arg02 = arith.mulf %arg0, %two0 : f64
                    // CHECK: [[t0:%.*]] = "test.op"() : () -> f64
                    %t0 = "test.op"() : () -> f64
                    // CHECK: [[phi_sum:%.*]] = arith.addf [[t0:%.*]], [[t0:%.*]] : f64
                    // CHECK: quantum.gphase([[phi_sum:%.*]])
                    quantum.gphase %t0
                    quantum.gphase %t0
                    scf.yield %next_i : i32
                }
                // CHECK: [[phi_sum:%.*]] = arith.addf [[arg0:%.*]], [[arg1:%.*]] : f64
                // CHECK: quantum.gphase([[phi_sum:%.*]]) :
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                %2 = quantum.custom "RX"() %0 : !quantum.bit
                return
            }
        """

        ctx = Context()
        # Load scf.Scf dialects to ensure operations (scf.if) in the program str are registered.
        ctx.load_dialect(scf.Scf)
        # Load arith.Arith dialects to ensure operations (arith.constant, f64) in the program str are registered.
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(func.Func)
        ctx.load_dialect(test.Test)
        ctx.load_dialect(Quantum)

        module = xdsl.parser.Parser(ctx, program).parse_module()
        pipeline = xdsl.passes.PipelinePass((CombineGlobalPhasesPass(),))

        pipeline.apply(ctx, module)
        run_filecheck(program, module)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
