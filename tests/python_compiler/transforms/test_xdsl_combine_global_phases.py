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
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                quantum.gphase %arg0
                quantum.gphase %arg1
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
                // CHECK: [[ret:%.*]] = scf.if [[cond:%.*]] -> (f64) {
                // [[two:%.*]] = arith.constant [[two:2.*]] : f64
                // [[arg02:%.*]] = arith.mulf [[arg0:%.*]], [[two:%.*]] : f64
                // scf.yield [[arg02:%.*]] : f64
                //} else {
                // [[%two_1:%.*]] = arith.constant [[two_1:2.*]] : f64
                // [[arg12:%.*]] = arith.mulf [[arg1:%.*]], [[two_1:%.*]] : f64
                // scf.yield [[arg12:%.*]] : f64
                // }
                // CHECK: [[phi_sum:%.*]] = arith.addf [[ret:%.*]], [[arg0:%.*]] : f64
                // CHECK: quantum.gphase([[phi_sum:%.*]])
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                quantum.gphase %arg0
                %ret = scf.if %cond -> (f64) {
                    %two = arith.constant 2 : f64
                    %arg0x2 = arith.mulf %arg0, %two : f64
                    scf.yield %arg0x2 : f64
                } else {
                    %two = arith.constant 2 : f64
                    %arg1x2 = arith.mulf %arg1, %two : f64
                    scf.yield %arg1x2 : f64
                }
                quantum.gphase %ret
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
                // [[two:%.*]] = arith.constant [[two:%2.*]] : f64
                // CHECK: [[t0:%.*]] = "test.op"() : () -> f64
                // [[arg0x2:%.*]] = arith.mulf [[arg0:%.*]], [[two:%.*]] : f64
                // [[phi_sum:%.+]] = arith.addf [[t0:%.*]], [[t0:%.*]] : f64
                // quantum.gphase [[phi_sum]]
                // scf.yield [[arg0x2:%.*]] : f64
                //} else {
                // [[%two1:%.*]] = arith.constant [[two1:%2.*]] : f64
                // [[arg1x2:%.*]] = arith.mulf [[arg1:%.*]], [[two1:%.*]] : f64
                // quantum.gphase([[arg1x2:%.*]])
                // scf.yield [[arg1x2:%.*]] : f64
                // }
                // CHECK: [[phi_sum:%.*]] = arith.addf [[arg0:%.*]], [[arg1:%.*]] : f64
                // CHECK: quantum.gphase([[phi_sum:%.*]])
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                %ret = scf.if %cond -> (f64) {
                    %two0 = arith.constant 2 : f64
                    %arg0x2 = arith.mulf %arg0, %two0 : f64
                    %t0 = "test.op"() : () -> f64
                    quantum.gphase %t0
                    quantum.gphase %t0
                    scf.yield %arg0x2 : f64
                } else {
                    %two1 = arith.constant 2 : f64
                    %arg1x2 = arith.mulf %arg1, %two1 : f64
                    quantum.gphase %arg1x2
                    scf.yield %arg1x2 : f64
                }
                quantum.gphase %arg0
                quantum.gphase %arg1
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
