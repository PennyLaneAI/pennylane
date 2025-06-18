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
from pennylane.compiler.python_compiler.transforms import CombineGlobalPhasesPass


class TestCombineGlobalPhasesPass:
    """Unit tests for CombineGlobalPhasesPass."""

    # def test_no_global_phases_ops(self, run_filecheck):
    #     """Test that nothing changes when there are no composable gates."""
    #     program = """
    #         func.func @test_func(%arg0: f64, %arg1: f64) {
    #             // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
    #             %0 = "test.op"() : () -> !quantum.bit
    #             // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
    #             // CHECK: quantum.custom "RY"() [[q1:%.*]] : !quantum.bit
    #             %1 = quantum.custom "RX"() %0 : !quantum.bit
    #             %2 = quantum.custom "RY"() %1 : !quantum.bit
    #             return
    #         }
    #     """

    #     ctx = Context()
    #     ctx.load_dialect(func.Func)
    #     ctx.load_dialect(test.Test)
    #     ctx.load_dialect(Quantum)

    #     module = xdsl.parser.Parser(ctx, program).parse_module()
    #     pipeline = xdsl.passes.PipelinePass((CombineGlobalPhasesPass(),))
    #     pipeline.apply(ctx, module)

    #     run_filecheck(program, module)

    def test_combinable_ops(self, run_filecheck):
        """Test that composable gates are merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: %1 = arith.addf %arg1, %arg0 : f64
                // CHECK: quantum.gphase(%1)
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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
