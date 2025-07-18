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

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    CombineGlobalPhasesPass,
    combine_global_phases_pass,
)


class TestCombineGlobalPhasesPass:
    """Unit tests for CombineGlobalPhasesPass."""

    def test_combinable_ops_without_control_flow(self, run_filecheck):
        """Test that combines global phases in a func without control flow."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[phi_sum:%.+]] = arith.addf %arg1, %arg0 : f64
                // CHECK: quantum.gphase([[phi_sum]])
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: [[q1:%.+]] = quantum.custom "PauliX"() [[q0]] : !quantum.bit
                %2 = quantum.custom "PauliX"() %0 : !quantum.bit
                return
            }
        """

        pipeline = (CombineGlobalPhasesPass(),)
        run_filecheck(program, pipeline)

    def test_combinable_ops_with_control_flow(self, run_filecheck):
        """Test that combines global phases in a func with control flow."""
        program = """
            func.func @test_func(%cond: i32, %arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                quantum.gphase %arg0
                // CHECK: [[ret:%.+]] = scf.if %cond -> (f64) {
                %ret = scf.if %cond -> (f64) {
                    // CHECK: [[two:%.+]] = arith.constant {{2.+}} : f64
                    %two = arith.constant 2 : f64
                    // CHECK: [[arg02:%.+]] = arith.mulf %arg0, [[two]] : f64
                    %arg0x2 = arith.mulf %arg0, %two : f64
                    // CHECK: scf.yield [[arg02]] : f64
                    scf.yield %arg0x2 : f64
                } else {
                    // CHECK: [[two_1:%.+]] = arith.constant {{2.+}} : f64
                    %two_1 = arith.constant 2 : f64
                    // CHECK: [[arg12:%.+]] = arith.mulf %arg1, [[two_1]] : f64
                    %arg1x2 = arith.mulf %arg1, %two_1 : f64
                    // CHECK: scf.yield [[arg12:%.+]] : f64
                    scf.yield %arg1x2 : f64
                }
                // CHECK: [[phi_sum:%.+]] = arith.addf [[ret]], %arg0 : f64
                // CHECK: quantum.gphase([[phi_sum]]) :
                quantum.gphase %ret
                // CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
                %2 = quantum.custom "PauliX"() %0 : !quantum.bit
                return
            }
        """

        pipeline = (CombineGlobalPhasesPass(),)
        run_filecheck(program, pipeline)

    def test_combinable_ops_in_control_flow_if(self, run_filecheck):
        """Test that combines global phases in a func without control a flow.
        Here the control flow is an `if` operation.
        """
        program = """
            // CHECK: func.func @test_func(%cond : i32, [[arg0:%.+]] : f64, [[arg1:%.+]] : f64)
            func.func @test_func(%cond : i32, %arg0 : f64, %arg1 : f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[ret:%.+]] = scf.if [[cond:%.+]] -> (f64) {
                %ret = scf.if %cond -> (f64) {
                    // CHECK: [[two0:%.+]] = arith.constant {{2.+}} : f64
                    %two0 = arith.constant 2 : f64
                    // CHECK: [[arg02:%.+]] = arith.mulf [[arg0]], [[two0]] : f64
                    %arg02 = arith.mulf %arg0, %two0 : f64
                    // CHECK: [[t0:%.+]] = "test.op"() : () -> f64
                    %t0 = "test.op"() : () -> f64
                    // CHECK: [[phi_sum_0:%.+]] = arith.addf [[t0]], [[t0]] : f64
                    // CHECK: quantum.gphase([[phi_sum_0]])
                    quantum.gphase %t0
                    quantum.gphase %t0
                    // CHECK: scf.yield [[arg02]] : f64
                    scf.yield %arg02 : f64
                    // CHECK: } else {
                } else {
                    // CHECK: [[two1:%.+]] = arith.constant {{2.+}} : f64
                    %two1 = arith.constant 2 : f64
                    // CHECK: [[arg1x2:%.+]] = arith.mulf [[arg1]], [[two1]] : f64
                    %arg1x2 = arith.mulf %arg1, %two1 : f64
                    // CHECK: [[phi_sum_1:%.+]] = arith.addf [[arg1x2]], [[arg1x2]] : f64
                    // CHECK: quantum.gphase([[phi_sum_1]])
                    quantum.gphase %arg1x2
                    quantum.gphase %arg1x2
                    // CHECK: scf.yield [[arg1x2]] : f64
                    scf.yield %arg1x2 : f64
                    // CHECK: }
                }
                // CHECK: [[phi_sum_2:%.+]] = arith.addf [[arg1]], [[arg0]] : f64
                // CHECK: quantum.gphase([[phi_sum_2:%.+]])
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
                %2 = quantum.custom "PauliX"() %0 : !quantum.bit
                return
            }
        """

        pipeline = (CombineGlobalPhasesPass(),)
        run_filecheck(program, pipeline)

    def test_combinable_ops_in_control_flow_for(self, run_filecheck):
        """Test that combines global phases in a func with a control flow.
        Here the control flow is a `for` operation.
        """
        program = """
            // CHECK: func.func @test_func(%n : i32, [[arg0:%.+]] : f64, [[arg1:%.+]] : f64)
            func.func @test_func(%n : i32, %arg0 : f64, %arg1 : f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[c0:%.+]] = arith.constant 0 : i32
                // CHECK: [[c1:%.+]] = arith.constant 1 : i32
                %0 = "test.op"() : () -> !quantum.bit
                %c0 = arith.constant 0 : i32
                %c1 = arith.constant 1 : i32
                scf.for %i = %c0 to %n step %c1 {
                    // CHECK: [[two0:%.+]] = arith.constant {{2.+}} : f64
                    %two0 = arith.constant 2 : f64
                    // CHECK: [[arg02:%.+]] = arith.mulf [[arg0]], [[two0]] : f64
                    %arg02 = arith.mulf %arg0, %two0 : f64
                    // CHECK: [[t0:%.+]] = "test.op"() : () -> f64
                    %t0 = "test.op"() : () -> f64
                    // CHECK: [[phi_sum_0:%.+]] = arith.addf [[t0]], [[t0]] : f64
                    // CHECK: quantum.gphase([[phi_sum_0]])
                    quantum.gphase %t0
                    quantum.gphase %t0
                }
                // CHECK: [[phi_sum_1:%.+]] = arith.addf [[arg1]], [[arg0]] : f64
                // CHECK: quantum.gphase([[phi_sum_1]])
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
                %2 = quantum.custom "PauliX"() %0 : !quantum.bit
                return
            }
        """

        pipeline = (CombineGlobalPhasesPass(),)
        run_filecheck(program, pipeline)

    def test_combinable_ops_in_control_flow_while(self, run_filecheck):
        """Test that combines global phases in a func with control flow.
        Here the control flow is a `while` operation.
        """
        program = """
            // CHECK: func.func @test_func(%n : i32, [[arg0:%.+]] : f64, [[arg1:%.+]] : f64)
            func.func @test_func(%n : i32, %arg0 : f64, %arg1 : f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %c0 = arith.constant 0 : i32
                %c1 = arith.constant 1 : i32
                scf.while (%current_i = %n) : (i32) -> () {
                    %cond = arith.cmpi slt, %current_i, %n : i32
                    scf.condition(%cond) %current_i : i32
                } do {
                ^bb0(%i : i32):
                    %next_i = arith.addi %i, %c1 : i32
                    // CHECK: [[two0:%.+]] = arith.constant {{2.+}} : f64
                    %two0 = arith.constant 2 : f64
                    // CHECK: [[arg02:%.+]] = arith.mulf [[arg0]], [[two0]] : f64
                    %arg02 = arith.mulf %arg0, %two0 : f64
                    // CHECK: [[t0:%.+]] = "test.op"() : () -> f64
                    %t0 = "test.op"() : () -> f64
                    // CHECK: [[phi_sum_0:%.+]] = arith.addf [[t0]], [[t0]] : f64
                    // CHECK: quantum.gphase([[phi_sum_0]])
                    quantum.gphase %t0
                    quantum.gphase %t0
                    scf.yield %next_i : i32
                }
                // CHECK: [[phi_sum_1:%.+]] = arith.addf [[arg1]], [[arg0]] : f64
                // CHECK: quantum.gphase([[phi_sum_1]]) :
                quantum.gphase %arg0
                quantum.gphase %arg1
                // CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
                %2 = quantum.custom "PauliX"() %0 : !quantum.bit
                return
            }
        """

        pipeline = (CombineGlobalPhasesPass(),)
        run_filecheck(program, pipeline)


# pylint: disable=too-few-public-methods
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestCombineGlobalPhasesIntegration:
    """Integration tests for the CombineGlobalPhasesPass."""

    def test_qjit(self, run_filecheck_qjit):
        """Test that the CombineGlobalPhasesPass works correctly with qjit."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir", pass_plugins=[getXDSLPluginAbsolutePath()])
        @combine_global_phases_pass
        @qml.qnode(dev)
        def circuit(x: float, y: float):
            # CHECK: [[phi:%.+]] = arith.addf
            # CHECK: quantum.gphase([[phi]])
            # CHECK-NOT: quantum.gphase
            qml.GlobalPhase(x)
            qml.GlobalPhase(y)
            return qml.state()

        run_filecheck_qjit(circuit)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
