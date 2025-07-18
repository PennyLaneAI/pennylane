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

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import MergeRotationsPass, merge_rotations_pass


class TestMergeRotationsPass:
    """Unit tests for MergeRotationsPass."""

    def test_no_composable_ops(self, run_filecheck):
        """Test that nothing changes when there are no composable gates."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.*]] = quantum.custom "RX"() [[q0:%.*]] : !quantum.bit
                // CHECK: quantum.custom "RY"() [[q1:%.*]] : !quantum.bit
                %1 = quantum.custom "RX"() %0 : !quantum.bit
                %2 = quantum.custom "RY"() %1 : !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)

    def test_composable_ops(self, run_filecheck):
        """Test that composable gates are merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[phi0:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK-DAG: quantum.custom "RX"([[phi0:%.*]]) [[q0:%.*]] : !quantum.bit
                // CHECK-NOT: "quantum.custom"
                %1 = quantum.custom "RX"(%arg0) %0 : !quantum.bit
                %2 = quantum.custom "RX"(%arg1) %1 : !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)

    def test_many_composable_ops(self, run_filecheck):
        """Test that more than 2 composable ops are merged correctly."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) {
                // CHECK: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[phi0:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK-DAG: [[phi1:%.*]] = arith.addf [[phi0:%.*]], %arg2 : f64
                // CHECK-DAG: [[phi2:%.*]] = arith.addf [[phi1:%.*]], %arg3 : f64
                // CHECK-DAG: quantum.custom "RX"([[phi2:%.*]]) [[q0:%.*]] : !quantum.bit
                // CHECK-NOT: "quantum.custom"
                %1 = quantum.custom "RX"(%arg0) %0 : !quantum.bit
                %2 = quantum.custom "RX"(%arg1) %1 : !quantum.bit
                %3 = quantum.custom "RX"(%arg2) %2 : !quantum.bit
                %4 = quantum.custom "RX"(%arg3) %3 : !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)

    def test_non_consecutive_composable_ops(self, run_filecheck):
        """Test that non-consecutive composable gates are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK-DAG: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q1:%.*]] = quantum.custom "RX"(%arg0) [[q0:%.*]] : !quantum.bit
                // CHECK-DAG: [[q2:%.*]] = quantum.custom "RY"(%arg0) [[q1:%.*]] : !quantum.bit
                // CHECK-DAG: quantum.custom "RX"(%arg1) [[q2:%.*]] : !quantum.bit
                %1 = quantum.custom "RX"(%arg0) %0 : !quantum.bit
                %2 = quantum.custom "RY"(%arg0) %1 : !quantum.bit
                %3 = quantum.custom "RX"(%arg1) %2 : !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)

    def test_composable_ops_different_qubits(self, run_filecheck):
        """Test that composable gates on different qubits are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK-DAG: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: quantum.custom "RX"(%arg0) [[q0:%.*]] : !quantum.bit
                // CHECK-DAG: quantum.custom "RX"(%arg1) [[q1:%.*]] : !quantum.bit
                %2 = quantum.custom "RX"(%arg0) %0 : !quantum.bit
                %3 = quantum.custom "RX"(%arg1) %1 : !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)

    def test_controlled_composable_ops(self, run_filecheck):
        """Test that controlled composable ops can be merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst = "arith.constant"() <{value = true}> : () -> i1
                // CHECK-DAG: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[phi0:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK-DAG: quantum.custom "RX"([[phi0:%.*]]) [[q0:%.*]] ctrls([[q1:%.*]], [[q2:%.*]]) ctrlvals(%cst, %cst) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                // CHECK-NOT: "quantum.custom"
                %3, %4, %5 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%cst, %cst) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %6, %7, %8 = quantum.custom "RX"(%arg1) %3 ctrls(%4, %5) ctrlvals(%cst, %cst) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)

    def test_controlled_composable_ops_same_control_values(self, run_filecheck):
        """Test that controlled composable ops with the same control values
        can be merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst0 = "arith.constant"() <{value = true}> : () -> i1
                %cst1 = "arith.constant"() <{value = false}> : () -> i1
                // CHECK-DAG: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[phi0:%.*]] = arith.addf %arg0, %arg1 : f64
                // CHECK-DAG: quantum.custom "RX"([[phi0:%.*]]) [[q0:%.*]] ctrls([[q1:%.*]], [[q2:%.*]]) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                // CHECK-NOT: quantum.custom
                %3, %4, %5 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %6, %7, %8 = quantum.custom "RX"(%arg1) %3 ctrls(%4, %5) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)

    def test_controlled_composable_ops_different_control_values(self, run_filecheck):
        """Test that controlled composable ops with different control values
        are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst0 = "arith.constant"() <{value = true}> : () -> i1
                %cst1 = "arith.constant"() <{value = false}> : () -> i1
                // CHECK-DAG: [[q0:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q1:%.*]] = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q2:%.*]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK-DAG: [[q3:%.*]], [[q4:%.*]], [[q5:%.*]] = quantum.custom "RX"(%arg0) [[q0:%.*]] ctrls([[q1:%.*]], [[q2:%.*]]) ctrlvals(%cst1, %cst0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                // CHECK-DAG: quantum.custom "RX"(%arg1) [[q5:%.*]] ctrls([[q4:%.*]], [[q5:%.*]]) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %3, %4, %5 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%cst1, %cst0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %6, %7, %8 = quantum.custom "RX"(%arg1) %3 ctrls(%4, %5) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (MergeRotationsPass(),)
        run_filecheck(program, pipeline)


# pylint: disable=too-few-public-methods
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestMergeRotationsIntegration:
    """Integration tests for the MergeRotationsPass."""

    def test_qjit(self, run_filecheck_qjit):
        """Test that the MergeRotationsPass works correctly with qjit."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir", pass_plugins=[getXDSLPluginAbsolutePath()])
        @merge_rotations_pass
        @qml.qnode(dev)
        def circuit(x: float, y: float):
            # CHECK: [[phi:%.*]] = arith.addf
            # CHECK: quantum.custom "RX"([[phi]])
            # CHECK-NOT: quantum.custom
            qml.RX(x, 0)
            qml.RX(y, 0)
            return qml.state()

        run_filecheck_qjit(circuit)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
