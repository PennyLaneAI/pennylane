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
"""Unit test module for the ParitySynth transform"""
import pytest

pytestmark = pytest.mark.external

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
# from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

# import pennylane as qml
from pennylane.compiler.python_compiler.transforms import ParitySynthPass  # , parity_synth_pass


class TestParitySynthPass:
    """Unit tests for ParitySynthPass."""

    def test_no_phase_polynomial_ops(self, run_filecheck):
        """Test that nothing changes when there are no phase polynomial gates."""
        program = """
            func.func @test_func(%arg0: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = quantum.custom "Hadamard"() [[q0]] : !quantum.bit
                // CHECK: quantum.custom "RX"(%arg0) [[q1]] : !quantum.bit
                %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
                %2 = quantum.custom "RX"(%arg0) %1 : !quantum.bit
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)

    def test_composable_cnots(self, run_filecheck):
        """Test that two out of three CNOT gates are merged."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK: quantum.custom "CNOT"() [[q0]], [[q1]] : !quantum.bit, !quantum.bit
                %2, %3 = quantum.custom "CNOT"() %0, %1 : !quantum.bit, !quantum.bit
                %4, %5 = quantum.custom "CNOT"() %2, %3 : !quantum.bit, !quantum.bit
                %6, %7 = quantum.custom "CNOT"() %4, %5 : !quantum.bit, !quantum.bit
                // CHECK-NOT: "quantum.custom"
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)

    def test_two_cnots_single_rotation_no_merge(self, run_filecheck):
        """Test that a phase polynomial of two CNOTs separated by a rotation on the target
        is maintained."""
        program = """
            func.func @test_func(%arg0: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // In the following check, q1 and q0 are exchanged. This is a symmetry
                // of the test case and ParitySynth chooses to flip the CNOTs.
                // CHECK: [[q2:%.+]], [[q3:%.+]] = quantum.custom "CNOT"() [[q1]], [[q0]] : !quantum.bit, !quantum.bit
                %2, %3 = quantum.custom "CNOT"() %0, %1 : !quantum.bit, !quantum.bit
                // CHECK: [[q4:%.+]] = quantum.custom "RZ"(%arg0) [[q3]] : !quantum.bit
                %4 = quantum.custom "RZ"(%arg0) %3 : !quantum.bit
                // CHECK: quantum.custom "CNOT"() [[q2]], [[q4]] : !quantum.bit, !quantum.bit
                %5, %6 = quantum.custom "CNOT"() %2, %4 : !quantum.bit, !quantum.bit
                // CHECK-NOT: "quantum.custom"
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)

    def test_two_cnots_single_rotation_with_merge(self, run_filecheck):
        """Test that a phase polynomial of two CNOTs separated by a rotation on the control
        is reduced."""
        program = """
            func.func @test_func(%arg0: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2, %3 = quantum.custom "CNOT"() %0, %1 : !quantum.bit, !quantum.bit
                // CHECK: quantum.custom "RZ"(%arg0) [[q0]] : !quantum.bit
                %4 = quantum.custom "RZ"(%arg0) %2 : !quantum.bit
                %5, %6 = quantum.custom "CNOT"() %4, %3 : !quantum.bit, !quantum.bit
                // CHECK-NOT: "quantum.custom"
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)


# pylint: disable=pointless-string-statement
'''

    def test_composable_ops_different_qubits(self, run_filecheck):
        """Test that composable gates on different qubits are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK: quantum.custom "RX"(%arg0) [[q0]] : !quantum.bit
                // CHECK: quantum.custom "RX"(%arg1) [[q1]] : !quantum.bit
                %2 = quantum.custom "RX"(%arg0) %0 : !quantum.bit
                %3 = quantum.custom "RX"(%arg1) %1 : !quantum.bit
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)

    def test_controlled_composable_ops(self, run_filecheck):
        """Test that controlled composable ops can be merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst = "arith.constant"() <{value = true}> : () -> i1
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK: [[phi0:%.+]] = arith.addf %arg0, %arg1 : f64
                // CHECK: quantum.custom "RX"([[phi0]]) [[q0]] ctrls([[q1]], [[q2]]) ctrlvals(%cst, %cst) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                // CHECK-NOT: "quantum.custom"
                %3, %4, %5 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%cst, %cst) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %6, %7, %8 = quantum.custom "RX"(%arg1) %3 ctrls(%4, %5) ctrlvals(%cst, %cst) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)

    def test_controlled_composable_ops_same_control_values(self, run_filecheck):
        """Test that controlled composable ops with the same control values
        can be merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst0 = "arith.constant"() <{value = true}> : () -> i1
                %cst1 = "arith.constant"() <{value = false}> : () -> i1
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK: [[phi0:%.+]] = arith.addf %arg0, %arg1 : f64
                // CHECK: quantum.custom "RX"([[phi0]]) [[q0]] ctrls([[q1]], [[q2]]) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                // CHECK-NOT: quantum.custom
                %3, %4, %5 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %6, %7, %8 = quantum.custom "RX"(%arg1) %3 ctrls(%4, %5) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)

    def test_controlled_composable_ops_different_control_values(self, run_filecheck):
        """Test that controlled composable ops with different control values
        are not merged."""
        program = """
            func.func @test_func(%arg0: f64, %arg1: f64) {
                %cst0 = "arith.constant"() <{value = true}> : () -> i1
                %cst1 = "arith.constant"() <{value = false}> : () -> i1
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q3:%.+]], [[q4:%.+]], [[q5:%.+]] = quantum.custom "RX"(%arg0) [[q0]] ctrls([[q1]], [[q2]]) ctrlvals(%cst1, %cst0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                // CHECK: quantum.custom "RX"(%arg1) [[q3]] ctrls([[q4]], [[q5]]) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %3, %4, %5 = quantum.custom "RX"(%arg0) %0 ctrls(%1, %2) ctrlvals(%cst1, %cst0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %6, %7, %8 = quantum.custom "RX"(%arg1) %3 ctrls(%4, %5) ctrlvals(%cst0, %cst1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (ParitySynthPass(),)
        run_filecheck(program, pipeline)


# pylint: disable=too-few-public-methods
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestParitySynthIntegration:
    """Integration tests for the ParitySynthPass."""

    def test_qjit(self, run_filecheck_qjit):
        """Test that the ParitySynthPass works correctly with qjit."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir", pass_plugins=[getXDSLPluginAbsolutePath()])
        @parity_synth_pass
        @qml.qnode(dev)
        def circuit(x: float, y: float):
            # CHECK: [[phi:%.+]] = arith.addf
            # CHECK: quantum.custom "RX"([[phi]])
            # CHECK-NOT: quantum.custom
            qml.RX(x, 0)
            qml.RX(y, 0)
            return qml.state()

        run_filecheck_qjit(circuit)

'''

if __name__ == "__main__":
    pytest.main(["-x", __file__])
