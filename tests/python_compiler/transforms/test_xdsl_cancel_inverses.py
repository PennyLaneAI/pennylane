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

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")

# pylint: disable=wrong-import-position
from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    IterativeCancelInversesPass,
    iterative_cancel_inverses_pass,
)


class TestIterativeCancelInversesPass:
    """Unit tests for IterativeCancelInversesPass."""

    def test_no_inverses_same_qubit(self, run_filecheck):
        """Test that nothing changes when there are no inverses."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = quantum.custom "PauliX"() [[q0]] : !quantum.bit
                // CHECK: quantum.custom "PauliY"() [[q1]] : !quantum.bit
                %1 = quantum.custom "PauliX"() %0 : !quantum.bit
                %2 = quantum.custom "PauliY"() %1 : !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)

    def test_inverses_different_qubits(self, run_filecheck):
        """Test that nothing changes when there are no inverses."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                %1 = "test.op"() : () -> !quantum.bit
                // CHECK: quantum.custom "PauliX"() [[q0]] : !quantum.bit
                // CHECK: quantum.custom "PauliX"() [[q1]] : !quantum.bit
                %2 = quantum.custom "PauliX"() %0 : !quantum.bit
                %3 = quantum.custom "PauliX"() %1 : !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)

    def test_simple_self_inverses(self, run_filecheck):
        """Test that inverses are cancelled."""
        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: quantum.custom
                %1 = quantum.custom "PauliX"() %0 : !quantum.bit
                %2 = quantum.custom "PauliX"() %1 : !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)

    def test_nested_self_inverses(self, run_filecheck):
        """Test that nested self-inverses are cancelled."""
        program = """
            func.func @test_func() {
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: quantum.custom
                %1 = quantum.custom "PauliX"() %0 : !quantum.bit
                %2 = quantum.custom "PauliY"() %1 : !quantum.bit
                %3 = quantum.custom "PauliZ"() %2 : !quantum.bit
                %4 = quantum.custom "PauliZ"() %3 : !quantum.bit
                %5 = quantum.custom "PauliY"() %4 : !quantum.bit
                %6 = quantum.custom "PauliX"() %5 : !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)

    def test_cancel_ops_with_control_qubits(self, run_filecheck):
        """Test that ops with control qubits can be cancelled."""
        program = """
            func.func @test_func() {
                %0 = "arith.constant"() <{value = true}> : () -> i1
                %1 = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit
                // CHECK-NOT: quantum.custom
                %4, %5, %6 = quantum.custom "PauliY"() %1 ctrls(%2, %3) ctrlvals(%0, %0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %7, %8, %9 = quantum.custom "PauliY"() %4 ctrls(%5, %6) ctrlvals(%0, %0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)

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
                %5, %6, %7 = quantum.custom "PauliY"() %2 ctrls(%3, %4) ctrlvals(%1, %0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %8, %9, %10 = quantum.custom "PauliY"() %5 ctrls(%6, %7) ctrlvals(%1, %0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)

    def test_ops_with_control_qubits_different_control_values(self, run_filecheck):
        """Test that ops with the same control qubits but different
        control values don't cancel."""
        program = """
            func.func @test_func() {
                // CHECK-DAG: [[cval0:%.+]] = arith.constant false
                // CHECK-DAG: [[cval1:%.+]] = arith.constant true
                %0 = arith.constant false
                %1 = arith.constant true
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = "test.op"() : () -> !quantum.bit
                // CHECK: [[q2:%.+]] = "test.op"() : () -> !quantum.bit
                %2 = "test.op"() : () -> !quantum.bit
                %3 = "test.op"() : () -> !quantum.bit
                %4 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q3:%.+]], [[q4:%.+]], [[q5:%.+]] = quantum.custom "PauliY"() [[q0]] ctrls([[q1]], [[q2]]) ctrlvals([[cval1]], [[cval0]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                // CHECK: quantum.custom "PauliY"() [[q3]] ctrls([[q4]], [[q5]]) ctrlvals([[cval0]], [[cval1]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %5, %6, %7 = quantum.custom "PauliY"() %2 ctrls(%3, %4) ctrlvals(%1, %0) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                %8, %9, %10 = quantum.custom "PauliY"() %5 ctrls(%6, %7) ctrlvals(%0, %1) : !quantum.bit ctrls !quantum.bit, !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)

    def test_non_consecutive_self_inverse_ops(self, run_filecheck):
        """Test that self-inverse gates on the same qubit that are not
        consecutive are not cancelled."""
        program = """
            func.func @test_func() {
                // CHECK: [[q0:%.+]] = "test.op"() : () -> !quantum.bit
                %0 = "test.op"() : () -> !quantum.bit
                // CHECK: [[q1:%.+]] = quantum.custom "PauliX"() [[q0]] : !quantum.bit
                // CHECK: [[q2:%.+]] = quantum.custom "PauliY"() [[q1]] : !quantum.bit
                // CHECK: quantum.custom "PauliX"() [[q2]] : !quantum.bit
                %1 = quantum.custom "PauliX"() %0 : !quantum.bit
                %2 = quantum.custom "PauliY"() %1 : !quantum.bit
                %3 = quantum.custom "PauliX"() %2 : !quantum.bit
                return
            }
        """

        pipeline = (IterativeCancelInversesPass(),)
        run_filecheck(program, pipeline)


# pylint: disable=too-few-public-methods
@pytest.mark.usefixtures("enable_disable_plxpr")
class TestIterativeCancelInversesIntegration:
    """Integration tests for the IterativeCancelInversesPass."""

    def test_qjit(self, run_filecheck_qjit):
        """Test that the IterativeCancelInversesPass works correctly with qjit."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir", pass_plugins=[getXDSLPluginAbsolutePath()])
        @iterative_cancel_inverses_pass
        @qml.qnode(dev)
        def circuit():
            # CHECK-NOT: quantum.custom
            qml.H(0)
            qml.X(0)
            qml.X(0)
            qml.H(0)
            return qml.state()

        run_filecheck_qjit(circuit)

    def test_qjit_no_cancellation(self, run_filecheck_qjit):
        """Test that the IterativeCancelInversesPass works correctly with qjit when
        there are no operations that can be cancelled."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit(target="mlir", pass_plugins=[getXDSLPluginAbsolutePath()])
        @iterative_cancel_inverses_pass
        @qml.qnode(dev)
        def circuit():
            # CHECK-NOT: quantum.custom
            qml.H(1)
            qml.X(1)
            qml.X(0)
            qml.H(0)
            return qml.state()

        with pytest.raises(AssertionError, match="filecheck failed"):
            run_filecheck_qjit(circuit)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
