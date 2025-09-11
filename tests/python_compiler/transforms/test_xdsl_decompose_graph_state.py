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

"""Unit and integration tests for the Python compiler `decompose-graph-state` transform.

FileCheck notation hint:

    Qubit variable names are written as q0a, q0b, q1a, etc. in the FileCheck program. The leading
    number indicates the wire index of that qubit, and the second letter increments by one after
    each use.
"""

# pylint: disable=wrong-import-position

import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")

from pennylane.compiler.python_compiler.transforms import (
    DecomposeGraphStatePass,
    NullDecomposeGraphStatePass,
)


class TestDecomposeGraphStatePass:
    """Unit tests for the decompose-graph-state pass."""

    def test_1_qubit(self, run_filecheck):
        """Test the decompose-graph-state pass for a 1-qubit graph state."""
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[]> : tensor<0xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(1) : !quantum.reg
            // CHECK: [[q0a:%.+]] = quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK: [[q0b:%.+]] = quantum.custom "Hadamard"() [[q0a]] : !quantum.bit
            // CHECK: [[out_reg0:%.+]] = quantum.insert [[graph_reg]][0], [[q0b]] : !quantum.reg, !quantum.bit

            %adj_matrix = arith.constant dense<[]> : tensor<0xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<0xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_2_qubit_chain(self, run_filecheck):
        """Test the decompose-graph-state pass for a 2-qubit graph state. The qubit connectivity is:

            0 -- 1

        which has the adjacency matrix representation

            0  1
            1  0

        and densely packed adjacency matrix representation

            [1]
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1]> : tensor<1xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(2) : !quantum.reg

            // CHECK:      [[q0a:%.+]] = quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: [[q1a:%.+]] = quantum.extract [[graph_reg]][1] : !quantum.reg -> !quantum.bit

            // CHECK:      [[q0b:%.+]] = quantum.custom "Hadamard"() [[q0a]] : !quantum.bit
            // CHECK-NEXT: [[q1b:%.+]] = quantum.custom "Hadamard"() [[q1a]] : !quantum.bit

            // CHECK: [[q0c:%.+]], [[q1c:%.+]] = quantum.custom "CZ"() [[q0b]], [[q1b]] : !quantum.bit, !quantum.bit

            // CHECK:      [[out_reg00:%.+]] = quantum.insert [[graph_reg]][0], [[q0c]] : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_reg01:%.+]] = quantum.insert [[out_reg00]][1], [[q1c]] : !quantum.reg, !quantum.bit

            %adj_matrix = arith.constant dense<[1]> : tensor<1xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<1xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_3_qubit_chain(self, run_filecheck):
        """Test the decompose-graph-state pass for a 3-qubit graph state. The qubit connectivity is:

            0 -- 1 -- 2

        which has the adjacency matrix representation

            0  1  0
            1  0  1
            0  1  0

        and densely packed adjacency matrix representation

            [1, 0, 1]
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(3) : !quantum.reg

            // CHECK:      [[q0a:%.+]] = quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: [[q1a:%.+]] = quantum.extract [[graph_reg]][1] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: [[q2a:%.+]] = quantum.extract [[graph_reg]][2] : !quantum.reg -> !quantum.bit

            // CHECK:      [[q0b:%.+]] = quantum.custom "Hadamard"() [[q0a]] : !quantum.bit
            // CHECK-NEXT: [[q1b:%.+]] = quantum.custom "Hadamard"() [[q1a]] : !quantum.bit
            // CHECK-NEXT: [[q2b:%.+]] = quantum.custom "Hadamard"() [[q2a]] : !quantum.bit

            // CHECK:      [[q0c:%.+]], [[q1c:%.+]] = quantum.custom "CZ"() [[q0b]], [[q1b]] : !quantum.bit, !quantum.bit
            // CHECK-NEXT: [[q1d:%.+]], [[q2c:%.+]] = quantum.custom "CZ"() [[q1c]], [[q2b]] : !quantum.bit, !quantum.bit

            // CHECK:      [[out_reg00:%.+]] = quantum.insert [[graph_reg]][0], [[q0c]] : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_reg01:%.+]] = quantum.insert [[out_reg00]][1], [[q1d]] : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_reg02:%.+]] = quantum.insert [[out_reg01]][2], [[q2c]] : !quantum.reg, !quantum.bit

            %adj_matrix = arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_4_qubit_square_lattice(self, run_filecheck):
        """Test the decompose-graph-state pass for a 4-qubit graph state. The qubit connectivity is:

            0 -- 1
            |    |
            2 -- 3

        which has the adjacency matrix representation

            0  1  1  0
            1  0  0  1
            1  0  0  1
            0  1  1  0

        and densely packed adjacency matrix representation

            [1, 1, 0, 0, 1, 1]

        [(0, 1), (0, 2), (1, 3), (2, 3)]
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1, 1, 0, 0, 1, 1]> : tensor<6xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(4) : !quantum.reg

            // CHECK:      [[q0a:%.+]] = quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: [[q1a:%.+]] = quantum.extract [[graph_reg]][1] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: [[q2a:%.+]] = quantum.extract [[graph_reg]][2] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: [[q3a:%.+]] = quantum.extract [[graph_reg]][3] : !quantum.reg -> !quantum.bit

            // CHECK:      [[q0b:%.+]] = quantum.custom "Hadamard"() [[q0a]] : !quantum.bit
            // CHECK-NEXT: [[q1b:%.+]] = quantum.custom "Hadamard"() [[q1a]] : !quantum.bit
            // CHECK-NEXT: [[q2b:%.+]] = quantum.custom "Hadamard"() [[q2a]] : !quantum.bit
            // CHECK-NEXT: [[q3b:%.+]] = quantum.custom "Hadamard"() [[q3a]] : !quantum.bit

            // CHECK:      [[q0c:%.+]], [[q1c:%.+]] = quantum.custom "CZ"() [[q0b]], [[q1b]] : !quantum.bit, !quantum.bit
            // CHECK-NEXT: [[q0d:%.+]], [[q2c:%.+]] = quantum.custom "CZ"() [[q0c]], [[q2b]] : !quantum.bit, !quantum.bit
            // CHECK-NEXT: [[q1d:%.+]], [[q3c:%.+]] = quantum.custom "CZ"() [[q1c]], [[q3b]] : !quantum.bit, !quantum.bit
            // CHECK-NEXT: [[q2d:%.+]], [[q3d:%.+]] = quantum.custom "CZ"() [[q2c]], [[q3c]] : !quantum.bit, !quantum.bit

            // CHECK:      [[out_reg00:%.+]] = quantum.insert [[graph_reg]][0], [[q0d]] : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_reg01:%.+]] = quantum.insert [[out_reg00]][1], [[q1d]] : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_reg02:%.+]] = quantum.insert [[out_reg01]][2], [[q2d]] : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_reg03:%.+]] = quantum.insert [[out_reg02]][3], [[q3d]] : !quantum.reg, !quantum.bit

            %adj_matrix = arith.constant dense<[1, 1, 0, 0, 1, 1]> : tensor<6xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_2_qubit_chain_non_standard_init(self, run_filecheck):
        """Test the decompose-graph-state pass for a 2-qubit graph state, using non-standard `init`
        and `entangle` attributes.
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1]> : tensor<1xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(2) : !quantum.reg

            // CHECK:      [[q0a:%.+]] = quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: [[q1a:%.+]] = quantum.extract [[graph_reg]][1] : !quantum.reg -> !quantum.bit

            // CHECK:      [[q0b:%.+]] = quantum.custom "S"() [[q0a]] : !quantum.bit
            // CHECK-NEXT: [[q1b:%.+]] = quantum.custom "S"() [[q1a]] : !quantum.bit

            // CHECK: [[q0c:%.+]], [[q1c:%.+]] = quantum.custom "CNOT"() [[q0b]], [[q1b]] : !quantum.bit, !quantum.bit

            // CHECK:      [[out_reg00:%.+]] = quantum.insert [[graph_reg]][0], [[q0c]] : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_reg01:%.+]] = quantum.insert [[out_reg00]][1], [[q1c]] : !quantum.reg, !quantum.bit

            %adj_matrix = arith.constant dense<[1]> : tensor<1xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<1xi1>) [init "S", entangle "CNOT"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_register_use(self, run_filecheck):
        """Test that the uses of the register resulting from a graph_state_prep op are still correct
        after decomposing it into its quantum ops with the decompose-graph-state pass.

        We do not rigorously test the decomposition here, only that the last resulting register from
        the decomposition (specifically, from the last quantum.insert op) is correctly picked up by
        the ops that used the register resulting from the original graph_state_prep op.
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1]> : tensor<1xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(2) : !quantum.reg

            // CHECK:      quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: quantum.extract [[graph_reg]][1] : !quantum.reg -> !quantum.bit

            // CHECK:      quantum.custom "Hadamard"() {{%.+}} : !quantum.bit
            // CHECK-NEXT: quantum.custom "Hadamard"() {{%.+}} : !quantum.bit

            // CHECK: quantum.custom "CZ"() {{%.+}}, {{%.+}} : !quantum.bit, !quantum.bit

            // CHECK:                         quantum.insert {{%.+}}[0], {{%.+}} : !quantum.reg, !quantum.bit
            // CHECK-NEXT: [[out_qreg0:%.+]] = quantum.insert {{%.+}}[1], {{%.+}} : !quantum.reg, !quantum.bit

            %adj_matrix = arith.constant dense<[1]> : tensor<1xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<1xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg

            // CHECK:      quantum.extract [[out_qreg0]][0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: quantum.extract [[out_qreg0]][1] : !quantum.reg -> !quantum.bit
            %q0a = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
            %q1a = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit

            // CHECK: arith.constant
            // CHECK: mbqc.measure_in_basis
            %angle_0 = arith.constant 0.0 : f64
            %m0, %q0b = mbqc.measure_in_basis [XY, %angle_0] %q0a : i1, !quantum.bit

            // CHECK: arith.constant
            // CHECK: mbqc.measure_in_basis
            %angle_pi_2 = arith.constant 1.5707963267948966 : f64
            %m1, %q1b = mbqc.measure_in_basis [XY, %angle_pi_2] %q1a : i1, !quantum.bit

            // CHECK: [[out_qreg1:%.+]] = quantum.insert [[out_qreg0]][0], {{%.+}} : !quantum.reg, !quantum.bit
            // CHECK: [[out_qreg2:%.+]] = quantum.insert [[out_qreg1]][1], {{%.+}} : !quantum.reg, !quantum.bit
            %reg0 = quantum.insert %qreg[0], %q0b : !quantum.reg, !quantum.bit
            %reg1 = quantum.insert %reg0[1], %q1b : !quantum.reg, !quantum.bit

            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_adj_matrix_reuse(self, run_filecheck):
        """Test that the decompose-graph-state pass supports the case where we reuse the adjacency
        matrix resulting from a constant op multiple times.
        """

        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: quantum.alloc(3)
            // CHECK: quantum.alloc(3)

            %adj_matrix = arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            %qreg1 = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            %qreg2 = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_with_stablehlo_constant(self, run_filecheck):
        """Test that the decompose-graph-state pass supports the case where the adjacency matrix
        results from a `stablehlo.constant` op (rather than an `arith.constant` op).
        """

        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: stablehlo.constant
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: quantum.alloc(3)

            %adj_matrix = "stablehlo.constant"() <{value = dense<[1, 0, 1]> : tensor<3xi1>}> : () -> tensor<3xi1>
            %qreg1 = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)


class TestNullDecomposeGraphStatePass:
    """Unit tests for the null-decompose-graph-state pass."""

    def test_1_qubit(self, run_filecheck):
        """Test the null-decompose-graph-state pass for a 1-qubit graph state."""
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[]> : tensor<0xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(1) : !quantum.reg
            // CHECK-NOT: quantum.extract
            // CHECK-NOT: quantum.custom "Hadamard"
            // CHECK-NOT: quantum.custom "CZ"
            // CHECK-NOT: quantum.insert

            %adj_matrix = arith.constant dense<[]> : tensor<0xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<0xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (NullDecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_2_qubit_chain(self, run_filecheck):
        """Test the null-decompose-graph-state pass for a 2-qubit graph state."""
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1]> : tensor<1xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(2) : !quantum.reg
            // CHECK-NOT: quantum.extract
            // CHECK-NOT: quantum.custom "Hadamard"
            // CHECK-NOT: quantum.custom "CZ"
            // CHECK-NOT: quantum.insert

            %adj_matrix = arith.constant dense<[1]> : tensor<1xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<1xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (NullDecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_3_qubit_chain(self, run_filecheck):
        """Test the null-decompose-graph-state pass for a 3-qubit graph state."""
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(3) : !quantum.reg
            // CHECK-NOT: quantum.extract
            // CHECK-NOT: quantum.custom "Hadamard"
            // CHECK-NOT: quantum.custom "CZ"
            // CHECK-NOT: quantum.insert

            %adj_matrix = arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (NullDecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_register_use(self, run_filecheck):
        """Test that the uses of the register resulting from a graph_state_prep op are still correct
        after decomposing it into its quantum ops with the null-decompose-graph-state pass.

        We do not rigorously test the decomposition here, only that the resulting register from the
        decomposition (specifically, from the quantum.alloc op) is correctly picked up by the ops
        that used the register resulting from the original graph_state_prep op.
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1]> : tensor<1xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(2) : !quantum.reg

            %adj_matrix = arith.constant dense<[1]> : tensor<1xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<1xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg

            // CHECK:      quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK-NEXT: quantum.extract [[graph_reg]][1] : !quantum.reg -> !quantum.bit
            %q0a = quantum.extract %qreg[0] : !quantum.reg -> !quantum.bit
            %q1a = quantum.extract %qreg[1] : !quantum.reg -> !quantum.bit

            // CHECK: arith.constant
            // CHECK: mbqc.measure_in_basis
            %angle_0 = arith.constant 0.0 : f64
            %m0, %q0b = mbqc.measure_in_basis [XY, %angle_0] %q0a : i1, !quantum.bit

            // CHECK: arith.constant
            // CHECK: mbqc.measure_in_basis
            %angle_pi_2 = arith.constant 1.5707963267948966 : f64
            %m1, %q1b = mbqc.measure_in_basis [XY, %angle_pi_2] %q1a : i1, !quantum.bit

            // CHECK: [[out_qreg1:%.+]] = quantum.insert [[graph_reg]][0], {{%.+}} : !quantum.reg, !quantum.bit
            // CHECK: [[out_qreg2:%.+]] = quantum.insert [[out_qreg1]][1], {{%.+}} : !quantum.reg, !quantum.bit
            %reg0 = quantum.insert %qreg[0], %q0b : !quantum.reg, !quantum.bit
            %reg1 = quantum.insert %reg0[1], %q1b : !quantum.reg, !quantum.bit

            func.return
        }
        """

        pipeline = (NullDecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_adj_matrix_reuse(self, run_filecheck):
        """Test that the null-decompose-graph-state pass supports the case where we reuse the
        adjacency matrix resulting from a constant op multiple times.
        """

        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: quantum.alloc(3)
            // CHECK: quantum.alloc(3)

            %adj_matrix = arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            %qreg1 = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            %qreg2 = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (NullDecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_with_stablehlo_constant(self, run_filecheck):
        """Test that the null-decompose-graph-state pass supports the case where the adjacency matrix
        results from a `stablehlo.constant` op (rather than an `arith.constant` op).
        """

        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: stablehlo.constant
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: quantum.alloc(3)

            %adj_matrix = "stablehlo.constant"() <{value = dense<[1, 0, 1]> : tensor<3xi1>}> : () -> tensor<3xi1>
            %qreg1 = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (NullDecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)
