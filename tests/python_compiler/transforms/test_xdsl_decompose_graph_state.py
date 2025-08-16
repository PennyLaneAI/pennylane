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

"""Unit and integration tests for the Python compiler `decompose-graph-state` transform."""

# pylint: disable=wrong-import-position

import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")
from catalyst.passes import xdsl_plugin

import pennylane as qml
from pennylane.compiler.python_compiler.transforms import (
    DecomposeGraphStatePass,
    decompose_graph_state_pass,
)


class TestDecomposeGraphStatePass:
    """Unit tests for the decompose-graph-state pass."""

    def test_2_qubit_chain(self, run_filecheck):
        """

        0 -- 1

        0  1
        1  0

        [1]
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            // CHECK-NOT: arith.constant dense<[1]> : tensor<1xi1>
            // CHECK-NOT: mbqc.graph_state_prep

            // CHECK: [[graph_reg:%.+]] = quantum.alloc(2) : !quantum.reg
            // CHECK: [[q00:%.+]] = quantum.extract [[graph_reg]][0] : !quantum.reg -> !quantum.bit
            // CHECK: [[q10:%.+]] = quantum.extract [[graph_reg]][1] : !quantum.reg -> !quantum.bit

            // CHECK: [[q01:%.+]] = quantum.custom "Hadamard"() [[q00]] : !quantum.bit
            // CHECK: [[q11:%.+]] = quantum.custom "Hadamard"() [[q10]] : !quantum.bit
            // CHECK: [[q20:%.+]], [[q21:%.+]] = quantum.custom "CZ"() [[q01]], [[q11]] : !quantum.bit, !quantum.bit

            // CHECK: [[out_reg0:%.+]] = quantum.insert [[graph_reg]][0], [[q20]] : !quantum.reg, !quantum.bit
            // CHECK: [[out_reg1:%.+]] = quantum.insert [[out_reg0]][1], [[q21]] : !quantum.reg, !quantum.bit

            %adj_matrix = arith.constant dense<[1]> : tensor<1xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<1xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_3_qubit_chain(self, run_filecheck):
        """

        0 -- 1 -- 2

        0  1  0
        1  0  1
        0  1  0

        [1, 0, 1]
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            %adj_matrix = arith.constant dense<[1, 0, 1]> : tensor<3xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<3xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_4_qubit_square_lattice(self, run_filecheck):
        """

        0 -- 1
        |    |
        2 -- 3

        0  1  1  0
        1  0  0  1
        1  0  0  1
        0  1  1  0

        [1, 1, 0, 0, 1, 1]
        """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            %adj_matrix = arith.constant dense<[1, 1, 0, 0, 1, 1]> : tensor<6xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<6xi1>) [init "Hadamard", entangle "CZ"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    def test_2_qubit_chain_non_standard_init(self, run_filecheck):
        """ """
        program = """
        // CHECK-LABEL: circuit
        func.func @circuit() {
            %adj_matrix = arith.constant dense<[1]> : tensor<1xi1>
            %qreg = mbqc.graph_state_prep (%adj_matrix : tensor<1xi1>) [init "S", entangle "CNOT"] : !quantum.reg
            func.return
        }
        """

        pipeline = (DecomposeGraphStatePass(),)
        run_filecheck(program, pipeline)

    class TestDecomposeGraphStateIntegration:
        pass
