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

"""Unit and integration tests for the Python compiler `outline_state_evolution` transform."""

import pytest

pytestmark = pytest.mark.external

xdsl = pytest.importorskip("xdsl")
catalyst = pytest.importorskip("catalyst")

import pennylane as qml
from pennylane.compiler.python_compiler.transforms.outline_state_evolution import (
    OutlineStateEvolutionPass,
)


class TestOutlineStateEvolutionPass:
    """Unit tests for the outline-state-evolution pass."""

    def test_1_wire_expval(self, run_filecheck):
        """Test the outline-state-evolution pass on a simple 1-wire, 1-gate circuit."""
        program = """
        module @module_circuit {
            // CHECK-LABEL: circuit
            func.func public @circuit() -> tensor<f64> {
                %c0_i64 = arith.constant 0 : i64
                quantum.device shots(%c0_i64) ["", "", ""]
                %0 = quantum.alloc( 1) : !quantum.reg
                %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
                %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
                %2 = quantum.namedobs %out_qubits[ PauliZ] : !quantum.obs
                %3 = quantum.expval %2 : f64
                %from_elements = tensor.from_elements %3 : tensor<f64>
                %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
                quantum.dealloc %4 : !quantum.reg
                quantum.device_release
                return %from_elements : tensor<f64>
            }
        }
        """

        pipeline = (OutlineStateEvolutionPass(),)
        run_filecheck(program, pipeline)
