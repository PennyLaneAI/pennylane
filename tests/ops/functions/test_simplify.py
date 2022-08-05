# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the qml.simplify function
"""
import pennylane as qml
from pennylane.tape import QuantumTape


class TestSimplify:
    """Tests for the qml.simplify method."""

    def test_simplify_method_with_default_depth(self):
        """Test simplify method with default depth."""
        op = qml.adjoint(
            qml.adjoint(
                qml.op_sum(
                    qml.ops.Pow(qml.RX(1, wires=0)),
                    qml.PauliX(0),
                    qml.op_sum(qml.adjoint(qml.PauliX(0)), qml.PauliZ(0)),
                )
            )
        )

        sim_op = qml.simplify(op)
        assert sim_op.arithmetic_depth == 2

    def test_simplify_method_with_queuing(self):
        """Test the simplify method while queuing."""
        with qml.tape.QuantumTape() as tape:
            op = qml.adjoint(qml.adjoint(qml.PauliX(0)))
            simplified_op = qml.simplify(op)
        tape: QuantumTape
        assert len(tape.circuit) == 1
        assert tape.circuit[0] is simplified_op
