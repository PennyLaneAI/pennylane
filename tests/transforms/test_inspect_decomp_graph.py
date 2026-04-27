# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the inspect_decomp_graph transform."""

import pytest

import pennylane as qp
from pennylane.transforms import inspect_decomp_graph


@pytest.mark.usefixtures("disable_graph_decomposition")
def test_error_raised_graph_disabled():
    """Tests that an error is raised if graph is disabled."""

    @inspect_decomp_graph
    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        qp.CRX(0.5, [0, 1])
        return qp.probs()

    with pytest.raises(ValueError, match="only relevant with the new graph-based decomposition"):
        circuit()


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestInspectDecompGraph:
    """Tests the inspect_decomp_graph transform."""

    def test_non_existent_op(self):
        """Tests that the correct message is produced for a non existent op."""

        @inspect_decomp_graph(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.CRX(0.5, wires=[0, 1])
            return qp.probs()

        inspector = circuit()
        assert inspector.inspect_decomps(qp.CRY(0.5, wires=[0, 1])) == (
            "This operator is not found in the decomposition graph! This typically "
            "means that this operator was not part of the original circuit, nor is it "
            "produced by any of the operators' decomposition rules."
        )

    def test_op_type_error(self):
        """Tests that a proper error is raised when an operator type is provided."""

        @inspect_decomp_graph(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.CRX(0.5, wires=[0, 1])
            return qp.probs()

        inspector = circuit()
        with pytest.raises(TypeError, match="takes a concrete operator instance as"):
            inspector.inspect_decomps(qp.CRX)

    def test_work_wire_budget_mismatch(self):
        """Tests that a correct message is produced when the work wire budget is wrong."""

        @inspect_decomp_graph(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT, num_work_wires=1)
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.MultiControlledX([0, 1, 2, 3, 4])
            return qp.probs()

        inspector = circuit()
        assert inspector.inspect_decomps(
            qp.MultiControlledX([0, 1, 2, 3, 4]), num_work_wires=2
        ) == (
            "The decomposition graph was solved with 1 work wires available for dynamic "
            "allocation at the top level. There is not a point where a MultiControlledX("
            "num_control_wires=4, num_work_wires=0, num_zero_control_values=0, work_wire_type"
            "=borrowed) is decomposed with a dynamic allocation budget of 2."
        )
