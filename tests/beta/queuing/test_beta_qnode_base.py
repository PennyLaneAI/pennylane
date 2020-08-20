# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` :class:`QNode` class.
"""
import contextlib
import io
import textwrap

import pytest
import numpy as np

import pennylane as qml
from pennylane._device import Device
from pennylane.variable import Variable
from pennylane.wires import Wires, WireError

# Beta imports

# No BetaBaseQNode import
from pennylane.qnodes.base import QuantumFunctionError, decompose_queue

# BetaBetaBaseQNode import
from pennylane.beta.queuing.base import BetaBaseQNode
from pennylane.beta.queuing.queuing import QueuingContext
from pennylane.beta.queuing.operation import BetaTensor
import pennylane.beta.queuing.measure as beta_measure

@pytest.fixture(scope="function")
def mock_qnode(mock_device, monkeypatch):
    """Provides a circuit for the subsequent tests of the operation queue"""

    with monkeypatch.context() as m:
        m.setattr(qml, 'QueuingContext', QueuingContext)

        def circuit(x):
            qml.RX(x, wires=[0])
            qml.CNOT(wires=[0, 1])
            qml.RY(0.4, wires=[0])
            qml.RZ(-0.2, wires=[1])
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))

        node = BetaBaseQNode(circuit, mock_device)
        node._construct([1.0], {})
        return node

class TestQNodeOperationQueue:
    """Tests that the QNode operation queue is properly filled and interacted with"""

    def test_operation_ordering(self, mock_qnode, mock_device):
        """Tests that the ordering of the operations is correct"""

        qnode = mock_qnode
        assert qnode.ops[0].name == "RX"
        assert qnode.ops[1].name == "CNOT"
        assert qnode.ops[2].name == "RY"
        assert qnode.ops[3].name == "RZ"
        assert qnode.ops[4].name == "PauliX"
        assert qnode.ops[5].name == "PauliZ"

    def test_op_descendants_operations_only(self, mock_qnode):
        """Tests that _op_descendants properly extracts the successors that are operations"""

        qnode = mock_qnode
        operation_successors = qnode._op_descendants(qnode.ops[0], only="G")
        assert qnode.ops[0] not in operation_successors
        assert qnode.ops[1] in operation_successors
        assert qnode.ops[4] not in operation_successors

    def test_op_descendants_observables_only(self, mock_qnode):
        """Tests that _op_descendants properly extracts the successors that are observables"""

        qnode = mock_qnode
        observable_successors = qnode._op_descendants(qnode.ops[0], only="O")
        assert qnode.ops[0] not in observable_successors
        assert qnode.ops[1] not in observable_successors
        assert qnode.ops[4] in observable_successors

    def test_op_descendants_both_operations_and_observables(self, mock_qnode):
        """Tests that _op_descendants properly extracts all successors"""

        qnode = mock_qnode
        successors = qnode._op_descendants(qnode.ops[0], only=None)
        assert qnode.ops[0] not in successors
        assert qnode.ops[1] in successors
        assert qnode.ops[4] in successors

    def test_op_descendants_both_operations_and_observables_nodes(self, mock_qnode):
        """Tests that _op_descendants properly extracts all successor nodes"""

        qnode = mock_qnode
        successors = qnode._op_descendants(qnode.ops[0], only=None)
        assert qnode.circuit.operations[0] not in successors
        assert qnode.circuit.operations[1] in successors
        assert qnode.circuit.operations[2] in successors
        assert qnode.circuit.operations[3] in successors
        assert qnode.circuit.observables[0] in successors

    def test_op_descendants_both_operations_and_observables_strict_ordering(self, mock_qnode):
        """Tests that _op_descendants properly extracts all successors"""

        qnode = mock_qnode
        successors = qnode._op_descendants(qnode.ops[2], only=None)
        assert qnode.circuit.operations[0] not in successors
        assert qnode.circuit.operations[1] not in successors
        assert qnode.circuit.operations[2] not in successors
        assert qnode.circuit.operations[3] not in successors
        assert qnode.circuit.observables[0] in successors

    def test_op_descendants_extracts_all_successors(self, mock_qnode):
        """Tests that _op_descendants properly extracts all successors"""

        qnode = mock_qnode
        successors = qnode._op_descendants(qnode.ops[2], only=None)
        assert qnode.ops[4] in successors
        assert qnode.ops[5] not in successors

    def test_operation_appending(self, mock_device, monkeypatch):
        """Tests that operations are correctly appended."""
        with monkeypatch.context() as m:
            m.setattr(qml, 'QueuingContext', QueuingContext)

            CNOT = qml.CNOT(wires=[0, 1])

            def circuit(x):
                qml.QueuingContext.append(CNOT)
                qml.RY(0.4, wires=[0])
                qml.RZ(-0.2, wires=[1])

                return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))

            qnode = BetaBaseQNode(circuit, mock_device)
            qnode._construct([1.0], {})

            assert qnode.ops[0].name == "CNOT"
            assert qnode.ops[1].name == "RY"
            assert qnode.ops[2].name == "RZ"
            assert qnode.ops[3].name == "PauliX"

    def test_operation_removal(self, mock_device, monkeypatch):
        """Tests that operations are correctly removed."""

        with monkeypatch.context() as m:
            m.setattr(qml, 'QueuingContext', QueuingContext)

            def circuit(x):
                RX = qml.RX(x, wires=[0])
                qml.CNOT(wires=[0, 1])
                qml.RY(0.4, wires=[0])
                qml.RZ(-0.2, wires=[1])

                qml.QueuingContext.remove(RX)

                return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliZ(1))

            qnode = BetaBaseQNode(circuit, mock_device)
            qnode._construct([1.0], {})

            assert qnode.ops[0].name == "CNOT"
            assert qnode.ops[1].name == "RY"
            assert qnode.ops[2].name == "RZ"
            assert qnode.ops[3].name == "PauliX"
            assert qnode.ops[4].name == "PauliZ"
