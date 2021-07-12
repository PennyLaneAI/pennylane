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
Tests for the Grover Diffusion Operator template
"""
import pytest
import numpy as np
import pennylane as qml


def test_work_wires():
    """Assert work wires get passed to MultiControlledX"""
    wires = ("a", "b")
    work_wire = ("aux",)

    op = qml.templates.GroverOperator(wires=wires, work_wires=work_wire)

    assert op.work_wires == work_wire

    ops = op.expand().operations

    assert ops[2]._work_wires == work_wire


def test_do_queue():
    """Assert do_queue=False is not queued"""

    with qml.tape.QuantumTape() as tape:
        qml.templates.GroverOperator(wires=(0, 1), do_queue=False)

    assert len(tape.operations) == 0


def test_id():
    """Assert id keyword works"""

    op = qml.templates.GroverOperator(wires=(0, 1), id="hello")

    assert op.id == "hello"


class TestDecomposition:
    def test_expand(self):
        """Tests correct expansion"""
        wires = tuple(range(3))

        op = qml.templates.GroverOperator(wires=wires)

        ops = op.expand().operations

        assert isinstance(ops[0], qml.Hadamard)
        assert isinstance(ops[1], qml.Hadamard)
        assert isinstance(ops[2], qml.PauliZ)
        assert isinstance(ops[3], qml.MultiControlledX)
        assert isinstance(ops[4], qml.PauliZ)
        assert isinstance(ops[5], qml.Hadamard)
        assert isinstance(ops[6], qml.Hadamard)

        assert ops[0].wires == (0,)
        assert ops[1].wires == (1,)
        assert ops[2].wires == (2,)
        assert ops[3].wires == wires
        assert ops[4].wires == (2,)
        assert ops[5].wires == (0,)
        assert ops[6].wires == (1,)

    def test_custom_labels(self):
        """assert decomposition works with string labels"""
        wires = ("a", "b")

        op = qml.templates.GroverOperator(wires=wires)
        ops = op.expand().operations

        assert ops[0].wires == ("a",)
        assert ops[1].wires == ("b",)
        assert ops[2].wires == wires
        assert ops[3].wires == ("b",)
        assert ops[4].wires == ("a",)


class TestIntegration:
    def test_findstate(self):
        """Asserts can find state marked by oracle."""
        wires = range(6)

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circ():
            for wire in wires:
                qml.Hadamard(wire)

            for _ in range(5):
                qml.Hadamard(wires[0])
                qml.MultiControlledX(wires=wires[0], control_wires=wires[1:])
                qml.Hadamard(wires[0])
                qml.templates.GroverOperator(wires=wires)

            return qml.probs(wires=wires)

        probs = circ()

        assert np.argmax(probs) == len(probs) - 1
