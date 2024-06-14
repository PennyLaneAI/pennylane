# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Snapshot operation."""
import numpy as np

# pylint: disable=protected-access
import pytest

import pennylane as qml
from pennylane import Snapshot


class TestBarrier:
    """Tests that the Barrier gate is correct."""

    def test_use_barrier(self):
        r"""Test that the barrier influences compilation."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(qfunc, dev)
        gates = qml.specs(qnode)()["resources"].gate_sizes[1]

        assert gates == 3

        optimized_qfunc = qml.compile(qfunc)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["resources"].gate_sizes[1]

        assert optimized_gates == 2

    def test_barrier_only_visual(self):
        r"""Test that the barrier doesn't influence compilation when the only_visual parameter is True."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Barrier(only_visual=True, wires=0)
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        optimized_qfunc = qml.compile(qfunc)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["resources"].gate_sizes[1]

        assert optimized_gates == 0

    def test_barrier_edge_cases(self):
        r"""Test that the barrier works in edge cases."""

        def qfunc():
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.Barrier(wires=0)
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(qfunc, dev)
        gates = qml.specs(qnode)()["resources"].gate_sizes[1]

        assert gates == 4

        optimized_qfunc = qml.compile(qfunc)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["resources"].gate_sizes[1]

        assert optimized_gates == 0

        def qfunc1():
            qml.Hadamard(wires=0)
            qml.Barrier(wires=0)
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)

            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(qfunc1, dev)
        gates = qml.specs(qnode)()["resources"].gate_sizes[1]

        assert gates == 4

        def qfunc2():
            qml.Hadamard(wires=0)
            qml.Barrier(only_visual=True, wires=0)
            qml.Barrier(wires=0)
            qml.Hadamard(wires=0)

            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=3)
        optimized_qfunc = qml.compile(qfunc2)
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_gates = qml.specs(optimized_qnode)()["resources"].gate_sizes[1]

        assert optimized_gates == 2

    def test_barrier_adjoint(self):
        """Test if adjoint of a Barrier is decomposed correctly."""

        base = qml.Barrier(wires=(0, 1))
        adj = qml.ops.op_math.Adjoint(base)

        assert adj.decomposition()[0].name == "Barrier"

    def test_barrier_control(self):
        """Test if Barrier is correctly included in queue when controlling"""
        dev = qml.device("default.qubit", wires=3)

        def barrier():
            qml.PauliX(wires=0)
            qml.Barrier(wires=[0, 1])
            qml.CNOT(wires=[0, 1])

        @qml.qnode(dev)
        def circuit():
            barrier()
            qml.ctrl(barrier, 2)()
            return qml.state()

        circuit()
        tape = circuit.tape.expand(stop_at=lambda op: op.name in ["Barrier", "PauliX", "CNOT"])

        assert tape.operations[1].name == "Barrier"
        assert tape.operations[4].name == "Barrier"

    def test_barrier_empty_wire_list_no_error(self):
        """Test that barrier does not raise an error when instantiated with wires=[]."""
        barrier = qml.Barrier(wires=[])
        assert isinstance(barrier, qml.Barrier)

    def test_simplify_only_visual_one_wire(self):
        """Test that if `only_visual=True`, the operation simplifies to the identity."""
        op = qml.Barrier(wires="a", only_visual=True)
        simplified = op.simplify()
        assert qml.equal(simplified, qml.Identity("a"))

    def test_simplify_only_visual_multiple_wires(self):
        """Test that if `only_visual=True`, the operation simplifies to a product of identities."""
        op = qml.Barrier(wires=(0, 1, 2), only_visual=True)
        simplified = op.simplify()
        assert isinstance(simplified, qml.ops.op_math.Prod)
        for i, op in enumerate(simplified.operands):
            assert qml.equal(op, qml.Identity(i))

    def test_simplify_only_visual_False(self):
        """Test that no simplification occurs if only_visual is False."""
        op = qml.Barrier(wires=(0, 1, 2, 3), only_visual=False)
        assert op.simplify() is op

    def test_qml_matrix_fails(self):
        """Test that qml.matrix(op) and op.matrix() both fail."""
        op = qml.Barrier(0)
        with pytest.raises(qml.operation.MatrixUndefinedError):
            qml.matrix(op)
        with pytest.raises(qml.operation.MatrixUndefinedError):
            op.matrix()


class TestWireCut:
    """Tests for the WireCut operator"""

    def test_behaves_as_identity(self):
        """Tests that the WireCut operator behaves as the Identity in the
        absence of cutting"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def with_wirecut():
            qml.PauliX(wires=0)
            qml.WireCut(wires=0)
            return qml.state()

        @qml.qnode(dev)
        def without_wirecut():
            qml.PauliX(wires=0)
            return qml.state()

        assert np.allclose(with_wirecut(), without_wirecut())

    def test_wires_empty_list_raises_error(self):
        """Test that the WireCut operator raises an error when instantiated with an empty list."""
        with pytest.raises(
            ValueError, match="WireCut: wrong number of wires. At least one wire has to be given."
        ):
            qml.WireCut(wires=[])

    def test_qml_matrix_fails(self):
        """Test that qml.matrix(op) and op.matrix() both fail."""
        op = qml.WireCut(0)
        with pytest.raises(qml.operation.MatrixUndefinedError):
            qml.matrix(op)
        with pytest.raises(qml.operation.MatrixUndefinedError):
            op.matrix()


def test_decomposition():
    """Test the decomposition of the Snapshot operation."""

    assert Snapshot.compute_decomposition() == []
    assert Snapshot().decomposition() == []


def test_label_method():
    """Test the label method for the Snapshot operation."""
    assert Snapshot().label() == "|Snap|"
    assert Snapshot("my_label").label() == "|Snap|"


def test_control():
    """Test the _controlled method for the Snapshot operation."""
    assert isinstance(Snapshot()._controlled(0), Snapshot)
    assert Snapshot("my_label")._controlled(0).tag == Snapshot("my_label").tag


def test_adjoint():
    """Test the adjoint method for the Snapshot operation."""
    assert isinstance(Snapshot().adjoint(), Snapshot)
    assert Snapshot("my_label").adjoint().tag == Snapshot("my_label").tag


def test_snapshot_no_empty_wire_list_error():
    """Test that Snapshot does not raise an empty wire error."""
    snapshot = qml.Snapshot()
    assert isinstance(snapshot, qml.Snapshot)
