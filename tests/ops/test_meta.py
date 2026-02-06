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

import pennylane as qp
from pennylane import Snapshot


class TestBarrier:
    """Tests that the Barrier gate is correct."""

    def test_use_barrier(self):
        r"""Test that the barrier influences compilation."""

        def qfunc():
            qp.Hadamard(wires=0)
            qp.Barrier(wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        dev = qp.device("default.qubit", wires=3)
        qnode = qp.QNode(qfunc, dev)
        gates = qp.specs(qnode)()["resources"].gate_sizes[1]

        assert gates == 3

        optimized_qfunc = qp.compile(qfunc)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)
        optimized_gates = qp.specs(optimized_qnode)()["resources"].gate_sizes[1]

        assert optimized_gates == 2

    def test_barrier_only_visual(self):
        r"""Test that the barrier doesn't influence compilation when the only_visual parameter is True."""

        def qfunc():
            qp.Hadamard(wires=0)
            qp.Barrier(only_visual=True, wires=0)
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        dev = qp.device("default.qubit", wires=3)
        optimized_qfunc = qp.compile(qfunc)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)

        assert 1 not in qp.specs(optimized_qnode)()["resources"].gate_sizes

    def test_barrier_edge_cases(self):
        r"""Test that the barrier works in edge cases."""

        def qfunc():
            qp.Barrier(wires=0)
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.Barrier(wires=0)
            return qp.expval(qp.PauliZ(0))

        dev = qp.device("default.qubit", wires=3)
        qnode = qp.QNode(qfunc, dev)
        gates = qp.specs(qnode)()["resources"].gate_sizes[1]

        assert gates == 4

        optimized_qfunc = qp.compile(qfunc)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)
        assert 1 not in qp.specs(optimized_qnode)()["resources"].gate_sizes

        def qfunc1():
            qp.Hadamard(wires=0)
            qp.Barrier(wires=0)
            qp.Barrier(wires=0)
            qp.Hadamard(wires=0)

            return qp.expval(qp.PauliZ(0))

        dev = qp.device("default.qubit", wires=3)
        qnode = qp.QNode(qfunc1, dev)
        gates = qp.specs(qnode)()["resources"].gate_sizes[1]

        assert gates == 4

        def qfunc2():
            qp.Hadamard(wires=0)
            qp.Barrier(only_visual=True, wires=0)
            qp.Barrier(wires=0)
            qp.Hadamard(wires=0)

            return qp.expval(qp.PauliZ(0))

        dev = qp.device("default.qubit", wires=3)
        optimized_qfunc = qp.compile(qfunc2)
        optimized_qnode = qp.QNode(optimized_qfunc, dev)
        optimized_gates = qp.specs(optimized_qnode)()["resources"].gate_sizes[1]

        assert optimized_gates == 2

    def test_barrier_adjoint(self):
        """Test if adjoint of a Barrier is decomposed correctly."""

        base = qp.Barrier(wires=(0, 1))
        adj = qp.ops.op_math.Adjoint(base)

        assert adj.decomposition()[0].name == "Barrier"

    def test_barrier_control(self):
        """Test if Barrier is correctly included in queue when controlling"""
        dev = qp.device("default.qubit", wires=3)

        def barrier():
            qp.PauliX(wires=0)
            qp.Barrier(wires=[0, 1])
            qp.CNOT(wires=[0, 1])

        @qp.qnode(dev)
        def circuit():
            barrier()
            qp.ctrl(barrier, 2)()
            return qp.state()

        tape = qp.workflow.construct_tape(circuit)()
        tape = tape.expand(stop_at=lambda op: op.name in ["Barrier", "PauliX", "CNOT"])

        assert tape.operations[1].name == "Barrier"
        assert tape.operations[4].name == "Barrier"

    def test_barrier_empty_wire_list_no_error(self):
        """Test that barrier does not raise an error when instantiated with wires=[]."""
        barrier = qp.Barrier(wires=[])
        assert isinstance(barrier, qp.Barrier)

    def test_simplify_only_visual_one_wire(self):
        """Test that if `only_visual=True`, the operation simplifies to the identity."""
        op = qp.Barrier(wires="a", only_visual=True)
        simplified = op.simplify()
        qp.assert_equal(simplified, qp.Identity("a"))

    def test_simplify_only_visual_multiple_wires(self):
        """Test that if `only_visual=True`, the operation simplifies to a product of identities."""
        op = qp.Barrier(wires=(0, 1, 2), only_visual=True)
        simplified = op.simplify()
        assert isinstance(simplified, qp.ops.op_math.Prod)
        for i, op in enumerate(simplified.operands):
            qp.assert_equal(op, qp.Identity(i))

    def test_simplify_only_visual_False(self):
        """Test that no simplification occurs if only_visual is False."""
        op = qp.Barrier(wires=(0, 1, 2, 3), only_visual=False)
        assert op.simplify() is op

    def test_qp_matrix_gives_identity(self):
        """Test that qp.matrix(op) gives an identity."""
        op = qp.Barrier(0)
        assert np.allclose(qp.matrix(op), np.eye(2))
        op = qp.Barrier()
        assert np.allclose(qp.matrix(op, wire_order=[0, 3]), np.eye(4))

    def test_op_matrix_fails(self):
        """Test that qp.matrix(op) and op.matrix() both fail."""
        op = qp.Barrier(0)
        with pytest.raises(qp.operation.MatrixUndefinedError):
            op.matrix()


class TestWireCut:
    """Tests for the WireCut operator"""

    def test_behaves_as_identity(self):
        """Tests that the WireCut operator behaves as the Identity in the
        absence of cutting"""

        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def with_wirecut():
            qp.PauliX(wires=0)
            qp.WireCut(wires=0)
            return qp.state()

        @qp.qnode(dev)
        def without_wirecut():
            qp.PauliX(wires=0)
            return qp.state()

        assert np.allclose(with_wirecut(), without_wirecut())

    def test_wires_empty_list_raises_error(self):
        """Test that the WireCut operator raises an error when instantiated with an empty list."""
        with pytest.raises(
            ValueError,
            match="WireCut: wrong number of wires. At least one wire has to be provided.",
        ):
            qp.WireCut(wires=[])

    def test_qp_matrix_gives_identity(self):
        """Test that qp.matrix(op) gives an identity."""
        op = qp.WireCut(0)
        assert np.allclose(qp.matrix(op), np.eye(2))

    def test_op_matrix_fails(self):
        """Test that qp.matrix(op) and op.matrix() both fail."""
        op = qp.WireCut(0)
        with pytest.raises(qp.operation.MatrixUndefinedError):
            op.matrix()


class TestSnapshot:
    """Unit tests for the snapshot class."""

    def test_repr(self):
        """Test the repr for a Snapshot."""

        op = qp.Snapshot("my_tag", measurement=qp.expval(qp.Z(0)), shots=2)
        assert repr(op) == "<Snapshot: tag=my_tag, measurement=expval(Z(0)), shots=Shots(total=2)>"

    def test_update_tag(self):
        """Test that update_tag generates a copy with a new tag."""

        op1 = qp.Snapshot("initial_tag", measurement=qp.probs(), shots=5)

        op2 = op1.update_tag("new_tag")
        assert op2.tag == "new_tag"
        assert op2.hyperparameters["shots"] == qp.measurements.Shots(5)
        assert op2.hyperparameters["measurement"] == qp.probs()
        assert op1.tag == "initial_tag"

    def test_decomposition(self):
        """Test the decomposition of the Snapshot operation."""

        assert Snapshot.compute_decomposition() == []
        assert Snapshot().decomposition() == []

    def test_label_method(self):
        """Test the label method for the Snapshot operation."""
        assert Snapshot().label() == "|Snap|"
        assert Snapshot("my_label").label() == "|Snap|"

    def test_control(self):
        """Test the _controlled method for the Snapshot operation."""
        assert isinstance(Snapshot()._controlled(0), Snapshot)
        assert Snapshot("my_label")._controlled(0).tag == Snapshot("my_label").tag

    def test_adjoint(self):
        """Test the adjoint method for the Snapshot operation."""
        assert isinstance(Snapshot().adjoint(), Snapshot)
        assert Snapshot("my_label").adjoint().tag == Snapshot("my_label").tag

    def test_snapshot_no_empty_wire_list_error(self):
        """Test that Snapshot does not raise an empty wire error."""
        snapshot = qp.Snapshot()
        assert isinstance(snapshot, qp.Snapshot)

    def test_shots_none_for_no_measurement(self):
        """Test that the shots become None if no measurement is provided."""

        op = qp.Snapshot()
        assert op.hyperparameters["shots"] == qp.measurements.Shots(None)

    @pytest.mark.parametrize(
        "mp", (qp.expval(qp.Z(0)), qp.measurements.StateMP(wires=(2, 1, 0)))
    )
    def test_map_wires(self, mp):
        """Test that the wires of the measurement are mapped"""
        op = Snapshot(measurement=mp, tag="my tag")
        wire_map = {0: "a", 1: "b"}
        new_op = op.map_wires(wire_map)
        target_mp = mp.map_wires(wire_map)
        qp.assert_equal(target_mp, new_op.hyperparameters["measurement"])
        assert new_op.tag == "my tag"

    # pylint: disable=unused-argument
    @pytest.mark.capture
    @pytest.mark.parametrize("measurement", (None, "state"))
    def test_capture_measurement(self, measurement):
        """Test that a snapshot can be captured into plxpr."""

        import jax

        def f():
            if measurement is None:
                qp.Snapshot()
            else:
                qp.Snapshot(measurement=qp.state())

        jaxpr = jax.make_jaxpr(f)()

        if measurement is None:
            assert jaxpr.eqns[0].primitive == qp.Snapshot._primitive
        else:
            assert jaxpr.eqns[0].primitive == qp.measurements.StateMP._wires_primitive
            assert jaxpr.eqns[1].primitive == qp.Snapshot._primitive
            assert jaxpr.eqns[1].invars[0] == jaxpr.eqns[0].outvars[0]
