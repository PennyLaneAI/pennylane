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
"""
Unit tests for the PhaseGradientStatePrep template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.exceptions import WireError
from pennylane.templates.state_preparations.phase_gradient import (
    _phase_gradient_state_prep_ppr_decomposition,
)


def _expected_state(num_wires):
    dim = 2**num_wires
    return np.exp((-2j * np.pi / dim) * np.arange(dim)) / np.sqrt(dim)


@pytest.mark.parametrize("num_wires", [1, 2, 3, 4, 5, 6])
@pytest.mark.usefixtures("enable_and_disable_capture")
def test_standard_validity(num_wires):
    """Check the operation using the assert_valid function."""
    op = qp.PhaseGradientStatePrep(wires=range(num_wires))
    qp.ops.functions.assert_valid(op, skip_differentiation=True)


def test_label():
    """Test the label of the template."""
    op = qp.PhaseGradientStatePrep(wires=[0, 1])
    assert op.label() == "|∇z⟩"
    assert op.label(base_label="grad") == "grad"


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("use_qjit", [False, pytest.param(True, marks=pytest.mark.catalyst)])
    @pytest.mark.parametrize("num_wires", [1, 2, 3, 4, 7])
    def test_decomposition_prepares_state(self, num_wires, use_qjit):
        """Test that executing the decomposition prepares the phase gradient state."""

        gate_set = {"PPR", "GlobalPhase"}

        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            qp.PhaseGradientStatePrep(wires=range(num_wires))
            return qp.state()

        if use_qjit:
            import catalyst

            # TODO: Use `decompose` for this branch as well once graph_decomposition is integrated
            circuit = qp.transforms.to_ppr(
                catalyst.passes.graph_decomposition(circuit, gate_set=gate_set)
            )
            circuit = qp.qjit(circuit, capture=True)

        else:
            circuit = qp.transforms.decompose(circuit, gate_set=gate_set)
            tape = qp.workflow.construct_tape(circuit)()
            assert all(op.name in gate_set for op in tape.operations)
        assert np.allclose(circuit(), _expected_state(num_wires))

    def test_custom_wire_labels(self):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        @qp.qnode(qp.device("default.qubit", wires=3))
        def circuit():
            qp.PhaseGradientStatePrep(wires=range(3))
            return qp.state()

        @qp.qnode(qp.device("default.qubit", wires=["z", "a", "k"]))
        def circuit2():
            qp.PhaseGradientStatePrep(wires=["z", "a", "k"])
            return qp.state()

        assert np.allclose(circuit(), circuit2())


class TestPPRDecomposition:
    """Tests for the decomposition into Pauli product rotations."""

    @pytest.mark.parametrize("num_wires", [1, 2, 3, 4, 7, 10, 14, 16])
    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_prepares_state(self, num_wires):
        """Test that decomposing into PPRs prepares the phase gradient state."""

        @qp.transforms.decompose(gate_set={"PPR", "GlobalPhase"})
        @qp.qnode(qp.device("default.qubit", wires=num_wires))
        def circuit():
            qp.PhaseGradientStatePrep(wires=range(num_wires))
            return qp.state()

        tape = qp.workflow.construct_tape(circuit)()
        assert all(isinstance(op, (qp.PPR, qp.GlobalPhase)) for op in tape.operations)
        assert np.allclose(circuit(), _expected_state(num_wires), atol=1e-12, rtol=0)

    def test_precision_on_30_wires(self):
        """Test that the decomposition prepares the phase gradient state on 30 wires up to an
        error of 1e-12 in the 2-norm. The error of the product state is bounded by combining
        the single-wire errors (up to phase) in quadrature and adding the global phase error."""
        num_wires = 30
        with qp.queuing.AnnotatedQueue() as q:
            _phase_gradient_state_prep_ppr_decomposition(wires=range(num_wires))

        squared_errors = []
        phase = -q.queue[-1].data[0]
        for j in range(num_wires):
            wire_ops = [op for op in q.queue if op.wires == qp.wires.Wires(j)]
            state = qp.matrix(wire_ops, wire_order=[j])[:, 0]
            target = np.array([1, np.exp(-1j * np.pi / 2**j)]) / np.sqrt(2)
            overlap = np.vdot(target, state)
            squared_errors.append(np.linalg.norm(state - overlap / np.abs(overlap) * target) ** 2)
            phase += np.angle(overlap)
        phase_error = np.abs(np.angle(np.exp(1j * phase)))
        assert np.sqrt(np.sum(squared_errors)) + phase_error < 1e-12

    @pytest.mark.parametrize("num_wires", [1, 3, 8, 20, 30])
    def test_gate_types(self, num_wires):
        """Test that the decomposition consists of pi/8 PPRs, at most one other PPR per wire,
        and a global phase."""
        with qp.queuing.AnnotatedQueue() as q:
            _phase_gradient_state_prep_ppr_decomposition(wires=range(num_wires))

        assert all(isinstance(op, qp.PPR) for op in q.queue[:-1])
        assert isinstance(q.queue[-1], qp.GlobalPhase)
        for wire in range(num_wires):
            denominators = [abs(op.angle_denominator) for op in q.queue if op.wires == [wire]]
            assert sum(d != 8 for d in denominators) <= 1

    def test_condition(self):
        """Test that the decomposition is only applicable to up to 30 wires."""
        assert _phase_gradient_state_prep_ppr_decomposition.is_applicable(wires=range(30))
        assert not _phase_gradient_state_prep_ppr_decomposition.is_applicable(wires=range(31))


class TestStateVector:
    """Test the state_vector() method."""

    @pytest.mark.parametrize("num_wires", [1, 2, 3, 5])
    def test_state_vector(self, num_wires):
        """Tests that the state vector is correct."""
        res = qp.PhaseGradientStatePrep(wires=range(num_wires)).state_vector()
        assert res.shape == (2,) * num_wires
        assert np.allclose(np.reshape(res, (-1,)), _expected_state(num_wires))

    def test_state_vector_bad_wire_order(self):
        """Tests that the provided wire_order must contain the wires in the operation."""
        op = qp.PhaseGradientStatePrep(wires=[0, 1])
        with pytest.raises(
            WireError, match="wire_order must contain all PhaseGradientStatePrep wires"
        ):
            op.state_vector(wire_order=[1, 2])

    def test_state_vector_wire_order(self):
        """Tests that the state vector works with a different order of wires."""
        op = qp.PhaseGradientStatePrep(wires=[0, 1])
        res = op.state_vector(wire_order=[1, 0])
        expected = np.transpose(np.reshape(_expected_state(2), (2, 2)))
        assert np.allclose(res, expected)

    def test_state_vector_subset_of_wires(self):
        """Tests that the state vector works with not all state wires."""
        op = qp.PhaseGradientStatePrep([2, 1])
        res = op.state_vector(wire_order=[0, 1, 2])
        assert res.shape == (2, 2, 2)

        expected_10 = qp.PhaseGradientStatePrep([0, 1]).state_vector(wire_order=[1, 0])
        expected = np.stack([expected_10, np.zeros_like(expected_10)])
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("num_wires", [1, 2, 4])
    def test_state_vector_matches_decomposition(self, num_wires):
        """Test that the state vector matches the decomposition."""
        op = qp.PhaseGradientStatePrep(wires=range(num_wires))
        state = qp.matrix(op.decomposition, wire_order=range(num_wires))()[:, 0]
        assert np.allclose(np.reshape(op.state_vector(), (-1,)), state)


class TestPhaseGradientConsistency:
    """Test that the prepared state is consistent with the phase gradient features of PennyLane."""

    @pytest.mark.parametrize("num_wires, value", [(1, 1), (2, 1), (3, 5), (4, 11)])
    def test_addition_imprints_phase(self, num_wires, value):
        """Test that adding an integer M with SemiAdder imprints the phase exp(2πiM/B)."""
        x_wires = [f"x{i}" for i in range(num_wires)]
        grad_wires = [f"g{i}" for i in range(num_wires)]
        work_wires = [f"w{i}" for i in range(num_wires - 1)]

        @qp.qnode(qp.device("default.qubit", wires=x_wires + grad_wires + work_wires))
        def circuit(add):
            qp.BasisState(qp.math.int_to_binary(value, num_wires), wires=x_wires)
            qp.PhaseGradientStatePrep(wires=grad_wires)
            if add:
                qp.SemiAdder(x_wires, grad_wires, work_wires)
            return qp.state()

        phase = np.exp(2j * np.pi * value / 2**num_wires)
        assert np.allclose(circuit(True), phase * circuit(False))

    def test_rz_phase_gradient_transform(self):
        """Test that the template can be used as resource state for ``rz_phase_gradient``."""
        precision = 3
        phi = (1 / 2 + 1 / 8) * 2 * np.pi
        angle_wires = [f"ang_{i}" for i in range(precision)]
        phase_grad_wires = [f"phg_{i}" for i in range(precision)]
        work_wires = [f"work_{i}" for i in range(precision - 1)]

        @qp.transforms.rz_phase_gradient(
            angle_wires=angle_wires, phase_grad_wires=phase_grad_wires, work_wires=work_wires
        )
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.PhaseGradientStatePrep(wires=phase_grad_wires)
            qp.H("targ")
            qp.RZ(phi, "targ")
            qp.H("targ")
            return qp.probs("targ")

        expected = np.abs(qp.RX(phi, 0).matrix()[:, 0]) ** 2
        assert np.allclose(circuit(), expected)
