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

from functools import partial

import numpy as np
import pytest

import pennylane as qp
from pennylane.exceptions import WireError


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
    assert op.label() == "|∇⟩"
    assert op.label(base_label="grad") == "grad"


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("num_wires", [1, 2, 3, 4, 5, 6])
    def test_correct_gates_in_decomposition(self, num_wires):
        """Test that only discrete gates are used for up to three wires."""
        wires = ["a", "b", "c", "d", "e", "f"][:num_wires]
        op = qp.PhaseGradientStatePrep(wires=wires)
        with qp.queuing.AnnotatedQueue() as q:
            returned_list = op.decomposition()
        # queued_list = qp.tape.QuantumScript.from_queue(q)

        phase_gates = [qp.Z, qp.adjoint(qp.S), qp.adjoint(qp.T)]
        phase_gates += [partial(qp.PhaseShift, phi=-np.pi / 2**i) for i in range(3, num_wires)]
        expected = [qp.H(w) for w in wires] + [gate(wires=w) for gate, w in zip(phase_gates, wires)]
        assert returned_list == expected
        assert q.queue == expected

    @pytest.mark.parametrize("num_wires", [1, 2, 3, 4, 7])
    def test_decomposition_prepares_state(self, num_wires):
        """Test that executing the decomposition prepares the phase gradient state."""
        gate_set = {"Hadamard", "PauliZ", "Adjoint(S)", "Adjoint(T)", "PhaseShift"}

        @qp.transforms.decompose(gate_set=gate_set)
        @qp.qnode(qp.device("default.qubit", wires=num_wires))
        def circuit():
            qp.PhaseGradientStatePrep(wires=range(num_wires))
            return qp.state()

        tape = qp.workflow.construct_tape(circuit)()
        print([op.name for op in tape.operations])
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
