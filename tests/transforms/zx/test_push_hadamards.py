# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the `transforms.zx.push_hadamards` transform.
"""
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.tape import QuantumScript

pytest.importorskip("pyzx")


def test_import_pyzx_error(monkeypatch):
    """Test that a ModuleNotFoundError is raised by the push_hadamards transform
    when the pyzx external package is not installed."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pyzx", None)

        qs = QuantumScript(ops=[], measurements=[])

        with pytest.raises(ModuleNotFoundError, match="The `pyzx` package is required."):
            qml.transforms.zx.push_hadamards(qs)


@pytest.mark.external
class TestPushHadamards:

    @pytest.mark.parametrize(
        "gate",
        (
            # 1-qubit hermitian gates
            qml.Identity(wires=0),
            qml.PauliX(wires=0),
            qml.PauliY(wires=0),
            qml.PauliZ(wires=0),
            qml.Hadamard(wires=0),
            # 2-qubit hermitian gates
            qml.CNOT(wires=[0, 1]),
            qml.CY(wires=[0, 1]),
            qml.CZ(wires=[0, 1]),
            qml.CH(wires=[0, 1]),
            qml.SWAP(wires=[0, 1]),
        ),
    )
    def test_hermitian_involutory_gates_cancellation(self, gate):
        """Test cancellation for each supported Hermitian gate (involution property HH=I)"""
        ops = [gate, gate]

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.push_hadamards(qs)

        assert new_qs.operations == []

    @pytest.mark.parametrize(
        "num_gates, expected_ops",
        (
            (1, [qml.S(0)]),
            (2, [qml.Z(0)]),
            (4, []),
        ),
    )
    def test_S_gate_simplification(self, num_gates, expected_ops):
        """Test S gate simplification/cancellation."""
        ops = [qml.S(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.push_hadamards(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "num_gates, expected_ops",
        (
            (1, [qml.T(0)]),
            (2, [qml.S(0)]),
            (4, [qml.Z(0)]),
            (8, []),
        ),
    )
    def test_T_gate_simplification(self, num_gates, expected_ops):
        """Test T gate simplification/cancellation."""
        ops = [qml.T(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.push_hadamards(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "rot_gate",
        (qml.RX, qml.RY),
    )
    def test_rotation_gates_error(self, rot_gate):
        """Test that an error is raised when the input circuit contains RX or RY rotation gates."""
        qs = QuantumScript(ops=[rot_gate(0.5, wires=0)])

        with pytest.raises(
            TypeError,
            match=r"The input quantum circuit must be a phase-polynomial \+ Hadamard circuit.",
        ):
            qml.transforms.zx.push_hadamards(qs)

    @pytest.mark.parametrize(
        "measurements",
        (
            [],
            [qml.expval(qml.Z(0))],
            [qml.probs()],
            [qml.state()],
        ),
    )
    def test_transformed_tape(self, measurements):
        """Test that the operations of the transformed tape match the expected operations
        and that the original measurements are not touched."""
        ops = [
            qml.T(wires=0),
            qml.Hadamard(wires=0),
            qml.Hadamard(wires=0),
            qml.T(wires=1),
            qml.Hadamard(wires=1),
            qml.CNOT(wires=[1, 2]),
            qml.Hadamard(wires=1),
            qml.Hadamard(wires=2),
        ]
        original_tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        (transformed_tape,), _ = qml.transforms.zx.push_hadamards(original_tape)

        expected_ops = [
            qml.T(wires=0),
            qml.T(wires=1),
            qml.Hadamard(wires=2),
            qml.CNOT(wires=[2, 1]),
        ]

        assert transformed_tape.operations == expected_ops
        assert transformed_tape.measurements == measurements

    def test_equivalent_state(self):
        """Test that the output state returned by the transformed QNode matches
        the output state returned by the original QNode for a simple circuit."""
        num_wires = 3
        device = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(device)
        def original_circ():
            for i in range(num_wires):
                qml.Hadamard(wires=i)
            qml.T(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=0)
            qml.S(wires=2)
            qml.CNOT(wires=[1, 2])
            return qml.state()

        reduced_circ = qml.transforms.zx.push_hadamards(original_circ)

        state1 = original_circ()
        state2 = reduced_circ()

        assert np.allclose(state1, state2)
