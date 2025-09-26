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
Unit tests for the `transforms.zx.todd` transform.
"""
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.tape import QuantumScript

pytest.importorskip("pyzx")


def test_import_pyzx_error(monkeypatch):
    """Test that a ModuleNotFoundError is raised by the todd transform
    when the pyzx external package is not installed."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pyzx", None)

        qs = QuantumScript(ops=[], measurements=[])

        with pytest.raises(ModuleNotFoundError, match="The `pyzx` package is required."):
            qml.transforms.zx.todd(qs)


@pytest.mark.external
class TestTODD:

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
        (new_qs,), _ = qml.transforms.zx.todd(qs)

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
        (new_qs,), _ = qml.transforms.zx.todd(qs)

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
        (new_qs,), _ = qml.transforms.zx.todd(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "gate",
        (
            # non-Clifford or T gates
            qml.RX(0.5, wires=0),
            qml.RY(0.5, wires=0),
            qml.RZ(0.5, wires=0),
            qml.U1(0.1, wires=0),
            qml.U2(0.1, 0.2, wires=0),
            qml.U3(0.1, 0.2, 0.3, wires=0),
            qml.CRX(0.5, wires=[0, 1]),
            qml.CRY(0.5, wires=[0, 1]),
            qml.CRZ(0.5, wires=[0, 1]),
        ),
    )
    def test_non_clifford_or_T_gates_error(self, gate):
        """Test that an error is raised when the input circuit is not Clifford + T."""
        qs = QuantumScript(ops=[gate])

        with pytest.raises(TypeError, match=r"The input circuit must be a Clifford \+ T circuit."):
            qml.transforms.zx.todd(qs)

    @pytest.mark.parametrize(
        "angle, expected_ops",
        (
            (0, []),
            (0.25 * np.pi, [qml.T(0)]),
            (0.5 * np.pi, [qml.S(0)]),
            (0.75 * np.pi, [qml.T(0), qml.S(0)]),
            (np.pi, [qml.Z(0)]),
            (1.25 * np.pi, [qml.T(0), qml.Z(0)]),
            (1.5 * np.pi, [qml.adjoint(qml.S(0))]),
            (1.75 * np.pi, [qml.T(0), qml.adjoint(qml.S(0))]),
            (2 * np.pi, []),
        ),
    )
    def test_RZ_rotation_with_Clifford_T_angle(self, angle, expected_ops):
        """Test that RZ rotation gates are transformed into the corresponding sequence of Clifford + T gates."""
        ops = [qml.RZ(angle, wires=0)]

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.todd(qs)

        assert new_qs.operations == expected_ops

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
            qml.CNOT(wires=[0, 1]),
            qml.S(wires=0),
            qml.T(wires=0),
            qml.T(wires=1),
            qml.CNOT(wires=[0, 2]),
            qml.T(wires=1),
        ]
        original_tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        (transformed_tape,), _ = qml.transforms.zx.todd(original_tape)

        expected_ops = [
            qml.adjoint(qml.S(wires=0)),
            qml.S(wires=1),
            qml.CZ(wires=[1, 0]),
            qml.CNOT(wires=[0, 2]),
            qml.CNOT(wires=[0, 1]),
        ]

        assert transformed_tape.operations == expected_ops
        assert transformed_tape.measurements == measurements

    def test_equivalent_state(self):
        """Test that the output state returned by the transformed QNode matches
        the output state returned by the original QNode up to a global phase."""
        num_wires = 3
        device = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(device)
        def original_circ():
            for i in range(num_wires):
                qml.Hadamard(wires=i)
            qml.T(wires=0)
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(np.pi / 2, wires=0)
            qml.S(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.RZ(-np.pi, wires=2)
            return qml.state()

        reduced_circ = qml.transforms.zx.todd(original_circ)

        state1 = original_circ()
        state2 = reduced_circ()

        # test that the states are equivalent up to a global phase
        check = qml.math.fidelity_statevector(state1, state2)
        assert np.isclose(check, 1)

        u1 = qml.matrix(original_circ, wire_order=range(num_wires))()
        u2 = qml.matrix(reduced_circ, wire_order=range(num_wires))()

        # test that the unitaries are equivalent up to a global phase
        prod = u1 @ np.conj(u2.T)
        glob_phase = prod[0, 0]
        assert np.allclose(prod, glob_phase * np.eye(2**num_wires))
