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
Unit tests for the `transforms.zx.optimize_t_count` transform.
"""
import sys

import numpy as np
import pytest

import pennylane as qp
from pennylane.tape import QuantumScript

pytest.importorskip("pyzx")


def test_import_pyzx_error(monkeypatch):
    """Test that a ModuleNotFoundError is raised by the optimize_t_count transform
    when the pyzx external package is not installed."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pyzx", None)

        qs = QuantumScript(ops=[], measurements=[])

        with pytest.raises(ModuleNotFoundError, match="The `pyzx` package is required."):
            qp.transforms.zx.optimize_t_count(qs)


@pytest.mark.external
class TestOptimizeTCount:

    @pytest.mark.parametrize(
        "gate",
        (
            # 1-qubit hermitian gates
            qp.Identity(wires=0),
            qp.PauliX(wires=0),
            qp.PauliY(wires=0),
            qp.PauliZ(wires=0),
            qp.Hadamard(wires=0),
            # 2-qubit hermitian gates
            qp.CNOT(wires=[0, 1]),
            qp.CY(wires=[0, 1]),
            qp.CZ(wires=[0, 1]),
            qp.CH(wires=[0, 1]),
            qp.SWAP(wires=[0, 1]),
        ),
    )
    def test_hermitian_involutory_gates_cancellation(self, gate):
        """Test cancellation for each supported Hermitian gate (involution property HH=I)"""
        ops = [gate, gate]

        qs = QuantumScript(ops)
        (new_qs,), _ = qp.transforms.zx.optimize_t_count(qs)

        assert new_qs.operations == []

    @pytest.mark.parametrize(
        "num_gates, expected_ops",
        (
            (1, [qp.S(0)]),
            (2, [qp.Z(0)]),
            (4, []),
        ),
    )
    def test_S_gate_simplification(self, num_gates, expected_ops):
        """Test S gate simplification/cancellation."""
        ops = [qp.S(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qp.transforms.zx.optimize_t_count(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "num_gates, expected_ops",
        (
            (1, [qp.T(0)]),
            (2, [qp.S(0)]),
            (4, [qp.Z(0)]),
            (8, []),
        ),
    )
    def test_T_gate_simplification(self, num_gates, expected_ops):
        """Test T gate simplification/cancellation."""
        ops = [qp.T(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qp.transforms.zx.optimize_t_count(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "gate",
        (
            # non-Clifford or T gates
            qp.RX(0.5, wires=0),
            qp.RY(0.5, wires=0),
            qp.RZ(0.5, wires=0),
            qp.U1(0.1, wires=0),
            qp.U2(0.1, 0.2, wires=0),
            qp.U3(0.1, 0.2, 0.3, wires=0),
            qp.CRX(0.5, wires=[0, 1]),
            qp.CRY(0.5, wires=[0, 1]),
            qp.CRZ(0.5, wires=[0, 1]),
        ),
    )
    def test_non_clifford_or_T_gates_error(self, gate):
        """Test that an error is raised when the input circuit is not Clifford + T."""
        qs = QuantumScript(ops=[gate])

        with pytest.raises(TypeError, match=r"The input circuit must be a Clifford \+ T circuit."):
            qp.transforms.zx.optimize_t_count(qs)

    @pytest.mark.parametrize(
        "angle, expected_ops",
        (
            (0, []),
            (0.25 * np.pi, [qp.T(0)]),
            (0.5 * np.pi, [qp.S(0)]),
            (0.75 * np.pi, [qp.Z(0), qp.adjoint(qp.T(0))]),
            (np.pi, [qp.Z(0)]),
            (1.25 * np.pi, [qp.Z(0), qp.T(0)]),
            (1.5 * np.pi, [qp.adjoint(qp.S(0))]),
            (1.75 * np.pi, [qp.adjoint(qp.T(0))]),
            (2 * np.pi, []),
        ),
    )
    def test_RZ_rotation_with_Clifford_T_angle(self, angle, expected_ops):
        """Test that RZ rotation gates are transformed into the corresponding sequence of Clifford + T gates."""
        ops = [qp.RZ(angle, wires=0)]

        qs = QuantumScript(ops)
        (new_qs,), _ = qp.transforms.zx.optimize_t_count(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "measurements",
        (
            [],
            [qp.expval(qp.Z(0))],
            [qp.probs()],
            [qp.state()],
        ),
    )
    def test_transformed_tape(self, measurements):
        """Test that the operations of the transformed tape match the expected operations
        and that the original measurements are not touched."""
        ops = [
            qp.T(wires=0),
            qp.CNOT(wires=[0, 1]),
            qp.S(wires=0),
            qp.T(wires=0),
            qp.T(wires=1),
            qp.CNOT(wires=[0, 2]),
            qp.T(wires=1),
        ]
        original_tape = qp.tape.QuantumScript(ops=ops, measurements=measurements)

        (transformed_tape,), _ = qp.transforms.zx.optimize_t_count(original_tape)

        expected_ops = [
            qp.Z(wires=0),
            qp.CNOT(wires=[0, 1]),
            qp.S(wires=1),
            qp.CNOT(wires=[0, 2]),
        ]

        assert transformed_tape.operations == expected_ops
        assert transformed_tape.measurements == measurements

    def test_equivalent_state(self):
        """Test that the output state returned by the transformed QNode matches
        the output state returned by the original QNode up to a global phase."""
        num_wires = 3
        device = qp.device("default.qubit", wires=num_wires)

        @qp.qnode(device)
        def original_circ():
            for i in range(num_wires):
                qp.Hadamard(wires=i)
            qp.T(wires=0)
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            qp.RZ(np.pi / 2, wires=0)
            qp.S(wires=2)
            qp.CNOT(wires=[1, 2])
            qp.RZ(-np.pi, wires=2)
            return qp.state()

        reduced_circ = qp.transforms.zx.optimize_t_count(original_circ)

        state1 = original_circ()
        state2 = reduced_circ()

        # test that the states are equivalent up to a global phase
        check = qp.math.fidelity_statevector(state1, state2)
        assert np.isclose(check, 1)

        u1 = qp.matrix(original_circ, wire_order=range(num_wires))()
        u2 = qp.matrix(reduced_circ, wire_order=range(num_wires))()

        # test that the unitaries are equivalent up to a global phase
        prod = u1 @ np.conj(u2.T)
        glob_phase = prod[0, 0]
        assert np.allclose(prod, glob_phase * np.eye(2**num_wires))
