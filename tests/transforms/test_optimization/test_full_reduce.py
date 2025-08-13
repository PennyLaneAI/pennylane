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
Unit tests for the `zx.full_reduce` transform.
"""
import sys

import numpy as np
import pytest

import pennylane as qml
from pennylane.tape import QuantumScript

pytest.importorskip("pyzx")


def test_import_pyzx_error(monkeypatch):
    """Test that a ModuleNotFoundError is raised by the full_reduce transform
    when the pyzx external package is not installed."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pyzx", None)

        qs = QuantumScript(ops=[], measurements=[])

        with pytest.raises(ModuleNotFoundError, match="The `pyzx` package is required."):
            qml.transforms.zx.full_reduce(qs)


@pytest.mark.external
class TestFullReduce:

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
            # 3-qubit hermitian gates
            qml.Toffoli(wires=[0, 1, 2]),
            qml.CCZ(wires=[0, 1, 2]),
        ),
    )
    def test_hermitian_involutory_gates_cancellation(self, gate):
        ops = [gate, gate]

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.full_reduce(qs)

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
        ops = [qml.S(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.full_reduce(qs)

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
        ops = [qml.T(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.full_reduce(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "params",
        (
            (1.7, 0.0),
            (3.1, -0.5),
            (0.1, 0.9, -2.8),
        ),
    )
    def test_merge_RX_rotations(self, params):
        ops = [qml.RX(angle, wires=0) for angle in params]

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.full_reduce(qs)

        assert len(new_qs.operations) == 3

        rot = new_qs.operations[1]
        assert new_qs.operations[0] == qml.H(0)
        assert isinstance(rot, qml.RZ)
        assert new_qs.operations[2] == qml.H(0)

        new_angle = rot.parameters[0]
        exp_angle = np.mod(np.sum(params), 2 * np.pi)

        assert np.isclose(new_angle, exp_angle)

    @pytest.mark.parametrize(
        "params",
        (
            (1.7, 0.0),
            (3.1, -0.5),
            (0.1, 0.9, -2.8),
        ),
    )
    def test_merge_RY_rotations(self, params):
        ops = [qml.RY(angle, wires=0) for angle in params]

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.full_reduce(qs)

        assert len(new_qs.operations) == 5

        rot = new_qs.operations[2]
        assert new_qs.operations[0] == qml.S(0)
        assert new_qs.operations[1] == qml.H(0)
        assert isinstance(rot, qml.RZ)
        assert new_qs.operations[3] == qml.H(0)
        assert new_qs.operations[4] == qml.adjoint(qml.S(0))

        new_angle = rot.parameters[0]
        exp_angle = np.mod(np.sum(params), 2 * np.pi)

        assert np.isclose(new_angle, 2 * np.pi - exp_angle)

    @pytest.mark.parametrize(
        "params",
        (
            (1.7, 0.0),
            (3.1, -0.5),
            (0.1, 0.9, -2.8),
        ),
    )
    def test_merge_RZ_rotations(self, params):
        ops = [qml.RZ(angle, wires=0) for angle in params]

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.full_reduce(qs)

        assert len(new_qs.operations) == 1

        rot = new_qs.operations[0]
        assert isinstance(rot, qml.RZ)

        new_angle = rot.parameters[0]
        exp_angle = np.mod(np.sum(params), 2 * np.pi)

        assert np.isclose(new_angle, exp_angle)

    @pytest.mark.parametrize(
        "angle, expected_ops",
        (
            (0, []),
            (0.25 * np.pi, [qml.T(0)]),
            (0.5 * np.pi, [qml.S(0)]),
            (0.75 * np.pi, [qml.Z(0), qml.adjoint(qml.T(0))]),
            (np.pi, [qml.Z(0)]),
            (1.25 * np.pi, [qml.Z(0), qml.T(0)]),
            (1.5 * np.pi, [qml.adjoint(qml.S(0))]),
            (1.75 * np.pi, [qml.adjoint(qml.T(0))]),
            (2 * np.pi, []),
        ),
    )
    def test_RZ_rotation_with_Clifford_T_angle(self, angle, expected_ops):
        """Test that RZ rotation gates are transformed into the corresponding sequence of Clifford + T gates."""
        ops = [qml.RZ(angle, wires=0)]

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.full_reduce(qs)

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
        ops = [
            qml.CNOT(wires=[0, 1]),
            qml.T(wires=0),
            qml.CNOT(wires=[3, 2]),
            qml.T(wires=1),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=2),
            qml.RZ(0.5, wires=1),
            qml.CNOT(wires=[1, 2]),
            qml.T(wires=1),
            qml.CNOT(wires=[3, 2]),
            qml.T(wires=0),
            qml.CNOT(wires=[0, 1]),
        ]
        original_tape = qml.tape.QuantumScript(ops=ops, measurements=measurements)

        (transformed_tape,), _ = qml.transforms.zx.full_reduce(original_tape)

        expected_ops = [
            qml.S(wires=0),
            qml.CNOT(wires=[2, 3]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(2.070796326790258, wires=[1]),
            qml.CNOT(wires=[1, 3]),
            qml.T(wires=3),
            qml.CNOT(wires=[1, 3]),
            qml.CNOT(wires=[2, 3]),
            qml.CNOT(wires=[0, 1]),
        ]

        assert transformed_tape.operations == expected_ops
        assert transformed_tape.measurements == measurements

    @pytest.mark.parametrize(
        "params",
        (
            (0.0, 0.0),
            (1.7, 0.0),
            (0.0, -1.7),
            (-3.2, 2.2),
            (3.2, -2.2),
        ),
    )
    def test_equivalent_state(self, params):
        num_wires = 3
        device = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(device)
        def original_circ(x, y):
            for i in range(num_wires):
                qml.Hadamard(wires=i)
            qml.T(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=0)
            qml.RX(x, wires=1)
            qml.S(wires=2)
            qml.RX(y, wires=1)
            qml.CNOT(wires=[1, 2])
            return qml.state()

        reduced_circ = qml.transforms.zx.full_reduce(original_circ)

        state1 = original_circ(*params)
        state2 = reduced_circ(*params)

        # test that the states are equivalent up to a global phase
        check = qml.math.fidelity_statevector(state1, state2)
        assert np.isclose(check, 1)
