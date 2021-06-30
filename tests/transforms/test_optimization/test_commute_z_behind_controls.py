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

import pytest

import pennylane as qml
from pennylane.wires import Wires
from pennylane.transforms.optimization import commute_z_behind_controls


class TestCommuteZBehindControls:
    """Test that diagonal gates are properly pushed behind X-based target operations."""

    def test_single_z_after_cnot_gate(self):
        """Test that a single Z after a CNOT is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 2])
            qml.PauliZ(wires=0)

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "PauliZ"
        assert ops[1].wires == Wires(0)

        assert ops[2].name == "CNOT"
        assert ops[2].wires == Wires([0, 2])

    def test_multiple_z_after_cnot_gate(self):
        """Test that multiple Z rotations after a CNOT both get pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["b", "a"])
            qml.RZ(0.2, wires="b")
            qml.PauliZ(wires="b")

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "RZ"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("b")

        assert ops[2].name == "PauliZ"
        assert ops[2].wires == Wires("b")

        assert ops[3].name == "CNOT"
        assert ops[3].wires == Wires(["b", "a"])

    def test_single_z_after_cry_gate(self):
        """Test that a single Z rotation after a CRY is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRY(0.1, wires=[0, "a"])
            qml.RZ(0.2, wires=0)

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "RZ"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires(0)

        assert ops[2].name == "CRY"
        assert ops[2].parameters[0] == 0.1
        assert ops[2].wires == Wires([0, "a"])

    def test_multiple_z_after_crx_gate(self):
        """Test that multiple Z rotations after a CRX are pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CRX(0.3, wires=["b", "a"])
            qml.PhaseShift(0.2, wires="b")
            qml.T(wires="b")

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "PhaseShift"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("b")

        assert ops[2].name == "T"
        assert ops[2].wires == Wires("b")

        assert ops[3].name == "CRX"
        assert ops[3].parameters[0] == 0.3
        assert ops[3].wires == Wires(["b", "a"])

    def test_no_commuting_gates_after_crx(self):
        """Test that pushing commuting X gates behind targets is properly 'blocked'."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRX(0.1, wires=[0, "a"])
            # The Hadamard blocks the CRX from moving ahead of the PauliX
            qml.Hadamard(wires=0)
            qml.S(wires=0)

        transformed_qfunc = commute_z_behind_controls(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "CRX"
        assert ops[1].parameters[0] == 0.1
        assert ops[1].wires == Wires([0, "a"])

        assert ops[2].name == "Hadamard"
        assert ops[2].wires == Wires(0)

        assert ops[3].name == "S"
        assert ops[3].wires == Wires(0)
