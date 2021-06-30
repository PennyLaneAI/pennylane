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

from pennylane.transforms.optimization import commute_x_behind_targets


class TestCommuteXBehindTargets:
    """Test that X rotations are properly pushed behind targets of X-based controlled operations."""

    def test_single_x_after_cnot_gate(self):
        """Test that a single X after a CNOT is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 2])
            qml.PauliX(wires=2)

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "PauliX"
        assert ops[1].wires == Wires(2)

        assert ops[2].name == "CNOT"
        assert ops[2].wires == Wires([0, 2])

    def test_multiple_x_after_cnot_gate(self):
        """Test that multiple X rotations after a CNOT both get pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CNOT(wires=["b", "a"])
            qml.RX(0.2, wires="a")
            qml.PauliX(wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "PauliX"
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "CNOT"
        assert ops[3].wires == Wires(["b", "a"])

    def test_single_x_after_crx_gate(self):
        """Test that a single X rotation after a CRX is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRX(0.1, wires=[0, "a"])
            qml.RX(0.2, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "CRX"
        assert ops[2].parameters[0] == 0.1
        assert ops[2].wires == Wires([0, "a"])

    def test_multiple_x_after_crx_gate(self):
        """Test that multiple X rotations after a CRX are pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.CRX(0.3, wires=["b", "a"])
            qml.PauliX(wires="a")
            qml.RX(0.1, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "PauliX"
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "RX"
        assert ops[2].parameters[0] == 0.1
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "CRX"
        assert ops[3].parameters[0] == 0.3
        assert ops[3].wires == Wires(["b", "a"])

    def test_single_x_after_toffoli_gate(self):
        """Test that a single X rotation after a Toffoli is pushed behind."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.Toffoli(wires=[0, 3, "a"])
            qml.RX(0.2, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.2
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "Toffoli"
        assert ops[2].wires == Wires([0, 3, "a"])

    def test_multiple_x_after_toffoli_gate(self):
        """Test that multiple X rotations after a Toffoli are pushed behind."""

        def qfunc():
            qml.Hadamard(wires="a")
            qml.Toffoli(wires=["b", "c", "a"])
            qml.RX(0.1, wires="a")
            qml.PauliX(wires="b")
            qml.RX(0.2, wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 5

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("a")

        assert ops[1].name == "RX"
        assert ops[1].parameters[0] == 0.1
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "RX"
        assert ops[2].parameters[0] == 0.2
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "Toffoli"
        assert ops[3].wires == Wires(["b", "c", "a"])

        assert ops[4].name == "PauliX"
        assert ops[4].wires == Wires(["b"])

    def test_no_commuting_gates_after_crx(self):
        """Test that pushing commuting X gates behind targets is properly 'blocked'."""

        def qfunc():
            qml.Hadamard(wires=0)
            qml.CRX(0.1, wires=[0, "a"])
            # The Hadamard blocks the CRX from moving ahead of the PauliX
            qml.Hadamard(wires="a")
            qml.PauliX(wires="a")

        transformed_qfunc = commute_x_behind_targets(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 4

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires(0)

        assert ops[1].name == "CRX"
        assert ops[1].parameters[0] == 0.1
        assert ops[1].wires == Wires([0, "a"])

        assert ops[2].name == "Hadamard"
        assert ops[2].wires == Wires("a")

        assert ops[3].name == "PauliX"
        assert ops[3].wires == Wires("a")
