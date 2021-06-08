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

from pennylane.transforms import cnot_to_cz


class TestTranspilation:
    """Test that adjacent inverse gates are cancelled."""

    def test_cnot_to_cz_isolated(self):
        """Test that a single instance of CNOT is transformed."""

        def qfunc():
            qml.CNOT(wires=["a", "b"])

        transformed_qfunc = cnot_to_cz(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 3

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("b")

        assert ops[1].name == "CZ"
        assert ops[1].wires == Wires(["a", "b"])

        assert ops[2].name == "Hadamard"
        assert ops[2].wires == Wires("b")

    def test_cnot_to_cz_with_others(self):
        """Test that CNOT is transformed in a circuit with multiple operations"""

        def qfunc():
            qml.Hadamard(wires="b")
            qml.CNOT(wires=["c", "a"])
            qml.RX(0.3, wires="a")

        transformed_qfunc = cnot_to_cz(qfunc)

        ops = qml.transforms.make_tape(transformed_qfunc)().operations

        assert len(ops) == 5

        assert ops[0].name == "Hadamard"
        assert ops[0].wires == Wires("b")

        assert ops[1].name == "Hadamard"
        assert ops[1].wires == Wires("a")

        assert ops[2].name == "CZ"
        assert ops[2].wires == Wires(["c", "a"])

        assert ops[3].name == "Hadamard"
        assert ops[3].wires == Wires("a")

        assert ops[4].name == "RX"
        assert ops[4].wires == Wires("a")
