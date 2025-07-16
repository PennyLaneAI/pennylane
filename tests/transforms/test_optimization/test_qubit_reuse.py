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
Unit tests for the optimization transform ``undo_swaps``.
"""
import random
from collections import Counter

import pennylane as qml
from pennylane import queuing


class TestQubitReuse:
    """Test that check the main functionalities of the `qubit_reuse` transform"""

    def test_transform_circuit(self):
        """A simple test case."""
        dev = qml.device("default.qubit")

        # define the 5-qubit circuit
        #
        # 0: ──H─╭●──────────┤ < Z >
        # 1: ─╭●─│──╭●───────┤
        # 2: ─╰X─│──│──╭●────┤
        # 3: ────╰X─│──│──╭●─┤
        # 4: ───────╰X─╰X─╰X─┤

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 4])
            qml.CNOT(wires=[3, 4])
            return qml.expval(qml.PauliZ(0))

        # apply the transform to get the new 3-qubit circuit i.e.
        #
        # 0: ──H─╭X───────────╭●──┤↗│  │0⟩───────────╭●──┤↗│  │0⟩───────────┤
        # 1: ────╰●──┤↗│  │0⟩─╰X─╭●─────────┤↗│  │0⟩─│──╭●─────────┤↗│  │0⟩─┤
        # 2: ────────────────────╰X──────────────────╰X─╰X──────────────────┤

        random.seed(10)  # for test reproducibility
        new_circuit = qml.transforms.qubit_reuse(circuit)

        # execute the circuit
        with queuing.AnnotatedQueue() as q:
            new_circuit()

        # check that we now use less (3) wires
        found_wires = Counter()
        for op in q.queue:
            for wire in op.wires:
                found_wires[wire] += 1

        assert len(found_wires) == 3
