# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests of the Fast Fermionic Fourier Transform (FFFT)."""
import pytest

from pennylane import PauliZ, device, list_decomps, math, probs, qnode, workflow
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates.subroutines.ffft import FFFT, TwoQubitFermionicFourierTransform
from pennylane.wires import Wires


dev = device("default.qubit")


@qnode(dev)
def ffft(wires):
    FFFT(wires)
    return probs(wires)


@pytest.mark.parametrize("wires", [(0, 1), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5, 6, 7)])
def test_ffft_decomposition_new(wires):
    op = FFFT(wires)

    for rule in list_decomps(FFFT):
        _test_decomposition_rule(op, rule)


@pytest.mark.parametrize(
    "wires, error_type, error_msg",
    [
        (tuple(), ValueError, "The number of wires must be at least 2"),
        ((0, 1, 2), NotImplementedError, "powers of two"),
        ((0, 1, 2, 3, 4, 5), NotImplementedError, "powers of two"),
    ],
)
def test_raises(wires, error_type, error_msg):
    with pytest.raises(error_type, match=error_msg):
        FFFT(wires)


@pytest.mark.jax
@pytest.mark.parametrize(
    "wires, expected_circuit",
    [
        ((0, 1), [TwoQubitFermionicFourierTransform(Wires([0, 1]))]),
        (
            (0, 1, 2, 3),
            [
                TwoQubitFermionicFourierTransform(wires=[0, 1]),
                TwoQubitFermionicFourierTransform(wires=[2, 3]),
                PauliZ(2) ** 0.0,
                PauliZ(3) ** 0.5,
                TwoQubitFermionicFourierTransform(wires=[0, 2]),
                TwoQubitFermionicFourierTransform(wires=[1, 3]),
            ],
        ),
        (
            (0, 1, 2, 3, 4, 5, 6, 7),
            [
                TwoQubitFermionicFourierTransform(wires=[0, 1]),
                TwoQubitFermionicFourierTransform(wires=[2, 3]),
                PauliZ(2) ** 0.0,
                PauliZ(3) ** 0.5,
                TwoQubitFermionicFourierTransform(wires=[0, 2]),
                TwoQubitFermionicFourierTransform(wires=[1, 3]),
                TwoQubitFermionicFourierTransform(wires=[4, 5]),
                TwoQubitFermionicFourierTransform(wires=[6, 7]),
                PauliZ(6) ** 0.0,
                PauliZ(7) ** 0.5,
                TwoQubitFermionicFourierTransform(wires=[4, 6]),
                TwoQubitFermionicFourierTransform(wires=[5, 7]),
                PauliZ(4) ** 0.0,
                PauliZ(5) ** 0.25,
                PauliZ(6) ** 0.5,
                PauliZ(7) ** 0.75,
                TwoQubitFermionicFourierTransform(wires=[0, 4]),
                TwoQubitFermionicFourierTransform(wires=[1, 5]),
                TwoQubitFermionicFourierTransform(wires=[2, 6]),
                TwoQubitFermionicFourierTransform(wires=[3, 7]),
            ],
        ),
    ],
)
def test_ffft_circuit(wires, expected_circuit):
    tape = workflow.construct_tape(ffft, level="device")(wires)
    assert tape.operations == expected_circuit
