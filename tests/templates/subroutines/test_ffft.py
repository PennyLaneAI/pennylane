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

import numpy as np
import pytest

from pennylane import FermionicSWAP, PauliZ, device, list_decomps, qnode, workflow
from pennylane.measurements import sample, state
from pennylane.ops.functions.assert_valid import _test_decomposition_rule
from pennylane.templates import BasisEmbedding
from pennylane.templates.subroutines.ffft import FFFT, TwoWireFFT
from pennylane.wires import Wires

dev = device("default.qubit")


@qnode(dev)
def ffft(wires, input=None):
    if input is not None:
        BasisEmbedding(input, wires)
    FFFT(wires)
    return state()


@pytest.mark.parametrize(
    "wires",
    [
        (0, 1),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    ],
)
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


@pytest.mark.capture
@pytest.mark.parametrize(
    "wires, expected_circuit, parallel",
    [
        ((0, 1), [TwoWireFFT(Wires([0, 1]))], False),
        (
            (0, 1, 2, 3),
            [
                TwoWireFFT(wires=[0, 1]),
                TwoWireFFT(wires=[2, 3]),
                PauliZ(2) ** 0.0,
                PauliZ(3) ** 0.5,
                FermionicSWAP(np.pi, wires=[2, 1]),
                TwoWireFFT(wires=[0, 1]),
                FermionicSWAP(np.pi, wires=[1, 2]),
                FermionicSWAP(np.pi, wires=[3, 2]),
                TwoWireFFT(wires=[1, 2]),
                FermionicSWAP(np.pi, wires=[2, 3]),
            ],
            False,
        ),
        (
            (0, 1, 2, 3, 4, 5, 6, 7),
            [
                TwoWireFFT(wires=[0, 1]),
                TwoWireFFT(wires=[2, 3]),
                PauliZ(2) ** 0.0,
                PauliZ(3) ** 0.5,
                FermionicSWAP(np.pi, wires=[2, 1]),
                TwoWireFFT(wires=[0, 1]),
                FermionicSWAP(np.pi, wires=[1, 2]),
                FermionicSWAP(np.pi, wires=[3, 2]),
                TwoWireFFT(wires=[1, 2]),
                FermionicSWAP(np.pi, wires=[2, 3]),
                TwoWireFFT(wires=[4, 5]),
                TwoWireFFT(wires=[6, 7]),
                PauliZ(6) ** 0.0,
                PauliZ(7) ** 0.5,
                FermionicSWAP(np.pi, wires=[6, 5]),
                TwoWireFFT(wires=[4, 5]),
                FermionicSWAP(np.pi, wires=[5, 6]),
                FermionicSWAP(np.pi, wires=[7, 6]),
                TwoWireFFT(wires=[5, 6]),
                FermionicSWAP(np.pi, wires=[6, 7]),
                PauliZ(4) ** 0.0,
                PauliZ(5) ** 0.25,
                PauliZ(6) ** 0.5,
                PauliZ(7) ** 0.75,
                FermionicSWAP(np.pi, wires=[4, 3]),
                FermionicSWAP(np.pi, wires=[3, 2]),
                FermionicSWAP(np.pi, wires=[2, 1]),
                TwoWireFFT(wires=[0, 1]),
                FermionicSWAP(np.pi, wires=[1, 2]),
                FermionicSWAP(np.pi, wires=[2, 3]),
                FermionicSWAP(np.pi, wires=[3, 4]),
                FermionicSWAP(np.pi, wires=[5, 4]),
                FermionicSWAP(np.pi, wires=[4, 3]),
                FermionicSWAP(np.pi, wires=[3, 2]),
                TwoWireFFT(wires=[1, 2]),
                FermionicSWAP(np.pi, wires=[2, 3]),
                FermionicSWAP(np.pi, wires=[3, 4]),
                FermionicSWAP(np.pi, wires=[4, 5]),
                FermionicSWAP(np.pi, wires=[6, 5]),
                FermionicSWAP(np.pi, wires=[5, 4]),
                FermionicSWAP(np.pi, wires=[4, 3]),
                TwoWireFFT(wires=[2, 3]),
                FermionicSWAP(np.pi, wires=[3, 4]),
                FermionicSWAP(np.pi, wires=[4, 5]),
                FermionicSWAP(np.pi, wires=[5, 6]),
                FermionicSWAP(np.pi, wires=[7, 6]),
                FermionicSWAP(np.pi, wires=[6, 5]),
                FermionicSWAP(np.pi, wires=[5, 4]),
                TwoWireFFT(wires=[3, 4]),
                FermionicSWAP(np.pi, wires=[4, 5]),
                FermionicSWAP(np.pi, wires=[5, 6]),
                FermionicSWAP(np.pi, wires=[6, 7]),
            ],
            False,
        ),
    ],
)
def test_ffft_circuit(wires, expected_circuit, parallel):
    @qnode(device("default.qubit", wires=wires), shots=1)
    def ffft(wires, parallel):  # pylint: disable=redefined-outer-name
        FFFT(wires, parallel)
        return sample(wires=wires)

    tape = workflow.construct_tape(ffft, level="device")(wires, parallel)
    for i, op in enumerate(tape.operations):
        assert type(op) == type(expected_circuit[i])
        assert op.hyperparameters == expected_circuit[i].hyperparameters
        assert op.data == expected_circuit[i].data
        assert op.wires == expected_circuit[i].wires


def fermionic_superposition_state(amplitudes):
    n = len(amplitudes)
    st = np.zeros(1 << n)
    for m in range(n):
        st[1 << (n - m - 1)] = amplitudes[m]
    return st / np.linalg.norm(st)


@pytest.mark.parametrize(
    "amplitudes",
    [
        (1, 0),
        (0, 1),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ],
)
def test_ffft_correct(amplitudes):
    n = len(amplitudes)

    def ft(k, m):
        return sum(np.exp(-2j * (k / m) * np.pi) ** j * amplitudes[j] for j in range(m)) / np.sqrt(
            m
        )

    modes = [ft(k, n) for k in range(n)]

    initial = fermionic_superposition_state(amplitudes)
    expected = fermionic_superposition_state(modes)

    result = ffft(list(range(n**2)), initial, False)

    for elem in np.argwhere(expected):
        assert elem in np.argwhere(result)

    result_parallel = ffft(list(range(n**2)), initial, True)

    for elem in np.argwhere(expected):
        assert elem in np.argwhere(result_parallel)
