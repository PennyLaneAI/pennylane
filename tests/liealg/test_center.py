# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/dla/center.py functionality"""

import numpy as np
import pytest

import pennylane as qml
from pennylane import center
from pennylane.pauli import PauliSentence, PauliWord


def test_trivial_center():
    """Test that the center of an empty list of generators is an empty list of generators."""
    assert center([]) == []


DLA_CENTERS = (
    ([qml.I()], [qml.I()]),  # just the identity
    ([qml.I(), qml.X(0)], [qml.I(), qml.X(0)]),  # identity and some other operator
    ([qml.X(0), qml.X(1)], [qml.X(0), qml.X(1)]),  # two non-overlapping wires
    ([qml.X(0), qml.Y(1)], [qml.X(0), qml.Y(1)]),  # two non-overlapping wires with different ops
    ([qml.X(0), qml.Y(0), qml.Z(0), qml.I()], [qml.I()]),  # non-trivial DLA, but trivial center
    (
        [qml.X(0) + qml.Y(0), qml.X(0) - qml.Y(0), qml.Z(0)],
        [],
    ),  # non-trivial DLA, but trivial center
)


@pytest.mark.parametrize("ops, true_res", DLA_CENTERS)
def test_center(ops, true_res):
    """Test centers with Identity operators or non-overlapping wires"""
    assert center(ops) == true_res


@pytest.mark.parametrize("ops, true_res", DLA_CENTERS)
def test_center_pauli(ops, true_res):
    """Test centers with Identity operators or non-overlapping wires using their pauli_rep"""
    ops = [op.pauli_rep for op in ops]
    res = center(ops, pauli=True)

    assert all(isinstance(op, PauliSentence) for op in res)
    assert res == [op.pauli_rep for op in true_res]


@pytest.mark.parametrize("pauli", [False, True])
def test_center_pauli_word(pauli):
    """Test that PauliWord instances can be passed for both pauli=True/False"""
    words = [{0: "X"}, {0: "X", 1: "X"}, {1: "Y"}, {0: "X", 1: "Z"}]
    ops = list(map(PauliWord, words))
    if pauli:
        assert qml.center(ops, pauli=pauli) == [PauliWord({0: "X"})]
    else:
        assert qml.center(ops, pauli=pauli) == [qml.X(0)]


@pytest.mark.parametrize("pauli", [False, True])
def test_center_pauli_sentence(pauli):
    """Test that PauliSentence instances can be passed for both pauli=True/False"""
    words = [{0: "X"}, {0: "X", 1: "X"}, {1: "Y"}, {0: "X", 1: "Z"}]
    words = list(map(PauliWord, words))
    sentences = [
        {words[0]: 0.5, words[1]: 3.2},
        {words[0]: -0.2, words[2]: 2.5},
        {words[2]: 1.2, words[3]: 0.72, words[1]: 0.6},
        {words[1]: 0.9, words[2]: 1.8},
    ]
    sentences = list(map(PauliSentence, sentences))
    if pauli:
        cent = qml.center(sentences, pauli=pauli)
        assert isinstance(cent, list) and len(cent) == 1
        assert isinstance(cent[0], PauliSentence)
        assert PauliWord({0: "X"}) in cent[0]
    else:
        cent = qml.center(sentences, pauli=pauli)
        assert isinstance(cent, list) and len(cent) == 1
        assert isinstance(cent[0], qml.ops.op_math.SProd)
        assert cent[0].base == qml.X(0)


c = 1 / np.sqrt(2)

GENERATOR_CENTERS = (
    ([qml.X(0), qml.X(0) @ qml.X(1), qml.Y(1)], [qml.X(0)]),
    ([qml.X(0) @ qml.X(1), qml.Y(1), qml.X(0)], [qml.X(0)]),
    ([qml.X(0) @ qml.X(1), qml.Y(1), qml.X(1)], []),
    ([qml.X(0) @ qml.X(1), qml.Y(1), qml.Z(0)], []),
    ([p(0) @ p(1) for p in [qml.X, qml.Y, qml.Z]], [p(0) @ p(1) for p in [qml.X, qml.Y, qml.Z]]),
    ([qml.X(0), qml.X(1), sum(p(0) @ p(1) for p in [qml.Y, qml.Z])], [c * qml.X(0) + c * qml.X(1)]),
)


@pytest.mark.parametrize("generators, true_res", GENERATOR_CENTERS)
def test_center_dla(generators, true_res):
    """Test computing the center for a non-trivial DLA"""
    g = qml.lie_closure(generators)
    assert center(g) == true_res
