# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the MultiX template."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.wires import Wires


def test_input_arguments_parsed_correctly():
    """Tests that MultiX handles and sanitizes its input arguments correctly."""
    bitstring_input = [0, 1, 1, 0]
    wires_input = ("a", "b", "c", "d")
    multix_op = qp.MultiX(bitstring_input, wires=wires_input)

    assert isinstance(multix_op.bitstring, np.ndarray)
    assert np.all(multix_op.bitstring == np.array(bitstring_input))
    assert multix_op.wires == Wires(wires_input)
    assert multix_op.dynamic_args == {"bitstring": multix_op.bitstring}
    assert multix_op.wire_args == {"wires": Wires(wires_input)}
    assert multix_op.num_wires == len(wires_input)


@pytest.mark.parametrize(
    ("bitstring", "wires", "error_match"),
    [
        ([0, 1, 1, 0], ["a", "b", "c"], "length"),
        ([0, 1, 2], ["a", "b", "c"], "binary"),
        ([[0, 1, 0]], ["a", "b", "c"], "dimension"),
        ([0], [], "wire"),
    ],
)
def test_invalid_arguments(bitstring, wires, error_match):
    """Tests that MultiX raises clear errors when input arguments are invalid."""
    with pytest.raises(ValueError, match=error_match):
        qp.MultiX(bitstring, wires=wires)


@pytest.mark.parametrize(
    ("bitstring", "expected_matrix"),
    [
        ([0], np.eye(2)),
        ([1], qp.X.compute_matrix()),
        ([1, 0], np.kron(qp.X.compute_matrix(), np.eye(2))),
        ([0, 1, 1], np.kron(np.eye(2), np.kron(qp.X.compute_matrix(), qp.X.compute_matrix()))),
    ],
)
def test_matrix(bitstring, expected_matrix):
    """Tests that MultiX computes the tensor product selected by the bitstring."""
    op = qp.MultiX(bitstring, wires=range(len(bitstring)))

    assert np.allclose(op.matrix(), expected_matrix)


@pytest.mark.parametrize(
    ("bitstring", "wires", "expected_index"),
    [
        ([0], [0], 0),
        ([1], [0], 1),
        ([1, 0], [0, 1], 2),
        ([0, 1, 1], [0, 1, 2], 3),
        ([1, 0, 1, 1], [0, 1, 2, 3], 11),
    ],
)
def test_evalutation(bitstring, wires, expected_index):
    """Tests that MultiX works correctly on 'default.qubit' device and |0...0> input state."""

    dev = qp.device("default.qubit")

    @qp.qnode(dev)
    def circuit():
        qp.MultiX(bitstring, wires=wires)
        return qp.probs(wires=wires)

    # qp.MultiX( bitstring, ... ) should set |0...0> to |bitstring>
    expected_result = np.zeros(2 ** len(wires))
    expected_result[expected_index] = 1

    obtained_result = circuit()

    assert np.allclose(obtained_result, expected_result)


@pytest.mark.parametrize(
    ("bitstring", "wires"),
    [
        ([0], [0]),
        ([1], [0]),
        ([1, 0], [0, 1]),
        ([0, 1, 1], [0, 1, 2]),
        ([1, 0, 1, 1], [0, 1, 2, 3]),
    ],
)
def test_decomposition(bitstring, wires):
    """Tests that MultiX decomposition contains X gates at the locations in the bitstring marked by 1."""

    multix_op = qp.MultiX(bitstring, wires=wires)
    assert multix_op.has_decomposition
    decomp = multix_op.decomposition()
    assert len(decomp) == sum(bitstring)  # each bit contributes one X gate

    # checking that the decomposed PauliX gates have the correct wire indices
    decomp_idx = 0
    for i, bit in enumerate(bitstring):
        if bit == 1:
            qp.assert_equal(decomp[decomp_idx], qp.X(wires[i]))
            decomp_idx += 1
