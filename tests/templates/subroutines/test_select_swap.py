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
"""
Tests for the SelectSwap template.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.decomposition import DecompositionGraph
from pennylane.testing import assert_decomposable

BITSTRINGS = (
    (0, 0, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize(
    "work_wires",
    [
        [],  # only loads one bitstring at a time
        [6, 7, 8],  # Select-SWAP without a work wire for the CSWAPs
        [6, 7, 8, 9, 10, 11],  # Select-SWAP with a work wire for the CSWAPs
    ],
)
def test_assert_valid_select_swap(work_wires):
    """Run standard validity tests."""
    op = qp.SelectSwap(BITSTRINGS, [0, 1, 2], [3, 4, 5], work_wires)
    qp.testing.assert_valid(op, skip_differentiation=True)


@pytest.mark.parametrize(
    ("num_bitstrings", "num_targets", "num_work_wires"),
    [(8, 3, 0), (8, 3, 3), (8, 3, 6), (5, 2, 3), (32, 1, 5)],
)
def test_load_and_unload(num_bitstrings, num_targets, num_work_wires, seed):
    """Test that SelectSwap loads the right bitstring into the target register and that its
    adjoint returns the target and work wires to the zero state."""
    bitstrings = np.random.default_rng(seed).integers(0, 2, size=(num_bitstrings, num_targets))
    num_control_wires = qp.math.ceil_log2(num_bitstrings)
    wires = qp.registers(
        {
            "control_wires": num_control_wires,
            "target_wires": num_targets,
            "work_wires": num_work_wires,
        }
    )
    dev = qp.device("default.qubit")

    @qp.set_shots(1)
    @qp.qnode(dev)
    def circuit(j, unload):
        qp.BasisState(qp.math.int_to_binary(j, num_control_wires), wires=wires["control_wires"])
        qp.SelectSwap(bitstrings, **wires)
        if unload:
            qp.adjoint(qp.SelectSwap(bitstrings, **wires))
        return qp.sample(wires=wires["target_wires"] + wires["work_wires"])

    for j in range(2**num_control_wires):
        expected = bitstrings[j] if j < num_bitstrings else [0] * num_targets
        assert np.allclose(circuit(j, False)[0][:num_targets], expected)
        assert np.allclose(circuit(j, True)[0], 0)


def test_decomposition():
    """Test that the decomposition loads four bitstrings in parallel and swaps the right one into
    the target register, without cleaning up afterwards."""
    bitstrings = [[(i // 3) % 2] for i in range(32)]
    decomposition = qp.SelectSwap(
        bitstrings,
        control_wires=[0, 1, 2, 3, 4],
        target_wires=[5],
        work_wires=[6, 7, 8, 9, 10],
    ).decomposition()

    # Four bitstrings are loaded in parallel, into the target wire and three work wires
    columns = [sum(bitstrings[4 * i : 4 * i + 4], []) for i in range(8)]
    qrom_load = qp.QROMLoad(
        columns, control_wires=[0, 1, 2], target_wires=[5, 6, 7, 8], work_wires=[9, 10]
    )
    cswaps = [
        qp.ctrl(qp.SWAP(wires), control=c, work_wires=[9], work_wire_type="zeroed")
        for c, wires in [(3, [6, 8]), (3, [5, 7]), (4, [5, 6])]
    ]
    expected_gates = [qrom_load] + cswaps

    assert len(decomposition) == len(expected_gates)
    for op1, op2 in zip(decomposition, expected_gates):
        qp.assert_equal(op1, op2)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize(
    ("num_bitstrings", "control_wires", "target_wires", "work_wires"),
    [
        (1, [], [0], [1]),
        (2, [0], [1], []),
        (3, [0, 1], [2], [3, 4]),
        (4, [2, 3], [0, 1], [4, 5, 6, 7, 8, 9]),
        (8, [0, 1, 2], [3, 4], [5]),
        (7, [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11]),
        (11, [0, 1, 2, 3], [4, 5], [6, 7]),
        (32, [0, 1, 2, 3, 4], [5], [6, 7, 8, 9, 10]),
    ],
)
def test_decomposition_new(num_bitstrings, control_wires, target_wires, work_wires, seed):
    """Tests the decomposition rule implemented with the new system."""
    rng = np.random.default_rng(seed)
    bitstrings = rng.integers(0, 2, size=(num_bitstrings, len(target_wires)))
    op = qp.SelectSwap(bitstrings, control_wires, target_wires, work_wires)
    assert_decomposable(op)


@pytest.mark.parametrize("sizes", [(32, 5, 1, 5), (64, 6, 1, 9), (64, 6, 2, 20), (256, 8, 2, 11)])
def test_t_count(sizes, seed):
    """Test that SelectSwap uses less than half as many T gates as QROMLoad, which has to clean
    up its work wires."""
    num_bitstrings, n_ctrl, n_target, n_work = sizes
    bitstrings = np.random.default_rng(seed).integers(0, 2, size=(num_bitstrings, n_target))
    wires = qp.registers({"control_wires": n_ctrl, "target_wires": n_target, "work_wires": n_work})
    # Weigh T gates heavily so that the graph minimizes the T count, which is what we compare
    gate_set = {"T": 100, "Adjoint(T)": 100, "H": 1, "S": 1, "Adjoint(S)": 1, "CNOT": 1, "X": 1}
    gate_set |= {"Z": 1, "CZ": 1, "GlobalPhase": 1, "MidMeasure": 1}

    def t_count(op):
        counts = DecompositionGraph([op], gate_set).solve().resource_estimate(op).gate_counts
        return sum(num for gate, num in counts.items() if str(gate) in {"T", "Adjoint(T)"})

    select_swap_t_count = t_count(qp.SelectSwap(bitstrings, **wires))
    assert 0 < 2 * select_swap_t_count < t_count(qp.QROMLoad(bitstrings, **wires))
