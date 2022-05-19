# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the functions of the structure module.
"""
import pytest
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
from pennylane.templates.subroutines import UCCSD


@pytest.mark.parametrize(
    (
        "electrons",
        "orbitals",
        "delta_sz",
        "n_singles",
        "n_doubles",
        "singles_exp",
        "doubles_exp",
    ),
    [
        (1, 5, 0, 2, 0, [[0, 2], [0, 4]], []),
        (1, 5, 1, 0, 0, [], []),
        (1, 5, -1, 2, 0, [[0, 1], [0, 3]], []),
        (2, 5, 0, 3, 2, [[0, 2], [0, 4], [1, 3]], [[0, 1, 2, 3], [0, 1, 3, 4]]),
        (2, 5, 1, 2, 1, [[1, 2], [1, 4]], [[0, 1, 2, 4]]),
        (2, 5, -1, 1, 0, [[0, 3]], []),
        (2, 5, 2, 0, 0, [], []),
        (3, 6, 1, 1, 0, [[1, 4]], []),
        (
            3,
            6,
            -1,
            4,
            4,
            [[0, 3], [0, 5], [2, 3], [2, 5]],
            [[0, 1, 3, 5], [0, 2, 3, 4], [0, 2, 4, 5], [1, 2, 3, 5]],
        ),
        (3, 6, -2, 0, 1, [], [[0, 2, 3, 5]]),
        (3, 4, 0, 1, 0, [[1, 3]], []),
        (3, 4, 1, 0, 0, [], []),
        (3, 4, -1, 2, 0, [[0, 3], [2, 3]], []),
        (3, 4, 2, 0, 0, [], []),
    ],
)
def test_excitations(electrons, orbitals, delta_sz, n_singles, n_doubles, singles_exp, doubles_exp):
    r"""Test the correctness of the generated configurations"""

    singles, doubles = qchem.excitations(electrons, orbitals, delta_sz)

    assert len(singles) == len(singles_exp)
    assert len(doubles) == len(doubles_exp)
    assert singles == singles_exp
    assert doubles == doubles_exp


@pytest.mark.parametrize(
    ("electrons", "orbitals", "delta_sz", "message_match"),
    [
        (0, 4, 0, "number of active electrons has to be greater than 0"),
        (3, 2, 0, "has to be greater than the number of active electrons"),
        (2, 4, 3, "Expected values for 'delta_sz'"),
        (2, 4, 1.5, "Expected values for 'delta_sz'"),
    ],
)
def test_inconsistent_excitations(electrons, orbitals, delta_sz, message_match):
    r"""Test that an error is raised if a set of inconsistent arguments is input"""

    with pytest.raises(ValueError, match=message_match):
        qchem.excitations(electrons, orbitals, delta_sz)


@pytest.mark.parametrize(
    ("singles", "doubles", "wires", "singles_wires_exp", "doubles_wires_exp"),
    [
        ([[0, 2]], [], None, [[0, 1, 2]], []),
        ([], [[0, 1, 2, 3]], None, [], [[[0, 1], [2, 3]]]),
        ([[0, 1]], [[0, 1, 2, 4]], None, [[0, 1]], [[[0, 1], [2, 3, 4]]]),
        (
            [[0, 1], [2, 4]],
            [[0, 1, 2, 3], [0, 2, 4, 6]],
            None,
            [[0, 1], [2, 3, 4]],
            [[[0, 1], [2, 3]], [[0, 1, 2], [4, 5, 6]]],
        ),
        (
            [[0, 1], [2, 4]],
            [[0, 1, 2, 3], [0, 2, 4, 6]],
            ["a0", "b1", "c2", "d3", "e4", "f5", "g6"],
            [["a0", "b1"], ["c2", "d3", "e4"]],
            [[["a0", "b1"], ["c2", "d3"]], [["a0", "b1", "c2"], ["e4", "f5", "g6"]]],
        ),
    ],
)
def test_mapping_from_excitations_to_wires(
    singles, doubles, wires, singles_wires_exp, doubles_wires_exp
):
    r"""Test the correctness of the mapping between indices of the single and double
    excitations and the list of wires to be passed to the quantum circuit"""

    singles_wires, doubles_wires = qchem.excitations_to_wires(singles, doubles, wires=wires)

    assert len(singles_wires) == len(singles_wires_exp)
    assert len(doubles_wires) == len(doubles_wires_exp)
    assert singles_wires == singles_wires_exp
    assert doubles_wires == doubles_wires_exp


@pytest.mark.parametrize(
    ("singles", "doubles", "wires", "message_match"),
    [
        ([], [], None, "'singles' and 'doubles' lists can not be both empty"),
        ([[0, 2, 3]], [], None, "Expected entries of 'singles' to be of shape"),
        ([[0, 2], [3]], [], None, "Expected entries of 'singles' to be of shape"),
        ([], [[0, 1, 2, 3], [1, 3]], None, "Expected entries of 'doubles' to be of shape"),
        ([], [[0, 1, 2, 3], [1, 3, 4, 5, 6]], None, "Expected entries of 'doubles' to be of shape"),
        (
            [[0, 2]],
            [[0, 1, 2, 3], [0, 2, 4, 6]],
            ["a0", "b1", "c2", "d3", "e4", "f5"],
            "Expected number of wires is",
        ),
    ],
)
def test_excitations_to_wires_exceptions(singles, doubles, wires, message_match):
    r"""Test that the function 'excitations_to_wires()' throws an exception if ``singles``,
    ``doubles`` or ``wires`` parameter has illegal shapes or size"""

    with pytest.raises(ValueError, match=message_match):
        qchem.excitations_to_wires(singles, doubles, wires=wires)


@pytest.mark.parametrize(
    ("weights", "singles", "doubles", "expected"),
    [
        (
            np.array([3.90575761, -1.89772083, -1.36689032]),
            [[0, 2], [1, 3]],
            [[0, 1, 2, 3]],
            [-0.14619406, -0.06502792, 0.14619406, 0.06502792],
        )
    ],
)
def test_integration_with_uccsd(weights, singles, doubles, expected):
    """Test integration with the UCCSD template"""

    s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
    N = 4
    wires = range(N)
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circuit(weights):
        UCCSD(weights, wires, s_wires=s_wires, d_wires=d_wires, init_state=np.array([1, 1, 0, 0]))
        return [qml.expval(qml.PauliZ(w)) for w in range(N)]

    res = circuit(weights)
    assert np.allclose(res, np.array(expected))
