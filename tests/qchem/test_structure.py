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
import os
import pytest
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
from pennylane.templates.subroutines import UCCSD


ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")


def test_reading_xyz_file(tmpdir):
    r"""Test reading of the generated file 'structure.xyz'"""

    ref_symbols = ["C", "C", "N", "H", "H", "H", "H", "H"]
    ref_coords = np.array(
        [
            0.68219113,
            -0.85415621,
            -1.04123909,
            -1.34926445,
            0.23621577,
            0.61794044,
            1.29068294,
            0.25133357,
            1.40784596,
            0.83525895,
            -2.88939124,
            -1.16974047,
            1.26989596,
            0.19275206,
            -2.69852891,
            -2.57758643,
            -1.05824663,
            1.61949529,
            -2.17129532,
            2.04090421,
            0.11338357,
            2.06547065,
            2.00877887,
            1.20186582,
        ]
    )
    name = os.path.join(ref_dir, "gdb3.mol5.XYZ")
    symbols, coordinates = qchem.read_structure(name, outpath=tmpdir)

    assert symbols == ref_symbols
    assert np.allclose(coordinates, ref_coords)


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
            [0.14619406, 0.06502792, -0.14619406, -0.06502792],
        )
    ],
)
def test_excitation_integration_with_uccsd(weights, singles, doubles, expected):
    """Test integration with the UCCSD template"""

    s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)
    n = 4
    wires = range(n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(weights):
        UCCSD(weights, wires, s_wires=s_wires, d_wires=d_wires, init_state=np.array([1, 1, 0, 0]))
        return [qml.expval(qml.PauliZ(w)) for w in range(n)]

    res = circuit(weights)
    assert np.allclose(res, np.array(expected))


@pytest.mark.parametrize(
    ("electrons", "orbitals", "exp_state"),
    [
        (2, 5, np.array([1, 1, 0, 0, 0])),
        (1, 5, np.array([1, 0, 0, 0, 0])),
        (5, 5, np.array([1, 1, 1, 1, 1])),
    ],
)
def test_hf_state(electrons, orbitals, exp_state):
    r"""Test the correctness of the generated occupation-number vector"""

    res_state = qchem.hf_state(electrons, orbitals)

    assert len(res_state) == len(exp_state)
    assert np.allclose(res_state, exp_state)


@pytest.mark.parametrize(
    ("electrons", "orbitals", "msg_match"),
    [
        (0, 5, "number of active electrons has to be larger than zero"),
        (-1, 5, "number of active electrons has to be larger than zero"),
        (6, 5, "number of active orbitals cannot be smaller than the number of active"),
    ],
)
def test__hf_state_inconsistent_input(electrons, orbitals, msg_match):
    r"""Test that an error is raised if a set of inconsistent arguments is input"""

    with pytest.raises(ValueError, match=msg_match):
        qchem.hf_state(electrons, orbitals)


@pytest.mark.parametrize(
    (
        "n_electrons",
        "n_orbitals",
        "multiplicity",
        "act_electrons",
        "act_orbitals",
        "core_ref",
        "active_ref",
    ),
    [
        (4, 6, 1, None, None, [], list(range(6))),
        (4, 6, 1, 4, None, [], list(range(6))),
        (4, 6, 1, 2, None, [0], list(range(1, 6))),
        (4, 6, 1, None, 4, [], list(range(4))),
        (4, 6, 1, 2, 3, [0], list(range(1, 4))),
        (5, 6, 2, 3, 4, [0], list(range(1, 5))),
        (5, 6, 2, 1, 4, [0, 1], list(range(2, 6))),
    ],
)
def test_active_spaces(
    n_electrons, n_orbitals, multiplicity, act_electrons, act_orbitals, core_ref, active_ref
):
    r"""Test the correctness of the generated active spaces"""

    core, active = qchem.active_space(
        n_electrons,
        n_orbitals,
        mult=multiplicity,
        active_electrons=act_electrons,
        active_orbitals=act_orbitals,
    )

    assert core == core_ref
    assert active == active_ref


@pytest.mark.parametrize(
    ("n_electrons", "n_orbitals", "multiplicity", "act_electrons", "act_orbitals", "message_match"),
    [
        (4, 6, 1, 6, 5, "greater than the total number of electrons"),
        (4, 6, 1, 1, 5, "should be even"),
        (4, 6, 1, -1, 5, "has to be greater than 0."),
        (4, 6, 1, 2, 6, "greater than the total number of orbitals"),
        (4, 6, 1, 2, 1, "there are no virtual orbitals"),
        (5, 6, 2, 2, 5, "should be odd"),
        (5, 6, 2, 3, -2, "has to be greater than 0."),
        (5, 6, 2, 3, 6, "greater than the total number of orbitals"),
        (5, 6, 2, 3, 2, "there are no virtual orbitals"),
        (6, 6, 3, 1, 2, "greater than or equal to"),
    ],
)
def test_inconsistent_active_spaces(
    n_electrons, n_orbitals, multiplicity, act_electrons, act_orbitals, message_match
):
    r"""Test that an error is raised if an inconsistent active space is generated"""

    with pytest.raises(ValueError, match=message_match):
        qchem.active_space(
            n_electrons,
            n_orbitals,
            mult=multiplicity,
            active_electrons=act_electrons,
            active_orbitals=act_orbitals,
        )
