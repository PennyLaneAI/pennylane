# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This module contains tests for the compact Hamiltonian dataclasses used in resource estimation.
"""
import pytest

import pennylane.estimator as qre
from pennylane.estimator import (
    CDFHamiltonian,
    THCHamiltonian,
    VibrationalHamiltonian,
    VibronicHamiltonian,
)
from pennylane.estimator.compact_hamiltonian import (
    _sort_and_freeze,
    _validate_pauli_terms,
)

# pylint: disable=too-many-arguments


class TestPauliHamiltonian:
    """Unit tests for the PauliHamiltonian class"""

    @pytest.mark.parametrize(
        "num_qubits, pauli_terms, one_norm, num_terms",
        (
            (5, {"XX": 10, "YY": 10, "Z": 5}, 0.01, 25),
            (10, {"X": 20, "YZY": 7, "ZXX": 3}, None, 30),
            (50, {"XXXXX": 10, "Z": 5}, 10.5, 15),
            (
                5,
                (
                    {"XX": 10, "X": 5},
                    {"ZZ": 10},
                ),
                0.01,
                25,
            ),
            (
                10,
                (
                    {"XX": 15, "X": 5},
                    {"ZZ": 10},
                    {"YY": 5, "X": 5},
                ),
                None,
                40,
            ),
            (
                50,
                ({"XXXXX": 10, "X": 3},),
                10.5,
                13,
            ),
        ),
    )
    def test_init(self, num_qubits, pauli_terms, one_norm, num_terms):
        """Test that we can instantiate a PauliHamiltonian"""
        pauli_ham = qre.PauliHamiltonian(
            num_qubits=num_qubits,
            pauli_terms=pauli_terms,
            one_norm=one_norm,
        )

        assert pauli_ham.num_qubits == num_qubits
        assert pauli_ham.pauli_terms == pauli_terms
        assert pauli_ham.one_norm == one_norm
        assert pauli_ham.num_terms == num_terms

    @pytest.mark.parametrize(
        "input_args, error_message",
        (
            (
                {
                    "num_qubits": 5,
                    "one_norm": -1,
                    "pauli_terms": {"X": 1, "Y": 2, "Z": 3},
                },
                "one_norm, if provided, must be a positive float or integer.",
            ),
            (
                {
                    "num_qubits": 5,
                    "one_norm": "one",
                    "pauli_terms": {"X": 1, "Y": 2, "Z": 3},
                },
                "one_norm, if provided, must be a positive float or integer.",
            ),
            (
                {
                    "num_qubits": 5,
                    "pauli_terms": ({"XX": "Wrong"}, {"YY": False, "ZZZ": 1.5}),
                },
                "The values represent frequencies and should be positive integers",
            ),
            (
                {
                    "num_qubits": 5,
                    "pauli_terms": {
                        "XX": "Wrong",
                        "YY": False,
                        "ZZZ": 1.5,
                    },
                },
                "The values represent frequencies and should be positive integers",
            ),
            (
                {
                    "num_qubits": 5,
                    "pauli_terms": {
                        "Wrong": 30,
                        False: 30,
                        5: 40,
                    },
                },
                "The keys represent Pauli words and should be strings",
            ),
            (
                {
                    "num_qubits": 5,
                    "pauli_terms": ({"XX": 30}, {False: 30, "Wrong": 40}),
                },
                "The keys represent Pauli words and should be strings",
            ),
        ),
    )
    def test_init_raises_error_from_incompatible_inputs(self, input_args, error_message):
        """Test that passing incompatible input arguments will raise the appropriate error"""
        with pytest.raises(ValueError, match=error_message):
            qre.PauliHamiltonian(**input_args)

    @pytest.mark.parametrize(
        "pauli_ham, expected_repr",
        (
            (
                qre.PauliHamiltonian(10, pauli_terms={"X": 20, "YZY": 7, "ZXX": 3}),
                "PauliHamiltonian(num_qubits=10, one_norm=None, pauli_terms={'X': 20, 'YZY': 7, 'ZXX': 3})",
            ),
            (
                qre.PauliHamiltonian(
                    15,
                    pauli_terms=(
                        {"XX": 15, "X": 5},
                        {"ZZ": 10},
                        {"YY": 5, "X": 5},
                    ),
                ),
                "PauliHamiltonian(num_qubits=15, one_norm=None, pauli_terms=({'XX': 15, 'X': 5}, {'ZZ': 10}, {'YY': 5, 'X': 5}))",
            ),
            (
                qre.PauliHamiltonian(10, pauli_terms={"X": 20, "YZY": 7, "ZXX": 3}, one_norm=0.01),
                "PauliHamiltonian(num_qubits=10, one_norm=0.01, pauli_terms={'X': 20, 'YZY': 7, 'ZXX': 3})",
            ),
            (
                qre.PauliHamiltonian(
                    15,
                    pauli_terms=(
                        {"XX": 15, "X": 5},
                        {"ZZ": 10},
                        {"YY": 5, "X": 5},
                    ),
                    one_norm=0.01,
                ),
                "PauliHamiltonian(num_qubits=15, one_norm=0.01, pauli_terms=({'XX': 15, 'X': 5}, {'ZZ': 10}, {'YY': 5, 'X': 5}))",
            ),
        ),
    )
    def test_repr(self, pauli_ham, expected_repr):
        """Test the repr dundar method of the PauliHamiltonian"""
        assert repr(pauli_ham) == expected_repr

    @pytest.mark.parametrize(
        "input_args",
        (
            {
                "num_qubits": 10,
                "one_norm": 10.5,
                "pauli_terms": {"XX": 5, "YY": 10, "Z": 1},
            },
            {
                "num_qubits": 10,
                "one_norm": 10.5,
                "pauli_terms": ({"XX": 5}, {"YY": 10}, {"Z": 1}),
            },
        ),
    )
    def test_hash_and_equality(self, input_args):
        """Test the hash method works as expected"""
        ph1 = qre.PauliHamiltonian(**input_args)
        ph2 = qre.PauliHamiltonian(**input_args)
        ph3 = qre.PauliHamiltonian(
            num_qubits=5, pauli_terms={"X": 5}
        )  # some other PauliHamiltonian

        assert ph1 == ph2
        assert ph1 != ph3
        assert hash(ph1) == hash(ph1)
        assert hash(ph1) == hash(ph2)

    def test_hash_and_equality_commuting_groups(self):
        """Test that hash and equality are dependant on
        the order of commuting groups in the tuple"""
        ph1 = qre.PauliHamiltonian(
            num_qubits=10,
            pauli_terms=({"XX": 5, "X": 1}, {"YY": 10}, {"Z": 1}),
        )
        ph2 = qre.PauliHamiltonian(
            num_qubits=10,
            pauli_terms=({"X": 1, "XX": 5}, {"YY": 10}, {"Z": 1}),
        )
        ph3 = qre.PauliHamiltonian(
            num_qubits=10,
            pauli_terms=({"Z": 1}, {"XX": 5, "X": 1}, {"YY": 10}),
        )

        assert ph1 == ph2
        assert hash(ph1) == hash(ph2)
        assert ph1 != ph3
        assert hash(ph1) != hash(ph3)

    def test_hash_and_equality_dict(self):
        """Test that hash and equality are independant of
        the order of terms in the pauli_terms dictionary"""
        ph1 = qre.PauliHamiltonian(
            num_qubits=10,
            pauli_terms={"XX": 5, "YY": 10, "Z": 1},
        )
        ph2 = qre.PauliHamiltonian(
            num_qubits=10,
            pauli_terms={"YY": 10, "Z": 1, "XX": 5},
        )

        assert ph1 == ph2
        assert hash(ph1) == hash(ph2)


@pytest.mark.parametrize(
    "pauli_terms, error_type, error_message",
    (
        (
            {"XX": 10, "YY": 5, "Z": 5, "XZYX": 103},
            None,
            None,
        ),
        (
            ("XX", "YY", "Z", "XZYX"),
            TypeError,
            "Expected `pauli_terms` to be a dictionary",
        ),
        (
            {"XX": 10, "YY": 5, "Z": 5, "XZYX": -103},
            ValueError,
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"XX": 10, "YY": 5, "Z": True, "XZYX": 103},
            ValueError,
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"XX": 10, "YY": "Five", "Z": 5, "XZYX": 103},
            ValueError,
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"XX": 10.0, "YY": 5.5, "Z": 5.1, "XZYX": 103.2},
            ValueError,
            "The values represent frequencies and should be positive integers",
        ),
        (
            {"IXXI": 10, "Wrong": 5, "Characters": 5, "XZYX": 103},
            ValueError,
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
        (
            {"xx": 10, "yy": 5, "z": 5, "XzyX": 103},
            ValueError,
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
        (
            {False: 10, "YY": 5, "Z": 5, "XZYX": 103},
            ValueError,
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
        (
            {"XX": 10, 15: 5, "Z": 5, "XZYX": 103},
            ValueError,
            "The keys represent Pauli words and should be strings containing either 'X','Y' or 'Z'",
        ),
    ),
)
def test_validate_pauli_terms(pauli_terms, error_type, error_message):
    """Test the private _validate_pauli_terms function"""
    if error_message is None:
        # No error message --> valid pauli distribution format
        assert _validate_pauli_terms(pauli_terms) is None
    else:
        with pytest.raises(error_type, match=error_message):
            _validate_pauli_terms(pauli_terms)


def test_sort_and_freeze():
    """Test the private sort and freeze function behaves as expected"""
    pauli_terms = {
        "XX": 30,
        "ZZ": 76,
        "X": 10,
        "YXY": 105,
        "YY": 15,
    }

    expected_result = (
        ("X", 10),
        ("XX", 30),
        ("YXY", 105),
        ("YY", 15),
        ("ZZ", 76),
    )
    assert _sort_and_freeze(pauli_terms) == expected_result


# Test that all of compact_hamiltonian classes are frozen
Test_Hamiltonians = [
    (CDFHamiltonian, "num_orbitals", {"num_orbitals": 10, "num_fragments": 30}),
    (THCHamiltonian, "num_orbitals", {"num_orbitals": 10, "tensor_rank": 30}),
    (VibrationalHamiltonian, "num_modes", {"num_modes": 5, "grid_size": 3, "taylor_degree": 2}),
    (
        VibronicHamiltonian,
        "num_modes",
        {"num_modes": 5, "num_states": 2, "grid_size": 3, "taylor_degree": 2},
    ),
]


@pytest.mark.parametrize("HamiltonianClass, attr_name, kwargs", Test_Hamiltonians)
def test_Hamiltonian_is_frozen(HamiltonianClass, attr_name, kwargs):
    """Verify that all Hamiltonian dataclasses are immutable (frozen=True)."""

    assert HamiltonianClass.__dataclass_params__.frozen is True

    hamiltonian = HamiltonianClass(**kwargs)

    with pytest.raises(AttributeError):
        setattr(hamiltonian, attr_name, 20)


@pytest.mark.parametrize(
    "num_orbitals, num_fragments, one_norm",
    [(4, 10, None), (16, 50, 50.0), (8, 20, 25)],
)
def test_cdf_instantiation(num_orbitals, num_fragments, one_norm):
    """Test successful instantiation with valid integer inputs."""
    hamiltonian = CDFHamiltonian(
        num_orbitals=num_orbitals, num_fragments=num_fragments, one_norm=one_norm
    )

    assert hamiltonian.num_orbitals == num_orbitals
    assert hamiltonian.num_fragments == num_fragments
    assert hamiltonian.one_norm == one_norm


@pytest.mark.parametrize(
    "invalid_num_orbitals, invalid_num_fragments, invalid_one_norm",
    [
        ("4", 10, None),
        (16, 4.5, None),
        (16, 50, "50.0"),
        (4.5, 10, None),
        (-4, 10, None),
        (16, -10, None),
        (16, 50, -5),
        (4, -5.5, None),
    ],
)
def test_cdf_invalid_types(invalid_num_orbitals, invalid_num_fragments, invalid_one_norm):
    """Test that TypeError is raised for invalid input types."""
    with pytest.raises(TypeError):
        CDFHamiltonian(
            num_orbitals=invalid_num_orbitals,
            num_fragments=invalid_num_fragments,
            one_norm=invalid_one_norm,
        )


@pytest.mark.parametrize(
    "num_orbitals, tensor_rank, one_norm",
    [(4, 10, None), (16, 50, 50.0), (8, 20, 25)],
)
def test_thc_instantiation(num_orbitals, tensor_rank, one_norm):
    """Test successful instantiation with valid integer inputs."""
    hamiltonian = THCHamiltonian(
        num_orbitals=num_orbitals, tensor_rank=tensor_rank, one_norm=one_norm
    )

    assert hamiltonian.num_orbitals == num_orbitals
    assert hamiltonian.tensor_rank == tensor_rank
    assert hamiltonian.one_norm == one_norm


@pytest.mark.parametrize(
    "invalid_num_orbitals, invalid_tensor_rank, invalid_one_norm",
    [
        ("4", 10, None),
        (16, 4.5, None),
        (16, 50, "50.0"),
        (4.5, 10, None),
        (-4, 10, None),
        (16, -10, None),
        (16, 50, -5),
        (4, -5.5, None),
    ],
)
def test_thc_invalid_types(invalid_num_orbitals, invalid_tensor_rank, invalid_one_norm):
    """Test that TypeError is raised for invalid input types."""
    with pytest.raises(TypeError):
        THCHamiltonian(
            num_orbitals=invalid_num_orbitals,
            tensor_rank=invalid_tensor_rank,
            one_norm=invalid_one_norm,
        )


@pytest.mark.parametrize(
    "num_modes, grid_size, taylor_degree, one_norm",
    [(4, 10, 3, None), (16, 50, 2, 50.0), (8, 20, 4, 25)],
)
def test_vibrational_instantiation(num_modes, grid_size, taylor_degree, one_norm):
    """Test successful instantiation with valid integer inputs."""
    hamiltonian = VibrationalHamiltonian(
        num_modes=num_modes, grid_size=grid_size, taylor_degree=taylor_degree, one_norm=one_norm
    )

    assert hamiltonian.num_modes == num_modes
    assert hamiltonian.grid_size == grid_size
    assert hamiltonian.taylor_degree == taylor_degree
    assert hamiltonian.one_norm == one_norm


@pytest.mark.parametrize(
    "invalid_num_modes, invalid_grid_size, invalid_taylor_degree, invalid_one_norm",
    [
        ("4", 10, 3, None),
        (16, 4.5, 2, None),
        (16, 50, "2", None),
        (4, 10, 3, "None"),
        (-4, 10, 3, None),
        (16, -10, 2, None),
        (16, 50, -2, 10.0),
        (16, 50, 2, -5),
        (5, -4.5, 1, None),
    ],
)
def test_vibrational_invalid_types(
    invalid_num_modes, invalid_grid_size, invalid_taylor_degree, invalid_one_norm
):
    """Test that TypeError is raised for invalid input types."""
    with pytest.raises(TypeError):
        VibrationalHamiltonian(
            num_modes=invalid_num_modes,
            grid_size=invalid_grid_size,
            taylor_degree=invalid_taylor_degree,
            one_norm=invalid_one_norm,
        )


@pytest.mark.parametrize(
    "num_modes, num_states, grid_size, taylor_degree, one_norm",
    [(4, 2, 10, 3, None), (16, 3, 50, 2, 50.0), (8, 5, 20, 4, 25)],
)
def test_vibronic_instantiation(num_modes, num_states, grid_size, taylor_degree, one_norm):
    """Test successful instantiation with valid integer inputs."""
    hamiltonian = VibronicHamiltonian(
        num_modes=num_modes,
        num_states=num_states,
        grid_size=grid_size,
        taylor_degree=taylor_degree,
        one_norm=one_norm,
    )

    assert hamiltonian.num_modes == num_modes
    assert hamiltonian.num_states == num_states
    assert hamiltonian.grid_size == grid_size
    assert hamiltonian.taylor_degree == taylor_degree
    assert hamiltonian.one_norm == one_norm


@pytest.mark.parametrize(
    "invalid_num_modes, invalid_num_states, invalid_grid_size, invalid_taylor_degree, invalid_one_norm",
    [
        ("4", 2, 10, 3, None),
        (16, 3.5, 50, 2, None),
        (16, 3, "50", 2, None),
        (4, 2, 10, 3.5, None),
        (4, 2, 10, 3, "None"),
        (-4, 2, 10, 3, None),
        (16, -3, 50, 2, None),
        (16, 3, -50, 2, None),
        (16, 3, 50, -2.5, 10.0),
        (16, 3, 50, 2, -5),
    ],
)
def test_vibronic_invalid_types(
    invalid_num_modes,
    invalid_num_states,
    invalid_grid_size,
    invalid_taylor_degree,
    invalid_one_norm,
):
    """Test that TypeError is raised for invalid input types."""
    with pytest.raises(TypeError):
        VibronicHamiltonian(
            num_modes=invalid_num_modes,
            num_states=invalid_num_states,
            grid_size=invalid_grid_size,
            taylor_degree=invalid_taylor_degree,
            one_norm=invalid_one_norm,
        )
