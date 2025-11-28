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

from pennylane.estimator import (
    CDFHamiltonian,
    THCHamiltonian,
    VibrationalHamiltonian,
    VibronicHamiltonian,
)

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
