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
"""Unit tests for the ``numeric_hamiltonians.py`` file."""

from dataclasses import dataclass

import numpy as np
import pytest

import pennylane as qp
from pennylane import CDFHamiltonian, CGFHamiltonian, NumericHamiltonian
from pennylane.typing import AbstractArray, Float

L, M, N = 2, 2, 3


def cdf_specs(num_fragments=L, num_orbitals=N):
    """Abstract CDF specifications, using the ``qp.typing.Float[...]`` notation."""
    shape = (num_fragments + 1, num_orbitals, num_orbitals)
    return {"core_tensors": Float[shape], "leaf_tensors": Float[shape], "nuc_constant": Float}


def cgf_specs(num_fragments=L, num_modes=M, num_modals=N):
    """Abstract CGF specifications."""
    return {
        "core_tensors": Float[num_fragments + 1, num_modes, num_modes, num_modals, num_modals],
        "leaf_tensors": Float[num_fragments + 1, num_modes, num_modals, num_modals],
        "nuc_constant": Float,
    }


class TestAbstract:
    """Tests for Hamiltonians built from ``qp.typing.Float[...]`` specifications."""

    def test_cgf_from_specs(self):
        """Test that abstract inputs surface as ``AbstractArray`` of the right shape."""
        ham = CGFHamiltonian(**cgf_specs())

        assert ham.is_abstract
        assert ham.core_tensors == AbstractArray((L + 1, M, M, N, N), float)
        assert ham.leaf_tensors == AbstractArray((L + 1, M, N, N), float)
        assert ham.nuc_constant == AbstractArray((), float)

    def test_cdf_from_specs(self):
        """Test abstract construction for CDF."""
        ham = CDFHamiltonian(**cdf_specs())

        assert ham.is_abstract
        assert ham.core_tensors == AbstractArray((L + 1, N, N), float)

    @pytest.mark.parametrize(
        "cls, specs, expected",
        [
            (CDFHamiltonian, cdf_specs, {"num_fragments": L, "num_orbitals": N}),
            (CGFHamiltonian, cgf_specs, {"num_fragments": L, "num_modes": M, "num_modals": N}),
        ],
    )
    def test_dimensions_derived_from_specs(self, cls, specs, expected):
        """Test that the dimensions are derived from abstract shapes too."""
        assert cls(**specs()).dimensions == expected

    def test_nuc_constant_defaults_to_abstract_scalar(self):
        """Test that omitting ``nuc_constant`` gives an abstract scalar."""
        ham = CGFHamiltonian(Float[L + 1, M, M, N, N], Float[L + 1, M, N, N])

        assert ham.is_abstract
        assert ham.nuc_constant == AbstractArray((), float)

    def test_abstractify_matches_abstract_construction(self):
        """Test that ``abstractify`` on concrete data reproduces the abstract instance."""
        assert qp.core.abstractify(CGFHamiltonian(**cgf_tensors())) == CGFHamiltonian(**cgf_specs())

    def test_unknown_axis_size_permitted(self):
        """Test that ``-1`` marks an axis of unknown size without pinning the symbol."""
        ham = CGFHamiltonian(Float[-1, M, M, N, N], Float[L + 1, M, N, N])

        assert ham.num_fragments == L
        assert ham.num_modes == M

    def test_fully_unknown_dimension(self):
        """Test that a dimension no tensor pins is reported as ``None``."""
        ham = CGFHamiltonian(Float[-1, M, M, N, N], Float[-1, M, N, N])

        assert ham.num_fragments is None
        assert ham.num_modes == M

    def test_inconsistent_shared_dimension(self):
        """Test that shape unification applies to abstract input identically."""
        with pytest.raises(ValueError, match="inconsistent 'num_modals'"):
            CGFHamiltonian(Float[L + 1, M, M, N, N], Float[L + 1, M, N, N + 1])
