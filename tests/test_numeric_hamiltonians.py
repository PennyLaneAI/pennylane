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

    def test_inconsistent_fragment_count(self):
        """Test that the leading axis must agree for abstract input too."""
        with pytest.raises(ValueError, match="inconsistent 'num_fragments'"):
            CDFHamiltonian(Float[L + 1, N, N], Float[L + 2, N, N])

    def test_wrong_rank(self):
        """Test that the rank check applies to abstract input too."""
        with pytest.raises(ValueError, match="'core_tensors' must have 5 dimensions"):
            CGFHamiltonian(Float[L + 1, N, N], Float[L + 1, M, N, N])

    def test_hash_matches_concrete_counterpart(self):
        """Test that hashing on shapes makes an abstract Hamiltonian hash equal to the
        concrete one it describes."""
        concrete = CGFHamiltonian(**cgf_tensors())

        assert hash(CGFHamiltonian(**cgf_specs())) == hash(concrete)
        assert hash(qp.core.abstractify(concrete)) == hash(concrete)

    def test_equality_ignores_values(self):
        """Test that comparing abstract to concrete compares shapes only."""
        assert CGFHamiltonian(**cgf_specs()) == CGFHamiltonian(**cgf_tensors(seed=7))

    def test_repr_shows_specs(self):
        """Test that an abstract Hamiltonian reports its ``AbstractArray`` specs."""
        assert "AbstractArray" in repr(CGFHamiltonian(**cgf_specs()))

    @pytest.mark.jax
    def test_jax_roundtrip(self):
        """Test that an abstract Hamiltonian survives a ``jax.tree_util`` round-trip."""
        import jax

        ham = CGFHamiltonian(**cgf_specs())
        leaves, treedef = jax.tree_util.tree_flatten(ham)

        assert jax.tree_util.tree_unflatten(treedef, leaves) == ham

    def test_new_subclass_supports_abstract_data(self):
        """Test that a new representation gets abstract construction for free."""

        # pylint: disable=too-few-public-methods
        @dataclass(frozen=True, eq=False, repr=False)
        class THCHamiltonian(NumericHamiltonian):
            """Tensor-hypercontracted shape family, for this test only."""

            core_shape = ("R", "R")
            leaf_shape = ("R", "N")
            symbol_metadata = {"R": ("tensor_rank", 0), "N": ("num_orbitals", 0)}

            core_tensors: object
            leaf_tensors: object
            nuc_constant: object = None

        ham = THCHamiltonian(Float[7, 7], Float[7, 4])

        assert ham.is_abstract
        assert ham.dimensions == {"tensor_rank": 7, "num_orbitals": 4}
