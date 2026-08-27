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

# pylint: disable=import-outside-toplevel,too-many-public-methods,no-member

from dataclasses import dataclass

import numpy as np
import pytest

import pennylane as qp
from pennylane.numeric_hamiltonians import CDFHamiltonian, CGFHamiltonian, NumericHamiltonian
from pennylane.typing import AbstractArray, Float

L, M, N = 2, 2, 3


def cdf_tensors(seed, num_fragments=L, num_orbitals=N):
    """Concrete CDF tensor data: core and leaf both ``(L+1, N, N)``."""
    rng = np.random.default_rng(seed)
    shape = (num_fragments + 1, num_orbitals, num_orbitals)
    return {
        "core_tensors": rng.random(shape),
        "leaf_tensors": rng.random(shape),
        "nuc_constant": 0.5,
    }


def cgf_tensors(seed, num_fragments=L, num_modes=M, num_modals=N):
    """Concrete CGF tensor data: core ``(L+1, M, M, N, N)``, leaf ``(L+1, M, N, N)``."""
    rng = np.random.default_rng(seed)
    return {
        "core_tensors": rng.random(
            (num_fragments + 1, num_modes, num_modes, num_modals, num_modals)
        ),
        "leaf_tensors": rng.random((num_fragments + 1, num_modes, num_modals, num_modals)),
        "nuc_constant": 0.5,
    }


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


class TestConcrete:
    """Tests for Hamiltonians built from concrete numeric data."""

    @pytest.mark.parametrize(
        "cls, data, expected",
        [
            (CDFHamiltonian, cdf_tensors, {"num_fragments": L, "num_orbitals": N}),
            (
                CGFHamiltonian,
                cgf_tensors,
                {"num_fragments": L, "num_modes": M, "num_modals": N},
            ),
        ],
    )
    def test_dimensions_derived_from_shapes(self, cls, data, expected, seed):
        """Test that the named dimensions are derived from the tensor shapes."""
        ham = cls(**data(seed))

        assert ham.dimensions == expected
        for name, size in expected.items():
            assert getattr(ham, name) == size
        assert not ham.is_abstract

    def test_direct_attribute_access(self, seed):
        """Test that the numeric data is readable off the instance, and that the tensors
        are the positional arguments."""
        data = cgf_tensors(seed)
        ham = CGFHamiltonian(data["core_tensors"], data["leaf_tensors"], data["nuc_constant"])

        assert qp.math.allclose(ham.core_tensors, data["core_tensors"])
        assert qp.math.allclose(ham.leaf_tensors, data["leaf_tensors"])
        assert qp.math.allclose(ham.nuc_constant, data["nuc_constant"])
        assert ham.tensors == (ham.core_tensors, ham.leaf_tensors, ham.nuc_constant)

    def test_nuc_constant_defaults_to_zero_array(self, seed):
        """Test that omitting ``nuc_constant`` gives a rank-0 array of zero. It is stored
        as an array so the pytree leaf has a stable shape and dtype, which matters when
        the Hamiltonian is used as a control-flow carry."""
        data = cgf_tensors(seed)
        del data["nuc_constant"]
        nuc = CGFHamiltonian(**data).nuc_constant

        assert qp.math.allclose(nuc, 0.0)
        assert qp.math.shape(nuc) == ()
        assert hasattr(nuc, "dtype")

    @pytest.mark.jax
    def test_jax_arrays_accepted(self, seed):
        """Test that jax arrays work as tensor data."""
        import jax

        data = cgf_tensors(seed)
        ham = CGFHamiltonian(
            jax.numpy.array(data["core_tensors"]),
            jax.numpy.array(data["leaf_tensors"]),
            data["nuc_constant"],
        )

        assert ham.num_modes == M
        assert not ham.is_abstract

    def test_inconsistent_shared_dimension(self):
        """Test that a symbol appearing in both tensors must unify."""
        with pytest.raises(ValueError, match="inconsistent 'num_modals'"):
            CGFHamiltonian(np.zeros((L + 1, M, M, N, N)), np.zeros((L + 1, M, N, N + 1)))

    def test_inconsistent_fragment_count(self):
        """Test that the leading axis must agree between the two tensors."""
        with pytest.raises(ValueError, match="inconsistent 'num_fragments'"):
            CDFHamiltonian(np.zeros((L + 1, N, N)), np.zeros((L + 2, N, N)))

    def test_wrong_rank(self):
        """Test that a tensor with the wrong number of dimensions is rejected by name."""
        with pytest.raises(ValueError, match="'core_tensors' must have 5 dimensions"):
            CGFHamiltonian(np.zeros((L + 1, N, N)), np.zeros((L + 1, M, N, N)))

    def test_cdf_and_cgf_ranks_are_distinguished(self, seed):
        """Test that CGF-shaped data is rejected by CDF and vice versa: the class, not
        the rank, decides the representation."""
        cgf, cdf = cgf_tensors(seed), cdf_tensors(seed)

        with pytest.raises(ValueError, match="must have 3 dimensions"):
            CDFHamiltonian(cgf["core_tensors"], cgf["leaf_tensors"])

        with pytest.raises(ValueError, match="must have 5 dimensions"):
            CGFHamiltonian(cdf["core_tensors"], cdf["leaf_tensors"])

    @pytest.mark.parametrize(
        "shape, dimension",
        [
            # The leading axis holds L+1 entries, so L must be at least 1 (offset 1).
            ((1, N, N), "num_fragments"),
            # A zero-length axis with no offset is also rejected.
            ((L + 1, 0, 0), "num_orbitals"),
        ],
    )
    def test_degenerate_dimension_rejected(self, shape, dimension):
        """Test that every derived dimension must come out at least 1, offset or not."""
        with pytest.raises(ValueError, match=f"'{dimension}' must be at least 1"):
            CDFHamiltonian(np.zeros(shape), np.zeros(shape))

    def test_non_scalar_nuc_constant(self, seed):
        """Test that ``nuc_constant`` must be a scalar."""
        data = cgf_tensors(seed)
        data["nuc_constant"] = np.zeros(4)

        with pytest.raises(ValueError, match="'nuc_constant' must be a scalar"):
            CGFHamiltonian(**data)

    def test_missing_tensors(self):
        """Test that both tensors are required."""
        with pytest.raises(TypeError):
            CGFHamiltonian(np.zeros((L + 1, M, M, N, N)))  # pylint: disable=no-value-for-parameter

    @pytest.mark.parametrize(
        "cls, data", [(CDFHamiltonian, cdf_tensors), (CGFHamiltonian, cgf_tensors)]
    )
    def test_registered_with_pennylane_pytrees(self, cls, data, seed):
        """Test that both classes are registered with PennyLane's own pytree backend, not
        only with jax, so ``qp.pytrees.flatten`` sees them."""
        assert qp.pytrees.is_pytree(cls)

        leaves, structure = qp.pytrees.flatten(cls(**data(seed)))

        assert len(leaves) == 3
        assert qp.pytrees.unflatten(leaves, structure) == cls(**data(seed))

    @pytest.mark.jax
    @pytest.mark.parametrize(
        "cls, data", [(CDFHamiltonian, cdf_tensors), (CGFHamiltonian, cgf_tensors)]
    )
    def test_jax_roundtrip_preserves_values_and_structure(self, cls, data, seed):
        """Test that a ``jax.tree_util`` round-trip preserves values and structure."""
        import jax

        ham = cls(**data(seed))
        leaves, treedef = jax.tree_util.tree_flatten(ham)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        assert len(leaves) == 3
        assert isinstance(rebuilt, cls)
        for original, restored in zip(ham.tensors, rebuilt.tensors, strict=True):
            assert qp.math.allclose(original, restored)
        assert rebuilt == ham

    @pytest.mark.jax
    def test_derived_dimensions_survive_roundtrip(self, seed):
        """Test that the dimensions travel in the treedef, since unflattening skips
        validation and cannot re-derive them."""
        import jax

        ham = CGFHamiltonian(**cgf_tensors(seed))
        leaves, treedef = jax.tree_util.tree_flatten(ham)

        assert jax.tree_util.tree_unflatten(treedef, leaves).dimensions == ham.dimensions

    @pytest.mark.jax
    @pytest.mark.parametrize("leaf", [0.5, 3, True, complex(1, 2)])
    def test_tree_map_to_python_scalars(self, seed, leaf):
        """Test that a ``tree_map`` producing plain Python scalars stays usable."""
        import jax

        ham = CGFHamiltonian(**cgf_tensors(seed))
        mapped = jax.tree_util.tree_map(lambda _: leaf, ham)

        assert all(isinstance(t, type(leaf)) for t in mapped.tensors)
        assert isinstance(hash(mapped), int)
        assert mapped == jax.tree_util.tree_map(lambda _: leaf, ham)

    @pytest.mark.jax
    def test_traces_as_tracers(self, seed):
        """Test that concrete data becomes tracers when passed as a runtime argument,
        while the shape metadata stays statically available."""
        import jax

        seen = {}

        def f(h):
            seen["core"] = h.core_tensors
            seen["nuc"] = h.nuc_constant
            seen["num_modes"] = h.num_modes
            return h.core_tensors.sum() + h.nuc_constant

        jax.make_jaxpr(f)(CGFHamiltonian(**cgf_tensors(seed)))

        assert qp.math.is_abstract(seen["core"])
        assert qp.math.is_abstract(seen["nuc"])
        assert seen["core"].shape == (L + 1, M, M, N, N)
        assert seen["num_modes"] == M

    @pytest.mark.jax
    def test_constructed_from_tracers(self, seed):
        """Test that a Hamiltonian built inside a trace is concrete, not abstract: a
        tracer has a known shape and dtype."""
        import jax

        data = cgf_tensors(seed)
        seen = {}

        def f(core, leaf, nuc):
            ham = CGFHamiltonian(core, leaf, nuc)
            seen["is_abstract"] = ham.is_abstract
            seen["dimensions"] = ham.dimensions
            return ham.core_tensors.sum()

        jax.make_jaxpr(f)(data["core_tensors"], data["leaf_tensors"], data["nuc_constant"])

        assert seen["is_abstract"] is False
        assert seen["dimensions"] == {"num_fragments": L, "num_modes": M, "num_modals": N}

    @pytest.mark.jax
    def test_validation_applies_to_tracers(self, seed):
        """Test that shape validation still fires for traced data."""
        import jax

        data = cgf_tensors(seed)

        def f(core, leaf):
            return CGFHamiltonian(core, leaf).core_tensors.sum()

        with pytest.raises(ValueError, match="must have 5 dimensions with shape"):
            jax.make_jaxpr(f)(data["leaf_tensors"], data["leaf_tensors"])

    @pytest.mark.jax
    def test_differently_shaped_hamiltonians_have_distinct_structures(self, seed):
        """Test that the derived dimensions are part of the treedef, so differently
        shaped data does not silently share a structure, and hence a compiled
        signature."""
        import jax

        _, tree_a = jax.tree_util.tree_flatten(CGFHamiltonian(**cgf_tensors(seed)))
        _, tree_b = jax.tree_util.tree_flatten(
            CGFHamiltonian(**cgf_tensors(seed, num_fragments=L + 1))
        )

        assert tree_a != tree_b

    @pytest.mark.jax
    def test_cdf_and_cgf_structures_differ(self, seed):
        """Test that a CDF and a CGF payload are never conflated."""
        import jax

        _, cdf_tree = jax.tree_util.tree_flatten(CDFHamiltonian(**cdf_tensors(seed)))
        _, cgf_tree = jax.tree_util.tree_flatten(CGFHamiltonian(**cgf_tensors(seed)))

        assert cdf_tree != cgf_tree

    def test_hashable_with_array_data(self, seed):
        """Test that a tensor-backed Hamiltonian is hashable despite holding arrays."""
        assert isinstance(hash(CGFHamiltonian(**cgf_tensors(seed))), int)

    def test_hash_distinguishes_shapes(self, seed):
        """Test that different shape families do not collide."""
        assert hash(CGFHamiltonian(**cgf_tensors(seed))) != hash(
            CGFHamiltonian(**cgf_tensors(seed, num_modes=M + 1))
        )

    def test_equality_compares_values(self, seed):
        """Test that two Hamiltonians differing only in values are not equal."""
        assert CGFHamiltonian(**cgf_tensors(seed)) == CGFHamiltonian(**cgf_tensors(seed))
        assert CGFHamiltonian(**cgf_tensors(seed)) != CGFHamiltonian(**cgf_tensors(seed + 1))

    def test_equality_across_representations(self, seed):
        """Test that a CDF and a CGF Hamiltonian are never equal."""
        assert CDFHamiltonian(**cdf_tensors(seed)) != CGFHamiltonian(**cgf_tensors(seed))

    def test_equality_differing_shapes(self, seed):
        """Test that same-type Hamiltonians with different shapes are unequal."""
        small = CGFHamiltonian(**cgf_tensors(seed))
        large = CGFHamiltonian(**cgf_tensors(seed, num_fragments=L + 1))

        assert small != large
        # Confirm the guard is load-bearing rather than merely redundant.
        with pytest.raises(ValueError, match="could not be broadcast"):
            np.allclose(small.core_tensors, large.core_tensors)

    def test_equality_differing_dtypes(self):
        """Test that same-shape Hamiltonians with different dtypes are unequal."""
        shapes = ((L + 1, M, M, N, N), (L + 1, M, N, N))
        f64 = CGFHamiltonian(*(np.zeros(s, np.float64) for s in shapes))
        f32 = CGFHamiltonian(*(np.zeros(s, np.float32) for s in shapes))

        assert f64 != f32
        assert hash(f64) != hash(f32)
        # The values are identical, so only the dtype can be distinguishing them.
        assert qp.math.allclose(f64.core_tensors, f32.core_tensors)

    @pytest.mark.parametrize(
        "cls, data", [(CDFHamiltonian, cdf_tensors), (CGFHamiltonian, cgf_tensors)]
    )
    def test_repr_does_not_dump_arrays(self, cls, data, seed):
        """Test that the repr stays readable, summarizing arrays by shape."""
        text = repr(cls(**data(seed)))

        assert text.startswith(f"{cls.__name__}(")
        assert "tensor(shape=" in text
        assert len(text) < 300

    @pytest.mark.parametrize(
        "cls, data", [(CDFHamiltonian, cdf_tensors), (CGFHamiltonian, cgf_tensors)]
    )
    def test_immutable(self, cls, data, seed):
        """Test that instances are frozen dataclasses."""
        assert cls.__dataclass_params__.frozen is True
        ham = cls(**data(seed))

        with pytest.raises(AttributeError):
            ham.core_tensors = None

    def test_shared_base_and_exports(self):
        """Test that both classes share a base and are exported at the top level.

        The base is reachable on the module namespace rather than at the top level, since
        it is only needed for subclassing and ``isinstance`` checks.
        """
        assert issubclass(CDFHamiltonian, NumericHamiltonian)
        assert issubclass(CGFHamiltonian, NumericHamiltonian)
        assert qp.CDFHamiltonian is CDFHamiltonian
        assert qp.CGFHamiltonian is CGFHamiltonian
        assert qp.numeric_hamiltonians.NumericHamiltonian is NumericHamiltonian

    def test_new_subclass_from_shape_family_alone(self):
        """Test that defining a new representation needs only a shape family, with no
        new validation code."""

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

        ham = THCHamiltonian(np.zeros((7, 7)), np.zeros((7, 4)))

        assert ham.tensor_rank == 7
        assert ham.num_orbitals == 4
        assert qp.pytrees.is_pytree(THCHamiltonian)

        with pytest.raises(ValueError, match="inconsistent 'tensor_rank'"):
            THCHamiltonian(np.zeros((7, 7)), np.zeros((6, 4)))

    def test_cdf_normalize_leaf_determinant(self, seed):
        """Force every per-mode leaf to determinant ``+1``"""

        rng = np.random.default_rng(seed)
        _, n_states, L = 2, 3, 2
        leaves = rng.random((L + 1, n_states, n_states))

        ham = CDFHamiltonian(
            core_tensors=np.zeros((L + 1, n_states, n_states)),
            leaf_tensors=leaves,
            nuc_constant=0.0,
        )

        normalized = ham.normalize_leaf_determinant().leaf_tensors
        dets = qp.math.sign(qp.math.linalg.det(normalized))
        assert np.allclose(dets, np.ones_like(dets))

    def test_cgf_normalize_leaf_determinant(self, seed):
        """Force every per-mode leaf to determinant ``+1``"""

        rng = np.random.default_rng(seed)
        num_modes, n_states, L = 2, 3, 2
        leaves = rng.random((L + 1, num_modes, n_states, n_states))

        ham = CGFHamiltonian(
            core_tensors=np.zeros((L + 1, num_modes, num_modes, n_states, n_states)),
            leaf_tensors=leaves,
            nuc_constant=0.0,
        )

        normalized = ham.normalize_leaf_determinant().leaf_tensors
        dets = qp.math.sign(qp.math.linalg.det(normalized))
        assert np.allclose(dets, np.ones_like(dets))

    def test_cgf_align_one_body_leaf(self, seed):
        """The one-body leaf (eigenvectors stored as columns) is transposed per mode to match
        the two-body row convention; the two-body leaves are returned untouched."""

        rng = np.random.default_rng(seed)
        num_modes, n_states, L = 2, 3, 2
        leaf = np.stack(
            [
                np.stack([random_orthogonal(n_states, rng) for _ in range(num_modes)])
                for _ in range(L + 1)
            ]
        )

        ham = CGFHamiltonian(
            core_tensors=np.zeros((L + 1, num_modes, num_modes, n_states, n_states)),
            leaf_tensors=leaf,
            nuc_constant=0.0,
        )
        aligned = ham.align_one_body_leaf().leaf_tensors
        assert np.allclose(aligned[0], np.swapaxes(leaf[0], -2, -1))
        assert np.allclose(aligned[1:], leaf[1:])


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

    def test_abstractify_matches_abstract_construction(self, seed):
        """Test that ``abstractify`` on concrete data reproduces the abstract instance."""
        assert qp.core.abstractify(CGFHamiltonian(**cgf_tensors(seed))) == CGFHamiltonian(
            **cgf_specs()
        )

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

    def test_hash_matches_concrete_counterpart(self, seed):
        """Test that hashing on shapes makes an abstract Hamiltonian hash equal to the
        concrete one it describes."""
        concrete = CGFHamiltonian(**cgf_tensors(seed))

        assert hash(CGFHamiltonian(**cgf_specs())) == hash(concrete)
        assert hash(qp.core.abstractify(concrete)) == hash(concrete)

    def test_equality_ignores_values(self, seed):
        """Test that comparing abstract to concrete compares shapes only."""
        assert CGFHamiltonian(**cgf_specs()) == CGFHamiltonian(**cgf_tensors(seed))

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
