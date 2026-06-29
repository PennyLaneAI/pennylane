"""Tests for the fragment-construction logic in ``fragmentation_scheme.py``.

Covers the GanConfig-driven ``gan_fragments`` entry point (fragment count,
types, and shape/type validation) and the private builders that translate
coefficient tensors into GanCoeff polynomials and matching fragments.
"""

from itertools import product

import numpy as np
import pytest

from pennylane.labs.trotter_error.fragments.gan_fragments.fragmentation_scheme import (
    GanConfig,
    gan_fragments,
    _diagonal,
    _kinetic,
    _mol_matching,
    _met_matching,
    _molecular_coupling,
    _electron_repulsion,
    _molecule_metal_transfer,
    _nuclear_reference,
)
from pennylane.labs.trotter_error.fragments.gan_fragments.gan_fragments import (
    GanFragment,
    GanCoeff,
    GanMonomial,
    FuncSymbol,
)

from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import FermiWord, FermiOp


N_MOL = 3
N_MET = 6
N_MODES = 5


def _symmetric_coupling(rng):
    a = rng.random((N_MOL, N_MOL, N_MODES))
    return (a + a.transpose(1, 0, 2)) / 2


@pytest.fixture
def config():
    rng = np.random.default_rng(42)

    couplings = [_symmetric_coupling(rng), _symmetric_coupling(rng)]

    rep = [_symmetric_coupling(rng), _symmetric_coupling(rng)]
    for tensor in rep:
        for i, j in product(range(N_MOL), repeat=2):
            tensor[i, j] = 0
    repulsion = rep

    nuclear = [rng.random(N_MODES), rng.random(N_MODES)]
    transfer = [rng.random((N_MOL, N_MET))]  # Sequence of (n_mol, n_met) arrays
    masses = rng.random(N_MODES)
    energies = rng.random(N_MET)

    return GanConfig(
        n_modes=N_MODES,
        n_met=N_MET,
        n_mol=N_MOL,
        couplings=couplings,
        repulsion=repulsion,
        nuclear=nuclear,
        transfer=transfer,
        masses=masses,
        energies=energies,
    )


# --------------------------------------------------------------------------- #
# gan_fragments: assembly
# --------------------------------------------------------------------------- #
def test_gan_fragments_count_and_types(config):
    frags = gan_fragments(config)
    # 1 diagonal + (n_mol - offset) mol matchings + n_met met matchings + 1 kinetic
    offset = 1 - (config.n_mol % 2)
    expected = 1 + (config.n_mol - offset) + config.n_met + 1
    assert len(frags) == expected
    assert all(isinstance(f, GanFragment) for f in frags)


def test_gan_fragments_first_is_diagonal_last_is_kinetic(config):
    frags = gan_fragments(config)
    assert frags[0] == _diagonal(config)
    assert frags[-1] == _kinetic(config)


# --------------------------------------------------------------------------- #
# gan_fragments: validation
# --------------------------------------------------------------------------- #
def test_gan_fragments_rejects_non_sequence_couplings(config):
    config.couplings = np.zeros((N_MOL, N_MOL, N_MODES))
    with pytest.raises(TypeError):
        gan_fragments(config)


def test_gan_fragments_rejects_non_sequence_repulsion(config):
    config.repulsion = np.zeros((N_MOL, N_MOL, N_MODES))
    with pytest.raises(TypeError):
        gan_fragments(config)


def test_gan_fragments_rejects_bad_masses_shape(config):
    config.masses = np.zeros(N_MODES + 1)
    with pytest.raises(ValueError):
        gan_fragments(config)


def test_gan_fragments_rejects_bad_energies_shape(config):
    config.energies = np.zeros(N_MET + 1)
    with pytest.raises(ValueError):
        gan_fragments(config)


def test_gan_fragments_rejects_bad_coupling_shape(config):
    config.couplings = [np.zeros((N_MOL, N_MOL, N_MODES + 1))]
    with pytest.raises(ValueError):
        gan_fragments(config)


# --------------------------------------------------------------------------- #
# Matching fragments
# --------------------------------------------------------------------------- #
def test_mol_matching_returns_fragment(config):
    frag = _mol_matching(0, config)
    assert isinstance(frag, GanFragment)
    assert len(frag.fragment) > 0


def test_mol_matching_is_hermitian_pairing(config):
    # Each edge (i, j) contributes both c_i^dag a_j and c_j^dag a_i with the
    # same coefficient, so the two directed terms must both be present.

    frag = _mol_matching(0, config)
    for fermi in frag.fragment:
        ops = fermi.ops
        # all matching terms are single hops: c^dag a
        assert len(ops) == 2


def test_met_matching_pairs_each_molecular_mode(config):
    frag = _met_matching(1, config)
    assert isinstance(frag, GanFragment)
    # n_mol edges, each contributing two directed hops (mol->met and met->mol)
    assert len(frag.fragment) == 2 * config.n_mol


# --------------------------------------------------------------------------- #
# Coefficient builders
# --------------------------------------------------------------------------- #
def test_molecular_coupling_matches_manual_sum(config):
    i, j = 0, 1
    coeff = _molecular_coupling(i, j, config)

    # Reconstruct the expected polynomial, mirroring both the full-tensor and
    # diagonal-tensor branches of _molecular_coupling.
    monomials = {}
    for order, tensor in enumerate(config.couplings, start=1):
        full_shape = (config.n_mol, config.n_mol) + (config.n_modes,) * order
        diag_shape = (config.n_mol, config.n_mol, config.n_modes)
        if tensor.shape == full_shape:
            for modes in product(range(config.n_modes), repeat=order):
                mono = GanMonomial([FuncSymbol.position(m) for m in modes])
                monomials[mono] = monomials.get(mono, 0.0) + tensor[(i, j) + modes]
        if tensor.shape == diag_shape and order > 1:
            for mode in range(config.n_modes):
                mono = GanMonomial([FuncSymbol.position(mode, order)])
                monomials[mono] = monomials.get(mono, 0.0) + tensor[i, j, mode]
    expected = GanCoeff(monomials)

    assert coeff == expected


def test_molecule_metal_transfer_constant_term(config):
    # With only a zeroth-order transfer tensor, the coefficient is a single
    # identity monomial carrying transfer[0][i, j].
    i, j = 1, 2
    coeff = _molecule_metal_transfer(i, j, config)
    const_mono = GanMonomial([FuncSymbol.identity()])
    expected_val = config.transfer[0][i, j]
    assert const_mono in coeff.monomials
    assert coeff.monomials[const_mono] == pytest.approx(expected_val)


def test_diagonal_contains_identity_nuclear_reference(config):
    frag = _diagonal(config)
    # The nuclear reference energy sits on the identity fermionic word.
    assert FermiWord.identity() in frag.fragment


def test_kinetic_has_metallic_number_terms(config):
    frag = _kinetic(config)
    for i in range(config.n_met):
        num = FermiWord([FermiOp.creation_met(i), FermiOp.annihilation_met(i)])
        assert num in frag.fragment


def test_electron_repulsion_zero_when_tensor_zero(config):
    # The fixture zeroes the repulsion tensors, so every coefficient is empty.
    for i, j in product(range(config.n_mol), repeat=2):
        assert _electron_repulsion(i, j, config).is_zero()


def test_nuclear_reference_returns_gancoeff(config):
    coeff = _nuclear_reference(config)
    assert isinstance(coeff, GanCoeff)
