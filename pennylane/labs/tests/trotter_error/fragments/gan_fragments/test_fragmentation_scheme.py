"""Tests for the fragment-construction logic in ``fragmentation_scheme.py``.

Covers the GanConfig-driven ``gan_fragments`` entry point (fragment count,
types, and shape/type validation) and the private builders that translate
coefficient tensors into GanCoeff polynomials and matching fragments.
"""

from itertools import product

import numpy as np
import pytest

from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import FermiOp, FermiWord
from pennylane.labs.trotter_error.fragments.gan_fragments.fragmentation_scheme import (
    GanConfig,
    _diagonal,
    _kinetic,
    _met_matching,
    _mol_matching,
    _molecular_coupling,
    _molecule_metal_transfer,
    _nuclear_reference,
    gan_fragments,
)
from pennylane.labs.trotter_error.fragments.gan_fragments.gan_fragments import (
    FuncSymbol,
    GanCoeff,
    GanFragment,
    GanMonomial,
)

N_MOL = 3
N_MET = 6
N_MODES = 5


def _symmetric_coupling(rng):
    a = rng.random((N_MOL, N_MOL, N_MODES))
    return (a + a.transpose(1, 0, 2)) / 2


@pytest.fixture(name="config")
def fixture_config():
    """Returns a GanConfig to be used in testing"""
    rng = np.random.default_rng(42)

    couplings = [_symmetric_coupling(rng), _symmetric_coupling(rng)]
    repulsion = [_symmetric_coupling(rng), _symmetric_coupling(rng)]

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


def test_gan_fragments_count_and_types(config):
    """Test that the fragments are of the right form"""
    frags = gan_fragments(config)
    offset = 1 - (config.n_mol % 2)
    expected = 1 + (config.n_mol - offset) + config.n_met + 1
    assert len(frags) == expected
    assert all(isinstance(f, GanFragment) for f in frags)


def test_gan_fragments_first_is_diagonal_last_is_kinetic(config):
    """Test the order of the fragments"""
    frags = gan_fragments(config)
    assert frags[0] == _diagonal(config)
    assert frags[-1] == _kinetic(config)


def test_gan_fragments_rejects_non_sequence_couplings(config):
    """Test that couplings are a sequence"""
    config.couplings = np.zeros((N_MOL, N_MOL, N_MODES))
    with pytest.raises(TypeError):
        gan_fragments(config)


def test_gan_fragments_rejects_non_sequence_repulsion(config):
    """Test that repulsion is a sequence"""
    config.repulsion = np.zeros((N_MOL, N_MOL, N_MODES))
    with pytest.raises(TypeError):
        gan_fragments(config)


def test_gan_fragments_rejects_bad_masses_shape(config):
    """Test the shape of massess"""
    config.masses = np.zeros(N_MODES + 1)
    with pytest.raises(ValueError):
        gan_fragments(config)


def test_gan_fragments_rejects_bad_energies_shape(config):
    """Test the shape of energies"""
    config.energies = np.zeros(N_MET + 1)
    with pytest.raises(ValueError):
        gan_fragments(config)


def test_gan_fragments_rejects_bad_coupling_shape(config):
    """Test the shape of couplings"""
    config.couplings = [np.zeros((N_MOL, N_MOL, N_MODES + 1))]
    with pytest.raises(ValueError):
        gan_fragments(config)


def test_mol_matching_returns_fragment(config):
    """Test the return type of mol_matching"""
    frag = _mol_matching(0, config)
    assert isinstance(frag, GanFragment)
    assert len(frag.fragment) > 0


def test_mol_matching_is_hermitian_pairing(config):
    """Test that mol_matching returns pairs"""

    frag = _mol_matching(0, config)
    for fermi in frag.fragment:
        ops = fermi.ops
        assert len(ops) == 2


def test_met_matching_pairs_each_molecular_mode(config):
    """Test that each molecular orbital appears in fragment"""
    frag = _met_matching(1, config)
    assert isinstance(frag, GanFragment)
    # n_mol edges, each contributing two directed hops (mol->met and met->mol)
    assert len(frag.fragment) == 2 * config.n_mol


def test_molecular_coupling_matches_manual_sum(config):
    """Test that the molecular coupling fragment matches expected value"""
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
    """Test that the constant term is the identity monomial"""
    i, j = 1, 2
    coeff = _molecule_metal_transfer(i, j, config)
    const_mono = GanMonomial([FuncSymbol.identity()])
    expected_val = config.transfer[0][i, j]
    assert const_mono in coeff.monomials
    assert coeff.monomials[const_mono] == pytest.approx(expected_val)


def test_diagonal_contains_identity_nuclear_reference(config):
    """Test that the diagonal contains the nuclear reference"""
    frag = _diagonal(config)
    assert FermiWord.identity() in frag.fragment


def test_kinetic_has_metallic_number_terms(config):
    """Test that kinetic term contains the correct number of terms"""
    frag = _kinetic(config)
    for i in range(config.n_met):
        num = FermiWord([FermiOp.creation_met(i), FermiOp.annihilation_met(i)])
        assert num in frag.fragment


def test_nuclear_reference_returns_gancoeff(config):
    """Test return type of nuclear reference"""
    coeff = _nuclear_reference(config)
    assert isinstance(coeff, GanCoeff)
