"""Pytest suite for GAN-fragment commutator identities.

Each test corresponds to one commutator identity from the reference notes and
is parametrized over several matching-index inputs. Shared, deterministic
setup (the ``GanConfig`` and the energy fragment) is provided through fixtures
so the tests are order-independent.

Index ranges used for parametrization
-------------------------------------
Inferred from the edge generators and ``config`` (n_mol=3, n_met=6):
  * molecular matching index reduces mod (n_mol + t - 1) = 3  -> s, k in {0, 1, 2}
  * metal matching index uses    mod n_met = 6                -> s, k in {0..5}
These were NOT verified against the source of ``_mol_matching`` /
``_met_matching`` (repo unavailable at conversion time). If a parametrized
case raises IndexError, narrow the corresponding range.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pytest

from pennylane.labs.trotter_error import GanConfig
from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import FermiOp, FermiWord
from pennylane.labs.trotter_error.fragments.gan_fragments.fragmentation_scheme import (
    _diagonal,
    _electron_repulsion,
    _met_matching,
    _mol_matching,
    _molecular_coupling,
    _molecule_metal_transfer,
)
from pennylane.labs.trotter_error.fragments.gan_fragments.gan_fragments import (
    GanCoeff,
    GanFragment,
    GanMonomial,
)
from pennylane.labs.trotter_error.product_formulas.commutator import CommutatorNode, SymbolNode


@pytest.fixture(name="config")
def fixture_config():
    """Deterministic GanConfig built from a fixed random seed."""
    np.random.seed(42)

    n_modes = 4

    n_mol = 3
    n_met = 6

    arr1 = np.random.random(size=(n_mol, n_mol, n_modes))
    arr1 = (arr1 + arr1.transpose(1, 0, 2)) / 2

    arr2 = np.random.random(size=(n_mol, n_mol, n_modes))
    arr2 = (arr2 + arr2.transpose(1, 0, 2)) / 2

    couplings = [arr1, arr2]

    arr1 = np.random.random(size=(n_mol, n_mol, n_modes))
    arr1 = (arr1 + arr1.transpose(1, 0, 2)) / 2

    arr2 = np.random.random(size=(n_mol, n_mol, n_modes))
    arr2 = (arr2 + arr2.transpose(1, 0, 2)) / 2

    for i in range(n_mol):
        arr1[i, i] = 0
        arr2[i, i] = 0

    repulsions = [arr1, arr2]

    nuclear = [np.random.random(size=n_modes), np.random.random(size=n_modes)]
    transfer = [
        np.random.random(size=(n_mol, n_met)),
        np.random.random(size=(n_mol, n_met)),
        np.random.random(size=(n_mol, n_met)),
    ]
    masses = np.random.random(size=n_modes)
    energies = np.random.random(size=n_met)

    return GanConfig(
        n_modes=n_modes,
        n_met=n_met,
        n_mol=n_mol,
        couplings=couplings,
        repulsion=repulsions,
        nuclear=nuclear,
        transfer=transfer,
        masses=masses,
        energies=energies,
    )


@pytest.fixture(name="feps")
def fixture_feps(config):
    """The energy (epsilon) fragment F_eps = sum_i eps_i c_i^dag c_i (metal)."""
    terms = defaultdict(GanCoeff.identity)
    for i, energy in enumerate(config.energies):
        monomial = GanMonomial.identity()
        coeff = GanCoeff({monomial: energy})
        fermi = FermiWord([FermiOp.creation_met(i), FermiOp.annihilation_met(i)])
        terms[fermi] += coeff
    return GanFragment(terms)


def _mol_edges(s: int, config: GanConfig):
    """Returns the edges of the mol matching for mode s"""
    t = config.n_mol % 2
    k = (config.n_mol + t) // 2

    edges = set()

    for i in range(1, k):
        u = (s + i) % (config.n_mol + t - 1)
        v = (s - i) % (config.n_mol + t - 1)
        edges.add((("mol", u), ("mol", v)))

    if t == 0:
        edges.add((("mol", s), ("mol", config.n_mol - 1)))

    return edges


def _met_edges(s: int, config: GanConfig):
    """Returns the edges of the met/mol matching for mode s"""
    edges = set()

    for i in range(config.n_mol):
        u = i
        v = (i + s) % config.n_met
        edges.add((("mol", u), ("met", v)))

    return edges


def _expected_mol_mol(s, k, config):
    """Returns the expected evaluation of [F_s, F_k] where s and k are in the mol space"""
    terms = defaultdict(GanCoeff.identity)
    for e1 in _mol_edges(s, config):
        for e2 in _mol_edges(k, config):
            e1 = set(e1)
            e2 = set(e2)

            if len(e1.intersection(e2)) != 1:
                continue

            i, j = e1
            m, n = e2

            if i == m:
                i, j = j, i
            elif i == n:
                i, j = j, i
                m, n = n, m
            elif j == n:
                m, n = n, m

            g_ij = _molecular_coupling(i[1], j[1], config)
            g_mn = _molecular_coupling(m[1], n[1], config)

            fermi1 = FermiWord([FermiOp.creation_mol(i[1]), FermiOp.annihilation_mol(n[1])])
            fermi2 = FermiWord([FermiOp.creation_mol(n[1]), FermiOp.annihilation_mol(i[1])])

            terms[fermi1] += g_ij @ g_mn
            terms[fermi2] -= g_ij @ g_mn
    return GanFragment(terms)


def _expected_met_met(s, k, config):
    """Returns the expected evluation of [F_s, F_k] where s and k are in the met space"""
    terms = defaultdict(GanCoeff.identity)
    for e1 in _met_edges(s, config):
        for e2 in _met_edges(k, config):

            e1 = set(e1)
            e2 = set(e2)

            if len(e1.intersection(e2)) != 1:
                continue

            i, j = e1
            m, n = e2

            if i == m:
                i, j = j, i
            elif i == n:
                i, j = j, i
                m, n = n, m
            elif j == n:
                m, n = n, m

            mol_index = i[1] if i[0] == "mol" else j[1]
            met_index = i[1] if i[0] == "met" else j[1]

            g_ij = _molecule_metal_transfer(mol_index, met_index, config)

            mol_index = m[1] if m[0] == "mol" else n[1]
            met_index = m[1] if m[0] == "met" else n[1]

            g_mn = _molecule_metal_transfer(mol_index, met_index, config)

            if i[0] == "mol" and n[0] == "mol":
                fermi1 = FermiWord([FermiOp.creation_mol(i[1]), FermiOp.annihilation_mol(n[1])])
                fermi2 = FermiWord([FermiOp.creation_mol(n[1]), FermiOp.annihilation_mol(i[1])])
            elif i[0] == "mol" and n[0] == "met":
                fermi1 = FermiWord([FermiOp.creation_mol(i[1]), FermiOp.annihilation_met(n[1])])
                fermi2 = FermiWord([FermiOp.creation_met(n[1]), FermiOp.annihilation_mol(i[1])])
            elif i[0] == "met" and n[0] == "mol":
                fermi1 = FermiWord([FermiOp.creation_met(i[1]), FermiOp.annihilation_mol(n[1])])
                fermi2 = FermiWord([FermiOp.creation_mol(n[1]), FermiOp.annihilation_met(i[1])])
            elif i[0] == "met" and n[0] == "met":
                fermi1 = FermiWord([FermiOp.creation_met(i[1]), FermiOp.annihilation_met(n[1])])
                fermi2 = FermiWord([FermiOp.creation_met(n[1]), FermiOp.annihilation_met(i[1])])
            else:
                raise ValueError(f"Unexpected spaces: {i[0]}, {i[1]}.")

            terms[fermi1] += g_ij @ g_mn
            terms[fermi2] -= g_ij @ g_mn
    return GanFragment(terms)


def _expected_mol_met(s, k, config):
    """Returns the expected evaluation of [F_s, F_k] where s is in the mol space and k is in the met space"""
    terms = defaultdict(GanCoeff.identity)
    for e1 in _mol_edges(s, config):
        for e2 in _met_edges(k, config):

            e1 = set(e1)
            e2 = set(e2)

            if len(e1.intersection(e2)) != 1:
                continue

            i, j = e1
            m, n = e2

            if i == m:
                i, j = j, i
            elif i == n:
                i, j = j, i
                m, n = n, m
            elif j == n:
                m, n = n, m

            g_ij = _molecular_coupling(i[1], j[1], config)

            mol_index = m[1] if m[0] == "mol" else n[1]
            met_index = m[1] if m[0] == "met" else n[1]

            g_mn = _molecule_metal_transfer(mol_index, met_index, config)

            if i[0] == "mol" and n[0] == "mol":
                fermi1 = FermiWord([FermiOp.creation_mol(i[1]), FermiOp.annihilation_mol(n[1])])
                fermi2 = FermiWord([FermiOp.creation_mol(n[1]), FermiOp.annihilation_mol(i[1])])
            elif i[0] == "mol" and n[0] == "met":
                fermi1 = FermiWord([FermiOp.creation_mol(i[1]), FermiOp.annihilation_met(n[1])])
                fermi2 = FermiWord([FermiOp.creation_met(n[1]), FermiOp.annihilation_mol(i[1])])
            elif i[0] == "met" and n[0] == "mol":
                fermi1 = FermiWord([FermiOp.creation_met(i[1]), FermiOp.annihilation_mol(n[1])])
                fermi2 = FermiWord([FermiOp.creation_mol(n[1]), FermiOp.annihilation_met(i[1])])
            elif i[0] == "met" and n[0] == "met":
                fermi1 = FermiWord([FermiOp.creation_met(i[1]), FermiOp.annihilation_met(n[1])])
                fermi2 = FermiWord([FermiOp.creation_met(n[1]), FermiOp.annihilation_met(i[1])])
            else:
                raise ValueError(f"Unexpected spaces {i[0]}, {i[1]}.")

            terms[fermi1] += g_ij @ g_mn
            terms[fermi2] -= g_ij @ g_mn
    return GanFragment(terms)


def _expected_feps_met(s, config):
    """Returns the expected evaluation of [F_eps, F_s] where s is in the met space"""
    terms = defaultdict(GanCoeff.identity)
    for i, a in _met_edges(s, config):

        met_index = i[1] if i[0] == "met" else a[1]
        mol_index = i[1] if i[0] == "mol" else a[1]

        e_a = config.energies[met_index]
        g_ia = _molecule_metal_transfer(mol_index, met_index, config)

        fermi1 = FermiWord([FermiOp.creation_mol(mol_index), FermiOp.annihilation_met(met_index)])
        fermi2 = FermiWord([FermiOp.creation_met(met_index), FermiOp.annihilation_mol(mol_index)])

        terms[fermi1] -= e_a * g_ia
        terms[fermi2] += e_a * g_ia

    return GanFragment(terms)


def _expected_f0_mol(s, config):
    """Returns the expected evaluation of [F_0, F_s] where s is in the mol space"""
    F0 = _diagonal(config)
    Fs = _mol_matching(s, config)

    comm = CommutatorNode(SymbolNode("F0"), SymbolNode("Fs"))
    comm = comm.eval({"F0": F0, "Fs": Fs})

    terms = defaultdict(GanCoeff.identity)
    for r in range(config.n_mol):
        for i, j in _mol_edges(s, config):

            i, j = i[1], j[1]
            ij = {i, j}

            if r not in ij:
                continue

            a = ij.difference({r}).pop()

            g_rr = _molecular_coupling(r, r, config)
            U_ij = _molecular_coupling(i, j, config)

            fermi1 = FermiWord([FermiOp.creation_mol(r), FermiOp.annihilation_mol(a)])
            fermi2 = FermiWord([FermiOp.creation_mol(a), FermiOp.annihilation_mol(r)])

            terms[fermi1] += g_rr @ U_ij
            terms[fermi2] -= g_rr @ U_ij

    for p, q in combinations(range(config.n_mol), r=2):
        for i, j in _mol_edges(s, config):

            i, j = i[1], j[1]
            pq = {p, q}
            ij = {i, j}

            if len(pq.intersection(ij)) != 1:
                continue

            c = pq.intersection(ij)
            a = ij.difference(c).pop()
            b = pq.difference(c).pop()
            c = c.pop()

            V_pq = _electron_repulsion(p, q, config)
            U_ij = _molecular_coupling(i, j, config)

            fermi1 = FermiWord(
                [
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                    FermiOp.creation_mol(c),
                    FermiOp.annihilation_mol(a),
                ]
            )
            for fermi, coeff in fermi1.normal_order().words.items():
                terms[fermi] += coeff * V_pq @ U_ij

            fermi2 = FermiWord(
                [
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                    FermiOp.creation_mol(a),
                    FermiOp.annihilation_mol(c),
                ]
            )
            for fermi, coeff in fermi2.normal_order().words.items():
                terms[fermi] -= coeff * V_pq @ U_ij

            fermi3 = FermiWord(
                [
                    FermiOp.creation_mol(c),
                    FermiOp.annihilation_mol(a),
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                ]
            )
            for fermi, coeff in fermi3.normal_order().words.items():
                terms[fermi] += coeff * V_pq @ U_ij

            fermi4 = FermiWord(
                [
                    FermiOp.creation_mol(a),
                    FermiOp.annihilation_mol(c),
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                ]
            )
            for fermi, coeff in fermi4.normal_order().words.items():
                terms[fermi] -= coeff * V_pq @ U_ij

    return GanFragment(terms)


def _expected_f0_met(s, config):
    """Returns the expected evaluation of [F_0, F_s] where s is in the met space"""

    terms = defaultdict(GanCoeff.identity)
    for r in range(config.n_mol):
        for i, a in _met_edges(s, config):

            i, a = i[1], a[1]

            if r != i:
                continue

            g_rr = _molecular_coupling(r, r, config)
            W_ia = _molecule_metal_transfer(i, a, config)

            fermi1 = FermiWord([FermiOp.creation_mol(r), FermiOp.annihilation_met(a)])
            fermi2 = FermiWord([FermiOp.creation_met(a), FermiOp.annihilation_mol(r)])

            terms[fermi1] += g_rr @ W_ia
            terms[fermi2] -= g_rr @ W_ia

    for p, q in combinations(range(config.n_mol), r=2):

        if p == q:
            continue

        for i, a in _met_edges(s, config):
            i, a = i[1], a[1]
            pq = {p, q}

            if i not in pq:
                continue

            c = pq.intersection({i}).pop()
            b = pq.difference({c}).pop()

            V_pq = _electron_repulsion(p, q, config)
            W_ia = _molecule_metal_transfer(i, a, config)

            fermi1 = FermiWord(
                [
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                    FermiOp.creation_mol(c),
                    FermiOp.annihilation_met(a),
                ]
            )
            for fermi, coeff in fermi1.normal_order().words.items():
                terms[fermi] += coeff * V_pq @ W_ia

            fermi2 = FermiWord(
                [
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                    FermiOp.creation_met(a),
                    FermiOp.annihilation_mol(c),
                ]
            )
            for fermi, coeff in fermi2.normal_order().words.items():
                terms[fermi] -= coeff * V_pq @ W_ia

            fermi3 = FermiWord(
                [
                    FermiOp.creation_mol(c),
                    FermiOp.annihilation_met(a),
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                ]
            )
            for fermi, coeff in fermi3.normal_order().words.items():
                terms[fermi] += coeff * V_pq @ W_ia

            fermi4 = FermiWord(
                [
                    FermiOp.creation_met(a),
                    FermiOp.annihilation_mol(c),
                    FermiOp.creation_mol(b),
                    FermiOp.annihilation_mol(b),
                ]
            )
            for fermi, coeff in fermi4.normal_order().words.items():
                terms[fermi] -= coeff * V_pq @ W_ia

    return GanFragment(terms)


@pytest.mark.parametrize("s, k", [(0, 1), (1, 2), (2, 0), (0, 2), (1, 0)])
def test_commutator_mol_mol(s, k, config):
    """Test [F_s, F_k] evaluates correctly where s and k are in the mol space"""
    Fs = _mol_matching(s, config)
    Fk = _mol_matching(k, config)

    comm = CommutatorNode(SymbolNode("Fs"), SymbolNode("Fk"))
    comm = comm.eval({"Fs": Fs, "Fk": Fk})

    expected = _expected_mol_mol(s, k, config)

    assert comm.fragment.keys() == expected.fragment.keys()
    assert comm == expected


@pytest.mark.parametrize("s, k", [(0, 1), (1, 3), (2, 5), (0, 3), (4, 2)])
def test_commutator_met_met(s, k, config):
    """Test [F_s, F_k] evaluates correctly where s and k are in the met space"""
    Fs = _met_matching(s, config)
    Fk = _met_matching(k, config)

    comm = CommutatorNode(SymbolNode("Fs"), SymbolNode("Fk"))
    comm = comm.eval({"Fs": Fs, "Fk": Fk})

    expected = _expected_met_met(s, k, config)

    assert comm.fragment.keys() == expected.fragment.keys()
    assert comm == expected


@pytest.mark.parametrize("s, k", [(0, 1), (1, 2), (2, 4), (0, 5), (1, 0)])
def test_commutator_mol_met(s, k, config):
    """Test [F_s, F_k] evaluates correclty where s is in the mol space and k is in the met space"""
    Fs = _mol_matching(s, config)
    Fk = _met_matching(k, config)

    comm = CommutatorNode(SymbolNode("Fs"), SymbolNode("Fk"))
    comm = comm.eval({"Fs": Fs, "Fk": Fk})

    expected = _expected_mol_met(s, k, config)

    assert comm.fragment.keys() == expected.fragment.keys()
    assert comm == expected


def test_commutator_feps_f0_vanishes(config, feps):
    """Test [F_eps, F_0] evaulates correctly"""
    F0 = _diagonal(config)

    comm = CommutatorNode(SymbolNode("Feps"), SymbolNode("F0"))
    comm = comm.eval({"Feps": feps, "F0": F0})

    expected = GanFragment({})

    assert comm.fragment.keys() == expected.fragment.keys()
    assert comm == expected


@pytest.mark.parametrize("s", [0, 1, 2, 3, 4, 5])
def test_commutator_feps_met(s, config, feps):
    """Test [F_eps, F_s] evaluates correclty where s is in the met space"""
    Fs = _met_matching(s, config)

    comm = CommutatorNode(SymbolNode("Feps"), SymbolNode("Fs"))
    comm = comm.eval({"Feps": feps, "Fs": Fs})

    expected = _expected_feps_met(s, config)

    assert comm.fragment.keys() == expected.fragment.keys()
    assert comm == expected


@pytest.mark.parametrize("s", [0, 1, 2])
def test_commutator_f0_mol(s, config):
    """Test [F_0, F_s] evaluates correctly where s is in the mol space"""
    F0 = _diagonal(config)
    Fs = _mol_matching(s, config)

    comm = CommutatorNode(SymbolNode("F0"), SymbolNode("Fs"))
    comm = comm.eval({"F0": F0, "Fs": Fs})

    expected = _expected_f0_mol(s, config)

    assert comm.fragment.keys() == expected.fragment.keys()
    assert comm == expected


@pytest.mark.parametrize("s", [0, 1, 2, 3, 4, 5])
def test_commutator_f0_met(s, config):
    """Test [F_0, F_s] evaluates correctly where s is in the met space"""
    F0 = _diagonal(config)
    Fs = _met_matching(s, config)

    comm = CommutatorNode(SymbolNode("F0"), SymbolNode("Fs"))
    comm = comm.eval({"F0": F0, "Fs": Fs})

    expected = _expected_f0_met(s, config)

    assert comm.fragment.keys() == expected.fragment.keys()
    assert comm == expected
