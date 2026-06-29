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
r"""Construction of GAN Hamiltonian fragments from coefficient arrays.

This module turns a user-supplied description of a GAN Hamiltonian --- the
mode counts and the coefficient tensors collected in a :class:`GanConfig` ---
into a list of :class:`~.GanFragment` objects suitable for Trotter-error
analysis. The fragmentation groups the Hamiltonian's terms so that the
fragments can be exponentiated and recombined in a product formula.

The fragments produced by :func:`gan_fragments` are:

* a single *diagonal* fragment (on-site molecular couplings, electron repulsion,
  and the nuclear reference energy),
* one *molecular matching* fragment per molecular matching index, grouping
  hopping terms between molecular modes into commuting edge sets,
* one *metallic matching* fragment per metallic matching index, grouping
  molecule--metal transfer (hybridization) terms,
* a single *kinetic* fragment (nuclear kinetic energy and metallic on-site
  energies).

The matching construction is a graph-edge-colouring style decomposition: each
matching index selects a set of disjoint mode pairs (edges) whose corresponding
hopping terms mutually commute and can therefore live in the same fragment.

The private ``_molecular_coupling``, ``_electron_repulsion``,
``_molecule_metal_transfer``, and ``_nuclear_reference`` helpers translate the
coefficient tensors into :class:`~.GanCoeff` polynomials in the nuclear
position/momentum functions; each supports both a "full" tensor (a distinct
coefficient per mode tuple) and a "diagonal" tensor (one coefficient per mode).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Sequence

from numpy.typing import ArrayLike

from pennylane.labs.trotter_error.fragments.gan_fragments.fermi import FermiOp, FermiWord
from pennylane.labs.trotter_error.fragments.gan_fragments.gan_fragments import (
    FuncSymbol,
    GanCoeff,
    GanFragment,
    GanMonomial,
)


@dataclass
class GanConfig:
    r"""Dimensions and coefficient tensors defining a GAN Hamiltonian.

    Collects everything needed to build the GAN fragments. The coefficient
    sequences are indexed by polynomial order in the nuclear coordinates: entry
    ``order`` holds the rank-``order`` tensor of expansion coefficients (so,
    e.g., ``couplings[0]`` is the constant term, ``couplings[1]`` the linear
    term, and so on).

    Args:
        n_modes (int): the number of nuclear (vibrational) modes.
        n_met (int): the number of metallic modes.
        n_mol (int): the number of molecular modes.
        couplings (Sequence[ArrayLike]): the molecular coupling tensors
            :math:`U`, per polynomial order.
        repulsion (Sequence[ArrayLike]): the electron repulsion tensors
            :math:`V`, per polynomial order.
        transfer (Sequence[ArrayLike]): the molecule--metal transfer
            (hybridization) tensors :math:`W`, per polynomial order.
        nuclear (Sequence[ArrayLike]): the nuclear reference energy tensors
            :math:`U_0`, per polynomial order.
        masses (ArrayLike): the nuclear mode masses :math:`m`, shape
            ``(n_modes,)``.
        energies (ArrayLike): the metallic on-site energies
            :math:`\epsilon`, shape ``(n_met,)``.
    """

    n_modes: int
    n_met: int
    n_mol: int
    couplings: Sequence[ArrayLike]  ## U
    repulsion: Sequence[ArrayLike]  ## V
    transfer: Sequence[ArrayLike]  ## W
    nuclear: Sequence[ArrayLike]  ## U0
    masses: ArrayLike  ## m
    energies: ArrayLike  ## epsilon


def gan_fragments(config: GanConfig) -> list[GanFragment]:
    """Construct the fragments of a GAN Hamiltonian from its configuration.

    Validates the shapes of every coefficient tensor in ``config`` against the
    declared mode counts, then assembles the fragment list: the diagonal
    fragment, the molecular matching fragments, the metallic matching fragments,
    and the kinetic fragment (in that order).

    Args:
        config (GanConfig): the dimensions and coefficient tensors of the
            Hamiltonian.

    Returns:
        list[GanFragment]: the fragments whose sum is the full GAN Hamiltonian.

    Raises:
        TypeError: if ``couplings``, ``repulsion``, ``transfer``, or ``nuclear``
            is not a ``Sequence``.
        ValueError: if any coefficient tensor, the masses, or the energies has a
            shape inconsistent with the declared mode counts.
    """

    if not isinstance(config.couplings, Sequence):
        raise TypeError(
            f"Electron coupling coefficients must be Sequence type, got type {type(config.couplings)}."
        )

    if not isinstance(config.repulsion, Sequence):
        raise TypeError(
            f"Electron repulsion coefficients must be Sequence type, got type {type(config.repulsion)}."
        )

    if not isinstance(config.transfer, Sequence):
        raise TypeError(
            f"Electron transfer coefficients must be Sequence type, got type {type(config.nuclear)}."
        )

    if not isinstance(config.nuclear, Sequence):
        raise TypeError(
            f"Nuclear coordinates must be Sequence type, got type {type(config.nuclear)}."
        )

    for order, tensor in enumerate(config.couplings):
        full_shape = (config.n_mol, config.n_mol) + ((config.n_modes,) * order)
        diag_shape = (config.n_mol, config.n_mol) + (config.n_modes,)

        if tensor.shape == full_shape:
            continue

        if tensor.shape == diag_shape:
            continue

        raise ValueError(
            f"Electron coupling coefficients for order {order} must be shape {full_shape} or shape {diag_shape}, got shape {tensor.shape}."
        )

    for order, tensor in enumerate(config.repulsion):
        full_shape = (config.n_mol, config.n_mol) + ((config.n_modes,) * order)
        diag_shape = (config.n_mol, config.n_mol) + (config.n_modes,)

        if tensor.shape == full_shape:
            continue

        if tensor.shape == diag_shape:
            continue

        raise ValueError(
            f"Electron repulsion coefficients for order {order} must be shape {full_shape} or shape {diag_shape}, got shape {tensor.shape}."
        )

    for order, tensor in enumerate(config.nuclear):
        full_shape = (config.n_modes,) * order
        diag_shape = (config.n_modes,)

        if tensor.shape == full_shape:
            continue

        if tensor.shape == diag_shape:
            continue

        raise ValueError(
            f"Nuclear reference coefficients for order {order} must be shape {full_shape} or shape {diag_shape}, got shape {tensor.shape}."
        )

    for tensor in config.transfer:
        shape = (config.n_mol, config.n_met)

        if tensor.shape != shape:
            raise ValueError(
                f"Electron transfer coefficients for order {order} must be shape {shape}, got shape {tensor.shape}."
            )

    if config.masses.shape != (config.n_modes,):
        raise ValueError(
            f"Masses must be shape {(config.n_modes, )}, got shape {config.masses.shape}."
        )

    if config.energies.shape != (config.n_met,):
        raise ValueError(
            f"Energies must be shape {(config.n_met, )}, got shape {config.energies.shape}."
        )

    fragments = []
    fragments.append(_diagonal(config))

    offset = 1 - (config.n_mol % 2)
    for i in range(0, config.n_mol - offset):
        fragments.append(_mol_matching(i, config))

    for i in range(0, config.n_met):
        fragments.append(_met_matching(i, config))

    fragments.append(_kinetic(config))

    return fragments


def _diagonal(config: GanConfig) -> GanFragment:
    r"""Build the diagonal fragment of the GAN Hamiltonian.

    Collects the number-conserving on-site terms: the molecular self-couplings
    :math:`c_i^\dagger c_i`, the electron-repulsion terms
    :math:`c_i^\dagger c_i c_j^\dagger c_j`, and the (identity-fermionic)
    nuclear reference energy.

    Args:
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanFragment: the diagonal fragment.
    """
    terms = defaultdict(GanCoeff.identity)

    for i in range(config.n_mol):
        gan_coeff = _molecular_coupling(i, i, config)
        fermi = FermiWord([FermiOp.creation_mol(i), FermiOp.annihilation_mol(i)])
        terms[fermi] += gan_coeff

    for i, j in product(range(config.n_mol), repeat=2):
        gan_coeff = _electron_repulsion(i, j, config)
        fermi = FermiWord(
            [
                FermiOp.creation_mol(i),
                FermiOp.annihilation_mol(i),
                FermiOp.creation_mol(j),
                FermiOp.annihilation_mol(j),
            ]
        )
        terms[fermi] += gan_coeff

    gan_coeff = _nuclear_reference(config)
    terms[FermiWord.identity()] += gan_coeff

    return GanFragment(terms)


def _kinetic(config: GanConfig) -> GanFragment:
    r"""Build the kinetic fragment of the GAN Hamiltonian.

    Collects the nuclear kinetic energy :math:`P(i)^2 / (2 m_i)` (carried on the
    identity fermionic word) and the metallic on-site energies
    :math:`\epsilon_i\, c_i^\dagger c_i`.

    Args:
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanFragment: the kinetic fragment.
    """
    terms = defaultdict(GanCoeff.identity)

    for i, mass in enumerate(config.masses):
        func = FuncSymbol.momentum(i)
        monomial = GanMonomial([func])
        coeff = GanCoeff({monomial: 1 / (2 * mass)})
        fermi = FermiWord.identity()
        terms[fermi] += coeff

    for i, energy in enumerate(config.energies):
        monomial = GanMonomial.identity()
        coeff = GanCoeff({monomial: energy})
        fermi = FermiWord([FermiOp.creation_met(i), FermiOp.annihilation_met(i)])
        terms[fermi] += coeff

    return GanFragment(terms)


def _mol_matching(s: int, config: GanConfig) -> GanFragment:
    r"""Build the ``s``-th molecular matching fragment.

    Selects a matching (a set of disjoint molecular mode pairs) determined by
    the index ``s`` and collects the corresponding hopping terms
    :math:`c_i^\dagger c_j + c_j^\dagger c_i`, each weighted by its molecular
    coupling coefficient. The edges are generated symmetrically around ``s`` modulo
    the number of molecular modes, with a special wrap-around edge added when
    ``n_mol`` is even.

    Args:
        s (int): the molecular matching index selecting the edge set.
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanFragment: the molecular matching fragment for index ``s``.
    """
    t = config.n_mol % 2
    k = (config.n_mol + t) // 2

    edges = set()

    for i in range(1, k):
        u = (s + i) % (config.n_mol + t - 1)
        v = (s - i) % (config.n_mol + t - 1)
        edges.add((u, v))

    if t == 0:
        edges.add((s, config.n_mol - 1))

    terms = defaultdict(GanCoeff.identity)
    for i, j in edges:
        gan_coeff = _molecular_coupling(i, j, config)
        fermi1 = FermiWord([FermiOp.creation_mol(i), FermiOp.annihilation_mol(j)])
        fermi2 = FermiWord([FermiOp.creation_mol(j), FermiOp.annihilation_mol(i)])
        terms[fermi1] += gan_coeff
        terms[fermi2] += gan_coeff

    return GanFragment(terms)


def _met_matching(s: int, config: GanConfig):
    r"""Build the ``s``-th metallic matching fragment.

    Selects a matching that pairs each molecular mode ``i`` with the metallic
    mode ``(i + s) mod n_met`` and collects the corresponding molecule--metal
    transfer (hybridization) terms
    :math:`c^\dagger_{\text{mol},i} c_{\text{met},j} + c^\dagger_{\text{met},j} c_{\text{mol},i}`,
    each weighted by its transfer coefficient.

    Args:
        s (int): the metallic matching index (the molecule-to-metal offset).
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanFragment: the metallic matching fragment for index ``s``.
    """
    edges = set()

    for i in range(config.n_mol):
        u = i
        v = (i + s) % config.n_met
        edges.add((u, v))

    terms = defaultdict(GanCoeff.identity)
    for i, j in edges:
        gan_coeff = _molecule_metal_transfer(i, j, config)
        fermi1 = FermiWord([FermiOp.creation_mol(i), FermiOp.annihilation_met(j)])
        fermi2 = FermiWord([FermiOp.creation_met(j), FermiOp.annihilation_mol(i)])
        terms[fermi1] += gan_coeff
        terms[fermi2] += gan_coeff

    return GanFragment(terms)


def _molecular_coupling(i: int, j: int, config: GanConfig) -> GanCoeff:
    r"""Return the molecular coupling coefficient :math:`U_{ij}` as a polynomial.

    Builds the nuclear-function polynomial multiplying the molecular term on
    modes ``(i, j)`` by summing over the coupling tensors of every order. A
    "full" tensor contributes a position monomial per mode tuple; a "diagonal"
    tensor (orders above one) contributes a single position raised to that
    order, per mode.

    Args:
        i (int): the first molecular mode index.
        j (int): the second molecular mode index.
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanCoeff: the coupling coefficient as a linear combination of monomials.
    """
    monomials = defaultdict(float)

    for order, tensor in enumerate(config.couplings, start=1):
        full_shape = (config.n_mol, config.n_mol) + (config.n_modes,) * order
        diag_shape = (config.n_mol, config.n_mol, config.n_modes)

        if tensor.shape == full_shape:
            for modes in product(range(config.n_modes), repeat=order):
                monomial = GanMonomial([FuncSymbol.position(mode) for mode in modes])
                index = (i, j) + tuple(modes)
                coeff = tensor[index]
                monomials[monomial] += coeff

        if tensor.shape == diag_shape and order > 1:
            for mode in range(config.n_modes):
                monomial = GanMonomial([FuncSymbol.position(mode, order)])
                coeff = tensor[i, j, mode]
                monomials[monomial] += coeff

    return GanCoeff(monomials)


def _electron_repulsion(i: int, j: int, config: GanConfig) -> GanCoeff:
    r"""Return the electron repulsion coefficient :math:`V_{ij}` as a polynomial.

    Builds the nuclear-function polynomial multiplying the repulsion term on
    molecular modes ``(i, j)`` by summing over the repulsion tensors of every
    order, using the same full/diagonal tensor handling as
    :func:`_molecular_coupling`.

    Args:
        i (int): the first molecular mode index.
        j (int): the second molecular mode index.
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanCoeff: the repulsion coefficient as a linear combination of monomials.
    """
    monomials = defaultdict(float)

    for order, tensor in enumerate(config.repulsion):
        full_shape = (config.n_mol, config.n_mol) + (config.n_modes,) * order
        diag_shape = (config.n_mol, config.n_mol, config.n_modes)

        if tensor.shape == full_shape:
            for modes in product(range(config.n_modes), repeat=order):
                monomial = GanMonomial([FuncSymbol.position(mode) for mode in modes])
                index = (i, j) + tuple(modes)
                coeff = tensor[index]
                monomials[monomial] += coeff

        if tensor.shape == diag_shape and order > 1:
            for mode in range(config.n_modes):
                monomial = GanMonomial([FuncSymbol.position(mode, order)])
                coeff = tensor[i, j, mode]
                monomials[monomial] += coeff

    return GanCoeff(monomials)


def _molecule_metal_transfer(i: int, j: int, config: GanConfig) -> GanCoeff:
    r"""Return the molecule--metal transfer coefficient :math:`W_{ij}` as a polynomial.

    Builds the nuclear-function polynomial multiplying the hybridization term
    coupling molecular mode ``i`` to metallic mode ``j``. The zeroth-order
    tensor contributes a constant (identity) term; higher-order tensors each
    contribute a position function on the dedicated transfer mode
    (``n_modes - 1``) raised to the corresponding order.

    Args:
        i (int): the molecular mode index.
        j (int): the metallic mode index.
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanCoeff: the transfer coefficient as a linear combination of monomials.
    """
    trans_mode = config.n_modes - 1
    monomials = defaultdict(float)

    const = GanMonomial([FuncSymbol.identity()])
    monomials[const] += config.transfer[0][i, j]

    for order, tensor in enumerate(config.transfer[1:]):
        monomial = GanMonomial([FuncSymbol.position(trans_mode, order)])
        monomials[monomial] += tensor[i, j]

    return GanCoeff(monomials)


def _nuclear_reference(config: GanConfig) -> GanCoeff:
    r"""Return the nuclear reference energy :math:`U_0` as a polynomial.

    Builds the mode-only nuclear-function polynomial (no fermionic dependence)
    by summing over the nuclear reference tensors of every order, using the same
    full/diagonal tensor handling as the other coefficient builders.

    Args:
        config (GanConfig): the Hamiltonian configuration.

    Returns:
        GanCoeff: the nuclear reference coefficient as a linear combination of
        monomials.
    """
    monomials = defaultdict(float)

    for order, tensor in enumerate(config.nuclear):
        full_shape = (config.n_modes,) * order
        diag_shape = (config.n_modes,)

        if tensor.shape == full_shape:
            for modes in product(range(config.n_modes), repeat=order):
                monomial = GanMonomial([FuncSymbol.position(mode) for mode in modes])
                coeff = tensor[tuple(modes)]
                monomials[monomial] += coeff

        if tensor.shape == diag_shape and order > 1:
            for mode in range(config.n_modes):
                monomial = GanMonomial([FuncSymbol.position(mode, order)])
                coeff = tensor[mode]
                monomials[monomial] += coeff

    return GanCoeff(monomials)
