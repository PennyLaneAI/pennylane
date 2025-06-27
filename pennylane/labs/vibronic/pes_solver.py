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
"""This module contains functions to calculate potential energy surfaces
per normal modes on a grid."""

import numpy as np
import scipy as sp

from pathlib import Path
from tempfile import TemporaryDirectory

import pyscf
from pyscf.hessian import thermo

from scipy.spatial.transform import Rotation

from itertools import combinations

from pyscf import gto, scf, mcscf, tdscf, cc
from pyscf.fci import direct_spin0
from pyscf.symm.geom import SymmSys

from pennylane import concurrency, qchem

from pennylane.qchem.vibrational.vibrational_class import (
    VibrationalPES,
    optimize_geometry,
)


def _run_casscf(
    symbols,
    coords,
    ncas,
    nelecas,
    basis="6-31g*",
    max_macro=50,
    max_micro=20,
    conv_tol=1e-7,
    nroots=2,
    restrict_to_singlet=True,
    spin=0,
    point_group=None,
):
    """
    Run CASSCF calculation and return energies.

    Args:


    Returns:
        List of CASSCF energies for each root
    """

    coords = [[symbol, tuple(np.array(coords)[i])] for i, symbol in enumerate(symbols)]

    mol = gto.Mole()
    mol.atom = coords
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = spin
    if point_group:
        mol.symmetry = point_group.lower()
        mol.symmetry_subgroup = point_group.lower()

    mol.build()

    # Run RHF/UHF based on spin
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    mf.kernel()

    # Run CASSCF
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = conv_tol

    # Set spin restriction if requested
    if restrict_to_singlet:
        if mol.spin != 0:
            raise ValueError("Cannot restrict to singlet for open-shell system")
        mc.fcisolver = direct_spin0.FCISolver(mol)
        mc.fcisolver.spin = 0
        mc.fix_spin_(shift=0.2, ss=0)
    else:
        # For open-shell systems, use appropriate FCI solver
        if mol.spin != 0:
            from pyscf.fci import direct_spin1

            mc.fcisolver = direct_spin1.FCISolver(mol)
            mc.fcisolver.spin = mol.spin

    # Compute multiple roots if requested
    if nroots > 1:
        weights = [1.0 / nroots] * nroots
        mc.fcisolver.nroots = nroots
        mc = mc.state_average_(weights)
        mc.verbose = 3
        mc.kernel()
        if not mc.converged:
            raise ValueError("The CASSCF did NOT converge.")
        # CASSCF energies for all states:
        energies = mc.e_states
    else:
        energies = [mc.kernel()[0]]

    return energies


def _run_tddft(
    symbols,
    coords,
    basis="6-31g*",
    functional="cam-b3lyp",
    nroots=2,
    conv_tol=1e-7,
    max_cycle=50,
    restrict_to_singlet=True,
    spin=0,
    point_group=None,
):
    """
    Run TDDFT calculation and return energies.

    Args:
        mol: PySCF Mole object

    Returns:
        List of TDDFT energies in Hartree, including ground state
    """

    coords = [[symbol, tuple(np.array(coords)[i])] for i, symbol in enumerate(symbols)]

    mol = gto.Mole()
    mol.atom = coords
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = spin
    if point_group:
        mol.symmetry = point_group.lower()
        mol.symmetry_subgroup = point_group.lower()

    mol.build()

    # Run DFT based on spin
    if mol.spin == 0:
        mf = scf.RKS(mol)
    else:
        mf = scf.UKS(mol)
    mf.xc = functional
    mf.kernel()

    # Get ground state energy
    gs_energy = mf.e_tot

    # Run TDDFT
    td = tdscf.TDA(mf)
    td.nstates = nroots - 1
    td.conv_tol = conv_tol
    td.max_cycle = max_cycle

    # Set spin restriction if requested
    if restrict_to_singlet:
        if mol.spin != 0:
            raise ValueError("Cannot restrict to singlet for open-shell system")
        td.singlet = True
    else:
        td.singlet = False

    # Run calculation
    td.kernel()
    td.analyze(verbose=4)

    # Get excitation energies (in eV) and convert to Hartree
    excitation_energies = td.e / 27.2114

    # Add ground state energy to get total energies
    energies = [gs_energy] + [gs_energy + e for e in excitation_energies]

    return energies


def _run_eom_ccsd(
    symbols,
    coords,
    basis="6-31g*",
    nroots=2,
    conv_tol=1e-7,
    conv_tol_eom=1e-6,
    max_cycle=50,
    max_cycle_eom=20,
    restrict_to_singlet=True,
    spin=0,
    point_group=None,
    frozen=None,
):
    """
    Run EOM-CCSD calculation and return energies.

    Args:
        mol: PySCF Mole object

    Returns:
        List of EOM-CCSD energies in Hartree, including ground state
    """
    coords = [[symbol, tuple(np.array(coords)[i])] for i, symbol in enumerate(symbols)]

    mol = gto.Mole()
    mol.atom = coords
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = spin
    if point_group:
        mol.symmetry = point_group.lower()
        mol.symmetry_subgroup = point_group.lower()

    mol.build()

    # Run HF based on spin
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)

    mf.kernel()

    # Get ground state energy
    gs_energy = mf.e_tot

    # Run CCSD
    if mol.spin == 0:
        mycc = cc.RCCSD(mf)
    else:
        mycc = cc.UCCSD(mf)

    # Set convergence parameters
    mycc.conv_tol = conv_tol
    mycc.max_cycle = max_cycle

    # Set frozen orbitals if specified
    if frozen is not None:
        mycc.frozen = frozen

    mycc.kernel()

    if not mycc.converged:
        raise RuntimeError("CCSD calculation did not converge")

    # Run EOM-CCSD
    if mol.spin == 0:
        energies_eomcc, cvecs_eomcc = mycc.eomee_ccsd_singlet(nroots=nroots - 1)
    else:
        energies_eomcc, cvecs_eomcc = cc.EOM(mycc)  # placeholder

    # Get excitation energies and convert to total energies
    excitation_energies = energies_eomcc

    energies = [mycc.e_tot] + [gs_energy + e for e in [excitation_energies]]

    return energies
