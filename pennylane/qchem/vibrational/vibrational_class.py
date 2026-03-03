# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions and classes to generate data needed to construct a
vibrational Hamiltonian for a given molecule."""

from dataclasses import dataclass

import numpy as np

from pennylane.qchem.openfermion_pyscf import _import_pyscf

# pylint: disable=import-outside-toplevel, unused-variable, too-many-instance-attributes, too-many-arguments


@dataclass
class VibrationalPES:
    r"""Data class to store information needed to construct a vibrational Hamiltonian for a molecule.

    Args:
        freqs (array[float]): normal-mode frequencies in atomic units
        grid (array[float]): grid points to compute potential energy surface data.
            Should be the sample points of the Gauss-Hermite quadrature.
        gauss_weights (array[float]): weights associate with each point in ``grid``.
            Should be the weights of the Gauss-Hermite quadrature.
        uloc (TensorLike[float]): normal mode localization matrix with shape ``(m, m)`` where
            ``m = len(freqs)``
        pes_data (list[TensorLike[float]]): list of one-mode, two-mode and three-mode potential
            energy surface data, with shapes ``(m, l)``, ``(m, m, l, l)`` ``(m, m, m, l, l, l)``,
            respectively, where ``m = len(freqs)`` and ``l > 0``
        dipole_data (list[TensorLike[float]]): list of  one-mode, two-mode and three-mode dipole
            moment data, with shapes ``(m, l, 3)``, ``(m, m, l, l, 3)`` ``(m, m, m, l, l, l, 3)``,
            respectively, where ``m = len(freqs)`` and ``l > 0``
        localized (bool): Whether the potential energy surface data correspond to localized normal
            modes. Default is ``True``.
        dipole_level (int): The level up to which dipole moment data are to be calculated. Input
            values can be ``1``, ``2``, or ``3`` for up to one-mode dipole, two-mode dipole and
            three-mode dipole, respectively. Default value is ``1``.

    **Example**

    This example shows how to construct the :class:`~.qchem.vibrational.VibrationalPES` object for a
    linear diatomic molecule, e.g., :math:`H_2`, with only one vibrational normal mode. The one-mode
    potential energy surface data is obtained by sampling ``9`` points along the normal mode, with
    grid points and weights that correspond to a Gauss-Hermite quadrature.

    >>> freqs = np.array([0.01885397])
    >>> grid, weights = np.polynomial.hermite.hermgauss(9)
    >>> pes_onemode = [[0.05235573, 0.03093067, 0.01501878, 0.00420778, 0.0,
    ...                 0.00584504, 0.02881817, 0.08483433, 0.22025702]]
    >>> vib_pes = qml.qchem.VibrationalPES(freqs=freqs, grid=grid,
    ...           gauss_weights=weights, pes_data=[pes_onemode])
    >>> vib_pes.freqs
    array([0.01885397])

    The following example shows how to construct the :class:`~.qchem.vibrational.VibrationalPES`
    object for a nonlinear triatomic molecule, e.g., :math:`H_3^+`, with three vibrational
    normal modes. We assume that the potential energy surface and dipole data are obtained by
    sampling ``5`` points along the normal mode, with grid points and weights that correspond to a
    Gauss-Hermite quadrature.

    >>> freqs = np.array([0.00978463, 0.00978489, 0.01663723])
    >>> grid, weights = np.polynomial.hermite.hermgauss(5)
    >>>
    >>> uloc = np.array([[-0.99098585,  0.13396657,  0.],
    ...                  [-0.13396657, -0.99098585,  0.],
    ...                  [ 0.        ,  0.        ,  1.]])
    >>>
    >>> pes_onemode = np.random.rand(3, 5)
    >>> pes_twomode = np.random.rand(3, 3, 5, 5)
    >>> pes_threemode = np.random.rand(3, 3, 3, 5, 5, 5)
    >>>
    >>> dipole_onemode = np.random.rand(3, 5, 3)
    >>> dipole_twomode = np.random.rand(3, 3, 5, 5, 3)
    >>> dipole_threemode = np.random.rand(3, 3, 3, 5, 5, 5, 3)
    >>>
    >>> localized = True
    >>> dipole_level = 3
    >>>
    >>> vib_obj = qml.qchem.VibrationalPES(freqs=freqs, grid=grid, gauss_weights=weights,
    ...           uloc=uloc, pes_data=[pes_onemode, pes_twomode, pes_threemode],
    ...           dipole_data=[dipole_onemode, dipole_twomode, dipole_threemode],
    ...           localized=True, dipole_level=3)
    >>> print(vib_obj.dipole_threemode.shape)
    (3, 3, 3, 5, 5, 5, 3)
    """

    def __init__(
        self,
        freqs=None,
        grid=None,
        gauss_weights=None,
        uloc=None,
        pes_data=None,
        dipole_data=None,
        localized=True,
        dipole_level=1,
    ):
        self.freqs = freqs
        self.grid = grid
        self.gauss_weights = gauss_weights
        self.uloc = uloc
        self.pes_onemode = pes_data[0] if pes_data else None
        self.pes_twomode = pes_data[1] if pes_data and len(pes_data) > 1 else None
        self.pes_threemode = pes_data[2] if pes_data and len(pes_data) > 2 else None
        self.dipole_onemode = dipole_data[0] if dipole_data else None
        self.dipole_twomode = dipole_data[1] if dipole_level >= 2 else None
        self.dipole_threemode = dipole_data[2] if dipole_level >= 3 else None
        self.localized = localized
        self.dipole_level = dipole_level


def _harmonic_analysis(scf_result, method="rhf"):
    r"""Performs harmonic analysis by evaluating the Hessian using PySCF routines.

    Args:
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        tuple: A tuple containing the following:
         - list[float]: normal mode frequencies in ``cm^-1``
         - TensorLike[float]: corresponding displacement vectors for each normal mode

    """
    pyscf = _import_pyscf()
    from pyscf.hessian import thermo

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method} is not available.")

    hess = getattr(pyscf.hessian, method).Hessian(scf_result).kernel()
    harmonic_res = thermo.harmonic_analysis(scf_result.mol, hess)

    return harmonic_res["freq_wavenumber"], harmonic_res["norm_mode"]


def _single_point(molecule, method="rhf"):
    r"""Runs electronic structure calculation.

    Args:
        molecule (:func:`~pennylane.qchem.molecule.Molecule`): Molecule object.
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        pyscf.scf object from electronic structure calculation

    """
    pyscf = _import_pyscf()

    method = method.strip().lower()
    if method not in ["rhf", "uhf"]:
        raise ValueError(f"Specified electronic structure method, {method}, is not available.")

    geom = [
        [symbol, tuple(np.array(molecule.coordinates)[i])]
        for i, symbol in enumerate(molecule.symbols)
    ]
    spin = int((molecule.mult - 1) / 2)
    mol = pyscf.gto.Mole(atom=geom, symmetry="C1", spin=spin, charge=molecule.charge, unit="Bohr")
    mol.basis = molecule.basis_name
    mol.build()
    if method == "rhf":
        scf_obj = pyscf.scf.RHF(mol).run(verbose=0)
    else:
        scf_obj = pyscf.scf.UHF(mol).run(verbose=0)
    return scf_obj


def _import_geometric():
    """Import geometric."""
    try:
        import geometric
    except ImportError as Error:
        raise ImportError(
            "This feature requires geometric. It can be installed with: pip install geometric."
        ) from Error

    return geometric


def optimize_geometry(molecule, method="rhf"):
    r"""Computes the equilibrium geometry of a molecule.

    Args:
        molecule (~qchem.molecule.Molecule): the molecule object
        method (str): Electronic structure method used to perform geometry optimization.
            Available options are ``"rhf"`` and ``"uhf"`` for restricted and unrestricted
            Hartree-Fock, respectively. Default is ``"rhf"``.

    Returns:
        array[array[float]]: optimized atomic positions in Cartesian coordinates

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, 0.0],
    ...                      [0.0, 0.0, 1.0]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> eq_geom = qml.qchem.optimize_geometry(mol)
    >>> eq_geom
    array([[ 0.        ,  0.        , -0.40277116],
           [ 0.        ,  0.        ,  1.40277116]])

    """
    pyscf = _import_pyscf()
    geometric = _import_geometric()
    from pyscf.geomopt.geometric_solver import optimize

    scf_res = _single_point(molecule, method)
    geom_eq = optimize(scf_res, maxsteps=100)

    if molecule.unit == "angstrom":
        return geom_eq.atom_coords(unit="A")
    return geom_eq.atom_coords(unit="B")


def _get_rhf_dipole(scf_result):
    """
    Given a restricted Hartree-Fock object, evaluate the dipole moment
    in the restricted Hartree-Fock state.

    Args:
        scf_result(pyscf.scf object): pyscf object from electronic structure calculations

    Returns:
        TensorLike[float]: dipole moment
    """

    charges = scf_result.mol.atom_charges()
    coords = scf_result.mol.atom_coords()
    masses = scf_result.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum("z,zx->x", masses, coords) / masses.sum()
    scf_result.mol.set_common_orig_(nuc_mass_center)
    dip_ints = scf_result.mol.intor("int1e_r", comp=3)

    t_dm1 = scf_result.make_rdm1()
    if len(t_dm1.shape) == 3:
        dipole_e_alpha = np.einsum("xij,ji->x", dip_ints, t_dm1[0, ::])
        dipole_e_beta = np.einsum("xij,ji->x", dip_ints, t_dm1[1, ::])
        dipole_e = dipole_e_alpha + dipole_e_beta
    else:
        dipole_e = np.einsum("xij,ji->x", dip_ints, t_dm1)

    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom, :] -= nuc_mass_center
    dipole_n = np.einsum("z,zx->x", charges, centered_coords)
    dipole = -dipole_e + dipole_n
    return dipole


def _get_uhf_dipole(scf_result):
    """
    Given an unrestricted Hartree-Fock object, evaluate the dipole moment
    in the unrestricted Hartree-Fock state.

    Args:
        scf_result(pyscf.scf object): pyscf object from electronic structure calculations

    Returns:
        TensorLike[float]: dipole moment

    """

    charges = scf_result.mol.atom_charges()
    coords = scf_result.mol.atom_coords()
    masses = scf_result.mol.atom_mass_list(isotope_avg=True)
    nuc_mass_center = np.einsum("z,zx->x", masses, coords) / masses.sum()
    scf_result.mol.set_common_orig_(nuc_mass_center)

    t_dm1_alpha, t_dm1_beta = scf_result.make_rdm1()

    dip_ints = scf_result.mol.intor("int1e_r", comp=3)
    dipole_e_alpha = np.einsum("xij,ji->x", dip_ints, t_dm1_alpha)
    dipole_e_beta = np.einsum("xij,ji->x", dip_ints, t_dm1_beta)
    dipole_e = dipole_e_alpha + dipole_e_beta

    centered_coords = np.copy(coords)
    for num_atom in range(len(charges)):
        centered_coords[num_atom, :] -= nuc_mass_center
    dipole_n = np.einsum("z,zx->x", charges, centered_coords)

    dipole = -dipole_e + dipole_n
    return dipole


def _get_dipole(scf_result, method):
    r"""Evaluate the dipole moment for a Hartree-Fock state.

    Args:
        scf_result (pyscf.scf object): pyscf object from electronic structure calculations
        method (str): Electronic structure method that can be either restricted and unrestricted
            Hartree-Fock,  ``'rhf'`` and ``'uhf'``, respectively. Default is ``'rhf'``.

    Returns:
        TensorLike[float]: dipole moment

    """
    method = method.strip().lower()
    if method == "rhf":
        return _get_rhf_dipole(scf_result)

    return _get_uhf_dipole(scf_result)
