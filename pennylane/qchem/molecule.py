# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains functions and classes to create a
:class:`~pennylane.qchem.molecule.Molecule` object. This object stores all
the necessary information to perform a Hartree-Fock calculation for a given molecule.
"""
import collections

# pylint: disable=too-many-arguments,too-many-instance-attributes
import itertools
import warnings

import pennylane as qml

from .basis_data import atomic_numbers
from .basis_set import BasisFunction, mol_basis_data
from .integrals import contracted_norm, primitive_norm

# Bohr-Angstrom correlation coefficient (https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0)
bohr_angs = 0.529177210903


class Molecule:
    r"""Create a molecule object that stores molecular information and default basis set parameters.

    The molecule object can be passed to functions that perform a Hartree-Fock calculation.

    Args:
        symbols (list[str]): Symbols of the atomic species in the molecule. Currently, atoms with
            atomic numbers 1-10 are supported.
        coordinates (array[float]): 1D array with the atomic positions in Cartesian coordinates. The
            coordinates must be given in atomic units and the size of the array should be ``3*N``
            where ``N`` is the number of atoms.
        charge (int): net charge of the molecule
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
        basis_name (str): Atomic basis set used to represent the molecular orbitals. Currently, the
            only supported basis sets are ``STO-3G``, ``6-31G``, ``6-311G`` and ``CC-PVDZ``. Other
            basis sets can be loaded from the basis-set-exchange library using ``load_data``.
        load_data (bool): flag to load data from the basis-set-exchange library
        l (tuple[int]): angular momentum quantum numbers of the basis function
        alpha (array[float]): exponents of the primitive Gaussian functions
        coeff (array[float]): coefficients of the contracted Gaussian functions
        r (array[float]): positions of the Gaussian functions
        normalize (bool): if True, the basis functions get normalized
        unit (str): unit of atomic coordinates. Available options are ``unit="bohr"`` and ``unit="angstrom"``.

    .. note::
        :class:`~.qchem.Molecule` is not currently compatible with :func:`~.qjit` and ``jax.jit``.

    **Example**

    Import necessary modules:

    >>> from pennylane import numpy as np
    >>> from pennylane.qchem import Molecule

    Define molecular symbols and geometry:

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, -0.694349],
    ...                      [0.0, 0.0,  0.694349]], requires_grad = True)
    >>> mol = Molecule(symbols, geometry)
    >>> print(mol.n_electrons)
    2

    """

    def __init__(
        self,
        symbols,
        coordinates,
        charge=0,
        mult=1,
        basis_name="sto-3g",
        name="molecule",
        load_data=False,
        l=None,
        alpha=None,
        coeff=None,
        normalize=True,
        unit="bohr",
    ):
        if (
            basis_name.lower()
            not in [
                "sto-3g",
                "6-31g",
                "6-311g",
                "cc-pvdz",
            ]
            and load_data is False
        ):
            raise ValueError(
                "Currently, the only supported basis sets are 'sto-3g', '6-31g', '6-311g' and "
                "'cc-pvdz'. Please consider using `load_data=True` to download the basis set from "
                "an external library that can be installed with: pip install basis-set-exchange."
            )

        if set(symbols) - set(atomic_numbers):
            raise ValueError(f"Atoms in {set(symbols) - set(atomic_numbers)} are not supported.")

        self.symbols = symbols
        self.coordinates = coordinates
        self.unit = unit.strip().lower()
        self.charge = charge
        self.mult = mult
        self.basis_name = basis_name.lower()
        self.name = name
        self.load_data = load_data
        self.n_basis, self.basis_data = mol_basis_data(self.basis_name, self.symbols, load_data)

        if self.unit not in ("angstrom", "bohr"):
            raise ValueError(
                f"The provided unit '{unit}' is not supported. "
                f"Please set 'unit' to 'bohr' or 'angstrom'."
            )

        if self.unit == "angstrom":
            self.coordinates = self.coordinates / bohr_angs

        self.nuclear_charges = [atomic_numbers[s] for s in self.symbols]

        self.n_electrons = sum(self.nuclear_charges) - self.charge

        if l is None:
            l = [i[0] for i in self.basis_data]

        use_jax = any(qml.math.get_interface(x) == "jax" for x in [coordinates, alpha, coeff])
        interface_args = [{"like": "autograd", "requires_grad": False}, {"like": "jax"}][use_jax]
        if alpha is None:
            alpha = [qml.math.array(i[1], **interface_args) for i in self.basis_data]

        if coeff is None:
            coeff = [qml.math.array(i[2], **interface_args) for i in self.basis_data]

            if normalize:
                coeff = [
                    qml.math.array(c * primitive_norm(l[i], alpha[i]), **interface_args)
                    for i, c in enumerate(coeff)
                ]

        if (
            len(
                {
                    qml.math.get_deep_interface(x)
                    for x in [coordinates, alpha, coeff]
                    if qml.math.get_deep_interface(x) != "numpy"
                }
            )
            > 1
        ):
            warnings.warn(
                "The parameters coordinates, coeff, and alpha are not of the same interface. Please use the same interface for all 3 or there may be unintended behavior.",
                UserWarning,
            )

        r = list(
            itertools.chain(
                *[[self.coordinates[i]] * self.n_basis[i] for i in range(len(self.n_basis))]
            )
        )

        self.l = l
        self.alpha = alpha
        self.coeff = coeff
        self.r = r

        self.basis_set = [
            BasisFunction(self.l[i], self.alpha[i], self.coeff[i], self.r[i]) for i in range(len(l))
        ]
        self.n_orbitals = len(self.l)

        self.mo_coefficients = None
        self.normalize = normalize

    def __repr__(self):
        """Returns the molecule representation in string format"""

        elements, counter, flags = set(self.symbols), collections.Counter(self.symbols), []
        if counter["C"]:  # Hill Notation
            flags = ["C", "H"] if counter["H"] else ["C"]
        ordered_elems = flags + list(sorted(elements.difference(set(flags))))
        formula = "".join([x + str(counter[x]) if counter[x] > 1 else x for x in ordered_elems])

        return f"<Molecule = {formula}, Charge: {self.charge}, Basis: {self.basis_name.upper()}, Orbitals: {self.n_orbitals}, Electrons: {self.n_electrons}>"

    def atomic_orbital(self, index):
        r"""Return a function that evaluates an atomic orbital at a given position.

        Args:
            index (int): index of the atomic orbital, order follwos the order of atomic symbols

        Returns:
            function: function that computes the value of the orbital at a given position

        **Example**

        >>> symbols  = ['H', 'H']
        >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
        >>> mol = qml.qchem.Molecule(symbols, geometry)
        >>> ao = mol.atomic_orbital(0)
        >>> ao(0.0, 0.0, 0.0)
        0.62824688
        """
        l = self.basis_set[index].l
        alpha = self.basis_set[index].alpha
        coeff = self.basis_set[index].coeff
        r = self.basis_set[index].r

        coeff = coeff * contracted_norm(l, alpha, coeff)

        lx, ly, lz = l

        def orbital(x, y, z):
            r"""Evaluate a basis function at a given position.

            Args:
                x (float): x component of the position
                y (float): y component of the position
                z (float): z component of the position

            Returns:
                array[float]: value of a basis function
            """
            c = ((x - r[0]) ** lx) * ((y - r[1]) ** ly) * ((z - r[2]) ** lz)
            e = qml.math.exp(
                qml.math.tensordot(
                    -alpha, (x - r[0]) ** 2 + (y - r[1]) ** 2 + (z - r[2]) ** 2, axes=0
                )
            )
            return c * qml.math.dot(coeff, e)

        return orbital

    def molecular_orbital(self, index):
        r"""Return a function that evaluates a molecular orbital at a given position.

        Args:
            index (int): index of the molecular orbital

        Returns:
            function: function to evaluate the molecular orbital

        **Example**

        >>> symbols  = ['H', 'H']
        >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
        >>> mol = qml.qchem.Molecule(symbols, geometry)
        >>> qml.qchem.scf(mol)() # run scf to obtain the optimized molecular orbitals
        >>> mo = mol.molecular_orbital(1)
        >>> mo(0.0, 0.0, 0.0)
        0.01825128
        """
        # molecular coefficients are set by other modules
        c = self.mo_coefficients[index]  # pylint:disable=unsubscriptable-object

        def orbital(x, y, z):
            r"""Evaluate a molecular orbital at a given position.

            Args:
                x (float): x component of the position
                y (float): y component of the position
                z (float): z component of the position

            Returns:
                array[float]: value of a molecular orbital
            """
            m = 0.0
            for i in range(self.n_orbitals):
                m = m + c[i] * self.atomic_orbital(i)(x, y, z)
            return m

        return orbital
