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
This module contains functions and classes to create a Molecule object. This object stores all
the necessary information to perform a Hartree-Fock calculation for a given molecule.
"""
# pylint: disable=too-few-public-methods, too-many-arguments, too-many-instance-attributes
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.hf.basis_data import atomic_numbers
from pennylane.hf.basis_set import BasisFunction, mol_basis_data


class Molecule:
    r"""Create a molecule object that stores molecular information and default basis set parameters.

    The molecule object can be passed to functions that perform a Hartree-Fock calculation.

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian coordinates. The
            coordinates must be given in atomic units and the size of the array should be ``3*N``
             where ``N`` is the number of atoms.
        charge (int): Net charge of the molecule. If not specified, a neutral molecule is assumed.
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals. Possible
            values of ``mult`` are :math:`1, 2, 3, \ldots`. If not specified, a closed-shell HF
            state is assumed.
        basis_name(str): Atomic basis set used to represent the molecular orbitals.
        active_electrons (int): Number of active electrons. If not specified, all electrons are
            considered to be active.
        active_orbitals (int): Number of active orbitals. If not specified, all orbitals are
            considered to be active.
        l (tuple[int]): angular momentum numbers of the basis function.
        alpha (array(float)): exponents of the primitive Gaussian functions
        coeff (array(float)): coefficients of the contracted Gaussian functions
        r (array(float)): positions of the Gaussian functions
    """

    def __init__(
        self,
        symbols,
        coordinates,
        charge=0,
        mult=1,
        basis_name="sto-3g",
        active_electrons=None,
        active_orbitals=None,
        l=None,
        alpha=None,
        coeff=None,
        r=None,
    ):

        if basis_name not in ["sto-3g", "STO-3G"]:
            raise ValueError("The only supported basis set is 'sto-3g'.")

        if set(symbols) - set(atomic_numbers):
            raise ValueError(f"Atoms in {set(symbols) - set(atomic_numbers)} are not supported.")

        self.symbols = symbols
        self.coordinates = coordinates
        self.charge = charge
        self.mult = mult
        self.basis_name = basis_name

        self.n_basis, self.basis_data = mol_basis_data(self.basis_name, self.symbols)

        if l is None:
            l = [i[0] for i in self.basis_data]

        if alpha is None:
            alpha = [pnp.array(i[1], requires_grad=False) for i in self.basis_data]

        if coeff is None:
            coeff = [pnp.array(i[2], requires_grad=False) for i in self.basis_data]

        if r is None:
            r = sum([[self.coordinates[i]] * self.n_basis[i] for i in range(len(self.n_basis))], [])

        self.l = l
        self.alpha = alpha
        self.coeff = coeff
        self.r = r

        self.basis_set = generate_basis_set(self.l, self.alpha, self.coeff, self.r)
        self.n_orbitals = len(self.l)

        self.nuclear_charges = generate_nuclear_charges(self.symbols)

        self.n_electrons = sum(np.array(self.nuclear_charges)) - self.charge
        self.core, self.active = qml.qchem.active_space(
            self.n_electrons,
            self.n_orbitals,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )


def generate_basis_set(l, alpha, coeff, r):
    r"""Generate a set of basis function objects.

    Args:
        l list((tuple[int])): angular momentum numbers of the basis function.
        alpha list((array(float))): exponents of the Gaussian functions forming basis functions
        coeff list((array(float))): coefficients of the contracted Gaussian functions
        r list((array(float))): positions of the Gaussian functions forming the basis functions

    Returns:
        list(BasisFunction): list containing a set of basis function objects.

    **Example**

    >>> l = [(0, 0, 0), (0, 0, 0)]
    >>> exponents = [[3.42525091, 0.62391373, 0.1688554], [3.42525091, 0.62391373, 0.1688554]]
    >>> coefficients = [[0.15432897, 0.53532814, 0.44463454], [0.15432897, 0.53532814, 0.44463454]]
    >>> centers = [[0.0, 0.0, -0.694349], [0.0, 0.0, 0.694349]]
    >>>  basis_set = generate_basis_set(l, exponents, coefficients, centers)
    >>> print(basis_set)
    [<molecule.BasisFunction object at 0x7f7566db2910>, <molecule.BasisFunction object at 0x7f7566db2a30>]
    """
    return [BasisFunction(l[i], alpha[i], coeff[i], r[i]) for i in range(len(l))]


def generate_nuclear_charges(symbols):
    r"""Generate a list of atomic nuclear charges.

    Args:
    symbols (list[str]): symbols of the atomic species in the molecule

    Returns:
        list(int): list containing atomic nuclear charges.

    **Example**

    >>> symbols = ["H", "F"]
    >>> z = generate_nuclear_charges(symbols)
    >>> print(z)
    [1, 9]
    """
    return [atomic_numbers[s] for s in symbols]
