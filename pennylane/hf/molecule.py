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
import itertools

# pylint: disable=too-few-public-methods, too-many-arguments, too-many-instance-attributes
import numpy as np
from pennylane import numpy as pnp
from pennylane.hf.basis_data import atomic_numbers
from pennylane.hf.basis_set import BasisFunction, mol_basis_data, atom_basis_data
from pennylane.hf.integrals import primitive_norm, contracted_norm

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Colormap as cmap

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
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals. Possible
            values of ``mult`` are :math:`1, 2, 3, \ldots`.
        basis_name (str): Atomic basis set used to represent the molecular orbitals. Currently, the
            only supported basis set is 'sto-3g'.
        l (tuple[int]): angular momentum quantum numbers of the basis function
        alpha (array[float]): exponents of the primitive Gaussian functions
        coeff (array[float]): coefficients of the contracted Gaussian functions
        r (array[float]): positions of the Gaussian functions

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = pnp.array([[0.0, 0.0, -0.694349],
    >>>                       [0.0, 0.0,  0.694349]], requires_grad = True)
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
        l=None,
        alpha=None,
        coeff=None,
        r=None,
    ):

        if basis_name not in ["sto-3g", "STO-3G"]:
            raise ValueError("Currently, the only supported basis set is 'sto-3g'.")

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

        self.nuclear_charges = [atomic_numbers[s] for s in self.symbols]

        self.n_electrons = sum(np.array(self.nuclear_charges)) - self.charge

    def get_atomic_orbital(self,i_atom):
        '''
        input : i_atom atom index for the ATOMIC orbital

        output : ATOMIC orbital f(x,y,z)
        '''

        atom_symbol = self.symbols[i_atom]
        r = self.r[i_atom]
        basis_data = atom_basis_data(self.basis_name, atom_symbol)
        lmn = basis_data[0][0]
        coeff = basis_data[0][1]
        alpha = basis_data[0][2]

        N = [primitive_norm(lmn,a) for a in alpha]
        C = [n * c for n, c in zip(N, coeff)]

        x0, y0, z0 = r[0], r[1], r[2]
        norm = contracted_norm(lmn, np.asarray(alpha), np.asarray(C))

        l, m, n = lmn
#         x, y, z = r[0], r[1], r[2]
        def f_orbital(x,y,z):
            ang = ((x - x0) ** l) * ((y - y0) ** m) * ((z - z0) ** n)
            val = ang * np.dot(np.array(C), np.array([np.exp(-a * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)) for a in alpha]))
            return norm * val
        return f_orbital

    def get_molecular_orbital(self,i_mo,M=None):
        '''
        input : i_atom atom index for the ATOMIC orbital
                M linear combination for the atomic orbitals

        output : MOLECULAR orbital f(x,y,z)
        '''

        if i_mo > self.n_orbitals:
            print('The molecular orbital number does not exists!')
            i_mo = self.n_orbitals


        if not M.any(): # RANDOM WEIGHTS
            M = np.random.rand(self.n_orbitals)
            M = M/np.linalg.norm(M)
        # TODO
        # READ THE LINEAR COMBINATION FROM THE HF CALCULATIONS
        # else:
            # M = molecule.D[i_mo]

        def f_orbital(x,y,z):
            mo = 0.
            for i in range(self.n_orbitals):
                mo_t = self.get_atomic_orbital(i)
                mo = mo + M[i]*mo_t(x,y,z)
            return mo
        return f_orbital
