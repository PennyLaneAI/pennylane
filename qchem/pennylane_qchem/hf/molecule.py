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
This module contains functions and classes to create a Molecule object. This object contains all
the necessary information to perform a Hartree-Fock calculation for a given molecule.
"""

import numpy as np
import pennylane as qml
# from basis_data import atomic_numbers
# from basis_data import STO3G
import autograd.numpy as anp
from pennylane import numpy as pnp


class Molecule:
    r"""Create a molecule object that stores molecular information and default basis set parameters.

    The molecule object can be passed to functions that perform Hartree-Fock calculations.

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
        alpha (array[float]): exponents of the Gaussian functions forming the basis function
        coeff (array[float]): coefficients of the contracted Gaussian functions
        rgaus (array[float]): positions of the Gaussian functions forming the basis function
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
        alpha = None,
        coeff = None,
        rgaus = None
    ):

        if basis_name not in ["sto-3g", "STO-3G"]:
            raise ValueError("The only supported basis set is 'sto-3g'.")

        self.symbols = symbols
        self.coordinates = coordinates
        self.charge = charge
        self.mult = mult
        self.basis_name = basis_name

        # self.n_basis, self.basis_data = mol_basis_data(self.basis_name, self.symbols)
        #
        # self.l = [i[0] for i in self.basis_data]
        #
        # if alpha is None:
        #     alpha = [pnp.array(i[1], requires_grad=False) for i in self.basis_data]
        #
        # if coeff is None:
        #     coeff = [pnp.array(i[2], requires_grad=False) for i in self.basis_data]
        #
        # if rgaus is None:
        #     r_atom = [i for i in self.coordinates]
        #     rgaus = sum([[r_atom[i]] * self.n_basis[i] for i in range(len(self.n_basis))], [])

        # self.alpha = alpha
        # self.coeff = coeff
        # self.rgaus = rgaus

        # self.basis_set = generate_basis_functions(self.l, alpha, coeff, rgaus)

        # self.n_orbitals = len(self.l)
        #
        # self.nuclear_charges = [atomic_numbers[s] for s in symbols]
        #
        # self.n_electrons = sum(np.array(self.nuclear_charges))

        # self.core, self.active = qml.qchem.active_space(
        #     self.n_electrons,
        #     self.n_orbitals,
        #     active_electrons=active_electrons,
        #     active_orbitals=active_orbitals
        # )
        #
        # self.wires = [i for i in range(len(self.active * 2))]
