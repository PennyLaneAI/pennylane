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
:class:`~pennylane.qchem.basis_set.BasisFunction` object from standard basis sets such as STO-3G.
"""
# pylint: disable=too-few-public-methods, too-many-branches
from .basis_data import basis_sets, load_basisset


class BasisFunction:
    r"""Create a basis function object.

    A basis set is composed of a set of basis functions that are typically constructed as a linear
    combination of primitive Gaussian functions. For instance, a basis function in the STO-3G basis
    set is formed as

    .. math::

        \psi = a_1 G_1 + a_2 G_2 + a_3 G_3,

    where :math:`a` denotes the contraction coefficients and :math:`G` is a Gaussian function
    defined as

    .. math::

        G = x^l y^m z^n e^{-\alpha r^2}.

    Each Gaussian function is characterized by the angular momentum numbers :math:`(l, m, n)` that
    determine the type of the orbital, the exponent :math:`\alpha` and the position vector
    :math:`r = (x, y, z)`. These parameters and the contraction coefficients :math:`a` define
    atomic basis functions. Predefined values of the exponents and contraction coefficients for
    each atomic orbital of a given chemical element can be obtained from reference libraries such as
    the Basis Set Exchange `library <https://www.basissetexchange.org>`_.

    The basis function object created by the BasisFunction class stores all the basis set parameters
    including the angular momentum, exponents, positions and coefficients of the Gaussian functions.

    The basis function object can be easily passed to the functions that compute various types of
    integrals over such functions, e.g., overlap integrals, which are essential for Hartree-Fock
    calculations.

    Args:
        l (tuple[int]): angular momentum numbers of the basis function.
        alpha (array(float)): exponents of the primitive Gaussian functions
        coeff (array(float)): coefficients of the contracted Gaussian functions
        r (array(float)): positions of the Gaussian functions
    """

    def __init__(self, l, alpha, coeff, r):
        self.l = l
        self.alpha = alpha
        self.coeff = coeff
        self.r = r
        self.params = [self.alpha, self.coeff, self.r]


def atom_basis_data(name, atom, load_data=False):
    r"""Generate default basis set parameters for an atom.

    This function extracts the angular momentum, exponents, and contraction coefficients of
    Gaussian functions forming atomic orbitals for a given atom. These values are taken, by default,
    from the basis set data provided in :mod:`~pennylane.qchem.basis_data`. If `load_data = True`,
    the basis set data is loaded from the basis-set-exchange library.

    Args:
        name (str): name of the basis set
        atom (str): atomic symbol of the chemical element
        load_data (bool): flag to load data from the basis-set-exchange library

    Returns:
        list(tuple): tuple containing the angular momentum, the exponents and contraction
        coefficients of a basis function

    **Example**

    >>> params = atom_basis_data('sto-3g', 'H')
    >>> print(params)
    [((0, 0, 0), [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422])]
    """

    name = name.lower()

    s = [(0, 0, 0)]
    p = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # for px, py, pz, respectively
    # for dxy, dxz, dyz, dxx, dyy, dzz, respectively:
    d = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)]

    if load_data:
        basis = load_basisset(name, atom)
    else:
        basis = basis_sets[name][atom]

    params = []
    sp_count = 0
    for i, j in enumerate(basis["orbitals"]):
        if j == "S":
            params.append((s[0], basis["exponents"][i], basis["coefficients"][i]))
        if j == "SP":
            for term in j:
                if term == "S":
                    params.append(
                        (s[0], basis["exponents"][i], basis["coefficients"][i + sp_count])
                    )
                if term == "P":
                    for l in p:
                        params.append(
                            (l, basis["exponents"][i], basis["coefficients"][i + sp_count + 1])
                        )
            sp_count += 1
        if j == "P":
            for l in p:
                params.append((l, basis["exponents"][i], basis["coefficients"][i]))
        if j == "D":
            for l in d:
                params.append((l, basis["exponents"][i], basis["coefficients"][i]))
    return params


def mol_basis_data(name, symbols, load_data=False):
    r"""Generates default basis set parameters for a molecule.

    This function generates the default basis set parameters for a list of atomic symbols and
    computes the total number of basis functions for each atom.

    Args:
        name (str): name of the basis set
        symbols (list[str]): symbols of the atomic species in the molecule
        load_data (bool): flag to load data from the basis-set-exchange library

    Returns:
        tuple(list, tuple): the number of atomic basis functions and the basis set parameters for
        each atom in the molecule

    **Example**

    >>> n_basis, params = mol_basis_data('sto-3g', ['H', 'H'])
    >>> print(n_basis)
    [1, 1]
    >>> print(params)
    (((0, 0, 0), [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422]),
     ((0, 0, 0), [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422]))
    """
    n_basis = []
    basis_set = []
    for s in symbols:
        basis = atom_basis_data(name, s, load_data=load_data)
        n_basis += [len(basis)]
        basis_set += basis
    return n_basis, tuple(basis_set)
