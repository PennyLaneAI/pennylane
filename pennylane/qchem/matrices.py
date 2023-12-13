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
This module contains the functions needed for computing matrices.
"""
# pylint: disable= too-many-branches
import itertools as it

import numpy as np
import pennylane as qml

from .integrals import (
    attraction_integral,
    kinetic_integral,
    moment_integral,
    overlap_integral,
    repulsion_integral,
)


def mol_density_matrix(n_electron, c):
    r"""Compute the molecular density matrix.

    The density matrix :math:`P` is computed from the molecular orbital coefficients :math:`C` as

    .. math::

        P_{\mu \nu} = \sum_{i=1}^{N} C_{\mu i} C_{\nu i},

    where :math:`N = N_{electrons} / 2` is the number of occupied orbitals. Note that the total
    density matrix is the sum of the :math:`\alpha` and :math:`\beta` density
    matrices, :math:`P = P^{\alpha} + P^{\beta}`.

    Args:
        n_electron (integer): number of electrons
        c (array[array[float]]): molecular orbital coefficients

    Returns:
        array[array[float]]: density matrix

    **Example**

    >>> c = np.array([[-0.54828771,  1.21848441], [-0.54828771, -1.21848441]])
    >>> n_electron = 2
    >>> mol_density_matrix(n_electron, c)
    array([[0.30061941, 0.30061941], [0.30061941, 0.30061941]])
    """
    p = qml.math.dot(c[:, : n_electron // 2], qml.math.conjugate(c[:, : n_electron // 2]).T)
    return p


def overlap_matrix(basis_functions):
    r"""Return a function that computes the overlap matrix for a given set of basis functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions

    Returns:
        function: function that computes the overlap matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> overlap_matrix(mol.basis_set)(*args)
    array([[1.0, 0.7965883009074122], [0.7965883009074122, 1.0]])
    """

    def overlap(*args):
        r"""Construct the overlap matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the overlap matrix
        """
        n = len(basis_functions)
        matrix = qml.math.eye(n)

        for (i, a), (j, b) in it.combinations(enumerate(basis_functions), r=2):
            args_ab = []
            if args:
                args_ab.extend([arg[i], arg[j]] for arg in args)
            integral = overlap_integral(a, b, normalize=False)(*args_ab)

            o = qml.math.zeros((n, n))
            o[i, j] = o[j, i] = 1.0
            matrix = matrix + integral * o

        return matrix

    return overlap


def moment_matrix(basis_functions, order, idx):
    r"""Return a function that computes the multipole moment matrix for a set of basis functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions
        order (integer): exponent of the position component
        idx (integer): index determining the dimension of the multipole moment integral

    Returns:
        function: function that computes the multipole moment matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> order, idx = 1, 0
    >>> moment_matrix(mol.basis_set, order, idx)(*args)
    tensor([[0.0, 0.4627777], [0.4627777, 2.0]], requires_grad=True)
    """

    def _moment_matrix(*args):
        r"""Construct the multipole moment matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the multipole moment matrix
        """
        n = len(basis_functions)
        matrix = qml.math.zeros((n, n))

        for (i, a), (j, b) in it.combinations_with_replacement(enumerate(basis_functions), r=2):
            args_ab = []
            if args:
                args_ab.extend([arg[i], arg[j]] for arg in args)
            integral = moment_integral(a, b, order, idx, normalize=False)(*args_ab)

            o = qml.math.zeros((n, n))
            o[i, j] = o[j, i] = 1.0
            matrix = matrix + integral * o

        return matrix

    return _moment_matrix


def kinetic_matrix(basis_functions):
    r"""Return a function that computes the kinetic matrix for a given set of basis functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions

    Returns:
        function: function that computes the kinetic matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> kinetic_matrix(mol.basis_set)(*args)
    array([[0.76003189, 0.38325367], [0.38325367, 0.76003189]])
    """

    def kinetic(*args):
        r"""Construct the kinetic matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the kinetic matrix
        """
        n = len(basis_functions)
        matrix = qml.math.zeros((n, n))

        for (i, a), (j, b) in it.combinations_with_replacement(enumerate(basis_functions), r=2):
            args_ab = []
            if args:
                args_ab.extend([arg[i], arg[j]] for arg in args)
            integral = kinetic_integral(a, b, normalize=False)(*args_ab)

            o = qml.math.zeros((n, n))
            o[i, j] = o[j, i] = 1.0
            matrix = matrix + integral * o

        return matrix

    return kinetic


def attraction_matrix(basis_functions, charges, r):
    r"""Return a function that computes the electron-nuclear attraction matrix for a given set of
    basis functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions
        charges (list[int]): nuclear charges
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the electron-nuclear attraction matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> attraction_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    array([[-2.03852057, -1.60241667], [-1.60241667, -2.03852057]])
    """

    def attraction(*args):
        r"""Construct the electron-nuclear attraction matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the electron-nuclear attraction matrix
        """
        n = len(basis_functions)
        matrix = qml.math.zeros((n, n))
        for (i, a), (j, b) in it.combinations_with_replacement(enumerate(basis_functions), r=2):
            integral = 0
            if args:
                args_ab = []

                if r.requires_grad:
                    args_ab.extend([arg[i], arg[j]] for arg in args[1:])
                else:
                    args_ab.extend([arg[i], arg[j]] for arg in args)

                for k, c in enumerate(r):
                    if c.requires_grad:
                        args_ab = [args[0][k]] + args_ab
                    integral = integral - charges[k] * attraction_integral(
                        c, a, b, normalize=False
                    )(*args_ab)
                    if c.requires_grad:
                        args_ab = args_ab[1:]
            else:
                for k, c in enumerate(r):
                    integral = (
                        integral - charges[k] * attraction_integral(c, a, b, normalize=False)()
                    )

            o = qml.math.zeros((n, n))
            o[i, j] = o[j, i] = 1.0
            matrix = matrix + integral * o

        return matrix

    return attraction


def repulsion_tensor(basis_functions):
    r"""Return a function that computes the electron repulsion tensor for a given set of basis
    functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions

    Returns:
        function: function that computes the electron repulsion tensor

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> repulsion_tensor(mol.basis_set)(*args)
    array([[[[0.77460595, 0.56886144], [0.56886144, 0.65017747]],
            [[0.56886144, 0.45590152], [0.45590152, 0.56886144]]],
           [[[0.56886144, 0.45590152], [0.45590152, 0.56886144]],
            [[0.65017747, 0.56886144],[0.56886144, 0.77460595]]]])
    """

    def repulsion(*args):
        r"""Construct the electron repulsion tensor for a given set of basis functions.

        Permutational symmetries are taken from [D.F. Brailsford and G.G. Hall, International
        Journal of Quantum Chemistry, 1971, 5, 657-668].

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the electron repulsion tensor
        """
        n = len(basis_functions)
        tensor = qml.math.zeros((n, n, n, n))
        e_calc = qml.math.full((n, n, n, n), np.nan)

        for (i, a), (j, b), (k, c), (l, d) in it.product(enumerate(basis_functions), repeat=4):
            if qml.math.isnan(e_calc[(i, j, k, l)]):
                args_abcd = []
                if args:
                    args_abcd.extend([arg[i], arg[j], arg[k], arg[l]] for arg in args)
                integral = repulsion_integral(a, b, c, d, normalize=False)(*args_abcd)

                permutations = [
                    (i, j, k, l),
                    (k, l, i, j),
                    (j, i, l, k),
                    (l, k, j, i),
                    (j, i, k, l),
                    (l, k, i, j),
                    (i, j, l, k),
                    (k, l, j, i),
                ]

                o = qml.math.zeros((n, n, n, n))
                for perm in permutations:
                    o[perm] = 1.0
                    e_calc[perm] = 1.0
                tensor = tensor + integral * o
        return tensor

    return repulsion


def core_matrix(basis_functions, charges, r):
    r"""Return a function that computes the core matrix for a given set of basis functions.

    The core matrix is computed as a sum of the kinetic and electron-nuclear attraction matrices.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions
        charges (list[int]): nuclear charges
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the core matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    array([[-1.27848869, -1.21916299], [-1.21916299, -1.27848869]])
    """

    def core(*args):
        r"""Construct the core matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the core matrix
        """
        if r.requires_grad:
            t = kinetic_matrix(basis_functions)(*args[1:])
        else:
            t = kinetic_matrix(basis_functions)(*args)

        a = attraction_matrix(basis_functions, charges, r)(*args)
        return t + a

    return core
