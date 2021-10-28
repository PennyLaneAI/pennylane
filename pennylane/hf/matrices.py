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
import autograd.numpy as anp
from pennylane.hf.integrals import (
    generate_attraction,
    generate_kinetic,
    generate_overlap,
    generate_repulsion,
)


def molecular_density_matrix(n_electron, c):
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
    >>> density_matrix(n_electron, c)
    array([[0.30061941, 0.30061941], [0.30061941, 0.30061941]])
    """
    p = anp.dot(c[:, : n_electron // 2], anp.conjugate(c[:, : n_electron // 2]).T)
    return p


def generate_overlap_matrix(basis_functions):
    r"""Return a function that computes the overlap matrix for a given set of basis functions.

    Args:
        basis_functions (list[BasisFunction]): basis functions

    Returns:
        function: function that computes the overlap matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True),
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> overlap_matrix(mol.basis_set)(*args)
    array([[1.0, 0.7965883009074122], [0.7965883009074122, 1.0]])
    """

    def overlap(*args):
        r"""Construct the overlap matrix for a given set of basis functions.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the overlap matrix
        """
        n = len(basis_functions)
        overlap_matrix = anp.eye(len(basis_functions))
        for i, a in enumerate(basis_functions):
            for j, b in enumerate(basis_functions):
                if i < j:
                    if args:
                        args_ab = []
                        for arg in args:
                            args_ab.append(arg[[i, j]])
                        overlap_integral = generate_overlap(a, b)(*args_ab)
                    else:
                        overlap_integral = generate_overlap(a, b)()
                    o = anp.zeros((n, n))
                    o[i, j] = o[j, i] = 1.0
                    overlap_matrix = overlap_matrix + overlap_integral * o
        return overlap_matrix

    return overlap


def generate_kinetic_matrix(basis_functions):
    r"""Return a function that computes the kinetic matrix for a given set of basis functions.

    Args:
        basis_functions (list[BasisFunction]): basis functions

    Returns:
        function: function that computes the kinetic matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True),
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> kinetic_matrix(mol.basis_set)(*args)
    array([[0.76003189, 0.38325367], [0.38325367, 0.76003189]])
    """

    def kinetic(*args):
        r"""Construct the kinetic matrix for a given set of basis functions.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the kinetic matrix
        """
        n = len(basis_functions)
        kinetic_matrix = anp.zeros((n, n))
        for i, a in enumerate(basis_functions):
            for j, b in enumerate(basis_functions):
                if i <= j:
                    if args:
                        args_ab = []
                        for arg in args:
                            args_ab.append(arg[[i, j]])
                        kinetic_integral = generate_kinetic(a, b)(*args_ab)
                    else:
                        kinetic_integral = generate_kinetic(a, b)()
                    o = anp.zeros((n, n))
                    o[i, j] = o[j, i] = 1.0
                    kinetic_matrix = kinetic_matrix + kinetic_integral * o
        return kinetic_matrix

    return kinetic


def generate_attraction_matrix(basis_functions, charges, r):
    r"""Return a function that computes the electron-nuclear attraction matrix for a given set of
    basis functions.

    Args:
        basis_functions (list[BasisFunction]): basis functions
        charges (list[int]): nuclear charges
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the electron-nuclear attraction matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True),
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> attraction_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    array([[-2.03852057, -1.60241667], [-1.60241667, -2.03852057]])
    """

    def attraction(*args):
        r"""Construct the electron-nuclear attraction matrix for a given set of basis functions.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the electron-nuclear attraction matrix
        """
        n = len(basis_functions)
        attraction_matrix = anp.zeros((n, n))
        for i, a in enumerate(basis_functions):
            for j, b in enumerate(basis_functions):
                attraction_integral = 0
                if i <= j:
                    if args:
                        args_ab = []

                        if r.requires_grad:
                            for l in range(len(args) - 1):
                                args_ab.append(args[l + 1][[i, j]])
                        else:
                            for arg in args:
                                args_ab.append(arg[[i, j]])

                        for k, c in enumerate(r):
                            if c.requires_grad:
                                args_ab = [args[0][k]] + args_ab
                            attraction_integral = attraction_integral - charges[
                                k
                            ] * generate_attraction(c, a, b)(*args_ab)
                            if c.requires_grad:
                                args_ab = args_ab[1:]
                    else:
                        for k, c in enumerate(r):
                            attraction_integral = (
                                attraction_integral - charges[k] * generate_attraction(c, a, b)()
                            )

                    o = anp.zeros((n, n))
                    o[i, j] = o[j, i] = 1.0
                    attraction_matrix = attraction_matrix + attraction_integral * o
        return attraction_matrix

    return attraction


def generate_repulsion_tensor(basis_functions):
    r"""Return a function that computes the electron repulsion tensor for a given set of basis
    functions.

    Args:
        basis_functions (list[BasisFunction]): basis functions

    Returns:
        function: function that computes the electron repulsion tensor

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True),
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
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
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the electron repulsion tensor
        """
        n = len(basis_functions)
        repulsion_tensor = anp.zeros((n, n, n, n))
        e_calc = []

        for i, a in enumerate(basis_functions):
            for j, b in enumerate(basis_functions):
                for k, c in enumerate(basis_functions):
                    for l, d in enumerate(basis_functions):

                        if [i, j, k, l] not in e_calc:
                            if args:
                                args_abcd = []
                                for arg in args:
                                    args_abcd.append(arg[[i, j, k, l]])
                                repulsion_integral = generate_repulsion(a, b, c, d)(*args_abcd)
                            else:
                                repulsion_integral = generate_repulsion(a, b, c, d)()

                            o = anp.zeros((n, n, n, n))
                            o[i, j, k, l] = o[k, l, i, j] = o[j, i, l, k] = o[l, k, j, i] = 1.0
                            o[j, i, k, l] = o[l, k, i, j] = o[i, j, l, k] = o[k, l, j, i] = 1.0
                            repulsion_tensor = repulsion_tensor + repulsion_integral * o
                            e_calc = e_calc + [
                                [i, j, k, l],
                                [k, l, i, j],
                                [j, i, l, k],
                                [l, k, j, i],
                                [j, i, k, l],
                                [l, k, i, j],
                                [i, j, l, k],
                                [k, l, j, i],
                            ]

        return repulsion_tensor

    return repulsion


def generate_core_matrix(basis_functions, charges, r):
    r"""Return a function that computes the core matrix for a given set of basis functions.

    The core matrix is computed as a sum of the kinetic and electron-nuclear attraction matrices.

    Args:
        basis_functions (list[BasisFunction]): basis functions
        charges (list[int]): nuclear charges
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the core matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True),
    >>> mol = Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    array([[-1.27848869, -1.21916299], [-1.21916299, -1.27848869]])
    """

    def core(*args):
        r"""Construct the core matrix for a given set of basis functions.

        Args:
            args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the core matrix
        """
        if r.requires_grad:
            t = generate_kinetic_matrix(basis_functions)(*args[1:])
        else:
            t = generate_kinetic_matrix(basis_functions)(*args)

        a = generate_attraction_matrix(basis_functions, charges, r)(*args)
        return t + a

    return core
