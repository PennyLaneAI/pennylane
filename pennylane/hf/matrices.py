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

import autograd.numpy as anp
import numpy as np
from integrals import *

def density_matrix(n_electron, c):
    """Construct the density matrix.
    """
    p = anp.dot(c[:,:n_electron//2], anp.conjugate(c[:,:n_electron//2]).T)
    return p


def overlap_matrix(basis_set):
    r"""Construct the the overlap matrix for a given set of basis functions.
    The diagonal elements of the matrix are set to one.
    """
    def overlap(*args):
        s = anp.eye(len(basis_set))
        for i, a in enumerate(basis_set):
            for j, b in enumerate(basis_set):
                if i < j:
                    s[i, j] = s [j, i] = generate_overlap(a, b)(args[i], args[j])
        return s
    return overlap


def kinetic_matrix(basis_set):
    r"""Construct the the kinetic energy matrix for a given set of basis functions.
    """
    def kinetic(*args):
        k = anp.zeros((len(basis_set), len(basis_set)))
        for i, a in enumerate(basis_set):
            for j, b in enumerate(basis_set):
                if i == j:
                    k[i, j] = generate_kinetic(a, b)(args[i], args[j])
                if i < j:
                    k[i, j] = k[j, i] = generate_kinetic(a, b)(args[i], args[j])
        return k
    return kinetic


def attraction_matrix(basis_set, charges):
    r"""Construct the electron-nucleus attraction matrix for a given set of basis functions.
    """
    def attraction(r, *args):

        v = anp.zeros((len(basis_set), len(basis_set)))

        for i, a in enumerate(basis_set):
            for j, b in enumerate(basis_set):
                nuclear_attraction = 0
                if i == j:
                    for k, c in enumerate(r):
                        nuclear_attraction = nuclear_attraction + charges[k] * generate_attraction(a, b)(c, args[i], args[j])
                    v[i, j] = nuclear_attraction
                if i < j:
                    for k, c in enumerate(r):
                        nuclear_attraction = nuclear_attraction + charges[k] * generate_attraction(a, b)(c, args[i], args[j])
                    v[i, j] = v[j, i] = nuclear_attraction
        return v

    return attraction


def core_matrix(basis_set, charges):
    r"""Construct the core matrix for a given set of basis functions.
    """
    def core(r, *args):
        return -1 * attraction_matrix(basis_set, charges)(r, *args) + kinetic_matrix(basis_set)(*args)
    return core


def repulsion_tensor(basis_set):
    """Construct the electron repulsion tensor for a given set of basis functions.
    """
    def repulsion(*args):

        n = len(basis_set)
        e = anp.full((n, n, n, n), anp.inf)

        for i, a in enumerate(basis_set):
            for j, b in enumerate(basis_set):
                for k, c in enumerate(basis_set):
                    for l, d in enumerate(basis_set):

                        if e[i, j, k, l] == anp.inf:

                            electron_repulsion = generate_repulsion(a, b, c, d)(args[i], args[j], args[k], args[l])
                            e[i, j, k, l] = e[k, l, i, j] = e[j, i, l, k] = e[l, k, j, i] = electron_repulsion
                            e[j, i, k, l] = e[l, k, i, j] = e[i, j, l, k] = e[k, l, j, i] = electron_repulsion
        return e
    return repulsion