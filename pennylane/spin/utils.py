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
"""
This module contains helper functions to create 
:class:`~pennylane.spin.lattice` objects.
"""
import numpy as np

# pylint: disable=too-many-arguments
# pylint: disable=use-a-generator


def map_vertices(basis_coords, sl, L, basis):
    """Generates lattice site indices for unit cell + sublattice coordinates."""

    basis_coords = basis_coords % L

    site_indices = np.zeros(basis_coords.shape[0], dtype=int)

    num_sl = len(basis)
    num_dim = len(L)

    nsites_axis = np.zeros(num_dim, dtype=int)
    nsites_axis[-1] = num_sl

    for j in range(num_dim - 1, 0, -1):
        nsites_axis[j - 1] = nsites_axis[j] * L[num_dim - j]

    for index in range(basis_coords.shape[0]):
        site_indices[index] = np.dot(basis_coords[index], nsites_axis)
    site_indices += sl

    return site_indices
