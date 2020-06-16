# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains functions to construct many-body observables whose expectation
values can be used to simulate molecular properties."""
# pylint: disable=too-many-arguments, too-few-public-methods
import os
import numpy as np
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import bravyi_kitaev, jordan_wigner

from pennylane import qchem

def s2_me_table(sz, n_spin_orbs):
    r"""Computes the matrix elements
    :math:`\langle \alpha, beta \vert \hat{s}_1 \cdot \hat{s}_2 \vert \gamma, \delta \rangle`
    of the two-particle spin operator :math:`\hat{s}_1 \cdot \hat{s}_2`.

    The latter matrix elements are computed as follows:

    .. math:

        \langle \alpha, \beta \vert \hat{\bf{s}}_1 \cdot \hat {\bf{s}}_2
        \vert \gamma, \delta \rangle = && \delta_{\alpha,\delta} \delta_{\beta,\gamma} \times \\
        && \left( \frac{1}{2} \delta_{m_\alpha, m_\delta+1} \delta_{m_\beta, m_\gamma-1}
        + \frac{1}{2} \delta_{m_\alpha, m_\delta-1} \delta_{m_\beta, m_\gamma+1}
        + m_\alpha m_\beta \delta_{m_\alpha, m_\delta} \delta_{m_\beta, m_\gamma} \right),

    where :math:`\alpha` and :math:`m_\alpha` refer to the quantum numbers of the spatial
    :math:`\varphi_\alpha({\bf r})` and spin :math:`\chi_{m_\alpha}(s_z)` wave functions,
    respectively, of the single-particle state :math:`\vert \alpha \rangle`.

    Args:
        sz (array[float]): spin-projection quantum number of the spin-orbitals
        n_spin_orbs (int): number of spin orbitals
   
    Returns: 
        array: NumPy array with the table of matrix elements
    """

    n = np.arange(n_spin_orbs)

    alpha = n.reshape(-1, 1, 1, 1)
    beta = n.reshape(1, -1, 1, 1)
    gamma = n.reshape(1, 1, -1, 1)
    delta = n.reshape(1, 1, 1, -1)
    
    # we only care about indices satisfying the following boolean mask
    mask = np.logical_and(alpha // 2 == delta // 2, beta // 2 == gamma // 2)

    # diagonal elements
    diag_mask = np.logical_and(sz[alpha] == sz[delta], sz[beta] == sz[gamma])
    diag_indices = np.argwhere(np.logical_and(mask, diag_mask))
    diag_values = (sz[alpha] * sz[beta]).flatten()

    diag = np.vstack([diag_indices.T, diag_values]).T
    
    # off-diagonal elements
    m1 = np.logical_and(sz[alpha] == sz[delta] + 1, sz[beta] == sz[gamma] - 1)
    m2 = np.logical_and(sz[alpha] == sz[delta] - 1, sz[beta] == sz[gamma] + 1)

    off_diag_mask = np.logical_and(mask, np.logical_or(m1, m2))
    off_diag_indices = np.argwhere(off_diag_mask)
    off_diag_values = np.full([len(off_diag_indices)], 0.5)

    off_diag = np.vstack([off_diag_indices.T, off_diag_values]).T
    
    # combine the off diagonal and diagonal tables into a single table
    return np.vstack([diag, off_diag])


def get_s2_me(mol_name, hf_data, n_active_electrons=None, n_active_orbitals=None):
    r"""Reads the Hartree-Fock (HF) electronic structure data file, defines an active space and
    generates the table with the matrix elements of the two-particle spin operator
    :math:`\langle \alpha, beta \vert \hat{s}_1 \cdot \hat{s}_2 \vert \gamma, \delta \rangle`

    Args:
        mol_name (str): name of the molecule
        hf_data (str): path to the directory with the HF electronic structure data file
        n_active_electrons (int): number of active electrons
        n_active_orbitals (int): number of active orbitals
   
    Returns: 
        tuple: the contribution of doubly-occupied orbitals, if any, or other quantity
        required to initialize the many-body observable and a Numpy array with the table
        of matrix elements
    """
    
    docc_indices, active_indices = qchem.active_space(
        mol_name,
        hf_data,
        n_active_electrons=n_active_electrons,
        n_active_orbitals=n_active_orbitals
    )
    
    if n_active_electrons == None:
        hf_elect_struct = MolecularData(filename=os.path.join(hf_data.strip(), mol_name.strip()))
        n_electrons = hf_elect_struct.n_electrons
    else:
        n_electrons = n_active_electrons        
        
    n_spin_orbs = 2*len(active_indices)
    
    sz = np.where(np.arange(n_spin_orbs) % 2 == 0, 0.5, -0.5)
    
    return s2_me_table(sz, n_spin_orbs), 3/4*n_electrons

mol_name = "h2"
geo_file = "h2.xyz"
charge = 0
multiplicity = 1
basis = "sto-3g"

n_electrons = 2
n_orbitals = 2

geometry = qchem.read_structure(geo_file)
hf_data = qchem.meanfield_data("h2", geometry, charge, multiplicity, basis)
s2_me_table, init_term = get_s2_me(
    mol_name, hf_data,
    n_active_electrons=n_electrons,
    n_active_orbitals=n_orbitals
)
