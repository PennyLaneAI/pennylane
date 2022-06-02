# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for resource estimation with double factorization method.
"""

from pennylane import numpy as np


def norm(one, two, eigvals):
    r"""Return the 1-norm of a molecular Hamiltonian from the one- and two-electron and factorized
    two-electron integrals.

     Args:
         one (array[array[float]]): one-electron integrals
         two (array[array[float]]): two-electron integrals
         eigvals (array[float]): eigenvalues of the matrices obtained from factorizing the
             two-electron integral tensor


     Returns:
         array[float]: 1-norm of the Hamiltonian

     **Example**

     >>> symbols  = ['H', 'H', 'O']
     >>> geometry = np.array([[0.0,  0.000000000,  0.150166845],
     >>>                      [0.0,  0.768778665, -0.532681406],
     >>>                      [0.0, -0.768778665, -0.532681406]], requires_grad = False) / 0.529177
     >>> mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g')
     >>> core, one, two = qml.qchem.electron_integrals(mol)()
     >>> two = np.swapaxes(two, 1, 3) # convert to the chemists notation
     >>> l, w, v = factorize(two, 1e-5)
     >>> print(norm(one, two, w))
     52.98762043980203
     """
    lambda_one = 0.25 * np.sum([np.sum(abs(val)) ** 2 for val in eigvals])

    l_inv = one - 0.5 * np.einsum('illj', two) + np.einsum('llij', two)

    val, vec = np.linalg.eigh(l_inv)

    lambda_two = np.sum(abs(val))

    return lambda_one + lambda_two
