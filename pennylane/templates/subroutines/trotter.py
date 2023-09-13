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
Contains templates for Suzuki-Trotter approximation based subroutines.
"""

import pennylane as qml

# from pennylane.math ...
from pennylane import numpy as np
from pennylane.operation import Operation


def _scaler(order):
    """Assumes that order is an even integer > 2"""
    return (4 - 4 ** (order - 1))**-1

def


class TrotterProduct(Operation):
    """Representing the Suzuki-Trotter product approximation"""

    def __init__(self, hamiltonian, time, n=1, order=1, check_hermitian=True, id=None):

        if isinstance(hamiltonian, qml.Hamiltonian):
            coeffs, ops = hamiltonian.terms()
        else:


        if check_hermitian:
            pass

