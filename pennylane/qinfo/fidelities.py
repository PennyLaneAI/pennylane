# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Differentiable quantum fidelity"""

from collections.abc import Iterable
import pennylane as qml

def fidelity(qnode0, qnode1, indices0, indices1):
    """Compute the Fidelity entropy from two :class:`.QNode` returning a :func:`~.state`.

    """
    density_matrix_qnode0 = qml.qinfo.density_matrix_transform(qnode0, indices0)
    density_matrix_qnode1 = qml.qinfo.density_matrix_transform(qnode1, indices1)

    def wrapper(signature0=None, signature1=None):

        if signature0 is not None:
            if isinstance(signature0, Iterable):
                density_matrix0 = density_matrix_qnode0(signature0)
            else:
                density_matrix0 = density_matrix_qnode0(*signature0)
        else:
            density_matrix0 = density_matrix_qnode0()

        if signature1 is not None:
            if isinstance(signature0, Iterable):
                density_matrix1 = density_matrix_qnode1(signature1)
            else:
                density_matrix1 = density_matrix_qnode1(*signature1)
        else:
            density_matrix1 = density_matrix_qnode1()

        fidelity = qml.math.to_fidelity(density_matrix0, density_matrix1)
        return fidelity

    return wrapper
