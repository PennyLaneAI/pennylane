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

def fidelity(qnode, indices, base=None):
    """Compute the Von Neumann entropy from a :class:`.QNode` returning a :func:`~.state`.

    """

    density_matrix_qnode = qml.qinfo.density_matrix_transform(qnode, indices)

    def wrapper(*args, **kwargs):
        density_matrix = density_matrix_qnode(*args, **kwargs)
        entropy = qml.math.compute_vn_entropy(density_matrix, base)
        return entropy

    return wrapper