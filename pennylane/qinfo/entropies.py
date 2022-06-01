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
"""Differentiable quantum entropies"""

import pennylane as qml


def vn_entropy_transform(state, wires=None, base=None):
    """Get Von Neumann entropies from a state."""

    def wrapper(*args, **kwargs):
        # Check for the QNode return type
        density_matrix = qml.qinfo.density_matrix_transform(state, wires)(*args, **kwargs)
        entropy = qml.math.compute_vn_entropy(density_matrix, base)
        return entropy

    return wrapper


def mutual_info_transform(qnode, wires0, wires1, base=None):
    """Get the mutual information between subsystems from a QNode"""

    def wrapper(*args, **kwargs):
        density_matrix = qml.qinfo.density_matrix_transform(qnode, qnode.device.wires.tolist())(
            *args, **kwargs
        )
        entropy = qml.math.to_mutual_info(density_matrix, wires0, wires1, base=base)
        return entropy

    return wrapper
