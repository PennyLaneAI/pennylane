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


def vn_entropy_transform(qnode, indices=None, base=None):
    """Compute the Von Neumann entropy from a :class:`.QNode` returning a :func:`~.state`.

    Args:
        qnode (tensor_like): A :class:`.QNode` returning a :func:`~.state`.
        indices (list(int)): List of indices in the considered subsystem.
        base (float, int): Base for the logarithm.

    Returns:
        float: Von Neumann entropy of the considered subsystem.

    **Example**

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)
            @qml.qnode(dev)
            def circuit(x):
                qml.IsingXX(x, wires=[0, 1])
                return qml.state()

    >>> vn_entropy_transform(circuit, indices=[0])(np.pi/2)
    0.6931472

    """

    def wrapper(*args, **kwargs):
        density_matrix = qml.qinfo.density_matrix_transform(qnode, indices)(*args, **kwargs)
        entropy = qml.math.compute_vn_entropy(density_matrix, base)
        return entropy

    return wrapper
