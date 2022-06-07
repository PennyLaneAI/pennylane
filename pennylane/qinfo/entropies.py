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


def vn_entropy_transform(qnode, indices, base=None):
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

    density_matrix_qnode = qml.qinfo.density_matrix_transform(qnode, indices)

    def wrapper(*args, **kwargs):
        density_matrix = density_matrix_qnode(*args, **kwargs)
        entropy = qml.math.compute_vn_entropy(density_matrix, base)
        return entropy

    return wrapper


def mutual_info_transform(qnode, indices0, indices1, base=None):
    """Compute the mutual information from a :class:`.QNode` returning a :func:`~.state`.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system.

    Args:
        qnode (QNode): A :class:`.QNode` returning a :func:`~.state`.
        indices0 (list[int]): List of indices in the first subsystem.
        indices1 (list[int]): List of indices in the second subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        func: A function with the same arguments as the QNode that returns
        the mutual information from its output state.

    **Example**

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(x):
                qml.IsingXX(x, wires=[0, 1])
                return qml.state()

    >>> mutual_info_circuit = qinfo.mutual_info_transform(circuit, indices0=[0], indices1=[1])
    >>> mutual_info_circuit(np.pi/2)
    1.3862943611198906
    >>> mutual_info_circuit(0.4)
    0.3325090393262875
    """

    density_matrix_qnode = qml.qinfo.density_matrix_transform(qnode, qnode.device.wires.tolist())

    def wrapper(*args, **kwargs):
        density_matrix = density_matrix_qnode(*args, **kwargs)
        entropy = qml.math.to_mutual_info(density_matrix, indices0, indices1, base=base)
        return entropy

    return wrapper
