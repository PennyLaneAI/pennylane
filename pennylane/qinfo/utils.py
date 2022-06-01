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
"""Utils function for the quantum information module.
"""
import pennylane as qml


def density_matrix_transform(qnode, indices):
    """Compute the reduced density matrix from a QNode.

     Args:
         qnode (QNode): A :class:`~.QNode` returning :func:`~.state`.
         indices (list(int)): List of indices in the considered subsystem.

     Returns:
         tensor_like: (Reduced) Density matrix of size ``(2**len(wires), 2**len(wires))``

     **Example**

     .. code-block:: python

         import numpy as np

         dev = qml.device("default.qubit", wires=2)
         @qml.qnode(dev)
         def circuit(x):
           qml.IsingXX(x, wires=[0,1])
           return qml.state()

    >>> density_matrix_transform(circuit, wires=[0])(np.pi/2)
     [[0.5+0.j 0.+0.j]
      [0.+0.j 0.5+0.j]]

    """

    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)
        return_type = qnode.tape.observables[0].return_type
        if len(qnode.tape.observables) != 1 or not return_type == qml.measurements.State:
            raise ValueError("The qfunc return type needs to be a state.")

        # TODO: optimize given the wires
        state_built = qnode(*args, **kwargs)
        density_matrix = qml.math.to_density_matrix(state_built, indices=indices)
        return density_matrix

    return wrapper
