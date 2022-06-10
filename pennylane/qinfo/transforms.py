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
"""Utils function for the quantum information module."""

import pennylane as qml


def reduced_dm(qnode, wires):
    """Compute the reduced density matrix from a :class:`~.QNode` returning :func:`~.state`.

     Args:
         qnode (QNode): A :class:`~.QNode` returning :func:`~.state`.
         wires (Sequence(int)): List of indices in the considered subsystem.

     Returns:
         func: Function which wraps the QNode and accepts the same arguments. When called, this function will
            return the density matrix.

     **Example**

     .. code-block:: python

         import numpy as np

         dev = qml.device("default.qubit", wires=2)
         @qml.qnode(dev)
         def circuit(x):
           qml.IsingXX(x, wires=[0,1])
           return qml.state()

    >>> reduced_dm(circuit, wires=[0])(np.pi/2)
     [[0.5+0.j 0.+0.j]
      [0.+0.j 0.5+0.j]]

    .. seealso:: :func:`pennylane.density_matrix` and :func:`pennylane.math.reduced_dm`
    """

    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)
        return_type = qnode.tape.observables[0].return_type
        if len(qnode.tape.observables) != 1 or not return_type == qml.measurements.State:
            raise ValueError("The qfunc return type needs to be a state.")

        # TODO: optimize given the wires by creating a tape with relevant operations
        state_built = qnode(*args, **kwargs)
        density_matrix = qml.math.reduced_dm(
            state_built, indices=wires, c_dtype=qnode.device.C_DTYPE
        )
        return density_matrix

    return wrapper


def vn_entropy(qnode, wires, base=None):
    r"""Compute the Von Neumann entropy from a :class:`.QNode` returning a :func:`~.state`.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        qnode (tensor_like): A :class:`.QNode` returning a :func:`~.state`.
        wires (Sequence(int)): List of indices in the considered subsystem.
        base (float): Base for the logarithm, default is None the natural logarithm is used in this case.

    Returns:
        float: Von Neumann entropy of the considered subsystem.

    **Example**

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)
            @qml.qnode(dev)
            def circuit(x):
                qml.IsingXX(x, wires=[0, 1])
                return qml.state()

    >>> vn_entropy(circuit, indices=[0])(np.pi/2)
    0.6931472

    """

    density_matrix_qnode = qml.qinfo.reduced_dm(qnode, qnode.device.wires)

    def wrapper(*args, **kwargs):
        # If pure state directly return 0.
        if len(wires) == len(qnode.device.wires):
            qnode.construct(args, kwargs)
            return_type = qnode.tape.observables[0].return_type
            if len(qnode.tape.observables) != 1 or not return_type == qml.measurements.State:
                raise ValueError("The qfunc return type needs to be a state.")
            density_matrix = qnode(*args, **kwargs)
            if density_matrix.shape == (density_matrix.shape[0],):
                return 0.0
            entropy = qml.math.vn_entropy(density_matrix, wires, base, c_dtype=qnode.device.C_DTYPE)
            return entropy

        density_matrix = density_matrix_qnode(*args, **kwargs)
        entropy = qml.math.vn_entropy(density_matrix, wires, base, c_dtype=qnode.device.C_DTYPE)
        return entropy

    return wrapper
