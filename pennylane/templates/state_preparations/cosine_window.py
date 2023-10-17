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
r"""
Contains the CosineWindow template.
"""
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, StatePrepBase
from pennylane.wires import Wires, WireError
from pennylane import math


class CosineWindow(StatePrepBase):
    r"""CosineWindow(wires)
    Prepare the initial state following a cosine wave function.

    .. math::

        |\psi\rangle = \sqrt{2^{1-m}} \sum_{k=0}^{2^m-1} \cos(\frac{\pi k}{2^m} - \frac{\pi}{2}) |k\rangle,

    where :math:`m` is the number of wires.

    .. figure:: ../../_static/templates/state_preparations/cosine_window.png
        :align: center
        :width: 65%
        :target: javascript:void(0);

    .. note::

        The wave function is shifted by :math:`\frac{\pi}{2}` units so that the window is centered.


    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None


    Args:
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    **Example**

    >>> dev = qml.device('default.qubit', wires=2)
    >>> @qml.qnode(dev)
    ... def example_circuit():
    ...     qml.CosineWindow(wires=range(2))
    ...     return qml.probs()
    >>> print(example_circuit())
    [1.87469973e-33 2.50000000e-01 5.00000000e-01 2.50000000e-01]
    """
    num_wires = AnyWires
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def __init__(self, wires, id=None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).
        It is efficiently decomposed from one QFT over all qubits and one-qubit rotation gates.

        Args:
            wires (Iterable, Wires): the wire(s) the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations
        """

        decomp_ops = []

        decomp_ops.append(qml.Hadamard(wires=wires[-1]))
        decomp_ops.append(qml.RZ(-np.pi, wires=wires[-1]))
        decomp_ops.append(qml.adjoint(qml.QFT)(wires=wires))

        for ind, wire in enumerate(wires):
            decomp_ops.append(qml.PhaseShift(np.pi * 2 ** (-ind - 1), wires=wire))

        return decomp_ops

    def state_vector(self, wire_order=None):
        num_op_wires = len(self.wires)
        op_vector_shape = (-1,) + (2,) * num_op_wires if self.batch_size else (2,) * num_op_wires
        vector = np.array(
            [
                np.sqrt(2)
                * np.cos(-np.pi / 2 + np.pi * x / 2 ** (num_op_wires))
                / np.sqrt(2**num_op_wires)
                for x in range(2**num_op_wires)
            ]
        )
        op_vector = math.reshape(vector, op_vector_shape)

        if wire_order is None or Wires(wire_order) == self.wires:
            return op_vector

        wire_order = Wires(wire_order)
        if not wire_order.contains_wires(self.wires):
            raise WireError(f"Custom wire_order must contain all {self.name} wires")

        num_total_wires = len(wire_order)
        indices = tuple(
            [Ellipsis] + [slice(None)] * num_op_wires + [0] * (num_total_wires - num_op_wires)
        )
        ket_shape = [2] * num_total_wires
        if self.batch_size:
            # Add broadcasted dimension to the shape of the state vector
            ket_shape = [self.batch_size] + ket_shape

        ket = np.zeros(ket_shape, dtype=np.complex128)
        ket[indices] = op_vector

        # unless wire_order is [*self.wires, *rest_of_wire_order], need to rearrange
        if self.wires != wire_order[:num_op_wires]:
            current_order = self.wires + list(Wires.unique_wires([wire_order, self.wires]))
            desired_order = [current_order.index(w) for w in wire_order]
            if self.batch_size:
                # If the operation is broadcasted, the desired order must include the batch dimension
                # as the first dimension.
                desired_order = [0] + [d + 1 for d in desired_order]

            ket = ket.transpose(desired_order)

        return math.convert_like(ket, op_vector)
