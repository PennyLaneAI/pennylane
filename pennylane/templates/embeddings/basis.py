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
Contains the BasisEmbedding template.
"""

from pennylane.decomposition import add_decomps
from pennylane.ops.qubit.state_preparation import BasisState, _basis_state_decomp


class BasisEmbedding(BasisState):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=np.array([0, 1, 0])`` or ``features=2`` (binary 010), the
    quantum system will be prepared in state :math:`|010 \rangle`.

    .. warning::

        ``BasisEmbedding`` calls a circuit whose architecture depends on the binary features.
        The ``features`` argument is therefore not differentiable when using the template, and
        gradients with respect to the argument cannot be computed by PennyLane.

    Args:
        features (tensor_like or int): Binary input of shape ``(len(wires), )`` or integer
            that represents the binary input.
        wires (Any or Iterable[Any]): the wire(s) that the template acts on

    Example:

        Basis embedding encodes the binary feature vector into a basis state.

        .. code-block:: python

            dev = qml.device('reference.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(feature_vector):
                qml.BasisEmbedding(features=feature_vector, wires=range(3))
                return qml.state()

            X = [1,1,1]

        The resulting circuit is:

        >>> print(qml.draw(circuit, level="device")(X))
        0: ──X─┤ ╭State
        1: ──X─┤ ├State
        2: ──X─┤ ╰State

        And, the output state is:

        >>> print(circuit(X))
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j]

        Thus, ``[1,1,1]`` is mapped to :math:`|111 \rangle`.

    """

    def __init__(self, features, wires, id=None):
        super().__init__(features, wires=wires, id=id)


add_decomps(BasisEmbedding, _basis_state_decomp)
