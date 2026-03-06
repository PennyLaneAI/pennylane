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
import functools
from pennylane.ops import PauliX
from pennylane.ops.qubit.state_preparation import _basis_state_decomp, _process_state
from pennylane.templates import Subroutine
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike


# pylint: disable=unused-argument
def basis_embedding_resources(features, wires):
    """Calculate the resources for BasisEmbedding."""
    return {PauliX: len(wires)}


@functools.partial(
    Subroutine,
    static_argnames=[],
    compute_resources=basis_embedding_resources,
)
def BasisEmbedding(features: TensorLike, wires: WiresLike):
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
    wires = Wires(wires)
    features = _process_state(features, wires)

    _basis_state_decomp(features, wires=wires)
