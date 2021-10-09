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
"""
Contains the batch dimension transform.
"""
# pylint: disable=import-outside-toplevel
import pennylane as qml


from .batch_transform import batch_transform


@batch_transform
def batch_dim(tape):
    """Inserts a batch dimension to all trainable operation
    parameters of a QNode.

    Args:
        qnode (pennylane.QNode or .QuantumTape): quantum tape or QNode to add a batch dimension to

    Returns:
        func: Function which accepts the same arguments as the QNode, however the
        first dimension of each argument will be treated as a batch dimension.

    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.batch_dim
        @qml.beta.qnode(dev)
        def circuit(x, weights):
            qml.RX(x, wires=0)
            qml.RY(0.2, wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.expval(qml.Hadamard(0))

    The ``qml.batch_dim`` decorator allows us to pass arguments ``x`` and ``weights``
    that have a batch dimension. For example,

    >>> batch_size = 3
    >>> x = np.linspace(0.1, 0.5, batch_size)
    >>> weights = np.random.random((batch_size, 10, 3, 3))

    If we evaluate the QNode with these inputs, we will get an output
    of shape ``(batch_size,)``:

    >>> circuit(x, weights)
    [-0.30773348  0.23135516  0.13086565]

    QNodes with a batch dimension remain fully differentiable:

    >>> cost_fn = lambda x, weights: np.sum(circuit(x, weights))
    >>> cost_fn(x, weights)
    -0.8581269507766536
    >>> qml.grad(cost_fn)(x, weights)[0]
    [ 0.23235464  0.00928953 -0.30083487]
    """
    params = list(tape.get_parameters())
    output_tapes = []

    for p in zip(*params):
        new_tape = tape.copy(copy_operations=True)
        new_tape.set_parameters(p)
        output_tapes.append(new_tape)

    return output_tapes, qml.math.stack
