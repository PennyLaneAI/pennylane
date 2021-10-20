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
def batch_params(tape, all_operations=False):
    """Transform a QNode to support an initial batch dimension
    for operation parameters.

    .. note::

        This transform will create multiple circuits inside the QNode, one per batch dimension.
        As a result, it is both simulator and hardware compatible. When using
        a simulator device, however, this means that a separate simulation
        will be performed per batch dimension.

    .. warning::

        Currently, not all templates have been updated to support a batch
        dimension. If you run into an error attempting to use a template
        with this transform, please open a GitHub issue detailing
        the error.

    Args:
        qnode (pennylane.QNode or .QuantumTape): quantum tape or QNode to add a batch dimension to
        all_operations (bool): If ``True``, a batch dimension will be added to *all* operations
            in the QNode, rather than just trainable QNode parameters.

    Returns:
        func: Function which accepts the same arguments as the QNode, however the
        first dimension of each argument will be treated as a batch dimension.
        The function output will also contain an initial batch dimension.

    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.batch_params
        @qml.beta.qnode(dev)
        def circuit(x, weights):
            qml.RX(x, wires=0)
            qml.RY(0.2, wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.expval(qml.Hadamard(0))

    The ``qml.batch_params`` decorator allows us to pass arguments ``x`` and ``weights``
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

    If we pass the ``all_operations`` argument, we can specify that
    *all* operation parameters in the transformed QNode, regardless of whether they
    are QNode input parameters, have a batch dimension:

    .. code-block:: python

        @functools.partial(qml.batch_params, all_operations=True)
        @qml.beta.qnode(dev)
        def circuit(x, weights):
            qml.RX(x, wires=0)
            qml.RY([0.2, 0.2, 0.2], wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.expval(qml.Hadamard(0))

    >>> cost_fn = lambda x, weights: np.sum(circuit(x, weights))
    >>> weights.requires_grad = False
    >>> cost_fn(x, weights)
    0.5497108163237583
    >>> qml.grad(cost_fn)(x, weights)[0]
    0.43792281188363347
    """
    params = tape.get_parameters(trainable_only=not all_operations)
    output_tapes = []

    try:
        batch_dim = qml.math.shape(params[0])[0]
    except IndexError:
        raise ValueError(f"Parameter {params[0]} does not contain a batch dimension.") from None

    unbatched_params = [[] for i in range(batch_dim)]

    for p in params:
        for i in range(batch_dim):
            try:
                unbatched_params[i].append(p[i])
            except IndexError:
                raise ValueError(
                    f"Parameter {p} has incorrect batch dimension. Expecting "
                    f"first dimension of length {batch_dim}."
                ) from None

    for p in unbatched_params:
        new_tape = tape.copy(copy_operations=True)
        new_tape.set_parameters(p, trainable_only=not all_operations)
        output_tapes.append(new_tape)

    return output_tapes, qml.math.stack
