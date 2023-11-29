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
from typing import Callable, Sequence

import pennylane as qml


from .core import transform


def _nested_stack(res):
    """
    Given a list of identical nested tuple structures, stack the arrays at the leaves
    """
    # for some reason pylint thinks qml.numpy.builtins is a dict
    # pylint: disable=no-member
    if not isinstance(res[0], (tuple, qml.numpy.builtins.SequenceBox)):
        return qml.math.stack(res)

    stacked_results = []
    for i in range(len(res[0])):
        stacked_results.append(_nested_stack([r[i] for r in res]))

    return tuple(stacked_results)


def _split_operations(ops, params, split_indices, num_tapes):
    """
    Given a list of operators, return a list (with length ``num_tapes``) containing lists
    of new operators with the parameters at the given indices unbatched.

    Args:
        ops (Sequence[.Operator]): list of operators to split
        params (Sequence[TensorLike]): list of parameters which may have a batch dimension.
            The size of this list must be the total number of parameters in the ``ops`` list.
        split_indices (Sequence[int]): the parameter indices that need to be unbatched. The
            index of a parameter, say ``p``, is defined to be its index in the list
            ``[p for op in ops for p in op.data]``. If any parameter of an operator has an index
            contained in ``split_indices``, then a new operator is created using the corresponding
            entry of ``params``. Otherwise, the original operator is used.
        num_tapes (int): the number of new tapes to create, which is also equal to the batch size.
    """
    # for some reason pylint thinks "qml.ops" is a set
    # pylint: disable=no-member
    new_ops = [[] for _ in range(num_tapes)]
    idx = 0

    for op in ops:
        # determine if any parameters of the operator are batched
        if any(i in split_indices for i in range(idx, idx + len(op.data))):
            for b in range(num_tapes):
                new_params = tuple(
                    params[i][b] if i in split_indices else params[i]
                    for i in range(idx, idx + len(op.data))
                )
                new_op = qml.ops.functions.bind_new_parameters(op, new_params)
                new_ops[b].append(new_op)
        else:
            # no batching in the operator; don't copy
            for b in range(num_tapes):
                new_ops[b].append(op)

        idx += len(op.data)

    return new_ops


@transform
def batch_params(
    tape: qml.tape.QuantumTape, all_operations=False
) -> (Sequence[qml.tape.QuantumTape], Callable):
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
        tape (QNode or QuantumTape or Callable): a quantum circuit to add a batch dimension to
        all_operations (bool): If ``True``, a batch dimension will be added to *all* operations
            in the QNode, rather than just trainable QNode parameters.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the batched results, with the first dimension treated as the batch dimension.

    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.batch_params
        @qml.qnode(dev)
        def circuit(x, weights):
            qml.RX(x, wires=0)
            qml.RY(0.2, wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.expval(qml.Hadamard(0))

    The ``qml.batch_params`` decorator allows us to pass arguments ``x`` and ``weights``
    that have a batch dimension. For example,

    >>> batch_size = 3
    >>> x = np.linspace(0.1, 0.5, batch_size)
    >>> rng = np.random.default_rng(seed=1234)
    >>> weights = rng.random((batch_size, 10, 3, 3), requires_grad=True)

    If we evaluate the QNode with these inputs, we will get an output
    of shape ``(batch_size,)``:

    >>> circuit(x, weights)
    tensor([ 0.00800498,  0.2735391 , -0.24395442], requires_grad=True)

    QNodes with a batch dimension remain fully differentiable:

    >>> cost_fn = lambda x, weights: np.sum(circuit(x, weights))
    >>> cost_fn(x, weights)
    tensor(0.03758966, requires_grad=True)
    >>> qml.grad(cost_fn)(x, weights)[0]
    array([-0.30262974,  0.06320878,  0.00811555])

    If we pass the ``all_operations`` argument, we can specify that
    *all* operation parameters in the transformed QNode, regardless of whether they
    are QNode input parameters, have a batch dimension:

    .. code-block:: python

        from functools import partial

        @partial(qml.batch_params, all_operations=True)
        @qml.qnode(dev)
        def circuit(x, weights):
            qml.RX(x, wires=0)
            qml.RY([0.2, 0.2, 0.2], wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.expval(qml.Hadamard(0))

    >>> cost_fn = lambda x, weights: np.sum(circuit(x, weights))
    >>> weights.requires_grad = False
    >>> cost_fn(x, weights)
    tensor(0.03758966, requires_grad=True)
    >>> qml.grad(cost_fn)(x, weights)[0]
    -0.30262974103192636
    """
    # pylint: disable=protected-access
    params = tape.get_parameters(trainable_only=False)
    indices = list(range(len(params))) if all_operations else list(tape.trainable_params)

    if not indices:
        raise ValueError(
            "There are no operations to transform. Either add trainable parameters, "
            "or specify `all_operations=True`."
        )

    try:
        batch_dim = qml.math.shape(params[indices[0]])[0]
    except IndexError:
        raise ValueError(f"Parameter {params[0]} does not contain a batch dimension.") from None

    for i in indices:
        shape = qml.math.shape(params[i])
        if len(shape) == 0 or shape[0] != batch_dim:
            raise ValueError(
                f"Parameter {params[i]} has incorrect batch dimension. Expecting "
                f"first dimension of length {batch_dim}."
            )

    output_tapes = []
    for ops in _split_operations(tape.operations, params, indices, batch_dim):
        new_tape = qml.tape.QuantumScript(
            ops, tape.measurements, shots=tape.shots, trainable_params=tape.trainable_params
        )
        output_tapes.append(new_tape)

    def processing_fn(res):
        return _nested_stack(res)

    return output_tapes, processing_fn
