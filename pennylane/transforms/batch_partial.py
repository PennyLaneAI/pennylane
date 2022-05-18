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
"""
Contains the batch dimension transform.
"""
import functools
import inspect

import pennylane as qml


def _convert_to_args(func, args, kwargs):
    """
    Given a function, convert the positional and
    keyword arguments to purely positional arguments.
    """
    sig = inspect.signature(func).parameters

    new_args = []
    for i, param in enumerate(sig):
        if param in kwargs:
            # first check if the name is provided in kwargs
            new_args.append(kwargs[param])
        elif i < len(sig):
            # next check if the argnum is provided
            new_args.append(args[i])

    return tuple(new_args)


def batch_partial(qnode, all_operations=False, **partial_kwargs):
    """
    Create a wrapper function around the QNode with partially
    evaluated parameters, which supports an initial batch dimension
    for other unevaluated parameters.

    Args:
        qnode (pennylane.QNode): QNode to partially evaluate
        all_operations (bool): If ``True``, a batch dimension will be added to *all* operations
            in the QNode, rather than just trainable QNode parameters.
        partial_kwargs (dict): partially-evaluated parameters to pass to the QNode

    Returns:
        func: Function which accepts the same arguments as the QNode minus the
        partially evaluated arguments provided, and behaves the same as the QNode
        called with both the partially evaluated arguments and the extra arguments.
        However, the first dimension of each argument of the returned function
        will be treated as a batch dimension. The function output will also contain
        an initial batch dimension.

    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y[..., 0], wires=0)
            qml.RY(y[..., 1], wires=1)
            return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    The ``qml.batch_partial`` decorator allows us to create a partially evaluated
    function that wraps the QNode. For example,

    >>> y = np.array([0.2, 0.3])
    >>> batched_partial_circuit = qml.batch_partial(circuit, y=y)

    The unevaluated arguments of the resulting function must now have a batch
    dimension, and the output of the function also has a batch dimension:

    >>> batch_size = 4
    >>> x = np.linspace(0.1, 0.5, batch_size)
    >>> batched_partial_circuit(x)
    tensor([0.9316158 , 0.91092081, 0.87405565, 0.82167473], requires_grad=True)

    Gradients can be computed for the arguments of the wrapper function, but
    not for any partially evaluated arguments passed to ``qml.batch_partial``:

    >>> qml.jacobian(batched_partial_circuit)(x)
    array([[-0.09347337,  0.        ,  0.        ,  0.        ],
           [ 0.        , -0.21649144,  0.        ,  0.        ],
           [ 0.        ,  0.        , -0.33566648,  0.        ],
           [ 0.        ,  0.        ,  0.        , -0.44888295]])
    """
    qnode = qml.batch_params(qnode, all_operations=all_operations)

    # store whether this decorator is being used as a pure
    # analog of functools.partial, or whether it is used to
    # wrap a QNode in a more complex lambda statement
    is_partial = False
    if not any(callable(val) for val in partial_kwargs.values()):
        # none of the kwargs passed in are callable
        is_partial = True

    sig = inspect.signature(qnode).parameters
    if is_partial:
        # the partially evaluated function must have at least one more
        # parameter, otherwise batching doesn't make sense
        if len(sig) <= len(partial_kwargs):
            raise ValueError("Partial evaluation must leave at least one unevaluated parameter")
    else:
        # if used to wrap a QNode in a lambda statement, then check that
        # all arguments are provided
        if len(sig) > len(partial_kwargs):
            raise ValueError("Callable argument requires all other arguments to QNode be provided")

    @functools.wraps(qnode)
    def wrapper(*args, **kwargs):

        # raise an error if keyword arguments are passed, since the
        # arguments are passed to the lambda statement instead of the QNode
        if not is_partial and kwargs:
            raise ValueError(
                "Arguments must not be passed as keyword arguments to "
                "callable within partial function"
            )

        # get the batch dimension (we don't have to check if all arguments
        # have the same batch dim since that's done in qml.batch_params)
        try:
            if args:
                batch_dim = qml.math.shape(args[0])[0]
            else:
                batch_dim = qml.math.shape(list(kwargs.values())[0])[0]
        except IndexError:
            raise ValueError("Batch dimension must be provided") from None

        for key, val in partial_kwargs.items():
            if callable(val):
                val = qml.math.stack([val(*a) for a in zip(*args)])
                kwargs[key] = val
            else:
                kwargs[key] = qml.math.stack([val] * batch_dim)

        if is_partial:
            return qnode(*_convert_to_args(qnode, args, kwargs))

        # don't pass the arguments to the lambda itself into the QNode
        return qnode(*_convert_to_args(qnode, (), kwargs))

    return wrapper
