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
Contains the batch dimension transform for partial use of QNodes.
"""
import functools
import inspect

import pennylane as qml


def _convert_to_args(sig, args, kwargs):
    """
    Given the signature of a function, convert the positional and
    keyword arguments to purely positional arguments.
    """
    new_args = []
    for i, param in enumerate(sig):
        if param in kwargs:
            # first check if the name is provided in the keyword arguments
            new_args.append(kwargs[param])
        else:
            # if not, then the argument must be positional
            new_args.append(args[i])

    return tuple(new_args)


def batch_partial(qnode, all_operations=False, preprocess=None, **partial_kwargs):
    """
    Create a batched partial callable object from the QNode specified.

    This transform provides functionality akin to ``functools.partial`` and
    allows batching the arguments used for calling the batched partial object.

    Args:
        qnode (pennylane.QNode): QNode to pre-supply arguments to
        all_operations (bool): If ``True``, a batch dimension will be added to *all* operations
            in the QNode, rather than just trainable QNode parameters.
        preprocess (dict): If provided, maps every QNode argument name to a preprocessing
            function. When the returned partial function is called, the arguments are
            first passed to the preprocessing functions, and the return values are
            passed to the QNode.
        partial_kwargs (dict): pre-supplied arguments to pass to the QNode.

    Returns:
        function: Function which wraps the QNode and accepts the same arguments minus the
        pre-supplied arguments provided. The first dimension of each argument of the
        wrapper function will be treated as a batch dimension.

    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

    The ``qml.batch_partial`` decorator allows us to create a partial callable
    object that wraps the QNode. For example,

    >>> y = np.array(0.2)
    >>> batched_partial_circuit = qml.batch_partial(circuit, y=y)

    The unevaluated arguments of the resulting function must now have a batch
    dimension, and the output of the function also has a batch dimension:

    >>> batch_size = 4
    >>> x = np.linspace(0.1, 0.5, batch_size)
    >>> batched_partial_circuit(x)
    tensor([0.97517033, 0.95350781, 0.91491915, 0.86008934], requires_grad=True)

    Jacobians can be computed for the arguments of the wrapper function, but
    not for any pre-supplied argument passed to ``qml.batch_partial``:

    >>> qml.jacobian(batched_partial_circuit)(x)
    array([[-0.0978434 ,  0.        ,  0.        ,  0.        ],
           [ 0.        , -0.22661276,  0.        ,  0.        ],
           [ 0.        ,  0.        , -0.35135943,  0.        ],
           [ 0.        ,  0.        ,  0.        , -0.46986895]])

    The same ``qml.batch_partial`` function can also be used to replace arguments
    of a QNode with functions, and calling the wrapper would evaluate
    those functions and pass the results into the QNode. For example,

    >>> x = np.array(0.1)
    >>> y_fn = lambda y0: y0 * 0.2 + 0.3
    >>> batched_lambda_circuit = qml.batch_partial(circuit, x=x, preprocess={"y": y_fn})

    The wrapped function ``batched_lambda_circuit`` also expects arguments to
    have an initial batch dimension:

    >>> batch_size = 4
    >>> y0 = np.linspace(0.5, 2, batch_size)
    >>> batched_lambda_circuit(y0)
    tensor([0.91645953, 0.8731983 , 0.82121237, 0.76102116], requires_grad=True)

    Jacobians can be computed in this scenario as well:

    >>> qml.jacobian(batched_lambda_circuit)(y0)
    array([[-0.07749457,  0.        ,  0.        ,  0.        ],
           [ 0.        , -0.09540608,  0.        ,  0.        ],
           [ 0.        ,  0.        , -0.11236432,  0.        ],
           [ 0.        ,  0.        ,  0.        , -0.12819986]])
    """
    qnode = qml.batch_params(qnode, all_operations=all_operations)

    preprocess = {} if preprocess is None else preprocess

    # store whether this decorator is being used as a pure
    # analog of functools.partial, or whether it is used to
    # wrap a QNode in a more complex lambda statement
    is_partial = preprocess == {}

    # determine which arguments need to be stacked along the batch dimension
    to_stack = []
    for key, val in partial_kwargs.items():
        try:
            # check if the value is a tensor
            if qml.math.asarray(val).dtype != object:
                to_stack.append(key)
        except ImportError:
            # autoray can't find a backend for val, so it cannot be stacked
            pass

    sig = inspect.signature(qnode).parameters
    if is_partial:
        # the batched partial function must have at least one more
        # parameter, otherwise batching doesn't make sense
        if len(sig) <= len(partial_kwargs):
            raise ValueError("Partial evaluation must leave at least one unevaluated parameter")
    else:
        # if used to wrap a QNode in a lambda statement, then check that
        # all arguments are provided
        if len(sig) > len(partial_kwargs) + len(preprocess):
            raise ValueError("Callable argument requires all other arguments to QNode be provided")

    @functools.wraps(qnode)
    def wrapper(*args, **kwargs):
        # pylint: disable=not-callable

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
            raise ValueError("Parameter with batch dimension must be provided") from None

        for key, val in preprocess.items():
            unstacked_args = (qml.math.unstack(arg) for arg in args)
            val = qml.math.stack([val(*a) for a in zip(*unstacked_args)])
            kwargs[key] = val

        for key, val in partial_kwargs.items():
            if key in to_stack:
                kwargs[key] = qml.math.stack([val] * batch_dim)
            else:
                kwargs[key] = val

        if is_partial:
            return qnode(*_convert_to_args(sig, args, kwargs))

        # don't pass the arguments to the lambda itself into the QNode
        return qnode(*_convert_to_args(sig, (), kwargs))

    return wrapper
