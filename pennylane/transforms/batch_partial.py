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
import copy
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
        else:
            raise ValueError(f"Argument {param} must be provided")

    return tuple(new_args)


def batch_partial(qnode, **partial_kwargs):
    qnode = qml.batch_params(qnode)

    # store whether this decorator is being used as a pure
    # analog of functools.partial, or whether it is used to
    # wrap a QNode in a more complex lambda statement
    is_partial = False
    if not any([callable(val) for val in partial_kwargs.values()]):
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
        if args:
            batch_dim = qml.math.shape(args[0])[0]
        else:
            batch_dim = qml.math.shape(list(kwargs.values())[0])[0]

        for key, val in partial_kwargs.items():
            if callable(val):
                val = qml.math.stack([val(*a) for a in zip(*args)])
                kwargs[key] = val
            else:
                kwargs[key] = qml.math.stack([val] * batch_dim)

        if is_partial:
            return qnode(*_convert_to_args(qnode, args, kwargs))
        else:
            # don't pass the arguments to the lambda itself into the QNode
            return qnode(*_convert_to_args(qnode, (), kwargs))

    return wrapper
