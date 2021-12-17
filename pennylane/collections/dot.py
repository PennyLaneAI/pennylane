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
Contains functions to implement the dot product between QNode collections
"""
# pylint: disable=too-many-arguments,import-outside-toplevel


def _get_dot_func(interface, x=None):
    """Helper function for :func:`~.dot` to determine
    the correct dot product function depending on the QNodeCollection
    interface.

    Args:
        interface (str): the interface to get the dot product function for
        x (Sequence): A non-QNodeCollection sequence. If it isn't the correct
            type for the interface, it is automatically converted.

    Returns:
        tuple[callable, Sequence or torch.Tensor or tf.Variable]: a tuple
        containing the required dot product function, as well as the
        (potentially converted) sequence.
    """
    if interface == "tf":
        import tensorflow as tf

        if x is not None and not isinstance(x, (tf.Tensor, tf.Variable)):
            x = tf.Variable(x, dtype=tf.float64)

        return lambda a, b: tf.tensordot(a, b, 1), x

    if interface == "torch":
        import torch

        if x is not None and not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)

        return torch.matmul, x

    if interface in ("autograd", "numpy"):
        from autograd import numpy as np

        if x is not None and not isinstance(x, np.ndarray):
            x = np.array(x)

        return np.dot, x

    if interface == "jax":
        import jax.numpy as jnp

        if x is not None and not isinstance(x, jnp.ndarray):
            x = jnp.array(x)

        return jnp.dot, x

    if interface is None:
        import numpy as np

        return np.dot, x

    raise ValueError(f"Unknown interface {interface}")


def dot(x, y):
    r"""Lazily perform the dot product between arrays, tensors, and :class:`QNodeCollection`.

    Using this function, lazy dot products can be computed between two :class:`QNodeCollection`
    objects, or a :class:`QNodeCollection` object and an array/tensor object. In the latter
    case, only one-dimensional arrays/tensors are supported.

    Args:
        x (array or tensor or QNodeCollection): A QNode collection of independent QNodes,
            or an array/tensor object.
        y (array or tensor or QNodeCollection): A QNode collection of independent QNodes,
            or an array/tensor object.

    .. seealso:: :func:`~.apply`, :func:`~.sum`

    **Example:**

    We can create a QNodeCollection using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    The returned QNodeCollection contains 2 QNodes, as we mapped the :func:`~.StronglyEntanglingLayers`
    over a list of two observables:

    >>> len(qnodes)
    2

    For the cost function, we now perform the dot product between a vector of coefficients
    and the QNodeCollection:

    >>> coeffs = torch.tensor([0.32, -0.2], dtype=torch.double)
    >>> cost = qml.dot(coeffs, qnodes)

    .. note::

        The ``cost`` function is equivalent to computing :math:`\langle 0 | U(\theta)^\dagger H U(\theta) | 0\rangle`
        where

        * :math:`U(\theta)` is the unitary applied by the strongly entangling layers, and
        * :math:`H = 0.32 X\otimes Z - 0.2 Z \otimes Z`.

    This is a lazy dot product --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=3, n_wires=2)
    >>> x = np.random.random(shape) # generate random parameters
    >>> cost(x)
    tensor(-0.2183, dtype=torch.float64, grad_fn=<DotBackward>)
    """
    if hasattr(x, "interface") and hasattr(y, "interface"):

        if x.interface != y.interface:
            raise ValueError("QNodeCollections have non-matching interfaces")

        interface = x.interface
        fn, _ = _get_dot_func(interface)
        func = lambda params, **kwargs: fn(x(params, **kwargs), y(params, **kwargs))

    elif hasattr(x, "interface"):
        interface = x.interface
        fn, y = _get_dot_func(interface, y)
        func = lambda params, **kwargs: fn(x(params, **kwargs), y)

    elif hasattr(y, "interface"):
        interface = y.interface
        fn, x = _get_dot_func(interface, x)
        func = lambda params, **kwargs: fn(x, y(params, **kwargs))

    else:
        raise ValueError("At least one argument must be a QNodeCollection")

    func.interface = interface
    return func
