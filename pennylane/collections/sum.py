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
Contains the sum function, for summing QNode collections.
"""
# pylint: disable=too-many-arguments,import-outside-toplevel

from .apply import apply


def sum(x):
    """Lazily sum the constituent QNodes of a :class:`QNodeCollection`.

    Args:
        x (QNodeCollection): a QNode collection of independent QNodes.

    .. seealso:: :func:`~.apply`, :func:`~.dot`

    **Example:**

    We can create a QNodeCollection using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    For the cost function, we now sum the results of all QNodes in the collection:

    >>> cost = qml.sum(qnodes)

    This is a lazy summation --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> np.random.seed(42)
    >>> shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=3, n_wires=2)
    >>> x = np.random.random(shape)
    >>> cost(x)
    tensor(0.9177, dtype=torch.float64)
    """
    if hasattr(x, "interface") and x.interface is not None:
        if x.interface == "tf":
            import tensorflow as tf

            return apply(tf.reduce_sum, x)

        if x.interface == "torch":
            import torch

            return apply(torch.sum, x)

        if x.interface in ("autograd", "numpy"):
            from autograd import numpy as np

            return apply(np.sum, x)

        if x.interface == "jax":
            import jax.numpy as jnp

            return apply(jnp.sum, x)

        raise ValueError(f"Unknown interface {x.interface}")

    import numpy as np

    return apply(np.sum, x)
