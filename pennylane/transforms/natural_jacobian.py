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
Contains the quantum natural Jacobian transform
"""
# pylint: disable=import-outside-toplevel
import pennylane as qml


def _get_solve_fn(qnode):
    if qnode.__class__.__name__ == "ExpvalCost":
        qnode = qnode.qnodes.qnodes[0]

    if qnode.interface == "torch":
        import torch

        return lambda A, x: torch.solve(x, A)[0]

    elif qnode.interface == "tf":
        import tensorflow as tf

        return tf.linalg.solve

    elif qnode.interface == "jax":
        from jax import numpy as jnp

        return jnp.linalg.solve

    from pennylane import numpy as np

    return np.linalg.solve


def natural_jacobian(qnode):
    r"""Returns a function to extract the quantum natural Jacobian
    matrix of a QNode.

    Consider a QNode :math:`Q = q \circ f: \mathbb{R}^m\rightarrow \mathbb{R}^p`,
    where:

    - :math:`f: \mathbb{R}^m \rightarrow \mathbb{R}^n` is the function representing the
      transformation from QNode arguments to gate arguments,

    - :math:`q: \mathbb{R}^n \rightarrow \mathbb{R}^p` is the function representing
      the transformation from gate arguments to QNode output, and

    - :math:`g^{-1}` is the pseudo-inverse of the Fubini-Study metric tensor of :math:`q`.

    The quantum natural Jacobian of the QNode is given by:

    .. math:: \mathbf{J}_Q^{\text{natural}}(x) =\mathbf{J}_q(x) ~ g^{-1} ~\mathbf{J}_f(x).

    For more details, see `Quantum Natural Gradient, Stokes et al., (2020)
    <https://arxiv.org/abs/1909.02108>`__.

    Args:
        qnode (.QNode): QNode to compute the quantum natural Jacobian of

    Returns:
        function: Function which accepts the same arguments as the QNode.
        When called, this function will return the quantum natural Jacobian of the QNode output
        with respect to the QNode arugments.

    **Example**

    Consider the following QNode:

    >>> @qml.qnode(dev)
    ... def circuit(weights):
    ...     qml.RX(weights[0], wires=0)
    ...     qml.RY(weights[0], wires=1)
    ...     qml.RZ(weights[1] ** 2, wires=1)
    ...     return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

    We can use this transform to extract the quantum natural Jacobian of
    this QNode:

    >>> nat_jac_fn = qml.transforms.natural_jacobian(circuit)
    >>> weights = np.array([1., 1.], requires_grad=True)
    >>> nat_jac = nat_jac_fn(weights)
    >>> print(np.round(nat_jac, 4))
    [[-3.3659  0.    ]
     [ 1.1677 -8.    ]]

    The returned Jacobian has rows corresponding to the QNode output, and columns
    corresponding to gate arguments.

    .. seealso:: :func:`~.metric_tensor`, :func:`~.quantum_jacobian`, :func:`~.classical_jacobian`
    """
    Jq = qml.transforms.quantum_jacobian(qnode)
    g = qml.metric_tensor(qnode)
    Jf = qml.transforms.classical_jacobian(qnode)
    solve = _get_solve_fn(qnode)

    def _natural_jacobian(*args, **kwargs):
        """Computes the Jacobian of the QNode, by computing
        the metric tensor, quantum Jacobian, and classical Jacobian."""
        mt = g(*args, **kwargs)
        c = Jf(*args, **kwargs)
        q = Jq(*args, **kwargs)

        mt = qml.math.cast_like(mt, q)
        ng = qml.math.T(solve(mt, qml.math.T(q)))
        return qml.math.squeeze(qml.math.dot(ng, c))

    return _natural_jacobian
