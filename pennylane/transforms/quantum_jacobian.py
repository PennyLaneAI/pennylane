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
Contains the quantum Jacobian transform
"""
# pylint: disable=import-outside-toplevel
from functools import partial

import pennylane as qml


def quantum_jacobian(qnode):
    r"""Returns a function to extract the Jacobian
    matrix of the quantum part of a QNode.

    This transform allows the Jacobian of the QNode output with respect
    to quantum gate arguments to be extracted.

    Args:
        qnode (.QNode): QNode to compute the (quantum) Jacobian of

    Returns:
        function: Function which accepts the same arguments as the QNode.
        When called, this function will return the Jacobian of the QNode output
        with respect to the gate arguments (*not* the QNode arugments).

    **Example**

    Consider the following QNode:

    >>> @qml.qnode(dev)
    ... def circuit(weights):
    ...     qml.RX(weights[0], wires=0)
    ...     qml.RY(weights[0], wires=1)
    ...     qml.RZ(weights[1] ** 2, wires=1)
    ...     return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))

    We can use this transform to extract the Jacobian of :math:`f: \mathbb{R}^n \rightarrow
    \mathbb{R}^m`, a function that maps the gate arguments :math:`x` to the QNode output:

    >>> qjac_fn = qml.transforms.quantum_jacobian(circuit)
    >>> weights = np.array([1., 1.], requires_grad=True)
    >>> qjac = qjac_fn(weights)
    >>> print(np.round(qjac, 4))
    [[-0.8415  0.      0.    ]
     [ 0.      0.2919 -0.7081]]

    The returned Jacobian has rows corresponding to the QNode output, and columns
    corresponding to gate arguments.

    .. UsageDetails::

        Since QNodes may consist of both quantum and classical processing, we
        may use the ``quantum_jacobian`` transform alongside the :func:`~.classical_jacobian`
        function to reconstruct the Jacobian of the QNode output with respect to QNode
        inputs.

        For example, using the same QNode as above,

        >>> cjac_fn = qml.transforms.classical_jacobian(circuit)
        >>> cjac = cjac_fn(weights)
        >>> print(cjac)
        [[1. 0.]
         [1. 0.]
         [0. 2.]]

        The Jacobian of the QNode with respect to its arguments can be reconstructed
        simply by matrix multiplying the quantum and classical Jacobians:

        >>> np.round(qjac @ cjac, 4)
        tensor([[-0.8415,  0.    ],
                [ 0.2919, -1.4161]], requires_grad=True)

        Let's verify this result against the :func:`~.jacobian` function:

        >>> jac = qml.jacobian(circuit)(weights)
        >>> np.round(jac, 4)
        tensor([[-0.8415,  0.    ],
                [ 0.2919, -1.4161]], requires_grad=True)
    """
    if qnode.__class__.__name__ == "ExpvalCost":
        qnodes = qnode.qnodes
        coeffs = qnode.hamiltonian.coeffs

        def _jacobian(*args, **kwargs):
            jacs = [quantum_jacobian(q)(*args, **kwargs) for q in qnodes]
            return qml.math.sum([c * j for c, j in zip(coeffs, jacs)], axis=0)

        return _jacobian

    dev = qnode.device

    def quantum_processing(*args, **kwargs):
        """A function that isolates solely the quantum processing of a QNode."""
        qnode.construct(args, kwargs)
        params = qml.math.stack(qnode.qtape.get_parameters())

        def func(params):
            return qml.math.stack(qnode.qtape.execute(params=params, device=qnode.device))

        return params, func

    def _jacobian(jac_fn, *args, **kwargs):
        """Computes the Jacobian of the quantum_processing function.

        Args:
            jac_fn (function): function with signature ``jac_fn(func, params)``
                that computes the Jacobian of a function ``func`` at parameter
                values ``params``.
            *args: QNode arguments
            **kwargs: QNode keyword arguments

        Returns:
            tensor_like[float]: the Jacobian of the quantum_processing function.
        """
        params, func = quantum_processing(*args, **kwargs)
        return jac_fn(func, params)

    if qnode.interface == "autograd":
        return partial(_jacobian, lambda f, p: qml.jacobian(f)(p))

    if qnode.interface == "torch":
        import torch

        return partial(_jacobian, torch.autograd.functional.jacobian)

    if qnode.interface == "jax":
        import jax

        return partial(_jacobian, lambda f, p: jax.jacobian(f)(p))

    if qnode.interface == "tf":
        import tensorflow as tf

        def _jacobian(*args, **kwargs):
            params, func = quantum_processing(*args, **kwargs)

            with tf.GradientTape() as tape:
                tape.watch(params)
                res = func(params)

            return qml.math.stack(tape.jacobian(res, params))

        return _jacobian
