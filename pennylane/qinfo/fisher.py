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
"""Differentiable classical fisher information"""
# pylint: disable=import-outside-toplevel, not-callable
import functools
import pennylane as qml

from pennylane.transforms import batch_transform

# TODO: create qml.math.jacobian and replace it here
def _torch_jac(circ):
    """Torch jacobian as a callable function"""
    import torch

    def wrapper(*args, **kwargs):
        loss = functools.partial(circ, **kwargs)
        if len(args) > 1:
            return torch.autograd.functional.jacobian(loss, (args))
        return torch.autograd.functional.jacobian(loss, *args)

    return wrapper


# TODO: create qml.math.jacobian and replace it here
def _tf_jac(circ):
    """TF jacobian as a callable function"""
    import tensorflow as tf

    def wrapper(*args, **kwargs):
        with tf.GradientTape() as tape:
            loss = circ(*args, **kwargs)
        return tape.jacobian(loss, (args))

    return wrapper


def classical_fisher(qnode, argnums=0):
    r"""Returns a function that computes the classical fisher information matrix (CFIM) of a given :class:`.QNode` or quantum tape.

    Given a parametrized (classical) probability distribution :math:`p(\bm{\theta})`, the classical fisher information matrix quantifies how changes to the parameters :math:`\bm{\theta}`
    are reflected in the probability distribution. For a parametrized quantum state, we apply the concept of classical fisher information to the computational
    basis measurement.
    More explicitly, this function implements eq. (15) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_:

    .. math::

        \text{CFIM}_{i, j} = \sum_{\ell=0}^{2^N-1} \frac{1}{p_\ell(\bm{\theta})} \frac{\partial p_\ell(\bm{\theta})}{\partial \theta_i} \frac{\partial p_\ell(\bm{\theta})}{\partial \theta_j}

    for :math:`N` qubits.

    Args:
        tape (:class:`.QNode` or qml.QuantumTape): A :class:`.QNode` or quantum tape that may have arbitrary return types.

    Returns:
        func: The function that computes the classical fisher information matrix. This function accepts the same signature as the :class:`.QNode`.

    .. warning::

        In its current form, this functionality is not hardware compatible and can only be used by simulators.

.. seealso:: :func:`metric_tensor`, :func:`qinfo.quantum_fisher`

    :func:`metric_tensor`, :func:`qinfo.quantum_fisher`

    **Example**

    First, let us define a parametrized quantum state and return its (classical) probability distribution for all computational basis elements:

    .. code-block:: python

        import pennylane.numpy as pnp
        n_wires = 2

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=0)
            qml.CNOT(wires=(0,1))
            return qml.probs(wires=range(n_wires))

    Executing this circuit yields the ``2**n_wires`` elements of :math:`p_\ell(\bm{\theta})`

    >>> params = pnp.random.random(2)
    >>> circ(params)
    tensor([0.77708372, 0.        , 0.        , 0.22291628], requires_grad=True)

    We can obtain its ``(2, 2)`` classical fisher information matrix (CFIM) by simply calling the function returned by ``classical_fisher()``:

    >>> cfim_func = qml.qinfo.classical_fisher(circ)
    >>> cfim_func(params)
    tensor([[1., 1.],
        [1., 1.]], requires_grad=True)

    This function has the same signature as the :class:`.QNode`. Here is a small example with multiple arguments:

    .. code-block:: python

        @qml.qnode(dev)
        def circ(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.probs(wires=range(n_wires))

    >>> x, y = pnp.array([0.5, 0.6], requires_grad=True)
    >>> circ(x, y)
    (tensor([0.87380224, 0.        , 0.12619776, 0.        ], requires_grad=True)
    >>> qml.qinfo.classical_fisher(circ)(x, y)
     [tensor([[0.15828019]], requires_grad=True),
      tensor([[0.74825326]], requires_grad=True)])

    Note how in the case of multiple variables we get a list of matrices with sizes
    ``[(n_params0, n_params0), (n_params1, n_params1)]``, which in this case is simply two ``(1, 1)`` matrices.


    A typical setting where the classical fisher information matrix is used is in variational quantum algorithms.
    Closely related to the `quantum natural gradient <https://arxiv.org/abs/1909.02108>`_, which employs the `quantum` fisher information matrix,
    we can compute a rescaled gradient using the CFIM. In this scenario, typically a Hamiltonian objective function :math:`\langle H \rangle` is minimized:

    .. code-block:: python

        H = qml.Hamiltonian(coeffs = [0.5, 0.5], ops = [qml.PauliZ(0), qml.PauliZ(1)])

        @qml.qnode(dev)
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RX(params[2], wires=1)
            qml.RY(params[3], wires=1)
            qml.CNOT(wires=(0,1))
            return qml.expval(H)

        params = pnp.random.random(4)

    We can compute both the gradient of :math:`\langle H \rangle` and the CFIM with the same :class:`.QNode` ``circ`` in this example since ``classical_fisher()`` ignores the return types
    and assumes ``qml.probs()`` for all wires.

    >>> grad = qml.grad(circ)(params)
    >>> cfim = qml.qinfo.classical_fisher(circ)(params)
    >>> print(grad.shape, cfim.shape)
    (4,) (4, 4)

    Combined together, we can get a rescaled gradient to be employed for optimization schemes like natural gradient descent.

    >>> rescaled_grad = cfim @ grad
    >>> print(rescaled_grad)
    [-0.66772533 -0.16618756 -0.05865127 -0.06696078]

    """
    new_qnode = _make_probs(qnode, post_processing_fn=lambda x: qml.math.squeeze(qml.math.stack(x)))

    interface = qnode.interface

    if interface in ("jax", "jax-jit"):
        import jax

        jac = jax.jacobian(new_qnode, argnums=argnums)

    if interface == "torch":
        jac = _torch_jac(new_qnode)

    if interface == "autograd":
        jac = qml.jacobian(new_qnode)

    if interface == "tf":
        jac = _tf_jac(new_qnode)

    def wrapper(*args, **kwargs):
        j = jac(*args, **kwargs)
        p = new_qnode(*args, **kwargs)

        # In case multiple variables are used
        if isinstance(j, tuple) and len(j) > 1:
            res = []
            for j_i in j:
                if interface == "tf":
                    j_i = qml.math.transpose(qml.math.cast(j_i, dtype=p.dtype))

                res.append(_compute_cfim(p, j_i))

            return res

        return _compute_cfim(p, j)

    return wrapper


def _compute_cfim(p, dp):
    r"""Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`classical_fisher_{ij} = \sum_\ell (\partial_i p_\ell) (\partial_i p_\ell) / p_\ell`
    """
    # Note that casting and being careful about dtypes is necessary as interfaces
    # typically treat derivatives (dp) with float32, while standard execution (p) comes in float64

    nonzeros_p = qml.math.where(p > 0, p, qml.math.ones_like(p))
    one_over_p = qml.math.where(p > 0, qml.math.ones_like(p), qml.math.zeros_like(p))
    one_over_p = qml.math.divide(one_over_p, nonzeros_p)

    # Multiply dp and p
    dp = qml.math.cast(dp, dtype=p.dtype)
    dp = qml.math.reshape(
        dp, (len(p), -1)
    )  # Squeeze does not work, as you could have shape (num_probs, num_params) with num_params = 1
    dp_over_p = qml.math.transpose(dp) * one_over_p  # creates (n_params, n_probs) array

    return dp_over_p @ dp  # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)


@batch_transform
def _make_probs(tape, wires=None, post_processing_fn=None):
    """Ignores the return types of any qnode and creates a new one that outputs probabilities"""
    if wires is None:
        wires = tape.wires

    with qml.tape.QuantumTape() as new_tape:
        for op in tape.operations:
            qml.apply(op)
        qml.probs(wires=wires)

    if post_processing_fn is None:
        post_processing_fn = lambda x: qml.math.squeeze(qml.math.stack(x))

    return [new_tape], post_processing_fn
