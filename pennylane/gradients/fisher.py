# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains functions for computing classical and quantum fisher information matrices."""
# pylint: disable=import-outside-toplevel, not-callable
from collections.abc import Callable, Sequence
from functools import partial

import pennylane as qml
from pennylane import transform
from pennylane.devices import DefaultQubit, DefaultQubitLegacy
from pennylane.gradients import adjoint_metric_tensor


# TODO: create qml.math.jacobian and replace it here
def _torch_jac(circ):
    """Torch jacobian as a callable function"""
    import torch

    def wrapper(*args, **kwargs):
        loss = partial(circ, **kwargs)
        if len(args) > 1:
            return torch.autograd.functional.jacobian(loss, args, create_graph=True)
        return torch.autograd.functional.jacobian(loss, *args, create_graph=True)

    return wrapper


# TODO: create qml.math.jacobian and replace it here
def _tf_jac(circ):
    """TF jacobian as a callable function"""
    import tensorflow as tf

    def wrapper(*args, **kwargs):
        with tf.GradientTape() as tape:
            loss = circ(*args, **kwargs)
        return tape.jacobian(loss, args)

    return wrapper


def _compute_cfim(p, dp):
    r"""Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`classical_fisher_{ij} = \sum_\ell (\partial_i p_\ell) (\partial_i p_\ell) / p_\ell`
    """
    # Exclude values where p=0 and calculate 1/p
    nonzeros_p = qml.math.where(p > 0, p, qml.math.ones_like(p))
    one_over_p = qml.math.where(p > 0, qml.math.ones_like(p), qml.math.zeros_like(p))
    one_over_p = one_over_p / nonzeros_p

    # Multiply dp and p
    # Note that casting and being careful about dtypes is necessary as interfaces
    # typically treat derivatives (dp) with float32, while standard execution (p) comes in float64
    dp = qml.math.cast_like(dp, p)
    dp = qml.math.reshape(
        dp, (len(p), -1)
    )  # Squeeze does not work, as you could have shape (num_probs, num_params) with num_params = 1
    dp_over_p = qml.math.transpose(dp) * one_over_p  # creates (n_params, n_probs) array

    # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)
    return dp_over_p @ dp


@transform
def _make_probs(tape: qml.tape.QuantumTape) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
    """Ignores the return types of the provided circuit and creates a new one
    that outputs probabilities"""
    qscript = qml.tape.QuantumScript(tape.operations, [qml.probs(tape.wires)], shots=tape.shots)

    def post_processing_fn(res):
        # only a single probs measurement, so no stacking needed
        return res[0]

    return [qscript], post_processing_fn


def classical_fisher(qnode, argnums=0):
    r"""Returns a function that computes the classical fisher information matrix (CFIM) of a given :class:`.QNode` or
    quantum tape.

    Given a parametrized (classical) probability distribution :math:`p(\bm{\theta})`, the classical fisher information
    matrix quantifies how changes to the parameters :math:`\bm{\theta}` are reflected in the probability distribution.
    For a parametrized quantum state, we apply the concept of classical fisher information to the computational
    basis measurement.
    More explicitly, this function implements eq. (15) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_:

    .. math::

        \text{CFIM}_{i, j} = \sum_{\ell=0}^{2^N-1} \frac{1}{p_\ell(\bm{\theta})} \frac{\partial p_\ell(\bm{\theta})}{
        \partial \theta_i} \frac{\partial p_\ell(\bm{\theta})}{\partial \theta_j}

    for :math:`N` qubits.

    Args:
        tape (:class:`.QNode` or qml.QuantumTape): A :class:`.QNode` or quantum tape that may have arbitrary return types.
        argnums (Optional[int or List[int]]): Arguments to be differentiated in case interface ``jax`` is used.

    Returns:
        func: The function that computes the classical fisher information matrix. This function accepts the same
        signature as the :class:`.QNode`. If the signature contains one differentiable variable ``params``, the function
        returns a matrix of size ``(len(params), len(params))``. For multiple differentiable arguments ``x, y, z``,
        it returns a list of sizes ``[(len(x), len(x)), (len(y), len(y)), (len(z), len(z))]``.


    .. seealso:: :func:`~.pennylane.metric_tensor`, :func:`~.pennylane.gradient.transforms.quantum_fisher`

    **Example**

    First, let us define a parametrized quantum state and return its (classical) probability distribution for all
    computational basis elements:

    .. code-block:: python

        import pennylane.numpy as pnp

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.CNOT([0, 1])
            qml.CRY(params[1], wires=[1, 0])
            qml.Hadamard(1)
            return qml.probs(wires=[0, 1])

    Executing this circuit yields the ``2**2=4`` elements of :math:`p_\ell(\bm{\theta})`

    >>> pnp.random.seed(25)
    >>> params = pnp.random.random(2)
    >>> circ(params)
    [0.41850088 0.41850088 0.08149912 0.08149912]

    We can obtain its ``(2, 2)`` classical fisher information matrix (CFIM) by simply calling the function returned
    by ``classical_fisher()``:

    >>> cfim_func = qml.gradient.classical_fisher(circ)
    >>> cfim_func(params)
    [[ 0.901561 -0.125558]
     [-0.125558  0.017486]]

    This function has the same signature as the :class:`.QNode`. Here is a small example with multiple arguments:

    .. code-block:: python

        @qml.qnode(dev)
        def circ(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.probs(wires=range(n_wires))

    >>> x, y = pnp.array([0.5, 0.6], requires_grad=True)
    >>> circ(x, y)
    [0.86215007 0.         0.13784993 0.        ]
    >>> qml.gradient.classical_fisher(circ)(x, y)
    [array([[0.32934729]]), array([[0.51650396]])]

    Note how in the case of multiple variables we get a list of matrices with sizes
    ``[(n_params0, n_params0), (n_params1, n_params1)]``, which in this case is simply two ``(1, 1)`` matrices.


    A typical setting where the classical fisher information matrix is used is in variational quantum algorithms.
    Closely related to the `quantum natural gradient <https://arxiv.org/abs/1909.02108>`_, which employs the
    `quantum` fisher information matrix, we can compute a rescaled gradient using the CFIM. In this scenario,
    typically a Hamiltonian objective function :math:`\langle H \rangle` is minimized:

    .. code-block:: python

        H = qml.Hamiltonian(coeffs=[0.5, 0.5], observables=[qml.Z(0), qml.Z(1)])

        @qml.qnode(dev)
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RX(params[2], wires=1)
            qml.RY(params[3], wires=1)
            qml.CNOT(wires=(0,1))
            return qml.expval(H)

        params = pnp.random.random(4)

    We can compute both the gradient of :math:`\langle H \rangle` and the CFIM with the same :class:`.QNode` ``circ``
    in this example since ``classical_fisher()`` ignores the return types and assumes ``qml.probs()`` for all wires.

    >>> grad = qml.grad(circ)(params)
    >>> cfim = qml.gradient.classical_fisher(circ)(params)
    >>> print(grad.shape, cfim.shape)
    (4,) (4, 4)

    Combined together, we can get a rescaled gradient to be employed for optimization schemes like natural gradient
    descent.

    >>> rescaled_grad = cfim @ grad
    >>> print(rescaled_grad)
    [-0.66772533 -0.16618756 -0.05865127 -0.06696078]

    The ``classical_fisher`` matrix itself is again differentiable:

    .. code-block:: python

        @qml.qnode(dev)
        def circ(params):
            qml.RX(qml.math.cos(params[0]), wires=0)
            qml.RX(qml.math.cos(params[0]), wires=1)
            qml.RX(qml.math.cos(params[1]), wires=0)
            qml.RX(qml.math.cos(params[1]), wires=1)
            return qml.probs(wires=range(2))

        params = pnp.random.random(2)

    >>> qml.gradient.classical_fisher(circ)(params)
    [[4.18575068e-06 2.34443943e-03]
     [2.34443943e-03 1.31312079e+00]]
    >>> qml.jacobian(qml.gradient.classical_fisher(circ))(params)
    array([[[9.98030491e-01, 3.46944695e-18],
            [1.36541817e-01, 5.15248592e-01]],
           [[1.36541817e-01, 5.15248592e-01],
            [2.16840434e-18, 2.81967252e-01]]]))

    """
    new_qnode = _make_probs(qnode)

    def wrapper(*args, **kwargs):
        old_interface = qnode.interface

        if old_interface == "auto":
            qnode.interface = qml.math.get_interface(*args, *list(kwargs.values()))

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

        j = jac(*args, **kwargs)
        p = new_qnode(*args, **kwargs)

        if old_interface == "auto":
            qnode.interface = "auto"

        # In case multiple variables are used, we create a list of cfi matrices
        if isinstance(j, tuple):
            res = []
            for j_i in j:
                res.append(_compute_cfim(p, j_i))

            if len(j) == 1:
                return res[0]

            return res

        return _compute_cfim(p, j)

    return wrapper


@partial(transform, is_informative=True)
def quantum_fisher(
    tape: qml.tape.QuantumTape, device, *args, **kwargs
) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
    r"""Returns a function that computes the quantum fisher information matrix (QFIM) of a given :class:`.QNode`.

    Given a parametrized quantum state :math:`|\psi(\bm{\theta})\rangle`, the quantum fisher information matrix (QFIM) quantifies how changes to the parameters :math:`\bm{\theta}`
    are reflected in the quantum state. The metric used to induce the QFIM is the fidelity :math:`f = |\langle \psi | \psi' \rangle|^2` between two (pure) quantum states.
    This leads to the following definition of the QFIM (see eq. (27) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_):

    .. math::

        \text{QFIM}_{i, j} = 4 \text{Re}\left[ \langle \partial_i \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle
        - \langle \partial_i \psi(\bm{\theta}) | \psi(\bm{\theta}) \rangle \langle \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle \right]

    with short notation :math:`| \partial_j \psi(\bm{\theta}) \rangle := \frac{\partial}{\partial \theta_j}| \psi(\bm{\theta}) \rangle`.

    .. seealso::
        :func:`~.pennylane.metric_tensor`, :func:`~.pennylane.adjoint_metric_tensor`, :func:`~.pennylane.gradient.transforms.classical_fisher`

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit that may have arbitrary return types.
        *args: In case finite shots are used, further arguments according to :func:`~.pennylane.metric_tensor` may be passed.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the quantum Fisher information in the form of a tensor.

    .. note::

        ``quantum_fisher`` coincides with the ``metric_tensor`` with a prefactor of :math:`4`. Internally, :func:`~.pennylane.adjoint_metric_tensor` is used when executing on a device with
        exact expectations (``shots=None``) that inherits from ``"default.qubit"``. In all other cases, i.e. if a device with finite shots is used, the hardware compatible transform :func:`~.pennylane.metric_tensor` is used.
        Please refer to their respective documentations for details on the arguments.

    **Example**

    The quantum Fisher information matrix (QIFM) can be used to compute the `natural` gradient for `Quantum Natural Gradient Descent <https://arxiv.org/abs/1909.02108>`_.
    A typical scenario is optimizing the expectation value of a Hamiltonian:

    .. code-block:: python

        n_wires = 2

        dev = qml.device("default.qubit", wires=n_wires)

        H = 1.*qml.X(0) @ qml.X(1) - 0.5 * qml.Z(1)

        @qml.qnode(dev)
        def circ(params):
            qml.RY(params[0], wires=1)
            qml.CNOT(wires=(1,0))
            qml.RY(params[1], wires=1)
            qml.RZ(params[2], wires=1)
            return qml.expval(H)

        params = pnp.array([0.5, 1., 0.2], requires_grad=True)

    The natural gradient is then simply the QFIM multiplied by the gradient:

    >>> grad = qml.grad(circ)(params)
    >>> grad
    [ 0.59422561 -0.02615095 -0.05146226]
    >>> qfim = qml.gradient.quantum_fisher(circ)(params)
    >>> qfim
    [[1.         0.         0.        ]
     [0.         1.         0.        ]
     [0.         0.         0.77517241]]
    >>> qfim @ grad
    tensor([ 0.59422561, -0.02615095, -0.03989212], requires_grad=True)

    When using real hardware or finite shots, ``quantum_fisher`` is internally calling :func:`~.pennylane.metric_tensor`.
    To obtain the full QFIM, we need an auxilary wire to perform the Hadamard test.

    >>> dev = qml.device("default.qubit", wires=n_wires+1, shots=1000)
    >>> @qml.qnode(dev)
    ... def circ(params):
    ...     qml.RY(params[0], wires=1)
    ...     qml.CNOT(wires=(1,0))
    ...     qml.RY(params[1], wires=1)
    ...     qml.RZ(params[2], wires=1)
    ...     return qml.expval(H)
    >>> qfim = qml.gradient.quantum_fisher(circ)(params)

    Alternatively, we can fall back on the block-diagonal QFIM without the additional wire.

    >>> qfim = qml.gradient.quantum_fisher(circ, approx="block-diag")(params)

    """

    if device.shots or not isinstance(device, (DefaultQubitLegacy, DefaultQubit)):
        tapes, processing_fn = qml.gradients.metric_tensor(tape, *args, **kwargs)

        def processing_fn_multiply(res):
            res = qml.execute(res, device=device)
            return 4 * processing_fn(res)

        return tapes, processing_fn_multiply

    res = adjoint_metric_tensor(tape, *args, **kwargs)

    def processing_fn_multiply(r):  # pylint: disable=function-redefined
        r = qml.math.stack(r)
        return 4 * r

    return res, processing_fn_multiply


@quantum_fisher.custom_qnode_transform
def qnode_execution_wrapper(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to take into account that classical processing may be present
    inside the QNode."""

    tkwargs["device"] = qnode.device

    return self.default_qnode_transform(qnode, targs, tkwargs)
