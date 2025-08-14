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
from functools import partial

from pennylane import math
from pennylane._grad import jacobian
from pennylane.devices import DefaultQubit
from pennylane.gradients import adjoint_metric_tensor
from pennylane.gradients.metric_tensor import _contract_metric_tensor_with_cjac
from pennylane.measurements import probs
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms.core import transform
from pennylane.typing import PostprocessingFn
from pennylane.workflow import execute

from .metric_tensor import metric_tensor


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
def _tf_jac(circ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
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
    nonzeros_p = math.where(p > 0, p, math.ones_like(p))
    one_over_p = math.where(p > 0, math.ones_like(p), math.zeros_like(p))
    one_over_p = one_over_p / nonzeros_p

    # Multiply dp and p
    # Note that casting and being careful about dtypes is necessary as interfaces
    # typically treat derivatives (dp) with float32, while standard execution (p) comes in float64
    dp = math.cast_like(dp, p)
    dp = math.reshape(
        dp, (len(p), -1)
    )  # Squeeze does not work, as you could have shape (num_probs, num_params) with num_params = 1
    dp_over_p = math.transpose(dp) * one_over_p  # creates (n_params, n_probs) array

    # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)
    return dp_over_p @ dp


@transform
def _make_probs(
    tape: QuantumScript,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Ignores the return types of the provided circuit and creates a new one
    that outputs probabilities"""
    qscript = QuantumScript(tape.operations, [probs(tape.wires)], shots=tape.shots)

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

        import pennylane.numpy as np

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.CNOT([0, 1])
            qml.CRY(params[1], wires=[1, 0])
            qml.Hadamard(1)
            return qml.probs(wires=[0, 1])

    Executing this circuit yields the ``2**2=4`` elements of :math:`p_\ell(\bm{\theta})`

    >>> np.random.seed(25)
    >>> params = np.random.random(2)
    >>> circ(params)
    tensor([0.41850088, 0.41850088, 0.08149912, 0.08149912], requires_grad=True)

    We can obtain its ``(2, 2)`` classical fisher information matrix (CFIM) by simply calling the function returned
    by ``classical_fisher()``:

    >>> cfim_func = qml.gradients.classical_fisher(circ)
    >>> cfim_func(params)
    tensor([[ 0.90156094, -0.12555804],
            [-0.12555804,  0.01748614]], requires_grad=True)

    This function has the same signature as the :class:`.QNode`. Here is a small example with multiple arguments:

    .. code-block:: python

        @qml.qnode(dev)
        def circ(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.probs(wires=range(1))

    >>> x, y = np.array([0.5, 0.6], requires_grad=True)
    >>> circ(x, y)
    tensor([0.86215007, 0.13784993], requires_grad=True)
    >>> qml.gradients.classical_fisher(circ)(x, y)
    [tensor([[0.32934729]], requires_grad=True),
    tensor([[0.51650396]], requires_grad=True)]

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

        params = np.random.random(4)

    We can compute both the gradient of :math:`\langle H \rangle` and the CFIM with the same :class:`.QNode` ``circ``
    in this example since ``classical_fisher()`` ignores the return types and assumes ``qml.probs()`` for all wires.

    >>> grad = qml.grad(circ)(params)
    >>> cfim = qml.gradients.classical_fisher(circ)(params)
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

        params = np.random.random(2)

    >>> qml.gradients.classical_fisher(circ)(params)
    tensor([[0.86929514, 0.76134441],
            [0.76134441, 0.6667992 ]], requires_grad=True)
    >>> qml.jacobian(qml.gradients.classical_fisher(circ))(params)
    array([[[ 1.98284265e+00, -1.60461922e-16],
            [ 8.68304725e-01,  1.07654307e+00]],
           [[ 8.68304725e-01,  1.07654307e+00],
            [ 7.30752264e-17,  1.88571178e+00]]])

    """
    new_qnode = _make_probs(qnode)

    def wrapper(*args, **kwargs):
        old_interface = qnode.interface

        if old_interface == "auto":
            qnode.interface = math.get_interface(*args, *list(kwargs.values()))

        interface = qnode.interface

        if interface in ("jax", "jax-jit"):
            import jax

            jac = jax.jacobian(new_qnode, argnums=argnums)

        elif interface == "torch":
            jac = _torch_jac(new_qnode)

        elif interface == "autograd":
            jac = jacobian(new_qnode)

        elif (
            interface == "tf"
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            jac = _tf_jac(new_qnode)
        else:
            raise ValueError(
                f"Interface {interface} not supported for jacobian calculations."
            )  # pragma: no cover

        j = jac(*args, **kwargs)
        p = new_qnode(*args, **kwargs)

        if old_interface == "auto":
            qnode.interface = "auto"

        # In case multiple variables are used, we create a list of cfi matrices
        if isinstance(j, tuple):
            res = []
            for j_i in j:
                res.append(_compute_cfim(p, j_i))

            if len(j) == 1:  # pragma: no cover (TensorFlow tests were disabled during deprecation)
                return res[0]

            return res

        return _compute_cfim(p, j)

    return wrapper


@partial(transform, classical_cotransform=_contract_metric_tensor_with_cjac, is_informative=True)
def quantum_fisher(
    tape: QuantumScript, device, *args, **kwargs
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
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

        ``quantum_fisher`` coincides with the ``metric_tensor`` with a prefactor of :math:`4`.
        Internally, :func:`~.pennylane.adjoint_metric_tensor` is used when executing on ``"default.qubit"``
        with exact expectations (``shots=None``). In all other cases, e.g. if a device with finite shots
        is used, the hardware-compatible transform :func:`~.pennylane.metric_tensor` is used, which
        may require an additional wire on the device.
        Please refer to the respective documentations for details.

    **Example**

    The quantum Fisher information matrix (QIFM) can be used to compute the `natural` gradient for `Quantum Natural Gradient Descent <https://arxiv.org/abs/1909.02108>`_.
    A typical scenario is optimizing the expectation value of a Hamiltonian:

    .. code-block:: python

        from pennylane import numpy as np

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

        params = np.array([0.5, 1., 0.2], requires_grad=True)

    The natural gradient is then simply the QFIM multiplied by the gradient:

    >>> grad = qml.grad(circ)(params)
    >>> grad
    array([ 0.59422561, -0.02615095, -0.05146226])
    >>> qfim = qml.gradients.quantum_fisher(circ)(params)
    >>> qfim
    tensor([[1.        , 0.        , 0.        ],
            [0.        , 1.        , 0.        ],
            [0.        , 0.        , 0.77517241]], requires_grad=True)
    >>> qfim @ grad
    tensor([ 0.59422561, -0.02615095, -0.03989212], requires_grad=True)

    When using real hardware or finite shots, ``quantum_fisher`` is internally calling :func:`~.pennylane.metric_tensor`.
    To obtain the full QFIM, we need an auxilary wire to perform the Hadamard test.

    >>> from functools import partial
    >>> dev = qml.device("default.qubit", wires=n_wires+1)
    >>> @partial(qml.set_shots, shots=1000)
    ... @qml.qnode(dev)
    ... def circ(params):
    ...     qml.RY(params[0], wires=1)
    ...     qml.CNOT(wires=(1,0))
    ...     qml.RY(params[1], wires=1)
    ...     qml.RZ(params[2], wires=1)
    ...     return qml.expval(H)
    >>> qfim = qml.gradients.quantum_fisher(circ)(params)

    Alternatively, we can fall back on the block-diagonal QFIM without the additional wire.

    >>> qfim = qml.gradients.quantum_fisher(circ, approx="block-diag")(params)

    """

    if tape.shots or not isinstance(device, DefaultQubit):
        tapes, processing_fn = metric_tensor(tape, *args, **kwargs)

        def processing_fn_multiply(res):
            res = execute(res, device=device)
            return 4 * processing_fn(res)

        return tapes, processing_fn_multiply

    res = adjoint_metric_tensor(tape, *args, **kwargs)

    def processing_fn_multiply(r):  # pylint: disable=function-redefined
        r = math.stack(r)
        return 4 * r

    return res, processing_fn_multiply


@quantum_fisher.custom_qnode_transform
def qnode_execution_wrapper(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to take into account that classical processing may be present
    inside the QNode."""

    tkwargs["device"] = qnode.device

    return self.default_qnode_transform(qnode, targs, tkwargs)
