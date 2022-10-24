# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""QNode transforms for the quantum information quantities."""
# pylint: disable=import-outside-toplevel,not-callable,unnecessary-lambda-assignment
import functools

import pennylane as qml
from pennylane.devices import DefaultQubit
from pennylane.qnode import QNode
from pennylane.tape import QuantumTape
from pennylane.transforms import adjoint_metric_tensor, batch_transform, metric_tensor


def reduced_dm(qnode, wires):
    """Compute the reduced density matrix from a :class:`~.QNode` returning
    :func:`~.state`.

    Args:
        qnode (QNode): A :class:`~.QNode` returning :func:`~.state`.
        wires (Sequence(int)): List of wires in the considered subsystem.

    Returns:
        func: Function which wraps the QNode and accepts the same arguments. When called, this function will
        return the density matrix.

    **Example**

    .. code-block:: python

        import numpy as np

        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def circuit(x):
          qml.IsingXX(x, wires=[0,1])
          return qml.state()

    >>> reduced_dm(circuit, wires=[0])(np.pi/2)
     [[0.5+0.j 0.+0.j]
      [0.+0.j 0.5+0.j]]

    .. seealso:: :func:`pennylane.density_matrix` and :func:`pennylane.math.reduced_dm`
    """

    def wrapper(*args, **kwargs):
        qnode.construct(args, kwargs)
        return_type = qnode.tape.observables[0].return_type
        if len(qnode.tape.observables) != 1 or not return_type == qml.measurements.State:
            raise ValueError("The qfunc return type needs to be a state.")

        # TODO: optimize given the wires by creating a tape with relevant operations
        state_built = qnode(*args, **kwargs)
        density_matrix = qml.math.reduced_dm(
            state_built, indices=wires, c_dtype=qnode.device.C_DTYPE
        )
        return density_matrix

    return wrapper


def vn_entropy(qnode, wires, base=None):
    r"""Compute the Von Neumann entropy from a :class:`.QNode` returning a :func:`~.state`.

    .. math::
        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    Args:
        qnode (tensor_like): A :class:`.QNode` returning a :func:`~.state`.
        wires (Sequence(int)): List of wires in the considered subsystem.
        base (float): Base for the logarithm, default is None the natural logarithm is used in this case.

    Returns:
        float: Von Neumann entropy of the considered subsystem.

    **Example**

    It is possible to obtain the entropy of a subsystem from a :class:`.QNode` returning a :func:`~.state`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> vn_entropy(circuit, wires=[0])(np.pi/2)
    0.6931472

    The function is differentiable with backpropagation for all interfaces, e.g.:

    >>> param = np.array(np.pi/4, requires_grad=True)
    >>> qml.grad(vn_entropy(circuit, wires=[0]))(param)
    0.6232252401402305

    .. seealso:: :func:`pennylane.math.vn_entropy` and :func:`pennylane.vn_entropy`
    """

    density_matrix_qnode = qml.qinfo.reduced_dm(qnode, qnode.device.wires)

    def wrapper(*args, **kwargs):
        # If pure state directly return 0.
        if len(wires) == len(qnode.device.wires):
            qnode.construct(args, kwargs)
            return_type = qnode.tape.observables[0].return_type
            if len(qnode.tape.observables) != 1 or not return_type == qml.measurements.State:
                raise ValueError("The qfunc return type needs to be a state.")
            density_matrix = qnode(*args, **kwargs)
            if density_matrix.shape == (density_matrix.shape[0],):
                return 0.0
            entropy = qml.math.vn_entropy(density_matrix, wires, base, c_dtype=qnode.device.C_DTYPE)
            return entropy

        density_matrix = density_matrix_qnode(*args, **kwargs)
        entropy = qml.math.vn_entropy(density_matrix, wires, base, c_dtype=qnode.device.C_DTYPE)
        return entropy

    return wrapper


def mutual_info(qnode, wires0, wires1, base=None):
    r"""Compute the mutual information from a :class:`.QNode` returning a :func:`~.state`:

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})

    where :math:`S` is the von Neumann entropy.

    The mutual information is a measure of correlation between two subsystems.
    More specifically, it quantifies the amount of information obtained about
    one system by measuring the other system.

    Args:
        qnode (QNode): A :class:`.QNode` returning a :func:`~.state`.
        wires0 (Sequence(int)): List of wires in the first subsystem.
        wires1 (Sequence(int)): List of wires in the second subsystem.
        base (float): Base for the logarithm. If None, the natural logarithm is used.

    Returns:
        func: A function with the same arguments as the QNode that returns
        the mutual information from its output state.

    **Example**

    It is possible to obtain the mutual information of two subsystems from a
    :class:`.QNode` returning a :func:`~.state`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

    >>> mutual_info_circuit = qinfo.mutual_info(circuit, wires0=[0], wires1=[1])
    >>> mutual_info_circuit(np.pi/2)
    1.3862943611198906
    >>> x = np.array(0.4, requires_grad=True)
    >>> mutual_info_circuit(x)
    0.3325090393262875
    >>> qml.grad(mutual_info_circuit)(np.array(0.4, requires_grad=True))
    1.2430067731198946

    .. seealso:: :func:`~.qinfo.vn_entropy`, :func:`pennylane.math.mutual_info` and :func:`pennylane.mutual_info`
    """

    density_matrix_qnode = qml.qinfo.reduced_dm(qnode, qnode.device.wires)

    def wrapper(*args, **kwargs):
        density_matrix = density_matrix_qnode(*args, **kwargs)
        entropy = qml.math.mutual_info(density_matrix, wires0, wires1, base=base)
        return entropy

    return wrapper


# TODO: create qml.math.jacobian and replace it here
def _torch_jac(circ):
    """Torch jacobian as a callable function"""
    import torch

    def wrapper(*args, **kwargs):
        loss = functools.partial(circ, **kwargs)
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


@batch_transform
def _make_probs(tape: QuantumTape, wires=None, post_processing_fn=None):
    """Ignores the return types of any qnode and creates a new one that outputs probabilities"""
    if wires is None:
        wires = tape.wires

    with QuantumTape(shots=tape.raw_shots) as new_tape:
        for op in tape.operations:
            qml.apply(op)
        qml.probs(wires=wires)

    if post_processing_fn is None:
        post_processing_fn = lambda x: qml.math.squeeze(qml.math.stack(x))

    return [new_tape], post_processing_fn


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


    .. seealso:: :func:`~.pennylane.metric_tensor`, :func:`~.pennylane.qinfo.transforms.quantum_fisher`

    **Example**

    First, let us define a parametrized quantum state and return its (classical) probability distribution for all
    computational basis elements:

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

    We can obtain its ``(2, 2)`` classical fisher information matrix (CFIM) by simply calling the function returned
    by ``classical_fisher()``:

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
    Closely related to the `quantum natural gradient <https://arxiv.org/abs/1909.02108>`_, which employs the
    `quantum` fisher information matrix, we can compute a rescaled gradient using the CFIM. In this scenario,
    typically a Hamiltonian objective function :math:`\langle H \rangle` is minimized:

    .. code-block:: python

        H = qml.Hamiltonian(coeffs=[0.5, 0.5], observables=[qml.PauliZ(0), qml.PauliZ(1)])

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
    >>> cfim = qml.qinfo.classical_fisher(circ)(params)
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

    >>> print(qml.qinfo.classical_fisher(circ)(params))
    (tensor([[0.13340679, 0.03650311],
             [0.03650311, 0.00998807]], requires_grad=True)
    >>> print(qml.jacobian(qml.qinfo.classical_fisher(circ))(params))
    array([[[9.98030491e-01, 3.46944695e-18],
            [1.36541817e-01, 5.15248592e-01]],
           [[1.36541817e-01, 5.15248592e-01],
            [2.16840434e-18, 2.81967252e-01]]]))

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


def quantum_fisher(qnode: QNode, *args, **kwargs):
    r"""Returns a function that computes the quantum fisher information matrix (QFIM) of a given :class:`.QNode`.

    Given a parametrized quantum state :math:`|\psi(\bm{\theta})\rangle`, the quantum fisher information matrix (QFIM) quantifies how changes to the parameters :math:`\bm{\theta}`
    are reflected in the quantum state. The metric used to induce the QFIM is the fidelity :math:`f = |\langle \psi | \psi' \rangle|^2` between two (pure) quantum states.
    This leads to the following definition of the QFIM (see eq. (27) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_):

    .. math::

        \text{QFIM}_{i, j} = 4 \text{Re}\left[ \langle \partial_i \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle
        - \langle \partial_i \psi(\bm{\theta}) | \psi(\bm{\theta}) \rangle \langle \psi(\bm{\theta}) | \partial_j \psi(\bm{\theta}) \rangle \right]

    with short notation :math:`| \partial_j \psi(\bm{\theta}) \rangle := \frac{\partial}{\partial \theta_j}| \psi(\bm{\theta}) \rangle`.

    .. seealso::
        :func:`~.pennylane.metric_tensor`, :func:`~.pennylane.adjoint_metric_tensor`, :func:`~.pennylane.qinfo.transforms.classical_fisher`

    Args:
        qnode (:class:`.QNode`): A :class:`.QNode` that may have arbitrary return types.
        *args: In case finite shots are used, further arguments according to :func:`~.pennylane.metric_tensor` may be passed.

    Returns:
        func: The function that computes the quantum fisher information matrix.

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

        H = 1.*qml.PauliX(0) @ qml.PauliX(1) - 0.5 * qml.PauliZ(1)

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
    array([ 0.59422561, -0.02615095, -0.05146226])
    >>> qfim = qml.qinfo.quantum_fisher(circ)(params)
    >>> qfim
    tensor([[1.        , 0.        , 0.        ],
            [0.        , 1.        , 0.        ],
            [0.        , 0.        , 0.77517241]], requires_grad=True)
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
    >>> qfim = qml.qinfo.quantum_fisher(circ)(params)

    Alternatively, we can fall back on the block-diagonal QFIM without the additional wire.

    >>> qfim = qml.qinfo.quantum_fisher(circ, approx="block-diag")(params)

    """

    if qnode.shots is not None and isinstance(qnode.device, DefaultQubit):

        def wrapper(*args0, **kwargs0):
            return 4 * metric_tensor(qnode, *args, **kwargs)(*args0, **kwargs0)

    else:

        def wrapper(*args0, **kwargs0):
            return 4 * adjoint_metric_tensor(qnode, *args, **kwargs)(*args0, **kwargs0)

    return wrapper


def fidelity(qnode0, qnode1, wires0, wires1):
    r"""Compute the fidelity for two :class:`.QNode` returning a :func:`~.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size.

    The fidelity for two mixed states given by density matrices :math:`\rho` and :math:`\sigma`
    is defined as

    .. math::
        F( \rho , \sigma ) = \text{Tr}( \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}})^2

    If one of the states is pure, say :math:`\rho=\ket{\psi}\bra{\psi}`, then the expression
    for fidelity simplifies to

    .. math::
        F( \ket{\psi} , \sigma ) = \bra{\psi} \sigma \ket{\psi}

    Finally, if both states are pure, :math:`\sigma=\ket{\phi}\bra{\phi}`, then the
    fidelity is simply

    .. math::
        F( \ket{\psi} , \ket{\phi}) = \left|\braket{\psi, \phi}\right|^2

    .. note::
        The second state is coerced to the type and dtype of the first state. The fidelity is returned in the type
        of the interface of the first state.

    Args:
        state0 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        state1 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        wires0 (Sequence[int]): the wires of the first subsystem
        wires1 (Sequence[int]): the wires of the second subsystem

    Returns:
        func: A function that returns the fidelity between the states outputted by the QNodes.

    **Example**

    First, let's consider two QNodes with potentially different signatures: a circuit with two parameters
    and another circuit with a single parameter. The output of the :func:`~.qinfo.fidelity` transform then requires
    two tuples to be passed as arguments, each containing the args and kwargs of their respective circuit, e.g.
    ``all_args0 = (0.1, 0.3)`` and ``all_args1 = (0.2)`` in the following case:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit_rx(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit_ry(y):
            qml.RY(y, wires=0)
            return qml.state()

    >>> qml.qinfo.fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])((0.1, 0.3), (0.2))
    0.9905158135644924

    It is also possible to use QNodes that do not depend on any parameters. When it is the case for the first QNode, it
    is required to pass an empty tuple as an argument for the first QNode.

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit_rx():
            return qml.state()

        @qml.qnode(dev)
        def circuit_ry(x):
            qml.RY(x, wires=0)
            return qml.state()

    >>> qml.qinfo.fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])((), (0.2))
    0.9900332889206207

    On the other hand, if the second QNode is the one that does not depend on parameters then a single tuple can also be
    passed:

    >>> qml.qinfo.fidelity(circuit_ry, circuit_rx, wires0=[0], wires1=[0])((0.2))
    0.9900332889206207

    The :func:`~.qinfo.fidelity` transform is also differentiable and the gradient can be obtained in the different frameworks
    with backpropagation, the following example uses ``jax`` and ``backprop``.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, interface="jax")
        def circuit0(x):
            qml.RX(x, wires=0)
            return qml.state()

        @qml.qnode(dev, interface="jax")
        def circuit1():
            qml.PauliZ(wires=0)
            return qml.state()

    >>> jax.grad(qml.qinfo.fidelity(circuit0, circuit1, wires0=[0], wires1=[0]))((jax.numpy.array(0.3)))
    -0.14776011

    There is also the possibility to pass a single dictionary at the end of the tuple for fixing args,
    you can follow this example:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit_rx(x, y):
            qml.RX(x, wires=0)
            qml.RZ(y, wires=0)
            return qml.state()

        @qml.qnode(dev)
        def circuit_ry(y, use_ry):
            if use_ry:
                qml.RY(y, wires=0)
            return qml.state()

    >>> fidelity(circuit_rx, circuit_ry, wires0=[0], wires1=[0])((0.1, 0.3), (0.9, {'use_ry': True}))
    0.8208074192135424

    .. seealso:: :func:`pennylane.math.fidelity`
    """

    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    # Get the state vector if all wires are selected
    if len(wires0) == len(qnode0.device.wires):
        state_qnode0 = qnode0
    else:
        state_qnode0 = qml.qinfo.reduced_dm(qnode0, wires=wires0)

    # Get the state vector if all wires are selected
    if len(wires1) == len(qnode1.device.wires):
        state_qnode1 = qnode1
    else:
        state_qnode1 = qml.qinfo.reduced_dm(qnode1, wires=wires1)

    def evaluate_fidelity(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the fidelity between two states computed from QNodes. It allows giving
        the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Fidelity between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args0[-1], dict):
                args0 = all_args0[:-1]
                kwargs0 = all_args0[-1]
            else:
                args0 = all_args0
                kwargs0 = {}
            state0 = state_qnode0(*args0, **kwargs0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args1[-1], dict):
                args1 = all_args1[:-1]
                kwargs1 = all_args1[-1]
            else:
                args1 = all_args1
                kwargs1 = {}
            state1 = state_qnode1(*args1, **kwargs1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the fidelity.
        fid = qml.math.fidelity(state0, state1)
        return fid

    return evaluate_fidelity


def relative_entropy(qnode0, qnode1, wires0, wires1):
    r"""
    Compute the relative entropy for two :class:`.QNode` returning a :func:`~.state` (a state can be a state vector
    or a density matrix, depending on the device) acting on quantum systems with the same size.

    .. math::
        S(\rho\,\|\,\sigma)=-\text{Tr}(\rho\log\sigma)-S(\rho)=\text{Tr}(\rho\log\rho)-\text{Tr}(\rho\log\sigma)
        =\text{Tr}(\rho(\log\rho-\log\sigma))

    Roughly speaking, quantum relative entropy is a measure of distinguishability between two
    quantum states. It is the quantum mechanical analog of relative entropy.

    Args:
        qnode0 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        qnode1 (QNode): A :class:`.QNode` returning a :func:`~.state`.
        wires0 (Sequence[int]): the subsystem of the first QNode
        wires1 (Sequence[int]): the subsystem of the second QNode

    Returns:
        func: A function that takes as input the joint arguments of the two QNodes,
        and returns the relative entropy from their output states.

    **Example**

    Consider the following QNode:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

    The ``qml.qinfo.relative_entropy`` transform can be used to compute the relative
    entropy between the output states of the QNode:

    >>> relative_entropy_circuit = qml.qinfo.relative_entropy(circuit, circuit, wires0=[0], wires1=[0])

    The returned function takes two tuples as input, the first being the arguments to the
    first QNode and the second being the arguments to the second QNode:

    >>> x, y = np.array(0.4), np.array(0.6)
    >>> relative_entropy_circuit((x,), (y,))
    0.017750012490703237

    This transform is fully differentiable:

    .. code-block:: python

        def wrapper(x, y):
            return relative_entropy_circuit((x,), (y,))

    >>> wrapper(x, y)
    0.017750012490703237
    >>> qml.grad(wrapper)(x, y)
    (array(-0.16458856), array(0.16953273))
    """

    if len(wires0) != len(wires1):
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    # Get the state vector if all wires are selected
    if len(wires0) == len(qnode0.device.wires):
        state_qnode0 = qnode0
    else:
        state_qnode0 = qml.qinfo.reduced_dm(qnode0, wires=wires0)

    # Get the state vector if all wires are selected
    if len(wires1) == len(qnode1.device.wires):
        state_qnode1 = qnode1
    else:
        state_qnode1 = qml.qinfo.reduced_dm(qnode1, wires=wires1)

    def evaluate_relative_entropy(all_args0=None, all_args1=None):
        """Wrapper used for evaluation of the fidelity between two states computed from QNodes. It allows giving
        the args and kwargs to each :class:`.QNode`.

        Args:
            all_args0 (tuple): Tuple containing the arguments (*args, kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, kwargs) of the second :class:`.QNode`.

        Returns:
            float: Fidelity between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args0[-1], dict):
                args0 = all_args0[:-1]
                kwargs0 = all_args0[-1]
            else:
                args0 = all_args0
                kwargs0 = {}
            state0 = state_qnode0(*args0, **kwargs0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            # Handle a dictionary as last argument
            if isinstance(all_args1[-1], dict):
                args1 = all_args1[:-1]
                kwargs1 = all_args1[-1]
            else:
                args1 = all_args1
                kwargs1 = {}
            state1 = state_qnode1(*args1, **kwargs1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the relative entropy
        return qml.math.relative_entropy(state0, state1)

    return evaluate_relative_entropy
