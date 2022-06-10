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
# pylint: disable=import-outside-toplevel, not-callable
import functools
import pennylane as qml
from pennylane.transforms import batch_transform


def reduced_dm(qnode, wires):
    """Compute the reduced density matrix from a :class:`~.QNode` returning :func:`~.state`.

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

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)
            @qml.qnode(dev)
            def circuit(x):
                qml.IsingXX(x, wires=[0, 1])
                return qml.state()

    >>> vn_entropy(circuit, wires=[0])(np.pi/2)
    0.6931472

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
    >>> qml.grad(mutual_info_circuit)(0.4)
    1.2430067731198946

    .. seealso::

        :func:`~.qinfo.vn_entropy_transform`
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


def _compute_cfim(p, dp):
    r"""Computes the (num_params, num_params) classical fisher information matrix from the probabilities and its derivatives
    I.e. it computes :math:`classical_fisher_{ij} = \sum_\ell (\partial_i p_\ell) (\partial_i p_\ell) / p_\ell`
    """
    # Exclude values where p=0 and calculate 1/p
    nonzeros_p = qml.math.where(p > 0, p, qml.math.ones_like(p))
    one_over_p = qml.math.where(p > 0, qml.math.ones_like(p), qml.math.zeros_like(p))
    one_over_p = qml.math.divide(one_over_p, nonzeros_p)

    # Multiply dp and p
    # Note that casting and being careful about dtypes is necessary as interfaces
    # typically treat derivatives (dp) with float32, while standard execution (p) comes in float64
    dp = qml.math.cast(dp, dtype=p.dtype)
    dp = qml.math.reshape(
        dp, (len(p), -1)
    )  # Squeeze does not work, as you could have shape (num_probs, num_params) with num_params = 1
    dp_over_p = qml.math.transpose(dp) * one_over_p  # creates (n_params, n_probs) array

    # (n_params, n_probs) @ (n_probs, n_params) = (n_params, n_params)
    return dp_over_p @ dp


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

    Returns: func: The function that computes the classical fisher information matrix. This function accepts the same
    signature as the :class:`.QNode`. If the signature contains one differentiable variable ``params``, the function
    returns a matrix of size ``(len(params), len(params))``. For multiple differentiable arguments ``x, y, z``,
    it returns a list of sizes ``[(len(x), len(x)), (len(y), len(y)), (len(z), len(z))]``.

    .. warning::

        The ``classical_fisher()`` matrix is currently not differentiable.

    .. seealso:: :func:`~.pennylane.metric_tensor`

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
        if isinstance(j, tuple) and len(j) > 1:
            res = []
            for j_i in j:
                if interface == "tf":
                    j_i = qml.math.transpose(qml.math.cast(j_i, dtype=p.dtype))

                res.append(_compute_cfim(p, j_i))

            return res

        return _compute_cfim(p, j)

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
    and another circuit with a single parameter. The output of the `qml.qinfo.fidelity` transform then requires
    two tuples to be passed as arguments, each containing the args and kwargs of their respective circuit, e.g. `all_args0 = (0.1, 0.3)` and
    `all_args1 = (0.2)` in the following case:

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

    It is also possible to use QNodes that do not depend on any parameters. When it is the case for the first QNode, you
    need to pass an empty tuple as an argument for the first QNode.

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

    The `qml.qinfo.fidelity` transform is also differentiable and you can use the gradient in the different frameworks
    with backpropagation, the following example uses `jax` and `backprop`.

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
            all_args0 (tuple): Tuple containing the arguments (*args, **kwargs) of the first :class:`.QNode`.
            all_args1 (tuple): Tuple containing the arguments (*args, **kwargs) of the second :class:`.QNode`.

        Returns:
            float: Fidelity between two quantum states
        """
        if not isinstance(all_args0, tuple) and all_args0 is not None:
            all_args0 = (all_args0,)

        if not isinstance(all_args1, tuple) and all_args1 is not None:
            all_args1 = (all_args1,)

        # If no all_args is given, evaluate the QNode without args
        if all_args0 is not None:
            state0 = state_qnode0(*all_args0)
        else:
            # No args
            state0 = state_qnode0()

        # If no all_args is given, evaluate the QNode without args
        if all_args1 is not None:
            state1 = state_qnode1(*all_args1)
        else:
            # No args
            state1 = state_qnode1()

        # From the two generated states, compute the fidelity.
        fid = qml.math.fidelity(state0, state1)
        return fid

    return evaluate_fidelity
