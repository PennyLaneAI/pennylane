# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper Functionality to compute the kak decomposition variationally, as outlined in https://arxiv.org/abs/2104.00728"""
# pylint: disable=too-many-arguments, too-many-positional-arguments
import warnings
from datetime import datetime
from functools import partial

import numpy as np

import pennylane as qml
from pennylane.liealg import adjvec_to_op, op_to_adjvec
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence

try:
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", True)
    has_jax = True
except ImportError:
    has_jax = False

try:
    import matplotlib.pyplot as plt

    has_plt = True
except ImportError:
    has_plt = False


def variational_kak_adj(H, g, dims, adj, verbose=False, opt_kwargs=None, pick_min=False):
    r"""
    Variational KaK decomposition of Hermitian ``H`` using the adjoint representation.

    Given a Cartan decomposition (:func:`~cartan_decomp`) :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`,
    a Hermitian operator :math:`H \in \mathfrak{m}`,
    and a horizontal Cartan subalgebra (:func:`~horizontal_cartan_subalgebra`) :math:`\mathfrak{a} \subset \mathfrak{m}`,
    this function computes
    :math:`a \in \mathfrak{a}` and :math:`K_c \in e^{i\mathfrak{k}}` such that

    .. math:: H = K_c a K_c^\dagger.

    In particular, :math:`a = \sum_j c_j a_j` is decomposed in terms of commuting operators :math:`a_j \in \mathfrak{a}`.
    This allows for the immediate decomposition

    .. math:: e^{-i t H} = K_c e^{-i t a} K_c^\dagger = K_c \left(\prod_j e^{-i t c_j a_j} \right) K_c^\dagger.

    The result is provided in terms of the adjoint vector representation of :math:`a \in \mathfrak{a}`
    (see :func:`adjvec_to_op`), i.e. the ordered coefficients :math:`c_j` in :math:`a = \sum_j c_j m_j`
    with the basis elements :math:`m_j \in (\tilde{\mathfrak{m}} \oplus \mathfrak{a})` and
    the optimal parameters :math:`\theta` such that

    .. math:: K_c = \prod_{j=|\mathfrak{k}|}^{1} e^{-i \theta_j k_j}

    for the ordered basis of :math:`\mathfrak{k}` given by the first :math:`|\mathfrak{k}|` elements of ``g``.
    Note that we define :math:`K_c` mathematically with the descending order of basis elements :math:`k_j \in \mathfrak{k}` such that
    the resulting circuit has the canonical ascending order. In particular, a PennyLane quantum function that describes the circuit given
    the optimal parameters ``theta_opt`` and the basis ``k`` containing the operators, is given by the following.

    .. code-block:: python

        def Kc(theta_opt: Iterable[float], k: Iterable[Operator]):
            assert len(theta_opt) == len(k)
            for theta_j, k_j in zip(theta_opt, k):
                qml.exp(-1j * theta_j * k_j)

    Internally, this function performs a modified version of `2104.00728 <https://arxiv.org/abs/2104.00728>`__,
    in particular minimizing the cost function

    .. math:: f(\theta) = \langle H, K(\theta) e^{-i \sum_{j=1}^{|\mathfrak{a}|} \pi^j a_j} K(\theta)^\dagger \rangle,

    see eq. (6) therein and our `demo <demos/tutorial_fixed_depth_hamiltonian_simulation_via_cartan_decomposition>`__ for more details.
    Instead of relying on having Pauli words, we use the adjoint representation
    for a more general evaluation of the cost function. The rest is the same.

    .. seealso:: `The KAK decomposition in theory (demo) <demos/tutorial_kak_decomposition>`__, `The KAK decomposition in practice (demo) <demos/tutorial_fixed_depth_hamiltonian_simulation_via_cartan_decomposition>`__.

    Args:
        H (Union[Operator, PauliSentence, np.ndarray]): Hamiltonian to decompose
        g (List[Union[Operator, PauliSentence, np.ndarray]]): DLA of the Hamiltonian
        dims (Tuple[int]): Tuple of dimensions ``(dim_k, dim_mtilde, dim_a)`` of
            Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus (\tilde{\mathfrak{m}} \oplus \mathfrak{a})`
        adj (np.ndarray): Adjoint representation of dimension ``(dim_g, dim_g, dim_g)``,
            with the implicit ordering ``(k, mtilde, a)``.
        verbose (bool): Plot the optimization. Requires matplotlib to be installed (``pip install matplotlib``)
        opt_kwargs (dict): Keyword arguments for the optimization like initial starting values
            for :math:`\theta` of dimension ``(dim_k,)``, given as ``theta0``.
            Also includes ``n_epochs``, ``lr``, ``b1``, ``b2``, ``verbose``, ``interrupt_tol``, see :func:`~run_opt`
        pick_min (bool): Whether to pick the parameter set with lowest cost function value during the optimization
            as optimal parameters. Otherwise picks the last parameter set.
    Returns:
        Tuple(np.ndarray, np.ndarray): ``(adjvec_a, theta_opt)``: The adjoint vector representation
        ``adjvec_a`` of dimension ``(dim_mtilde + dim_a,)``, with respect to the basis of
        :math:`\mathfrak{m} = \tilde{\mathfrak{m}} + \mathfrak{a}` of the CSA element
        :math:`a \in \mathfrak{a}` s.t. :math:`H = K a K^\dagger`. For a successful optimization, the entries
        corresponding to :math:`\tilde{\mathfrak{m}}` should be close to zero.
        The second return value, ``theta_opt``, are the optimal coefficients :math:`\theta` of the
        decomposition :math:`K = \prod_{j=|\mathfrak{k}|}^{1} e^{-i \theta_j k_j}` for the basis :math:`k_j \in \mathfrak{k}`.


    **Example**

    Let us perform a KaK decomposition for the transverse field Ising model Hamiltonian, exemplarily for :math:`n=3` qubits on a chain.
    We start with some boilerplate code to perform a Cartan decomposition using the :func:`~concurrence_involution`, which places the Hamiltonian
    in the horizontal subspace :math:`\mathfrak{m}`. From this we re-order :math:`\mathfrak{g} = \mathfrak{k} + \mathfrak{m}` and finally compute a
    :func:`~horizontal_cartan_subalgebra` :math:`\mathfrak{a}` in :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{a}`.

    .. code-block:: python

        import pennylane as qml
        import numpy as np
        import jax.numpy as jnp
        import jax

        from pennylane import X, Z
        from pennylane.liealg import (
            cartan_decomp,
            horizontal_cartan_subalgebra,
            check_cartan_decomp,
            concurrence_involution,
            adjvec_to_op,
        )
        from pennylane.labs.dla import (
            validate_kak,
            variational_kak_adj,
        )

        n = 3

        gens = [X(i) @ X(i + 1) for i in range(n - 1)]
        gens += [Z(i) for i in range(n)]
        H = qml.sum(*gens)

        g = qml.lie_closure(gens)
        g = [op.pauli_rep for op in g]

        involution = concurrence_involution

        assert not involution(H)
        k, m = cartan_decomp(g, involution=involution)
        assert check_cartan_decomp(k, m)

        g = k + m
        adj = qml.structure_constants(g)

        g, k, mtilde, a, adj = horizontal_cartan_subalgebra(g, k, m, adj, tol=1e-14, start_idx=0)

    Due to the canonical ordering of all constituents, it suffices to tell ``variational_kak_adj`` the dimensions of ``dims = (len(k), len(mtilde), len(a))``,
    alongside the Hamiltonian ``H``, the Lie algebra ``g`` and its adjoint representation ``adj``. Internally, the function is performing a variational
    optimization to find a local extremum of a suitably constructed loss function that finds as its extremum the decomposition

    .. math:: K_c = \prod_{j=1}^{|\mathfrak{k}|} e^{-i \theta_j k_j}

    in form of the optimal parameters :math:`\{\theta_j\}` for the respective :math:`k_j \in \mathfrak{k}`.
    The resulting :math:`K` then informs the CSA element ``a``
    of the KaK decomposition via :math:`a = K_c H K_c^\dagger`. This is detailed in `2104.00728 <https://arxiv.org/abs/2104.00728>`__.


    >>> dims = (len(k), len(mtilde), len(a))
    >>> adjvec_a, theta_opt = variational_kak_adj(H, g, dims, adj, opt_kwargs={"n_epochs": 3000})

    As a result, we are provided with the adjoint vector representation of the CSA element
    :math:`a \in \mathfrak{a}` with respect to the basis ``mtilde+a`` and the optimal parameters of dimension :math:`|\mathfrak{k}|`

    Let us perform some sanity checks to better understand the resulting outputs.
    We can turn that element back to an operator using :func:`adjvec_to_op` and from that to a matrix for which we can check Hermiticity.

    .. code-block:: python

        m = mtilde + a
        [a_op] = adjvec_to_op([adjvec_a], m)
        a_m = qml.matrix(a_op, wire_order=range(n))
        assert np.allclose(a_m, a_m.conj().T)

    Let us now confirm that we get back the original Hamiltonian from the resulting :math:`K_c` and :math:`a`.
    In particular, we want to confirm :math:`H = K_c a K_c^\dagger` for :math:`K_c = \prod_{j=1}^{|\mathfrak{k}|} e^{-i \theta_j k_j}`.

    .. code-block:: python

        assert len(theta_opt) == len(k)

        def Kc(theta_opt):
            for th, op in zip(theta_opt, k):
                qml.exp(-1j * th * op.operation())

        Kc_m = qml.matrix(Kc, wire_order=range(n))(theta_opt)

        # check Unitary property of Kc
        assert np.allclose(Kc_m.conj().T @ Kc_m, np.eye(2**n))

        H_reconstructed = Kc_m @ a_m @ Kc_m.conj().T

        H_m = qml.matrix(H, wire_order=range(len(H.wires)))

        # check Hermitian property of reconstructed Hamiltonian
        assert np.allclose(
            H_reconstructed, H_reconstructed.conj().T
        )

        # confirm reconstruction was successful to some given numerical tolerance
        assert np.allclose(H_m, H_reconstructed, atol=1e-6)

    Instead of performing these checks by hand, we can use the helper function :func:`~validate_kak`.

    >>> assert validate_kak(H, g, k, (adjvec_a, theta_opt), n, 1e-6)

    """

    if not has_jax:  # pragma: no cover
        raise ImportError(
            "jax and optax are required for variational_kak_adj. You can install them with pip install jax~=0.6.0 jaxlib~=0.6.0 optax."
        )  # pragma: no cover
    if verbose >= 1 and not has_plt:  # pragma: no cover
        print(
            "variational_kak_adj requires matplotlib to display a figure with the optimization "
            "progress (for verbose>=1). You can install it with pip install matplotlib"
        )

    if opt_kwargs is None:
        opt_kwargs = {}
    if not isinstance(H, PauliSentence):
        H = H.pauli_rep

    dim_k, dim_mtilde, dim_h = dims
    dim_m = dim_mtilde + dim_h

    adj_cropped = adj[-dim_m:, :dim_k, -dim_m:]

    ## creating the gamma vector expanded on the whole m
    gammavec = jnp.zeros(dim_m)
    gammavec = gammavec.at[-dim_h:].set([np.pi**i for i in range(dim_h)])

    def loss(theta, vec_H, adj):
        # this is different to Appendix F 1 in https://arxiv.org/pdf/2104.00728
        # Making use of adjoint representation
        # should be faster, and most importantly allow for treatment of sums of paulis

        assert adj.shape == (len(vec_H), len(theta), len(vec_H))
        # Implement Ad_(K_1 .. K_|k|) (vec_H), so that we get K_1 .. K_|k| H K^†_|k| .. K^†_1

        for i in range(dim_k - 1, -1, -1):
            vec_H = jax.scipy.linalg.expm(theta[i] * adj[:, i]) @ vec_H

        return (gammavec @ vec_H).real

    if verbose >= 1:
        print([H], g[-dim_m:])

    [vec_H] = op_to_adjvec([H], g[-dim_m:], is_orthogonal=False)

    theta0 = opt_kwargs.pop("theta0", None)
    if theta0 is None:
        theta0 = jax.random.normal(jax.random.PRNGKey(0), (dim_k,))

    opt_kwargs["verbose"] = verbose

    thetas, energy, _ = run_opt(partial(loss, vec_H=vec_H, adj=adj_cropped), theta0, **opt_kwargs)

    if verbose >= 1:
        plt.plot(energy - np.min(energy))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.show()

    idx = np.argmin(energy) if pick_min else -1

    if verbose:
        n_epochs = opt_kwargs.get("n_epochs", 500)
        print(f"Picking entry with index {idx} out of {n_epochs-1} ({pick_min=}).")
    theta_opt = thetas[idx]

    # Implement Ad_(K_1 .. K_|k|) (vec_H) like in the loss, with optimized parameters now.
    for i in range(dim_k - 1, -1, -1):
        vec_H = jax.scipy.linalg.expm(theta_opt[i] * adj_cropped[:, i]) @ vec_H

    return vec_H, theta_opt


def validate_kak(H, g, k, kak_res, n, error_tol, verbose=False):
    """Helper function to validate a khk decomposition"""
    # validate h_elem is Hermitian
    _is_dense = all(isinstance(op, np.ndarray) for op in k) and all(
        isinstance(op, np.ndarray) for op in k
    )

    vec_a, theta_opt = kak_res
    [a_elem] = adjvec_to_op([vec_a], g[len(k) :])  # sum(c * op for c, op in zip(vec_h, m))

    if isinstance(a_elem, Operator):
        a_elem_m = qml.matrix(a_elem, wire_order=range(n))
    elif isinstance(a_elem, PauliSentence):
        a_elem_m = a_elem.to_mat(wire_order=range(n))
    else:
        a_elem_m = a_elem

    assert np.allclose(a_elem_m, a_elem_m.conj().T), "CSA element `a` not Hermitian"

    # validate K_c a K_c^† reproduces H
    # Compute the ansatz K_c = K(theta_c) = K_1(theta_1) .. K_|k|(theta_|k|)
    Km = jnp.eye(2**n)
    assert len(theta_opt) == len(k)
    for th, op in zip(theta_opt, k):
        opm = qml.matrix(op.operation(), wire_order=range(n)) if not _is_dense else op
        Km @= jax.scipy.linalg.expm(1j * th * opm)

    assert np.allclose(Km @ Km.conj().T, np.eye(2**n))

    # Compute K_c^† a K_c
    H_reconstructed = Km.conj().T @ a_elem_m @ Km

    H_m = qml.matrix(H, wire_order=range(len(H.wires)))

    if verbose:
        print(f"Original matrix: {H_m}")
        print(f"Reconstructed matrix: {H_reconstructed}")

    assert np.allclose(
        H_reconstructed, H_reconstructed.conj().T
    ), "Reconstructed Hamiltonian not Hermitian"

    success = np.allclose(H_m, H_reconstructed, atol=error_tol)

    if not success:
        error = np.linalg.norm(H_m - H_reconstructed, ord="fro")

        warnings.warn(
            "The reconstructed H is not numerical identical to the original H.\n"
            f"We can still check for unitary equivalence: {error}",
            UserWarning,
        )

    return success


def run_opt(
    cost,
    theta,
    n_epochs=500,
    optimizer=None,
    verbose=False,
    interrupt_tol=None,
):
    r"""Boilerplate jax optimization

    Args:
        cost (callable): Cost function with scalar valued real output
        theta (Iterable): Initial values for argument of ``cost``
        n_epochs (int): Number of optimization iterations
        optimizer (optax.GradientTransformation): ``optax`` optimizer. Default is ``optax.adam(learning_rate=0.1)``.
        verbose (bool): Whether progress is output during optimization
        interrupt_tol (float): If not None, interrupt the optimization if the norm of the gradient is smaller than ``interrupt_tol``.

    **Example**

    .. code-block:: python

        from pennylane.labs.dla import run_opt
        import jax
        import jax.numpy as jnp
        import optax
        jax.config.update("jax_enable_x64", True)

        def cost(x):
            return x**2

        x0 = jnp.array(0.4)

        thetas, energy, gradients = run_opt(cost, x0)

    When no ``optimizer`` is passed, we use ``optax.adam(learning_rate=0.1)``.
    We can also use other optimizers, like ``optax.lbfgs``.

    >>> optimizer = optax.lbfgs(learning_rate=0.1, memory_size=1000)
    >>> thetas, energy, gradients = run_opt(cost, x0, optimizer=optimizer)

    """

    if not has_jax:  # pragma: no cover
        raise ImportError(
            "jax and optax are required for run_opt. You can install them with pip install jax~=0.6.0 jaxlib~=0.6.0 optax."
        )  # pragma: no cover

    if optimizer is None:
        optimizer = optax.adam(learning_rate=0.1)

    value_and_grad = jax.jit(jax.value_and_grad(cost))
    compiled_cost = jax.jit(cost)

    opt_state = optimizer.init(theta)

    energy, gradients, thetas = [], [], []

    @jax.jit
    def step(opt_state, theta):
        val, grad_circuit = value_and_grad(theta)
        updates, opt_state = optimizer.update(
            grad_circuit, opt_state, theta, value=val, grad=grad_circuit, value_fn=compiled_cost
        )
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta, val, grad_circuit

    t0 = datetime.now()
    ## Optimization loop
    try:
        for n in range(n_epochs):
            opt_state, theta, val, grad_circuit = step(opt_state, theta)

            energy.append(val)
            gradients.append(grad_circuit)
            thetas.append(theta)
            if (
                interrupt_tol is not None
                and (norm := np.linalg.norm(gradients[-1])) < interrupt_tol
            ):
                print(
                    f"Interrupting after {n} epochs because gradient norm is {norm} < {interrupt_tol}"
                )
                break
            if verbose:
                if n == 0:
                    print("First optimization round performed")
                if n % (n_epochs // 20) == 0:
                    print(f"Epoch {n:5d}: {val:.8f}")
    except KeyboardInterrupt:
        print(
            "KeyboardInterrupt received. Cancelled the optimization and will return intermediate result."
        )

    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val}; min loss: {np.min(energy)}; after {t1 - t0}")

    return thetas, energy, gradients
