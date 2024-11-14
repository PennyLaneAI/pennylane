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
"""Helper Functionality to compute the khk decomposition variationally, as outlined in https://arxiv.org/abs/2104.00728"""
# pylint: disable=too-many-arguments, too-many-positional-arguments
import warnings
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence

from .cartan_subalgebra import adjvec_to_op, op_to_adjvec

jax.config.update("jax_enable_x64", True)


def variational_kak(H, g, dims, adj, verbose=False, opt_kwargs=None, pick_min=False):
    r"""
    Variational KaK decomposition of Hermitian ``H``

    Given a Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \tilde{\mathfrak{m}} \oplus \mathfrak{a}`
    and a Hermitian operator :math:`H \in \tilde{\mathfrak{m}} \oplus \mathfrak{a}`, this function computes
    :math:`a \in \mathfrak{a}` and :math:`K_c \in e^{i\mathfrak{k}}` such that

    .. math:: H = K_c^\dagger a K_c

    The result is provided in terms of the adjoint representation vector of :math:`a` (see :func:`adjvec_to_op`) and
    the optimal parameters :math:`\theta` such that

    .. math:: K_c = \prod_{j=1}^{|\mathfrak{k}|} e^{-i \theta_j k_j}

    for the ordered basis of :math:`\mathfrak{k}` given by the first ``dim_k`` elements of ``g``.

    Internally, this function performs a modified version of `2104.00728 <https://arxiv.org/abs/2104.00728>`__,
    in particular minimizing the cost function eq. (6) therein. Instead of relying on having Pauli words, we use the adjoint representation
    for a more general evaluation of the cost function. The rest is the same.

    Args:
        H (Union[Operator, PauliSentence, np.ndarray]): Hamiltonian to decompose
        g (List[Union[Operator, PauliSentence, np.ndarray]]): DLA of the Hamiltonian
        dims (Tuple[int]): Tuple of dimensions ``(dim_k, dim_mtilde, dim_h)`` of
            Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} + \tilde{\mathfrak{m}} + \mathfrak{h}`
        adj (np.ndarray): Adjoint representation of dimension ``(dim_g, dim_g, dim_g)``,
            with the implicit ordering ``(k, mtilde, h)``.
        verbose (bool): Plot the optimization
        opt_kwargs (dict): Keyword arguments for the optimization like initial starting values for :math:`\theta` of dimension ``(dim_k,)``.
            Also includes ``n_epochs``, ``lr``, ``b1``, ``b2``, ``verbose``, ``interrupt_tol``, see :func:`~run_opt`

    Returns:
        np.ndarray: The adjoint vector representation ``adjvec_h`` of dimension ``(dim_mtilde + dim_h,)``, with respect to the basis of
            :math:`\mathfrak{m} = \tilde{\mathfrak{m}} + \mathfrak{h}` of the CSA element
            :math:`h \in \mathfrak{h}` s.t. :math:`H = K h K^\dagger`
        np.ndarray: The optimal coefficients :math:`\theta` of the decomposition :math:`K = \prod_{j=1}^{|\mathfrak{k}|} e^{-i \theta_j k_j}` for the basis :math:`k_j \in`


    **Example**

    Let us perform a KaK decomposition for the transverse field Ising model Hamiltonian, exemplarily for :math:`n=3` qubits on a chain.
    We start with some boilerplate code to perform a Cartan decomposition using the :func:`~concurrence_involution`, which places the Hamiltonian
    in the horizontal subspace :math:`\mathfrak{m}`. From this we re-order :math:`\mathfrak{g} = \mathfrak{k} + \mathfrak{m}` and finally compute a
    :func:`~cartan_subalgebra` :math:`\mathfrak{a}` in :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{a}`.

    .. code-block:: python

        import pennylane as qml
        import numpy as np
        import jax.numpy as jnp
        import jax

        from pennylane import X, Z
        from pennylane.labs.dla import (
            cartan_decomposition,
            cartan_subalgebra,
            check_cartan_decomp,
            concurrence_involution,
            validate_kak,
            variational_kak,
            adjvec_to_op,
        )

        n = 3

        gens = [X(i) @ X(i + 1) for i in range(n - 1)]
        gens += [Z(i) for i in range(n)]
        H = qml.sum(*gens)

        g = qml.lie_closure(gens)
        g = [op.pauli_rep for op in g]

        involution = concurrence_involution

        assert not involution(H)
        k, m = cartan_decomposition(g, involution=involution)
        assert check_cartan_decomp(k, m)

        g = k + m
        adj = qml.structure_constants(g)

        g, k, mtilde, a, adj = cartan_subalgebra(g, k, m, adj, tol=1e-14, start_idx=0)

    Due to the canonical ordering of all constituents, it suffices to tell ``variational_kak`` the dimensions of ``dims = (len(k), len(mtilde), len(a))``,
    alongside the Hamiltonian ``H``, the Lie algebra ``g`` and its adjoint representation ``adj``. Internally, the function is performing a variational
    optimization to find a local extremum of a suitably constructed loss function that finds as its extremum the decomposition

    .. math:: K_c = \prod_{j=1}^{|\mathfrak{k}|} e^{-i \theta_j k_j}

    in form of the optimal parameters :math:`\{\theta_j\}` for the respective :math:`k_j \in \mathfrak{k}`.
    The resulting :math:`K` then informs the CSA element ``a``
    of the KaK decomposition via :math:`a = K_c H K_c^\dagger`. This is detailed in `2104.00728 <https://arxiv.org/abs/2104.00728>`__.


    >>> dims = (len(k), len(mtilde), len(a))
    >>> adjvec_a, theta_opt = variational_kak(H, g, dims, adj, opt_kwargs={"n_epochs": 3000})

    As a result, we are provided the adjoint representation vector of the CSA element
    :math:`a \in \mathfrak{a}` and the optimal parameters of dimension :math:`|\mathfrak{k}|`

    Let us perform some sanity checks to better understand the resulting outputs.
    We can turn that element back to an operator using :func:`adjvec_to_op` and from that to a matrix for which we can check Hermiticity.
    .. code-block:: python

        [a] = adjvec_to_op([adjvec_a], g)
        a_m = qml.matrix(a, wire_order=range(n))
        assert np.allclose(a_m, a_m.conj().T)

    Let us now confirm that we get back the original Hamiltonian from the resulting :math:`K_c` and :math:`a`.
    In particular, we want to confirm :math:`H = K_c^\dagger a K_c` for :math:`K_c = \prod_{j=1}^{|\mathfrak{k}|} e^{-i \theta_j k_j}`.

    .. code-block:: python

        assert len(theta_opt) == len(k)
        def Kc():
            # note the reverse order of the multiplication because this is a circuit
            for th, op in zip(theta_opt[::-1], k[::-1]):
                qml.exp(-1j * th * op.operation())

        Kc_m = qml.matrix(Kc, wire_order=range(n))()

        # check Unitary property of Kc
        assert np.allclose(Kc_m.conj().T @ Kc_m, np.eye(2**n))

        H_reconstructed = Kc_m.conj().T @ a_m @ Kc_m

        H_m = qml.matrix(H, wire_order=range(len(H.wires)))

        # check Hermitian property of reconstructed Hamiltonian
        assert np.allclose(
            H_reconstructed, H_reconstructed.conj().T
        )

        # confirm reconstruction was successful to some given numerical tolerance
        assert np.allclose(H_m, H_reconstructed, atol=1e-6)

    Instead of performing these checks by hand, we can use the helper function :func:`~validat_kak`.

    >>> assert validate_kak(H, g, k, (adjvec_a, theta_opt), n, 1e-6)


    """
    if opt_kwargs is None:
        opt_kwargs = {}
    if not isinstance(H, PauliSentence):
        H = H.pauli_rep

    dim_k, dim_mtilde, dim_h = dims
    dim_m = dim_mtilde + dim_h
    dim_g = dim_k + dim_m

    adj_cropped = adj[:dim_k]  # [:, -dim_m:][:, :, -dim_m:]

    ## creating the gamma vector expanded on the whole m
    gammas = [np.pi**i for i in range(dim_h)]
    gammavec = np.zeros(dim_g)
    gammavec[-dim_h:] = gammas
    gammavec = jnp.array(gammavec)

    def loss(theta, vec_H, adj):
        # this is different to Appendix F 1 in https://arxiv.org/pdf/2104.00728
        # Making use of adjoint representation
        # should be faster, and most importantly allow for treatment of sums of paulis

        # gammavec @ (K_|k| .. K_1) @ vec_H
        res = jnp.eye(adj.shape[-1])

        for i in range(dim_k):
            res @= jax.scipy.linalg.expm(theta[i] * adj[i])

        return (gammavec @ res @ vec_H).real

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    [vec_H] = op_to_adjvec([H], g)

    theta0 = opt_kwargs.pop("theta0", None)
    if theta0 is None:
        theta0 = jax.random.normal(jax.random.PRNGKey(0), (dim_k,))

    thetas, energy, _ = run_opt(
        partial(value_and_grad, vec_H=vec_H, adj=adj_cropped), theta0, **opt_kwargs
    )

    if verbose >= 1:
        plt.plot(energy - np.min(energy))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.show()

    if pick_min:
        theta_opt = thetas[np.argmin(energy)]
    else:
        theta_opt = thetas[-1]

    M = jnp.eye(dim_g)

    for i in range(dim_k):  # TODO make matrix-vector multiplications instead
        M @= jax.scipy.linalg.expm(theta_opt[i] * adj_cropped[i])

    vec_h = M @ vec_H

    return vec_h, theta_opt


def validate_kak(H, g, k, kak_res, n, error_tol, verbose=False):
    """Helper function to validate a khk decomposition"""
    # validate h_elem is Hermitian
    _is_dense = all(isinstance(op, np.ndarray) for op in k) and all(
        isinstance(op, np.ndarray) for op in k
    )

    vec_h, theta_opt = kak_res
    [h_elem] = adjvec_to_op([vec_h], g)  # sum(c * op for c, op in zip(vec_h, m))

    if isinstance(h_elem, Operator):
        h_elem_m = qml.matrix(h_elem, wire_order=range(n))
    elif isinstance(h_elem, PauliSentence):
        h_elem_m = h_elem.to_mat(wire_order=range(n))
    else:
        h_elem_m = h_elem

    assert np.allclose(h_elem_m, h_elem_m.conj().T), "CSA element h not Hermitian"

    # validate KhK reproduces H
    Km = jnp.eye(2**n)
    assert len(theta_opt) == len(k)
    for th, op in zip(theta_opt, k):
        opm = qml.matrix(op.operation(), wire_order=range(n)) if not _is_dense else op
        Km @= jax.scipy.linalg.expm(-1j * th * opm)

    assert np.allclose(Km @ Km.conj().T, np.eye(2**n))

    H_reconstructed = Km.conj().T @ h_elem_m @ Km

    H_m = qml.matrix(H, wire_order=range(len(H.wires)))

    if verbose:
        print(f"Original matrix: {H_m}")
        print(f"Reconstructed matrix: {H_reconstructed}")

    assert np.allclose(
        H_reconstructed, H_reconstructed.conj().T
    ), "Reconstructed Hamiltonian not Hermitian"

    success = np.allclose(H_m, H_reconstructed, atol=error_tol)

    if not success:
        error = np.sqrt(
            np.trace((H_m - H_reconstructed) @ (H_m - H_reconstructed).conj().T)
        )  # Frobenius norm

        warnings.warn(
            f"The reconstructed H is not numerical identical to the original H.\n"
            f"We can still check for unitary equivalence: {error}",
            UserWarning,
        )

    return success


def run_opt(
    value_and_grad,
    theta,
    n_epochs=500,
    lr=0.1,
    b1=0.99,
    b2=0.999,
    verbose=True,
    interrupt_tol=None,
):
    """Boilerplate jax optimization"""
    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = []
    gradients = []
    thetas = []

    @jax.jit
    def partial_step(grad_circuit, opt_state, theta):
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta

    t0 = datetime.now()
    ## Optimization loop
    for n in range(n_epochs):
        # val, theta, grad_circuit, opt_state = step(theta, opt_state)
        val, grad_circuit = value_and_grad(theta)
        opt_state, theta = partial_step(grad_circuit, opt_state, theta)

        energy.append(val)
        gradients.append(grad_circuit)
        thetas.append(theta)
        if interrupt_tol is not None and (norm := np.linalg.norm(gradients[-1])) < interrupt_tol:
            print(
                f"Interrupting after {n} epochs because gradient norm is {norm} < {interrupt_tol}"
            )
            break
    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val}; min loss: {np.min(energy)}; after {t1 - t0}")

    return thetas, energy, gradients
