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
# pylint: disable=too-many-arguments
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


def variational_kak(H, g, dims, adj, verbose=False, opt_kwargs=None):
    r"""
    Variational KHK decomposition function

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



    """
    if opt_kwargs is None:
        opt_kwargs = {}
    if not isinstance(H, PauliSentence):
        H = H.pauli_rep

    dim_k, dim_mtilde, dim_h = dims
    dim_m = dim_mtilde + dim_h

    adj_cropped = adj[:dim_k][:, -dim_m:][:, :, -dim_m:]

    ## creating the gamma vector expanded on the whole m
    gammas = [np.pi**i for i in range(dim_h)]
    gammavec = np.zeros(dim_m)
    gammavec[-dim_h:] = gammas
    gammavec = jnp.array(gammavec)

    def loss(theta, vec_H, adj):
        # this is different to Appendix F 1 in https://arxiv.org/pdf/2104.00728
        # Making use of adjoint representation
        # should be faster, and most importantly allow for treatment of sums of paulis

        res = jnp.eye(dim_m)

        for i in range(dim_k):
            res @= jax.scipy.linalg.expm(theta[i] * adj[i])

        return gammavec @ res @ vec_H

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    [vec_H] = op_to_adjvec(
        [H], g[-dim_m:]
    )  # TODO update to also allow for dense representations using vstack

    theta0 = opt_kwargs.pop("theta0", None)
    if theta0 is None:
        theta0 = jax.random.normal(jax.random.PRNGKey(0), (dim_k,))

    thetas, energy, _ = run_opt(
        partial(value_and_grad, vec_H=vec_H, adj=adj_cropped), theta0, **opt_kwargs
    )

    if verbose >= 1:
        plt.plot(energy)  # - np.min(energy))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        # plt.yscale("log")
        plt.show()

    theta_opt = thetas[-1]

    M = jnp.eye(dim_m)

    for i in range(dim_k):
        M @= jax.scipy.linalg.expm(theta_opt[i] * adj_cropped[i])

    vec_h = M @ vec_H

    return vec_h, theta_opt


def validate_khk(H, k, m, khk_res, n, error_tol):
    """Helper function to validate a khk decomposition"""
    # validate h_elem is Hermitian
    vec_h, theta_opt = khk_res
    [h_elem] = adjvec_to_op([vec_h], m)  # sum(c * op for c, op in zip(vec_h, m))

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
    for th, op in zip(theta_opt[:], k[:]):
        Km @= jax.scipy.linalg.expm(1j * th * qml.matrix(op.operation(), wire_order=range(n)))

    H_reconstructed = Km @ qml.matrix(h_elem, wire_order=range(n)) @ Km.conj().T
    assert np.allclose(
        H_reconstructed, H_reconstructed.conj().T
    ), "Reconstructed Hamiltonian not Hermitian"

    H_m = qml.matrix(H, wire_order=range(len(H.wires)))
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
