import warnings
from datetime import datetime
from functools import reduce

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.linalg import null_space

import pennylane as qml
import pennylane.numpy as pnp
from pennylane import I, X, Y, Z
from pennylane.pauli import PauliSentence, PauliVSpace, PauliWord
from pennylane.pauli.dla.center import _intersect_bases  # From the fix-center branch

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def run_opt(
    value_and_grad, theta, n_epochs=100, lr=0.1, b1=0.9, b2=0.999, E_exact=0.0, verbose=True
):
    """Boilerplate jax optimization"""
    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
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

        energy[n] = val
        gradients.append(grad_circuit)
        thetas.append(theta)
    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val - E_exact}; min loss: {np.min(energy) - E_exact}; after {t1 - t0}")

    return thetas, energy, gradients


def EvenOdd(op: PauliSentence):
    """Generalization of EvenOdd involution to sums of Paulis"""
    parity = []
    for pw in op.keys():
        parity.append(len(pw) % 2)

    assert all(
        parity[0] == p for p in parity
    )  # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    return parity[0]


def Involution0(op: PauliSentence):
    """Standard involution used in paper :math:`\Theta(g) = -g^T`, comes down to counting Ys"""
    parity = []
    for pw in op.keys():
        result = sum([1 if el == "Y" else 0 for el in pw.values()])
        parity.append(result % 2)

    assert all(
        parity[0] == p for p in parity
    )  # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    return parity[0]


def CartanDecomp(g, involution=None):
    """Cartan Decomposition g = k + m

    Args:
        g (List[PauliSentence]): the (dynamical) Lie algebra to decompose
        involution (callable): Involution function :math:`\Theta(\cdot)` to act on PauliSentence ops, should return ``0/1`` or ``True/False``.

    Returns:
        k (List[PauliSentence]): the even parity subspace :math:`\Theta(\mathfrak{k}) = \mathfrak{k}`
        m (List[PauliSentence]): the odd parity subspace :math:`\Theta(\mathfrak{m}) = \mathfrak{m}`
    """
    m = []
    k = []

    if involution is None:
        involution = Involution0

    for op in g:
        if involution(op):  # odd parity
            k.append(op)
        else:  # even parity
            m.append(op)
    return k, m


def make_op(vec, g):
    """Helper function for CSA"""
    if isinstance(g, PauliVSpace):
        basis = g.basis
    else:
        basis = g
    assert len(vec) == len(basis)
    res = sum([c * op for c, op in zip(vec, basis)])
    res.simplify()
    return res


def project(op, vspace):
    """Helper function for CSA"""
    if isinstance(vspace, PauliVSpace):
        basis = vspace.basis
    else:
        basis = vspace

    rep = np.zeros((len(basis),), dtype=complex)

    for i, basis_i in enumerate(basis):
        value = (basis_i @ op).trace()
        value = value / (basis_i @ basis_i).trace()  # v = ∑ (v · e_j / ||e_j||^2) * e_j

        rep[i] = value

    return rep


def gram_schmidt(X):
    """Orthogonalize basis of volumn vectors in X"""
    _, R = np.linalg.qr(X.T, mode="complete")
    return R.T


def compute_csa(g, m, which=0):
    r"""
    Compute cartan subalgebra of g given m, the odd parity subspace. I.e. the maximal Abelian subalgebra in m.

    Args:
        g (List[PauliSentence]): DLA in terms of PauliSentence instnaces
        m (List[PauliSentence]): Odd parity subspace :math:`\mathfrak{m}` from Cartan decomposition g = k + m

    Returns:
        mtilde (List[PauliSentence]): the remaining operators in m, i.e. :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{h}`
        h (List[PauliSentence]): the Cartan subalgebra :math:`\mathfrak{h}`
    """
    # brute force translation from implementation of therooler in https://github.com/therooler/homogeneous_spaces/blob/main/A%20zoo%20of%20quantum%20gates.ipynb
    # not at all optimized, some redundancies in re-computing adjoint representations
    m = m.copy()

    all_pure_words = all(len(op) == 1 for op in g) and all(len(op) == 1 for op in m)
    if all_pure_words:
        return _compute_csa_words(m, which)

    h_vspace = PauliVSpace([m[which]])
    g_vspace = PauliVSpace(g)
    m_in_basis_of_g = np.array([project(mop, g_vspace) for mop in m]).T
    while True:
        kernel_intersection = m_in_basis_of_g
        for h_i in h_vspace.basis:
            adjoint_of_h_i = []
            for x in g_vspace.basis:
                _com = h_i.commutator(x)
                # `project` does not exist yet. Could project into `m` directly?
                adjoint_of_h_i.append(project(_com, g_vspace))
            new_kernel = null_space(np.stack(adjoint_of_h_i))

            kernel_intersection = _intersect_bases(kernel_intersection, new_kernel)

        if kernel_intersection.shape[1] == len(h_vspace.basis):
            # No new vector was added from all the kernels
            break

        kernel_intersection = gram_schmidt(kernel_intersection)  # orthogonalize
        for vec in kernel_intersection.T:
            op = make_op(vec, g_vspace)

            # The following is checking linear independence and then adds operator to vspace
            # The construction is like this to avoid recomputing linear independence in vspace.add
            # TODO: add functionality in PennyLane to do this without having to assign private attributes
            (
                _M,
                _pw_to_idx,
                _rank,
                _num_pw,
                is_independent,
            ) = h_vspace._check_independence(
                h_vspace._M, op, h_vspace._pw_to_idx, h_vspace._rank, h_vspace._num_pw, h_vspace.tol
            )
            if is_independent:
                (
                    h_vspace._M,
                    h_vspace._pw_to_idx,
                    h_vspace._rank,
                    h_vspace._num_pw,
                ) = (
                    _M,
                    _pw_to_idx,
                    _rank,
                    _num_pw,
                )
                h_vspace.add(op)  # Recomputes _check_independence
                break

    mtilde_vspace = PauliVSpace([])
    for m_i in m:
        if h_vspace.is_independent(m_i):
            mtilde_vspace.add(m_i)

    return mtilde_vspace.basis, h_vspace.basis


def is_independent(v, A, tol=1e-14):
    v /= np.linalg.norm(v)
    v = v - A @ qml.math.linalg.solve(qml.math.conj(A.T) @ A, A.conj().T) @ v
    return np.linalg.norm(v) > tol


def orthogonal_complement_basis(h, m, tol):
    """find mtilde = m - h"""
    # Step 1: Find the span of h
    h = np.array(h)
    m = np.array(m)

    # Compute the orthonormal basis of h using QR decomposition
    Q, _ = np.linalg.qr(h.T)

    # Step 2: Project each vector in m onto the orthogonal complement of span(h)
    projections = m - np.dot(np.dot(m, Q), Q.T)

    # Step 3: Find a basis for the non-zero projections
    # We'll use SVD to find the basis
    U, S, _ = np.linalg.svd(projections.T)

    # Choose columns of U corresponding to non-zero singular values
    rank = np.sum(S > tol)
    basis = U[:, :rank]

    return basis.T  # Transpose to get row vectors


def linearly_independent_set_svd(vectors):
    """Find a basis given a set of vectors"""
    A = np.array(vectors).T  # Transpose to get vectors as columns

    # Perform Singular Value Decomposition
    U, S, VT = np.linalg.svd(A)

    # Determine the rank of the matrix (number of non-zero singular values)
    rank = np.sum(S > 1e-10)  # Use a threshold to determine non-zero singular values

    # Select the first 'rank' columns of U (corresponding to non-zero singular values)
    independent_vectors = U[:, :rank]

    return independent_vectors.T  # Transpose back to original orientation


def compute_csa_new(g, m, ad, which=0, tol=1e-14, verbose=0):
    r"""
    Compute cartan subalgebra of g given m, the odd parity subspace. I.e. the maximal Abelian subalgebra in m.

    Args:
        g (List[PauliSentence]): DLA in terms of PauliSentence instnaces
        m (List[PauliSentence]): Odd parity subspace :math:`\mathfrak{m}` from Cartan decomposition g = k + m
        ad (Array): The ``(|g|, |g|, |g|)`` dimensional adjoint representation of g

    Returns:
        mtilde (List[PauliSentence]): the remaining operators in m, i.e. :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{h}`
        h (List[PauliSentence]): the Cartan subalgebra :math:`\mathfrak{h}`
    """
    # brute force translation from implementation of therooler in https://github.com/therooler/homogeneous_spaces/blob/main/A%20zoo%20of%20quantum%20gates.ipynb
    # not at all optimized, some redundancies in re-computing adjoint representations
    m = m.copy()

    # all_pure_words = all(len(op)==1 for op in g) and all(len(op)==1 for op in m)
    # if all_pure_words:
    #     return _compute_csa_words(m, which)

    g_vspace = PauliVSpace(g)
    np_m = np.array([project(mop, g_vspace) for mop in m]).T
    np_h = [project(m[which], g_vspace)]

    iteration = 1
    while True:
        print(f"iteration: {iteration}") if verbose else None
        kernel_intersection = np_m
        for h_i in np_h:

            # obtain adjoint rep of candidate h_i
            adjoint_of_h_i = np.einsum("gab,a->gb", ad, h_i)
            # compute kernel of adjoint
            new_kernel = null_space(adjoint_of_h_i, rcond=tol)

            # intersect kernel to stay in m
            kernel_intersection = _intersect_bases(kernel_intersection, new_kernel, rcond=tol)

        if kernel_intersection.shape[1] == len(np_h):
            # No new vector was added from all the kernels
            break

        kernel_intersection = gram_schmidt(kernel_intersection)  # orthogonalize
        for vec in kernel_intersection.T:
            if is_independent(vec, np.array(np_h).T, tol):
                np_h.append(vec)
                break

        iteration += 1

    # turn numpy array into operators
    h_vspace = PauliVSpace([])
    for h_i in np_h:
        h_i_op = make_op(h_i, g_vspace)
        h_vspace.add(h_i_op, tol=tol)

    # project
    np_mtilde = orthogonal_complement_basis(np_h, np.array(np_m).T, tol=tol)
    mtilde_vspace = PauliVSpace([])
    for m_i in np_mtilde:
        m_i_op = make_op(m_i, g_vspace)
        mtilde_vspace.add(m_i_op, tol=tol)

    return mtilde_vspace.basis, h_vspace.basis


def _compute_csa_words(m, which=0):
    """compute the Cartan subalgebra from the odd parity space :math:`\mathfrak{m}` of the Cartan decomposition

    The Cartan subalgebra is the maximal Abelian subalgebra of :math:`\mathfrak{m}`.

    This implementation is specific for cases of bases of m with pure Pauli words as detailed in Appendix C in `2104.00728 <https://arxiv.org/abs/2104.00728>`__.

    Args:
        m (List[PauliSentence]): the odd parity subspace :math:`\Theta(\mathfrak{m}) = \mathfrak{m}
        which (int): Choice for initial element of m from which to construct the maximal Abelian subalgebra

    Returns:
        mtilde (List): remaining elements of :math:`\mathfrak{m}` s.t. :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{h}`.
        h (List): Cartan subalgebra

    """
    m = m.copy()

    m0 = m[which]

    h = [m0]

    for m_i in m:
        commutes_with = 0
        for h_j in h:
            com = h_j.commutator(m_i)
            com.simplify()

            if len(com) != 0:
                commutes_with += 1

        if commutes_with == 0:
            if m_i not in h:
                h.append(m_i)

    for h_i in h:
        m.remove(h_i)

    return m, h


def khk_decompose(
    generators, H, theta0=None, n_epochs=500, validate=True, involution=None, verbose=1, tol=1e-12
):
    r"""The full KhK decomposition of a Hamiltonian H

    We are decomposing an :math:`H \ in \mathfrak{m}` into :math:`H = K^\dagger h K`
    using the techniques outlines in https://arxiv.org/abs/2104.00728 with the extension
    to handle sums of Paulis by computing expressions :math:`KgK^\dagger` using the algebra (i.e. adjoint representation)
    This is outlined in the temporary doc https://typst.app/project/rhhKohoJZ3hGFhqXZmfS8S

    Args:
        generators (List[PauliSentence]): generators of the Lie algebra
        H (Operator): Hamiltonian to decompose
        theta0 (JaxArray): initial guess for the optimization (requires knowledge of the resulting Lie algebra)
        n_epochs (int): number of iterations for the optimization
        validate (bool): check that the decomposition indeed reproduces H = K* h K

    Returns:
        vec_h: coefficients of the element :math:`h \in \mathfrak{h}` in the basis of :math:`\mathfrak{g}`
        theta_opt: optimal coefficients for :math:`\prod_j exp(-i \theta_j k_j)` for the operators :math:`k_j \in \mathfrak{k}`
        k: even parity subspace :math:`\mathfrak{k}` of :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`
        mtilde: remainder of odd parity subspace :math:`\mathfrak{m} = \tilde{\mathfrak{m}} \oplus \mathfrak{h}`
        h: the maximal Abelian subalgebra of :math:`\mathfrak{m}`
        ad: the ``(dim(g), dim(g), dim(g))`` full adjoint representation of :math:`\mathfrak{g}`, sorted in the order
            :math:`\mathfrak{g} = \mathfrak{k} \oplus \tilde{\mathfrak{m}} \oplus \mathfrak{h}`


    """
    if involution is None:
        involution = Involution0
    elif involution == "EvenOdd":
        involution = EvenOdd

    generators = [op.pauli_rep for op in generators]
    g = qml.lie_closure(generators, pauli=True)
    g = orthonormalize(g)

    print("Computing Cartan decomposition g = m + k") if verbose else None
    k, m = CartanDecomp(g, involution=involution)

    g = k + m  # reorder g

    k = orthonormalize(k)
    m = orthonormalize(m)

    (
        print(f"Cartan decomposition g = k + m with dimensions {len(g)} = {len(k)} + {len(m)}")
        if verbose
        else None
    )

    print("Computing adjoint representation of g = k + m") if verbose else None
    ad = qml.structure_constants(g)

    print("Computing Cartan subalgebra m = mtilde + h") if verbose else None
    mtilde, h = compute_csa_new(g, m, ad, tol=tol)

    # if validate:
    #     # TODO this validation is not correct
    #     ranks = []
    #     for adrep in ad[-len(m) :]:
    #         kernel = null_space(adrep, rcond=tol)
    #         ranks.append(kernel.shape[1])

    #     CSA_dim = np.min(ranks)

    #     correct_CSA_dim = CSA_dim == len(h)
    #     print(f"Correct CSA dimension of {CSA_dim}: {correct_CSA_dim} (i.e. len(h) = {len(h)})")

    h = orthonormalize(h)
    k = orthonormalize(k)
    mtilde = orthonormalize(mtilde)

    g = k + mtilde + h  # reorder g

    (
        print(
            f"Cartan decomposition g = k + mtilde + h with dimensions {len(g)} = {len(k)} + {len(mtilde)} + {len(h)}"
        )
        if verbose
        else None
    )

    print("Computing adjoint representation of g = k + mtilde + h") if verbose else None
    ad = qml.structure_constants(g)

    # khk decomposition

    # mututally irrational, see Example 10 on page 10 in https://arxiv.org/pdf/quant-ph/0505128
    gammas = [np.pi**i for i in range(len(h))]

    vec_H = project(H.pauli_rep, g).real

    ## creating the gamma vector expanded on the whole g
    gammavec = np.zeros(len(g))
    gammavec[-len(h) :] = gammas

    gammavec = jnp.array(gammavec)

    def loss(theta):
        # this is different to Appendix F 1 in https://arxiv.org/pdf/2104.00728
        # Making use of adjoint representation
        # should be faster, and most importantly allow for treatment of sums of paulis

        M = jax.scipy.linalg.expm(theta[0] * ad[0])

        for i in range(1, len(k)):
            M @= jax.scipy.linalg.expm(theta[i] * ad[i])

        return gammavec @ M @ vec_H

    if theta0 is None:
        theta0 = jnp.ones(len(k), dtype=float)

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    thetas, energy, _ = run_opt(value_and_grad, theta0, n_epochs=n_epochs)
    if verbose:
        plt.plot(energy)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()

    theta_opt = thetas[-1]

    M = jnp.eye(len(g))

    for i in range(len(k)):
        M @= jax.scipy.linalg.expm(theta_opt[i] * ad[i])

    vec_h = M @ vec_H

    if validate:
        h_elem = sum(c * op for c, op in zip(vec_h, g))
        h_elem.simplify(1e-10)

        n = len(H.wires)

        Km = jnp.eye(2**n)
        for th, op in zip(theta_opt, k):
            Km @= jax.scipy.linalg.expm(1j * th * qml.matrix(op.operation(), wire_order=range(n)))

        H_reconstructed = Km.conj().T @ qml.matrix(h_elem, wire_order=range(n)) @ Km

        H_m = qml.matrix(H, wire_order=range(len(H.wires)))
        success = np.allclose(H_m, H_reconstructed)

        if not success:
            # more expensive check for unitary equivalence
            success = 1 - np.linalg.norm(
                np.linalg.eigvalsh(H_m) - np.linalg.eigvalsh(H_reconstructed)
            )
            warnings.warn(
                "The reconstructed H is not numerical identical to the original H.\n"
                f"We can still check for unitary equivalence: {success}",
                UserWarning,
            )

        print(f"success: {success}")

    return vec_h, theta_opt, k, mtilde, h, ad


# gram schmidt with respect to R2 metric


def orthonormalize(vspace):
    if isinstance(vspace, PauliVSpace):
        vspace = vspace.basis

    if not all(isinstance(op, PauliSentence) for op in vspace):
        vspace = [op.pauli_rep for op in vspace]

    all_pws = list(reduce(set.__or__, [set(ps.keys()) for ps in vspace]))
    num_pw = len(all_pws)

    _pw_to_idx = {pw: i for i, pw in enumerate(all_pws)}
    _idx_to_pw = {i: pw for i, pw in enumerate(all_pws)}
    _M = np.zeros((num_pw, len(vspace)), dtype=float)

    for i, gen in enumerate(vspace):
        for pw, value in gen.items():
            _M[_pw_to_idx[pw], i] = value

    def gram_schmidt(X):
        Q, _ = np.linalg.qr(X)
        return Q

    OM = gram_schmidt(_M)
    for i in range(2):
        for j in range(2):
            prod = OM[:, i] @ OM[:, j]
            if i == j:
                assert np.isclose(prod, 1)
            else:
                assert np.isclose(prod, 0)

    # reconstruct normalized operators

    generators_orthogonal = []
    for i in range(len(vspace)):
        u1 = PauliSentence({})
        for j in range(num_pw):
            u1 += _idx_to_pw[j] * OM[j, i]
        u1.simplify()
        generators_orthogonal.append(u1)

    return generators_orthogonal
