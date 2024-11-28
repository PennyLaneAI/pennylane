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
"""A function to compute the structure constants that make up
the adjoint representation of a Lie algebra in a dense matrix representation."""

import numpy as np

from pennylane.typing import TensorLike


def structure_constants_dense(g: TensorLike, is_orthonormal: bool = True) -> TensorLike:
    r"""
    Compute the structure constants that make up the adjoint representation of a Lie algebra.

    This function computes the structure constants of a Lie algebra provided by their dense matrix representation,
    obtained from, e.g., :func:`~lie_closure_dense`.
    This is sometimes more efficient than using the sparse Pauli representations of :class:`~PauliWord` and
    :class:`~PauliSentence` that are employed in :func:`~structure_constants`, e.g., when there are few generators
    that are sums of many Paulis.

    .. seealso:: For details on the mathematical definitions, see :func:`~structure_constants` and the section "Lie algebra basics" in our `g-sim demo <https://pennylane.ai/qml/demos/tutorial_liesim/#lie-algebra-basics>`__.

    Args:
        g (np.array): The (dynamical) Lie algebra provided as dense matrices, as generated from :func:`pennylane.labs.lie.lie_closure_dense`.
            ``g`` should have shape ``(d, 2**n, 2**n)`` where ``d`` is the dimension of the algebra and ``n`` is the number of qubits. Each matrix ``g[i]`` should be Hermitian.
        is_orthonormal (bool): Whether or not the matrices in ``g`` are orthonormal with respect to the Hilbert-Schmidt inner product on
            (skew-)Hermitian matrices. If the inputs are orthonormal, it is recommended to set ``is_orthonormal`` to ``True`` to reduce
            computational cost. Defaults to ``True``.

    Returns:
        TensorLike: The adjoint representation of shape ``(d, d, d)``, corresponding to
        the indices ``(gamma, alpha, beta)``.

    **Example**

    Let us generate the DLA of the transverse field Ising model using :func:`~lie_closure_dense`.

    >>> from pennylane.labs.dla import lie_closure_dense
    >>> n = 4
    >>> gens = [qml.X(i) @ qml.X(i+1) + qml.Y(i) @ qml.Y(i+1) + qml.Z(i) @ qml.Z(i+1) for i in range(n-1)]
    >>> g = lie_closure_dense(gens)
    >>> g.shape
    (12, 16, 16)

    The DLA is represented by a collection of twelve :math:`2^4 \times 2^4` matrices.
    Hence, the dimension of the DLA is :math:`d = 12` and the structure constants have shape ``(12, 12, 12)``.

    >>> from pennylane.labs.dla import structure_constants_dense
    >>> adj = structure_constants_dense(g)
    >>> adj.shape
    (12, 12, 12)

    **Internal representation**

    As mentioned above, the input is assumed to be a batch of Hermitian matrices, even though
    algebra elements are usually skew-Hermitian. That is, the input should represent the operators
    :math:`G_\alpha` for an algebra basis :math:`\{iG_\alpha\}_\alpha`.
    In an orthonormal basis of this form, the structure constants can then be computed simply via

    .. math::

        f^\gamma_{\alpha, \beta} = \text{tr}[-i G_\gamma[iG_\alpha, iG_\beta]] = i\text{tr}[G_\gamma [G_\alpha, G_\beta]] \in \mathbb{R}.

    **Structure constants in non-orthonormal bases**

    Structure constants are often discussed using an orthonormal basis of the algebra.
    This function can deal with non-orthonormal bases as well. For this, the Gram
    matrix :math:`g` between the basis elements is taken into account when computing the overlap
    of a commutator :math:`[iG_\alpha, iG_\beta]` with all algebra elements :math:`iG_\gamma`.
    The resulting formula reads

    .. math::

        f^\gamma_{\alpha, \beta} &= \sum_\eta g^{-1}_{\gamma\eta} i \text{tr}[G_\eta [G_\alpha, G_\beta]]\\
        g_{\gamma \eta} &= \text{tr}[G_\gamma G_\eta] \quad(\in\mathbb{R})

    Internally, the commutators are computed by evaluating all operator products and subtracting
    suitable pairs of products from each other. These products can be reused to evaluate the
    Gram matrix as well.
    """

    if isinstance(g, list):
        g = np.array(g)

    assert g.shape[2] == g.shape[1]
    chi = g.shape[1]
    # Assert Hermiticity of the input. Otherwise we'll get the sign wrong
    assert np.allclose(g.conj().transpose((0, 2, 1)), g)

    # compute all commutators by computing all products first.
    # Axis ordering is (dimg, chi, _chi_) x (dimg, _chi_, chi) -> (dimg, chi, dimg, chi)
    prod = np.tensordot(g, g, axes=[[2], [1]])
    # The commutators now are the difference of prod with itself, with dimg axes swapped
    all_coms = prod - np.transpose(prod, (2, 1, 0, 3))

    # project commutators on the basis of g, see docstring for details.
    # Axis ordering is (dimg, _chi_, *chi*) x (dimg, *chi*, dimg, _chi_) -> (dimg, dimg, dimg)
    # Normalize trace inner product by dimension chi
    adj = (1j * np.tensordot(g / chi, all_coms, axes=[[1, 2], [3, 1]])).real

    if not is_orthonormal:
        # If the basis is not orthonormal, compute the Gram matrix and apply its
        # (pseudo-)inverse to the obtained projections. See the docstring for details.
        # The Gram matrix is just one additional diagonal contraction of the ``prod`` tensor,
        # across the Hilbert space dimensions. (dimg, _chi_, dimg, _chi_) -> (dimg, dimg)
        # This contraction is missing the normalization factor 1/chi of the trace inner product.
        gram_inv = np.linalg.pinv(np.sum(np.diagonal(prod, axis1=1, axis2=3), axis=-1).real)
        # Axis ordering for contraction with gamma axis of raw structure constants:
        # (dimg, _dimg_), (_dimg_, dimg, dimg) -> (dimg, dimg, dim)
        # Here we add the missing normalization factor of the trace inner product (after inversion)
        adj = np.tensordot(gram_inv * chi, adj, axes=1)

    return adj
