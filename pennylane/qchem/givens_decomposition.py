# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the functions needed for performing basis transformations defined by a set of fermionic ladder operators.
"""

import pennylane as qml


def _givens_matrix(a, b, left=True, tol=1e-8):
    r"""Build a :math:`2 \times 2` Givens rotation matrix :math:`G`.

    When the matrix :math:`G` is applied to a vector :math:`[a,\ b]^T` the following would happen:

    .. math::

            G \times \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} 0 \\ r \end{bmatrix} \quad \quad \quad \begin{bmatrix} a \\ b \end{bmatrix} \times G = \begin{bmatrix} r \\ 0 \end{bmatrix},

    where :math:`r` is a complex number.

    Args:
        a (float or complex): first element of the vector for which the Givens matrix is being computed
        b (float or complex): second element of the vector for which the Givens matrix is being computed
        left (bool): determines if the Givens matrix is being applied from the left side or right side.
        tol (float): determines tolerance limits for :math:`|a|` and :math:`|b|` under which they are considered as zero.

    Returns:
        tensor_like: Givens rotation matrix

    """
    abs_a, abs_b, interface = qml.math.abs(a), qml.math.abs(b), qml.math.get_interface(a)
    aprod = qml.math.nan_to_num(abs_b * abs_a)
    hypot = qml.math.hypot(abs_a, abs_b)

    cosine = qml.math.where(abs_a < tol, 1.0, qml.math.where(abs_b < tol, 0.0, abs_b / hypot))
    sine = qml.math.where(abs_a < tol, 0.0, qml.math.where(abs_b < tol, 1.0, abs_a / hypot))
    phase = qml.math.where(
        abs_a < tol,
        1.0,
        qml.math.where(abs_b < tol, 1.0, (1.0 * b * qml.math.conj(a)) / (aprod + 1e-15)),
    )

    if left:
        return qml.math.array([[phase * cosine, -sine], [phase * sine, cosine]], like=interface)

    return qml.math.array([[phase * sine, cosine], [-phase * cosine, sine]], like=interface)


# pylint:disable = too-many-branches
def givens_decomposition(unitary):
    r"""Decompose a unitary into a sequence of Givens rotation gates with phase shifts and a diagonal phase matrix.

    This decomposition is based on the construction scheme given in `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ ,
    which allows one to write any unitary matrix :math:`U` as:

    .. math::

        U = D \left(\prod_{(m, n) \in G} T_{m, n}(\theta, \phi)\right),

    where :math:`D` is a diagonal phase matrix, :math:`T(\theta, \phi)` is the Givens rotation gates with phase shifts and :math:`G` defines the
    specific ordered sequence of the Givens rotation gates acting on wires :math:`(m, n)`. The unitary for the :math:`T(\theta, \phi)` gates
    appearing in the decomposition is of the following form:

    .. math:: T(\theta, \phi) = \begin{bmatrix}
                                    1 & 0 & 0 & 0 \\
                                    0 & e^{i \phi} \cos(\theta) & -\sin(\theta) & 0 \\
                                    0 & e^{i \phi} \sin(\theta) & \cos(\theta) & 0 \\
                                    0 & 0 & 0 & 1
                                \end{bmatrix},

    where :math:`\theta \in [0, \pi/2]` is the angle of rotation in the :math:`\{|01\rangle, |10\rangle \}` subspace
    and :math:`\phi \in [0, 2 \pi]` represents the phase shift at the first wire.

    **Example**

    .. code-block:: python

        unitary = np.array([[ 0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                            [-0.21271+0.34938j, -0.38853+0.36497j,  0.61467-0.41317j],
                            [ 0.41356-0.20765j, -0.00651-0.66689j,  0.32839-0.48293j]])

        phase_mat, ordered_rotations = givens_decomposition(unitary)

    >>> phase_mat
    tensor([-0.20604358+0.9785369j , -0.82993272+0.55786114j,
            0.56230612-0.82692833j], requires_grad=True)
    >>> ordered_rotations
    [(tensor([[-0.65087861-0.63937521j, -0.40933651-0.j        ],
            [-0.29201359-0.28685265j,  0.91238348-0.j        ]], requires_grad=True),
    (0, 1)),
    (tensor([[ 0.47970366-0.33308926j, -0.8117487 -0.j        ],
            [ 0.66677093-0.46298215j,  0.5840069 -0.j        ]], requires_grad=True),
    (1, 2)),
    (tensor([[ 0.36147547+0.73779454j, -0.57008306-0.j        ],
            [ 0.2508207 +0.51194108j,  0.82158706-0.j        ]], requires_grad=True),
    (0, 1))]

    Args:
        unitary (tensor): unitary matrix on which decomposition will be performed

    Returns:
        (tensor_like, list[(tensor_like, tuple)]): diagonal elements of the phase matrix :math:`D` and Givens rotation matrix :math:`T` with their indices.

    Raises:
        ValueError: if the provided matrix is not square.

    .. details::
        :title: Theory and Pseudocode

        **Givens Rotation**

        Applying the Givens rotation :math:`T(\theta, \phi)` performs the following transformation of the basis states:

        .. math::

            &|00\rangle \mapsto |00\rangle\\
            &|01\rangle \mapsto e^{i \phi} \cos(\theta) |01\rangle - \sin(\theta) |10\rangle\\
            &|10\rangle \mapsto e^{i \phi} \sin(\theta) |01\rangle + \cos(\theta) |10\rangle\\
            &|11\rangle \mapsto |11\rangle.

        **Pseudocode**

        The algorithm that implements the decomposition is the following:

        .. code-block:: python

            def givens_decomposition(U):
                for i in range(1, N):
                    if i % 2:
                        for j in range(0, i):
                            # Find T^-1(i-j, i-j+1) matrix that nulls element (N-j, i-j) of U
                            # Update U = U @ T^-1(i-j, i-j+1)
                    else:
                        for j in range(1, i):
                            # Find T(N+j-i-1, N+j-i) matrix that nulls element (N+j-i, j) of U
                            # Update U = T(N+j-i-1, N+j-i) @ U

    """
    interface = qml.math.get_deep_interface(unitary)
    unitary = qml.math.copy(unitary) if interface == "jax" else qml.math.toarray(unitary).copy()
    M, N = qml.math.shape(unitary)

    if M != N:
        raise ValueError(f"The unitary matrix should be of shape NxN, got {unitary.shape}")

    left_givens, right_givens = [], []
    if interface == "jax":
        for i in range(1, N):
            if i % 2:
                for j in range(0, i):
                    indices = [i - j - 1, i - j]
                    grot_mat = _givens_matrix(*unitary[N - j - 1, indices].T, left=True)
                    unitary = unitary.at[:, indices].set(
                        unitary[:, indices] @ grot_mat.T,
                        indices_are_sorted=True,
                        unique_indices=True,
                    )
                    right_givens.append((qml.math.conj(grot_mat), indices))
            else:
                for j in range(1, i + 1):
                    indices = [N + j - i - 2, N + j - i - 1]
                    grot_mat = _givens_matrix(*unitary[indices, j - 1], left=False)
                    unitary = unitary.at[indices, :].set(
                        grot_mat @ unitary[indices, :], indices_are_sorted=True, unique_indices=True
                    )
                    left_givens.append((grot_mat, indices))
    else:
        for i in range(1, N):
            if i % 2:
                for j in range(0, i):
                    indices = [i - j - 1, i - j]
                    grot_mat = _givens_matrix(*unitary[N - j - 1, indices].T, left=True)
                    unitary[:, indices] = unitary[:, indices] @ grot_mat.T
                    right_givens.append((qml.math.conj(grot_mat), indices))
            else:
                for j in range(1, i + 1):
                    indices = [N + j - i - 2, N + j - i - 1]
                    grot_mat = _givens_matrix(*unitary[indices, j - 1], left=False)
                    unitary[indices] = grot_mat @ unitary[indices]
                    left_givens.append((grot_mat, indices))

    nleft_givens = []
    for grot_mat, (i, j) in reversed(left_givens):
        sphase_mat = qml.math.diag(qml.math.diag(unitary)[qml.math.array([i, j])])
        decomp_mat = qml.math.conj(grot_mat).T @ sphase_mat
        givens_mat = _givens_matrix(*decomp_mat[1, :].T)
        nphase_mat = decomp_mat @ givens_mat.T

        # check for T_{m,n}^{-1} x D = D x T.
        if not qml.math.is_abstract(decomp_mat) and not qml.math.allclose(
            nphase_mat @ qml.math.conj(givens_mat), decomp_mat
        ):  # pragma: no cover
            raise ValueError("Failed to shift phase transposition.")

        if interface == "jax":
            unitary = unitary.at[i, i].set(qml.math.diag(nphase_mat)[0])
            unitary = unitary.at[j, j].set(qml.math.diag(nphase_mat)[1])
        else:
            unitary[i, i], unitary[j, j] = qml.math.diag(nphase_mat)
        nleft_givens.append((qml.math.conj(givens_mat), (i, j)))

    phases, ordered_rotations = qml.math.diag(unitary), []
    for grot_mat, (i, j) in list(reversed(nleft_givens)) + list(reversed(right_givens)):
        if not qml.math.is_abstract(grot_mat) and not qml.math.all(
            qml.math.isreal(grot_mat[0, 1]) and qml.math.isreal(grot_mat[1, 1])
        ):  # pragma: no cover
            raise ValueError(f"Incorrect Givens Rotation encountered, {grot_mat}")
        ordered_rotations.append((grot_mat, (i, j)))

    return phases, ordered_rotations
