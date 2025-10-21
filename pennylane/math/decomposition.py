# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for decompositions are available from ``qml.math.decomposition``."""

from functools import lru_cache, partial

import numpy as np

from pennylane import math

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover
    ...


def zyz_rotation_angles(U, return_global_phase=False):
    r"""Compute the rotation angles :math:`\phi`, :math:`\theta`, and :math:`\omega` and the
    phase :math:`\alpha` of a 2x2 unitary matrix as a product of Z and Y rotations in the form
    :math:`e^{i\alpha} RZ(\omega) RY(\theta) RZ(\phi)`

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    Returns:
        tuple: The rotation angles :math:`\phi`, :math:`\theta`, and :math:`\omega` and the
        global phase :math:`\alpha` if ``return_global_phase=True``.

    """

    U, alpha = math.convert_to_su2(U, return_global_phase=True)
    # assume U = [[a, b], [c, d]], then here we take U[0, 1] as b
    abs_b = math.clip(math.abs(U[..., 0, 1]), 0, 1)
    theta = 2 * math.arcsin(abs_b)

    EPS = math.finfo(U.dtype).eps
    half_phi_plus_omega = math.angle(U[..., 1, 1] + EPS)
    half_omega_minus_phi = math.angle(U[..., 1, 0] + EPS)

    phi = half_phi_plus_omega - half_omega_minus_phi
    omega = half_phi_plus_omega + half_omega_minus_phi

    # Normalize the angles
    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    omega = math.squeeze(omega % (4 * np.pi))

    return (phi, theta, omega, alpha) if return_global_phase else (phi, theta, omega)


def xyx_rotation_angles(U, return_global_phase=False):
    r"""Compute the rotation angles :math:`\lambda`, :math:`\theta`, and :math:`\phi` and the
    phase :math:`\alpha` of a 2x2 unitary matrix as a product of X and Y rotations in the form
    :math:`e^{i\alpha} RX(\phi) RY(\theta) RX(\lambda)`.

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    Returns:
        tuple: The rotation angles :math:`\lambda`, :math:`\theta`, and :math:`\phi` and the
        global phase :math:`\alpha` if ``return_global_phase=True``.

    """

    U, alpha = math.convert_to_su2(U, return_global_phase=True)

    EPS = math.finfo(U.dtype).eps
    half_lam_plus_phi = math.arctan2(-math.imag(U[..., 0, 1]), math.real(U[..., 0, 0]) + EPS)
    half_lam_minus_phi = math.arctan2(math.imag(U[..., 0, 0]), -math.real(U[..., 0, 1]) + EPS)
    lam = half_lam_plus_phi + half_lam_minus_phi
    phi = half_lam_plus_phi - half_lam_minus_phi

    theta = math.where(
        math.isclose(math.sin(half_lam_plus_phi), math.zeros_like(half_lam_plus_phi)),
        2 * math.arccos(math.clip(math.real(U[..., 1, 1]) / math.cos(half_lam_plus_phi), -1, 1)),
        2 * math.arccos(math.clip(-math.imag(U[..., 0, 1]) / math.sin(half_lam_plus_phi), -1, 1)),
    )

    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    lam = math.squeeze(lam % (4 * np.pi))

    return (lam, theta, phi, alpha) if return_global_phase else (lam, theta, phi)


def xzx_rotation_angles(U, return_global_phase=False):
    r"""Compute the rotation angles :math:`\lambda`, :math:`\theta`, and :math:`\phi` and the
    phase :math:`\alpha` of a 2x2 unitary matrix as a product of X and Z rotations in the form
    :math:`e^{i\alpha} RX(\phi) RZ(\theta) RX(\lambda)`.

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    Returns:
        tuple: The rotation angles :math:`\lambda`, :math:`\theta`, and :math:`\phi` and the
        global phase :math:`\alpha` if ``return_global_phase=True``.

    """

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)
    EPS = math.finfo(U.dtype).eps

    # Compute \phi, \theta and \lambda after analytically solving for them from
    # U = RX(\phi) RZ(\theta) RX(\lambda)
    sum_diagonal_real = math.real(U[..., 0, 0] + U[..., 1, 1])
    sum_off_diagonal_imag = math.imag(U[..., 0, 1] + U[..., 1, 0])
    half_phi_plus_lambdas = math.arctan2(-sum_off_diagonal_imag, sum_diagonal_real + EPS)
    diff_diagonal_imag = math.imag(U[..., 0, 0] - U[..., 1, 1])
    diff_off_diagonal_real = math.real(U[..., 0, 1] - U[..., 1, 0])
    half_phi_minus_lambdas = math.arctan2(diff_off_diagonal_real, -diff_diagonal_imag + EPS)
    lam = half_phi_plus_lambdas - half_phi_minus_lambdas
    phi = half_phi_plus_lambdas + half_phi_minus_lambdas

    # Compute \theta
    theta = math.where(
        math.isclose(math.sin(half_phi_plus_lambdas), math.zeros_like(half_phi_plus_lambdas)),
        2
        * math.arccos(
            math.clip(sum_diagonal_real / (2 * math.cos(half_phi_plus_lambdas) + EPS), -1, 1)
        ),
        2
        * math.arccos(
            math.clip(-sum_off_diagonal_imag / (2 * math.sin(half_phi_plus_lambdas) + EPS), -1, 1)
        ),
    )

    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    lam = math.squeeze(lam % (4 * np.pi))

    return (lam, theta, phi, global_phase) if return_global_phase else (lam, theta, phi)


def zxz_rotation_angles(U, return_global_phase=False):
    r"""Compute the rotation angles :math:`\lambda`, :math:`\theta`, and :math:`\phi` and the
    phase :math:`\alpha` of a 2x2 unitary matrix as a product of Z and X rotations in the form
    :math:`e^{i\alpha} RZ(\phi) RX(\theta) RZ(\lambda)`.

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    Returns:
        tuple: The rotation angles :math:`\lambda`, :math:`\theta`, and :math:`\phi` and the
        global phase :math:`\alpha` if ``return_global_phase=True``.

    """

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)
    EPS = math.finfo(U.dtype).eps

    abs_a = math.clip(math.abs(U[..., 0, 0]), 0, 1)
    abs_b = math.clip(math.abs(U[..., 0, 1]), 0, 1)
    theta = math.where(abs_a < abs_b, 2 * math.arccos(abs_a), 2 * math.arcsin(abs_b))

    half_phi_plus_lam = math.angle(U[..., 1, 1] + EPS)
    half_phi_minus_lam = math.angle(1j * U[..., 1, 0] + EPS)

    phi = half_phi_plus_lam + half_phi_minus_lam
    lam = half_phi_plus_lam - half_phi_minus_lam

    # Normalize the angles
    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    lam = math.squeeze(lam % (4 * np.pi))

    return (lam, theta, phi, global_phase) if return_global_phase else (lam, theta, phi)


def su2su2_to_tensor_products(U):
    r"""Given a matrix :math:`U = A \otimes B` in SU(2) x SU(2), extract A and B

    This process has been described in detail in the Appendix of Coffey & Deiotte
    https://link.springer.com/article/10.1007/s11128-009-0156-3

    """

    # First, write A = [[a1, a2], [-a2*, a1*]], which we can do for any SU(2) element.
    # Then, A \otimes B = [[a1 B, a2 B], [-a2*B, a1*B]] = [[C1, C2], [C3, C4]]
    # where the Ci are 2x2 matrices.
    C1 = U[0:2, 0:2]
    C2 = U[0:2, 2:4]
    C3 = U[2:4, 0:2]
    C4 = U[2:4, 2:4]

    # From the definition of A \otimes B, C1 C4^\dag = a1^2 I, so we can extract a1
    C14 = math.dot(C1, math.conj(math.T(C4)))
    a1 = math.sqrt(math.cast_like(C14[0, 0], 1j))

    # Similarly, -C2 C3^\dag = a2^2 I, so we can extract a2
    C23 = math.dot(C2, math.conj(math.T(C3)))
    a2 = math.sqrt(-math.cast_like(C23[0, 0], 1j))

    # This gets us a1, a2 up to a sign. To resolve the sign, ensure that
    # C1 C2^dag = a1 a2* I
    C12 = math.dot(C1, math.conj(math.T(C2)))
    a2 = math.cond(math.allclose(a1 * math.conj(a2), C12[0, 0]), lambda: a2, lambda: -1 * a2, ())

    # Construct A
    A = math.stack([math.stack([a1, a2]), math.stack([-math.conj(a2), math.conj(a1)])])

    # Next, extract B. Can do from any of the C, just need to be careful in
    # case one of the elements of A is 0.
    # We use B1 unless division by 0 would cause all elements to be inf.
    B = math.cond(
        math.allclose(a1, 0.0, atol=1e-6),
        lambda: C2 / math.cast_like(a2, 1j),
        lambda: C1 / math.cast_like(a1, 1j),
        (),
    )

    return math.convert_like(A, U), math.convert_like(B, U)


def decomp_int_to_powers_of_two(k: int, n: int) -> list[int]:
    r"""Decompose an integer :math:`k<=2^{n-1}` into additions and subtractions of the
    smallest-possible number of powers of two.

    Args:
        k (int): Integer to be decomposed
        n (int): Number of bits to consider

    Returns:
        list[int]: A list with length ``n``, with entry :math:`c_i` at position :math:`i`.

    This function is documented in ``pennylane/ops/qubit/pcphase_decomposition.md``.

    As an example, consider the number
    :math:`k=121_{10}=01111001_2`, which can be (trivially) decomposed into a sum of
    five powers of two by reading off the bits:
    :math:`k = 2^6 + 2^5 + 2^4 + 2^3 + 2^0 = 64 + 32 + 16 + 8 + 1`.
    The decomposition here, however, allows for minus signs and achieves the decomposition
    :math:`k = 2^7 - 2^3 + 2^0 = 128 - 8 + 1`, which only requires three powers of two.
    """
    R = []
    assert k <= 2 ** (n - 1)
    s = 0
    powers = 2 ** np.arange(n)
    for p in powers:  # p = 2**(n-1-i)
        if s & p == k & p:
            # Equal bit, move on
            factor = 0
        else:
            # Differing bit, consider pairs of bits
            if p >= 2 ** (n - 2):
                # 2**(n-1-i) >= 2**(n-2) is the same condition as i < 2
                factor = 1
            else:
                # Table entry from documentation
                in_middle_rows = (s & (p + 2 * p)).bit_count() == 1  # two bits of s are 01 or 10
                in_last_cols = bool(k & (2 * p))  # latter bit of k is 1
                if in_middle_rows != in_last_cols:  # xor between in_middle_rows and in_last_cols
                    factor = -1
                else:
                    factor = 1

            s += factor * p
        R.insert(0, factor)

    return R


def _set_unitary_matrix(unitary_matrix, index, value, like=None):
    """Set the values in the ``unitary_matrix`` at the specified index.

    Args:
        unitary_matrix (tensor_like): unitary being modified
        index (Tuple[Int | Ellipsis | List[Int]]): index for slicing the unitary
        value (tensor_like): new values for the specified index
        like (str): interface for the unitary matrix

    Returns:
        tensor_like: modified unitary

    Examples:
        A = np.eye(5)
        A = _set_unitary_matrix(A, (0, 0), 5)
        A = _set_unitary_matrix(A, (1, Ellipsis), np.array([1, 2, 3, 4, 5]))
        A = _set_unitary_matrix(A, (1, [1, 2]), np.array([3, 4]))
    """
    if like is None:
        like = math.get_interface(unitary_matrix)

    if like == "jax":
        return unitary_matrix.at[index[0], index[1]].set(
            value, indices_are_sorted=True, unique_indices=True
        )

    unitary_matrix[index[0], index[1]] = value
    return unitary_matrix


def _givens_matrix_core(a, b, left=True, tol=1e-8, real_valued=False):
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

    Raises:
        TypeError: if a and b have different interfaces

    """

    abs_a, abs_b = math.abs(a), math.abs(b)
    interface_a, interface_b = math.get_interface(a), math.get_interface(b)

    if interface_a != interface_b:
        raise TypeError(
            f"The interfaces of 'a' and 'b' do not match. Got {interface_a} and {interface_b}."
        )

    interface = interface_a

    EPS = math.finfo(math.asarray(a).dtype).eps
    hypot = math.hypot(abs_a, abs_b) + EPS  # avoid division by zero

    cosine = math.where(abs_b < tol, 0.0, abs_b / hypot)
    cosine = math.where(abs_a < tol, 1.0, cosine)

    sine = math.where(abs_b < tol, 1.0, abs_a / hypot)
    sine = math.where(abs_a < tol, 0.0, sine)

    # The following logic produces four different matrices for the four possible settings
    # of ``left`` and ``real_valued``:
    # (left, real_valued) = (True, True)
    # array = [[cosine, -phase * sine], [phase * sine, cosine]]
    # (left, real_valued) = (False, True)
    # array = [[sine, phase * cosine], [-phase * cosine, sine]]
    # (left, real_valued) = (True, False)
    # array = [[phase * cosine, -sine], [phase * sine, cosine]]
    # (left, real_valued) = (False, False)
    # array = [[phase * sine, cosine], [-phase * cosine, sine]]
    if interface == "jax":
        cosine, sine = jax.lax.cond(
            left,
            lambda cosine, sine: (cosine, sine),
            lambda cosine, sine: (sine, -cosine),
            cosine,
            sine,
        )
    elif not left:
        cosine, sine = sine, -cosine

    g00, g01 = cosine, -sine
    if real_valued:
        phase = math.where((abs_a < tol) + (abs_b < tol), 1.0, math.sign(a * b))  # previously sign
        g01 *= phase
    else:
        aprod = math.nan_to_num(abs_b * abs_a)
        phase = math.where(abs_b < tol, 1.0, (b * math.conj(a)) / (aprod + EPS))
        phase = math.where(abs_a < tol, 1.0, phase)
        g00 = phase * g00

    g10, g11 = phase * sine, cosine

    return math.array([[g00, g01], [g10, g11]], like=interface)


def _absorb_phases_so(left_givens, right_givens, phases, interface):
    r"""Function handling the diagonal phases left over from diagonalization via Givens
    rotations, for the real-valued case.

    Args:
        left_givens (list[tuple[array, int]]): Givens rotations applied from the left
            in the obtained decomposition (see details below). Each rotation is given as a
            :math:`2\times 2` matrix describing the rotation and two (sorted) integers encoding
            the matrix dimensions to which the rotation is applied.
        right_givens (list[tuple[array, int]]): Givens rotations applied from the right
            in the obtained decomposition (see details below). The format is as for ``left_givens``
        phases (array): Result of the diagonalization via Givens rotations (see details below).
            Will be diagonal and only contain :math:`\pm 1`.
        interface (str): The ML interface of ``phases``.

    Returns:
        tuple[array, list[tuple[array,int]]]: New phases with at most one entry :math:`-1`, and
        concatenated list of Givens rotations that form a new valid decomposition of the originally
        decomposed matrix. The format for each rotation is like that for ``left_givens``.

    The rotations in `left_givens` and `right_givens` are such that
    `reduce(dot, [mat.T for mat in left_givens] + reversed(right_givens)) == phases`
    and the phases are guaranteed to be :math:`\pm 1`.
    This function goes through the last `N-1` nearest-neighbour Givens rotations added
    in the diagonalization. They are the last ones in `left_givens` (`right_givens`) if `N` is
    odd (even). For each of these rotations, we denote the matrix index `idx0` (`idx1`) that will (not)
    be affected by any subsequent Givens rotation. Then, depending on the phases for these indices,
    we do the following:

    - If both phases are 1, do nothing.
    - If both phases are -1, absorb them in the rotation by multiplying it with -np.eye(2).
    - If the phases at `idx0, idx1` are `-1, 1`, absorb two phases in the rotation and commute
      the created phase at `idx1` through the rotation, causing it to transpose.
    - If the phases are `idx, idx1` are `1, -1`, commute the phase through the rotation, causing
      it to transpose.

    This can be simplified to two independent moves:
    - If the first phase is -1, absorb two phases in the rotation
    - If the second phase is -1, commute a phase on index `idx1` through the rotation, transposing it.

    The phases with -1 are guaranteed to come in an even number, so that this procedure will
    end up with modified rotations and the identity as phase matrix.
    """
    N = len(phases)
    mod = N % 2
    last_rotations = left_givens if mod else right_givens
    for k in range(len(last_rotations) - 1, len(last_rotations) - N, -1):
        grot_mat, (i, j) = last_rotations[k]

        ph0 = math.sign(phases[j, j]) if mod else math.sign(phases[i, i])
        phases = _set_unitary_matrix(phases, (i, i), ph0 * phases[i, i], like=interface)
        phases = _set_unitary_matrix(phases, (j, j), ph0 * phases[j, j], like=interface)
        grot_mat = ph0 * grot_mat
        ph1 = phases[i, i] if mod else phases[j, j]
        if interface == "jax":
            grot_mat = jax.lax.cond(ph1 < 0, lambda mat: mat.T, lambda mat: mat, grot_mat)

        elif ph1 < 0:
            grot_mat = grot_mat.T

        last_rotations[k] = (grot_mat, (i, j))

    left_givens = [(mat.T, indices) for mat, indices in left_givens]
    return math.diag(phases), left_givens + list(reversed(right_givens))


def _commute_phases_u(left_givens, right_givens, phases, interface):
    r"""Function handling the diagonal phases left over from diagonalization via Givens
    rotations, for the complex-valued case.

    Args:
        left_givens (list[tuple[array, int]]): Givens rotations applied from the left
            in the obtained decomposition (see details below). Each rotation is given as a
            :math:`2\times 2` matrix describing the rotation and two (sorted) integers encoding
            the matrix dimensions to which the rotation is applied.
        right_givens (list[tuple[array, int]]): Givens rotations applied from the right
            in the obtained decomposition (see details below). The format is as for ``left_givens``
        phases (array): Result of the diagonalization via Givens rotations (see details below).
            Will be diagonal and only contain complex phases :math:`e^{i\phi}`.
        interface (str): The ML interface of ``phases``.

    Returns:
        tuple[array, list[tuple[array,int]]]: New diagonal phases after commuting through Givens
        rotations, and concatenated list of Givens rotations that form a new valid decomposition
        of the originally decomposed matrix. The format for each rotation is like that
        for ``left_givens``.

    After the diagonalization, the phases sit at the diagonal of the square grid of Givens
    rotations. Instead, we want them to be moved to the front of the decomposition. For
    this, the condition `T_{m,n}^{-1} x D = D' x T'` is solved, see the Optica reference
    for details.

    After pulling the phases out, the Givens rotations are reordered so that they do not
    diagonalize, but reproduce, the original unitary.
    """
    nleft_givens = []
    for grot_mat, (i, j) in reversed(left_givens):
        # Manually compute new Givens matrix and new phase when commuting a phase through.
        abs_c, abs_s = math.abs(grot_mat[:, 0])
        first_col = phases[j, j] * math.conj(phases[i, i]) * grot_mat[1, 1] * grot_mat[0, 1]
        givens_mat = math.array(
            [[first_col / abs_s, -abs_s], [first_col / abs_c, abs_c]], like=interface
        )
        nphase_diag = [
            -math.conj(grot_mat[1, 0]) / abs_s * phases[j, j],
            grot_mat[1, 1] / abs_c * phases[j, j],
        ]
        for diag_idx, diag_val in zip([(i, i), (j, j)], nphase_diag, strict=True):
            phases = _set_unitary_matrix(phases, diag_idx, diag_val, like=interface)

        nleft_givens.append((math.conj(givens_mat), (i, j)))

    ordered_rotations = nleft_givens[::-1] + right_givens[::-1]

    return math.diag(phases), ordered_rotations


@lru_cache
def _givens_matrix_jax():

    @partial(jax.jit, static_argnames=("real_valued",))
    def givens_matrix_jax(a, b, left=True, tol=1e-8, real_valued=False):
        return _givens_matrix_core(a, b, left=left, tol=tol, real_valued=real_valued)

    return givens_matrix_jax


def _givens_matrix(a, b, left=True, tol=1e-8, real_valued=False):
    interface = math.get_interface(a)
    if interface == "jax" and isinstance(jnp.array(0), jax.core.Tracer):
        return _givens_matrix_jax()(a, b, left=left, tol=tol, real_valued=real_valued)
    return _givens_matrix_core(a, b, left=left, tol=tol, real_valued=real_valued)


def _right_givens_core(indices, unitary, N, j, real_valued):
    interface = math.get_interface(unitary)
    grot_mat = _givens_matrix(*unitary[N - j - 1, indices].T, left=True, real_valued=real_valued)
    unitary = _set_unitary_matrix(
        unitary, (Ellipsis, indices), unitary[:, indices] @ grot_mat.T, like=interface
    )
    return unitary, math.conj(grot_mat)


@lru_cache
def _right_givens_jax():

    @partial(jax.jit, static_argnames=("real_valued",))
    def _right_givens_jax(indices, unitary, N, j, real_valued):
        return _right_givens_core(indices, unitary, N, j, real_valued)

    return _right_givens_jax


def _right_givens(indices, unitary, N, j, real_valued):
    interface = math.get_interface(unitary)
    if interface == "jax" and isinstance(jnp.array(0), jax.core.Tracer):
        return _right_givens_jax()(indices, unitary, N, j, real_valued)
    return _right_givens_core(indices, unitary, N, j, real_valued)


def _left_givens_core(indices, unitary, j, real_valued):
    interface = math.get_interface(unitary)
    grot_mat = _givens_matrix(*unitary[indices, j - 1], left=False, real_valued=real_valued)
    unitary = _set_unitary_matrix(
        unitary, (indices, Ellipsis), grot_mat @ unitary[indices, :], like=interface
    )
    return unitary, grot_mat


@lru_cache
def _left_givens_jax():

    @partial(jax.jit, static_argnames=("real_valued",))
    def _left_givens_jax(indices, unitary, j, real_valued):
        return _left_givens_core(indices, unitary, j, real_valued)

    return _left_givens_jax


def _left_givens(indices, unitary, j, real_valued):
    interface = math.get_interface(unitary)
    if interface == "jax" and isinstance(jnp.array(0), jax.core.Tracer):
        return _left_givens_jax()(indices, unitary, j, real_valued)
    return _left_givens_core(indices, unitary, j, real_valued)


def givens_decomposition(unitary):
    r"""Decompose a unitary into a sequence of Givens rotation gates with phase shifts and a diagonal phase matrix.

    Args:
        unitary (tensor): unitary matrix on which decomposition will be performed

    Returns:
        (tensor_like, list[(tensor_like, tuple)]): diagonal elements of the phase matrix :math:`D` and Givens rotation matrix :math:`T` with their indices

    Raises:
        ValueError: if the provided matrix is not square.

    This decomposition is based on the construction scheme given in `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ ,
    which allows one to write any :math:`N\times N` unitary matrix :math:`U` as:

    .. math::

        U = D \left(\prod_{m \in G} T_m(\theta, \phi)\right),

    where :math:`D` is a diagonal phase matrix, :math:`T_m(\theta, \phi)` is the Givens rotation
    (with phase shift) between matrix indices :math:`m` and :math:`m+1` (zero-based indexing),
    and :math:`G` defines the ordered sequence of the Givens rotations. The explicit form of the
    Givens rotation with phase shift reads:

    .. math:: T(\theta, \phi) = \begin{bmatrix}
                                    \mathbb{I}_{m} & 0 & 0 & 0 \\
                                    0 & e^{i \phi} \cos(\theta) & -\sin(\theta) & 0 \\
                                    0 & e^{i \phi} \sin(\theta) & \cos(\theta) & 0 \\
                                    0 & 0 & 0 & \mathbb{I}_{N-m-2}
                                \end{bmatrix},

    where :math:`\theta \in [0, \pi/2]` is the angle of rotation
    and :math:`\phi \in [0, 2 \pi]` represents the phase shift.

    For real-valued matrices with unit determinant, i.e. special orthogonal :math:`U`,
    all phase angles :math:`\phi` vanish and :math:`D` can be fixed to the identity,
    absorbing its phases :math:`\pm 1` (with an even number of :math:`-1`\ s due to the
    determinant constraint) into the :math:`T` matrices.
    Whether :math:`U` is orthogonal is inferred from the data type of ``unitary``.
    If the determinant of an orthogonal :math:`U` is negative, this will show as a single
    negative phase in the first output value of ``givens_decomposition``.

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

    .. details::
        :title: Theory and Pseudocode

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
                        for j in range(i):
                            # Find T(N+j-i-1, N+j-i) matrix that nulls element (N+j-i, j) of U
                            # Update U = T(N+j-i-1, N+j-i) @ U

                if real_data_type:
                    # Absorb the diagonal phases, which can only be 1s and an even number of
                    # -1s, in the T gates. Adjust ordering and/or matrices of left-applied
                    # and right-applied T matrices so that they make up U instead of U^{-1}.
                else:
                    # Commute the diagonal phases from between the left-applied and right-applied
                    # T matrices to the left. Adjust ordering and/or matrices of left-applied
                    # and right-applied T matrices so that they make up U instead of U^{-1}.

    """
    interface = math.get_deep_interface(unitary)
    is_real = math.is_real_obj_or_close(unitary)
    unitary_mat = math.copy(unitary) if interface == "jax" else math.toarray(unitary).copy()
    converted_dtype = False
    if math.get_dtype_name(unitary).startswith("complex") and is_real:
        converted_dtype = True
        unitary_mat = math.real(unitary_mat)

    shape = math.shape(unitary_mat)

    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"The unitary matrix should be of shape NxN, got {shape}")

    N = shape[0]

    left_givens, right_givens = [], []
    for i in range(1, N):
        if i % 2:
            for j in range(i):
                indices = [i - j - 1, i - j]
                unitary_mat, grot_mat_conj = _right_givens(indices, unitary_mat, N, j, is_real)
                right_givens.append((grot_mat_conj, indices))
        else:
            for j in range(1, i + 1):
                indices = [N + j - i - 2, N + j - i - 1]
                unitary_mat, grot_mat = _left_givens(indices, unitary_mat, j, is_real)
                left_givens.append((grot_mat, indices))
    unitary_mat, all_givens = (_absorb_phases_so if is_real else _commute_phases_u)(
        left_givens, right_givens, unitary_mat, interface
    )
    if converted_dtype:
        unitary_mat = math.convert_like(unitary_mat, unitary)
        all_givens = [(math.convert_like(mat, unitary), indices) for mat, indices in all_givens]
    return unitary_mat, all_givens
