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

"""This module contains methods that decomposes unitary matrices."""

import numpy as np

from pennylane import math


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
    theta = math.where(abs_a > abs_b, 2 * math.arccos(abs_a), 2 * math.arcsin(abs_b))

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
