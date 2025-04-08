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
from scipy import sparse

from pennylane import math
from pennylane.typing import TensorLike

EPS = 1e-64


def zyz_rotation_angles(U: TensorLike, return_global_phase=False):
    """Decomposes a 2x2 unitary into ZYZ rotation angles and optionally a global phase.

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    """

    if sparse.issparse(U):
        U = U.todense()  # U is assumed to be 2x2 here so dense representation is fine.

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)

    abs_b = math.clip(math.abs(U[..., 0, 1]), 0, 1)
    theta = 2 * math.arcsin(abs_b)

    half_phi_plus_omega = math.angle(U[..., 1, 1] + EPS)
    half_omega_minus_phi = math.angle(U[..., 1, 0] + EPS)

    phi = half_phi_plus_omega - half_omega_minus_phi
    omega = half_phi_plus_omega + half_omega_minus_phi

    # Normalize the angles
    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    omega = math.squeeze(omega % (4 * np.pi))

    if return_global_phase:
        return phi, theta, omega, global_phase

    return phi, theta, omega


def xyx_rotation_angles(U: TensorLike, return_global_phase=False):
    """Decomposes a 2x2 unitary into XYX rotation angles and optionally a global phase.

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    """

    if sparse.issparse(U):
        U = U.todense()  # U is assumed to be 2x2 here so dense representation is fine.

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)

    half_lam_plus_phi = math.arctan2(-math.imag(U[..., 0, 1]), math.real(U[..., 0, 0]) + EPS)
    half_lam_minus_phi = math.arctan2(math.imag(U[..., 0, 0]), -math.real(U[..., 0, 1]) + EPS)
    lam = half_lam_plus_phi + half_lam_minus_phi
    phi = half_lam_plus_phi - half_lam_minus_phi

    theta = math.where(
        math.isclose(math.sin(half_lam_plus_phi), math.zeros_like(half_lam_plus_phi)),
        2 * math.arccos(math.real(U[..., 1, 1]) / (math.cos(half_lam_plus_phi) + EPS)),
        2 * math.arccos(-math.imag(U[..., 0, 1]) / (math.sin(half_lam_plus_phi) + EPS)),
    )

    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    lam = math.squeeze(lam % (4 * np.pi))

    if return_global_phase:
        return lam, theta, phi, global_phase

    return lam, theta, phi


def xzx_rotation_angles(U: TensorLike, return_global_phase=False):
    """Decomposes a 2x2 unitary into XZX rotation angles and optionally a global phase.

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    """

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)

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
        2 * math.arccos(sum_diagonal_real / (2 * math.cos(half_phi_plus_lambdas) + EPS)),
        2 * math.arccos(-sum_off_diagonal_imag / (2 * math.sin(half_phi_plus_lambdas) + EPS)),
    )

    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    lam = math.squeeze(lam % (4 * np.pi))

    if return_global_phase:
        return lam, theta, phi, global_phase

    return lam, theta, phi


def zxz_rotation_angles(U: TensorLike, return_global_phase=False):
    """Decomposes a 2x2 unitary into ZXZ rotation angles and optionally a global phase."""

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)

    abs_a = math.clip(math.abs(U[..., 0, 0]), 0, 1)
    abs_b = math.clip(math.abs(U[..., 0, 1]), 0, 1)
    theta = math.where(abs_a > abs_b, 2 * math.arccos(abs_a), 2 * math.arcsin(abs_b))

    half_phi_plus_omega = math.angle(U[..., 1, 1] + EPS)
    half_phi_minus_omega = math.angle(1j * U[..., 1, 0] + EPS)

    phi = half_phi_plus_omega + half_phi_minus_omega
    omega = half_phi_plus_omega - half_phi_minus_omega

    # Normalize the angles
    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    omega = math.squeeze(omega % (4 * np.pi))

    if return_global_phase:
        return omega, theta, phi, global_phase

    return omega, theta, phi
