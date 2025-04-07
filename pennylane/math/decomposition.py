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


def zyz_rotation_angles(U: TensorLike, return_global_phase=False):
    """Decomposes a 2x2 unitary into ZYZ rotation angles and optionally a global phase.

    Args:
        U (array): 2x2 unitary matrix
        return_global_phase (bool): if True, returns the global phase as well.

    """

    if sparse.issparse(U):
        U = U.todense()  # U is assumed to be 2x2 here so dense representation is fine.

    U, global_phase = math.convert_to_su2(U, return_global_phase=True)

    a = U[..., 0, 0]
    b = U[..., 0, 1]

    abs_a = math.clip(math.abs(a), 0, 1)
    abs_b = math.clip(math.abs(b), 0, 1)
    theta = math.where(abs_a > abs_b, 2 * math.arccos(abs_a), 2 * math.arcsin(abs_b))

    phi_plus_omega = 2 * math.angle(U[..., 1, 1])
    omega_minus_phi = 2 * math.angle(U[..., 1, 0])

    phi = (phi_plus_omega - omega_minus_phi) / 2
    omega = (phi_plus_omega + omega_minus_phi) / 2

    # Normalize the angles
    phi = math.squeeze(phi % (4 * np.pi))
    theta = math.squeeze(theta % (4 * np.pi))
    omega = math.squeeze(omega % (4 * np.pi))

    if return_global_phase:
        return phi, theta, omega, global_phase

    return phi, theta, omega
