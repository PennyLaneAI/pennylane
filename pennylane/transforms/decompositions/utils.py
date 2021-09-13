# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions required for decomposing two-qubit operations."""

import pennylane.math as math
from pennylane import numpy as np

# This gate E is called the "magic basis". It can be used to convert between
# SO(4) and SU(2) x SU(2). For A in SO(4), E A E^\dag is in SU(2) x SU(2).
E = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
Edag = E.conj().T


def _convert_to_su4(U):
    r"""Check unitarity of a 4x4 matrix and convert it to :math:`SU(4)` if the determinant is not 1.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`4 \times 4` and unitary.

    Returns:
        array[complex]: A :math:`4 \times 4` matrix in :math:`SU(4)` that is
        equivalent to U up to a global phase.
    """
    # Check unitarity
    if not math.allclose(math.dot(U, math.T(math.conj(U))), math.eye(4), atol=1e-7):
        raise ValueError("Operator must be unitary.")

    # Compute the determinant
    det = math.linalg.det(U)

    # Convert to SU(4) if it's not close to 1
    if not math.allclose(det, 1.0):
        exp_angle = -1j * math.cast_like(math.angle(det), 1j) / 4
        U = math.cast_like(U, det) * math.exp(exp_angle)

    return U


def _compute_num_cnots(U):
    r"""Compute the number of CNOTs required to implement a U in SU(4). This is based on
    the trace of

    .. math::

        \gamma(U) = (E^\dag U E) (E^\dag U E)^T,

    and follows the arguments of this paper: https://arxiv.org/abs/quant-ph/0308045.
    """
    u = math.dot(Edag, math.dot(U, E))
    gammaU = math.dot(u, math.T(u))

    trace = math.trace(gammaU)

    # For the case with 3 CNOTs, the trace is a non-zero complex number
    # with both real and imaginary parts.
    num_cnots = 3

    # Case: 0 CNOTs (tensor product), the trace is +/- 4
    # We need a tolerance of around 1e-7 here in order to work with the case where U
    # is specified with 8 decimal places.
    if math.allclose(trace, 4, atol=1e-7) or math.allclose(trace, -4, atol=1e-7):
        num_cnots = 0
    # Case: 1 CNOT, the trace is 0
    elif math.allclose(trace, 0.0, atol=1e-7):
        num_cnots = 1
    # Case: 2 CNOTs, the trace has only a real part
    elif math.allclose(math.imag(trace), 0, atol=1e-7):
        num_cnots = 2

    return num_cnots


def _su2su2_to_tensor_products(U):
    r"""Given a matrix :math:`U = A \otimes B` in SU(2) x SU(2), extract the two SU(2)
    operations A and B.

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

    if not math.allclose(a1 * np.conj(a2), C12[0, 0]):
        a2 *= -1

    # Construct A
    A = math.stack([[a1, a2], [-math.conj(a2), math.conj(a1)]])

    # Next, extract B. Can do from any of the C, just need to be careful in
    # case one of the elements of A is 0.
    if not math.allclose(A[0, 0], 0.0, atol=1e-6):
        B = C1 / math.cast_like(A[0, 0], 1j)
    else:
        B = C2 / math.cast_like(A[0, 1], 1j)

    return math.convert_like(A, U), math.convert_like(B, U)
