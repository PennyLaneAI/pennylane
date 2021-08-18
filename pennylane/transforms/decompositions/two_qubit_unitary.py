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
"""Contains transforms and helpers functions for decomposing arbitrary two-qubit
unitary operations into elementary gates.
"""
import pennylane as qml
from pennylane import math

from .single_qubit_unitary import zyz_decomposition


def _convert_to_su4(U):
    r"""Check unitarity of a 4x4 matrix and convert it to :math:`SU(4)` if possible.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`4 \times 4` and unitary.

    Returns:
        array[complex]: A :math:`4 \times 4` matrix in :math:`SU(4)` that is
        equivalent to U up to a global phase.
    """
    # Check unitarity
    if not math.allclose(math.dot(U, math.T(math.conj(U))), math.eye(2), atol=1e-7):
        raise ValueError("Operator must be unitary.")

    # Compute the determinant
    det = math.linalg.det(U)

    # Convert to SU(4) if it's not close to 1
    if not math.allclose(det, 1.0):
        exp_angle = -1j * math.cast_like(math.angle(det), 1j) / 4
        U = math.cast_like(U, exp_angle) * math.exp(exp_angle)

    return U


def _select_rotation_angles(U):
    r"""Choose the rotation angles of RZ, RY in the two-qubit decomposition.
    They are chosen as per Proposition V.1 in quant-ph/0308033 and are based
    on the phases of the eigenvalues of U.
    """

    # Choose any three eigenvalues of U, e^ix, e^iy, e^iz.
    evs = qml.math.linalg.eigvals(U)
    x, y, z = qml.math.angle(evs[0]), qml.math.angle(evs[1]), qml.math.angles(evs[2])

    # Then the rotation angles can be computed as follows
    alpha = (x + y) / 2
    beta = (x + z) / 2
    delta = (y + z) / 2

    return alpha, beta, delta


def two_qubit_decomposition(U, wires):
    r"""Recover the decomposition of a two-qubit matrix :math:`U` in terms of
    elementary operations.

    The work of https://arxiv.org/abs/quant-ph/0308033 presents a fixed-form
    decomposition of U in terms of single-qubit gates and CNOTs. Multiple such
    decompositions are possible (by choosing two of {``RX``, ``RY``, ``RZ``}),
    here we choose the ``RY``, ``RZ`` case (fig. 2 in the above) to match with
    the default decomposition of the single-qubit ``Rot`` operations as ``RZ RY
    RZ``. The form of the decomposition is:

     0: -C--X--RZ(d)--C---------X--A-|
     1: -D--C--RY(b)--X--RY(a)--C--B-|

    where A, B, C, D are SU(2) gates.

    Args:
        U (tensor): A 4 x 4 unitary matrix.
        wires (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.

    Returns:
        list[qml.Operation]: A list of operations that represent the decomposition
        of the matrix U.
    """

    # First, we note that this method works only for SU(4) gates, meaning that
    # we need to compute rescale the matrix by its determinant.
    U = _convert_to_su4(U)

    # Next, we can choose the angles of the RZ / RY rotations.
    # See documentation within the function used below.
    alpha, beta, delta = _select_rotation_angles(U)

    # This gives us the full interior portion of the decomposition
    interior_decomp = [
        qml.CNOT(wires=[wires[1], wires[0]]),
        qml.RZ(delta, wires=wires[0]),
        qml.RY(beta, wires=wires[1]),
        qml.CNOT(wires=[wires[0], wires[1]]),
        qml.RY(alpha, wires=wires[1]),
        qml.CNOT(wires=[wires[1], wires[0]]),
    ]

    # Now we need to find the four SU(2) operations A, B, C, D
    # TODO

    # Since we have only their qubit unitary form, we need to
    # decompose them as well.
    A_ops = zyz_decomposition(A, wires[0])
    B_ops = zyz_decomposition(B, wires[1])
    C_ops = zyz_decomposition(C, wires[0])
    D_ops = zyz_decomposition(A, wires[1])

    # Return the full decomposition
    return C_ops + D_ops + interior_decomp + A_ops + B_ops
