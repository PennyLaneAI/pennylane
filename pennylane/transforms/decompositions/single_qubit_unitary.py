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
"""Contains transforms and helpers functions for decomposing arbitrary unitary
operations into elementary gates.
"""
import numpy as np
import pennylane as qml
from pennylane import math


def _convert_to_su2(U):
    r"""Check unitarity of a matrix and convert it to :math:`SU(2)` if possible.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`2 \times 2` and unitary.

    Returns:
        array[complex]: A :math:`2 \times 2` matrix in :math:`SU(2)` that is
        equivalent to U up to a global phase.
    """
    # Check unitarity
    if not math.allclose(math.dot(U, math.T(math.conj(U))), math.eye(2), atol=1e-7):
        raise ValueError("Operator must be unitary.")

    # Compute the determinant
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]

    # Convert to SU(2) if it's not close to 1
    if not math.allclose(det, [1.0]):
        exp_angle = -1j * math.cast_like(math.angle(det), 1j) / 2
        U = math.cast_like(U, exp_angle) * math.exp(exp_angle)

    return U


def zyz_decomposition(U, wire):
    r"""Recover the decomposition of a single-qubit matrix :math:`U` in terms of
    elementary operations.

    Diagonal operations will be converted to a single :class:`.RZ` gate, while non-diagonal
    operations will be converted to a :class:`.Rot` gate that implements the original operation
    up to a global phase in the form :math:`RZ(\omega) RY(\theta) RZ(\phi)`.

    Args:
        U (tensor): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.

    Returns:
        list[qml.Operation]: A ``Rot`` gate on the specified wire that implements ``U``
        up to a global phase, or an equivalent ``RZ`` gate if ``U`` is diagonal.

    **Example**

    Suppose we would like to apply the following unitary operation:

    .. code-block:: python3

        U = np.array([
            [-0.28829348-0.78829734j,  0.30364367+0.45085995j],
            [ 0.53396245-0.10177564j,  0.76279558-0.35024096j]
        ])

    For PennyLane devices that cannot natively implement ``QubitUnitary``, we
    can instead recover a ``Rot`` gate that implements the same operation, up
    to a global phase:

    >>> decomp = zyz_decomposition(U, 0)
    >>> decomp
    [Rot(-0.24209529417800013, 1.14938178234275, 1.7330581433950871, wires=[0])]
    """
    U = _convert_to_su2(U)

    # Check if the matrix is diagonal; only need to check one corner.
    # If it is diagonal, we don't need a full Rot, just return an RZ.
    if math.allclose(U[0, 1], [0.0]):
        omega = 2 * math.angle(U[1, 1])
        return [qml.RZ(omega, wires=wire)]

    # If the top left element is 0, can only use the off-diagonal elements. We
    # have to be very careful with the math here to ensure things that get
    # multiplied together are of the correct type in the different interfaces.
    if math.allclose(U[0, 0], [0.0]):
        phi = 0.0
        theta = -np.pi
        omega = 1j * math.log(U[0, 1] / U[1, 0]) - np.pi
    else:
        # If not diagonal, compute the angle of the RY
        cos2_theta_over_2 = math.abs(U[0, 0] * U[1, 1])
        theta = 2 * math.arccos(math.sqrt(cos2_theta_over_2))

        el_division = U[0, 0] / U[1, 0]
        tan_part = math.cast_like(math.tan(theta / 2), el_division)
        omega = 1j * math.log(tan_part * el_division)
        phi = -omega - math.cast_like(2 * math.angle(U[0, 0]), omega)

    return [qml.Rot(math.real(phi), math.real(theta), math.real(omega), wires=wire)]
