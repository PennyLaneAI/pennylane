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
"""
Contains transforms for decomposing arbitrary unitary operations into elementary gates.
"""
import pennylane as qml
import numpy as np
from pennylane.transforms import qfunc_transform


def _convert_to_su2(U, zero_tol=1e-6):
    r"""Check unitarity of a matrix and convert it to :math:`SU(2)` if possible.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`2 \times 2` and unitary.

    Returns:
        array[complex]: A :math:`2 \times 2` matrix in :math:`SU(2)` that is
        equivalent to U up to a global phase.
    """
    # Check dimensions
    if qml.math.shape(U)[0] != 2 or qml.math.shape(U)[1] != 2:
        raise ValueError("Cannot convert matrix with shape {qml.math.shape(U)} to SU(2).")

    # Check unitarity
    if not qml.math.allclose(
        qml.math.dot(U, qml.math.T(qml.math.conj(U))), qml.math.eye(2), atol=zero_tol
    ):
        raise ValueError("Operator must be unitary.")

    # Compute the determinant
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]

    # Convert to SU(2) if it's not close to 1
    if not qml.math.allclose(det, [1.0], atol=zero_tol):
        exp_angle = -1j * qml.math.cast_like(qml.math.angle(det), 1j) / 2
        U = U * (qml.math.exp(exp_angle))

    return U


def _zyz_decomposition(U, wire, zero_tol=1e-6):
    r"""Helper function to recover the rotation angles of a single-qubit matrix :math:`U`.

    The set of angles are chosen so as to implement :math:`U` up to a global phase.

    Args:
        U (tensor): A 2 x 2 unitary matrix.
        tol (float): The tolerance at which an angle is considered to be close enough
            to 0. Needed to deal with varying precision across interfaces.

    Returns:
        (float, float, float): A set of angles (:math:`\phi`, :math:`\theta`, :math:`\omega`)
        that implement U as a sequence :math:`U = RZ(\omega) RY(\theta) RZ(\phi)`.
    """
    U = _convert_to_su2(U, zero_tol)

    # Compute the angle of the RY
    cos2_theta_over_2 = qml.math.abs(U[0, 0] * U[1, 1])
    theta = 2 * qml.math.arccos(qml.math.sqrt(cos2_theta_over_2))

    # If it's close to 0, the matrix is diagonal and we have just an RZ rotation
    if qml.math.allclose(theta, [0.0], atol=zero_tol):
        omega = 2 * qml.math.angle(U[1, 1])
        return [qml.RZ(omega, wires=wire)]

    # Otherwise recover the decomposition as a Rot, which can be further decomposed
    # if desired. If the top left element is 0, can only use the off-diagonal elements
    # We have to be very careful with the math here to ensure things that get multiplied
    # together are of the correct type.
    if qml.math.allclose(U[0, 0], [0.0], atol=zero_tol):
        phi = 1j * qml.math.log(U[0, 1] / U[1, 0])
        omega = -phi - qml.math.cast_like(2 * qml.math.angle(U[1, 0]), phi)
    else:
        el_division = U[0, 0] / U[1, 0]
        tan_part = qml.math.cast_like(qml.math.tan(theta / 2), el_division)
        omega = 1j * qml.math.log(tan_part * el_division)
        phi = -omega - qml.math.cast_like(2 * qml.math.angle(U[0, 0]), omega)

    return [qml.Rot(qml.math.real(phi), qml.math.real(theta), qml.math.real(omega), wires=wire)]


@qfunc_transform
def decompose_single_qubit_unitaries(tape):
    """Quantum function transform to decomposes all instances of single-qubit QubitUnitary
    operations to a sequence of rotations of the form ``RZ``, ``RY``, ``RZ``.

    Args:
        tape (qml.tape.QuantumTape): A quantum tape.
    """
    for op in tape.operations + tape.measurements:
        if isinstance(op, qml.QubitUnitary):
            dim_U = qml.math.shape(op.parameters[0])[0]

            if dim_U != 2:
                continue

            decomp = _zyz_decomposition(op.parameters[0], op.wire)

            for d_op in decomp:
                d_op.queue()
        else:
            op.queue()
