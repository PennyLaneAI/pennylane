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

from functools import partial
import pennylane as qml
from pennylane import math

# Conditionally import jax; check for special case with jax.lax.cond so that we can JIT
try:
    from jax.lax import cond
except (ModuleNotFoundError, ImportError) as e:
    pass


def _convert_to_su2(U):
    r"""Convert a 2x2 unitary matrix to :math:`SU(2)`.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`2 \times 2` and unitary.

    Returns:
        array[complex]: A :math:`2 \times 2` matrix in :math:`SU(2)` that is
        equivalent to U up to a global phase.
    """
    # Compute the determinant
    det = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]

    exp_angle = -1j * math.cast_like(math.angle(det), 1j) / 2
    return U * math.exp(exp_angle)


def _compute_rot_angles(U):
    # Derive theta from the off-diagonal element. We very very carefully rescale
    # the value to make sure it's between -1 and 1 by subtracting a small number
    # in the cases where we have a positive-valued element.
    element = math.abs(U[0, 1])
    theta = 2 * math.arcsin(element - math.sign(element) * 1e-16)

    # Compute phi and omega from the angles of the top row; use atan2 to keep
    # the angle within -np.pi and np.pi, and add very small values to avoid the
    # undefined case of 0/0. We add a smaller value to the imaginary part than
    # the real part because it is imag / real in the definition of atan2.
    angle_U00 = math.arctan2(math.imag(U[0, 0]) + 1e-128, math.real(U[0, 0]) + 1e-64)
    angle_U10 = math.arctan2(math.imag(U[1, 0]) + 1e-128, math.real(U[1, 0]) + 1e-64)

    phi = -angle_U10 - angle_U00
    omega = angle_U10 - angle_U00

    return phi, theta, omega



def zyz_decomposition(U, wire):
    r"""Recover the decomposition of a single-qubit matrix :math:`U` in terms of
    elementary operations.

    Diagonal operations can be converted to a single :class:`.RZ` gate, while non-diagonal
    operations will be converted to a :class:`.Rot` gate that implements the original operation
    up to a global phase in the form :math:`RZ(\omega) RY(\theta) RZ(\phi)`.

    .. warning:: 
        
        This decomposition is compatible with ``jax.jit``, however, all unitaries
        will be converted to :class:`.Rot` gates, including those that are diagonal.

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

    # If the value of U is abstract as is the case when we use something like
    # of jax.jit, we can't use the conditional, so simply compute the angles.
    if math.is_abstract(U):
        if math.get_interface(U) == "jax":
                
            angles = cond(
                True,
                lambda x: _compute_rot_angles(x),
                lambda x: _compute_rot_angles(x),
                U
            )
                
            return [qml.Rot(*angles, wires=wire)]            

    # If the value of U is not abstract, then things are running normally so we
    # include a conditional statement here to reduce the number of gates.
    if math.allclose(U[0, 1], 0.0):
        return [qml.RZ(2 * math.angle(U[1, 1]), wires=wire)]

    phi, theta, omega = _compute_rot_angles(U)
    return [qml.Rot(phi, theta, omega, wires=wire)]
