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

import pennylane as qml
from pennylane import math


def _convert_to_su2(U):
    r"""Convert a 2x2 unitary matrix to :math:`SU(2)`.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`2 \times 2` and unitary.

    Returns:
        array[complex]: A :math:`2 \times 2` matrix in :math:`SU(2)` that is
        equivalent to U up to a global phase.
    """
    # Compute the determinants
    dets = math.linalg.det(U)

    exp_angles = -1j * math.cast_like(math.angle(dets), 1j) / 2
    return math.cast_like(U, dets) * math.exp(exp_angles)[:, None, None]


def zyz_decomposition(U, wire):
    r"""Recover the decomposition of a single-qubit matrix :math:`U` in terms of
    elementary operations.

    Diagonal operations can be converted to a single :class:`.RZ` gate, while non-diagonal
    operations will be converted to a :class:`.Rot` gate that implements the original operation
    up to a global phase in the form :math:`RZ(\omega) RY(\theta) RZ(\phi)`.

    .. warning::

        When used with ``jax.jit``, all unitaries will be converted to :class:`.Rot` gates,
        including those that are diagonal.

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

    # Cast to batched format for more consistent code
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U

    U = _convert_to_su2(U)

    # If U is only one unitary and its value is not abstract, we can include a conditional
    # statement that will check if the off-diagonal elements are 0; if so, just use one RZ
    if len(U) == 1 and not math.is_abstract(U[0]):
        if math.allclose(U[0, 0, 1], 0.0):
            return [qml.RZ(2 * math.angle(U[0, 1, 1]), wires=wire)]

    # For batched U or single U with non-zero off-diagonal, compute the
    # Rot operator decomposition instead
    off_diagonal_elements = math.clip(math.abs(U[:, 0, 1]), 0, 1)
    thetas = 2 * math.arcsin(off_diagonal_elements)

    # Compute phi and omega from the angles of the top row; use atan2 to keep
    # the angle within -np.pi and np.pi, and add very small value to the real
    # part to avoid division by zero.
    epsilon = 1e-64
    angles_U00 = math.arctan2(
        math.imag(U[:, 0, 0]),
        math.real(U[:, 0, 0]) + epsilon,
    )
    angles_U10 = math.arctan2(
        math.imag(U[:, 1, 0]),
        math.real(U[:, 1, 0]) + epsilon,
    )

    phis = -angles_U10 - angles_U00
    omegas = angles_U10 - angles_U00

    phis, thetas, omegas = map(math.squeeze, [phis, thetas, omegas])

    return [qml.Rot(phis, thetas, omegas, wires=wire)]

def xyx_decomposition(U, wire):
    r"""Recover the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of X and Y rotations in the form
    :math:`e^{i\gamma} RX(\phi) RY(\theta) RX(\lambda)`.

    Args:
        U (tensor): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.

    Returns:
        list[qml.Operation]: Returns a list of 4 gates, an ``Identity`` times 
        the global phase, then an ``RX``, an ``RY``and another ``RX`` gate,
        which when applied in the order of appearance in the list is equivalent
        to the unitary :math:`U`.

    **Example**

    >>> U = np.array([[-0.28829348-0.78829734j,  0.30364367+0.45085995j],
                      [ 0.53396245-0.10177564j,  0.76279558-0.35024096j]])
    >>> decomp = xyx_decomposition(U, 0)
    >>> decomp
    [(0.38469215969008697-0.9230449297152207j)*(Identity(wires=[0])),
        RX(-2.689126816797089, wires=[0]), 
        RY(-1.3974974117211323, wires=[0]),
        RX(1.4205734045584246, wires=[0])]
    """
    
    # Small number to add to denominators to avoid division by zero
    EPS = 1e-64
    
    # Calculate the average phase of U[0, 0] and U[1, 1]. Use it to change 
    # the global phase of U such that U[0, 0] and U[1, 1] have the same 
    # phase but with opposite signs. Store this form with the name "U2".
    angle_U00 = math.arctan2(math.imag(U[0, 0]),
                             math.real(U[0, 0]) + EPS)
    angle_U11 = math.arctan2(math.imag(U[1, 1]),
                             math.real(U[1, 1]) + EPS)
    global_phase = (angle_U00 + angle_U11)/2
    U2 = math.exp(-1j*global_phase) * U
    
    # Compute \phi, \theta and \lambda after analytically solving for them from
    # U2 = expm(1j*\phi*PauliX) expm(1j*\theta*PauliY) expm(1j*\lambda*PauliX)
    lam_plus_phi  = math.arctan2(math.imag(U2[0, 1]),
                                 math.real(U2[0, 0]) + EPS);
    lam_minus_phi = math.arctan2(math.imag(U2[0, 0]),
                                 math.real(U2[0, 1]) + EPS);
    lam = (lam_plus_phi + lam_minus_phi)/2;
    phi = (lam_plus_phi - lam_minus_phi)/2;
    theta = math.arcsin(math.real(U2[0, 1]) / math.cos(lam_minus_phi));

    # Note: angles above are changed to half angles with opposite sign
    # to be consistent with how Pauli rotations are implemented in Pennylane
    return [qml.s_prod(math.exp(1j*global_phase),
                             qml.Identity(wire)),
                    qml.RX(-2*phi, wire),
                    qml.RY(-2*theta, wire),
                    qml.RX(-2*lam, wire)]
    