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
import warnings

import pennylane as qml
from pennylane import math


def _convert_to_su2(U, return_global_phase=False):
    r"""Convert a 2x2 unitary matrix to :math:`SU(2)`. (batched operation)

    Args:
        U (array[complex]): A matrix with a batch dimension, presumed to be
        of shape :math:`n \times 2 \times 2` and unitary for any positive integer n.
        return_global_phase (bool): If `True`, the return will include
        the global phase. If `False`, only the :math:`SU(2)` representative
        is returned.

    Returns:
        array[complex]: A :math:`n \times 2 \times 2` matrix in :math:`SU(2)` that is
        equivalent to U up to a global phase. If ``return_global_phase=True``,
        a 2-element tuple is returned, with the first element being the
        :math:`SU(2)` equivalent and the second, the global phase.
    """
    # Compute the determinants
    dets = math.linalg.det(U)

    exp_angles = math.cast_like(math.angle(dets), 1j) / 2
    U_SU2 = math.cast_like(U, dets) * math.exp(-1j * exp_angles)[:, None, None]
    return (U_SU2, exp_angles) if return_global_phase else U_SU2


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
    warnings.warn(
        "The ZYZ decomposition is deprecated and will be removed soon. Use :func:`one_qubit_decomposition` with"
        " the keyword rotations='ZYZ'"
    )
    return _zyz_decomposition(U=U, wire=wire)


def _zyz_decomposition(U, wire):
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


def xyx_decomposition(U, wire, return_global_phase=False):
    r"""Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of X and Y rotations in the form
    :math:`e^{i\gamma} RX(\phi) RY(\theta) RX(\lambda)`.

    Args:
        U (array[complex]): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase
            as a `qml.s_prod` between `exp(1j)*gamma` and `qml.Identity` as the last
            element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of of gates, an ``RX``, an ``RY`` and
        another ``RX`` gate, which when applied in the order of appearance in the list is equivalent
        to the unitary :math:`U` up to global phase. If `return_global_phase=True`,
        the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([[-0.28829348-0.78829734j,  0.30364367+0.45085995j],
    ...               [ 0.53396245-0.10177564j,  0.76279558-0.35024096j]])
    >>> decomp = xyx_decomposition(U, 0, return_global_phase=True)
    >>> decomp
    [RX(array(0.45246584), wires=[0]),
    RY(array(1.39749741), wires=[0]),
    RX(array(-1.72101925), wires=[0]),
    (0.38469215914523336-0.9230449299422961j)*(Identity(wires=[0]))]
    """
    warnings.warn(
        "The XYX decomposition is deprecated and will be removed soon. Use :func:`one_qubit_decomposition` with"
        "the keyword  `rotations=XYX`"
    )
    return _xyx_decomposition(U=U, wire=wire, return_global_phase=return_global_phase)


def _xyx_decomposition(U, wire, return_global_phase=False):
    r"""Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of X and Y rotations in the form
    :math:`e^{i\gamma} RX(\phi) RY(\theta) RX(\lambda)`.

    Args:
        U (array[complex]): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase
            as a `qml.s_prod` between `exp(1j)*gamma` and `qml.Identity` as the last
            element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of of gates, an ``RX``, an ``RY`` and
        another ``RX`` gate, which when applied in the order of appearance in the list is equivalent
        to the unitary :math:`U` up to global phase. If `return_global_phase=True`,
        the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([[-0.28829348-0.78829734j,  0.30364367+0.45085995j],
    ...               [ 0.53396245-0.10177564j,  0.76279558-0.35024096j]])
    >>> decomp = xyx_decomposition(U, 0, return_global_phase=True)
    >>> decomp
    [RX(array(0.45246584), wires=[0]),
    RY(array(1.39749741), wires=[0]),
    RX(array(-1.72101925), wires=[0]),
    (0.38469215914523336-0.9230449299422961j)*(Identity(wires=[0]))]
    """

    # Small number to add to denominators to avoid division by zero
    EPS = 1e-64

    # Choose gamma such that exp(-i*gamma)*U is special unitary (detU==1).
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, gamma = _convert_to_su2(U, return_global_phase=True)

    # Compute \phi, \theta and \lambda after analytically solving for them from
    # U_det1 = expm(1j*\phi*PauliX) expm(1j*\theta*PauliY) expm(1j*\lambda*PauliX)
    lam_plus_phi = math.arctan2(-math.imag(U_det1[:, 0, 1]), math.real(U_det1[:, 0, 0]) + EPS)
    lam_minus_phi = math.arctan2(math.imag(U_det1[:, 0, 0]), -math.real(U_det1[:, 0, 1]) + EPS)
    lam = lam_plus_phi + lam_minus_phi
    phi = lam_plus_phi - lam_minus_phi

    # The following conditional attempts to avoid 0 / 0 errors. Either the
    # sine is 0 or the cosine, but not both.
    if math.allclose(lam_plus_phi, 0):
        theta = 2 * math.arccos(math.real(U_det1[:, 1, 1]) / (math.cos(lam_plus_phi) + EPS))
    else:
        theta = 2 * math.arccos(-math.imag(U_det1[:, 0, 1]) / (math.sin(lam_plus_phi) + EPS))

    phi, theta, lam = map(math.squeeze, [phi, theta, lam])

    Operations = [qml.RX(phi, wire), qml.RY(theta, wire), qml.RX(lam, wire)]
    if return_global_phase:
        Operations += [qml.s_prod(math.exp(1j * gamma)[0], qml.Identity(wire))]

    return Operations
