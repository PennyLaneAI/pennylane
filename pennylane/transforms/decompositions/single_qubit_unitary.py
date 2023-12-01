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


def _convert_to_su2(U, return_global_phase=False):
    r"""Convert a 2x2 unitary matrix to :math:`SU(2)`. (batched operation)

    Args:
        U (array[complex]): A matrix with a batch dimension, presumed to be
            of shape :math:`n \times 2 \times 2` and unitary for any positive integer n.
        return_global_phase (bool): If `True`, the return will include the global phase.
            If `False`, only the :math:`SU(2)` representation is returned.

    Returns:
        array[complex]: A :math:`n \times 2 \times 2` matrix in :math:`SU(2)` that is
            equivalent to U up to a global phase. If ``return_global_phase=True``, a
            2-element tuple is returned, with the first element being the :math:`SU(2)`
            equivalent and the second, the global phase.
    """

    # Compute the determinants
    U = qml.math.cast(U, "complex128")
    determinants = math.linalg.det(U)
    phase = math.angle(determinants) / 2
    U = math.cast_like(U, determinants) * math.exp(-1j * math.cast_like(phase, 1j))[:, None, None]

    return (U, phase) if return_global_phase else U


def _zyz_get_rotation_angles(U):
    r"""Computes the rotation angles :math:`\phi`, :math:`\theta`, :math:`\omega`
    for a unitary :math:`U` that is :math:`SU(2)`

    Args:
        U (array[complex]): A matrix that is :math:`SU(2)`

    Returns:
        tuple[array[float]]: A tuple containing the rotation angles
            :math:`\phi`, :math:`\theta`, :math:`\omega`

    """

    # For batched U or single U with non-zero off-diagonal, compute the
    # normal decomposition instead
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

    # Normalize the angles
    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    omegas = omegas % (4 * np.pi)

    return phis, thetas, omegas


def _rot_decomposition(U, wire, return_global_phase=False):
    r"""Compute the decomposition of a single-qubit matrix :math:`U` in terms of
    elementary operations, as a single :class:`.RZ` gate or a :class:`.Rot` gate.

    Diagonal operations can be converted to a single :class:`.RZ` gate, while non-diagonal
    operations will be converted to a :class:`.Rot` gate that implements the original operation
    up to a global phase in the form :math:`RZ(\omega) RY(\theta) RZ(\phi)`.

    .. warning::

        When used with ``jax.jit``, all unitaries will be converted to :class:`.Rot` gates,
        including those that are diagonal.

    Args:
        U (tensor): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-alpha)``
            as the last element of the returned list of operations.

    Returns:
        list[qml.Operation]: A ``Rot`` gate on the specified wire that implements ``U``
            up to a global phase, or an equivalent ``RZ`` gate if ``U`` is diagonal. If
            `return_global_phase=True`, the global phase is included as the last element.

    **Example**

    Suppose we would like to apply the following unitary operation:

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])

    For PennyLane devices that cannot natively implement ``QubitUnitary``, we
    can instead recover a ``Rot`` gate that implements the same operation, up
    to a global phase:

    >>> decompositions = _rot_decomposition(U, 0)
    >>> decompositions
    [Rot(12.32427531154459, 1.1493817771511354, 1.733058145303424, wires=[0])]
    """

    # Cast to batched format for more consistent code
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U

    # Convert to SU(2) format and extract global phase
    U_det1, alphas = _convert_to_su2(U, return_global_phase=True)

    # If U is only one unitary and its value is not abstract, we can include a conditional
    # statement that will check if the off-diagonal elements are 0; if so, just use one RZ
    if len(U_det1) == 1 and not math.is_abstract(U_det1[0]):
        if math.allclose(U_det1[0, 0, 1], 0.0):
            angle = 2 * math.angle(U_det1[0, 1, 1]) % (4 * np.pi)
            operations = [qml.RZ(angle, wires=wire)]
            if return_global_phase:
                operations.append(qml.GlobalPhase(-alphas))
            return operations

    # Compute the zyz rotation angles
    phis, thetas, omegas = _zyz_get_rotation_angles(U_det1)

    operations = [qml.Rot(phis, thetas, omegas, wires=wire)]
    if return_global_phase:
        alphas = math.squeeze(alphas)
        operations.append(qml.GlobalPhase(-alphas))
    return operations


def _zyz_decomposition(U, wire, return_global_phase=False):
    r"""Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of Z and Y rotations in the form
    :math:`e^{i\alpha} RZ(\omega) RY(\theta) RZ(\phi)`. (batched operation)

    .. warning::

        When used with ``jax.jit``, all unitaries will be converted to :class:`.Rot` gates,
        including those that are diagonal.

    Args:
        U (tensor): A :math:`2 \times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-alpha)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RZ``, an ``RY`` and another ``RZ`` gate,
            which when applied in the order of appearance in the list is equivalent to the
            unitary :math:`U` up to a global phase. If `return_global_phase=True`, the global
            phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _zyz_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RZ(12.32427531154459, wires=[0]),
     RY(1.1493817771511354, wires=[0]),
     RZ(1.733058145303424, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]

    """

    # Cast to batched format for more consistent code
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U

    # Convert to SU(2) format and extract global phase
    U_det1, alphas = _convert_to_su2(U, return_global_phase=True)

    # Compute the zyz rotation angles
    phis, thetas, omegas = _zyz_get_rotation_angles(U_det1)

    operations = [qml.RZ(phis, wire), qml.RY(thetas, wire), qml.RZ(omegas, wire)]
    if return_global_phase:
        alphas = math.squeeze(alphas)
        operations.append(qml.GlobalPhase(-alphas))

    return operations


def _xyx_decomposition(U, wire, return_global_phase=False):
    r"""Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of X and Y rotations in the form
    :math:`e^{i\gamma} RX(\phi) RY(\theta) RX(\lambda)`.

    Args:
        U (array[complex]): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-gamma)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RX``, an ``RY`` and another ``RX`` gate,
            which when applied in the order of appearance in the list is equivalent to the unitary
            :math:`U` up to global phase. If `return_global_phase=True`, the global phase is returned
            as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _xyx_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RX(10.845351366405708, wires=[0]),
     RY(1.3974974118006183, wires=[0]),
     RX(0.45246583660683803, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]
    """

    # Small number to add to denominators to avoid division by zero
    EPS = 1e-64

    # Choose gamma such that exp(-i*gamma)*U is special unitary (detU==1).
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, gammas = _convert_to_su2(U, return_global_phase=True)

    # Compute \phi, \theta and \lambda after analytically solving for them from
    # U_det1 = expm(1j*\phi*PauliX) expm(1j*\theta*PauliY) expm(1j*\lambda*PauliX)
    lams_plus_phis = math.arctan2(-math.imag(U_det1[:, 0, 1]), math.real(U_det1[:, 0, 0]) + EPS)
    lams_minus_phis = math.arctan2(math.imag(U_det1[:, 0, 0]), -math.real(U_det1[:, 0, 1]) + EPS)
    lams = lams_plus_phis + lams_minus_phis
    phis = lams_plus_phis - lams_minus_phis

    # The following conditional attempts to avoid 0 / 0 errors. Either the
    # sine is 0 or the cosine, but not both.
    thetas = math.where(
        math.isclose(math.sin(lams_plus_phis), math.zeros_like(lams_plus_phis)),
        2 * math.arccos(math.real(U_det1[:, 1, 1]) / (math.cos(lams_plus_phis) + EPS)),
        2 * math.arccos(-math.imag(U_det1[:, 0, 1]) / (math.sin(lams_plus_phis) + EPS)),
    )

    phis, thetas, lams, gammas = map(math.squeeze, [phis, thetas, lams, gammas])

    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    lams = lams % (4 * np.pi)

    operations = [qml.RX(lams, wire), qml.RY(thetas, wire), qml.RX(phis, wire)]
    if return_global_phase:
        operations.append(qml.GlobalPhase(-gammas))

    return operations


def _xzx_decomposition(U, wire, return_global_phase=False):
    r"""Computes the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of Z and X rotations in the form
    :math:`e^{i\gamma} RX(\phi) RZ(\theta) RX(\lambda)`. (batched operation)

    Args:
        U (tensor): A :math:`2 \times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-gamma)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RX``, an ``RZ`` and
        another ``RX`` gate, which when applied in the order of appearance in the list is
        equivalent to the unitary :math:`U` up to a global phase. If `return_global_phase=True`,
        the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _xzx_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RX(12.416147693665032, wires=[0]),
     RZ(1.3974974090935608, wires=[0]),
     RX(11.448040119199066, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]

    """

    # Small number to add to denominators to avoid division by zero
    EPS = 1e-64

    # Choose gamma such that exp(-i*gamma)*U is special unitary (detU==1).
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, gammas = _convert_to_su2(U, return_global_phase=True)

    # Compute \phi, \theta and \lambda after analytically solving for them from
    # U_det1 = RX(\phi) RZ(\theta) RX(\lambda)
    sum_diagonal_real = math.real(U_det1[:, 0, 0] + U_det1[:, 1, 1])
    sum_off_diagonal_imag = math.imag(U_det1[:, 0, 1] + U_det1[:, 1, 0])
    phi_plus_lambdas_d2 = math.arctan2(-sum_off_diagonal_imag, sum_diagonal_real + EPS)
    diff_diagonal_imag = math.imag(U_det1[:, 0, 0] - U_det1[:, 1, 1])
    diff_off_diagonal_real = math.real(U_det1[:, 0, 1] - U_det1[:, 1, 0])
    phi_minus_lambdas_d2 = math.arctan2(diff_off_diagonal_real, -diff_diagonal_imag + EPS)
    lams = phi_plus_lambdas_d2 - phi_minus_lambdas_d2
    phis = phi_plus_lambdas_d2 + phi_minus_lambdas_d2

    # Compute \theta
    thetas = math.where(
        math.isclose(math.sin(phi_plus_lambdas_d2), math.zeros_like(phi_plus_lambdas_d2)),
        2 * math.arccos(sum_diagonal_real / (2 * math.cos(phi_plus_lambdas_d2) + EPS)),
        2 * math.arccos(-sum_off_diagonal_imag / (2 * math.sin(phi_plus_lambdas_d2) + EPS)),
    )

    phis, thetas, lams, gammas = map(math.squeeze, [phis, thetas, lams, gammas])

    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    lams = lams % (4 * np.pi)

    operations = [qml.RX(lams, wire), qml.RZ(thetas, wire), qml.RX(phis, wire)]
    if return_global_phase:
        operations.append(qml.GlobalPhase(-gammas))

    return operations


def _zxz_decomposition(U, wire, return_global_phase=False):
    r"""Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of X and Z rotations in the form
    :math:`e^{i\alpha} RZ(\phi) RY(\theta) RZ(\psi)`. (batched operation)

    Args:
        U (array[complex]): A :math:`2 \times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase as a
            ``qml.GlobalPhase(-alpha)`` as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RZ``, an ``RX`` and
            another ``RZ`` gate, which when applied in the order of appearance in the list is
            equivalent to the unitary :math:`U` up to a global phase. If `return_global_phase=True`,
            the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _zxz_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RZ(10.753478981934784, wires=[0]),
     RX(1.1493817777940707, wires=[0]),
     RZ(3.3038544749132295, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]

    """

    # Small number to add to denominators to avoid division by zero
    EPS = 1e-64

    # Get global phase \alpha and U in SU(2) form (determinant is 1)
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, alphas = _convert_to_su2(U, return_global_phase=True)

    # Use top row to solve for \phi and \psi
    phis_plus_psis = math.arctan2(-math.imag(U_det1[:, 0, 0]), math.real(U_det1[:, 0, 0]) + EPS)
    phis_minus_psis = math.arctan2(-math.real(U_det1[:, 0, 1]), -math.imag(U_det1[:, 0, 1]) + EPS)

    phis = phis_plus_psis + phis_minus_psis
    psis = phis_plus_psis - phis_minus_psis

    # Conditional to avoid divide by 0 errors
    thetas = math.where(
        math.isclose(math.sin(phis_plus_psis), math.zeros_like(phis_plus_psis)),
        math.real(U_det1[:, 0, 0]) / (math.cos(phis_plus_psis) + EPS),
        -math.imag(U_det1[:, 0, 0]) / (math.sin(phis_plus_psis) + EPS),
    )

    # Arcos is only defined between -1 and 1
    thetas = qml.math.clip(thetas, -1.0, 1.0)
    thetas = 2 * math.arccos(thetas)

    phis, thetas, psis, alphas = map(math.squeeze, [phis, thetas, psis, alphas])

    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    psis = psis % (4 * np.pi)

    # Return gates in the order they will be applied on the qubit
    operations = [qml.RZ(psis, wire), qml.RX(thetas, wire), qml.RZ(phis, wire)]
    if return_global_phase:
        operations.append(qml.GlobalPhase(-alphas))

    return operations


def one_qubit_decomposition(U, wire, rotations="ZYZ", return_global_phase=False):
    r"""Decompose a one-qubit unitary :math:`U` in terms of elementary operations. (batched operation)

    Any one qubit unitary operation can be implemented up to a global phase by composing RX, RY,
    and RZ gates.

    Currently supported values for ``rotations`` are "rot", "ZYZ", "XYX", "XZX", and "ZXZ".

    Args:
        U (tensor): A :math:`2 \times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        rotations (str): A string defining the sequence of rotations to decompose :math:`U` into.
        return_global_phase (bool): Whether to return the global phase as a ``qml.GlobalPhase(-alpha)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates which when applied in the order of appearance in
            the list is equivalent to the unitary :math:`U` up to a global phase. If
            ``return_global_phase=True``, the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = one_qubit_decomposition(U, 0, "ZXZ", return_global_phase=True)
    >>> decompositions
    [RZ(10.753478981934784, wires=[0]),
     RX(1.1493817777940707, wires=[0]),
     RZ(3.3038544749132295, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]
    """

    supported_rotations = {
        "rot": _rot_decomposition,
        "ZYZ": _zyz_decomposition,
        "XYX": _xyx_decomposition,
        "XZX": _xzx_decomposition,
        "ZXZ": _zxz_decomposition,
    }

    if rotations in supported_rotations:
        return supported_rotations[rotations](U, wire, return_global_phase)

    raise ValueError(
        f"Value {rotations} passed to rotations is either invalid or currently unsupported."
    )
