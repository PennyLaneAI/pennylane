# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the single_qubit_zyz_angles dispatch function."""

from functools import singledispatch

import numpy as np

from pennylane.core.operator import Operator
from pennylane.math.decomposition import zyz_rotation_angles
from pennylane.ops.op_math.adjoint import AdjointOperation
from pennylane.ops.qubit import (
    RX,
    RY,
    RZ,
    SX,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    Rot,
    S,
    T,
)
from pennylane.typing import TensorLike


@singledispatch
def single_qubit_zyz_angles(op: Operator) -> tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
    r"""Returns the equivalent ZYZ rotation angles of a single-qubit operator.

    A single-qubit unitary operator is equivalent to a product of Z and Y rotations in the form
    :math:`e^{i\alpha} RZ(\omega) RY(\theta) RZ(\phi)`.

    Args:
        op (Operator): the operator to obtain rotation angles for.

    Returns:
        A tuple of (:math:`\phi`, :math:`\theta`, :math:`\omega`, :math:`\alpha`) where the first
        three values are the rotation angles and :math:`\alpha` is the global phase.

    **Examples**

    >>> qp.single_qubit_zyz_angles(qp.H(0))
    (3.141..., 1.570..., 0.0, 1.570...)

    We can verify that this is correct:

    .. code-block:: python

        phi, theta, omega, alpha = qp.single_qubit_zyz_angles(qp.H(0))

        def circuit():
            qp.RZ(phi, wires=0)
            qp.RY(theta, wires=0)
            qp.RZ(omega, wires=0)
            qp.GlobalPhase(-alpha)  # note the negative sign convention

    >>> qp.math.allclose(qp.matrix(circuit, wire_order=[0])(), qp.H(0).matrix())
    True

    """
    if len(op.wires) != 1:
        raise ValueError(
            "qp.single_qubit_zyz_angles is not applicable to operators on more than one wire."
        )

    return zyz_rotation_angles(op.matrix(), return_global_phase=True)


@single_qubit_zyz_angles.register
def _(op: Hadamard):  # pylint: disable=unused-argument
    # H = RZ(0) RY(\pi/2) RZ(\pi)
    return (np.pi, np.pi / 2, 0.0, np.pi / 2)


@single_qubit_zyz_angles.register
def _(op: PauliX):  # pylint: disable=unused-argument
    # X = RZ(-\pi/2) RY(\pi) RZ(\pi/2)
    return (np.pi / 2, np.pi, -np.pi / 2, np.pi / 2)


@single_qubit_zyz_angles.register
def _(op: PauliY):  # pylint: disable=unused-argument
    # Y = RZ(0) RY(\pi) RZ(0)
    return (0.0, np.pi, 0.0, np.pi / 2)


@single_qubit_zyz_angles.register
def _(op: PauliZ):  # pylint: disable=unused-argument
    # Z = RZ(0) RY(0) RZ(\pi)
    return (np.pi, 0.0, 0.0, np.pi / 2)


@single_qubit_zyz_angles.register
def _(op: S):  # pylint: disable=unused-argument
    # S = RZ(0) RY(0) RZ(\pi/2)
    return (np.pi / 2, 0.0, 0.0, np.pi / 4)


@single_qubit_zyz_angles.register
def _(op: T):  # pylint: disable=unused-argument
    # T = RZ(0) RY(0) RZ(\pi/4)
    return (np.pi / 4, 0.0, 0.0, np.pi / 8)


@single_qubit_zyz_angles.register
def _(op: SX):  # pylint: disable=unused-argument
    # SX = RZ(-\pi/2) RY(\pi/2) RZ(\pi/2)
    return (np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 4)


@single_qubit_zyz_angles.register
def _(op: RX):
    # RX(\theta) = RZ(-\pi/2) RY(\theta) RZ(\pi/2)
    return (np.pi / 2, op.data[0], -np.pi / 2, 0.0)


@single_qubit_zyz_angles.register
def _(op: RY):
    # RY(\theta) = RZ(0) RY(\theta) RZ(0)
    return (0.0, op.data[0], 0.0, 0.0)


@single_qubit_zyz_angles.register
def _(op: RZ):
    # RZ(\theta) = RZ(0) RY(0) RZ(\theta)
    return (op.data[0], 0.0, 0.0, 0.0)


@single_qubit_zyz_angles.register
def _(op: PhaseShift):
    # PhaseShift(\theta) = RZ(0) RY(0) RZ(\theta)
    return (op.data[0], 0.0, 0.0, op.data[0] / 2)


@single_qubit_zyz_angles.register
def _(op: Rot):
    return tuple(op.data) + (0.0,)


@single_qubit_zyz_angles.register
def _(op: AdjointOperation):
    phi, theta, omega, alpha = single_qubit_zyz_angles(op.base)
    return (-omega, -theta, -phi, -alpha)
