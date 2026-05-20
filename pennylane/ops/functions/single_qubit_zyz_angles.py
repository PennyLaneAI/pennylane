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

from pennylane.operation import Operator
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
    """Returns the rotation angles for the ZYZ decomposition of this operator."""
    raise NotImplementedError


@single_qubit_zyz_angles.register
def _h_rot_angles(op: Hadamard):  # pylint: disable=unused-argument
    # H = RZ(\pi) RY(\pi/2) RZ(0)
    return (np.pi, np.pi / 2, 0.0, np.pi / 2)


@single_qubit_zyz_angles.register
def _x_rot_angles(op: PauliX):  # pylint: disable=unused-argument
    # X = RZ(-\pi/2) RY(\pi) RZ(\pi/2)
    return (np.pi / 2, np.pi, -np.pi / 2, np.pi / 2)


@single_qubit_zyz_angles.register
def _y_rot_angles(op: PauliY):  # pylint: disable=unused-argument
    # Y = RZ(0) RY(\pi) RZ(0)
    return (0.0, np.pi, 0.0, np.pi / 2)


@single_qubit_zyz_angles.register
def _z_rot_angles(op: PauliZ):  # pylint: disable=unused-argument
    # Z = RZ(\pi) RY(0) RZ(0)
    return (np.pi, 0.0, 0.0, np.pi / 2)


@single_qubit_zyz_angles.register
def _s_rot_angles(op: S):  # pylint: disable=unused-argument
    # S = RZ(\pi/2) RY(0) RZ(0)
    return (np.pi / 2, 0.0, 0.0, np.pi / 4)


@single_qubit_zyz_angles.register
def _t_rot_angles(op: T):  # pylint: disable=unused-argument
    # T = RZ(\pi/4) RY(0) RZ(0)
    return (np.pi / 4, 0.0, 0.0, np.pi / 8)


@single_qubit_zyz_angles.register
def _sx_rot_angles(op: SX):  # pylint: disable=unused-argument
    # SX = RZ(-\pi/2) RY(\pi/2) RZ(\pi/2)
    return (np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 4)


@single_qubit_zyz_angles.register
def _rx_rot_angles(op: RX):
    # RX(\theta) = RZ(-\pi/2) RY(\theta) RZ(\pi/2)
    return (np.pi / 2, op.data[0], -np.pi / 2, 0.0)


@single_qubit_zyz_angles.register
def _ry_rot_angles(op: RY):
    # RY(\theta) = RZ(0) RY(\theta) RZ(0)
    return (0.0, op.data[0], 0.0, 0.0)


@single_qubit_zyz_angles.register
def _rz_rot_angles(op: RZ):
    # RZ(\theta) = RZ(\theta) RY(0) RZ(0)
    return (op.data[0], 0.0, 0.0, 0.0)


@single_qubit_zyz_angles.register
def _ps_rot_angles(op: PhaseShift):
    # PhaseShift(\theta) = RZ(\theta) RY(0) RZ(0)
    return (op.data[0], 0.0, 0.0, op.data[0] / 2)


@single_qubit_zyz_angles.register
def _rot_rot_angles(op: Rot):
    return tuple(op.data) + (0.0,)


@single_qubit_zyz_angles.register
def _adjoint_rot_angles(op: AdjointOperation):
    omega, theta, phi, alpha = single_qubit_zyz_angles(op.base)
    return (-phi, -theta, -omega, -alpha)
