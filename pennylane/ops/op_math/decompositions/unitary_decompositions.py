# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module defines decomposition functions for unitary matrices."""

import pennylane as qml
from pennylane.decomposition.decomposition_rule import register_resources, resource_rep
from pennylane.decomposition.utils import DecompositionNotApplicableError
from pennylane.math.decomposition import (
    xyx_rotation_angles,
    xzx_rotation_angles,
    zyz_rotation_angles,
)


def _rot_resource(num_wires):
    if num_wires != 1:
        raise DecompositionNotApplicableError
    return {
        qml.Rot: 1,
        qml.GlobalPhase: 1,
    }


@register_resources(_rot_resource)
def rot_decomposition(U, wires, **__):
    """Decomposes a QubitUnitary into a Rot and a GlobalPhase."""
    phi, theta, omega, global_phase = zyz_rotation_angles(U, return_global_phase=True)
    qml.Rot(phi, theta, omega, wires=wires[0])
    qml.GlobalPhase(-global_phase)


def _zyz_resource(num_wires):
    if num_wires != 1:
        raise DecompositionNotApplicableError
    return {
        qml.RZ: 2,
        qml.RY: 1,
        qml.GlobalPhase: 1,
    }


@register_resources(_zyz_resource)
def zyz_decomposition(U, wires, **__):
    """Decomposes a QubitUnitary into a sequence of ZYZ rotations."""
    phi, theta, omega, global_phase = zyz_rotation_angles(U, return_global_phase=True)
    qml.RZ(phi, wires=wires[0])
    qml.RY(theta, wires=wires[0])
    qml.RZ(omega, wires=wires[0])
    qml.GlobalPhase(-global_phase)


def _xyx_resource(num_wires):
    if num_wires != 1:
        raise DecompositionNotApplicableError
    return {
        qml.RX: 2,
        qml.RY: 1,
        qml.GlobalPhase: 1,
    }


@register_resources(_xyx_resource)
def xyx_decomposition(U, wires, **__):
    """Decomposes a QubitUnitary into a sequence of XYX rotations."""
    phi, theta, omega, global_phase = xyx_rotation_angles(U, return_global_phase=True)
    qml.RX(phi, wires=wires[0])
    qml.RY(theta, wires=wires[0])
    qml.RX(omega, wires=wires[0])
    qml.GlobalPhase(-global_phase)


def _xzx_resource(num_wires):
    if num_wires != 1:
        raise DecompositionNotApplicableError
    return {
        qml.RX: 2,
        qml.RZ: 1,
        qml.GlobalPhase: 1,
    }


@register_resources(_xzx_resource)
def xzx_decomposition(U, wires, **__):
    """Decomposes a QubitUnitary into a sequence of XZX rotations."""
    phi, theta, omega, global_phase = xzx_rotation_angles(U, return_global_phase=True)
    qml.RX(phi, wires=wires[0])
    qml.RZ(theta, wires=wires[0])
    qml.RX(omega, wires=wires[0])
    qml.GlobalPhase(-global_phase)


def _zxz_resource(num_wires):
    if num_wires != 1:
        raise DecompositionNotApplicableError
    return {
        qml.RZ: 2,
        qml.RX: 1,
        qml.GlobalPhase: 1,
    }


@register_resources(_zxz_resource)
def zxz_decomposition(U, wires, **__):
    """Decomposes a QubitUnitary into a sequence of ZXZ rotations."""
    phi, theta, omega, global_phase = xzx_rotation_angles(U, return_global_phase=True)
    qml.RZ(phi, wires=wires[0])
    qml.RX(theta, wires=wires[0])
    qml.RZ(omega, wires=wires[0])
    qml.GlobalPhase(-global_phase)
