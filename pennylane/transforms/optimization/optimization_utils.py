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
"""Utility functions for circuit optimization."""
# pylint: disable=too-many-return-statements,import-outside-toplevel
from functools import partial

import numpy as np

from pennylane.math import abs as math_abs
from pennylane.math import sum as math_sum
from pennylane.math import allclose, arccos, arctan2, cos, get_interface, is_abstract, sin, stack
from pennylane.wires import Wires
from pennylane.ops.identity import GlobalPhase


def find_next_gate(wires, op_list):
    """Given a list of operations, finds the next operation that acts on at least one of
    the same set of wires, if present.

    Args:
        wires (Wires): A set of wires acted on by a quantum operation.
        op_list (list[Operation]): A list of operations that are implemented after the
            operation that acts on ``wires``.

    Returns:
        int or None: The index, in ``op_list``, of the earliest gate that uses one or more
        of the same wires, or ``None`` if no such gate is present.
    """
    next_gate_idx = None

    for op_idx, op in enumerate(op_list):
        if len(Wires.shared_wires([wires, op.wires])) > 0:
            next_gate_idx = op_idx
            break

    return next_gate_idx


def _zyz_to_quat(angles):
    """Converts a set of Euler angles in ZYZ format to a quaternion."""
    qw = cos(angles[1] / 2) * cos(0.5 * (angles[0] + angles[2]))
    qx = -sin(angles[1] / 2) * sin(0.5 * (angles[0] - angles[2]))
    qy = sin(angles[1] / 2) * cos(0.5 * (angles[0] - angles[2]))
    qz = cos(angles[1] / 2) * sin(0.5 * (angles[0] + angles[2]))

    return stack([qw, qx, qy, qz])


def _quaternion_product(q1, q2):
    """Compute the product of two quaternions, q = q1 * q2."""
    qw = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    qx = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    qy = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    qz = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return stack([qw, qx, qy, qz])


def _singular_quat_to_zyz(qw, qx, qy, qz, y_arg, abstract_jax=False):
    """Compute the ZYZ angles for the singular case of qx = qy = 0"""
    # pylint: disable=too-many-arguments
    z1_arg1 = 2 * (qx * qy + qz * qw)
    z1_arg2 = 1 - 2 * (qx**2 + qz**2)

    if abstract_jax:
        from jax.lax import cond

        return cond(
            y_arg > 0,
            lambda z1_arg1, z1_arg2: stack([arctan2(z1_arg1, z1_arg2), 0.0, 0.0]),
            lambda z1_arg1, z1_arg2: stack([-arctan2(z1_arg1, z1_arg2), np.pi, 0.0]),
            z1_arg1,
            z1_arg2,
        )

    if y_arg > 0:
        z1 = arctan2(z1_arg1, z1_arg2)
        y = z2 = 0.0
    else:
        z1 = -arctan2(z1_arg1, z1_arg2)
        y = np.pi
        z2 = 0.0
    return stack([z1, y, z2])


def _regular_quat_to_zyz(qw, qx, qy, qz, y_arg):
    """Compute the ZYZ angles for the regular case (qx != 0 or qy != 0)"""
    z1_arg1 = 2 * (qy * qz - qw * qx)
    z1_arg2 = 2 * (qx * qz + qw * qy)
    z1 = arctan2(z1_arg1, z1_arg2)

    y = arccos(y_arg)

    z2_arg1 = 2 * (qy * qz + qw * qx)
    z2_arg2 = 2 * (qw * qy - qx * qz)
    z2 = arctan2(z2_arg1, z2_arg2)

    return stack([z1, y, z2])


def _fuse(angles_1, angles_2, abstract_jax=False):
    """Perform fusion of two angle sets. Separated out so we can do JIT with conditionals."""
    # Compute the product of the quaternions
    qw, qx, qy, qz = _quaternion_product(_zyz_to_quat(angles_1), _zyz_to_quat(angles_2))

    # Convert the product back into the angles fed to Rot
    y_arg = 1 - 2 * (qx**2 + qy**2)

    # Require special treatment of the case qx = qy = 0. Note that we have to check
    # for "greater than" as well, because of imprecisions
    if abstract_jax:
        from jax.lax import cond

        return cond(
            math_abs(y_arg) >= 1,
            partial(_singular_quat_to_zyz, abstract_jax=True),
            _regular_quat_to_zyz,
            qw,
            qx,
            qy,
            qz,
            y_arg,
        )

    # Require special treatment of the case qx = qy = 0
    if abs(y_arg) >= 1:  # Have to check for "greater than" as well, because of imprecisions
        return _singular_quat_to_zyz(qw, qx, qy, qz, y_arg)
    return _regular_quat_to_zyz(qw, qx, qy, qz, y_arg)


def _no_fuse(angles_1, angles_2):
    """Special case: do not perform fusion when both Y angles are zero:
        Rot(a, 0, b) Rot(c, 0, d) = Rot(a + b + c + d, 0, 0)
    The quaternion math itself will fail in this case without a conditional.
    """
    return stack([angles_1[0] + angles_1[2] + angles_2[0] + angles_2[2], 0.0, 0.0])


def fuse_rot_angles(angles_1, angles_2):
    """Computed the set of rotation angles that is obtained when composing
    two ``qml.Rot`` operations.

    The ``qml.Rot`` operation represents the most general single-qubit operation.
    Two such operations can be fused into a new operation, however the angular dependence
    is non-trivial.

    Args:
        angles_1 (float): A set of three angles for the first ``qml.Rot`` operation.
        angles_2 (float): A set of three angles for the second ``qml.Rot`` operation.

    Returns:
        array[float]: Rotation angles for a single ``qml.Rot`` operation that
        implements the same operation as the two sets of input angles.
    """

    # Check if we are tracing; if so, use the special conditionals
    if is_abstract(angles_1) or is_abstract(angles_2):
        interface = get_interface(angles_1, angles_2)

        # TODO: implement something similar for torch and tensorflow interfaces
        # If the interface is JAX, use jax.lax.cond so that we can jit even with conditionals
        if interface == "jax":
            from jax.lax import cond

            return cond(
                allclose(angles_1[1], 0.0) * allclose(angles_2[1], 0.0),
                _no_fuse,
                partial(_fuse, abstract_jax=True),
                angles_1,
                angles_2,
            )

    # For other interfaces where we would not be jitting or tracing, we can simply check
    # if we are dealing with the special case of Rot(a, 0, b) Rot(c, 0, d).
    if allclose(angles_1[1], 0.0) and allclose(angles_2[1], 0.0):
        return _no_fuse(angles_1, angles_2)

    return _fuse(angles_1, angles_2)


def _fuse_global_phases(operations):
    """Fuse all the global phase operations into single one.

    Args:
        operations (list[Operation]): list of operations to be iterated over

    Returns:
        transformed list with a single :func:`~.pennylane.GlobalPhase` operation.
    """

    fused_ops, global_ops = [], []
    for op in operations:
        if isinstance(op, GlobalPhase):
            global_ops.append(op)
        else:
            fused_ops.append(op)

    fused_ops.append(GlobalPhase(math_sum(op.data[0] for op in global_ops)))
    return fused_ops
