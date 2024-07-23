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
from pennylane.math import arccos, arctan2, asarray, cos, sin, sqrt, stack, transpose
from pennylane.ops.identity import GlobalPhase
from pennylane.wires import Wires


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


def fuse_rot_angles(angles1, angles2):
    """Compute the set of rotation angles that is equivalent to performing
    two successive ``qml.Rot`` operations.

    The ``qml.Rot`` operation represents the most general single-qubit operation.
    Two such operations can be fused into a new operation, however the angular dependence
    is non-trivial.

    Args:
        angles_1 (tensor_like): A set of three angles for the first ``qml.Rot`` operation.
        angles_2 (tensor_like): A set of three angles for the second ``qml.Rot`` operation.

    Returns:
        tensor_like: Rotation angles for a single ``qml.Rot`` operation that
        implements the same operation as the two sets of input angles.

    .. note::

        The output angles are not always defined uniquely because Euler angles are not
        defined uniquely for some rotations. ``fuse_rot_angles`` makes a particular
        choice in this case.

    .. warning::

        This function is not differentiable everywhere. It has singularities for specific
        input values, where the derivative will be ``nan``.

    .. warning::

        This function is numerically unstable at singular points. It is recommended to use
        it with 64-bit floating point precision.

    """
    phi1, theta1, omega1 = transpose(asarray(angles1))
    phi2, theta2, omega2 = transpose(asarray(angles2))
    c1, c2, s1, s2 = cos(theta1 / 2), cos(theta2 / 2), sin(theta1 / 2), sin(theta2 / 2)

    mag = sqrt(c1**2 * c2**2 + s1**2 * s2**2 - 2 * c1 * c2 * s1 * s2 * cos(omega1 + phi2))
    print(f"{mag=}")
    theta_f = 2 * arccos(mag)

    alpha1, beta1 = (phi1 + omega1) / 2, (phi1 - omega1) / 2
    alpha2, beta2 = (phi2 + omega2) / 2, (phi2 - omega2) / 2

    alpha_arg1 = -c1 * c2 * sin(alpha1 + alpha2) - s1 * s2 * sin(beta2 - beta1)
    alpha_arg2 = c1 * c2 * cos(alpha1 + alpha2) - s1 * s2 * cos(beta2 - beta1)
    print(f"{alpha_arg1=}, {alpha_arg2=}")
    alpha_f = -1 * arctan2(alpha_arg1, alpha_arg2)

    beta_arg1 = -c1 * s2 * sin(alpha1 + beta2) + s1 * c2 * sin(alpha2 - beta1)
    beta_arg2 = c1 * s2 * cos(alpha1 + beta2) + s1 * c2 * cos(alpha2 - beta1)
    print(f"{beta_arg1=}, {beta_arg2=}")
    beta_f = -1 * arctan2(beta_arg1, beta_arg2)

    return stack([alpha_f + beta_f, theta_f, alpha_f - beta_f], axis=-1)


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

    fused_ops.append(GlobalPhase(sum(op.data[0] for op in global_ops)))
    return fused_ops
