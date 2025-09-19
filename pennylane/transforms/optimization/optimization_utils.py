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

import pennylane as qml
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
        if any(qml.math.is_abstract(w) for w in op.wires):
            break
        if len(Wires.shared_wires([wires, op.wires])) > 0:
            next_gate_idx = op_idx
            break

    return next_gate_idx


def _try_no_fuse(angles_1, angles_2):
    """Try to combine rotation angles without trigonometric identities
    if some angles in the input angles vanish."""
    # This sum is only computed to obtain a dtype-coerced object that respects
    # TensorFlow's coercion rules between Python/NumPy objects and TF objects.
    _sum = angles_1 + angles_2
    # moveaxis required for batched inputs
    phi1, theta1, omega1 = qml.math.moveaxis(qml.math.cast_like(angles_1, _sum), -1, 0)
    phi2, theta2, omega2 = qml.math.moveaxis(qml.math.cast_like(angles_2, _sum), -1, 0)

    if qml.math.allclose(omega1 + phi2, 0.0):
        return qml.math.stack([phi1, theta1 + theta2, omega2])
    if qml.math.allclose(theta1, 0.0):
        # No Y rotation in first Rot
        if qml.math.allclose(theta2, 0.0):
            # Z rotations only
            zero = qml.math.zeros_like(phi1) + qml.math.zeros_like(phi2)
            return qml.math.stack([phi1 + omega1 + phi2 + omega2, zero, zero])
        return qml.math.stack([phi1 + omega1 + phi2, theta2, omega2])
    if qml.math.allclose(theta2, 0.0):
        # No Y rotation in second Rot
        return qml.math.stack([phi1, theta1, omega1 + phi2 + omega2])
    return None


def fuse_rot_angles(angles_1, angles_2):
    r"""Compute the set of rotation angles that is equivalent to performing
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

    This function supports broadcasting/batching as long as the two inputs are standard
    broadcast-compatible.

    .. note::

        The output angles are not always defined uniquely because Euler angles are not
        unique for some rotations. ``fuse_rot_angles`` makes a particular
        choice in this case.

    .. warning::

        This function is not differentiable everywhere. It has singularities for specific
        input values where the derivative will be NaN.

    .. warning::

        This function is numerically unstable at singular points. It is recommended to use
        it with 64-bit floating point precision.

    See the documentation of :func:`~.pennylane.transforms.single_qubit_fusion` for a
    mathematical derivation of this function.
    """
    angles_1 = qml.math.asarray(angles_1)
    angles_2 = qml.math.asarray(angles_2)

    if not (
        qml.math.is_abstract(angles_1)
        or qml.math.is_abstract(angles_2)
        or qml.math.requires_grad(angles_1)
        or qml.math.requires_grad(angles_2)
    ):
        fused_angles = _try_no_fuse(angles_1, angles_2)
        if fused_angles is not None:
            return fused_angles

    # moveaxis required for batched inputs
    angles_1 = qml.math.moveaxis(angles_1, -1, 0)
    angles_2 = qml.math.moveaxis(angles_2, -1, 0)
    phi1, theta1, omega1 = angles_1[0], angles_1[1], angles_1[2]
    phi2, theta2, omega2 = angles_2[0], angles_2[1], angles_2[2]
    c1, c2 = qml.math.cos(theta1 / 2), qml.math.cos(theta2 / 2)
    s1, s2 = qml.math.sin(theta1 / 2), qml.math.sin(theta2 / 2)

    mag = qml.math.sqrt(
        c1**2 * c2**2 + s1**2 * s2**2 - 2 * c1 * c2 * s1 * s2 * qml.math.cos(omega1 + phi2)
    )
    theta_f = 2 * qml.math.arccos(mag)

    alpha1, beta1 = (phi1 + omega1) / 2, (phi1 - omega1) / 2
    alpha2, beta2 = (phi2 + omega2) / 2, (phi2 - omega2) / 2

    alpha_arg1 = -c1 * c2 * qml.math.sin(alpha1 + alpha2) - s1 * s2 * qml.math.sin(beta2 - beta1)
    alpha_arg2 = c1 * c2 * qml.math.cos(alpha1 + alpha2) - s1 * s2 * qml.math.cos(beta2 - beta1)
    alpha_f = -1 * qml.math.arctan2(alpha_arg1, alpha_arg2)

    beta_arg1 = -c1 * s2 * qml.math.sin(alpha1 + beta2) + s1 * c2 * qml.math.sin(alpha2 - beta1)
    beta_arg2 = c1 * s2 * qml.math.cos(alpha1 + beta2) + s1 * c2 * qml.math.cos(alpha2 - beta1)
    beta_f = -1 * qml.math.arctan2(beta_arg1, beta_arg2)

    return qml.math.stack([alpha_f + beta_f, theta_f, alpha_f - beta_f], axis=-1)


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
