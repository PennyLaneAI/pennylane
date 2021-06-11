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

from pennylane import numpy as np
from pennylane.math import allclose, isclose, sin, cos, arccos, arctan2


def yzy_to_zyz(y1, z, y2):
    """Converts a set of angles representing a sequence of rotations RY, RZ, RY into
    an equivalent sequence of the form RZ, RY, RZ.

    Any rotation in 3-dimensional space (or, equivalently, any single-qubit unitary)
    can be expressed as a sequence of rotations about 3 axes in 12 different ways.
    Typically, the arbitrary single-qubit rotation is expressed as RZ(a) RY(b) RZ(c),
    but there are some situations, e.g., composing two such rotations, where we need
    to convert between representations. This function converts the angles of a sequence

    .. math::

       RY(y_1) RZ(z) RY(y_2)

    into the form

    .. math::

       RZ(z_1) RY(y) RZ(z_2)

    This is accomplished by first converting the rotation to quaternion form, and then
    extracting the desired set of angles.

    Args:
        y1 (float): The angle of the first ``RY`` rotation.
        z (float): The angle of the inner ``RZ`` rotation.
        y2 (float): The angle of the second ``RY`` rotation.

    Returns:
        (float, float, float): A tuple of rotation angles in the ZYZ representation.
    """
    # Catch the case where everything is close to 0
    if allclose(np.array([y1, z, y2]), np.zeros(3)):
        return (0.0, 0.0, 0.0)

    # First, compute the quaternion representation
    # https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
    qw = cos(z / 2) * cos(0.5 * (y1 + y2))
    qx = sin(z / 2) * sin(0.5 * (y1 - y2))
    qy = cos(z / 2) * sin(0.5 * (y1 + y2))
    qz = sin(z / 2) * cos(0.5 * (y1 - y2))

    # Now convert from YZY Euler angles to ZYZ angles
    # Source: http://bediyap.com/programming/convert-quaternion-to-euler-rotations/
    z1_arg1 = 2 * (qy * qz - qw * qx)
    z1_arg2 = 2 * (qx * qz + qw * qy)
    z1 = arctan2(z1_arg1, z1_arg2)

    y = arccos(qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2)

    z2_arg1 = 2 * (qy * qz + qw * qx)
    z2_arg2 = -2 * (qx * qz - qw * qy)
    z2 = arctan2(z2_arg1, z2_arg2)

    return (z1, y, z2)


def fuse_rot(angles_1, angles_2):
    """Computed the set of rotation angles that is obtained when composing
    two ``qml.Rot`` operations.

    The ``qml.Rot`` operation represents the most general single-qubit operation.
    Two such operations can be fused into a new operation, however the angular dependence
    is non-trivial.

    Args:
        angles_1 (float): A set of three angles for the first ``qml.Rot`` operation.
        angles_2 (float): A set of three angles for the second ``qml.Rot`` operation.

    Returns:
        array[float]: A tuple of rotation angles for a single ``qml.Rot`` operation
        that implements the same operation as the two sets of input angles.
    """
    are_angles_1_zero = allclose(np.array(angles_1), np.zeros(3))
    are_angles_2_zero = allclose(np.array(angles_2), np.zeros(3))

    if are_angles_1_zero and are_angles_2_zero:
        return np.array([0.0, 0.0, 0.0])
    elif are_angles_1_zero:
        return np.array(angles_2)
    elif are_angles_2_zero:
        return np.array(angles_1)

    # RZ(a) RY(b) RZ(c) fused with RZ(d) RY(e) RZ(f)
    # first produces RZ(a) RY(b) RZ(c+d) RY(e) RZ(f)
    leftmost_z = angles_1[0]
    middle_yzy = angles_1[1], angles_1[2] + angles_2[0], angles_2[1]
    rightmost_z = angles_2[2]

    # There are a few other cases to consider where things can be 0 and
    # avoid having to use the quaternion conversion routine
    # If b = 0, then we have RZ(a + c + d) RY(e) RZ(f)
    if isclose(middle_yzy[0], 0.0):
        return np.array([leftmost_z + middle_yzy[1], middle_yzy[2], rightmost_z])
    # If c + d is close to 0, then we have the case RZ(a) RY(b + e) RZ(f)
    elif isclose(middle_yzy[1], 0.0):
        return np.array([leftmost_z, middle_yzy[0] + middle_yzy[2], rightmost_z])
    # If e is close to 0, then we have the case RZ(a) RY(b) RZ(c + d + f)
    elif isclose(middle_yzy[2], 0.0):
        return np.array([leftmost_z, middle_yzy[0], middle_yzy[1] + rightmost_z])

    # Otherwise, we need to turn the RY(b) RZ(c+d) RY(e) into something
    # of the form RZ(u) RY(v) RZ(w)
    u, v, w = yzy_to_zyz(*middle_yzy)

    # Then we can combine to create
    # RZ(a + u) RY(v) RZ(w + f)
    return np.array([leftmost_z + u, v, w + rightmost_z])
