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

import pytest
import numpy as np

import pennylane as qml

from pennylane.transforms.optimization_utils import yzy_to_zyz, fuse_rot


def normalize_angle(theta):
    """Normalize an angle into the range -np.pi to np.pi.

    Useful for testing matrix equivalence up to a global phase.
    """
    if theta > np.pi:
        theta -= 2 * np.pi * np.ceil(theta // np.pi)
    elif theta < -np.pi:
        theta += 2 * np.pi * np.ceil(theta // np.pi)
    return theta


class TestRotGateFusion:
    """Test that utility functions for fusing two qml.Rot gates function as expected."""

    @pytest.mark.parametrize(
        ("angles"),
        [([0.15, 0.25, -0.90]), ([0.0, 0.0, 0.0]), ([0.15, 0.25, -0.90]), ([0.05, -1.34, 4.12])],
    )
    def test_yzy_to_zyz(self, angles):
        """Test that a set of rotations of the form YZY is correctly converted
        to a sequence of the form ZYZ."""
        angles = [normalize_angle(x) for x in angles]

        z1, y, z2 = yzy_to_zyz(*angles)

        Y1 = qml.RY(angles[0], wires=0).matrix
        Z = qml.RZ(angles[1], wires=0).matrix
        Y2 = qml.RY(angles[2], wires=0).matrix
        product_yzy = np.linalg.multi_dot([Y1, Z, Y2])

        Z1 = qml.RZ(z1, wires=0).matrix
        Y = qml.RY(y, wires=0).matrix
        Z2 = qml.RZ(z2, wires=0).matrix
        product_zyz = np.linalg.multi_dot([Z1, Y, Z2])

        assert np.allclose(product_yzy, product_zyz)

    @pytest.mark.parametrize(
        ("angles_1", "angles_2"),
        [
            ([0.15, 0.25, -0.90], [-0.5, 0.25, 1.3]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.15, 0.25, -0.90], [-0.15, -0.25, 0.9]),
            ([0.05, -1.34, 4.12], [-0.8, 0.2, 0.12]),
        ],
    )
    def test_full_rot_fusion(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially."""
        angles_1 = [normalize_angle(x) for x in angles_1]
        angles_2 = [normalize_angle(x) for x in angles_2]

        rot_1_mat = qml.Rot(*angles_1, wires=0).matrix
        rot_2_mat = qml.Rot(*angles_2, wires=0).matrix
        matrix_expected = np.dot(rot_2_mat, rot_1_mat)

        fused_angles = [normalize_angle(x) for x in fuse_rot(angles_1, angles_2)]
        print(fused_angles)
        matrix_obtained = qml.Rot(*fused_angles, wires=0).matrix

        print(matrix_expected)
        print(matrix_obtained)

        assert np.allclose(matrix_obtained, matrix_expected)
