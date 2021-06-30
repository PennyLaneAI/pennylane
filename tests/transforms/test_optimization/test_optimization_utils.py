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

from pennylane.transforms.optimization.optimization_utils import _yzy_to_zyz, _fuse_rot_angles
from gate_data import I


class TestRotGateFusion:
    """Test that utility functions for fusing two qml.Rot gates function as expected."""

    @pytest.mark.parametrize(
        ("angles"),
        [([0.15, 0.25, -0.90]), ([0.0, 0.0, 0.0]), ([0.15, 0.25, -0.90]), ([0.05, -1.34, 4.12])],
    )
    def test_yzy_to_zyz(self, angles):
        """Test that a set of rotations of the form YZY is correctly converted
        to a sequence of the form ZYZ."""
        z1, y, z2 = _yzy_to_zyz(angles)

        Y1 = qml.RY(angles[0], wires=0).matrix
        Z = qml.RZ(angles[1], wires=0).matrix
        Y2 = qml.RY(angles[2], wires=0).matrix
        product_yzy = np.linalg.multi_dot([Y2, Z, Y1])

        Z1 = qml.RZ(z1, wires=0).matrix
        Y = qml.RY(y, wires=0).matrix
        Z2 = qml.RZ(z2, wires=0).matrix
        product_zyz = np.linalg.multi_dot([Z2, Y, Z1])

        # Check if U^\dag U is close to the identity
        mat_product = np.dot(np.conj(product_yzy.T), product_zyz)
        mat_product /= mat_product[0, 0]

        assert np.allclose(mat_product, I)

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

        rot_1_mat = qml.Rot(*angles_1, wires=0).matrix
        rot_2_mat = qml.Rot(*angles_2, wires=0).matrix
        matrix_expected = np.dot(rot_2_mat, rot_1_mat)

        fused_angles = _fuse_rot_angles(angles_1, angles_2)

        matrix_obtained = qml.Rot(*fused_angles, wires=0).matrix

        # Check if U^\dag U is close to the identity
        mat_product = np.dot(np.conj(matrix_obtained.T), matrix_expected)
        mat_product /= mat_product[0, 0]

        assert np.allclose(mat_product, I)
