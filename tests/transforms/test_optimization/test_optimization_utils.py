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

import pennylane as qml
from pennylane.transforms.optimization.optimization_utils import (
    find_next_gate,
    _zyz_to_quat,
    _quaternion_product,
    fuse_rot_angles,
)

from pennylane import numpy as np

from utils import check_matrix_equivalence


sample_op_list = [
    qml.Hadamard(wires="a"),
    qml.CNOT(wires=["a", "b"]),
    qml.Toffoli(wires=[0, 1, "b"]),
    qml.Hadamard(wires="c"),
]


class TestFindNextGate:
    @pytest.mark.parametrize(
        ("wires,op_list,next_gate_idx"),
        [
            ("a", sample_op_list, 0),
            ("b", sample_op_list, 1),
            ("e", sample_op_list, None),
            ([0, 2], sample_op_list, 2),
            ([0, 1], sample_op_list, 2),
            ("c", sample_op_list, 3),
        ],
    )
    def test_find_next_gate(self, wires, op_list, next_gate_idx):
        """Test find_next_gate correctly identifies the next gate in a list of operations that share any number of wires."""
        assert find_next_gate(qml.wires.Wires(wires), op_list) == next_gate_idx


class TestRotGateFusion:
    """Test that utility functions for fusing two qml.Rot gates function as expected."""

    @pytest.mark.parametrize(
        ("angles", "expected_quat"),
        # Examples generated at https://www.mathworks.com/help/robotics/ref/eul2quat.html
        [
            (
                [0.15, 0.25, -0.90],
                [0.923247491800509, -0.062488597726915, 0.107884031713695, -0.363414748929363],
            ),
            ([np.pi / 2, 0.0, 0.0], [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            ([0.15, 0, -0.90], [0.930507621912314, 0.0, 0.0, -0.366272529086048]),
            ([0.0, 0.2, 0.0], [0.995004165278026, 0.0, 0.099833416646828, 0.0]),
        ],
    )
    def test_zyz_to_quat(self, angles, expected_quat):
        """Test that ZYZ Euler angles are correctly converted to quaternions."""
        obtained_quat = _zyz_to_quat(angles)
        assert qml.math.allclose(obtained_quat, expected_quat)

    @pytest.mark.parametrize(
        ("angles_1", "angles_2", "expected_quat"),
        # Examples generated at https://www.vcalc.com/wiki/vCalc/Quaternion+Multiplication
        [
            ([0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [-60.0, 12.0, 30.0, 24.0]),
            ([0.1, 0.0, -0.2, 0.15], [1.0, 0.05, 1.65, -0.25], [0.4675, -0.1925, -0.0275, 0.135]),
            ([0.1, 0.0, 0.0, 0.15], [1.0, 0.0, 0.0, -0.25], [0.1375, 0.0, 0.0, 0.125]),
        ],
    )
    def test_quaternion_product(self, angles_1, angles_2, expected_quat):
        """Test that products of quaternions produce expected results."""
        obtained_quat = _quaternion_product(angles_1, angles_2)
        assert qml.math.allclose(obtained_quat, expected_quat)

    @pytest.mark.autograd
    @pytest.mark.parametrize(
        ("angles_1", "angles_2"),
        [
            ([0.15, 0.25, -0.90], [-0.5, 0.25, 1.3]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([0.1, 0.2, 0.3], [0.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.2, 0.4, -0.5]),
            ([0.15, 0.25, -0.90], [-0.15, -0.25, 0.9]),
            ([0.05, -1.34, 4.12], [-0.8, 0.2, 0.12]),
            ([0.05, 0.0, 4.12], [-0.8, 0.2, 0.12]),
            ([0.05, -1.34, 0.0], [-0.8, 0.2, 0.12]),
            ([0.05, 0.0, 0.1], [-0.2, 0.0, 0.12]),
            ([0.05, 0.0, 0.0], [0.0, 0.0, 0.12]),
            ([0.05, 0.0, 0.0], [0.0, 0.0, -0.05]),
            ([0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]),
            ([0.0, 0.0, 0.05], [-0.05, 0.0, 0.0]),
            ([0.0, 0.0, 0.05], [0.0, 0.0, -0.05]),
            ([0.05, 0.2, 0.0], [0.0, -0.6, 0.12]),
            ([0.05, -1.34, 4.12], [0.0, 0.2, 0.12]),
            ([0.05, -1.34, 4.12], [0.3, 0.0, 0.12]),
            ([np.pi, np.pi / 2, 0.0], [0.0, -np.pi / 2, 0.0]),
            ([0.9, np.pi / 2, 0.0], [0.0, -np.pi / 2, 0.0]),
            ([0.9, np.pi / 2, np.pi / 2], [-np.pi / 2, -np.pi / 2, 0.0]),
        ],
    )
    def test_full_rot_fusion_autograd(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially."""

        def original_ops():
            qml.Rot(*angles_1, wires=0)
            qml.Rot(*angles_2, wires=0)

        matrix_expected = qml.matrix(original_ops, [0])()

        fused_angles = fuse_rot_angles(angles_1, angles_2)
        matrix_obtained = qml.Rot(*fused_angles, wires=0).matrix()

        assert check_matrix_equivalence(matrix_expected, matrix_obtained)
