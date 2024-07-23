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
"""
Unit tests for utilities for optimization transforms.
"""
# pylint: disable=too-few-public-methods

from itertools import product

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.transforms.optimization.optimization_utils import find_next_gate, fuse_rot_angles

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
    def test_full_rot_fusion(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially."""

        def original_ops():
            qml.Rot(*angles_1, wires=0)
            qml.Rot(*angles_2, wires=0)

        matrix_expected = qml.matrix(original_ops, [0])()  # pylint:disable=too-many-function-args

        fused_angles = fuse_rot_angles(angles_1, angles_2)
        matrix_obtained = qml.Rot(*fused_angles, wires=0).matrix()

        assert qml.math.allclose(matrix_expected, matrix_obtained)

    @pytest.mark.slow
    def test_full_rot_fusion_special_angles(self):
        """Test the rotation angle fusion on special multiples of pi/2.
        Also tests that fuse_rot_angles works with batching/broadcasting.
        Do not change the test to non-broadcasted evaluation, as this will
        increase the runtime significantly."""

        special_points = np.array([3 / 2, 1, 1 / 2, 0, -1 / 2, -1, -3 / 2]) * np.pi
        special_angles = np.array(list(product(special_points, repeat=6))).reshape((-1, 2, 3))
        angles_1, angles_2 = np.transpose(special_angles, (1, 0, 2))

        def original_ops():
            qml.Rot(*angles_1.T, wires=0)  # Transpose to bring size-3 axis to front
            qml.Rot(*angles_2.T, wires=0)  # Transpose to bring size-3 axis to front

        matrix_expected = qml.matrix(original_ops, [0])()  # pylint:disable=too-many-function-args

        fused_angles = fuse_rot_angles(angles_1, angles_2)
        matrix_obtained = qml.Rot(
            *fused_angles.T, wires=0
        ).matrix()  # Transpose to bring size-3 axis to front

        assert qml.math.allclose(matrix_expected, matrix_obtained)

    @pytest.mark.slow
    @pytest.mark.jax
    def test_full_rot_fusion_jacobian(self):
        """Test the Jacobian of the rotation angle fusion. Uses batching for performance reasons.
        For known sources of singularities, the Jacobian is checked to indeed return NaN.
        These sources are related to the absolute value of the upper left entry of the matrix product:
         - If it is 1, the derivative of arccos becomes infinite (evaluated at 1), and
         - if its square is 0, the derivative of sqrt becomes infinite (evaluated at 0).
        """
        import jax

        special_points = np.array([3 / 2, 1, 1 / 2, 0, -1 / 2, -1, -3 / 2]) * np.pi
        special_angles = np.array(list(product(special_points, repeat=6))).reshape((-1, 2, 3))
        random_angles = np.random.random((1000, 2, 3))
        all_angles = jax.numpy.concatenate([special_angles, random_angles], dtype=complex)

        def mat_from_prod(angles):
            def original_ops():
                angles1, angles2 = angles[..., 0, :], angles[..., 1, :]
                qml.Rot(angles1[..., 0], angles1[..., 1], angles1[..., 2], wires=0)
                qml.Rot(angles2[..., 0], angles2[..., 1], angles2[..., 2], wires=0)

            return qml.matrix(original_ops, [0])()  # pylint:disable=too-many-function-args

        def mat_from_fuse(angles):
            angles1, angles2 = angles[..., 0, :], angles[..., 1, :]
            fused_angles = fuse_rot_angles(angles1, angles2)
            return qml.Rot(*fused_angles.T, wires=0).matrix()

        # Need holomorphic derivatives because the output matrices are complex-valued
        jac_from_prod = jax.vmap(jax.jacobian(mat_from_prod, holomorphic=True))(all_angles)
        jac_from_fuse = jax.vmap(jax.jacobian(mat_from_fuse, holomorphic=True))(all_angles)

        # expected failures based on the sources mentioned in the docstring above.
        thetas = all_angles[..., 1].T
        (c1, c2), (s1, s2) = np.cos(thetas / 2), np.sin(thetas / 2)
        omega1 = all_angles[:, 0, 2]
        phi2 = all_angles[:, 1, 0]
        # squared absolute value of the relevant entry of the product of the two rotation matrices
        pre_mag = c1**2 * c2**2 + s1**2 * s2**2 - 2 * c1 * c2 * s1 * s2 * np.cos(omega1 + phi2)
        # Compute condition for the two error sources combined
        error_sources = (np.abs(pre_mag - 1) < 1e-12) + (pre_mag == 0j)

        assert qml.math.allclose(jac_from_prod[~error_sources], jac_from_fuse[~error_sources])
        assert qml.math.all(
            qml.math.any(qml.math.isnan(jac_from_fuse[error_sources]), axis=[1, 2, 3, 4])
        )

        """
        mag = np.sqrt(pre_mag)

        alpha1, beta1 = (phi1 + omega1) / 2, (phi1 - omega1) / 2
        alpha2, beta2 = (phi2 + omega2) / 2, (phi2 - omega2) / 2

        alpha_arg1 = -c1 * c2 * np.sin(alpha1 + alpha2) - s1 * s2 * np.sin(beta2 - beta1)
        alpha_arg2 = c1 * c2 * np.cos(alpha1 + alpha2) - s1 * s2 * np.cos(beta2 - beta1)
        beta_arg1 = -c1 * s2 * np.sin(alpha1 + beta2) + s1 * c2 * np.sin(alpha2 - beta1)
        beta_arg2 = c1 * s2 * np.cos(alpha1 + beta2) + s1 * c2 * np.cos(alpha2 - beta1)

        def partial(all_angles):
            phi1, theta1, omega1 = all_angles[..., 0, :].T
            phi2, theta2, omega2 = all_angles[..., 1, :].T
            c1, c2, s1, s2 = qml.math.cos(theta1 / 2), qml.math.cos(theta2 / 2), qml.math.sin(theta1 / 2), qml.math.sin(theta2 / 2)
            mag = qml.math.sqrt(c1 ** 2 * c2 ** 2 + s1 ** 2 * s2 ** 2 - 2 * c1 * c2 * s1 * s2 * qml.math.cos(omega1 + phi2))

            #alpha1, beta1 = (phi1 + omega1) / 2, (phi1 - omega1) / 2
            #alpha2, beta2 = (phi2 + omega2) / 2, (phi2 - omega2) / 2

            #alpha_arg1 = -c1 * c2 * np.sin(alpha1 + alpha2) - s1 * s2 * np.sin(beta2 - beta1)
            #alpha_arg2 = c1 * c2 * np.cos(alpha1 + alpha2) - s1 * s2 * np.cos(beta2 - beta1)
            #beta_arg1 = -c1 * s2 * np.sin(alpha1 + beta2) + s1 * c2 * np.sin(alpha2 - beta1)
            #beta_arg2 = c1 * s2 * np.cos(alpha1 + beta2) + s1 * c2 * np.cos(alpha2 - beta1)
            #print(mag)
            return mag

        jac_partial = jax.vmap(jax.jacobian(partial))(all_angles.real)

        expected_singular_ids = np.where(
            np.any([
                (np.abs(pre_mag - 1) < 1e-12),
                (pre_mag == 0j),
                #np.isnan(mag),
                #(np.abs(alpha_arg1) + np.abs(alpha_arg1) < 1e-10),
                #(np.abs(beta_arg1) + np.abs(beta_arg1) < 1e-10)
            ], axis=0)
        )
        pre_mag2 = c1 ** 2 * s2 ** 2 + s1 ** 2 * c2 ** 2 + 2 * c1 * c2 * s1 * s2 * qml.math.cos(omega1 + phi2)
        print(pre_mag2[np.where(pre_mag==0j)[0]])
        for idx in np.where(qml.math.any(qml.math.isnan(jac_from_fuse), axis=[1, 2, 3, 4]))[0]:
            if idx not in expected_singular_ids[0]:
                weird_idx = idx
                break
        print(weird_idx)
        print(jac_from_fuse[weird_idx])
        print(all_angles[weird_idx])
        print(f"{pre_mag[weird_idx]:.20f}")
        print(jax.jacobian(qml.math.sqrt,holomorphic=True)(pre_mag[weird_idx]))
        print(jax.jacobian(qml.math.sqrt)(pre_mag.real[weird_idx]))
        print(len(all_angles))
        print(len(expected_singular_ids[0]))
        print(len(np.where(qml.math.any(qml.math.isnan(jac_from_fuse), axis=[1, 2, 3, 4]))[0]))
        print(jac_partial.shape)
        print(len(np.where(qml.math.any(qml.math.isnan(jac_partial), axis=[1, 2]))[0]))
        print(np.unique(mag[np.array(np.where(qml.math.any(qml.math.isnan(jac_partial), axis=[1, 2]))[0])]))
        """
