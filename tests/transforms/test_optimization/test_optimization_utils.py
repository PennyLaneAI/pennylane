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

    generic_test_angles = [
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
    ]

    def run_interface_test(self, angles_1, angles_2):
        """Execute standard test lines for different interfaces and batch tests.
        Note that the transpose calls only are relevant for tests with batching."""

        def original_ops():
            qml.Rot(*qml.math.transpose(angles_1), wires=0)
            qml.Rot(*qml.math.transpose(angles_2), wires=0)

        matrix_expected = qml.matrix(original_ops, [0])()  # pylint:disable=too-many-function-args

        fused_angles = fuse_rot_angles(angles_1, angles_2)
        # The reshape is only used in the _mixed_batching test. Otherwise it is irrelevant.
        matrix_obtained = qml.Rot(
            *qml.math.transpose(qml.math.reshape(fused_angles, (-1, 3))), wires=0
        ).matrix()

        assert qml.math.allclose(matrix_expected, matrix_obtained)

    @pytest.mark.parametrize("angles_1, angles_2", generic_test_angles)
    def test_full_rot_fusion_numpy(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially."""
        self.run_interface_test(angles_1, angles_2)

    mixed_batched_angles = [
        ([[0.4, 0.1, 0.0], [0.7, 0.2, 0.1]], [-0.9, 1.2, 0.6]),  # (2, None)
        ([-0.9, 1.2, 0.6], [[0.4, 0.1, 0.0], [0.7, 0.2, 0.1]]),  # (None, 2)
        ([-0.9, 1.2, 0.6], [[[0.4, 0.1, 0.0], [0.7, 0.2, 0.1]]] * 4),  # (None, (4, 2))
        (
            [[[-0.9, 1.2, 0.6]] * 2] * 4,
            [[[0.4, 0.1, 0.0], [0.7, 0.2, 0.1]]] * 4,
        ),  # ((4, 2), (4, 2))
    ]

    @pytest.mark.parametrize("angles_1, angles_2", mixed_batched_angles)
    def test_full_rot_fusion_mixed_batching(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially when the input angles are batched
        with mixed batching shapes."""

        reshaped_angles_1 = np.reshape(angles_1, (-1, 3) if np.ndim(angles_1) > 1 else (3,))
        reshaped_angles_2 = np.reshape(angles_2, (-1, 3) if np.ndim(angles_2) > 1 else (3,))
        self.run_interface_test(reshaped_angles_1, reshaped_angles_2)

    @pytest.mark.autograd
    @pytest.mark.parametrize("angles_1, angles_2", generic_test_angles)
    def test_full_rot_fusion_autograd(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially, in Autograd."""

        angles_1, angles_2 = qml.numpy.array(angles_1), qml.numpy.array(angles_1)
        self.run_interface_test(angles_1, angles_2)

    @pytest.mark.tf
    @pytest.mark.parametrize("angles_1, angles_2", generic_test_angles)
    def test_full_rot_fusion_tensorflow(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially, in Tensorflow."""
        import tensorflow as tf

        angles_1 = tf.Variable(angles_1, dtype=tf.float64)
        angles_2 = tf.Variable(angles_2, dtype=tf.float64)
        self.run_interface_test(angles_1, angles_2)

    @pytest.mark.torch
    @pytest.mark.parametrize("angles_1, angles_2", generic_test_angles)
    def test_full_rot_fusion_torch(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially, in torch."""
        import torch

        angles_1 = torch.tensor(angles_1, requires_grad=True, dtype=torch.float64)
        angles_2 = torch.tensor(angles_2, requires_grad=True, dtype=torch.float64)
        self.run_interface_test(angles_1, angles_2)

    @pytest.mark.jax
    @pytest.mark.parametrize("angles_1, angles_2", generic_test_angles)
    def test_full_rot_fusion_jax(self, angles_1, angles_2):
        """Test that the fusion of two Rot gates has the same effect as
        applying the Rots sequentially, in JAX."""
        import jax

        angles_1, angles_2 = jax.numpy.array(angles_1), jax.numpy.array(angles_1)
        self.run_interface_test(angles_1, angles_2)

    @pytest.mark.slow
    def test_full_rot_fusion_special_angles(self):
        """Test the rotation angle fusion on special multiples of pi/2.
        Also tests that fuse_rot_angles works with batching/broadcasting.
        Do not change the test to non-broadcasted evaluation, as this will
        increase the runtime significantly."""

        special_points = np.array([3 / 2, 1, 1 / 2, 0, -1 / 2, -1, -3 / 2]) * np.pi
        special_angles = np.array(list(product(special_points, repeat=6))).reshape((-1, 2, 3))
        angles_1, angles_2 = np.transpose(special_angles, (1, 0, 2))
        self.run_interface_test(angles_1, angles_2)

    # pylint: disable=too-many-arguments
    def run_jacobian_test(self, all_angles, jac_fn, is_batched, jit_fn=None, array_fn=None):
        """Execute standard test lines for testing Jacobians with different interfaces.
        #Note that the transpose calls only are relevant for tests with batching."""

        def mat_from_prod(angles):
            def original_ops():
                angles1, angles2 = angles[..., 0, :], angles[..., 1, :]
                qml.Rot(angles1[..., 0], angles1[..., 1], angles1[..., 2], wires=0)
                qml.Rot(angles2[..., 0], angles2[..., 1], angles2[..., 2], wires=0)

            return qml.matrix(original_ops, [0])()  # pylint:disable=too-many-function-args

        def mat_from_fuse(angles):
            angles1, angles2 = angles[..., 0, :], angles[..., 1, :]
            fused_angles = qml.math.transpose(fuse_rot_angles(angles1, angles2))
            return qml.Rot(fused_angles[0], fused_angles[1], fused_angles[2], wires=0).matrix()

        if jit_fn is not None:
            mat_from_fuse = jit_fn(mat_from_fuse)

        if is_batched:
            jac_from_prod = jac_fn(mat_from_prod)(all_angles)
            jac_from_fuse = jac_fn(mat_from_fuse)(all_angles)
        else:
            jac_from_prod = qml.math.stack([jac_fn(mat_from_prod)(a) for a in all_angles])
            jac_from_fuse = qml.math.stack([jac_fn(mat_from_fuse)(a) for a in all_angles])

        if array_fn is not None:
            # Convert to vanilla numpy
            all_angles = array_fn(all_angles)

        # expected failures based on the sources mentioned in the docstring above.
        thetas = qml.math.transpose(all_angles[..., 1])
        (c1, c2), (s1, s2) = qml.math.cos(thetas / 2), qml.math.sin(thetas / 2)
        omega1 = all_angles[:, 0, 2]
        phi2 = all_angles[:, 1, 0]
        # squared absolute value of the relevant entry of the product of the two rotation matrices
        pre_mag = (
            c1**2 * c2**2 + s1**2 * s2**2 - 2 * c1 * c2 * s1 * s2 * qml.math.cos(omega1 + phi2)
        )
        # Compute condition for the two error sources combined
        error_sources = (qml.math.abs(pre_mag - 1) < 1e-12) | (pre_mag == 0)

        assert qml.math.allclose(jac_from_prod[~error_sources], jac_from_fuse[~error_sources])
        nans = qml.math.isnan(jac_from_fuse[error_sources])
        nans = qml.math.reshape(nans, (len(nans), -1))
        assert qml.math.all(qml.math.any(nans, axis=1))

    @pytest.mark.slow
    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    def test_jacobian_jax(self, use_jit):
        """Test the Jacobian of the rotation angle fusion with JAX. Uses batching for performance
        reasons. For known sources of singularities, the Jacobian is checked to indeed return NaN.
        These sources are related to the absolute value of the upper left entry of the matrix
        product:
         - If it is 1, the derivative of arccos becomes infinite (evaluated at 1), and
         - if its square is 0, the derivative of sqrt becomes infinite (evaluated at 0).
        """
        import jax

        special_points = np.array([3 / 2, 1, 1 / 2, 0, -1 / 2, -1, -3 / 2]) * np.pi
        special_angles = np.array(list(product(special_points, repeat=6))).reshape((-1, 2, 3))
        random_angles = np.random.random((1000, 2, 3))
        # Need holomorphic derivatives and complex inputs because the output matrices are complex
        all_angles = jax.numpy.concatenate([special_angles, random_angles])

        # We need to define the Jacobian function manually because fuse_rot_angles is not guaranteed to be holomorphic,
        # and jax.jacobian requires real-valued outputs for non-holomorphic functions.
        def jac_fn(fn):
            real_fn = lambda arg: qml.math.real(fn(arg))
            imag_fn = lambda arg: qml.math.imag(fn(arg))
            real_jac_fn = jax.vmap(jax.jacobian(real_fn))
            imag_jac_fn = jax.vmap(jax.jacobian(imag_fn))
            return lambda arg: real_jac_fn(arg) + 1j * imag_jac_fn(arg)

        jit_fn = jax.jit if use_jit else None
        self.run_jacobian_test(all_angles, jac_fn, is_batched=True, jit_fn=jit_fn)

    @pytest.mark.slow
    @pytest.mark.torch
    def test_jacobian_torch(self):
        """Test the Jacobian of the rotation angle fusion with torch.
        For known sources of singularities, the Jacobian is checked to indeed return NaN.
        These sources are related to the absolute value of the upper left entry of the matrix
        product:
         - If it is 1, the derivative of arccos becomes infinite (evaluated at 1), and
         - if its square is 0, the derivative of sqrt becomes infinite (evaluated at 0).
        """
        import torch

        # Testing fewer points than with batching to limit test runtimes
        special_points = np.array([1, 0, -1]) * np.pi
        special_angles = np.array(list(product(special_points, repeat=6))).reshape((-1, 2, 3))
        random_angles = np.random.random((10, 2, 3))
        all_angles = np.concatenate([special_angles, random_angles])

        # Need holomorphic derivatives and complex inputs because the output matrices are complex
        all_angles = torch.tensor(all_angles, requires_grad=True)

        def jacobian(fn):
            real_fn = lambda arg: qml.math.real(fn(arg))
            imag_fn = lambda arg: qml.math.imag(fn(arg))
            real_jac_fn = lambda arg: torch.autograd.functional.jacobian(real_fn, (arg,))
            imag_jac_fn = lambda arg: torch.autograd.functional.jacobian(imag_fn, (arg,))
            return lambda arg: real_jac_fn(arg)[0] + 1j * imag_jac_fn(arg)[0]

        array_fn = lambda x: x.detach().numpy()
        self.run_jacobian_test(all_angles, jacobian, is_batched=False, array_fn=array_fn)

    @pytest.mark.slow
    @pytest.mark.autograd
    def test_jacobian_autograd(self):
        """Test the Jacobian of the rotation angle fusion with Autograd.
        For known sources of singularities, the Jacobian is checked to indeed return NaN.
        These sources are related to the absolute value of the upper left entry of the matrix
        product:
         - If it is 1, the derivative of arccos becomes infinite (evaluated at 1), and
         - if its square is 0, the derivative of sqrt becomes infinite (evaluated at 0).
        """
        special_points = np.array([1, 0, -1]) * np.pi
        special_angles = np.array(list(product(special_points, repeat=6))).reshape((-1, 2, 3))
        random_angles = np.random.random((100, 2, 3))
        # Need holomorphic derivatives and complex inputs because the output matrices are complex
        all_angles = qml.numpy.concatenate([special_angles, random_angles], requires_grad=True)

        def jacobian(fn):
            real_fn = lambda *args: qml.math.real(fn(*args))
            imag_fn = lambda *args: qml.math.imag(fn(*args))
            real_jac_fn = qml.jacobian(real_fn)
            imag_jac_fn = qml.jacobian(imag_fn)
            return lambda *args: real_jac_fn(*args) + 1j * imag_jac_fn(*args)

        self.run_jacobian_test(all_angles, jacobian, is_batched=False)

    @pytest.mark.skip
    @pytest.mark.slow
    @pytest.mark.tf
    def test_jacobian_tf(self):
        """Test the Jacobian of the rotation angle fusion with TensorFlow.
        For known sources of singularities, the Jacobian is checked to indeed return NaN.
        These sources are related to the absolute value of the upper left entry of the matrix
        product:
         - If it is 1, the derivative of arccos becomes infinite (evaluated at 1), and
         - if its square is 0, the derivative of sqrt becomes infinite (evaluated at 0).
        """
        import tensorflow as tf

        # Testing fewer points than with batching to limit test runtimes
        special_points = np.array([0, 1]) * np.pi
        special_angles = np.array(list(product(special_points, repeat=6))).reshape((-1, 2, 3))
        random_angles = np.random.random((3, 2, 3))
        all_angles = np.concatenate([special_angles, random_angles])

        def jacobian(fn):

            def jac_fn(arg):
                arg = tf.Variable(arg)
                with tf.GradientTape() as t:
                    out = fn(arg)
                return t.jacobian(out, arg)

            return jac_fn

        # Need holomorphic derivatives and complex inputs because the output matrices are complex
        all_angles = tf.Variable(all_angles)
        self.run_jacobian_test(all_angles, jacobian, is_batched=False)
