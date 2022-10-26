# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the (reduced) density matrix transform."""

import pytest

import pennylane as qml
from pennylane import numpy as np

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow", minversion="2.1")
torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")


angle_values = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
devices = [
    "default.qubit",
    "default.mixed",
]
interfaces = [
    "autograd",
    "torch",
    "tf",
    "jax",
]
wires_list = [[0], [1], [0, 1]]
check_state = [False, True]


class TestDensityMatrixQNode:
    """Tests for the (reduced) density matrix for QNodes returning states."""

    @pytest.mark.parametrize("check", check_state)
    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("angle", angle_values)
    @pytest.mark.parametrize("wires", wires_list)
    def test_density_matrix_from_qnode(self, device, wires, angle, interface, check, tol):
        """Test the density matrix from matrix for single wires."""
        dev = qml.device(device, wires=2)

        @qml.qnode(dev, interface=interface)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        density_matrix = qml.qinfo.reduced_dm(circuit, wires=wires)(angle)

        def expected_density_matrix(x, wires):
            if wires == [0] or wires == [1]:
                return [[np.cos(x / 2) ** 2, 0], [0, np.sin(x / 2) ** 2]]
            elif wires == [0, 1]:
                return [
                    [np.cos(x / 2) ** 2, 0, 0, 0.0 + np.cos(x / 2) * np.sin(x / 2) * 1j],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0.0 - np.cos(x / 2) * np.sin(x / 2) * 1j, 0, 0, np.sin(x / 2) ** 2],
                ]

        assert np.allclose(expected_density_matrix(angle, wires), density_matrix, atol=tol, rtol=0)

    def test_qnode_not_returning_state(self):
        """Test that the QNode of reduced_dm function must return state."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.RZ(0, wires=[0])
            return qml.expval(qml.PauliX(wires=0))

        with pytest.raises(ValueError, match="The qfunc return type needs to be a state"):
            qml.qinfo.reduced_dm(circuit, wires=[0])()

    def test_density_matrix_qnode_jax_jit(self, tol):
        """Test reduced_dm jitting for QNode."""
        from jax import jit
        import jax.numpy as jnp

        angle = jnp.array(0.1)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="jax-jit")
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        density_matrix = jit(qml.qinfo.reduced_dm(circuit, wires=[0]))(angle)
        expected_density_matrix = [[np.cos(angle / 2) ** 2, 0], [0, np.sin(angle / 2) ** 2]]

        assert np.allclose(density_matrix, expected_density_matrix, atol=tol, rtol=0)

    def test_density_matrix_qnode_tf_jit(self):
        """Test jitting the density matrix from state vector function with Tf."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, interface="tf")
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        density_matrix = tf.function(
            qml.qinfo.reduced_dm(circuit, wires=[0]),
            jit_compile=True,
            input_signature=(tf.TensorSpec(shape=(), dtype=tf.float32),),
        )
        density_matrix = density_matrix(tf.Variable(0.0, dtype=tf.float32))
        assert np.allclose(density_matrix, [[1, 0], [0, 0]])

    c_dtypes = [np.complex64, np.complex128]

    @pytest.mark.parametrize("c_dtype", c_dtypes)
    @pytest.mark.parametrize("check", check_state)
    @pytest.mark.parametrize("wires", wires_list)
    def test_density_matrix_c_dtype(self, wires, check, tol, c_dtype):
        """Test different complex dtype."""

        dev = qml.device("default.qubit", wires=2, c_dtype=c_dtype)

        @qml.qnode(dev, diff_method=None)
        def circuit(x):
            qml.IsingXX(x, wires=[0, 1])
            return qml.state()

        density_matrix = qml.qinfo.reduced_dm(circuit, wires=wires)(0.5)
        assert density_matrix.dtype == c_dtype
