# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the mutual_info module"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.interfaces import INTERFACE_MAP
from pennylane.measurements import Shots


@pytest.mark.parametrize("shots, shape", [(None, ()), (10, ()), ([1, 10], ((), ()))])
def test_shape(shots, shape):
    """Test that the shape is correct."""
    dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
    res = qml.mutual_info(wires0=[0], wires1=[1])
    assert res.shape(dev, Shots(shots)) == shape


class TestIntegration:
    """Tests for the mutual information functions"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tf", "torch"])
    @pytest.mark.parametrize(
        "state, expected",
        [
            ([1.0, 0.0, 0.0, 0.0], 0),
            ([qml.math.sqrt(2) / 2, 0.0, qml.math.sqrt(2) / 2, 0.0], 0),
            ([qml.math.sqrt(2) / 2, 0.0, 0.0, qml.math.sqrt(2) / 2], 2 * qml.math.log(2)),
            (qml.math.ones(4) * 0.5, 0.0),
        ],
    )
    def test_mutual_info_output(self, interface, state, expected):
        """Test the output of qml.mutual_info"""
        dev = qml.device("default.qubit.legacy", wires=4)

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.StatePrep(state, wires=[0, 1])
            return qml.mutual_info(wires0=[0, 2], wires1=[1, 3])

        res = circuit()
        new_res = qml.mutual_info(wires0=[0, 2], wires1=[1, 3]).process_state(
            state=circuit.device.state, wire_order=circuit.device.wires
        )
        assert np.allclose(res, expected, atol=1e-6)
        assert np.allclose(new_res, expected, atol=1e-6)
        assert INTERFACE_MAP.get(qml.math.get_interface(new_res)) == interface
        assert res.dtype == new_res.dtype  # pylint: disable=no-member

    def test_shot_vec_error(self):
        """Test an error is raised when using shot vectors with mutual_info."""
        dev = qml.device("default.qubit.legacy", wires=2, shots=[1, 10, 10, 1000])

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        with pytest.raises(
            NotImplementedError, match="mutual information is not supported with shot vectors"
        ):
            circuit(0.5)

    diff_methods = ["backprop", "finite-diff"]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", ["default.qubit.legacy", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 8))
    def test_qnode_state(self, device, interface, params):
        """Test that the mutual information transform works for QNodes by comparing
        against analytic values"""
        dev = qml.device(device, wires=2)

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = qml.qinfo.mutual_info(circuit, wires0=[0], wires1=[1])(params)

        # compare transform results with analytic values
        expected = -2 * np.cos(params / 2) ** 2 * np.log(
            np.cos(params / 2) ** 2 + 1e-10
        ) - 2 * np.sin(params / 2) ** 2 * np.log(np.sin(params / 2) ** 2 + 1e-10)

        assert np.allclose(actual, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", ["default.qubit.legacy", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize("params", zip(np.linspace(0, np.pi, 8), np.linspace(0, 2 * np.pi, 8)))
    def test_qnode_mutual_info(self, device, interface, params):
        """Test that the measurement process for mutual information works for QNodes
        by comparing against the mutual information transform"""
        dev = qml.device(device, wires=2)

        params = qml.math.asarray(np.array(params), like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit_mutual_info(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        @qml.qnode(dev, interface=interface)
        def circuit_state(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = circuit_mutual_info(params)

        # compare measurement results with transform results
        expected = qml.qinfo.mutual_info(circuit_state, wires0=[0], wires1=[1])(params)

        assert np.allclose(actual, expected)

    @pytest.mark.parametrize("device", ["default.qubit.legacy", "default.mixed", "lightning.qubit"])
    def test_mutual_info_wire_labels(self, device):
        """Test that mutual_info is correct with custom wire labels"""
        param = np.array([0.678, 1.234])
        wires = ["a", 8]
        dev = qml.device(device, wires=wires)

        @qml.qnode(dev)
        def circuit(param):
            qml.RY(param, wires=wires[0])
            qml.CNOT(wires=wires)
            return qml.state()

        actual = qml.qinfo.mutual_info(circuit, wires0=[wires[0]], wires1=[wires[1]])(param)

        # compare transform results with analytic values
        expected = -2 * np.cos(param / 2) ** 2 * np.log(np.cos(param / 2) ** 2) - 2 * np.sin(
            param / 2
        ) ** 2 * np.log(np.sin(param / 2) ** 2)

        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 8))
    def test_qnode_state_jax_jit(self, params):
        """Test that the mutual information transform works for QNodes by comparing
        against analytic values, for the JAX-jit interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.legacy", wires=2)

        params = jnp.array(params)

        @qml.qnode(dev, interface="jax-jit")
        def circuit(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = jax.jit(qml.qinfo.mutual_info(circuit, wires0=[0], wires1=[1]))(params)

        # compare transform results with analytic values
        expected = -2 * jnp.cos(params / 2) ** 2 * jnp.log(
            jnp.cos(params / 2) ** 2 + 1e-10
        ) - 2 * jnp.sin(params / 2) ** 2 * jnp.log(jnp.sin(params / 2) ** 2 + 1e-10)

        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", zip(np.linspace(0, np.pi, 8), np.linspace(0, 2 * np.pi, 8)))
    @pytest.mark.parametrize("interface", ["jax-jit"])
    def test_qnode_mutual_info_jax_jit(self, params, interface):
        """Test that the measurement process for mutual information works for QNodes
        by comparing against the mutual information transform, for the JAX-jit interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.legacy", wires=2)

        params = jnp.array(params)

        @qml.qnode(dev, interface=interface)
        def circuit_mutual_info(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        @qml.qnode(dev, interface="jax-jit")
        def circuit_state(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        actual = jax.jit(circuit_mutual_info)(params)

        # compare measurement results with transform results
        expected = jax.jit(qml.qinfo.mutual_info(circuit_state, wires0=[0], wires1=[1]))(params)

        assert np.allclose(actual, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["auto", "autograd"])
    def test_qnode_grad(self, param, diff_method, interface):
        """Test that the gradient of mutual information works for QNodes
        with the autograd interface"""
        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = np.sin(param) * (
                np.log(np.cos(param / 2) ** 2) - np.log(np.sin(param / 2) ** 2)
            )

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = qml.grad(circuit)(param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["jax"])
    def test_qnode_grad_jax(self, param, diff_method, interface):
        """Test that the gradient of mutual information works for QNodes
        with the JAX interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.legacy", wires=2)

        param = jnp.array(param)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = jnp.sin(param) * (
                jnp.log(jnp.cos(param / 2) ** 2) - jnp.log(jnp.sin(param / 2) ** 2)
            )

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = jax.grad(circuit)(param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.jax
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["jax-jit"])
    def test_qnode_grad_jax_jit(self, param, diff_method, interface):
        """Test that the gradient of mutual information works for QNodes
        with the JAX-jit interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.legacy", wires=2)

        param = jnp.array(param)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = jnp.sin(param) * (
                jnp.log(jnp.cos(param / 2) ** 2) - jnp.log(jnp.sin(param / 2) ** 2)
            )

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = jax.jit(jax.grad(circuit))(param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.tf
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["tf"])
    def test_qnode_grad_tf(self, param, diff_method, interface):
        """Test that the gradient of mutual information works for QNodes
        with the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit.legacy", wires=2)

        param = tf.Variable(param)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = np.sin(param) * (
                np.log(np.cos(param / 2) ** 2) - np.log(np.sin(param / 2) ** 2)
            )

        with tf.GradientTape() as tape:
            out = circuit(param)

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = tape.gradient(out, param)
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.torch
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["torch"])
    def test_qnode_grad_torch(self, param, diff_method, interface):
        """Test that the gradient of mutual information works for QNodes
        with the torch interface"""
        import torch

        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def circuit(param):
            qml.RY(param, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        if param == 0:
            # we don't allow gradients to flow through the discontinuity at 0
            expected = 0
        else:
            expected = np.sin(param) * (
                np.log(np.cos(param / 2) ** 2) - np.log(np.sin(param / 2) ** 2)
            )

        param = torch.tensor(param, requires_grad=True)
        out = circuit(param)
        out.backward()  # pylint: disable=no-member

        # higher tolerance for finite-diff method
        tol = 1e-8 if diff_method == "backprop" else 1e-5

        actual = param.grad
        assert np.allclose(actual, expected, atol=tol)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0.0, 0.0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_subsystem_overlap_error(self, interface, params):
        """Test that an error is raised when the subsystems overlap"""
        dev = qml.device("default.qubit.legacy", wires=3)

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            return qml.mutual_info(wires0=[0, 1], wires1=[1, 2])

        msg = "Subsystems for computing mutual information must not overlap"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            circuit(params)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "tensorflow", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0.0, 0.0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_custom_wire_labels_error(self, interface, params):
        """Tests that an error is raised when mutual information is measured
        with custom wire labels"""
        dev = qml.device("default.qubit.legacy", wires=["a", "b"])

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires="a")
            qml.RY(params[1], wires="b")
            qml.CNOT(wires=["a", "b"])
            return qml.mutual_info(wires0=["a"], wires1=["b"])

        msg = "Returning the mutual information is not supported when using custom wire labels"
        with pytest.raises(qml.QuantumFunctionError, match=msg):
            circuit(params)
