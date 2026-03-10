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
import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.measurements.mutual_info import MutualInfoMP
from pennylane.wires import Wires


class TestMutualInfoUnitTests:
    """Tests for the mutual_info function"""

    def test_queue(self):
        """Test that the right measurement class is queued."""

        with qml.queuing.AnnotatedQueue() as q:
            m = qml.mutual_info(wires0=[0], wires1=[1])

        assert q.queue[0] is m
        assert isinstance(q.queue[0], MutualInfoMP)

    @pytest.mark.parametrize("shots, shape", [(None, ()), (10, ())])
    def test_shape(self, shots, shape):
        """Test that the shape is correct."""
        res = qml.mutual_info(wires0=[0], wires1=[1])
        assert res.shape(shots, 3) == shape

    def test_properties(self):
        """Test that the properties are correct."""
        meas = qml.mutual_info(wires0=[0], wires1=[1])
        assert meas.numeric_type == float

    def test_copy(self):
        """Test that the ``__copy__`` method also copies the ``log_base`` information."""
        meas = qml.mutual_info(wires0=[0], wires1=[1], log_base=2)
        meas_copy = copy.copy(meas)
        assert meas_copy.log_base == 2
        assert meas_copy.wires == Wires([0, 1])

    def test_repr(self):
        """Test that the representation includes information about both wires and the log_base"""
        m1 = qml.mutual_info(wires0=[0], wires1=[1])
        assert repr(m1) == "MutualInfo(wires0=[0], wires1=[1], log_base=None)"

    def test_hash(self):
        """Test the hash property includes the log_base property and the separation of the wires into two subsytems."""
        m1 = MutualInfoMP(wires=[Wires(0), Wires(1)], log_base=2)
        m2 = MutualInfoMP(wires=[Wires(0), Wires(1)], log_base=10)
        assert m1.hash != m2.hash

        m3 = MutualInfoMP(wires=[Wires((0, 1)), Wires(2)])
        m4 = MutualInfoMP(wires=[Wires(0), Wires((1, 2))])
        assert m3.hash != m4.hash

    def test_map_wires(self):
        """Test that map_wires works as expected."""
        mapped1 = MutualInfoMP(wires=[Wires([0]), Wires([1])]).map_wires({0: 1, 1: 0})
        assert mapped1.raw_wires == [Wires([1]), Wires([0])]
        qml.assert_equal(mapped1, MutualInfoMP(wires=[Wires([1]), Wires([0])]))

        mapped2 = MutualInfoMP(wires=[Wires(["a", "b"]), Wires(["c"])]).map_wires(
            {"a": 0, "b": 1, "c": 2}
        )
        assert mapped2.raw_wires == [Wires([0, 1]), Wires([2])]
        qml.assert_equal(mapped2, MutualInfoMP(wires=[Wires([0, 1]), Wires([2])]))

    def test_mutual_info_overlapping_wires(self):
        """Test that an error is raised when subsystems overlap."""
        dm = qml.math.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
        wires = qml.wires.Wires(range(2))

        with pytest.raises(
            QuantumFunctionError,
            match="Subsystems for computing mutual information must not overlap.",
        ):
            qml.mutual_info(wires0=[0], wires1=[0, 1]).process_density_matrix(dm, wires)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "autograd"])
    @pytest.mark.parametrize(
        "wires0, wires1, log_base, expected_mutual_info",
        [
            ([0], [1], None, 1.3862943611198906),  # ln(4), natural log
            ([0], [1], 2, 2.0),  # log2(4)
        ],
    )
    def test_process_density_matrix_mutual_info(
        self, interface, wires0, wires1, log_base, expected_mutual_info
    ):  # pylint: disable=too-many-arguments
        """Test mutual information calculation for non-overlapping subsystems."""
        # Define a pure, entangled two-qubit state (|00> + |11>) / sqrt(2)
        dm = qml.math.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            like=interface,
        )

        if interface == "tensorflow":
            dm = qml.math.cast(dm, "float64")

        wires = qml.wires.Wires(range(2))

        mutual_info = qml.mutual_info(
            wires0=wires0, wires1=wires1, log_base=log_base
        ).process_density_matrix(dm, wires)

        # Set tolerance based on interface
        atol = 1.0e-7 if interface == "torch" else 1.0e-8

        assert qml.math.allclose(
            mutual_info, expected_mutual_info, atol=atol
        ), f"Wires0: {wires0}, Wires1: {wires1}, Log base: {log_base}, Mutual Info doesn't match expected value. Got {mutual_info}, expected {expected_mutual_info}"

        # Test if the result is real
        assert qml.math.allclose(
            qml.math.imag(mutual_info), 0, atol=atol
        ), f"Mutual Info should be real, but got imaginary part: {qml.math.imag(mutual_info)}"


class TestIntegration:
    """Tests for the mutual information functions"""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
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
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.StatePrep(state, wires=[0, 1])
            return qml.mutual_info(wires0=[0, 2], wires1=[1, 3])

        res = circuit()
        assert np.allclose(res, expected, atol=1e-6)

    @pytest.mark.parametrize("shots", [1000, [1, 10, 10, 1000]])
    def test_finite_shots_error(self, shots):
        """Test an error is raised when using shot vectors with mutual_info."""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(shots)
        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        with pytest.raises(DeviceError, match="not accepted with finite shots on default.qubit"):
            circuit(0.5)

    diff_methods = ["backprop", "finite-diff"]

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 8))
    def test_qnode_state(self, device, interface, params):
        """Test that the mutual information works for QNodes by comparing
        against analytic values"""
        dev = qml.device(device, wires=2)

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        actual = circuit(params)

        # compare transform results with analytic values
        expected = -2 * np.cos(params / 2) ** 2 * np.log(
            np.cos(params / 2) ** 2 + 1e-10
        ) - 2 * np.sin(params / 2) ** 2 * np.log(np.sin(params / 2) ** 2 + 1e-10)

        assert np.allclose(actual, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
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
            return qml.density_matrix(wires=[0, 1])

        actual = circuit_mutual_info(params)

        # compare measurement results with transform results
        dm = circuit_state(params)
        expected = qml.math.mutual_info(dm, indices0=[0], indices1=[1])

        assert np.allclose(actual, expected)

    @pytest.mark.jax
    @pytest.mark.parametrize("params", np.linspace(0, 2 * np.pi, 8))
    def test_qnode_state_jax_jit(self, params):
        """Test that the mutual information transform works for QNodes by comparing
        against analytic values, for the JAX-jit interface"""
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        params = jnp.array(params)

        @qml.qnode(dev, interface="jax-jit")
        def circuit(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        actual = circuit(params)

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

        dev = qml.device("default.qubit", wires=2)

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
            return qml.density_matrix(wires=[0, 1])

        actual = jax.jit(circuit_mutual_info)(params)

        # compare measurement results with transform results
        dm = circuit_state(params)
        expected = qml.math.mutual_info(dm, indices0=[0], indices1=[1])

        assert np.allclose(actual, expected)

    @pytest.mark.autograd
    @pytest.mark.parametrize("param", np.linspace(0, 2 * np.pi, 16))
    @pytest.mark.parametrize("diff_method", diff_methods)
    @pytest.mark.parametrize("interface", ["auto", "autograd"])
    def test_qnode_grad(self, param, diff_method, interface):
        """Test that the gradient of mutual information works for QNodes
        with the autograd interface"""
        dev = qml.device("default.qubit", wires=2)

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

        dev = qml.device("default.qubit", wires=2)

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

        dev = qml.device("default.qubit", wires=2)

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

        dev = qml.device("default.qubit", wires=2)

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

        dev = qml.device("default.qubit", wires=2)

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
    @pytest.mark.parametrize("device", ["default.qubit", "default.mixed", "lightning.qubit"])
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize(
        "params", [np.array([0.0, 0.0]), np.array([0.3, 0.4]), np.array([0.6, 0.8])]
    )
    def test_subsystem_overlap_error(self, device, interface, params):
        """Test that an error is raised when the subsystems overlap"""
        dev = qml.device(device, wires=3)

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            return qml.mutual_info(wires0=[0, 1], wires1=[1, 2])

        msg = "Subsystems for computing mutual information must not overlap"
        with pytest.raises(QuantumFunctionError, match=msg):
            circuit(params)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "jax", "torch"])
    @pytest.mark.parametrize("params", [np.array([0.0, 0.0]), np.array([0.3, 0.4])])
    def test_custom_wire_labels_works(self, interface, params):
        """Tests that no error is raised when mutual information is measured
        with custom wire labels"""
        dev = qml.device("default.qubit", wires=["a", "b"])

        params = qml.math.asarray(params, like=interface)

        @qml.qnode(dev, interface=interface)
        def circuit(params):
            qml.RY(params[0], wires="a")
            qml.RY(params[1], wires="b")
            qml.CNOT(wires=["a", "b"])
            return qml.mutual_info(wires0=["a"], wires1=["b"])

        @qml.qnode(dev, interface=interface)
        def circuit_expected(params):
            qml.RY(params[0], wires="a")
            qml.RY(params[1], wires="b")
            qml.CNOT(wires=["a", "b"])
            return qml.density_matrix(wires=["a", "b"])

        actual = circuit(params)

        dm = circuit_expected(params)
        expected = qml.math.mutual_info(dm, indices0=[0], indices1=[1])

        assert np.allclose(actual, expected)
