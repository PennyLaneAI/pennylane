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
"""Test that the supported configurations in the documentation
matches the supported configurations in the code"""
import pytest
import re

import pennylane as qml
from pennylane import numpy as np
from pennylane import QuantumFunctionError
from pennylane.measurements import State, Probability, Expectation, Variance, Sample

pytestmark = pytest.mark.all_interfaces

tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.autograd.functional")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

devices = ["default.qubit"]
interfaces = [None, "autograd", "jax", "tf", "torch"]
diff_interfaces = ["autograd", "jax", "tf", "torch"]
shotss = [None, 100]

diff_methods = ["device", "backprop", "adjoint", "parameter-shift", "finite-diff"]
return_types = [
    State,  # scalar cost function of the state
    "StateVector",  # the state directly
    "DensityMatrix",
    Probability,
    Sample,
    Expectation,
    "Hermitian",  # non-standard variant of expectation values
    "Projector",  # non-standard variant of expectation values
    Variance,
]

grad_types = [Expectation, "Hermitian", "Projector", Variance]


def get_qnode(interface, diff_method, return_type, shots):
    """Return a QNode with the given attributes"""
    dev = qml.device("default.qubit", wires=1, shots=shots)

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RX(x[0], wires=0)
        if return_type == State:
            return qml.state()
        elif return_type == "StateVector":
            return qml.state()
        elif return_type == "DensityMatrix":
            return qml.density_matrix(wires=0)
        elif return_type == Probability:
            return qml.probs(wires=0)
        elif return_type == Sample:
            return qml.sample(wires=0)
        elif return_type == Expectation:
            return qml.expval(qml.PauliZ(wires=0))
        elif return_type == "Hermitian":
            return qml.expval(qml.Hermitian(np.array([[1.0, 0.0], [0.0, -1.0]]), wires=0))
        elif return_type == "Projector":
            return qml.expval(qml.Projector(np.array([1]), wires=0))
        elif return_type == Variance:
            return qml.var(qml.PauliZ(wires=0))

    return circuit


def get_variable(interface, complex=False):
    """Return an interface-specific trainable variable"""
    if interface is None:
        return np.array([0.1])
    elif interface == "autograd":
        return np.array([0.1], requires_grad=True)
    elif interface == "jax":
        return jnp.array([0.1], dtype=np.complex64 if complex else None)
    elif interface == "tf":
        return tf.Variable([0.1], trainable=True, dtype=np.complex64 if complex else None)
    elif interface == "torch":
        return torch.tensor([0.1], requires_grad=True, dtype=torch.complex64 if complex else None)


def compute_gradient(x, interface, circuit, return_type, complex=False):
    """Return an interface-specific gradient or jacobian"""
    if interface == "autograd":
        if return_type in grad_types:
            return qml.grad(circuit)(x)
        elif return_type == State:

            def cost_fn(x):
                res = circuit(x)
                probs = np.abs(res) ** 2
                return probs[0]

            return qml.grad(cost_fn)(x)
        elif return_type == "DensityMatrix":

            def cost_fn(x):
                res = circuit(x)
                probs = np.abs(res) ** 2
                return probs[0][0]

            return qml.grad(cost_fn)(x)
        else:
            return qml.jacobian(circuit)(x)
    elif interface == "jax":
        if return_type in grad_types:
            return jax.grad(circuit, argnums=0)(x)
        elif return_type == State:

            def cost_fn(x):
                res = circuit(x)
                probs = jnp.abs(res) ** 2
                return probs[0]

            # compute the gradient of the scalar cost function instead
            # of the jacobian of the state directly - the latter is a
            # separate test case
            return jax.grad(cost_fn)(x)
        elif return_type == "DensityMatrix":

            def cost_fn(x):
                res = circuit(x)
                probs = jnp.abs(res) ** 2
                return probs[0][0]

            # compute the gradient of the scalar cost function instead
            # of the jacobian of the state directly - the latter is a
            # separate test case
            return jax.grad(cost_fn)(x)
        else:
            return jax.jacrev(circuit, holomorphic=complex)(x)
    elif interface == "tf":
        with tf.GradientTape() as tape:
            out = circuit(x)
        if return_type in grad_types:
            return tape.gradient(out, [x])
        elif return_type == State:

            def cost_fn(x):
                res = circuit(x)
                probs = tf.math.abs(res) ** 2
                return probs[0]

            with tf.GradientTape() as tape:
                out = cost_fn(x)

            # compute the gradient of the scalar cost function instead
            # of the jacobian of the state directly - the latter is a
            # separate test case
            return tape.gradient(out, [x])
        elif return_type == "DensityMatrix":

            def cost_fn(x):
                res = circuit(x)
                probs = tf.math.abs(res) ** 2
                return probs[0][0]

            with tf.GradientTape() as tape:
                out = cost_fn(x)

            # compute the gradient of the scalar cost function instead
            # of the jacobian of the state directly - the latter is a
            # separate test case
            return tape.gradient(out, [x])
        else:
            return tape.jacobian(out, [x])
    elif interface == "torch":
        if return_type in grad_types:
            res = circuit(x)
            res.backward()
            return x.grad
        elif return_type == State:

            def cost_fn(x):
                res = circuit(x)
                probs = torch.abs(res) ** 2
                return probs[0]

            # compute the gradient of the scalar cost function instead
            # of the jacobian of the state directly - the latter is a
            # separate test case
            res = cost_fn(x)
            res.backward()
            return x.grad
        elif return_type == "DensityMatrix":

            def cost_fn(x):
                res = circuit(x)
                probs = torch.abs(res) ** 2
                return probs[0][0]

            # compute the gradient of the scalar cost function instead
            # of the jacobian of the state directly - the latter is a
            # separate test case
            res = cost_fn(x)
            res.backward()
            return x.grad
        else:
            return F.jacobian(circuit, (x,))


class TestSupportedConfs:
    """Test that the supported configurations in the documentation
    matches the supported configurations in the code"""

    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("return_type", return_types)
    @pytest.mark.parametrize("shots", shotss)
    def test_all_device(self, interface, return_type, shots):
        """Test diff_method=device raises an error for all interfaces for default.qubit"""
        msg = (
            "The default.qubit device does not provide a native "
            "method for computing the jacobian."
        )

        with pytest.raises(QuantumFunctionError, match=msg):
            circuit = get_qnode(interface, "device", return_type, shots)

    @pytest.mark.parametrize("return_type", return_types)
    def test_none_backprop(self, return_type):
        """Test interface=None and diff_method=backprop raises an error"""
        msg = (
            "Device default.qubit only supports diff_method='backprop' when "
            "using the ['tf', 'torch', 'autograd', 'jax'] interfaces."
        )
        msg = re.escape(msg)

        with pytest.raises(QuantumFunctionError, match=msg):
            circuit = get_qnode(None, "backprop", return_type, None)

    @pytest.mark.parametrize("diff_method", ["adjoint", "parameter-shift", "finite-diff"])
    @pytest.mark.parametrize("return_type", return_types)
    @pytest.mark.parametrize("shots", shotss)
    def test_none_all(self, diff_method, return_type, shots):
        """Test interface=None and diff_method in [adjoint, parameter-shift,
        finite-diff] has a working forward pass"""
        warn_msg = (
            "Requested adjoint differentiation to be computed with finite shots. "
            "Adjoint differentiation always calculated exactly."
        )

        if diff_method == "adjoint" and shots is not None:
            # this warning is still raised in the forward pass
            with pytest.warns(UserWarning, match=warn_msg):
                circuit = get_qnode(None, diff_method, return_type, shots)
        else:
            circuit = get_qnode(None, diff_method, return_type, shots)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type",
        [State, "DensityMatrix", Probability, Expectation, "Hermitian", "Projector", Variance],
    )
    def test_all_backprop_none_shots(self, interface, return_type):
        """Test diff_method=backprop works for all interfaces when shots=None"""

        # DensityMatrix doesn't work with torch at the moment
        if interface == "torch" and return_type == "DensityMatrix":
            with pytest.raises(IndexError):
                circuit = get_qnode(interface, "backprop", return_type, None)
                x = get_variable(interface)
                grad = compute_gradient(x, interface, circuit, return_type)
            return

        # correctness is already tested in other test files
        circuit = get_qnode(interface, "backprop", return_type, None)
        x = get_variable(interface)
        grad = compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", return_types)
    def test_all_backprop_finite_shots(self, interface, return_type):
        """Test diff_method=backprop fails for all interfaces when shots>0"""
        msg = "Backpropagation is only supported when shots=None."

        # DensityMatrix doesn't work with torch at the moment
        if interface == "torch" and return_type == "DensityMatrix":
            with pytest.raises(IndexError):
                circuit = get_qnode(interface, "backprop", return_type, None)
                x = get_variable(interface)
                grad = compute_gradient(x, interface, circuit, return_type)
            return

        with pytest.raises(QuantumFunctionError, match=msg):
            circuit = get_qnode(interface, "backprop", return_type, 100)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", [State, "DensityMatrix", Probability, Variance])
    @pytest.mark.parametrize("shots", shotss)
    def test_all_adjoint_nonexp(self, interface, return_type, shots):
        """Test diff_method=adjoint raises an error for non-expectation
        measurements for all interfaces"""
        msg = "Adjoint differentiation method does not support measurement .*"

        warn_msg = (
            "Requested adjoint differentiation to be computed with finite shots. "
            "Adjoint differentiation always calculated exactly."
        )

        with pytest.raises(QuantumFunctionError, match=msg):
            with pytest.warns(UserWarning, match=warn_msg):
                circuit = get_qnode(interface, "adjoint", return_type, shots)
                x = get_variable(interface)
                grad = compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", [Expectation, "Hermitian", "Projector"])
    @pytest.mark.parametrize("shots", shotss)
    def test_all_adjoint_exp(self, interface, return_type, shots):
        """Test diff_method=adjoint works for expectation measurements for all interfaces"""
        warn_msg = (
            "Requested adjoint differentiation to be computed with finite shots. "
            "Adjoint differentiation always calculated exactly."
        )

        # Hermitian doesn't work with torch at the moment
        if interface == "torch" and return_type == "Hermitian":
            with pytest.raises(RuntimeError):
                with pytest.warns(UserWarning, match=warn_msg):
                    circuit = get_qnode(interface, "adjoint", return_type, shots)
                    x = get_variable(interface)
                    grad = compute_gradient(x, interface, circuit, return_type)
            return

        if shots is None:
            # test that everything runs
            # correctness is already tested in other test files
            circuit = get_qnode(interface, "adjoint", return_type, shots)
            x = get_variable(interface)
            grad = compute_gradient(x, interface, circuit, return_type)
        else:
            # test warning is raised when shots > 0
            with pytest.warns(UserWarning, match=warn_msg):
                circuit = get_qnode(interface, "adjoint", return_type, shots)
                x = get_variable(interface)
                grad = compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type", [Probability, Expectation, "Hermitian", "Projector", Variance]
    )
    @pytest.mark.parametrize("shots", shotss)
    def test_all_paramshift_nonstate(self, interface, return_type, shots):
        """Test diff_method=parameter-shift works for all interfaces and
        return_types except State and DensityMatrix"""

        # Hermitian doesn't work with torch at the moment
        if interface == "torch" and return_type == "Hermitian":
            with pytest.raises(RuntimeError):
                circuit = get_qnode(interface, "parameter-shift", return_type, shots)
                x = get_variable(interface)
                grad = compute_gradient(x, interface, circuit, return_type)
            return

        # correctness is already tested in other test files
        circuit = get_qnode(interface, "parameter-shift", return_type, shots)
        x = get_variable(interface)
        grad = compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", [State, "StateVector", "DensityMatrix"])
    @pytest.mark.parametrize("shots", shotss)
    def test_all_paramshift_state(self, interface, return_type, shots):
        """Test diff_method=parameter-shift fails for all interfaces and
        the return_types State and DensityMatrix"""
        msg = "Computing the gradient of circuits that return the state is not supported."
        complex = return_type == "StateVector"

        with pytest.raises(ValueError, match=msg):
            circuit = get_qnode(interface, "parameter-shift", return_type, shots)
            x = get_variable(interface, complex=complex)
            grad = compute_gradient(x, interface, circuit, return_type, complex=complex)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type", [Probability, Expectation, "Hermitian", "Projector", Variance]
    )
    @pytest.mark.parametrize("shots", shotss)
    def test_all_finitediff_nonstate(self, interface, return_type, shots):
        """Test diff_method=finite-diff works for all interfaces and
        return_types except State and DensityMatrix"""

        # Hermitian doesn't work with torch at the moment
        if interface == "torch" and return_type == "Hermitian":
            with pytest.raises(RuntimeError):
                circuit = get_qnode(interface, "parameter-shift", return_type, shots)
                x = get_variable(interface)
                grad = compute_gradient(x, interface, circuit, return_type)
            return

        # correctness is already tested in other test files
        circuit = get_qnode(interface, "finite-diff", return_type, shots)
        x = get_variable(interface)
        grad = compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", [State, "StateVector", "DensityMatrix"])
    @pytest.mark.parametrize("shots", shotss)
    def test_all_finitediff_state(self, interface, return_type, shots):
        """Test diff_method=finite-diff fails for all interfaces and
        the return_types State and DensityMatrix"""

        # this error message is a bit cryptic, but it's consistent across
        # all the interfaces
        msg = "state\\(wires=\\[0?\\]\\)\\ is\\ not\\ in\\ list"

        complex = return_type == "StateVector"

        with pytest.raises(ValueError, match=msg):
            circuit = get_qnode(interface, "finite-diff", return_type, shots)
            x = get_variable(interface, complex=complex)
            grad = compute_gradient(x, interface, circuit, return_type, complex=complex)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff"]
    )
    def test_all_sample_none_shots(self, interface, diff_method):
        """Test sample measurement fails for all interfaces and diff_methods
        when shots=None"""
        msg = (
            "The number of shots has to be explicitly set on the device "
            "when using sample-based measurements."
        )

        with pytest.raises(QuantumFunctionError, match=msg):
            circuit = get_qnode(interface, diff_method, Sample, None)
            x = get_variable(interface)
            circuit(x)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff"])
    def test_all_sample_finite_shots(self, interface, diff_method):
        """Test sample measurement works for all interfaces and diff_methods
        when shots>0 (but the results may be incorrect)"""

        # the only exception is JAX, which fails due to a dtype mismatch
        if interface == "jax":
            msg = "jacrev requires real-valued outputs .*"

            with pytest.raises(TypeError, match=msg):
                circuit = get_qnode(interface, diff_method, Sample, 100)
                x = get_variable(interface)
                grad = compute_gradient(x, interface, circuit, Sample)
        else:
            # should not raise an exception
            circuit = get_qnode(interface, diff_method, Sample, 100)
            x = get_variable(interface)
            grad = compute_gradient(x, interface, circuit, Sample)

        # test that forward pass still works
        circuit = get_qnode(interface, diff_method, Sample, 100)
        x = get_variable(interface)
        circuit(x)

    def test_autograd_state_backprop(self):
        """Test gradient of state directly fails for autograd interface"""
        msg = "cannot reshape array of size 4 into shape (2,1)"
        msg = re.escape(msg)

        with pytest.raises(ValueError, match=msg):
            circuit = get_qnode("autograd", "backprop", "StateVector", None)
            x = get_variable("autograd")
            grad = compute_gradient(x, "autograd", circuit, "StateVector")

    @pytest.mark.parametrize("interface", ["jax", "tf", "torch"])
    def test_all_state_backprop(self, interface):
        """Test gradient of state directly succeeds for non-autograd interfaces"""
        circuit = get_qnode(interface, "backprop", "StateVector", None)
        x = get_variable(interface, complex=True)
        grad = compute_gradient(x, interface, circuit, "StateVector", complex=True)
