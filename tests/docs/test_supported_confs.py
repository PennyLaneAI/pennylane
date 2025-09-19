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
match the supported configurations in the code.

A configuration is specified by:
    1. The quantum device, e.g. "default.qubit"
    2. The interface, e.g. "jax"
    3. The differentiation method, e.g. "parameter-shift"
    4. The return value of the QNode, e.g. qml.expval() or qml.probs()
    5. The number of shots, either None or an integer > 0

A configuration is supported if gradients can be computed for the
QNode without an exception being raised."""
# pylint: disable=too-many-arguments
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.exceptions import DeviceError, QuantumFunctionError

pytestmark = pytest.mark.all_interfaces

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.autograd.functional")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

interfaces = [None, "autograd", "jax", "torch"]
diff_interfaces = ["autograd", "jax", "torch"]
shots_list = [None, 100]

# Each of these tuples contain:
#   1. The argument 'wires' to pass to qml.device()
#   2. A list of the wire labels
#   3. The wire to measure in a single-qubit measurement process
#   4. The wires(s) to measure in a multi-qubit measurement process
wire_specs_list = [
    (1, [0], 0, 0),
    (2, [0, 1], 0, [0, 1]),
    # (["a"], ["a"], "a", "a"),
    # (["a", "b"], ["a", "b"], "a", ["a", "b"]),
]

diff_methods = [
    "device",
    "backprop",
    "adjoint",
    "parameter-shift",
    "finite-diff",
    "spsa",
    "hadamard",
]
return_types = [
    "StateCost",  # scalar cost function of the state
    "StateVector",  # the state directly
    "DensityMatrix",
    "Probability",
    "Sample",
    "Expectation",
    "Hermitian",  # non-standard variant of "Expectation" values
    "Projector",  # non-standard variant of "Expectation" values
    "Variance",
    "VnEntropy",
    "MutualInfo",
]

grad_return_cases = [
    "StateCost",
    "DensityMatrix",
    "Expectation",
    "Hermitian",
    "Projector",
    "Variance",
    "VnEntropy",
    "MutualInfo",
]


def get_qnode(interface, diff_method, return_type, shots, wire_specs):
    """Return a QNode with the given attributes.

    This function includes a general QNode definition that is used to create a
    specific QNode using the provided parameters:

    * input interface,
    * differentiation method,
    * return type and
    * the number of shots for the device.
    * the wire specifications, see the comment above
    """
    device_wires, wire_labels, single_meas_wire, multi_meas_wire = wire_specs

    dev = qml.device("default.qubit", wires=device_wires)

    # pylint: disable=too-many-return-statements
    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(x):
        for i, wire_label in enumerate(wire_labels):
            qml.Hadamard(wires=wire_label)
            qml.RX(x[i], wires=wire_label)

        if return_type == "StateCost":
            return qml.state()
        if return_type == "StateVector":
            return qml.state()
        if return_type == "DensityMatrix":
            return qml.density_matrix(wires=single_meas_wire)
        if return_type == "Probability":
            return qml.probs(wires=multi_meas_wire)
        if return_type == "Sample":
            return qml.sample(wires=multi_meas_wire)
        if return_type == "Expectation":
            return qml.expval(qml.PauliZ(wires=single_meas_wire))
        if return_type == "Hermitian":
            return qml.expval(
                qml.Hermitian(
                    np.array([[1.0, 0.0], [0.0, -1.0]], requires_grad=False), wires=single_meas_wire
                )
            )
        if return_type == "Projector":
            return qml.expval(qml.Projector(np.array([1]), wires=single_meas_wire))
        if return_type == "Variance":
            return qml.var(qml.PauliZ(wires=single_meas_wire))
        if return_type == "VnEntropy":
            return qml.vn_entropy(wires=single_meas_wire)
        if return_type == "MutualInfo":
            wires1 = [w for w in wire_labels if w != single_meas_wire]
            return qml.mutual_info(wires0=[single_meas_wire], wires1=wires1)
        return None

    return circuit


def get_variable(interface, wire_specs, complex=False):
    """Return an interface-specific trainable variable"""
    num_wires = len(wire_specs[1])

    if interface is None:
        return np.array([0.1] * num_wires)
    if interface == "autograd":
        return np.array([0.1] * num_wires, requires_grad=True)
    if interface == "jax":
        # complex dtype is required for JAX when holomorphic gradient is used
        return jnp.array([0.1] * num_wires, dtype=np.complex128 if complex else None)
    if interface == "torch":
        # complex dtype is required for torch when the gradients have non-zero
        # imaginary parts, otherwise they will be ignored
        return torch.tensor(
            [0.1] * num_wires, requires_grad=True, dtype=torch.complex64 if complex else None
        )
    return None


def get_state_cost_fn(circuit):
    """Get the scalar cost function dependent on the output state"""

    def cost_fn(x):
        res = circuit(x)
        probs = qml.math.abs(res) ** 2
        return probs[0]

    return cost_fn


def get_density_matrix_cost_fn(circuit):
    """Get the scalar cost function dependent on the output density matrix"""

    def cost_fn(x):
        res = circuit(x)
        probs = qml.math.abs(res) ** 2
        return probs[0][0]

    return cost_fn


# pylint: disable=too-many-return-statements
def compute_gradient(x, interface, circuit, return_type, complex=False):
    """Return an interface-specific gradient or jacobian using the
    provided parameters:

    * input weights,
    * interface,
    * the QNode to execute,
    * return type and
    * whether output is complex

    For the StateCost and DensityMatrix return types, this computes the
    gradient of a scalar cost function dependent on the state instead of
    the jacobian of the state directly. The latter is tested by the
    StateVector return type.
    """
    if return_type == "StateCost":
        cost_fn = get_state_cost_fn(circuit)
    elif return_type == "DensityMatrix":
        cost_fn = get_density_matrix_cost_fn(circuit)
    else:
        cost_fn = circuit

    if interface == "autograd":
        if return_type in grad_return_cases:
            return qml.grad(cost_fn)(x)
        return qml.jacobian(cost_fn)(x)
    if interface == "jax":
        if return_type in grad_return_cases:
            return jax.grad(cost_fn)(x)
        return jax.jacrev(cost_fn, holomorphic=complex)(x)
    if interface == "torch":
        if return_type in grad_return_cases:
            res = cost_fn(x)
            res.backward()
            return x.grad
        return F.jacobian(circuit, (x,))
    return None


class TestSupportedConfs:
    """Test that the supported configurations in the documentation
    matches the supported configurations in the code.

    Also test that the supported configurations do not raise any errors and
    the unsupported configurations raise the expected errors or otherwise
    behave as expected.

    These tests do not test for correctness."""

    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("return_type", return_types)
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_device(self, interface, return_type, shots, wire_specs):
        """Test diff_method=device raises an error for all interfaces for default.qubit"""
        msg = "does not support device with requested circuit"

        with pytest.raises(QuantumFunctionError, match=msg):
            get_qnode(interface, "device", return_type, shots, wire_specs)

    @pytest.mark.parametrize(
        "diff_method", ["adjoint", "parameter-shift", "finite-diff", "spsa", "hadamard", "backprop"]
    )
    @pytest.mark.parametrize("return_type", return_types)
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_none_all(self, diff_method, return_type, shots, wire_specs):
        """Test interface=None and diff_method in [adjoint, parameter-shift,
        finite-diff, spsa, hadamard] has a working forward pass"""
        circuit = get_qnode(None, diff_method, return_type, shots, wire_specs)
        x = get_variable(None, wire_specs)

        msg = None
        error_type = DeviceError
        if diff_method == "adjoint" and (shots is not None or return_type == "Sample"):
            error_type = QuantumFunctionError
            msg = f"does not support {diff_method} with requested circuit"
        elif diff_method == "backprop" and shots:
            error_type = QuantumFunctionError
            msg = f"does not support {diff_method} with requested circuit"
        elif not shots and return_type == "Sample":
            msg = "not accepted for analytic simulation"
        elif shots and return_type in (
            "VnEntropy",
            "MutualInfo",
            "DensityMatrix",
            "StateCost",
            "StateVector",
        ):
            msg = "not accepted with finite shots on"

        if msg is not None:
            with pytest.raises(error_type, match=msg):
                circuit(x)
        else:
            circuit(x)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type",
        [
            "StateCost",
            "DensityMatrix",
            "Probability",
            "Expectation",
            "Hermitian",
            "Projector",
            "Variance",
            "VnEntropy",
            "MutualInfo",
        ],
    )
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_backprop_none_shots(self, interface, return_type, wire_specs):
        """Test diff_method=backprop works for all interfaces when shots=None"""
        circuit = get_qnode(interface, "backprop", return_type, None, wire_specs)
        x = get_variable(interface, wire_specs)
        compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", return_types)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_backprop_finite_shots(self, interface, return_type, wire_specs):
        """Test diff_method=backprop fails for all interfaces when shots>0"""
        msg = "does not support backprop with requested circuit."

        x = get_variable(None, wire_specs)
        qn = get_qnode(interface, "backprop", return_type, 100, wire_specs)
        with pytest.raises(QuantumFunctionError, match=msg):
            qn(x)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type",
        ["StateCost", "DensityMatrix", "Probability", "Variance", "VnEntropy", "MutualInfo"],
    )
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_adjoint_nonexp(self, interface, return_type, shots, wire_specs):
        """Test diff_method=adjoint raises an error for non-"Expectation"
        measurements for all interfaces"""

        circuit = get_qnode(interface, "adjoint", return_type, shots, wire_specs)
        x = get_variable(interface, wire_specs, complex=True)

        if shots is not None:
            with pytest.raises(QuantumFunctionError):
                compute_gradient(x, interface, circuit, return_type)
        else:
            compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", ["Expectation", "Hermitian", "Projector"])
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_adjoint_exp(self, interface, return_type, shots, wire_specs):
        """Test diff_method=adjoint works for "Expectation" measurements for all interfaces"""
        if shots is None:
            # test that everything runs
            # correctness is already tested in other test files
            circuit = get_qnode(interface, "adjoint", return_type, shots, wire_specs)
            x = get_variable(interface, wire_specs)
            compute_gradient(x, interface, circuit, return_type)
        else:
            circuit = get_qnode(interface, "adjoint", return_type, shots, wire_specs)
            x = get_variable(interface, wire_specs)
            with pytest.raises(
                QuantumFunctionError,
                match="does not support adjoint with requested circuit",
            ):
                compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type",
        ["Probability", "Expectation", "Hermitian", "Projector", "Variance"],
    )
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_paramshift_nonstate(self, interface, return_type, shots, wire_specs):
        """Test diff_method=parameter-shift works for all interfaces and
        return_types except State and DensityMatrix"""

        # correctness is already tested in other test files
        circuit = get_qnode(interface, "parameter-shift", return_type, shots, wire_specs)
        x = get_variable(interface, wire_specs)
        compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type", ["StateCost", "StateVector", "DensityMatrix", "VnEntropy", "MutualInfo"]
    )
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_paramshift_state(self, interface, return_type, shots, wire_specs):
        """Test diff_method=parameter-shift fails for all interfaces and
        the return_types State and DensityMatrix"""
        msg = (
            "Computing the gradient of circuits that return the state with the "
            "parameter-shift rule gradient transform is not supported."
        )
        complex = return_type == "StateVector"

        # with pytest.raises(ValueError, match=msg):
        circuit = get_qnode(interface, "parameter-shift", return_type, shots, wire_specs)
        x = get_variable(interface, wire_specs, complex=complex)
        if shots is not None and interface != "jax":
            with pytest.raises(DeviceError, match="not accepted with finite shots"):
                compute_gradient(x, interface, circuit, return_type, complex=complex)
        else:
            if interface == "torch" and return_type == "StateVector":
                pytest.xfail(reason="see pytorch/pytorch/issues/94397")
            with pytest.raises(ValueError, match=msg):
                compute_gradient(x, interface, circuit, return_type, complex=complex)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type",
        [
            "Probability",
            "Expectation",
            "Hermitian",
            "Projector",
            "Variance",
            "VnEntropy",
            "MutualInfo",
        ],
    )
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    @pytest.mark.parametrize("diff_method", ["finite-diff", "spsa"])
    def test_all_finitediff_nonstate(self, interface, return_type, shots, wire_specs, diff_method):
        """Test diff_method in ['finite-diff', 'spsa'] works for all interfaces and
        return_types except State and DensityMatrix"""

        # correctness is already tested in other test files
        circuit = get_qnode(interface, diff_method, return_type, shots, wire_specs)
        x = get_variable(interface, wire_specs)
        if shots is not None and return_type in ("VnEntropy", "MutualInfo"):
            with pytest.raises(DeviceError, match="not accepted with finite shots"):
                compute_gradient(x, interface, circuit, return_type)
        else:
            compute_gradient(x, interface, circuit, return_type)

    @pytest.mark.xfail(reason="Finite diff for state measurement is being addressed.")
    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("return_type", ["StateCost", "StateVector", "DensityMatrix"])
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    @pytest.mark.parametrize("diff_method", ["finite-diff", "spsa"])
    def test_all_finitediff_state(self, interface, return_type, shots, wire_specs, diff_method):
        """Test diff_method in ['finite-diff', 'spsa'] fails for all interfaces and
        the return_types State and DensityMatrix."""

        # this error message is a bit cryptic, but it's consistent across
        # all the interfaces
        msg = "state\\(wires=\\[0?\\]\\)\\ is\\ not\\ in\\ list"

        complex = return_type == "StateVector"

        with pytest.raises(ValueError, match=msg):
            circuit = get_qnode(interface, diff_method, return_type, shots, wire_specs)
            x = get_variable(interface, wire_specs, complex=complex)

            if shots is not None:
                with pytest.warns(UserWarning, match="unaffected by sampling"):
                    compute_gradient(x, interface, circuit, return_type, complex=complex)
            else:
                compute_gradient(x, interface, circuit, return_type, complex=complex)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "diff_method", ["backprop", "adjoint", "parameter-shift", "finite-diff", "spsa", "hadamard"]
    )
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_sample_none_shots(self, interface, diff_method, wire_specs):
        """Test "Sample" measurement fails for all interfaces and diff_methods
        when shots=None"""
        if diff_method in {"adjoint"}:
            error_type = QuantumFunctionError
            msg = "does not support adjoint with requested circuit"
        else:
            error_type = DeviceError
            msg = "not accepted for analytic simulation"

        with pytest.raises(error_type, match=msg):
            circuit = get_qnode(interface, diff_method, "Sample", None, wire_specs)
            x = get_variable(interface, wire_specs)
            circuit(x)

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "spsa"])
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_sample_finite_shots(self, interface, diff_method, wire_specs):
        """Test "Sample" measurement works for all interfaces when shots>0 (but the results may be incorrect)"""
        # test that forward pass still works
        circuit = get_qnode(interface, diff_method, "Sample", 100, wire_specs)
        x = get_variable(interface, wire_specs)
        circuit(x)

    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_autograd_state_backprop(self, wire_specs):
        """Test gradient of state directly fails for autograd interface"""
        msg = "cannot reshape array of size .*"

        with pytest.raises(ValueError, match=msg):
            circuit = get_qnode("autograd", "backprop", "StateVector", None, wire_specs)
            x = get_variable("autograd", wire_specs)
            compute_gradient(x, "autograd", circuit, "StateVector")

    @pytest.mark.parametrize("interface", ["jax", "torch"])
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    def test_all_state_backprop(self, interface, wire_specs):
        """Test gradient of state directly succeeds for non-autograd interfaces"""
        circuit = get_qnode(interface, "backprop", "StateVector", None, wire_specs)
        x = get_variable(interface, wire_specs, complex=True)
        if interface == "torch":
            pytest.xfail(reason="see pytorch/pytorch/issues/94397")
        compute_gradient(x, interface, circuit, "StateVector", complex=True)

    wire_specs_list = [
        (2, [0], 0, 0),
        (3, [0, 1], 0, [0, 1]),
    ]

    @pytest.mark.parametrize("interface", diff_interfaces)
    @pytest.mark.parametrize(
        "return_type",
        [
            "Probability",
            "Expectation",
            "Hermitian",
            "Projector",
            "Variance",
            "VnEntropy",
            "MutualInfo",
        ],
    )
    @pytest.mark.parametrize("shots", shots_list)
    @pytest.mark.parametrize("wire_specs", wire_specs_list)
    @pytest.mark.parametrize("diff_method", ["hadamard"])
    def test_all_hadamard_nonstate_non_var(
        self, interface, return_type, shots, wire_specs, diff_method
    ):
        """Test diff_method "hadamard" works for all interfaces and
        return_types except State, DensityMatrix and Var"""
        # correctness is already tested in other test files
        circuit = get_qnode(interface, diff_method, return_type, shots, wire_specs)
        x = get_variable(interface, wire_specs)
        if return_type in ("VnEntropy", "MutualInfo"):
            if shots and interface != "jax":
                err_cls = DeviceError
                msg = "not accepted with finite shots"
            else:
                err_cls = ValueError
                msg = "Computing the gradient of circuits that return the state with the Hadamard test gradient transform is not supported"
            with pytest.raises(err_cls, match=msg):
                compute_gradient(x, interface, circuit, return_type)
        elif return_type == "Variance":
            with pytest.raises(
                ValueError,
                match=(
                    "Computing the gradient of variances with the Hadamard test "
                    "gradient transform is not supported."
                ),
            ):
                compute_gradient(x, interface, circuit, return_type)
        else:
            compute_gradient(x, interface, circuit, return_type)
