# Copyright 2019 Xanadu Quantum Technologies Inc.

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
Unit tests and integration tests for the :mod:`pennylane.plugin.Tensornet.tf` device.
"""
import cmath
# pylint: disable=protected-access,cell-var-from-loop
import math

import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.qnode_new import qnode
from itertools import product


try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        import tensorflow.contrib.eager as tfe
        tf.enable_eager_execution()
        Variable = tfe.Variable
    else:
        from tensorflow import Variable

except ImportError as e:
    pass

tensornetwork = pytest.importorskip("tensornetwork", minversion="0.1")


def prep_par(par, op):
    "Convert par into a list of parameters that op expects."
    if op.par_domain == "A":
        return [np.diag([x, 1]) for x in par]
    return par


@pytest.mark.usefixtures("skip_if_no_tf_support")
class TestTensorNetworkTFIntegration:
    """Integration tests for expt.tensornet.tf. This test ensures it integrates
    properly with the PennyLane interface, in particular QNode."""

    def test_load_tensornet_tf_device(self):
        """Test that the tensor network plugin loads correctly"""

        dev = qml.device("expt.tensornet.tf", wires=2)
        assert dev.num_wires == 2
        assert dev.shots == 1000
        assert dev.analytic
        assert dev.short_name == "expt.tensornet.tf"

    def test_args(self):
        """Test that the plugin requires correct arguments"""

        with pytest.raises(
                TypeError, match="missing 1 required positional argument: 'wires'"
        ):
            qml.device("expt.tensornet.tf")

    @pytest.mark.parametrize("gate", set(qml.ops.cv.ops))
    def test_unsupported_gate_error(self, tensornet_tf_device_3_wires, gate):
        """Tests that an error is raised if an unsupported gate is applied"""
        op = getattr(qml.ops, gate)

        if op.num_wires is qml.operation.Wires.Any or qml.operation.Wires.All:
            wires = [0]
        else:
            wires = list(range(op.num_wires))

        @qml.qnode(tensornet_tf_device_3_wires)
        def circuit(*x):
            """Test quantum function"""
            x = prep_par(x, op)
            op(*x, wires=wires)

            return qml.expval(qml.X(0))

        with pytest.raises(
                qml.DeviceError,
                match="Gate {} not supported on device expt.tensornet.tf".format(gate),
        ):
            x = np.random.random([op.num_params])
            circuit(*x)

    @pytest.mark.parametrize("observable", set(qml.ops.cv.obs))
    def test_unsupported_observable_error(self, tensornet_tf_device_3_wires, observable):
        """Test error is raised with unsupported observables"""

        op = getattr(qml.ops, observable)

        if op.num_wires is qml.operation.Wires.Any or qml.operation.Wires.All:
            wires = [0]
        else:
            wires = list(range(op.num_wires))

        @qml.qnode(tensornet_tf_device_3_wires)
        def circuit(*x):
            """Test quantum function"""
            x = prep_par(x, op)
            return qml.expval(op(*x, wires=wires))

        with pytest.raises(
                qml.DeviceError,
                match="Observable {} not supported on device expt.tensornet.tf".format(observable),
        ):
            x = np.random.random([op.num_params])
            circuit(*x)

    def test_qubit_circuit(self, tensornet_tf_device_1_wire, tol):
        """Test that the tensor network plugin provides correct result for a simple circuit"""

        p = 0.543

        @qml.qnode(tensornet_tf_device_1_wire)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        expected = -np.sin(p)

        assert np.isclose(circuit(p), expected, atol=tol, rtol=0)

    def test_qubit_identity(self, tensornet_tf_device_1_wire, tol):
        """Test that the tensor network plugin provides correct result for the Identity expectation"""

        p = 0.543

        @qml.qnode(tensornet_tf_device_1_wire)
        def circuit(x):
            """Test quantum function"""
            qml.RX(x, wires=0)
            return qml.expval(qml.Identity(0))

        assert np.isclose(circuit(p), 1, atol=tol, rtol=0)

    # This test is ran against the state |0> with one Z expval
    @pytest.mark.parametrize("name,expected_output", [
        ("PauliX", -1),
        ("PauliY", -1),
        ("PauliZ", 1),
        ("Hadamard", 0),
    ])
    def test_supported_gate_single_wire_no_parameters(self, tensornet_tf_device_1_wire, tol, name, expected_output):
        """Tests supported gates that act on a single wire that are not parameterized"""

        op = getattr(qml.ops, name)

        assert tensornet_tf_device_1_wire.supports_operation(name)

        @qml.qnode(tensornet_tf_device_1_wire)
        def circuit():
            op(wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state |Phi+> with two Z expvals
    @pytest.mark.parametrize("name,expected_output", [
        ("CNOT", [-1 / 2, 1]),
        ("SWAP", [-1 / 2, -1 / 2]),
        ("CZ", [-1 / 2, -1 / 2]),
    ])
    def test_supported_gate_two_wires_no_parameters(self, tensornet_tf_device_2_wires, tol, name, expected_output):
        """Tests supported gates that act on two wires that are not parameterized"""

        op = getattr(qml.ops, name)

        assert tensornet_tf_device_2_wires.supports_operation(name)

        @qml.qnode(tensornet_tf_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
            op(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,expected_output", [
        ("CSWAP", [-1, -1, 1]),
    ])
    def test_supported_gate_three_wires_no_parameters(self, tensornet_tf_device_3_wires, tol, name, expected_output):
        """Tests supported gates that act on three wires that are not parameterized"""

        op = getattr(qml.ops, name)

        assert tensornet_tf_device_3_wires.supports_operation(name)

        @qml.qnode(tensornet_tf_device_3_wires)
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            op(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran with two Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("BasisState", [0, 0], [1, 1]),
        ("BasisState", [1, 0], [-1, 1]),
        ("BasisState", [0, 1], [1, -1]),
        ("QubitStateVector", [1, 0, 0, 0], [1, 1]),
        ("QubitStateVector", [0, 0, 1, 0], [-1, 1]),
        ("QubitStateVector", [0, 1, 0, 0], [1, -1]),
    ])
    def test_supported_state_preparation(self, tensornet_tf_device_2_wires, tol, name, par, expected_output):
        """Tests supported state preparations"""

        op = getattr(qml.ops, name)

        assert tensornet_tf_device_2_wires.supports_operation(name)

        @qml.qnode(tensornet_tf_device_2_wires)
        def circuit():
            op(np.array(par), wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran on the state |0> with one Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("PhaseShift", [math.pi / 2], 1),
        ("PhaseShift", [-math.pi / 4], 1),
        ("RX", [math.pi / 2], 0),
        ("RX", [-math.pi / 4], 1 / math.sqrt(2)),
        ("RY", [math.pi / 2], 0),
        ("RY", [-math.pi / 4], 1 / math.sqrt(2)),
        ("RZ", [math.pi / 2], 1),
        ("RZ", [-math.pi / 4], 1),
        ("Rot", [math.pi / 2, 0, 0], 1),
        ("Rot", [0, math.pi / 2, 0], 0),
        ("Rot", [0, 0, math.pi / 2], 1),
        ("Rot", [math.pi / 2, -math.pi / 4, -math.pi / 4], 1 / math.sqrt(2)),
        ("Rot", [-math.pi / 4, math.pi / 2, math.pi / 4], 0),
        ("Rot", [-math.pi / 4, math.pi / 4, math.pi / 2], 1 / math.sqrt(2)),
        ("QubitUnitary",
         [np.array([[1j / math.sqrt(2), 1j / math.sqrt(2)], [1j / math.sqrt(2), -1j / math.sqrt(2)]])], 0),
        ("QubitUnitary",
         [np.array([[-1j / math.sqrt(2), 1j / math.sqrt(2)], [1j / math.sqrt(2), 1j / math.sqrt(2)]])], 0),
    ])
    def test_supported_gate_single_wire_with_parameters(self, tensornet_tf_device_1_wire, tol, name, par,
                                                        expected_output):
        """Tests supported gates that act on a single wire that are parameterized"""

        op = getattr(qml.ops, name)

        assert tensornet_tf_device_1_wire.supports_operation(name)

        @qml.qnode(tensornet_tf_device_1_wire)
        def circuit():
            op(*par, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    # This test is ran against the state 1/2|00>+sqrt(3)/2|11> with two Z expvals
    @pytest.mark.parametrize("name,par,expected_output", [
        ("CRX", [0], [-1 / 2, -1 / 2]),
        ("CRX", [-math.pi], [-1 / 2, 1]),
        ("CRX", [math.pi / 2], [-1 / 2, 1 / 4]),
        ("CRY", [0], [-1 / 2, -1 / 2]),
        ("CRY", [-math.pi], [-1 / 2, 1]),
        ("CRY", [math.pi / 2], [-1 / 2, 1 / 4]),
        ("CRZ", [0], [-1 / 2, -1 / 2]),
        ("CRZ", [-math.pi], [-1 / 2, -1 / 2]),
        ("CRZ", [math.pi / 2], [-1 / 2, -1 / 2]),
        ("CRot", [math.pi / 2, 0, 0], [-1 / 2, -1 / 2]),
        ("CRot", [0, math.pi / 2, 0], [-1 / 2, 1 / 4]),
        ("CRot", [0, 0, math.pi / 2], [-1 / 2, -1 / 2]),
        ("CRot", [math.pi / 2, 0, -math.pi], [-1 / 2, -1 / 2]),
        ("CRot", [0, math.pi / 2, -math.pi], [-1 / 2, 1 / 4]),
        ("CRot", [-math.pi, 0, math.pi / 2], [-1 / 2, -1 / 2]),
        ("QubitUnitary", [np.array(
            [[1, 0, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
             [0, 0, 0, 1]])], [-1 / 2, -1 / 2]),
        ("QubitUnitary", [np.array(
            [[-1, 0, 0, 0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2), 0], [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0],
             [0, 0, 0, -1]])], [-1 / 2, -1 / 2]),
    ])
    def test_supported_gate_two_wires_with_parameters(self, tensornet_tf_device_2_wires, tol, name, par,
                                                      expected_output):
        """Tests supported gates that act on two wires wires that are parameterized"""

        op = getattr(qml.ops, name)

        assert tensornet_tf_device_2_wires.supports_operation(name)

        @qml.qnode(tensornet_tf_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array([1 / 2, 0, 0, math.sqrt(3) / 2]), wires=[0, 1])
            op(*par, wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        assert np.allclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output", [
        ("PauliX", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1),
        ("PauliX", [1 / math.sqrt(2), -1 / math.sqrt(2)], -1),
        ("PauliX", [1, 0], 0),
        ("PauliY", [1 / math.sqrt(2), 1j / math.sqrt(2)], 1),
        ("PauliY", [1 / math.sqrt(2), -1j / math.sqrt(2)], -1),
        ("PauliY", [1, 0], 0),
        ("PauliZ", [1, 0], 1),
        ("PauliZ", [0, 1], -1),
        ("PauliZ", [1 / math.sqrt(2), 1 / math.sqrt(2)], 0),
        ("Hadamard", [1, 0], 1 / math.sqrt(2)),
        ("Hadamard", [0, 1], -1 / math.sqrt(2)),
        ("Hadamard", [1 / math.sqrt(2), 1 / math.sqrt(2)], 1 / math.sqrt(2)),
    ])
    def test_supported_observable_single_wire_no_parameters(self, tensornet_tf_device_1_wire, tol, name, state,
                                                            expected_output):
        """Tests supported observables on single wires without parameters."""

        obs = getattr(qml.ops, name)

        assert tensornet_tf_device_1_wire.supports_observable(name)

        @qml.qnode(tensornet_tf_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output,par", [
        ("Identity", [1, 0], 1, []),
        ("Identity", [0, 1], 1, []),
        ("Identity", [1 / math.sqrt(2), -1 / math.sqrt(2)], 1, []),
        ("Hermitian", [1, 0], 1, [np.array([[1, 1j], [-1j, 1]])]),
        ("Hermitian", [0, 1], 1, [np.array([[1, 1j], [-1j, 1]])]),
        ("Hermitian", [1 / math.sqrt(2), -1 / math.sqrt(2)], 1, [np.array([[1, 1j], [-1j, 1]])]),
    ])
    def test_supported_observable_single_wire_with_parameters(self, tensornet_tf_device_1_wire, tol, name, state,
                                                              expected_output, par):
        """Tests supported observables on single wires with parameters."""

        obs = getattr(qml.ops, name)

        assert tensornet_tf_device_1_wire.supports_observable(name)

        @qml.qnode(tensornet_tf_device_1_wire)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0])
            return qml.expval(obs(*par, wires=[0]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    @pytest.mark.parametrize("name,state,expected_output,par", [
        ("Hermitian", [1 / math.sqrt(3), 0, 1 / math.sqrt(3), 1 / math.sqrt(3)], 5 / 3,
         [np.array([[1, 1j, 0, 1], [-1j, 1, 0, 0], [0, 0, 1, -1j], [1, 0, 1j, 1]])]),
        ("Hermitian", [0, 0, 0, 1], 0, [np.array([[0, 1j, 0, 0], [-1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])]),
        ("Hermitian", [1 / math.sqrt(2), 0, -1 / math.sqrt(2), 0], 1,
         [np.array([[1, 1j, 0, 0], [-1j, 1, 0, 0], [0, 0, 1, -1j], [0, 0, 1j, 1]])]),
        ("Hermitian", [1 / math.sqrt(3), -1 / math.sqrt(3), 1 / math.sqrt(6), 1 / math.sqrt(6)], 1,
         [np.array([[1, 1j, 0, .5j], [-1j, 1, 0, 0], [0, 0, 1, -1j], [-.5j, 0, 1j, 1]])]),
        ("Hermitian", [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], 1,
         [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
        ("Hermitian", [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0], -1,
         [np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])]),
    ])
    def test_supported_observable_two_wires_with_parameters(self, tensornet_tf_device_2_wires, tol, name, state,
                                                            expected_output, par):
        """Tests supported observables on two wires with parameters."""

        obs = getattr(qml.ops, name)

        assert tensornet_tf_device_2_wires.supports_observable(name)

        @qml.qnode(tensornet_tf_device_2_wires)
        def circuit():
            qml.QubitStateVector(np.array(state), wires=[0, 1])
            return qml.expval(obs(*par, wires=[0, 1]))

        assert np.isclose(circuit(), expected_output, atol=tol, rtol=0)

    def test_cannot_overwrite_state(self, tensornet_tf_device_2_wires):
        """Tests that _state is a property and cannot be overwritten."""

        dev = tensornet_tf_device_2_wires

        with pytest.raises(AttributeError, match="can't set attribute"):
            dev._state = np.array([[1, 0],
                                   [0, 0]])

    def test_correct_state(self, tensornet_tf_device_2_wires):

        dev = tensornet_tf_device_2_wires
        state = dev._state

        expected = np.array([[1, 0],
                             [0, 0]])
        assert np.allclose(state, expected)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        circuit()
        state = dev._state

        expected = np.array([[1, 0],
                             [1, 0]]) / np.sqrt(2)
        assert np.allclose(state, expected)


class TestQNodeJacobianExceptions:
    """Tests that QNode.jacobian raises proper errors"""

    def test_undifferentiable_operation(self, tensornet_tf_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           qfunc contains an operation that is not differentiable."""

        def circuit(x):
            qml.BasisState(np.array([x, 0]), wires=[0, 1])
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, tensornet_tf_device_2_wires)

        with pytest.raises(ValueError, match="Cannot differentiate wrt parameter"):
            node.jacobian(0.5)

    def test_operation_not_supporting_analytic_gradient(self, tensornet_tf_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           qfunc contains an operation that does not support analytic gradients."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([x, 0]), 0))

        node = qml.QNode(circuit, tensornet_tf_device_2_wires)

        with pytest.raises(ValueError, match="analytic gradient method cannot be used with"):
            node.jacobian(0.5, method="A")

    def test_bogus_gradient_method_set(self, tensornet_tf_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           gradient method set is bogus."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        # in non-cached mode, the grad method would be
        # recomputed and overwritten from the
        # bogus value 'J'. Caching stops this from happening.
        node = qml.QNode(circuit, tensornet_tf_device_2_wires, cache=True)

        node.evaluate([0.0])
        keys = node.grad_method_for_par.keys()
        if keys:
            k0 = [k for k in keys][0]

        node.grad_method_for_par[k0] = "J"

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5)

    def test_indices_not_unique(self, tensornet_tf_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           jacobian is requested for non-unique indices."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, tensornet_tf_device_2_wires)

        with pytest.raises(ValueError, match="Parameter indices must be unique."):
            node.jacobian(0.5, which=[0, 0])

    def test_indices_nonexistant(self, tensornet_tf_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           jacobian is requested for non-existant parameters."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, tensornet_tf_device_2_wires)

        with pytest.raises(ValueError, match="Tried to compute the gradient wrt"):
            node.jacobian(0.5, which=[0, 6])

        with pytest.raises(ValueError, match="Tried to compute the gradient wrt"):
            node.jacobian(0.5, which=[1, -1])

    def test_unknown_method(self, tensornet_tf_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if the
           gradient method is unknown."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, tensornet_tf_device_2_wires)

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5, method="unknown")

    def test_wrong_order_in_finite_difference(self, tensornet_tf_device_2_wires):
        """Tests that QNode.jacobian properly raises an error if finite
           differences are attempted with wrong order."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = qml.QNode(circuit, tensornet_tf_device_2_wires)

        with pytest.raises(ValueError, match="Order must be 1 or 2"):
            node.jacobian(0.5, method="F", order=3)

    # Parameters obtained using the following three lines of code:
    # ALLOWED_DIFF_METHODS = ("best", "parameter-shift", "finite-diff")
    # ALLOWED_INTERFACE = ("autograd", "numpy", "torch", "tf", None)
    # diff_method_interface_products = product(ALLOWED_DIFF_METHODS, ALLOWED_INTERFACE)

    @pytest.mark.parametrize("diff_method, interface", [
        ('best', 'autograd'),
        ('best', 'numpy'),
        # ('best', 'torch'),
        ('best', 'tf'),
        ('best', None),
        ('parameter-shift', 'autograd'),
        ('parameter-shift', 'numpy'),
        # ('parameter-shift', 'torch'),
        ('parameter-shift', 'tf'),
        ('parameter-shift', None),
        ('finite-diff', 'autograd'),
        ('finite-diff', 'numpy'),
        # ('finite-diff', 'torch'),
        ('finite-diff', 'tf'),
        ('finite-diff', None),
    ])
    def test_jacobian(self, tensornet_tf_device_3_wires, diff_method, interface, tol):

        p = np.array([0.43316321, 0.2162158, 0.75110998, 0.94714242])

        dev1 = qml.device('expt.tensornet.tf', wires=3)

        @qnode(dev1, interface=interface, diff_method=diff_method)
        def circuit1(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device('default.qubit', wires=3)

        @qnode(dev, interface=interface, diff_method=diff_method)
        def circuit2(x):
            for i in range(0, len(p), 2):
                qml.RX(x[i], wires=0)
                qml.RY(x[i + 1], wires=1)
            for i in range(2):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit1(p), circuit2(p), atol=tol, rtol=0)
        a = circuit1.jacobian([p])
        b = circuit2.jacobian([p])

        assert np.allclose(circuit1.jacobian([p]), circuit2.jacobian([p]), atol=tol, rtol=0)
