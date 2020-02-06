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
"""
Unit tests for the :mod:`pennylane` :class:`JacobianQNode` class.
"""

import pytest
import numpy as np

import pennylane as qml
from pennylane._device import Device
from pennylane.operation import CVObservable
from pennylane.qnodes.base import QuantumFunctionError
from pennylane.qnodes.jacobian import JacobianQNode


@pytest.fixture(scope="function")
def operable_mock_device_2_wires(monkeypatch):
    """A mock instance of the abstract Device class that can support qfuncs."""

    dev = Device
    with monkeypatch.context() as m:
        m.setattr(dev, '__abstractmethods__', frozenset())
        m.setattr(dev, '_capabilities', {"model": "qubit"})
        m.setattr(dev, 'operations', ["BasisState", "RX", "RY", "CNOT", "Rot", "PhaseShift"])
        m.setattr(dev, 'observables', ["PauliX", "PauliY", "PauliZ"])
        m.setattr(dev, 'reset', lambda self: None)
        m.setattr(dev, 'apply', lambda self, x, y, z: None)
        m.setattr(dev, 'expval', lambda self, x, y, z: 1)
        yield Device(wires=2)


class TestAJacobianQNodeDetails:
    """Test configuration details of the autograd interface"""

    def test_interface_str(self, qubit_device_2_wires):
        """Test that the interface string is correctly identified
        as None"""
        def circuit(x, y, z):
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

        circuit = JacobianQNode(circuit, qubit_device_2_wires)
        assert circuit.interface == None


class TestJacobianQNodeExceptions:
    """Tests that JacobianQNode.jacobian raises proper errors."""

    def test_gradient_of_sample(self, operable_mock_device_2_wires):
        """Differentiation of a sampled output."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        node = JacobianQNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(QuantumFunctionError,
                           match="Circuits that include sampling can not be differentiated."):
            node.jacobian(1.0)

    def test_nondifferentiable_operator(self, operable_mock_device_2_wires):
        """Differentiating wrt. a parameter
        that appears as an argument to a nondifferentiable operator."""

        def circuit(x):
            qml.BasisState(np.array([x, 0]), wires=[0, 1])  # not differentiable
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JacobianQNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Cannot differentiate with respect to the parameters"):
            node.jacobian(0.5)

    def test_operator_not_supporting_pd_analytic(self, operable_mock_device_2_wires):
        """Differentiating wrt. a parameter that appears
        as an argument to an operation that does not support parameter-shift derivatives."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.Hermitian(np.diag([x, 0]), 0))

        node = JacobianQNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="analytic gradient method cannot be used with"):
            node.jacobian(0.5, method="A")

    def test_bogus_gradient_method_set(self, operable_mock_device_2_wires):
        """The gradient method set is bogus."""

        def circuit(x):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        # in mutable mode, the grad method would be
        # recomputed and overwritten from the
        # bogus value 'J'. Caching stops this from happening.
        node = JacobianQNode(circuit, operable_mock_device_2_wires, mutable=False)

        node.evaluate([0.0], {})
        node.par_to_grad_method[0] = "J"

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5)

    def test_indices_not_unique(self, operable_mock_device_2_wires):
        """The Jacobian is requested for non-unique indices."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JacobianQNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Parameter indices must be unique."):
            node.jacobian(0.5, wrt=[0, 0])

    def test_indices_nonexistant(self, operable_mock_device_2_wires):
        """ The Jacobian is requested for non-existant parameters."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JacobianQNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Tried to compute the gradient with respect to"):
            node.jacobian(0.5, wrt=[0, 6])

        with pytest.raises(ValueError, match="Tried to compute the gradient with respect to"):
            node.jacobian(0.5, wrt=[1, -1])

    def test_unknown_gradient_method(self, operable_mock_device_2_wires):
        """ The gradient method is unknown."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JacobianQNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Unknown gradient method"):
            node.jacobian(0.5, method="unknown")

    def test_wrong_order_in_finite_difference(self, operable_mock_device_2_wires):
        """Finite difference are attempted with wrong order."""

        def circuit(x):
            qml.Rot(0.3, x, -0.2, wires=[0])
            return qml.expval(qml.PauliZ(0))

        node = JacobianQNode(circuit, operable_mock_device_2_wires)

        with pytest.raises(ValueError, match="Order must be 1 or 2"):
            node.jacobian(0.5, method="F", options={'order': 3})


class TestBestMethod:
    """Test different flows of _best_method"""

    def test_all_finite_difference(self, operable_mock_device_2_wires):
        """Finite difference is the best method in almost all cases"""

        def circuit(x, y, z):
            qml.Rot(x, y, z, wires=[0])
            return qml.expval(qml.PauliZ(0))

        q = JacobianQNode(circuit, operable_mock_device_2_wires)
        q._construct([1.0, 1.0, 1.0], {})
        assert q.par_to_grad_method == {0: "F", 1: "F", 2: "F"}

    def test_no_following_observable(self, operable_mock_device_2_wires):
        """Test that the gradient is 0 if no observables succeed"""

        def circuit(x):
            qml.RX(x, wires=[1])
            return qml.expval(qml.PauliZ(0))

        q = JacobianQNode(circuit, operable_mock_device_2_wires)
        q._construct([1.0], {})
        assert q.par_to_grad_method == {0: "0"}

    def test_param_unused(self, operable_mock_device_2_wires):
        """Test that the gradient is 0 of an unused parameter"""

        def circuit(x, y):
            qml.RX(x, wires=[0])
            return qml.expval(qml.PauliZ(0))

        q = JacobianQNode(circuit, operable_mock_device_2_wires)
        q._construct([1.0, 1.0], {})
        assert q.par_to_grad_method == {0: "F", 1: "0"}

    def test_not_differentiable(self, operable_mock_device_2_wires):
        """Test that an operation with grad_method=None is marked as
        non-differentiable"""

        def circuit(x):
            qml.BasisState(x, wires=[1])
            return qml.expval(qml.PauliZ(0))

        q = JacobianQNode(circuit, operable_mock_device_2_wires)
        q._construct([np.array([1.0])], {})
        assert q.par_to_grad_method == {0: None}
