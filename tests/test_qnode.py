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
"""Unit tests for the QNode"""

import copy

# pylint: disable=import-outside-toplevel, protected-access, no-member
import warnings
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane import QNode
from pennylane import numpy as pnp
from pennylane import qnode
from pennylane.exceptions import DeviceError, PennyLaneDeprecationWarning, QuantumFunctionError
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn
from pennylane.workflow.qnode import _make_execution_config
from pennylane.workflow.set_shots import set_shots


def test_add_transform_deprecation():
    """Test that the add_transform method raises a deprecation warning."""

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    with pytest.warns(
        PennyLaneDeprecationWarning,
        match="The `qml.QNode.add_transform` method is deprecated and will be removed in v0.44",
    ):
        circuit.add_transform(
            qml.transforms.core.TransformContainer(
                qml.transform(qml.gradients.param_shift.expand_transform)
            )
        )


def dummyfunc():
    """dummy func."""
    return None


# pylint: disable=unused-argument
class CustomDevice(qml.devices.Device):
    """A null device that just returns 0."""

    def __repr__(self):
        return "CustomDevice"

    def execute(self, circuits, execution_config=None):
        return (0,)


class CustomDeviceWithDiffMethod(qml.devices.Device):
    """A device that defines a derivative."""

    def execute(self, circuits, execution_config=None):
        return 0

    def compute_derivatives(self, circuits, execution_config=None):
        """Device defines its own method to compute derivatives"""
        return 0


def test_no_measure():
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(QuantumFunctionError, match="must return either a single measurement"):
        _ = circuit(0.65)


def test_copy():
    """Test that a shallow copy also copies the execute kwargs, gradient kwargs, and transform program."""
    dev = CustomDevice()

    qn = qml.QNode(dummyfunc, dev)
    copied_qn = copy.copy(qn)
    assert copied_qn is not qn
    assert copied_qn.execute_kwargs == qn.execute_kwargs
    assert copied_qn.execute_kwargs is not qn.execute_kwargs
    assert list(copied_qn.transform_program) == list(qn.transform_program)
    assert copied_qn.transform_program is not qn.transform_program
    assert copied_qn.gradient_kwargs == qn.gradient_kwargs
    assert copied_qn.gradient_kwargs is not qn.gradient_kwargs

    assert copied_qn.func is qn.func
    assert copied_qn.device is qn.device
    assert copied_qn.interface is qn.interface
    assert copied_qn.diff_method == qn.diff_method


class TestUpdate:
    """Tests the update instance method of QNode"""

    def test_new_object_creation(self):
        """Test that a new object is created rather than mutated"""
        dev = qml.device("default.qubit")
        dummy_qn = qml.QNode(dummyfunc, dev)
        updated_qn = dummy_qn.update(device=dummy_qn.device)
        assert updated_qn is not dummy_qn

    def test_empty_update(self):
        """Test that providing no update parameters throws an error."""
        dev = qml.device("default.qubit")
        dummy_qn = qml.QNode(dummyfunc, dev)
        with pytest.raises(
            ValueError, match="Must specify at least one configuration property to update."
        ):
            dummy_qn.update()

    def test_update_args(self):
        """Test that arguments of QNode can be updated"""
        dev = qml.device("default.qubit")
        dummy_qn = qml.QNode(dummyfunc, dev)

        def circuit(x):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        new_circuit = dummy_qn.update(func=circuit)
        assert new_circuit.func is circuit

    @pytest.mark.torch
    def test_update_kwargs(self):
        """Test that keyword arguments can be updated"""
        dev = qml.device("default.qubit")
        dummy_qn = qml.QNode(dummyfunc, dev)

        def circuit(x):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        assert dummy_qn.interface == "auto"
        new_circuit = dummy_qn.update(func=circuit).update(interface="torch")
        assert qml.math.get_interface(new_circuit(1)) == "torch"

    def test_update_gradient_kwargs(self):
        """Test that gradient kwargs are updated correctly"""
        dev = qml.device("default.qubit")

        @qml.qnode(dev, gradient_kwargs={"atol": 1})
        def circuit(x):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        assert set(circuit.gradient_kwargs.keys()) == {"atol"}

        new_atol_circuit = circuit.update(gradient_kwargs={"atol": 2})
        assert set(new_atol_circuit.gradient_kwargs.keys()) == {"atol"}
        assert new_atol_circuit.gradient_kwargs["atol"] == 2

        new_kwarg_circuit = circuit.update(gradient_kwargs={"h": 1})
        assert set(new_kwarg_circuit.gradient_kwargs.keys()) == {"atol", "h"}
        assert new_kwarg_circuit.gradient_kwargs["atol"] == 1
        assert new_kwarg_circuit.gradient_kwargs["h"] == 1

    def test_update_multiple_arguments(self):
        """Test that multiple parameters can be updated at once."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(x):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        assert circuit.diff_method == "best"
        assert not circuit.execute_kwargs["device_vjp"]
        new_circuit = circuit.update(diff_method="adjoint", device_vjp=True)
        assert new_circuit.diff_method == "adjoint"
        assert new_circuit.execute_kwargs["device_vjp"]

    def test_update_transform_program(self):
        """Test that the transform program is properly preserved"""
        dev = qml.device("default.qubit", wires=2)

        @qml.transforms.combine_global_phases
        @qml.qnode(dev)
        def circuit(x):
            qml.RZ(x, wires=0)
            qml.GlobalPhase(phi=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        assert qml.transforms.combine_global_phases in circuit.transform_program
        assert len(circuit.transform_program) == 1
        new_circuit = circuit.update(diff_method="parameter-shift")
        assert new_circuit.diff_method == "parameter-shift"
        assert circuit.transform_program == new_circuit.transform_program


class TestInitialization:
    """Testing the initialization of the qnode."""

    def test_shots_initialization(self):
        """Test the initialization with shots."""

        # Default behavior: no shots from either device or qnode
        # shots should be None
        @qml.qnode(qml.device("default.qubit"))
        def f():
            return qml.state()

        assert not f.shots

        # Shots should be set correctly set from the qnode init
        @qml.qnode(qml.device("default.qubit"), shots=10)
        def f2():
            return qml.state()

        assert f2._shots_override_device  # pylint: disable=protected-access
        assert f2.shots == qml.measurements.Shots(10)

        # Shots from device should be set correctly
        with pytest.warns(PennyLaneDeprecationWarning, match="shots on device is deprecated"):

            @qml.qnode(qml.device("default.qubit", shots=5))
            def f3():
                return qml.state()

        assert f3.shots == qml.measurements.Shots(5)

        # Shots from device and also from qnode, then qnode should take precedence
        with pytest.warns(PennyLaneDeprecationWarning, match="shots on device is deprecated"):

            @qml.qnode(qml.device("default.qubit", shots=500), shots=None)
            def f4():
                return qml.state()

        assert f4.shots == qml.measurements.Shots(None)

    def test_cache_initialization_maxdiff_1(self):
        """Test that when max_diff = 1, the cache initializes to false."""

        @qml.qnode(qml.device("default.qubit"), max_diff=1)
        def f():
            return qml.state()

        assert f.execute_kwargs["cache"] == "auto"

    def test_cache_initialization_maxdiff_2(self):
        """Test that when max_diff = 2, the cache initialization to True."""

        @qml.qnode(qml.device("default.qubit"), max_diff=2)
        def f():
            return qml.state()

        assert f.execute_kwargs["cache"] == "auto"


# pylint: disable=too-many-public-methods
class TestValidation:
    """Tests for QNode creation and validation"""

    @pytest.mark.parametrize("return_type", (tuple, list))
    def test_return_behaviour_consistency(self, return_type):
        """Test that the QNode return typing stays consistent"""

        @qml.qnode(qml.device("default.qubit"))
        def circuit(return_type):
            return return_type([qml.expval(qml.Z(0))])

        assert isinstance(circuit(return_type), return_type)

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"
        expected_error = rf"'{test_interface}' is not a valid Interface\. Please use one of the supported interfaces: \[.*\]\."

        with pytest.raises(ValueError, match=expected_error):
            QNode(dummyfunc, dev, interface=test_interface)

    def test_changing_invalid_interface(self):
        """Test that an exception is raised for an invalid interface
        on a pre-existing QNode"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"

        @qnode(dev)
        def circuit(x):
            """a circuit."""
            qml.RX(x, wires=0)
            return qml.probs(wires=0)

        expected_error = rf"'{test_interface}' is not a valid Interface\. Please use one of the supported interfaces: \[.*\]\."

        with pytest.raises(ValueError, match=expected_error):
            circuit.interface = test_interface

    def test_invalid_device(self):
        """Test that an exception is raised for an invalid device"""
        with pytest.raises(QuantumFunctionError, match="Invalid device"):
            QNode(dummyfunc, None)

    # pylint: disable=protected-access, too-many-statements
    def test_diff_method(self):
        """Test that a user-supplied diff method correctly returns the right
        diff method."""
        dev = qml.device("default.qubit", wires=1)

        qn = QNode(dummyfunc, dev, diff_method="best")
        assert qn.diff_method == "best"

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="best")
        assert qn.diff_method == "best"

        qn = QNode(dummyfunc, dev, diff_method="backprop")
        assert qn.diff_method == "backprop"

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="backprop")
        assert qn.diff_method == "backprop"

        dev2 = CustomDeviceWithDiffMethod()
        qn = QNode(dummyfunc, dev2, diff_method="device")
        assert qn.diff_method == "device"

        qn = QNode(dummyfunc, dev2, interface="autograd", diff_method="device")
        assert qn.diff_method == "device"

        qn = QNode(dummyfunc, dev, diff_method="finite-diff")
        assert qn.diff_method == "finite-diff"

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="finite-diff")
        assert qn.diff_method == "finite-diff"

        qn = QNode(dummyfunc, dev, diff_method="spsa")
        assert qn.diff_method == "spsa"

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="hadamard")
        assert qn.diff_method == "hadamard"

        qn = QNode(dummyfunc, dev, diff_method="parameter-shift")
        assert qn.diff_method == "parameter-shift"

        qn = QNode(dummyfunc, dev, interface="autograd", diff_method="parameter-shift")
        assert qn.diff_method == "parameter-shift"

    @pytest.mark.autograd
    def test_gradient_transform(self, mocker):
        """Test passing a gradient transform directly to a QNode"""
        dev = qml.device("default.qubit", wires=1)
        spy = mocker.spy(qml.gradients.finite_difference, "finite_diff_coeffs")

        @qnode(dev, diff_method=qml.gradients.finite_diff)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qml.grad(circuit)(pnp.array(0.5, requires_grad=True))
        spy.assert_called()

    def test_unknown_diff_method_string(self):
        """Test that an exception is raised for an unknown differentiation method string"""
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            QuantumFunctionError,
            match="Differentiation method hello not recognized",
        ):
            QNode(dummyfunc, dev, interface="autograd", diff_method="hello")

    def test_unknown_diff_method_type(self):
        """Test that an exception is raised for an unknown differentiation method type"""
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(
            ValueError,
            match="Differentiation method 5 must be a str, TransformDispatcher, or None",
        ):
            QNode(dummyfunc, dev, interface="autograd", diff_method=5)

    def test_adjoint_finite_shots(self):
        """Tests that a DeviceError is raised with the adjoint differentiation method
        when the device has finite shots"""

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, diff_method="adjoint")
        def circ():
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(
            QuantumFunctionError,
            match="does not support adjoint with requested circuit",
        ):
            qml.set_shots(circ, shots=1)()

    @pytest.mark.autograd
    def test_sparse_diffmethod_error(self):
        """Test that an error is raised when the observable is SparseHamiltonian and the
        differentiation method is not parameter-shift."""
        dev = qml.device("default.qubit", wires=2)

        @qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.RX(param, wires=0)
            return qml.expval(qml.SparseHamiltonian(csr_matrix(np.eye(4)), [0, 1]))

        with pytest.raises(
            QuantumFunctionError,
            match="does not support backprop with requested circuit",
        ):
            qml.grad(circuit, argnum=0)([0.5])

    def test_qnode_print(self):
        """Test that printing a QNode object yields the right information."""
        dev = qml.device("default.qubit", wires=1)

        def func(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        assert (
            repr(qn)
            == f"<QNode: device='<default.qubit device (wires=1) at {hex(id(dev))}>', interface='auto', diff_method='best', shots='Shots(total=None)'>"
        )

        qn = QNode(func, dev, interface="autograd")

        assert (
            repr(qn)
            == f"<QNode: device='<default.qubit device (wires=1) at {hex(id(dev))}>', interface='autograd', diff_method='best', shots='Shots(total=None)'>"
        )

    @pytest.mark.autograd
    def test_diff_method_none(self, tol):
        """Test that diff_method=None creates a QNode with no interface, and no
        device swapping."""
        dev = qml.device("default.qubit", wires=1)

        @qnode(dev, diff_method=None)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.interface == "auto"
        assert circuit.device is dev

        # QNode can still be executed
        assert np.allclose(circuit(0.5), np.cos(0.5), atol=tol, rtol=0)

        with pytest.warns(UserWarning, match="Attempted to differentiate a function with no"):
            grad = qml.grad(circuit)(0.5)

        assert np.allclose(grad, 0)

    def test_not_giving_mode_kwarg_does_not_raise_warning(self):
        """Test that not providing a value for mode does not raise a warning."""
        with warnings.catch_warnings(record=True) as record:
            _ = qml.QNode(lambda f: f, qml.device("default.qubit", wires=1))

        assert len(record) == 0


# pylint: disable=too-few-public-methods
# pylint: disable=unnecessary-lambda
class TestPyTreeStructure:
    """Tests for preservation of pytree structure through execution"""

    @pytest.mark.parametrize(
        "measurement",
        [
            lambda: ({"probs": qml.probs()},),
            lambda: qml.probs(),
            lambda: [
                [qml.probs(wires=1), {"a": qml.probs(wires=0)}, qml.expval(qml.Z(0))],
                {"probs": qml.probs(wires=0), "exp": qml.expval(qml.X(1))},
            ],
            lambda: {"exp": qml.expval(qml.Z(0))},
            lambda: {
                "layer1": {
                    "layer2": {
                        "probs": qml.probs(wires=[0, 1]),
                        "expval": qml.expval(qml.PauliY(1)),
                    },
                    "single_prob": qml.probs(wires=0),
                }
            },
            lambda: (
                [qml.probs(wires=1), {"exp": qml.expval(qml.PauliZ(0))}],
                qml.expval(qml.PauliX(1)),
            ),
            lambda: [
                (qml.expval(qml.PauliX(0)), qml.var(qml.PauliY(1))),
                (qml.probs(wires=[1]), {"nested": qml.probs(wires=[0])}),
            ],
            lambda: [
                {
                    "first_layer": qml.probs(wires=0),
                    "second_layer": [
                        qml.expval(qml.PauliX(1)),
                        {"nested_exp": qml.expval(qml.PauliY(0))},
                    ],
                },
                (qml.probs(wires=[0, 1]), {"final": qml.expval(qml.PauliZ(0))}),
            ],
        ],
    )
    def test_pytree_structure_preservation(self, measurement):
        """Test that the result stucture matches the measurement structure."""

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(100)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1, wires=0)
            qml.RY(2, wires=1)
            qml.measure(0)
            qml.CNOT(wires=[0, 1])
            return measurement()

        result = circuit()

        result_structure = qml.pytrees.flatten(result)[1]
        measurement_structure = qml.pytrees.flatten(
            measurement(), is_leaf=lambda obj: isinstance(obj, qml.measurements.MeasurementProcess)
        )[1]

        assert result_structure == measurement_structure

    @pytest.mark.parametrize(
        "measurement",
        [
            lambda: qml.math.hstack([qml.expval(qml.Z(i)) for i in range(2)]),
            lambda: qml.math.stack([qml.expval(qml.Z(i)) for i in range(2)]),
        ],
    )
    def test_tensor_measurement(self, measurement):
        """Tests that measurements of tensor type are handled correctly"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():
            qml.RX(1, wires=0)
            qml.RY(2, wires=1)
            qml.measure(0)
            qml.CNOT(wires=[0, 1])
            return measurement()

        result = circuit()

        assert len(result) == 2


class TestTapeConstruction:
    """Tests for the tape construction"""

    def test_returning_non_measurements(self):
        """Test that an exception is raised if a non-measurement
        is returned from the QNode."""
        dev = qml.device("default.qubit", wires=2)

        def func0(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return 5

        qn = QNode(func0, dev)

        with pytest.raises(QuantumFunctionError, match="must return either a single measurement"):
            qn(5, 1)

        def func2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), 5

        qn = QNode(func2, dev)

        with pytest.raises(QuantumFunctionError, match="must return either a single measurement"):
            qn(5, 1)

        def func3(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return []

        qn = QNode(func3, dev)

        with pytest.raises(QuantumFunctionError, match="must return either a single measurement"):
            qn(5, 1)

    def test_inconsistent_measurement_order(self):
        """Test that an exception is raised if measurements are returned in an
        order different to how they were queued on the tape"""
        dev = qml.device("default.qubit", wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            m = qml.expval(qml.PauliZ(0))
            return qml.expval(qml.PauliX(1)), m

        qn = QNode(func, dev)

        with pytest.raises(
            QuantumFunctionError,
            match="measurements must be returned in the order they are measured",
        ):
            qn(5, 1)

    def test_consistent_measurement_order(self):
        """Test evaluation proceeds as expected if measurements are returned in the
        same order to how they were queued on the tape"""
        dev = qml.device("default.qubit", wires=2)

        contents = []

        def func(x, y):
            op1 = qml.RX(x, wires=0)
            op2 = qml.RY(y, wires=1)
            op3 = qml.CNOT(wires=[0, 1])
            m1 = qml.expval(qml.PauliZ(0))
            m2 = qml.expval(qml.PauliX(1))
            contents.append(op1)
            contents.append(op2)
            contents.append(op3)
            contents.append(m1)
            contents.append(m2)
            return [m1, m2]

        qn = QNode(func, dev)
        tape = qml.workflow.construct_tape(qn)(5, 1)
        assert tape.operations == contents[0:3]
        assert tape.measurements == contents[3:]

    @pytest.mark.jax
    @pytest.mark.parametrize("dev_name", ("default.qubit", "reference.qubit"))
    def test_jit_counts_raises_error(self, dev_name):
        """Test that returning counts in a quantum function with trainable parameters while
        jitting raises an error."""
        import jax

        dev = qml.device(dev_name, wires=2)

        def circuit1(param):
            qml.Hadamard(0)
            qml.RX(param, wires=1)
            qml.CNOT([1, 0])
            return qml.counts()

        qn = qml.set_shots(qml.QNode(circuit1, dev), shots=5)
        jitted_qnode1 = jax.jit(qn)

        with pytest.raises(
            NotImplementedError, match="The JAX-JIT interface doesn't support qml.counts."
        ):
            _ = jitted_qnode1(0.123)

        # Test with qnode decorator syntax
        @qml.set_shots(5)
        @qml.qnode(dev)
        def circuit2(param):
            qml.Hadamard(0)
            qml.RX(param, wires=1)
            qml.CNOT([1, 0])
            return qml.counts()

        jitted_qnode2 = jax.jit(circuit2)

        with pytest.raises(
            NotImplementedError, match="The JAX-JIT interface doesn't support qml.counts."
        ):
            jitted_qnode2(0.123)


def test_decorator(tol):
    """Test that the decorator correctly creates a QNode."""
    dev = qml.device("default.qubit", wires=2)

    @qnode(dev)
    def func(x, y):
        """My function docstring"""
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    assert isinstance(func, QNode)
    assert func.__doc__ == "My function docstring"

    x = pnp.array(0.12, requires_grad=True)
    y = pnp.array(0.54, requires_grad=True)

    res = func(x, y)
    expected = np.cos(x)
    assert np.allclose(res, expected, atol=tol, rtol=0)

    res2 = func(x, y)
    assert np.allclose(res, res2, atol=tol, rtol=0)


class TestIntegration:
    """Integration tests."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "torch", "jax"])
    def test_correct_number_of_executions(self, interface):
        """Test that number of executions can be tracked correctly and executiong
        returns results in the expected interface"""

        def func():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        qn = QNode(func, dev, interface=interface)

        with qml.Tracker(dev, persistent=True) as tracker:
            for _ in range(2):
                res = qn()

        assert tracker.totals["executions"] == 2
        assert qml.math.get_interface(res) == interface

        qn2 = QNode(func, dev, interface=interface)

        with tracker:
            for _ in range(3):
                res = qn2()

        assert tracker.totals["executions"] == 5
        assert qml.math.get_interface(res) == interface

        # qubit of different interface
        qn3 = QNode(func, dev, interface="autograd")
        with tracker:
            res = qn3()

        assert tracker.totals["executions"] == 6
        assert qml.math.get_interface(res) == "autograd"

    def test_num_exec_caching_with_backprop(self):
        """Tests that with diff_method='backprop', the number of executions
        recorded is correct."""
        dev = qml.device("default.qubit", wires=2)

        cache = {}

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit():
            qml.RY(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        with qml.Tracker(dev, persistent=True) as tracker:
            for _ in range(15):
                circuit()

        # Although we've evaluated the QNode more than once, due to caching,
        # there was one execution recorded
        assert tracker.totals["executions"] == 1
        assert cache

    def test_num_exec_caching_device_swap_two_exec(self):
        """Tests that when diff_method='backprop', the number of executions recorded is
        correct even with multiple QNode evaluations."""
        dev = qml.device("default.qubit", wires=2)

        cache = {}

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit0():
            qml.RY(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        with qml.Tracker(dev, persistent=True) as tracker:
            for _ in range(15):
                circuit0()

        @qml.qnode(dev, diff_method="backprop", cache=cache)
        def circuit2():
            qml.RZ(0.345, wires=0)
            return qml.expval(qml.PauliZ(0))

        with tracker:
            for _ in range(15):
                circuit2()

        # Although we've evaluated the QNode several times, due to caching,
        # there were two device executions recorded
        assert tracker.totals["executions"] == 2
        assert cache

    @pytest.mark.autograd
    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "spsa", "hadamard"])
    def test_single_expectation_value_with_argnum_one(self, diff_method, tol):
        """Tests correct output shape and evaluation for a QNode
        with a single expval output where only one parameter is chosen to
        estimate the jacobian.

        This test relies on the fact that exactly one term of the estimated
        jacobian will match the expected analytical value.
        """
        dev = qml.device("default.qubit", wires=3)

        x = pnp.array(0.543, requires_grad=True)
        y = pnp.array(-0.654, requires_grad=True)

        @qnode(
            dev, diff_method=diff_method, gradient_kwargs={"argnum": [1]}
        )  # <--- we only choose one trainable parameter
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        res = qml.grad(circuit)(x, y)
        assert len(res) == 2

        expected = (0, np.cos(y) * np.cos(x))

        assert np.allclose(res, expected, atol=tol, rtol=0)

    # pylint: disable=too-many-positional-arguments
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    @pytest.mark.parametrize("first_par", np.linspace(0.15, np.pi - 0.3, 3))
    @pytest.mark.parametrize("sec_par", np.linspace(0.15, np.pi - 0.3, 3))
    @pytest.mark.parametrize(
        "return_type", [qml.expval(qml.PauliZ(1)), qml.var(qml.PauliZ(1)), qml.probs(wires=[1])]
    )
    @pytest.mark.parametrize(
        "mv_return, mv_res",
        [
            (qml.expval, lambda x: np.sin(x / 2) ** 2),
            (qml.var, lambda x: np.sin(x / 2) ** 2 - np.sin(x / 2) ** 4),
            (qml.probs, lambda x: [np.cos(x / 2) ** 2, np.sin(x / 2) ** 2]),
        ],
    )
    def test_defer_meas_if_mcm_unsupported(
        self, dev_name, first_par, sec_par, return_type, mv_return, mv_res, mocker
    ):  # pylint: disable=too-many-arguments
        """Tests that the transform using the deferred measurement principle is
        applied if the device doesn't support mid-circuit measurements
        natively."""
        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def cry_qnode(x, y):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(x, wires=0)
            qml.CRY(y, wires=[0, 1])
            return qml.apply(return_type)

        @qml.qnode(dev)
        def conditional_ry_qnode(x, y):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(x, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(y, wires=1)
            return qml.apply(return_type), mv_return(op=m_0)

        spy = mocker.spy(qml.defer_measurements, "_transform")
        r1 = cry_qnode(first_par, sec_par)
        r2 = conditional_ry_qnode(first_par, sec_par)

        assert np.allclose(r1, r2[0])
        assert np.allclose(r2[1], mv_res(first_par))
        assert spy.call_count == 2

    def test_dynamic_one_shot_if_mcm_unsupported(self):
        """Test an error is raised if the dynamic one shot transform is a applied to a qnode with a device that
        does not support mid circuit measurements.
        """
        dev = qml.device("default.mixed", wires=2)

        with pytest.raises(
            TypeError,
            match="does not support mid-circuit measurements and/or one-shot execution mode",
        ):

            @qml.transforms.dynamic_one_shot
            @qml.set_shots(100)
            @qml.qnode(dev)
            def _():
                qml.RX(1.23, 0)
                ms = [qml.measure(0) for _ in range(10)]
                return qml.probs(op=ms)

    @pytest.mark.parametrize("basis_state", [[1, 0], [0, 1]])
    def test_sampling_with_mcm(self, basis_state, mocker):
        """Tests that a QNode with qml.sample and mid-circuit measurements
        returns the expected results."""
        dev = qml.device("default.qubit", wires=3)

        first_par = np.pi

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.BasisState(basis_state, wires=[0, 1])
            qml.CRY(x, wires=[0, 1])
            return qml.sample(qml.PauliZ(1))

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.BasisState(basis_state, wires=[0, 1])
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.sample(qml.PauliZ(1))

        r1 = cry_qnode(first_par)
        r2 = conditional_ry_qnode(first_par)
        assert np.allclose(r1, r2)

        cry_qnode_deferred = qml.defer_measurements(cry_qnode)
        conditional_ry_qnode_deferred = qml.defer_measurements(conditional_ry_qnode)

        r1 = cry_qnode_deferred(first_par)
        r2 = conditional_ry_qnode_deferred(first_par)
        assert np.allclose(r1, r2)

    @pytest.mark.tf
    @pytest.mark.parametrize("interface", ["auto"])
    def test_conditional_ops_tensorflow(self, interface):
        """Test conditional operations with TensorFlow."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        dm_conditional_ry_qnode = qml.defer_measurements(conditional_ry_qnode)

        x_ = -0.654
        x1 = tf.Variable(x_, dtype=tf.float64)
        x2 = tf.Variable(x_, dtype=tf.float64)
        x3 = tf.Variable(x_, dtype=tf.float64)

        with tf.GradientTape() as tape1:
            r1 = cry_qnode(x1)

        with tf.GradientTape() as tape2:
            r2 = conditional_ry_qnode(x2)

        with tf.GradientTape() as tape3:
            r3 = dm_conditional_ry_qnode(x3)

        assert np.allclose(r1, r2)
        assert np.allclose(r1, r3)

        grad1 = tape1.gradient(r1, x1)
        grad2 = tape2.gradient(r2, x2)
        grad3 = tape3.gradient(r3, x3)
        assert np.allclose(grad1, grad2)
        assert np.allclose(grad1, grad3)

    @pytest.mark.torch
    @pytest.mark.parametrize("interface", ["torch", "auto"])
    def test_conditional_ops_torch(self, interface):
        """Test conditional operations with Torch."""
        import torch

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface=interface, diff_method="parameter-shift")
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        x1 = torch.tensor(-0.654, dtype=torch.float64, requires_grad=True)
        x2 = torch.tensor(-0.654, dtype=torch.float64, requires_grad=True)

        r1 = cry_qnode(x1)
        r2 = conditional_ry_qnode(x2)

        assert np.allclose(r1.detach(), r2.detach())

        r1.backward()
        r2.backward()
        assert np.allclose(x1.grad.detach(), x2.grad.detach())

    @pytest.mark.jax
    @pytest.mark.parametrize("jax_interface", ["jax", "jax-jit", "auto"])
    def test_conditional_ops_jax(self, jax_interface):
        """Test conditional operations with JAX."""
        import jax

        jnp = jax.numpy
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev, interface=jax_interface, diff_method="parameter-shift")
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            qml.CRY(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        @qml.qnode(dev, interface=jax_interface, diff_method="parameter-shift")
        def conditional_ry_qnode(x):
            """QNode where the defer measurements transform is applied by
            default under the hood."""
            qml.Hadamard(1)
            qml.RY(1.234, wires=0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)
            return qml.expval(qml.PauliZ(1))

        x1 = jnp.array(-0.654)
        x2 = jnp.array(-0.654)

        r1 = cry_qnode(x1)
        r2 = conditional_ry_qnode(x2)

        assert np.allclose(r1, r2)
        assert np.allclose(jax.grad(cry_qnode)(x1), jax.grad(conditional_ry_qnode)(x2))

    def test_qnode_does_not_support_nested_queuing(self):
        """Test that operators in QNodes are not queued to surrounding contexts."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(0)
            return qml.expval(qml.PauliX(0))

        with qml.queuing.AnnotatedQueue() as q:
            circuit()

        tape = qml.workflow.construct_tape(circuit)()
        assert q.queue == []  # pylint: disable=use-implicit-booleaness-not-comparison
        assert len(tape.operations) == 1

    def test_qnode_preserves_inferred_numpy_interface(self):
        """Tests that the QNode respects the inferred numpy interface."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.8)
        res = circuit(x)
        assert qml.math.get_interface(res) == "numpy"

    def test_qnode_default_interface(self):
        """Tests that the default interface is set correctly for a QNode."""

        # pylint: disable=import-outside-toplevel
        import networkx as nx

        @qml.qnode(qml.device("default.qubit"))
        def circuit(graph: nx.Graph):
            for a in graph.nodes:
                qml.Hadamard(wires=a)
            for a, b in graph.edges:
                qml.CZ(wires=[a, b])
            return qml.expval(qml.PauliZ(0))

        graph = nx.complete_graph(3)
        res = circuit(graph)
        assert qml.math.get_interface(res) == "numpy"

    def test_qscript_default_interface(self):
        """Tests that the default interface is set correctly for a QuantumScript."""

        # pylint: disable=import-outside-toplevel
        import networkx as nx

        dev = qml.device("default.qubit")

        # pylint: disable=too-few-public-methods
        class DummyCustomGraphOp(qml.operation.Operation):
            """Dummy custom operation for testing purposes."""

            def __init__(self, graph: nx.Graph):
                super().__init__(graph, wires=graph.nodes)

            def decomposition(self) -> list:
                return []

        graph = nx.complete_graph(3)
        tape = qml.tape.QuantumScript([DummyCustomGraphOp(graph)], [qml.expval(qml.PauliZ(0))])
        res = qml.execute([tape], dev)
        assert qml.math.get_interface(res) == "numpy"

    def test_error_device_vjp_unsuppoprted(self):
        """Test that an error is raised in the device_vjp is unsupported."""

        class DummyDev(qml.devices.Device):
            def execute(self, circuits, execution_config=qml.devices.ExecutionConfig()):
                return 0

            def supports_derivatives(self, execution_config=None, circuit=None):
                return execution_config and execution_config.gradient_method == "vjp_grad"

            def supports_vjp(self, execution_config=None, circuit=None) -> bool:
                return execution_config and execution_config.gradient_method == "vjp_grad"

        @qml.qnode(DummyDev(), diff_method="parameter-shift", device_vjp=True)
        def circuit():
            return qml.expval(qml.Z(0))

        with pytest.raises(QuantumFunctionError, match="device_vjp=True is not supported"):
            circuit()

    @pytest.mark.parametrize(
        "interface",
        (
            pytest.param("autograd", marks=pytest.mark.autograd),
            pytest.param("jax", marks=pytest.mark.jax),
            pytest.param("torch", marks=pytest.mark.torch),
            pytest.param("tensorflow", marks=pytest.mark.tf),
        ),
    )
    def test_error_if_differentiate_diff_method_None(self, interface):
        """Test that an error is raised if differentiating a qnode with diff_method=None"""

        @qml.qnode(qml.device("reference.qubit", wires=1), diff_method=None)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.Z(0))

        x = qml.math.asarray(0.5, like=interface, requires_grad=True)

        res = circuit(x)  # execution works fine
        assert qml.math.allclose(res, np.cos(0.5))

        with pytest.raises(QuantumFunctionError, match="with diff_method=None"):
            qml.math.grad(circuit)(x)


class TestShots:
    """Unit tests for specifying shots per call."""

    def test_shots_setter_error(self):
        """Tests that an error is raised when setting shots on a QNode."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(AttributeError, match="Shots cannot be set on a qnode instance"):
            circuit.shots = 5

    # pylint: disable=unexpected-keyword-arg
    def test_specify_shots_per_call_sample(self):
        """Tests that shots can be set per call for a sample return type."""
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="shots on device is deprecated",
        ):
            dev = qml.device("default.qubit", wires=1, shots=10)

        @qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8)) == 10
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="Specifying 'shots' when executing a QNode is deprecated",
        ):
            assert len(circuit(0.8, shots=2)) == 2
            assert len(circuit(0.8, shots=3178)) == 3178
        assert len(circuit(0.8)) == 10

    # pylint: disable=unexpected-keyword-arg, protected-access
    def test_specify_shots_per_call_expval(self):
        """Tests that shots can be set per call for an expectation value.
        Note: this test has a vanishingly small probability to fail."""
        dev = qml.device("default.qubit", wires=1)

        @qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        # check that the circuit is analytic
        res1 = [circuit() for _ in range(100)]
        assert np.std(res1) == 0.0
        assert circuit.device._shots.total_shots is None

        # check that the circuit is temporary non-analytic
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="Specifying 'shots' when executing a QNode is deprecated",
        ):
            res1 = [circuit(shots=1) for _ in range(100)]
        assert np.std(res1) != 0.0

        # check that the circuit is analytic again
        res1 = [circuit() for _ in range(100)]
        assert np.std(res1) == 0.0
        assert circuit.device._shots.total_shots is None

    # pylint: disable=unexpected-keyword-arg
    def test_no_shots_per_call_if_user_has_shots_qfunc_kwarg(self):
        """Tests that the per-call shots overwriting is suspended if user
        has a shots keyword argument, but a warning is raised."""

        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="shots on device is deprecated",
        ):
            dev = qml.device("default.qubit", wires=2, shots=10)

        def circuit(a, shots=0):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = QNode(circuit, dev)

        assert len(circuit(0.8)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8)
        assert tape.operations[0].wires.labels == (0,)

        assert len(circuit(0.8, shots=1)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, shots=1)
        assert tape.operations[0].wires.labels == (1,)

        assert len(circuit(0.8, shots=0)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, shots=0)
        assert tape.operations[0].wires.labels == (0,)

    # pylint: disable=unexpected-keyword-arg
    def test_no_shots_per_call_if_user_has_shots_qfunc_arg(self):
        """Tests that the per-call shots overwriting is suspended
        if user has a shots argument, but a warning is raised."""
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="shots on device is deprecated",
        ):
            dev = qml.device("default.qubit", wires=[0, 1], shots=10)

        def ansatz0(a, shots):
            qml.RX(a, wires=shots)
            return qml.sample(qml.PauliZ(wires=0))

        # assert that warning is still raised
        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):
            circuit = QNode(ansatz0, dev)

        assert len(circuit(0.8, 1)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, 1)
        assert tape.operations[0].wires.labels == (1,)

        dev = qml.device("default.qubit", wires=2)

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):

            @qnode(dev, shots=10)
            def ansatz1(a, shots):
                qml.RX(a, wires=shots)
                return qml.sample(qml.PauliZ(wires=0))

        assert len(ansatz1(0.8, shots=0)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, 0)
        assert tape.operations[0].wires.labels == (0,)

    def test_warning_finite_shots_dev(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(5)
        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # no warning on the first execution
        circuit(0.3)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            circuit(0.3)

    # pylint: disable=unexpected-keyword-arg
    def test_warning_finite_shots_override(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(5)
        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # no warning on the first execution
        circuit(0.3)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            qml.set_shots(circuit, shots=5)(0.3)

    def test_warning_finite_shots_tape(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = qml.device("default.qubit", wires=1)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RZ(0.3, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = QuantumScript.from_queue(q, shots=5)
        # no warning on the first execution
        cache = {}
        qml.execute([tape], dev, None, cache=cache)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            qml.execute([tape], dev, None, cache=cache)

        cache2 = {}
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            qml.execute([tape, tape], dev, cache=cache2)

    def test_no_caching_by_default(self):
        """Test that caching is turned off by default."""

        dev = qml.device("default.qubit", wires=1)

        tape = qml.tape.QuantumScript([qml.H(0)], [qml.sample(wires=0)], shots=1000)

        res1, res2 = qml.execute([tape, tape], dev)
        assert not np.allclose(res1, res2)

    def test_no_warning_infinite_shots(self):
        """Tests that no warning is raised when caching is used with infinite shots."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Cached execution with finite shots detected")
            circuit(0.3)
            circuit(0.3)

    @pytest.mark.autograd
    def test_no_warning_internal_cache_reuse(self):
        """Tests that no warning is raised when only the internal cache is reused."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(5)
        @qml.qnode(dev, cache=True)
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.probs(wires=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Cached execution with finite shots detected")
            qml.jacobian(circuit, argnum=0)(0.3)

    # pylint: disable=unexpected-keyword-arg
    @pytest.mark.parametrize(
        "shots, total_shots, shot_vector",
        [
            (None, None, ()),
            (1, 1, ((1, 1),)),
            (10, 10, ((10, 1),)),
            ([1, 1, 2, 3, 1], 8, ((1, 2), (2, 1), (3, 1), (1, 1))),
        ],
    )
    def test_tape_shots_set_on_call(self, shots, total_shots, shot_vector):
        """test that shots are placed on the tape if they are specified during a call."""
        dev = qml.device("default.qubit", wires=2)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        # No override
        tape = qml.workflow.construct_tape(qn)(0.1, 0.2)
        assert tape.shots.total_shots is None

        # Override
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="Specifying 'shots' when executing a QNode is deprecated",
        ):
            tape = qml.workflow.construct_tape(qn)(0.1, 0.2, shots=shots)
        assert tape.shots.total_shots == total_shots
        assert tape.shots.shot_vector == shot_vector

        # Decorator syntax
        @qnode(dev)
        def qn2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.PauliZ(0))

        # No override
        tape = qml.workflow.construct_tape(qn2)(0.1, 0.2)
        assert tape.shots.total_shots is None

        # Override
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="Specifying 'shots' when executing a QNode is deprecated",
        ):
            tape = qml.workflow.construct_tape(qn2)(0.1, 0.2, shots=shots)
        assert tape.shots.total_shots == total_shots
        assert tape.shots.shot_vector == shot_vector

    def test_shots_not_updated_with_device(self):
        """Test that _shots is not updated when updating the QNode with a new device."""
        with pytest.warns(PennyLaneDeprecationWarning, match="shots on device is deprecated"):
            dev1 = qml.device("default.qubit", wires=1, shots=100)
        qn = qml.QNode(dummyfunc, dev1)
        assert qn._shots == qml.measurements.Shots(100)

        # _shots should take precedence over device shots
        with pytest.warns(PennyLaneDeprecationWarning, match="shots on device is deprecated"):
            dev2 = qml.device("default.qubit", wires=1, shots=200)
        with pytest.warns(
            UserWarning, match="The device's shots value does not match the QNode's shots value."
        ):
            updated_qnode = qn.update(device=dev2)
        assert updated_qnode._shots == qml.measurements.Shots(100)

    def test_shots_preserved_in_other_updates(self):
        """Test that _shots is preserved when updating other QNode parameters."""
        dev = qml.device("default.qubit", wires=1)
        qn = qml.set_shots(qml.QNode(dummyfunc, dev), shots=50)
        assert qn._shots == qml.measurements.Shots(50)

        # Update something unrelated to shots or device
        updated_qnode = qn.update(diff_method="parameter-shift")
        assert updated_qnode._shots == qml.measurements.Shots(50)

    def test_shots_direct_update(self):
        """Test that QNode shots can be updated directly using the update_shots method."""
        dev = qml.device("default.qubit", wires=1)
        qn = qml.set_shots(qml.QNode(dummyfunc, dev), shots=30)
        assert qn._shots == qml.measurements.Shots(30)

        # Update shots directly using update_shots method
        updated_qnode = qn.update_shots(shots=75)
        assert updated_qnode._shots == qml.measurements.Shots(75)


class TestTransformProgramIntegration:
    """Tests for the integration of the transform program with the qnode."""

    def test_transform_program_modifies_circuit(self):
        """Test qnode integration with a transform that turns the circuit into just a pauli x."""
        dev = qml.device("default.qubit", wires=1)

        def null_postprocessing(results):
            return results[0]

        @qml.transform
        def just_pauli_x_out(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript([qml.PauliX(0)], tape.measurements),
            ), null_postprocessing

        @just_pauli_x_out
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.transform_program[0].transform == just_pauli_x_out.transform

        assert qml.math.allclose(circuit(0.1), -1)

        with circuit.device.tracker as tracker:
            circuit(0.1)

        assert tracker.totals["executions"] == 1
        assert tracker.history["resources"][0].gate_types["PauliX"] == 1
        assert tracker.history["resources"][0].gate_types["RX"] == 0

    def tet_transform_program_modifies_results(self):
        """Test integration with a transform that modifies the result output."""

        dev = qml.device("default.qubit", wires=2)

        @qml.transform
        def pin_result(
            tape: QuantumScript, requested_result
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            def postprocessing(_: qml.typing.ResultBatch) -> qml.typing.Result:
                return requested_result

            return (tape,), postprocessing

        @partial(pin_result, requested_result=3.0)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.transform_program[0].transform == pin_result.transform
        assert circuit.transform_program[0].kwargs == {"requested_result": 3.0}

        assert qml.math.allclose(circuit(0.1), 3.0)

    def test_transform_order_circuit_processing(self):
        """Test that transforms are applied in the correct order in integration."""

        dev = qml.device("default.qubit", wires=2)

        def null_postprocessing(results):
            return results[0]

        @qml.transform
        def just_pauli_x_out(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript([qml.PauliX(0)], tape.measurements),
            ), null_postprocessing

        @qml.transform
        def repeat_operations(
            tape: QuantumScript,
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            new_tape = qml.tape.QuantumScript(
                tape.operations + copy.deepcopy(tape.operations), tape.measurements
            )
            return (new_tape,), null_postprocessing

        @repeat_operations
        @just_pauli_x_out
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit1(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        with circuit1.device.tracker as tracker:
            assert qml.math.allclose(circuit1(0.1), 1.0)

        assert tracker.history["resources"][0].gate_types["PauliX"] == 2

        @just_pauli_x_out
        @repeat_operations
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit2(x):
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        with circuit2.device.tracker as tracker:
            assert qml.math.allclose(circuit2(0.1), -1.0)

        assert tracker.history["resources"][0].gate_types["PauliX"] == 1

    def test_transform_order_postprocessing(self):
        """Test that transform postprocessing is called in the right order."""

        dev = qml.device("default.qubit", wires=2)

        def scale_by_factor(results, factor):
            return results[0] * factor

        def add_shift(results, shift):
            return results[0] + shift

        @qml.transform
        def scale_output(
            tape: QuantumScript, factor
        ) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (tape,), partial(scale_by_factor, factor=factor)

        @qml.transform
        def shift_output(tape: QuantumScript, shift) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (tape,), partial(add_shift, shift=shift)

        @partial(shift_output, shift=1.0)
        @partial(scale_output, factor=2.0)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit1():
            return qml.expval(qml.PauliZ(0))

        # first add one, then scale by 2.0.  Outer postprocessing transforms are applied first
        assert qml.math.allclose(circuit1(), 4.0)

        @partial(scale_output, factor=2.0)
        @partial(shift_output, shift=1.0)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit2():
            return qml.expval(qml.PauliZ(0))

        # first scale by 2, then add one. Outer postprocessing transforms are applied first
        assert qml.math.allclose(circuit2(), 3.0)

    def test_scaling_shots_transform(self):
        """Test a transform that scales the number of shots used in an execution."""

        # note that this won't work with the old device interface :(
        dev = qml.devices.DefaultQubit()

        def num_of_shots_from_sample(results):
            return len(results[0])

        @qml.transform
        def use_n_shots(tape: QuantumScript, n) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            return (
                qml.tape.QuantumScript(tape.operations, tape.measurements, shots=n),
            ), num_of_shots_from_sample

        @partial(use_n_shots, n=100)
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit():
            return qml.sample(wires=0)

        assert circuit() == 100


class TestNewDeviceIntegration:
    """Basic tests for integration of the new device interface and the QNode."""

    dev = CustomDevice()

    def test_initialization(self):
        """Test that a qnode can be initialized with the new device without error."""

        def f():
            return qml.expval(qml.PauliZ(0))

        qn = QNode(f, self.dev)
        assert qn.device is self.dev

    def test_repr(self):
        """Test that the repr works with the new device."""

        def f():
            return qml.expval(qml.PauliZ(0))

        qn = QNode(f, self.dev)
        assert (
            repr(qn)
            == "<QNode: device='CustomDevice', interface='auto', diff_method='best', shots='Shots(total=None)'>"
        )

    def test_device_with_custom_diff_method_name(self):
        """Test a device that has its own custom diff method."""

        class CustomDeviceWithDiffMethod2(qml.devices.DefaultQubit):
            """A device with a custom derivative named hello."""

            def supports_derivatives(self, execution_config=None, circuit=None):
                return getattr(execution_config, "gradient_method", None) == "hello"

            def setup_execution_config(
                self, config: qml.devices.ExecutionConfig | None = None, circuit=None
            ):
                if config.gradient_method in {"best", "hello"}:
                    return replace(config, gradient_method="hello", use_device_gradient=True)
                return config

            def compute_derivatives(
                self, circuits, execution_config: qml.devices.ExecutionConfig | None = None
            ):
                if self.tracker.active:
                    self.tracker.update(derivative_config=execution_config)
                    self.tracker.record()
                return super().compute_derivatives(circuits, execution_config)

        dev = CustomDeviceWithDiffMethod2()

        @qml.qnode(dev, diff_method="hello")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit.diff_method == "hello"

        with dev.tracker:
            qml.grad(circuit)(qml.numpy.array(0.5))

        assert dev.tracker.history["derivative_config"][0].gradient_method == "hello"
        assert dev.tracker.history["derivative_batches"] == [1]

    def test_shots_integration(self):
        """Test that shots provided at call time are passed through the workflow."""

        dev = qml.devices.DefaultQubit()

        @qml.qnode(dev, diff_method=None)
        def circuit():
            return qml.sample(wires=(0, 1))

        with pytest.raises(DeviceError, match="not accepted for analytic simulation"):
            circuit()

        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="Specifying 'shots' when executing a QNode is deprecated",
        ):
            results = circuit(shots=10)  # pylint: disable=unexpected-keyword-arg
            assert qml.math.allclose(results, np.zeros((10, 2)))

            results = circuit(shots=20)  # pylint: disable=unexpected-keyword-arg
            assert qml.math.allclose(results, np.zeros((20, 2)))


class TestMCMConfiguration:
    """Tests for MCM configuration arguments"""

    def test_one_shot_error_without_shots(self):
        """Test that an error is raised if mcm_method="one-shot" with no shots"""
        dev = qml.device("default.qubit", wires=3)
        param = np.pi / 4

        @qml.qnode(dev, mcm_method="one-shot")
        def f(x):
            qml.RX(x, 0)
            _ = qml.measure(0)
            return qml.probs(wires=[0, 1])

        with pytest.raises(
            ValueError,
            match="Cannot use the 'one-shot' method for mid-circuit measurements with",
        ):
            f(param)

    def test_invalid_postselect_mode_error(self):
        """Test that an error is raised if the requested postselect_mode is invalid"""
        shots = 100
        dev = qml.device("default.qubit", wires=3)

        def f(x):
            qml.RX(x, 0)
            _ = qml.measure(0, postselect=1)
            return qml.sample(wires=[0, 1])

        with pytest.raises(ValueError, match="Invalid postselection mode 'foo'"):
            _ = qml.set_shots(qml.QNode(f, dev, postselect_mode="foo"), shots=shots)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", [None, "best"])
    def test_defer_measurements_with_jit(self, diff_method, mocker, seed):
        """Test that using mcm_method="deferred" defaults to behaviour like
        postselect_mode="fill-shots" when using jax jit."""
        import jax  # pylint: disable=import-outside-toplevel

        shots = 100
        postselect = 1
        param = jax.numpy.array(np.pi / 2)
        spy = mocker.spy(qml.defer_measurements, "_transform")
        spy_one_shot = mocker.spy(qml.dynamic_one_shot, "_transform")

        dev = qml.device("default.qubit", wires=4, seed=jax.random.PRNGKey(seed))

        @qml.set_shots(shots)
        @qml.qnode(dev, diff_method=diff_method, mcm_method="deferred")
        def f(x):
            qml.RX(x, 0)
            qml.measure(0, postselect=postselect)
            return qml.sample(wires=0)

        f_jit = jax.jit(f)
        res = f(param)
        res_jit = f_jit(param)

        assert spy.call_count > 0
        spy_one_shot.assert_not_called()

        assert len(res) < shots
        assert len(res_jit) == shots
        assert qml.math.allclose(res, postselect)
        assert qml.math.allclose(res_jit, postselect)

    @pytest.mark.jax
    @pytest.mark.parametrize("diff_method", [None, "best"])
    def test_deferred_hw_like_error_with_jit(self, diff_method, seed):
        """Test that an error is raised if attempting to use postselect_mode="hw-like"
        with jax jit with mcm_method="deferred"."""
        import jax  # pylint: disable=import-outside-toplevel

        shots = 100
        postselect = 1
        param = jax.numpy.array(np.pi / 2)

        dev = qml.device("default.qubit", wires=4, seed=jax.random.PRNGKey(seed))

        @qml.set_shots(shots)
        @qml.qnode(dev, diff_method=diff_method, mcm_method="deferred", postselect_mode="hw-like")
        def f(x):
            qml.RX(x, 0)
            qml.measure(0, postselect=postselect)
            return qml.sample(wires=0)

        f_jit = jax.jit(f)

        # Checking that an error is not raised without jit
        _ = f(param)

        with pytest.raises(
            ValueError, match="Using postselect_mode='hw-like' is not supported with jax-jit."
        ):
            _ = f_jit(param)

    def test_single_branch_statistics_error_without_qjit(self):
        """Test that an error is raised if attempting to use mcm_method="single-branch-statistics
        without qml.qjit"""
        dev = qml.device("reference.qubit", wires=1)

        @qml.qnode(dev, mcm_method="single-branch-statistics")
        def circuit(x):
            qml.RX(x, 0)
            qml.measure(0, postselect=1)
            return qml.sample(wires=0)

        param = np.pi / 4
        with pytest.raises(ValueError, match="Cannot use mcm_method='single-branch-statistics'"):
            _ = circuit(param)

    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", [None, "one-shot", "deferred"])
    def test_execution_does_not_mutate_config(self, mcm_method, postselect_mode):
        """Test that executing a QNode does not mutate its mid-circuit measurement config options"""

        if postselect_mode == "fill-shots" and mcm_method != "deferred":
            pytest.skip("fill-shots is disabled for anything but deferred.")

        dev = qml.device("default.qubit", wires=2)

        original_config = qml.devices.MCMConfig(
            postselect_mode=postselect_mode, mcm_method=mcm_method
        )

        @qml.qnode(dev, postselect_mode=postselect_mode, mcm_method=mcm_method)
        def circuit(x, mp):
            qml.RX(x, 0)
            qml.measure(0, postselect=1)
            return mp(qml.PauliZ(0))

        _ = qml.set_shots(circuit, shots=10)(1.8, qml.expval)
        assert circuit.execute_kwargs["postselect_mode"] == original_config.postselect_mode
        assert circuit.execute_kwargs["mcm_method"] == original_config.mcm_method

        if mcm_method != "one-shot":
            _ = circuit(1.8, qml.expval)
            assert circuit.execute_kwargs["postselect_mode"] == original_config.postselect_mode
            assert circuit.execute_kwargs["mcm_method"] == original_config.mcm_method

        _ = qml.set_shots(circuit, shots=10)(1.8, qml.expval)
        assert circuit.execute_kwargs["postselect_mode"] == original_config.postselect_mode
        assert circuit.execute_kwargs["mcm_method"] == original_config.mcm_method


class TestTapeExpansion:
    """Test that tape expansion within the QNode works correctly"""

    @pytest.mark.parametrize(
        "diff_method,grad_on_execution",
        [("parameter-shift", False), ("adjoint", True), ("adjoint", False)],
    )
    def test_device_expansion(self, diff_method, grad_on_execution, mocker):
        """Test expansion of an unsupported operation on the device"""
        dev = qml.device("default.qubit", wires=1)

        # pylint: disable=too-few-public-methods
        class UnsupportedOp(qml.operation.Operation):
            """custom unsupported op."""

            num_wires = 1

            def decomposition(self):
                return [qml.RX(3 * self.data[0], wires=self.wires)]

        @qnode(dev, diff_method=diff_method, grad_on_execution=grad_on_execution)
        def circuit(x):
            UnsupportedOp(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        if diff_method == "adjoint" and grad_on_execution:
            spy = mocker.spy(circuit.device, "execute_and_compute_derivatives")
        else:
            spy = mocker.spy(circuit.device, "execute")

        x = pnp.array(0.5)
        circuit(x)

        tape = spy.call_args[0][0][0]
        assert len(tape.operations) == 1
        assert tape.operations[0].name == "RX"
        assert np.allclose(tape.operations[0].parameters, 3 * x)

    @pytest.mark.autograd
    def test_no_gradient_expansion(self):
        """Test that an unsupported operation with defined gradient recipe is
        not expanded"""
        dev = qml.device("default.qubit", wires=1)

        # pylint: disable=too-few-public-methods
        class UnsupportedOp(qml.operation.Operation):
            """custom unsupported op."""

            num_wires = 1

            grad_method = "A"
            grad_recipe = ([[3 / 2, 1, np.pi / 6], [-3 / 2, 1, -np.pi / 6]],)

            def decomposition(self):
                return [qml.RX(3 * self.data[0], wires=self.wires)]

        @qnode(dev, interface="autograd", diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            UnsupportedOp(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        x = pnp.array(0.5, requires_grad=True)
        qml.grad(circuit)(x)

        # check second derivative
        assert np.allclose(qml.grad(qml.grad(circuit))(x), -9 * np.cos(3 * x))

    @pytest.mark.autograd
    def test_gradient_expansion(self, mocker):
        """Test that a *supported* operation with no gradient recipe is
        expanded when applying the gradient transform, but not for execution."""
        dev = qml.device("default.qubit", wires=1)

        # pylint: disable=too-few-public-methods
        class PhaseShift(qml.PhaseShift):
            """custom phase shift."""

            grad_method = None

            def decomposition(self):
                return [qml.RY(3 * self.data[0], wires=self.wires)]

        @qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.Hadamard(wires=0)
            PhaseShift(x, wires=0)
            return qml.expval(qml.PauliX(0))

        x = pnp.array(0.5, requires_grad=True)
        circuit(x)

        res = qml.grad(circuit)(x)

        assert np.allclose(res, -3 * np.sin(3 * x))

        # test second order derivatives
        res = qml.grad(qml.grad(circuit))(x)
        assert np.allclose(res, -9 * np.cos(3 * x))

    def test_hamiltonian_expansion_analytic(self):
        """Test result if there are non-commuting groups and the number of shots is None"""
        dev = qml.device("default.qubit", wires=3)

        obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
        c = np.array([-0.6543, 0.24, 0.54])
        H = qml.Hamiltonian(c, obs)
        H.compute_grouping()

        assert len(H.grouping_indices) == 2

        @qnode(dev)
        def circuit():
            return qml.expval(H)

        res = circuit()
        assert np.allclose(res, c[2], atol=0.1)


def test_resets_after_execution_error():
    """Test that the interface is reset to ``"auto"`` if an error occurs during execution."""

    # pylint: disable=too-few-public-methods
    class BadOp(qml.operation.Operator):
        """An operator that will cause an error during execution."""

    @qml.qnode(qml.device("default.qubit"))
    def circuit(x):
        BadOp(x, wires=0)
        return qml.state()

    with pytest.raises(DeviceError):
        circuit(qml.numpy.array(0.1))

    assert circuit.interface == "auto"


class TestPrivateFunctions:
    """Tests for private functions in the QNode class."""

    def test_make_execution_config_with_no_qnode(self):
        """Test that the _make_execution_config function correctly creates an execution config."""
        diff_method = "best"
        mcm_config = qml.devices.MCMConfig(postselect_mode="fill-shots", mcm_method="deferred")
        config = _make_execution_config(None, diff_method, mcm_config)

        expected_config = qml.devices.ExecutionConfig(
            interface="numpy",
            gradient_keyword_arguments={},
            use_device_jacobian_product=False,
            grad_on_execution=None,
            gradient_method=diff_method,
            mcm_config=mcm_config,
        )

        assert config == expected_config

    @pytest.mark.parametrize("interface", ["autograd", "torch", "jax", "jax-jit"])
    def test_make_execution_config_with_qnode(self, interface):
        """Test that a execution config is made correctly with no QNode."""
        if "jax" in interface:
            grad_on_execution = False
        else:
            grad_on_execution = None

        @qml.qnode(qml.device("default.qubit"), interface=interface)
        def circuit():
            qml.H(0)
            return qml.probs()

        diff_method = "best"
        mcm_config = qml.devices.MCMConfig(postselect_mode="fill-shots", mcm_method="deferred")
        config = _make_execution_config(circuit, diff_method, mcm_config)

        expected_config = qml.devices.ExecutionConfig(
            interface=interface,
            gradient_keyword_arguments={},
            use_device_jacobian_product=False,
            grad_on_execution=grad_on_execution,
            gradient_method=diff_method,
            mcm_config=mcm_config,
        )

        assert config == expected_config


class TestSetShots:
    """Tests for the set_shots decorator functionality."""

    def test_shots_initialization(self):
        """Test that _shots is correctly initialized from the device with deprecation warning."""
        # Test that QNode inherits shots from device (with deprecation warning)
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="shots on device is deprecated",
        ):
            dev_with_shots = qml.device("default.qubit", wires=1, shots=42)
        qn_with_device_shots = qml.QNode(dummyfunc, dev_with_shots)
        assert qn_with_device_shots._shots == qml.measurements.Shots(42)

        # Test that QNode defaults to analytic mode when device has no shots
        dev_analytic = qml.device("default.qubit", wires=1)
        qnode_analytic = qml.QNode(dummyfunc, dev_analytic)
        assert qnode_analytic._shots.total_shots is None

        # Test that qnode creation with shots parameter initializes _shots correctly
        qn_with_shots = qml.QNode(dummyfunc, dev_with_shots)
        assert qn_with_shots._shots == qml.measurements.Shots(42)

    def test_shots_override_warning(self):
        """Test that a warning is raised when both set_shots transform and shots parameter are used."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.sample(qml.PauliZ(0))

        # First apply set_shots to override device shots
        modified_circuit = qml.set_shots(circuit, shots=50)
        assert modified_circuit._shots == qml.measurements.Shots(50)
        assert modified_circuit._shots_override_device is True

        # Then try to pass shots parameter when calling the QNode
        with pytest.warns(
            PennyLaneDeprecationWarning,
            match="Specifying 'shots' when executing a QNode is deprecated",
        ):
            with pytest.warns(
                UserWarning,
                match="Both 'shots=' parameter and 'set_shots' transform are specified. "
                "The transform will take precedence over",
            ):
                result = modified_circuit(shots=25)

        # Verify that the set_shots value (50) was used, not the parameter value (25)
        assert len(result) == 50

    def test_set_shots_direct_decorator(self):
        """Test set_shots with partial decorator syntax."""
        dev = qml.device("default.qubit", wires=1)

        @set_shots(shots=50)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        assert circuit._shots == qml.measurements.Shots(50)
        result = circuit()
        assert len(result) == 50

    def test_set_shots_partial_decorator(self):
        """Test set_shots with partial decorator syntax."""
        dev = qml.device("default.qubit", wires=1)

        @partial(set_shots, shots=50)
        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        assert circuit._shots == qml.measurements.Shots(50)
        result = circuit()
        assert len(result) == 50

    def test_set_shots_direct_application_argshots(self):
        """Test applying set_shots directly to an existing QNode."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def original_circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        # Apply set_shots directly
        new_circuit = set_shots(original_circuit, 75)

        assert original_circuit._shots == qml.measurements.Shots(10)
        assert new_circuit._shots == qml.measurements.Shots(75)

        result = new_circuit()
        assert len(result) == 75

    def test_set_shots_direct_application(self):
        """Test applying set_shots directly to an existing QNode."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def original_circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        # Apply set_shots directly
        new_circuit = set_shots(original_circuit, shots=75)

        assert original_circuit._shots == qml.measurements.Shots(10)
        assert new_circuit._shots == qml.measurements.Shots(75)

        result = new_circuit()
        assert len(result) == 75

    def test_set_shots_with_shot_vector(self):
        """Test set_shots with shot vectors."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        shot_vector = [(10, 3), (5, 2)]  # 10 shots × 3 times, 5 shots × 2 times
        new_circuit = set_shots(circuit, shots=shot_vector)

        assert new_circuit._shots.total_shots == 40  # 10*3 + 5*2
        assert new_circuit._shots.shot_vector == ((10, 3), (5, 2))
        result = new_circuit()
        assert isinstance(result, tuple)
        assert len(result) == 5  # 3 + 2 groups

    def test_set_shots_analytic_mode(self):
        """Test set_shots with None for analytic mode."""
        dev = qml.device("default.qubit", wires=1)

        @partial(set_shots, shots=None)
        @qml.set_shots(100)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit._shots is None or circuit._shots.total_shots is None
        result = circuit()
        assert isinstance(result, (float, np.floating))
        assert qml.math.allclose(result, np.cos(1.0))

    def test_set_shots_preserves_original_qnode(self):
        """Test that set_shots creates a new QNode without modifying the original."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(20)
        @qml.qnode(dev)
        def original_circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        new_circuit = set_shots(original_circuit, shots=100)

        # Original should be unchanged
        assert original_circuit._shots == qml.measurements.Shots(20)
        # New circuit should have updated shots
        assert new_circuit._shots == qml.measurements.Shots(100)

        # Both should be different objects
        assert original_circuit is not new_circuit

    @pytest.mark.parametrize(
        "invalid_input,expected_error",
        [
            ("not a function", "set_shots can only be applied to QNodes"),
            (42, "set_shots can only be applied to QNodes"),
            (lambda: 42, "set_shots can only be applied to QNodes"),
            (None, "set_shots can only be applied to QNodes"),
        ],
    )
    def test_set_shots_error_on_invalid_input(self, invalid_input, expected_error):
        """Test that set_shots raises appropriate errors for invalid inputs."""
        with pytest.raises(ValueError, match=expected_error):
            set_shots(invalid_input, shots=100)

    @pytest.mark.parametrize(
        "args,kwargs,error_message",
        [
            ((), {}, "Invalid arguments to set_shots"),
            ((100, 200), {}, "set_shots can only be applied to QNodes"),
        ],
    )
    def test_set_shots_error_on_invalid_argument_patterns(self, args, kwargs, error_message):
        """Test that set_shots raises ValueError for invalid argument patterns."""
        with pytest.raises(ValueError, match=error_message):
            set_shots(*args, **kwargs)

    def test_set_shots_with_measurements_requiring_shots(self):
        """Test set_shots works correctly with measurements that require shots."""
        dev = qml.device("default.qubit", wires=2)

        @partial(set_shots, shots=1000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts()

        result = circuit()
        assert isinstance(result, dict)
        assert sum(result.values()) == 1000

    def test_set_shots_preserves_qnode_properties(self):
        """Test that set_shots preserves other QNode properties."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev, diff_method="parameter-shift", interface="autograd")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        new_circuit = set_shots(circuit, shots=100)

        # Check that other properties are preserved
        assert new_circuit.diff_method == circuit.diff_method
        assert new_circuit.interface == circuit.interface
        assert new_circuit.device is circuit.device

    @pytest.mark.parametrize(
        "original_shots,override_shots,expected_tracking",
        [
            (10, None, "no_tracking"),  # finite -> analytic
            (None, 20, "shots_tracked"),  # analytic -> finite
        ],
    )
    def test_device_shots_override_tracking(
        self, original_shots, override_shots, expected_tracking
    ):
        """Test that QNode shots properly override device shots during execution."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(original_shots)
        @qml.qnode(dev, diff_method=None)
        def circuit():
            qml.RX(1.23, wires=0)
            return qml.probs(wires=0)

        # Test override behavior
        modified_circuit = set_shots(circuit, shots=override_shots)
        with qml.Tracker(dev) as tracker:
            modified_circuit()

        if expected_tracking == "no_tracking":
            assert "shots" not in tracker.history
        else:
            assert tracker.history["shots"][-1] == override_shots

    @pytest.mark.parametrize(
        "device_shots,override_shots,expected_executions",
        [
            (100, None, (3, 1)),  # shots -> analytic: param-shift to backprop
            (None, 100, (1, 3)),  # analytic -> shots: backprop to param-shift
        ],
    )
    def test_diff_method_adaptation(self, device_shots, override_shots, expected_executions):
        """Test that diff_method adapts when shots change."""
        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(device_shots)
        @qml.qnode(dev, diff_method="best")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        param = qml.numpy.array(0.5, requires_grad=True)

        # Test original execution count
        with qml.Tracker(dev) as tracker:
            qml.grad(circuit)(param)
        assert len(tracker.history["executions"]) == expected_executions[0]

        # Test modified execution count
        modified_circuit = set_shots(circuit, shots=override_shots)
        with qml.Tracker(dev) as tracker:
            qml.grad(modified_circuit)(param)
        assert len(tracker.history["executions"]) == expected_executions[1]

    def test_set_shots_integer_to_shot_vector(self):
        """Test converting from integer shots to shot vector."""
        dev = qml.device("default.qubit", wires=1)  # Start with integer shots

        @qml.set_shots(50)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        # Verify original has integer shots
        assert circuit._shots == qml.measurements.Shots(50)
        original_result = circuit()
        assert len(original_result) == 50
        assert not isinstance(original_result, tuple)  # Single array, not tuple of arrays

        # Convert to shot vector
        shot_vector = [(20, 2), (15, 3)]  # 20 shots × 2 times, 15 shots × 3 times = 85 total
        vector_circuit = set_shots(circuit, shots=shot_vector)

        # Verify shot vector conversion
        assert vector_circuit._shots.total_shots == 85  # 20*2 + 15*3
        assert vector_circuit._shots.shot_vector == ((20, 2), (15, 3))

        vector_result = vector_circuit()
        assert isinstance(vector_result, tuple)  # Now returns tuple of arrays
        assert len(vector_result) == 5  # 2 + 3 groups

        # Verify group sizes
        assert len(vector_result[0]) == 20  # First group: 20 shots
        assert len(vector_result[1]) == 20  # Second group: 20 shots
        assert len(vector_result[2]) == 15  # Third group: 15 shots
        assert len(vector_result[3]) == 15  # Fourth group: 15 shots
        assert len(vector_result[4]) == 15  # Fifth group: 15 shots

        # Original circuit should be unchanged
        assert circuit._shots == qml.measurements.Shots(50)

    def test_no_warning_if_shots_not_updated(self):
        """Test that no warning is raised if set_shots is called but the shots value is unchanged."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0))

        # No warning should be raised when calling with the same shots value
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            with pytest.warns(
                PennyLaneDeprecationWarning,
                match="Specifying 'shots' when executing a QNode is deprecated",
            ):
                result = circuit.update(diff_method="parameter-shift")(shots=50)
        # Filter for targeted warnings (by type and/or message)
        targeted = [
            w
            for w in record
            if issubclass(w.category, UserWarning)
            and "Both 'shots=' parameter and 'set_shots' transform are specified." in str(w.message)
        ]
        assert len(targeted) == 0
        assert len(result) == 50

    def test_no_warning_if_shots_not_updated_set_shots(self):
        """Test that no warning is raised if set_shots is called but the shots value is unchanged."""
        dev = qml.device("default.qubit")

        @partial(qml.set_shots, shots=100)
        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0))

        # No warning should be raised when calling with the same shots value
        result = circuit.update(diff_method="parameter-shift")()
        assert len(result) == 100

    def test_set_shots_positional_preserves_qnode_properties(self):
        """Test that @set_shots(500) preserves QNode properties."""
        dev = qml.device("default.qubit", wires=1)

        @set_shots(100)
        @qml.qnode(dev, diff_method="parameter-shift", interface="autograd")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Check that shots are set correctly
        assert circuit._shots == qml.measurements.Shots(100)

        # Check that other properties are preserved
        assert circuit.diff_method == "parameter-shift"
        assert circuit.interface == "autograd"
        assert circuit.device is dev

    def test_set_shots_positional_decorator_integer(self):
        """Test @set_shots(500) positional decorator syntax with integer shots."""
        dev = qml.device("default.qubit", wires=1)

        @set_shots(500)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        assert circuit._shots == qml.measurements.Shots(500)
        result = circuit()
        assert len(result) == 500

    def test_set_shots_positional_decorator_none(self):
        """Test @set_shots(None) positional decorator syntax for analytic mode."""
        dev = qml.device("default.qubit", wires=1)

        @set_shots(None)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit._shots is None or circuit._shots.total_shots is None
        result = circuit()
        assert isinstance(result, (float, np.floating))
        # Analytic result should be cos(1.0)
        assert qml.math.allclose(result, np.cos(1.0))

    def test_set_shots_positional_decorator_shot_vector(self):
        """Test @set_shots(shot_vector) positional decorator syntax with shot vectors."""
        dev = qml.device("default.qubit", wires=1)

        shot_vector = [(20, 2), (15, 3)]  # 20 shots × 2, 15 shots × 3

        @set_shots(shot_vector)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            return qml.sample(qml.PauliZ(0))

        assert circuit._shots.total_shots == 85  # 20*2 + 15*3
        assert circuit._shots.shot_vector == ((20, 2), (15, 3))
        result = circuit()
        assert isinstance(result, tuple)
        assert len(result) == 5  # 2 + 3 groups

    def test_set_shots_positional_vs_keyword_equivalence(self):
        """Test that @set_shots(500) and @set_shots(shots=500) are equivalent."""
        dev = qml.device("default.qubit", wires=1)

        @set_shots(None)  # Use analytic mode for exact comparison
        @qml.qnode(dev)
        def positional_circuit():
            qml.RX(1.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        @set_shots(shots=None)  # Use analytic mode for exact comparison
        @qml.qnode(dev)
        def keyword_circuit():
            qml.RX(1.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Both should have analytic mode
        assert positional_circuit._shots is None or positional_circuit._shots.total_shots is None
        assert keyword_circuit._shots is None or keyword_circuit._shots.total_shots is None

        # Both should produce the same results for analytic case
        pos_result = positional_circuit()
        kw_result = keyword_circuit()
        assert qml.math.allclose(pos_result, kw_result)

        # Should equal the expected analytic result
        expected = np.cos(1.0)
        assert qml.math.allclose(pos_result, expected)
        assert qml.math.allclose(kw_result, expected)

    def test_set_shots_positional_vs_direct_equivalence(self):
        """Test that @set_shots(500) and set_shots(qnode, shots=500) are equivalent."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def base_circuit():
            qml.RX(1.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        @set_shots(None)  # Use analytic mode for exact comparison
        @qml.qnode(dev)
        def positional_circuit():
            qml.RX(1.0, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Apply set_shots directly to base circuit
        direct_circuit = set_shots(base_circuit, shots=None)

        # Both should have analytic mode
        assert positional_circuit._shots is None or positional_circuit._shots.total_shots is None
        assert direct_circuit._shots is None or direct_circuit._shots.total_shots is None

        # Both should produce the same results
        pos_result = positional_circuit()
        direct_result = direct_circuit()
        assert qml.math.allclose(pos_result, direct_result)

        # Should equal the expected analytic result
        expected = np.cos(1.0)
        assert qml.math.allclose(pos_result, expected)
        assert qml.math.allclose(direct_result, expected)

    def test_set_shots_positional_analytic_vs_shot_difference(self):
        """Test that @set_shots(None) vs @set_shots(1000) produce different behaviors."""
        dev = qml.device("default.qubit", wires=1)

        @set_shots(None)
        @qml.qnode(dev)
        def analytic_circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        @set_shots(1000)
        @qml.qnode(dev)
        def shot_circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))

        # Analytic should be exactly 0.0
        analytic_results = [analytic_circuit() for _ in range(10)]
        assert all(result == 0.0 for result in analytic_results)

        # Shot-based should have variance
        shot_results = [shot_circuit() for _ in range(10)]
        # Should be close to 0 but with some variance
        assert abs(np.mean(shot_results)) < 0.1  # Should be close to 0
        assert np.std(shot_results) > 0.001  # Should have some variance

    def test_set_shots_positional_complex_shot_patterns(self):
        """Test @set_shots with complex shot patterns."""
        dev = qml.device("default.qubit", wires=2)

        # Test with tuple (shots, copies) format
        @set_shots([(10, 5), (20, 3), 50])
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample()

        # Total shots: 10*5 + 20*3 + 50 = 50 + 60 + 50 = 160
        assert circuit._shots.total_shots == 160
        result = circuit()
        assert isinstance(result, tuple)
        # Should have 5 + 3 + 1 = 9 result groups
        assert len(result) == 9

    def test_set_shots_positional_nested_decorators(self):
        """Test @set_shots works with other decorators."""
        dev = qml.device("default.qubit", wires=1)

        @qml.transforms.cancel_inverses  # Another transform
        @set_shots(250)
        @qml.qnode(dev)
        def circuit():
            qml.RX(1.0, wires=0)
            qml.RX(-1.0, wires=0)  # Should be cancelled
            return qml.sample(qml.PauliZ(0))

        assert circuit._shots == qml.measurements.Shots(250)
        result = circuit()
        assert len(result) == 250

    def test_set_shots_positional_multiple_calls_consistent(self):
        """Test that @set_shots(500) produces consistent behavior across multiple calls."""
        dev = qml.device("default.qubit", wires=1)

        @set_shots(None)  # Analytic mode
        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Multiple calls should give exactly the same result in analytic mode
        results = [circuit() for _ in range(5)]
        expected = np.cos(0.5)

        for result in results:
            assert qml.math.allclose(result, expected)

        # All results should be identical
        assert all(qml.math.allclose(results[0], r) for r in results[1:])
