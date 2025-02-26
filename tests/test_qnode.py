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
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn
from pennylane.workflow.qnode import _make_execution_config


def dummyfunc():
    """dummy func."""
    return None


def test_additional_kwargs_is_deprecated():
    """Test that passing gradient_kwargs as additional kwargs raises a deprecation warning."""
    dev = qml.device("default.qubit", wires=1)

    with pytest.warns(
        qml.PennyLaneDeprecationWarning,
        match=r"Specifying gradient keyword arguments \[\'atol\'\] as additional kwargs has been deprecated",
    ):
        QNode(dummyfunc, dev, atol=1)


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

    with pytest.raises(qml.QuantumFunctionError, match="must return either a single measurement"):
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

        with pytest.warns(
            UserWarning,
            match="Received gradient_kwarg blah, which is not included in the list of standard qnode gradient kwargs.",
        ):
            circuit.update(gradient_kwargs={"blah": 1})

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

    def test_cache_initialization_maxdiff_1(self):
        """Test that when max_diff = 1, the cache initializes to false."""

        @qml.qnode(qml.device("default.qubit"), max_diff=1)
        def f():
            return qml.state()

        assert f.execute_kwargs["cache"] is False

    def test_cache_initialization_maxdiff_2(self):
        """Test that when max_diff = 2, the cache initialization to True."""

        @qml.qnode(qml.device("default.qubit"), max_diff=2)
        def f():
            return qml.state()

        assert f.execute_kwargs["cache"] is True


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

    def test_expansion_strategy_error(self):
        """Test that an error is raised if expansion_strategy is passed to the qnode."""

        with pytest.raises(ValueError, match="'expansion_strategy' is no longer"):

            @qml.qnode(qml.device("default.qubit"), expansion_strategy="device")
            def _():
                return qml.state()

    def test_max_expansion_error(self):
        """Test that an error is raised if max_expansion is passed to the QNode."""

        with pytest.raises(ValueError, match="'max_expansion' is no longer a valid"):

            @qml.qnode(qml.device("default.qubit"), max_expansion=1)
            def _():
                qml.state()

    def test_invalid_interface(self):
        """Test that an exception is raised for an invalid interface"""
        dev = qml.device("default.qubit", wires=1)
        test_interface = "something"
        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(ValueError, match=expected_error):
            QNode(dummyfunc, dev, interface="something")

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

        expected_error = rf"Unknown interface {test_interface}\. Interface must be one of"

        with pytest.raises(ValueError, match=expected_error):
            circuit.interface = test_interface

    def test_invalid_device(self):
        """Test that an exception is raised for an invalid device"""
        with pytest.raises(qml.QuantumFunctionError, match="Invalid device"):
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
            qml.QuantumFunctionError, match="Differentiation method hello not recognized"
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
            qml.QuantumFunctionError,
            match="does not support adjoint with requested circuit",
        ):
            circ(shots=1)

    @pytest.mark.autograd
    def test_sparse_diffmethod_error(self):
        """Test that an error is raised when the observable is SparseHamiltonian and the
        differentiation method is not parameter-shift."""
        dev = qml.device("default.qubit", wires=2, shots=None)

        @qnode(dev, diff_method="backprop")
        def circuit(param):
            qml.RX(param, wires=0)
            return qml.expval(qml.SparseHamiltonian(csr_matrix(np.eye(4)), [0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError, match="does not support backprop with requested circuit"
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
            == f"<QNode: device='<default.qubit device (wires=1) at {hex(id(dev))}>', interface='auto', diff_method='best'>"
        )

        qn = QNode(func, dev, interface="autograd")

        assert (
            repr(qn)
            == f"<QNode: device='<default.qubit device (wires=1) at {hex(id(dev))}>', interface='autograd', diff_method='best'>"
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

    # pylint: disable=unused-variable
    def test_unrecognized_kwargs_raise_warning(self):
        """Test that passing gradient_kwargs not included in qml.gradients.SUPPORTED_GRADIENT_KWARGS raises warning"""
        dev = qml.device("default.qubit", wires=2)

        with warnings.catch_warnings(record=True) as w:

            @qml.qnode(dev, gradient_kwargs={"random_kwarg": qml.gradients.finite_diff})
            def circuit(params):
                qml.RX(params[0], wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

            assert len(w) == 1
            assert "not included in the list of standard qnode gradient kwargs" in str(w[0].message)

    # pylint: disable=unused-variable
    def test_incorrect_diff_method_kwargs_raise_warning(self):
        """Tests that using one of the incorrect kwargs previously used in some examples in PennyLane
        (grad_method, gradient_fn) to set the qnode diff_method raises a warning"""
        dev = qml.device("default.qubit", wires=2)

        with warnings.catch_warnings(record=True) as w:

            @qml.qnode(dev, grad_method=qml.gradients.finite_diff)
            def circuit0(params):
                qml.RX(params[0], wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

            @qml.qnode(dev, gradient_fn=qml.gradients.finite_diff)
            def circuit2(params):
                qml.RX(params[0], wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

        assert len(w) == 2
        assert "Use diff_method instead" in str(w[0].message)
        assert "Use diff_method instead" in str(w[1].message)

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

        dev = qml.device("default.qubit", wires=2, shots=100)

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
        dev = qml.device("default.qubit", wires=2)

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

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
            qn(5, 1)

        def func2(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), 5

        qn = QNode(func2, dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
            qn(5, 1)

        def func3(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return []

        qn = QNode(func3, dev)

        with pytest.raises(
            qml.QuantumFunctionError, match="must return either a single measurement"
        ):
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
            qml.QuantumFunctionError,
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

        dev = qml.device(dev_name, wires=2, shots=5)

        def circuit1(param):
            qml.Hadamard(0)
            qml.RX(param, wires=1)
            qml.CNOT([1, 0])
            return qml.counts()

        qn = qml.QNode(circuit1, dev)
        jitted_qnode1 = jax.jit(qn)

        with pytest.raises(
            NotImplementedError, match="The JAX-JIT interface doesn't support qml.counts."
        ):
            _ = jitted_qnode1(0.123)

        # Test with qnode decorator syntax
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
    @pytest.mark.parametrize("interface", ["autograd", "torch", "tensorflow", "jax"])
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
        dev = qml.device("default.mixed", wires=2, shots=100)

        with pytest.raises(
            TypeError,
            match="does not support mid-circuit measurements and/or one-shot execution mode",
        ):

            @qml.transforms.dynamic_one_shot
            @qml.qnode(dev)
            def _():
                qml.RX(1.23, 0)
                ms = [qml.measure(0) for _ in range(10)]
                return qml.probs(op=ms)

    @pytest.mark.parametrize("basis_state", [[1, 0], [0, 1]])
    def test_sampling_with_mcm(self, basis_state, mocker):
        """Tests that a QNode with qml.sample and mid-circuit measurements
        returns the expected results."""
        dev = qml.device("default.qubit", wires=3, shots=1000)

        first_par = np.pi

        @qml.qnode(dev)
        def cry_qnode(x):
            """QNode where we apply a controlled Y-rotation."""
            qml.BasisState(basis_state, wires=[0, 1])
            qml.CRY(x, wires=[0, 1])
            return qml.sample(qml.PauliZ(1))

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
    @pytest.mark.parametrize("interface", ["tf", "auto"])
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
    @pytest.mark.parametrize("jax_interface", ["jax-python", "jax-jit", "auto"])
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

        with pytest.raises(qml.QuantumFunctionError, match="device_vjp=True is not supported"):
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

        with pytest.raises(qml.QuantumFunctionError, match="with diff_method=None"):
            qml.math.grad(circuit)(x)


class TestShots:
    """Unit tests for specifying shots per call."""

    # pylint: disable=unexpected-keyword-arg
    def test_specify_shots_per_call_sample(self):
        """Tests that shots can be set per call for a sample return type."""
        dev = qml.device("default.qubit", wires=1, shots=10)

        @qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert len(circuit(0.8)) == 10
        assert len(circuit(0.8, shots=2)) == 2
        assert len(circuit(0.8, shots=3178)) == 3178
        assert len(circuit(0.8)) == 10

    # pylint: disable=unexpected-keyword-arg, protected-access
    def test_specify_shots_per_call_expval(self):
        """Tests that shots can be set per call for an expectation value.
        Note: this test has a vanishingly small probability to fail."""
        dev = qml.device("default.qubit", wires=1, shots=None)

        @qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        # check that the circuit is analytic
        res1 = [circuit() for _ in range(100)]
        assert np.std(res1) == 0.0
        assert circuit.device._shots.total_shots is None

        # check that the circuit is temporary non-analytic
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

        dev = qml.device("default.qubit", wires=2, shots=10)

        with pytest.warns(
            UserWarning, match="The 'shots' argument name is reserved for overriding"
        ):

            @qnode(dev)
            def ansatz1(a, shots):
                qml.RX(a, wires=shots)
                return qml.sample(qml.PauliZ(wires=0))

        assert len(ansatz1(0.8, shots=0)) == 10
        tape = qml.workflow.construct_tape(circuit)(0.8, 0)
        assert tape.operations[0].wires.labels == (0,)

    def test_shots_passed_as_unrecognized_kwarg(self):
        """Test that an error is raised if shots are passed to QNode initialization."""
        dev = qml.device("default.qubit", wires=[0, 1], shots=10)

        def ansatz0():
            return qml.expval(qml.X(0))

        with pytest.raises(ValueError, match="'shots' is not a valid gradient_kwarg."):
            qml.QNode(ansatz0, dev, gradient_kwargs={"shots": 100})

        with pytest.raises(ValueError, match="'shots' is not a valid gradient_kwarg."):

            @qml.qnode(dev, gradient_kwargs={"shots": 100})
            def _():
                return qml.expval(qml.X(0))

    # pylint: disable=unexpected-keyword-arg
    def test_shots_setting_does_not_mutate_device(self):
        """Tests that per-call shots setting does not change the number of shots in the device."""

        dev = qml.device("default.qubit", wires=1, shots=3)

        @qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.sample(qml.PauliZ(wires=0))

        assert dev.shots.total_shots == 3
        res = circuit(0.8, shots=2)
        assert len(res) == 2
        assert dev.shots.total_shots == 3

    def test_warning_finite_shots_dev(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = qml.device("default.qubit", wires=1, shots=5)

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
        dev = qml.device("default.qubit", wires=1, shots=5)

        @qml.qnode(dev, cache={})
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        # no warning on the first execution
        circuit(0.3)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            circuit(0.3, shots=5)

    def test_warning_finite_shots_tape(self):
        """Tests that a warning is raised when caching is used with finite shots."""
        dev = qml.device("default.qubit", wires=1, shots=5)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RZ(0.3, wires=0)
            qml.expval(qml.PauliZ(0))

        tape = QuantumScript.from_queue(q, shots=5)
        # no warning on the first execution
        cache = {}
        qml.execute([tape], dev, None, cache=cache)
        with pytest.warns(UserWarning, match="Cached execution with finite shots detected"):
            qml.execute([tape], dev, None, cache=cache)

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
        dev = qml.device("default.qubit", wires=1, shots=5)

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
        dev = qml.device("default.qubit", wires=2, shots=5)

        def func(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.PauliZ(0))

        qn = QNode(func, dev)

        # No override
        tape = qml.workflow.construct_tape(qn)(0.1, 0.2)
        assert tape.shots.total_shots == 5

        # Override
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
        assert tape.shots.total_shots == 5

        # Override
        tape = qml.workflow.construct_tape(qn2)(0.1, 0.2, shots=shots)
        assert tape.shots.total_shots == total_shots
        assert tape.shots.shot_vector == shot_vector


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


class TestGetGradientFn:
    """Test the get_gradient_fn static method."""

    dev = CustomDevice()

    def test_get_gradient_fn_custom_device(self):
        """Test get_gradient_fn is parameter for best for null device."""
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(self.dev, "autograd", "best")
        assert gradient_fn is qml.gradients.param_shift
        assert not kwargs
        assert new_dev is self.dev

    def test_get_gradient_fn_with_best_method_and_cv_ops(self):
        """Test that get_gradient_fn returns 'parameter-shift-cv' when CV operations are present on tape"""
        dev = qml.device("default.gaussian", wires=1)
        tape = qml.tape.QuantumScript([qml.Displacement(0.5, 0.0, wires=0)])
        res = qml.QNode.get_gradient_fn(dev, interface="autograd", diff_method="best", tape=tape)
        assert res == (qml.gradients.param_shift_cv, {"dev": dev}, dev)

    def test_get_gradient_fn_default_qubit(self):
        """Tests the get_gradient_fn is backprop for best for default qubit2."""
        dev = qml.devices.DefaultQubit()
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(dev, "autograd", "best")
        assert gradient_fn == "backprop"
        assert not kwargs
        assert new_dev is dev

    def test_get_gradient_fn_custom_dev_adjoint(self):
        """Test that an error is raised if adjoint is requested for a device that does not support it."""
        with pytest.raises(
            qml.QuantumFunctionError, match=r"Device CustomDevice does not support adjoint"
        ):
            QNode.get_gradient_fn(self.dev, "autograd", "adjoint")

    def test_error_for_backprop_with_custom_device(self):
        """Test that an error is raised when backprop is requested for a device that does not support it."""
        with pytest.raises(
            qml.QuantumFunctionError, match=r"Device CustomDevice does not support backprop"
        ):
            QNode.get_gradient_fn(self.dev, "autograd", "backprop")

    def test_custom_device_that_supports_backprop(self):
        """Test that a custom device and designate that it supports backprop derivatives."""

        # pylint: disable=unused-argument
        class BackpropDevice(qml.devices.Device):
            """A device that says it supports backpropagation."""

            def execute(self, circuits, execution_config=None):
                return 0

            def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
                return execution_config.gradient_method == "backprop"

        dev = BackpropDevice()
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(
            dev, interface="autograd", diff_method="backprop"
        )
        assert gradient_fn == "backprop"
        assert not kwargs
        assert new_dev is dev

    def test_custom_device_with_device_derivative(self):
        """Test that a custom device can specify that it supports device derivatives."""

        # pylint: disable=unused-argument
        class DerivativeDevice(qml.devices.Device):
            """A device that says it supports device derivatives."""

            def execute(self, circuits, execution_config=None):
                return 0

            def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
                return execution_config.gradient_method == "device"

        dev = DerivativeDevice()
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(dev, "tf", "device")
        assert gradient_fn == "device"
        assert not kwargs
        assert new_dev is dev

    def test_diff_method_is_none(self):
        """Test get_gradient_fn behaves correctly."""
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(
            self.dev, interface=None, diff_method=None
        )
        assert gradient_fn is None
        assert not kwargs
        assert new_dev is self.dev

    def test_transform_dispatcher_as_diff_method(self):
        """Test when diff_method is of type TransformDispatcher"""
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(
            self.dev, interface=None, diff_method=qml.gradients.param_shift
        )
        assert gradient_fn is qml.gradients.param_shift
        assert not kwargs
        assert new_dev is self.dev

    def test_invalid_diff_method(self):
        """Test that an invalid diff method raises an error."""
        with pytest.raises(
            qml.QuantumFunctionError, match="Differentiation method invalid-method not recognized"
        ):
            QNode.get_gradient_fn(self.dev, None, diff_method="invalid-method")

    @pytest.mark.parametrize("diff_method", ["parameter-shift", "finite-diff", "spsa", "hadamard"])
    def test_valid_diff_method_str(self, diff_method):
        """Test that gradient_fn are retrieved correctly."""
        gradient_transform_map = {
            "parameter-shift": qml.gradients.param_shift,
            "finite-diff": qml.gradients.finite_diff,
            "spsa": qml.gradients.spsa_grad,
            "hadamard": qml.gradients.hadamard_grad,
        }
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(
            self.dev, interface=None, diff_method=diff_method
        )
        assert gradient_fn is gradient_transform_map[diff_method]
        assert not kwargs
        assert new_dev is self.dev

    def test_param_shift_method_with_cv_ops(self):
        """Test that 'parameter-shift-cv' is used when CV operations are present."""
        tape = qml.tape.QuantumScript([qml.Displacement(0.5, 0.0, wires=0)])
        gradient_fn, kwargs, new_dev = QNode.get_gradient_fn(
            self.dev, interface=None, diff_method="parameter-shift", tape=tape
        )
        assert gradient_fn is qml.gradients.param_shift_cv
        assert kwargs == {"dev": self.dev}
        assert new_dev is self.dev


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
        assert repr(qn) == "<QNode: device='CustomDevice', interface='auto', diff_method='best'>"

    def test_device_with_custom_diff_method_name(self):
        """Test a device that has its own custom diff method."""

        class CustomDeviceWithDiffMethod2(qml.devices.DefaultQubit):
            """A device with a custom derivative named hello."""

            def supports_derivatives(self, execution_config=None, circuit=None):
                return getattr(execution_config, "gradient_method", None) == "hello"

            def _setup_execution_config(self, execution_config=qml.devices.DefaultExecutionConfig):
                if execution_config.gradient_method in {"best", "hello"}:
                    return replace(
                        execution_config, gradient_method="hello", use_device_gradient=True
                    )
                return execution_config

            def compute_derivatives(
                self, circuits, execution_config=qml.devices.DefaultExecutionConfig
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

        with pytest.raises(qml.DeviceError, match="not accepted for analytic simulation"):
            circuit()

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
        dev = qml.device("default.qubit", wires=3, shots=shots)

        def f(x):
            qml.RX(x, 0)
            _ = qml.measure(0, postselect=1)
            return qml.sample(wires=[0, 1])

        with pytest.raises(ValueError, match="Invalid postselection mode 'foo'"):
            _ = qml.QNode(f, dev, postselect_mode="foo")

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

        dev = qml.device("default.qubit", wires=4, shots=shots, seed=jax.random.PRNGKey(seed))

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

        dev = qml.device("default.qubit", wires=4, shots=shots, seed=jax.random.PRNGKey(seed))

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
        dev = qml.device("default.qubit", wires=1)

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
        dev = qml.device("default.qubit", wires=2)

        original_config = qml.devices.MCMConfig(
            postselect_mode=postselect_mode, mcm_method=mcm_method
        )

        @qml.qnode(dev, postselect_mode=postselect_mode, mcm_method=mcm_method)
        def circuit(x, mp):
            qml.RX(x, 0)
            qml.measure(0, postselect=1)
            return mp(qml.PauliZ(0))

        _ = circuit(1.8, qml.expval, shots=10)
        assert circuit.execute_kwargs["postselect_mode"] == original_config.postselect_mode
        assert circuit.execute_kwargs["mcm_method"] == original_config.mcm_method

        if mcm_method != "one-shot":
            _ = circuit(1.8, qml.expval)
            assert circuit.execute_kwargs["postselect_mode"] == original_config.postselect_mode
            assert circuit.execute_kwargs["mcm_method"] == original_config.mcm_method

        _ = circuit(1.8, qml.expval, shots=10)
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
        dev = qml.device("default.qubit", wires=3, shots=None)

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

    with pytest.raises(qml.DeviceError):
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

    @pytest.mark.parametrize("interface", ["autograd", "torch", "tf", "jax", "jax-jit"])
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
