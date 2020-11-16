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
"""Unit tests for the measure module"""
import contextlib
import pytest
import numpy as np

import pennylane as qml
from pennylane import QuantumFunctionError
from pennylane.devices import DefaultQubit

# Beta imports
from pennylane.tape import qnode
from pennylane.tape.queuing import AnnotatedQueue
from pennylane.tape.operation import mock_operations
from pennylane.tape.measure import (
    expval,
    var,
    sample,
    probs,
    state,
    density_matrix,
    Expectation,
    Sample,
    State,
    Variance,
    Probability,
    MeasurementProcess,
)


@pytest.fixture(autouse=True)
def patch_operator():
    with contextlib.ExitStack() as stack:
        for mock in mock_operations():
            stack.enter_context(mock)
        yield


@pytest.mark.parametrize(
    "stat_func,return_type", [(expval, Expectation), (var, Variance), (sample, Sample)]
)
class TestBetaStatistics:
    """Tests for annotating the return types of the statistics functions"""

    @pytest.mark.parametrize(
        "op",
        [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.Identity],
    )
    def test_annotating_obs_return_type(self, stat_func, return_type, op):
        """Test that the return_type related info is updated for a
        measurement"""
        with AnnotatedQueue() as q:
            A = op(0)
            stat_func(A)

        assert q.queue[:-1] == [A]
        meas_proc = q.queue[-1]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == return_type

        assert q._get_info(A) == {"owner": meas_proc}
        assert q._get_info(meas_proc) == {"owns": (A)}

    def test_annotating_tensor_hermitian(self, stat_func, return_type):
        """Test that the return_type related info is updated for a measurement
        when called for an Hermitian observable"""

        mx = np.array([[1, 0], [0, 1]])

        with AnnotatedQueue() as q:
            Herm = qml.Hermitian(mx, wires=[1])
            stat_func(Herm)

        assert q.queue[:-1] == [Herm]
        meas_proc = q.queue[-1]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == return_type

        assert q._get_info(Herm) == {"owner": meas_proc}
        assert q._get_info(meas_proc) == {"owns": (Herm)}

    @pytest.mark.parametrize(
        "op1,op2",
        [
            (qml.PauliY, qml.PauliX),
            (qml.Hadamard, qml.Hadamard),
            (qml.PauliY, qml.Identity),
            (qml.Identity, qml.Identity),
        ],
    )
    def test_annotating_tensor_return_type(self, op1, op2, stat_func, return_type):
        """Test that the return_type related info is updated for a measurement
        when called for an Tensor observable"""
        with AnnotatedQueue() as q:
            A = op1(0)
            B = op2(1)
            tensor_op = A @ B
            stat_func(tensor_op)

        assert q.queue[:-1] == [A, B, tensor_op]
        meas_proc = q.queue[-1]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == return_type

        assert q._get_info(A) == {"owner": tensor_op}
        assert q._get_info(B) == {"owner": tensor_op}
        assert q._get_info(tensor_op) == {"owns": (A, B), "owner": meas_proc}


@pytest.mark.parametrize("stat_func", [expval, var, sample])
class TestBetaStatisticsError:
    """Tests for errors arising for the beta statistics functions"""

    def test_not_an_observable(self, stat_func):
        """Test that a QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return stat_func(qml.CNOT(wires=[0, 1]))

        with pytest.raises(QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()


class TestBetaProbs:
    """Tests for annotating the return types of the probs function"""

    @pytest.mark.parametrize("wires", [[0], [0, 1], [1, 0, 2]])
    def test_annotating_probs(self, wires):
        with AnnotatedQueue() as q:
            probs(wires)

        assert len(q.queue) == 1

        meas_proc = q.queue[0]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == Probability


class TestProperties:
    """Test for the properties"""

    def test_wires_match_observable(self):
        """Test that the wires of the measurement process
        match an internal observable"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=["a", "b", "c"])
        m = MeasurementProcess(Expectation, obs=obs)

        assert np.all(m.wires == obs.wires)

    def test_eigvals_match_observable(self):
        """Test that the eigenvalues of the measurement process
        match an internal observable"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=[0, 1, 2])
        m = MeasurementProcess(Expectation, obs=obs)

        assert np.all(m.eigvals == np.array([1, 2, 3]))

        # changing the observable data should be reflected
        obs.data = [np.diag([5, 6, 7])]
        assert np.all(m.eigvals == np.array([5, 6, 7]))

    def test_error_obs_and_eigvals(self):
        """Test that providing both eigenvalues and an observable
        results in an error"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=[0, 1, 2])

        with pytest.raises(ValueError, match="Cannot set the eigenvalues"):
            MeasurementProcess(Expectation, obs=obs, eigvals=[0, 1])

    def test_error_obs_and_wires(self):
        """Test that providing both wires and an observable
        results in an error"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=[0, 1, 2])

        with pytest.raises(ValueError, match="Cannot set the wires"):
            MeasurementProcess(Expectation, obs=obs, wires=qml.wires.Wires([0, 1]))

    def test_observable_with_no_eigvals(self):
        """An observable with no eigenvalues defined should cause
        the eigvals property on the associated measurement process
        to be None"""
        obs = qml.NumberOperator(wires=0)
        m = MeasurementProcess(Expectation, obs=obs)
        assert m.eigvals is None

    def test_repr(self):
        """Test the string representation of a MeasurementProcess."""
        m = MeasurementProcess(Expectation, obs=qml.PauliZ(wires="a") @ qml.PauliZ(wires="b"))
        expected = "expval(PauliZ(wires=['a']) @ PauliZ(wires=['b']))"
        assert str(m) == expected

        m = MeasurementProcess(Probability, obs=qml.PauliZ(wires="a"))
        expected = "probs(PauliZ(wires=['a']))"
        assert str(m) == expected


class TestExpansion:
    """Test for measurement expansion"""

    def test_expand_pauli(self):
        """Test the expansion of a Pauli observable"""
        obs = qml.PauliX(0) @ qml.PauliY(1)
        m = MeasurementProcess(Expectation, obs=obs)
        tape = m.expand()

        assert len(tape.operations) == 4

        assert tape.operations[0].name == "Hadamard"
        assert tape.operations[0].wires.tolist() == [0]

        assert tape.operations[1].name == "PauliZ"
        assert tape.operations[1].wires.tolist() == [1]
        assert tape.operations[2].name == "S"
        assert tape.operations[2].wires.tolist() == [1]
        assert tape.operations[3].name == "Hadamard"
        assert tape.operations[3].wires.tolist() == [1]

        assert len(tape.measurements) == 1
        assert tape.measurements[0].return_type is Expectation
        assert tape.measurements[0].wires.tolist() == [0, 1]
        assert np.all(tape.measurements[0].eigvals == np.array([1, -1, -1, 1]))

    def test_expand_hermitian(self, tol):
        """Test the expansion of an hermitian observable"""
        H = np.array([[1, 2], [2, 4]])
        obs = qml.Hermitian(H, wires=["a"])

        m = MeasurementProcess(Expectation, obs=obs)
        tape = m.expand()

        assert len(tape.operations) == 1

        assert tape.operations[0].name == "QubitUnitary"
        assert tape.operations[0].wires.tolist() == ["a"]
        assert np.allclose(
            tape.operations[0].parameters[0],
            np.array([[-2, 1], [1, 2]]) / np.sqrt(5),
            atol=tol,
            rtol=0,
        )

        assert len(tape.measurements) == 1
        assert tape.measurements[0].return_type is Expectation
        assert tape.measurements[0].wires.tolist() == ["a"]
        assert np.all(tape.measurements[0].eigvals == np.array([0, 5]))

    def test_expand_no_observable(self):
        """Check that an exception is raised if the measurement to
        be expanded has no observable"""
        m = MeasurementProcess(Probability, wires=qml.wires.Wires([0, 1]))

        with pytest.raises(NotImplementedError, match="Cannot expand"):
            m.expand()


class TestState:
    """Tests for the state function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_shape_and_dtype(self, wires):
        """Test that the state is of correct size and dtype for a trivial circuit"""

        dev = qml.device("default.qubit", wires=wires)

        @qnode(dev)
        def func():
            return state()

        state_val = func()
        assert state_val.shape == (2 ** wires,)
        assert state_val.dtype == np.complex128

    def test_return_type_is_state(self):
        """Test that the return type of the observable is State"""

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev)
        def func():
            qml.Hadamard(0)
            return state()

        func()
        obs = func.qtape.observables
        assert len(obs) == 1
        assert obs[0].return_type is State

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_correct_ghz(self, wires):
        """Test that the correct state is returned when the circuit prepares a GHZ state"""

        dev = qml.device("default.qubit", wires=wires)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return state()

        state_val = func()
        assert np.allclose(np.sum(np.abs(state_val) ** 2), 1)
        assert np.allclose(state_val[0], 1 / np.sqrt(2))
        assert np.allclose(state_val[-1], 1 / np.sqrt(2))

    def test_return_with_other_types(self):
        """Test that an exception is raised when a state is returned along with another return
        type"""

        dev = qml.device("default.qubit", wires=2)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            return state(), expval(qml.PauliZ(1))

        with pytest.raises(
            QuantumFunctionError,
            match="The state or density matrix"
            " cannot be returned in combination"
            " with other return types",
        ):
            func()

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_equal_to_dev_state(self, wires):
        """Test that the returned state is equal to the one stored in dev.state for a template
        circuit"""

        dev = qml.device("default.qubit", wires=wires)

        weights = qml.init.strong_ent_layers_uniform(3, wires)

        @qnode(dev)
        def func():
            qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
            return state()

        state_val = func()
        assert np.allclose(state_val, dev.state)

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_interface_tf(self, skip_if_no_tf_support):
        """Test that the state correctly outputs in the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=4)

        @qnode(dev, interface="tf")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_expected = 0.25 * tf.ones(16)
        state_val = func()

        assert isinstance(state_val, tf.Tensor)
        assert state_val.dtype == tf.complex128
        assert np.allclose(state_expected, state_val.numpy())
        assert state_val.shape == (16,)

    def test_interface_torch(self):
        """Test that the state correctly outputs in the torch interface"""
        torch = pytest.importorskip("torch", minversion="1.6")

        dev = qml.device("default.qubit", wires=4)

        @qnode(dev, interface="torch")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_expected = 0.25 * torch.ones(16, dtype=torch.complex128)
        state_val = func()

        assert isinstance(state_val, torch.Tensor)
        assert state_val.dtype == torch.complex128
        assert torch.allclose(state_expected, state_val)
        assert state_val.shape == (16,)

    def test_jacobian_not_supported(self):
        """Test if an error is raised if the jacobian method is called via qml.grad"""
        dev = qml.device("default.qubit", wires=4)

        @qnode(dev)
        def func(x):
            for i in range(4):
                qml.RX(x, wires=i)
            return state()

        d_func = qml.jacobian(func)

        with pytest.raises(ValueError, match="The jacobian method does not support"):
            d_func(0.1)

    def test_no_state_capability(self, monkeypatch):
        """Test if an error is raised for devices that are not capable of returning the state.
        This is tested by changing the capability of default.qubit"""
        dev = qml.device("default.qubit", wires=1)
        capabilities = dev.capabilities().copy()
        capabilities["returns_state"] = False

        @qnode(dev)
        def func():
            return state()

        with monkeypatch.context() as m:
            m.setattr(DefaultQubit, "capabilities", lambda *args, **kwargs: capabilities)
            with pytest.raises(QuantumFunctionError, match="The current device is not capable"):
                func()

    def test_state_not_supported(self, monkeypatch):
        """Test if an error is raised for devices inheriting from the base Device class,
        which do not currently support returning the state"""
        dev = qml.device("default.gaussian", wires=1)

        @qnode(dev)
        def func():
            return state()

        with pytest.raises(QuantumFunctionError, match="Returning the state is not supported"):
            func()

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    @pytest.mark.parametrize(
        "device", ["default.qubit", "default.qubit.tf", "default.qubit.autograd"]
    )
    def test_devices(self, device, skip_if_no_tf_support):
        """Test that the returned state is equal to the expected returned state for all of
        PennyLane's built in statevector devices"""

        dev = qml.device(device, wires=4)

        @qnode(dev)
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_val = func()
        state_expected = 0.25 * np.ones(16)

        assert np.allclose(state_val, state_expected)
        assert np.allclose(state_val, dev.state)

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_gradient_with_passthru_tf(self, skip_if_no_tf_support):
        """Test that the gradient of the state is accessible when using default.qubit.tf with the
        backprop diff_method."""
        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=1)

        @qnode(dev, interface="tf", diff_method="backprop")
        def func(x):
            qml.RY(x, wires=0)
            return state()

        x = tf.Variable(0.1, dtype=tf.complex128)

        with tf.GradientTape() as tape:
            result = func(x)

        grad = tape.jacobian(result, x)
        expected = tf.stack([-0.5 * tf.sin(x / 2), 0.5 * tf.cos(x / 2)])
        assert np.allclose(grad, expected)

    def test_gradient_with_passthru_autograd(self):
        """Test that the gradient of the state is accessible when using default.qubit.autograd
        with the backprop diff_method."""
        from pennylane import numpy as anp

        dev = qml.device("default.qubit.autograd", wires=1)

        @qnode(dev, interface="autograd", diff_method="backprop")
        def func(x):
            qml.RY(x, wires=0)
            return state()

        x = anp.array(0.1, requires_grad=True)

        def loss_fn(x):
            res = func(x)
            return anp.real(res)  # This errors without the real. Likely an issue with complex
            # numbers in autograd

        d_loss_fn = qml.jacobian(loss_fn)

        grad = d_loss_fn(x)
        expected = np.array([-0.5 * np.sin(x / 2), 0.5 * np.cos(x / 2)])
        assert np.allclose(grad, expected)

    @pytest.mark.parametrize("wires", [[0, 2, 3, 1], ["a", -1, "b", 1000]])
    def test_custom_wire_labels(self, wires):
        """Test if an error is raised when custom wire labels are used"""
        dev = qml.device("default.qubit", wires=wires)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=wires[0])
            for i in range(3):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            return state()

        with pytest.raises(QuantumFunctionError, match="custom wire labels"):
            func()


class TestDensityMatrix:
    """Tests for the density matrix function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_density_matrix_shape_and_dtype(self, dev_name, wires):
        """Test that the density matrix is of correct size and dtype for a
        trivial circuit"""

        dev = qml.device(dev_name, wires=wires)

        @qnode(dev)
        def circuit():
            return density_matrix([0])

        state_val = circuit()

        assert state_val.shape == (2, 2)
        assert state_val.dtype == np.complex128

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_return_type_is_state(self, dev_name):
        """Test that the return type of the observable is State"""

        dev = qml.device(dev_name, wires=2)

        @qnode(dev)
        def func():
            qml.Hadamard(0)
            return density_matrix(0)

        func()
        obs = func.qtape.observables
        assert len(obs) == 1
        assert obs[0].return_type is State

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_product_state_first(self, dev_name):
        """Test that the correct density matrix is returned when
        tracing out a product state"""

        dev = qml.device(dev_name, wires=2)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix(0)

        density_first = func()

        assert np.allclose(
            np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]), density_first
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_product_state_second(self, dev_name):
        """Test that the correct density matrix is returned when
        tracing out a product state"""

        dev = qml.device(dev_name, wires=2)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix(1)

        density_second = func()
        assert np.allclose(
            np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]]), density_second
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_three_wires_first(self, dev_name):
        """Test that the correct density matrix for an example with three wires"""

        dev = qml.device(dev_name, wires=3)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix([0, 1])

        density_full = func()
        assert np.allclose(
            np.array(
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                ]
            ),
            density_full,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_three_wires_second(self, dev_name):
        """Test that the correct density matrix for an example with three wires"""

        dev = qml.device(dev_name, wires=3)

        @qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.CNOT(wires=[1, 2])
            return qml.density_matrix(wires=[1, 2])

        density = func()

        assert np.allclose(
            np.array(
                [
                    [
                        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    ]
                ]
            ),
            density,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_mixed_state(self, dev_name):
        """Test that the correct density matrix for an example with a mixed state"""

        dev = qml.device(dev_name, wires=2)

        @qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.density_matrix(wires=[1])

        density = func()

        assert np.allclose(np.array([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]), density)

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_all_wires(self, dev_name):
        """Test that the correct density matrix is returned when all wires are given"""

        dev = qml.device(dev_name, wires=2)

        @qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.density_matrix(wires=[0, 1])

        density = func()

        assert np.allclose(
            np.array(
                [
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                ]
            ),
            density,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_return_with_other_types(self, dev_name):
        """Test that an exception is raised when a state is returned along with another return
        type"""

        dev = qml.device(dev_name, wires=2)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            return density_matrix(0), expval(qml.PauliZ(1))

        with pytest.raises(
            QuantumFunctionError,
            match="The state or density matrix"
            " cannot be returned in combination"
            " with other return types",
        ):
            func()

    def test_no_state_capability(self, monkeypatch):
        """Test if an error is raised for devices that are not capable of returning
        the density matrix. This is tested by changing the capability of default.qubit"""
        dev = qml.device("default.qubit", wires=2)
        capabilities = dev.capabilities().copy()
        capabilities["returns_state"] = False

        @qnode(dev)
        def func():
            return density_matrix(0)

        with monkeypatch.context() as m:
            m.setattr(DefaultQubit, "capabilities", lambda *args, **kwargs: capabilities)
            with pytest.raises(
                QuantumFunctionError,
                match="The current device is not capable" " of returning the state",
            ):
                func()

    def test_density_matrix_not_supported(self):
        """Test if an error is raised for devices inheriting from the base Device class,
        which do not currently support returning the state"""
        dev = qml.device("default.gaussian", wires=2)

        @qnode(dev)
        def func():
            return density_matrix(0)

        with pytest.raises(QuantumFunctionError, match="Returning the state is not supported"):
            func()

    @pytest.mark.parametrize("wires", [[0, 2, 3, 1], ["a", -1, "b", 1000]])
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_custom_wire_labels(self, wires, dev_name):
        """Test if an error is raised when custom wire labels are used"""
        dev = qml.device(dev_name, wires=wires)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=wires[0])
            for i in range(3):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            return density_matrix(0)

        with pytest.raises(QuantumFunctionError, match="custom wire labels"):
            func()
