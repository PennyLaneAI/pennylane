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
import pytest
import contextlib
import numpy as np

import pennylane as qml
from pennylane import QuantumFunctionError
from pennylane.beta.tapes.qnode import QuantumFunctionError as QuantumFunctionErrorBeta  # TODO


# Beta imports
from pennylane.beta.tapes import qnode
from pennylane.operation import ObservableReturnTypes
from pennylane.beta.queuing import AnnotatedQueue, QueuingContext
from pennylane.beta.queuing.operation import mock_operations
from pennylane.beta.queuing.measure import (
    expval,
    var,
    sample,
    probs,
    Expectation,
    Sample,
    State,
    Variance,
    Probability,
    MeasurementProcess
)


def mock_queue(self):
    QueuingContext.append(self)
    return self


@pytest.fixture(autouse=True)
def patch_operator(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(qml.operation.Operator, "queue", mock_queue)
        yield


@pytest.mark.parametrize(
    "stat_func,return_type", [(expval, Expectation), (var, Variance), (sample, Sample)]
)
class TestBetaStatistics:
    """Tests for annotating the return types of the statistics functions"""

    @pytest.mark.parametrize(
        "op", [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.Identity],
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
        with contextlib.ExitStack() as stack:
            for mock in mock_operations():
                stack.enter_context(mock)

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
        assert q._get_info(tensor_op) == {"owns": (A,B), "owner": meas_proc}

@pytest.mark.parametrize(
    "stat_func", [expval, var, sample]
)
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


class TestState:
    """Tests for the state function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_shape_and_dtype(self, wires):
        """Test that the state is of correct size and dtype for a trivial circuit"""

        dev = qml.device("default.qubit", wires=wires)

        @qnode(dev)
        def func():
            return qml.state(range(wires))

        state = func()
        assert state.shape == (1, 2 ** wires)
        assert state.dtype == np.complex128

    def test_return_type_is_state(self):
        """Test that the return type of the observable is State"""

        dev = qml.device("default.qubit", wires=1)

        @qnode(dev)
        def func():
            qml.Hadamard(0)
            return qml.state([0])

        func()
        obs = func.qtape.observables
        assert len(obs) == 1
        assert obs[0].return_type is ObservableReturnTypes.State

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_correct_ghz(self, wires):
        """Test that the correct state is returned when the circuit prepares a GHZ state"""

        dev = qml.device("default.qubit", wires=wires)

        @qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.state(range(wires))

        state = func()[0]
        assert np.allclose(np.sum(np.abs(state) ** 2), 1)
        assert np.allclose(state[0], 1 / np.sqrt(2))
        assert np.allclose(state[-1], 1 / np.sqrt(2))

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_equal_to_dev_state(self, wires):
        """Test that the returned state is equal to the one stored in dev.state for a template
        circuit"""

        dev = qml.device("default.qubit", wires=wires)

        weights = qml.init.strong_ent_layers_uniform(3, wires)

        @qnode(dev)
        def func():
            qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
            return qml.state(range(wires))

        state = func()
        assert np.allclose(state, dev.state)

    def test_all_wires(self):
        """Test that an error is raised if the state is requested on a subset of wires"""
        wires = 4
        dev = qml.device("default.qubit", wires=wires)

        weights = qml.init.strong_ent_layers_uniform(3, wires)

        @qnode(dev)
        def func():
            qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
            return qml.state(range(wires - 1))

        with pytest.raises(QuantumFunctionError, match="The state must be returned over all wires"):
            func()

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_interface_tf(self, skip_if_no_tf_support):
        """Test that the state correctly outputs in the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=4)

        @qnode(dev, interface="tf")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return qml.state(range(4))

        state_expected = 0.25 * tf.ones(16)
        state = func()

        assert isinstance(state, tf.Tensor)
        assert state.dtype == tf.complex128
        assert np.allclose(state_expected, state.numpy())
        assert state.shape == (1, 16)

    def test_interface_torch(self):
        """Test that the state correctly outputs in the torch interface"""
        torch = pytest.importorskip("torch", minversion="1.6")

        dev = qml.device("default.qubit", wires=4)

        @qnode(dev, interface="torch")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return qml.state(range(4))

        state_expected = 0.25 * torch.ones(16, dtype=torch.complex128)
        state = func()

        assert isinstance(state, torch.Tensor)
        assert state.dtype == torch.complex128
        assert torch.allclose(state_expected, state)
        assert state.shape == (1, 16)

    @pytest.mark.usefixtures("skip_if_no_torch_support")
    def test_interface_torch_wrong_version(self, monkeypatch):
        """Test if an error is raised when a version of torch before 1.6.0 is used"""
        import torch
        dev = qml.device("default.qubit", wires=4)

        @qnode(dev, interface="torch")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return qml.state(range(4))

        with monkeypatch.context() as m:
            m.setattr(torch, "__version__", "1.5.0")

            with pytest.raises(QuantumFunctionErrorBeta, match="Version 1.6.0 or above of PyTorch"):
                func()

    def test_jacobian_not_supported(self):
        """Test if an error is raised if the jacobian method is called via qml.grad"""
        dev = qml.device("default.qubit", wires=4)

        @qnode(dev)
        def func(x):
            for i in range(4):
                qml.RX(x, wires=i)
            return qml.state(range(4))

        d_func = qml.jacobian(func)

        with pytest.raises(NotImplementedError, match="The jacobian method is not supported"):
            d_func(0.1)

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
            return qml.state(range(4))

        state = func()
        state_expected = 0.25 * np.ones(16)

        assert np.allclose(state, state_expected)
        assert np.allclose(state, dev.state)
