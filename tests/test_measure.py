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
import numpy as np

import pennylane as qml
from pennylane.qnodes import QuantumFunctionError
from pennylane.operation import Sample, Variance, Expectation, State


def test_no_measure(tol):
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(QuantumFunctionError, match="does not have the measurement"):
        res = circuit(0.65)


class TestExpval:
    """Tests for the expval function"""

    def test_value(self, tol):
        """Test that the expval interface works"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self):
        """Test that a QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.CNOT(wires=[0, 1]))

        with pytest.raises(QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()

    def test_observable_return_type_is_expectation(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        circuit()


class TestVar:
    """Tests for the var function"""

    def test_value(self, tol):
        """Test that the var function works"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = np.sin(x) ** 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self):
        """Test that a QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.CNOT(wires=[0, 1]))

        with pytest.raises(QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()

    def test_observable_return_type_is_variance(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Variance`"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.var(qml.PauliZ(0))
            assert res.return_type is Variance
            return res

        circuit()


class TestSample:
    """Tests for the sample function"""

    def test_sample_dimension(self, tol):
        """Test that the sample function outputs samples of the right size"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=2, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        sample = circuit()

        assert np.array_equal(sample.shape, (2, n_sample))

    def test_sample_combination(self, tol):
        """Test the output of combining expval, var and sample"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert np.array_equal(result.shape, (3,))
        assert np.array_equal(result[0].shape, (n_sample,))
        assert isinstance(result[1], float)
        assert isinstance(result[2], float)

    def test_single_wire_sample(self, tol):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=1, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (n_sample,))

    def test_multi_wire_sample_regular_shape(self, tol):
        """Test the return type and shape of sampling multiple wires
           where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (3, n_sample))
        assert result.dtype == np.dtype("int")

    def test_sample_output_type_in_combination(self, tol):
        """Test the return type and shape of sampling multiple works
           in combination with expvals and vars"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.dtype("object")
        assert np.array_equal(result.shape, (3,))
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        assert result[2].dtype == np.dtype("int")
        assert np.array_equal(result[2].shape, (n_sample,))

    def test_not_an_observable(self):
        """Test that a QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.sample(qml.CNOT(wires=[0, 1]))

        with pytest.raises(QuantumFunctionError, match="CNOT is not an observable"):
            sample = circuit()

    def test_observable_return_type_is_sample(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Sample`"""
        n_shots = 10
        dev = qml.device("default.qubit", wires=1, shots=n_shots)

        @qml.qnode(dev)
        def circuit():
            res = qml.sample(qml.PauliZ(0))
            assert res.return_type is Sample
            return res

        circuit()


class TestState:
    """Tests for the state function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_shape_and_dtype(self, wires):
        """Test that the state is of correct size and dtype for a trivial circuit"""

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def func():
            return qml.state()

        state = func()
        assert state.shape == (2 ** wires,)
        assert state.dtype == np.complex128

    def test_return_type_is_state(self):
        """Test that the return type of the observable is State"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def func():
            return qml.state()

        func()
        ob = func.ops[0]

        assert ob.return_type == State

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_correct_ghz(self, wires):
        """Test that the correct state is returned when the circuit prepares a GHZ state"""

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.state()

        state = func()
        assert np.allclose(np.sum(np.abs(state) ** 2), 1)
        assert np.allclose(state[0], 1 / np.sqrt(2))
        assert np.allclose(state[-1], 1 / np.sqrt(2))

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_equal_to_dev_state(self, wires):
        """Test that the returned state is equal to the one stored in dev.state for a template
        circuit"""

        dev = qml.device("default.qubit", wires=wires)

        weights = qml.init.strong_ent_layers_uniform(3, wires)

        @qml.qnode(dev)
        def func():
            qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
            return qml.state()

        state = func()
        assert np.allclose(state, dev.state)

    def test_combination(self):
        """Test that the state can be output in combination with other return types."""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return qml.state(), qml.expval(qml.PauliZ(0)), qml.var(qml.PauliX(1))

        state_expected = 0.25 * np.ones(16)

        out = func()
        assert len(out) == 3
        assert np.allclose(out[0], state_expected)
        assert np.allclose(out[1], 0)
        assert np.allclose(out[2], 0)

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_interface_tf(self, skip_if_no_tf_support):
        """Test that the state correctly outputs in the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="tf")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return qml.state()

        state_expected = 0.25 * tf.ones(16)
        state = func()

        assert isinstance(state, tf.Tensor)
        assert state.dtype == tf.complex128
        assert np.allclose(state_expected, state.numpy())

    def test_interface_torch(self):
        """Test that the state correctly outputs in the torch interface"""
        torch = pytest.importorskip("torch", minversion="1.6")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="torch")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return qml.state()

        state_expected = 0.25 * torch.ones(16, dtype=torch.complex128)
        state = func()

        assert isinstance(state, torch.Tensor)
        assert state.dtype == torch.complex128
        assert torch.allclose(state_expected, state)

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    @pytest.mark.parametrize(
        "device", ["default.qubit", "default.qubit.tf", "default.qubit.autograd"]
    )
    def test_devices(self, device, skip_if_no_tf_support):
        """Test that the returned state is equal to the expected returned state for all of
        PennyLane's built in statevector devices"""

        dev = qml.device(device, wires=4)

        @qml.qnode(dev)
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return qml.state()

        state = func()
        state_expected = 0.25 * np.ones(16)

        assert np.allclose(state, state_expected)
        assert np.allclose(state, dev.state)
