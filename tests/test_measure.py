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
from pennylane.operation import Sample, Variance, Expectation


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
        expected = np.sin(x)**2

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

        assert np.array_equal(sample.shape, (2,n_sample))

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
