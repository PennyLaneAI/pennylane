# Copyright 2018 Xanadu Quantum Technologies Inc.

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
from pennylane.qnode import QuantumFunctionError


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


class TestDeprecatedExpval:
    """Tests for the deprecated expval attribute getter.
    Once fully deprecated, this test can be removed"""
    #TODO: once `qml.expval.Observable` is deprecated, remove this test

    def test_value(self, tol):
        """Test that the old expval interface works,
        but a deprecation warning is raised"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval.PauliY(0)

        x = 0.54
        with pytest.warns(DeprecationWarning, match="is deprecated"):
            res = circuit(x)

        expected = -np.sin(x)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self):
        """Test that an attribute error is raised if the provided
        attribute is not an observable when using the old interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval.RX(0)

        with pytest.warns(DeprecationWarning, match="is deprecated"):
            with pytest.raises(AttributeError, match="RX is not an observable"):
                res = circuit()

    def test_not_an_operation(self):
        """Test that an attribute error is raised if an
        observable doesn't exist using the old interface"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval.R(0)

        with pytest.warns(DeprecationWarning, match="is deprecated"):
            with pytest.raises(AttributeError, match="has no observable 'R'"):
                res = circuit()


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


class TestSample:
    """Tests for the sample function"""

    def test_sample_dimension(self, tol):
        """Test that the sample function outputs samples of the right size"""
        dev = qml.device("default.qubit", wires=2)

        n_sample = 10

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0), n_sample), qml.sample(qml.PauliX(1), 2*n_sample)

        sample = circuit()

        assert np.array_equal(sample.shape, (2,))
        assert np.array_equal(sample[0].shape, (n_sample,))
        assert np.array_equal(sample[1].shape, (2*n_sample,))

    def test_sample_combination(self, tol):
        """Test the output of combining expval, var and sample"""
        dev = qml.device("default.qubit", wires=3)

        n_sample = 10

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0), n_sample), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert np.array_equal(result.shape, (3,))
        assert np.array_equal(result[0].shape, (n_sample,))
        assert isinstance(result[1], float)
        assert isinstance(result[2], float)

    def test_single_wire_sample(self, tol):
        """Test the return type and shape of sampling a single wire"""
        dev = qml.device("default.qubit", wires=1)

        n_sample = 10

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0), n_sample)

        result = circuit()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (n_sample,))

    def test_multi_wire_sample_regular_shape(self, tol):
        """Test the return type and shape of sampling multiple wires
           where a rectangular array is expected"""
        dev = qml.device("default.qubit", wires=3)

        n_sample = 10

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0), n_sample), qml.sample(qml.PauliZ(1), n_sample), qml.sample(qml.PauliZ(2), n_sample)

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (3, n_sample))
        assert result.dtype == np.dtype("float")

    def test_multi_wire_sample_ragged_shape(self, tol):
        """Test the return type and shape of sampling multiple wires
           where a ragged array is expected"""
        dev = qml.device("default.qubit", wires=3)

        n_sample = 10

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0), n_sample), qml.sample(qml.PauliZ(1), 2*n_sample), qml.sample(qml.PauliZ(2), 3*n_sample)

        result = circuit()

        # If the sample dimensions are not equal we expect the 
        # output to be an array of dtype="object"
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.dtype("object")
        assert np.array_equal(result.shape, (3,))
        assert np.array_equal(result[0].shape, (n_sample,))
        assert np.array_equal(result[1].shape, (2*n_sample,))
        assert np.array_equal(result[2].shape, (3*n_sample,))

    def test_sample_output_type_in_combination(self, tol):
        """Test the return type and shape of sampling multiple works 
           in combination with expvals and vars"""
        dev = qml.device("default.qubit", wires=3)

        n_sample = 10

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1)), qml.sample(qml.PauliZ(2), n_sample)

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.dtype("object")
        assert np.array_equal(result.shape, (3,))
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)
        assert result[2].dtype == np.dtype("float")
        assert np.array_equal(result[2].shape, (n_sample,))

    def test_sample_default_n(self, tol):
        """Test the return type and shape of sampling multiple works 
           in combination with expvals and vars"""

        n_shots = 10
        dev = qml.device("default.qubit", wires=1, shots=n_shots)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert np.array_equal(result.shape, (n_shots,))
        
    def test_sample_exception_device_context_missing(self):
        """Tests if the sampling raises an error when using a default
           sample number but the underlying device can't be accessed"""

        with pytest.raises(QuantumFunctionError, match="Could not find a bound device to determine the default number of samples."):
            qml.QNode._current_context = None
            qml.sample(qml.PauliZ(0, do_queue=False))

    def test_sample_exception_wrong_n(self):
        """Tests if the sampling raises an error for sample size n<=0
        or non-integer n
        """
        dev = qml.device("default.qubit", wires=2)

        with pytest.raises(ValueError, match="Calling sample with n = 0 is not possible."):
            @qml.qnode(dev)
            def circuit_a():
                qml.RX(0.52, wires=0)
                return qml.sample(qml.PauliZ(0), n=0)

            circuit_a()

        with pytest.raises(ValueError, match="The number of samples must be a positive integer."):
            @qml.qnode(dev)
            def circuit_b():
                qml.RX(0.52, wires=0)
                return qml.sample(qml.PauliZ(0), n=-12)

            circuit_b()

        with pytest.raises(ValueError, match="The number of samples must be a positive integer."):
            @qml.qnode(dev)
            def circuit_c():
                qml.RX(0.52, wires=0)
                return qml.sample(qml.PauliZ(0), n=20.4)

            circuit_c()

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
