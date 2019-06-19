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
