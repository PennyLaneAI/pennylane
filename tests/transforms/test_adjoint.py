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
import pytest
import numpy as np

import pennylane as qml
from pennylane.transforms.adjoint import adjoint


def test_adjoint_on_function():
    """Test that adjoint works when applied to a function"""

    dev = qml.device("default.qubit", wires=1)

    def my_op():
        qml.RX(0.123, wires=0)
        qml.RY(2.32, wires=0)
        qml.RZ(1.95, wires=0)

    @qml.qnode(dev)
    def my_circuit():
        qml.PauliX(wires=0)
        qml.PauliZ(wires=0)
        my_op()
        adjoint(my_op)()
        qml.PauliY(wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([1.0j, 0.0]), atol=1e-6, rtol=1e-6)


def test_adjoint_directly_on_op():
    """Test that adjoint works when directly applyed to an op"""

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(qml.RX)(np.pi / 4.0, wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([0.92388, 0.382683j]), atol=1e-6, rtol=1e-6)


def test_nested_adjoint():
    """Test that adjoint works when nested with other adjoints"""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(adjoint(qml.RX))(np.pi / 4.0, wires=0)
        return qml.state()

    np.testing.assert_allclose(my_circuit(), np.array([0.92388, -0.382683j]), atol=1e-6, rtol=1e-6)


def test_nested_adjoint_on_function():
    """Test that adjoint works when nested with other adjoints"""

    def my_op():
        qml.RX(0.123, wires=0)
        qml.RY(2.32, wires=0)
        qml.RZ(1.95, wires=0)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def my_circuit():
        adjoint(my_op)()
        qml.Hadamard(wires=0)
        adjoint(adjoint(my_op))()
        return qml.state()

    np.testing.assert_allclose(
        my_circuit(), np.array([-0.995707, 0.068644 + 6.209710e-02j]), atol=1e-6, rtol=1e-6
    )


test_functions = [
    lambda fn, *args, **kwargs: adjoint(fn)(*args, **kwargs),
    lambda fn, *args, **kwargs: qml.inv(fn(*args, **kwargs)),
]


@pytest.mark.parametrize("fn", test_functions)
class TestTemplateIntegration:
    """Test that templates work correctly with the adjoint transform"""

    def test_angle_embedding(self, fn):
        """Test that the adjoint correctly inverts angle embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.AngleEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.array([0.2, 0.5, 0.8])
        res = circuit(weights)
        assert len(np.nonzero(res)) == 1

    def test_amplitude_embedding(self, fn):
        """Test that the adjoint correctly inverts amplitude embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.AmplitudeEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.array([0.2, 0.5, 0.8, 0.6, 0.1, 0.6, 0.1, 0.5]) / np.sqrt(1.92)
        res = circuit(weights)
        assert len(np.nonzero(res)) == 1

    def test_basis_embedding(self, fn):
        """Test that the adjoint correctly inverts basis embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.BasisEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.state()

        weights = np.array([1, 0, 1])
        res = circuit(weights)
        expected = np.zeros([2 ** 3])
        expected[0] = 1.0
        assert np.allclose(res, expected)

    def test_displacement_embedding(self, fn):
        """Test that the adjoint correctly inverts displacement embedding"""
        dev = qml.device("default.gaussian", wires=3)
        template = qml.templates.DisplacementEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.expval(qml.NumberOperator(0))

        weights = np.array([0.6, 0.2, 0.1])
        res = circuit(weights)
        assert np.allclose(res, 0.0)

    def test_squeezing_embedding(self, fn):
        """Test that the adjoint correctly inverts squeezing embedding"""
        dev = qml.device("default.gaussian", wires=3)
        template = qml.templates.SqueezingEmbedding

        @qml.qnode(dev)
        def circuit(weights):
            template(features=weights, wires=[0, 1, 2])
            fn(template, features=weights, wires=[0, 1, 2])
            return qml.expval(qml.NumberOperator(0))

        weights = np.array([0.6, 0.2, 0.1])
        res = circuit(weights)
        assert np.allclose(res, 0.0)

    def test_qaoa_embedding(self, fn):
        """Test that the adjoint correctly inverts qaoa embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.QAOAEmbedding

        @qml.qnode(dev)
        def circuit(features, weights):
            template(features=features, weights=weights, wires=[0, 1, 2])
            fn(template, features=features, weights=weights, wires=[0, 1, 2])
            return qml.probs(wires=[0, 1, 2])

        features = np.array([1.0, 2.0, 3.0])
        weights = np.random.random(template.shape(2, 3))

        res = circuit(features, weights)
        expected = np.zeros([2 ** 3])
        expected[0] = 1.0

        assert np.allclose(res, expected)

    def test_iqp_embedding(self, fn):
        """Test that the adjoint correctly inverts iqp embedding"""
        dev = qml.device("default.qubit", wires=3)
        template = qml.templates.IQPEmbedding

        @qml.qnode(dev)
        def circuit(features):
            template(features=features, wires=[0, 1, 2])
            fn(template, features=features, wires=[0, 1, 2])
            return qml.probs(wires=[0, 1, 2])

        features = np.array([1.0, 2.0, 3.0])
        res = circuit(features)
        expected = np.zeros([2 ** 3])
        expected[0] = 1.0

        assert np.allclose(res, expected)
