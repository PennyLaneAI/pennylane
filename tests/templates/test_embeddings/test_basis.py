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
"""
Unit tests for the BasisEmbedding template.
"""
import pytest
import numpy as np
import pennylane as qml


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize("features", [[1, 0, 1],
                                          [1, 1, 1],
                                          [0, 1, 0]])
    def test_expansion(self, features):
        """Checks the queue for the default settings."""

        op = qml.templates.BasisEmbedding(features=features, wires=range(3))
        tape = op.expand()

        assert len(tape.operations) == features.count(1)
        for gate in tape.operations:
            assert gate.name == 'PauliX'

    @pytest.mark.parametrize("state", [[0, 1],
                                       [1, 1],
                                       [1, 0],
                                       [0, 0]])
    def test_state(self, state):
        """Checks that the correct state is prepared."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.BasisEmbedding(features=x, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        res = circuit(x=state)
        expected = [1 if s == 0 else -1 for s in state]
        assert np.allclose(res, expected)

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""
        features = [1, 0, 1]

        dev = qml.device("default.qubit", wires=3)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k"])

        @qml.qnode(dev)
        def circuit():
            qml.templates.BasisEmbedding(features, wires=range(3))
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.templates.BasisEmbedding(features, wires=["z", "a", "k"])
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)


class TestParameters:
    """Test inputs and pre-processing."""

    def test_too_many_input_bits_exception(self):
        """Verifies that exception thrown if there are more features than qubits."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError):
            circuit(x=np.array([0, 1, 1]))

    def test_not_enough_input_bits_exception(self):
        """Verifies that exception thrown if there are less features than qubits."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError):
            circuit(x=np.array([0]))

    def test_input_not_binary_exception(self):
        """Verifies that exception raised if the features contain
        values other than zero and one."""

        n_subsystems = 2
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.BasisEmbedding(features=x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Basis state must only consist of"):
            circuit(x=[2, 3])

    def test_exception_wrong_dim(self):
        """Verifies that exception is raised if the
        number of dimensions of features is incorrect."""

        n_subsystems = 2
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            qml.templates.BasisEmbedding(features=x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="Features must be one-dimensional"):
            circuit(x=[[1], [0]])
