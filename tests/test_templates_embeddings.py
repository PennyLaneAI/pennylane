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
"""
Unit tests for the :mod:`pennylane.templates.utils` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from math import pi
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import (AngleEmbedding,
                                            BasisEmbedding,
                                            AmplitudeEmbedding,
                                            DisplacementEmbedding,
                                            SqueezingEmbedding)
from pennylane import Beamsplitter


@pytest.fixture(scope="session",
                params=[1, 2, 5])
def n_subsystems(request):
    """Number of qubits or modes."""
    return request.param


@pytest.fixture(scope="session")
def qubit_device(n_subsystems):
    """Number of qubits or modes."""
    return qml.device('default.qubit', wires=n_subsystems)


@pytest.fixture(scope="session")
def gaussian_device(n_subsystems):
    """Number of qubits or modes."""
    return qml.device('default.gaussian', wires=n_subsystems)


class TestAngleEmbd:
    """ Tests the pennylane.templates.embeddings.AngleEmbedding method."""

    def test_angle_embedding_state_rotx(self, qubit_device, n_subsystems):
        """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='X' strategy."""

        features = [pi/2,  pi/2, pi/4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='X')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='X')
            return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [1, -1, 0, 1, 1]

        assert np.allclose(res, target[:n_subsystems])

    def test_angle_embedding_exception_rotx(self):
        """Verifies that pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='X' strategy raises an exception if there are fewer
           rotation gates than features."""

        features = [0, 0, 0, 0]
        n_subsystems = 1
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='X')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='X')
            return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=features)
        assert excinfo.value.args[0] == 'Number of features to embed cannot be larger than num_wires, but is 4.'

    def test_angle_embedding_state_roty(self, qubit_device, n_subsystems):
        """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='Y' strategy."""

        features = [pi/2,  pi/2, pi/4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Y')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Y')
            return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, -1, 0, 1, 1]

        assert np.allclose(res, target[:n_subsystems])

    def test_angle_embedding_exception_roty(self):
        """Verifies that pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='Z' strategy raises an exception if there are fewer
           rotation gates than features."""

        features = [0, 0, 0, 0]
        n_subsystems = 1
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Y')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Y')
            return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=features)
        assert excinfo.value.args[0] == 'Number of features to embed cannot be larger than num_wires, but is 4.'

    def test_angle_embedding_state_rotz(self, qubit_device, n_subsystems):
        """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='Z' strategy."""

        features = [pi/2,  pi/2, pi/4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Z')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Z')
            return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, 1, 1, 1, 1]
        assert np.allclose(res, target[:n_subsystems])

    def test_angle_embedding_exception_rotz(self):
        """Verifies that pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='Z' strategy raises an exception if there are fewer
           rotation gates than features."""

        features = [0, 0, 0, 0]
        n_subsystems = 1
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Z')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Z')
            return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=features)
        assert excinfo.value.args[0] == 'Number of features to embed cannot be larger than num_wires, but is 4.'

    def test_angle_embedding_exception_wrongrot(self):
        """Verifies that pennylane.templates.embeddings.AngleEmbedding() raises an exception if the
        rotation strategy is unknown."""

        n_subsystems = 1
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, n_wires=n_subsystems, rotation='A')
            return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[1])
        assert excinfo.value.args[0] == 'Rotation has to be `X`, `Y` or `Z`, got A.'


class TestBasisEmb:
    """ Tests the pennylane.templates.embeddings.BasisEmbedding method."""

    def test_basis_embedding_state(self):
        """Checks the state produced by pennylane.templates.embeddings.BasisEmbedding()."""

        state = np.array([0, 1])
        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(basis_state=x, n_qubits=2)
            return [qml.expval.PauliZ(i) for i in range(n_qubits)]

        res = circuit(x=state)
        assert np.allclose(res, [1, -1])

    def test_basis_embedding_exception_subsystems(self):
        """Verifies that pennylane.templates.embeddings.BasisEmbedding() throws exception
        if number of subsystems wrong."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(basis_state=x, n_qubits=n_qubits)
            return qml.expval.PauliZ(0)

        with pytest.raises(ValueError) as excinfo:
            circuit(x=np.array([0, 1, 1]))
        assert excinfo.value.args[0] == 'BasisEmbedding requires a feature vector of size n_qubits which is 2, got 3.'
        with pytest.raises(ValueError) as excinfo:
            circuit(x=np.array([1]))
        assert excinfo.value.args[0] == 'BasisEmbedding requires a feature vector of size n_qubits which is 2, got 1.'


class TestAmplitudeEmb:
    """ Tests the pennylane.templates.embeddings.AmplitudeEmbedding method."""

    def test_amplitude_embedding_state(self):
        """Checks the state produced by pennylane.templates.embeddings.AmplitudeEmbedding()."""

        features = np.array([0, 1, 0, 0])
        n_qubits = 2

        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, n_qubits=2)
            return [qml.expval.PauliZ(i) for i in range(n_qubits)]

        res = circuit(x=features)

        assert np.allclose(res, [1, -1])

    def test_amplitude_embedding_exception_subsystems(self):
        """Verifies that pennylane.templates.embeddings.AmplitudeEmbedding() throws exception
        if number of subsystems wrong."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, n_qubits=n_qubits)
            return qml.expval.PauliZ(0)

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[np.sqrt(0.2), np.sqrt(0.8), 0, 0, 0])
        assert excinfo.value.args[0] == 'AmplitudeEmbedding requires a feature vector of size 2**n_qubits ' \
                                        'which is 4, got 5.'
        with pytest.raises(ValueError) as excinfo:
            circuit(x=[np.sqrt(0.2), np.sqrt(0.8)])
        assert excinfo.value.args[0] == 'AmplitudeEmbedding requires a feature vector of size 2**n_qubits ' \
                                        'which is 4, got 2.'


class TestSqueezingEmb:
    """ Tests the pennylane.templates.embeddings.SqueezingEmbedding method."""

    def test_squeezing_embedding_state_execution_amplitude(self):
        """Checks the state produced by pennylane.templates.embeddings.SqueezingEmbedding()
        using the amplitude execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, n_wires=2, execution='amplitude', c=1)
            return [qml.expval.MeanPhoton(wires=0), qml.expval.MeanPhoton(wires=1)]

        assert np.allclose(circuit(x=features), [2.2784, 0.09273], atol=0.001)

    def test_squeezing_embedding_state_execution_phase(self):
        """Checks the state produced by pennylane.templates.embeddings.SqueezingEmbedding()
        using the phase execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, n_wires=2, execution='phase', c=1)
            Beamsplitter(pi/2, 0, wires=[0, 1])
            SqueezingEmbedding(features=[0, 0], n_wires=2, execution='phase', c=1)
            return [qml.expval.MeanPhoton(wires=0), qml.expval.MeanPhoton(wires=1)]

        assert np.allclose(circuit(x=features), [12.86036, 8.960306], atol=0.001)

    def test_squeezing_embedding_exception_subsystems(self):
        """Verifies that pennylane.templates.embeddings.SqueezingEmbedding() throws exception
        if number of subsystems wrong."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, n_wires=2, execution='phase')
            return [qml.expval.X(i) for i in range(n_wires)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[0.2, 0.3, 0.4])
        assert excinfo.value.args[0] == 'SqueezingEmbedding requires a feature vector of size n_wires ' \
                                        'which is 2, got 3.'
        with pytest.raises(ValueError) as excinfo:
            circuit(x=[0.2])
        assert excinfo.value.args[0] == 'SqueezingEmbedding requires a feature vector of size n_wires ' \
                                        'which is 2, got 1.'

    def test_squeezing_embedding_exception_strategy(self):
        """Verifies that pennylane.templates.embeddings.SqueezingEmbedding() throws exception
        if strategy unknown."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, n_wires=2, execution='A')
            return [qml.expval.X(i) for i in range(n_wires)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[1, 2])
        assert excinfo.value.args[0] == 'Execution strategy A not known. Has to be `phase` or `amplitude`.'


class TestDisplacementEmb:
    """ Tests the pennylane.templates.embeddings.DisplacementEmbedding method."""

    def test_displacement_embedding_state_execution_amplitude(self):
        """Checks the state produced by pennylane.templates.embeddings.DisplacementEmbedding()
        using the amplitude execution method."""

        features = np.array([0.1, 1.2])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, n_wires=2, execution='amplitude', c=1.)
            return [qml.expval.MeanPhoton(wires=0), qml.expval.MeanPhoton(wires=1)]

        assert np.allclose(circuit(x=features), [0.01, 1.44], atol=0.001)

    def test_displacement_embedding_state_execution_phase(self):
        """Checks the state produced by pennylane.templates.embeddings.DisplacementEmbedding()
        using the phase execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, n_wires=2, execution='phase', c=1.)
            Beamsplitter(pi/2, 0, wires=[0, 1])
            DisplacementEmbedding(features=[0, 0], n_wires=2, execution='phase', c=1.)
            return [qml.expval.MeanPhoton(wires=0), qml.expval.MeanPhoton(wires=1)]

        assert np.allclose(circuit(x=features), [0.089327, 2.724715], atol=0.01)

    def test_displacement_embedding_exception_subsystems(self):
        """Verifies that pennylane.templates.embeddings.DisplacementEmbedding() throws exception
        if number of subsystems wrong."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, n_wires=2, execution='phase')
            return [qml.expval.X(i) for i in range(n_wires)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[0.2, 0.3, 0.4])
        assert excinfo.value.args[0] == 'DisplacementEmbedding requires a feature vector of size n_wires ' \
                                        'which is 2, got 3.'
        with pytest.raises(ValueError) as excinfo:
            circuit(x=[0.2])
        assert excinfo.value.args[0] == 'DisplacementEmbedding requires a feature vector of size n_wires ' \
                                        'which is 2, got 1.'

    def test_displacement_embedding_exception_strategy(self):
        """Verifies that pennylane.templates.embeddings.DisplacementEmbedding() throws exception
        if strategy unknown."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, n_wires=2, execution='A')
            return [qml.expval.X(i) for i in range(n_wires)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[1, 2])
        assert excinfo.value.args[0] == 'Execution strategy A not known. Has to be `phase` or `amplitude`.'



