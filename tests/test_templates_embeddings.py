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


class TestAngleEmbd:
    """ Tests the pennylane.templates.embeddings.AngleEmbedding method."""

    def test_angle_embedding_state_rotx(self, qubit_device, n_subsystems):
        """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='X' strategy."""

        features = [pi/2, pi/2, pi/4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation='X')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation='X')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [1, -1, 0, 1, 1]

        assert np.allclose(res, target[:n_subsystems])

    def test_angle_embedding_state_roty(self, qubit_device, n_subsystems):
        """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='Y' strategy."""

        features = [pi/2, pi/2, pi/4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation='Y')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation='Y')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, -1, 0, 1, 1]
        assert np.allclose(res, target[:n_subsystems])

    def test_angle_embedding_state_rotz(self, qubit_device, n_subsystems):
        """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
           using the rotation='Z' strategy."""

        features = [pi/2, pi/2, pi/4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation='Z')
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation='Z')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, 1, 1, 1, 1]
        assert np.allclose(res, target[:n_subsystems])

    @pytest.mark.parametrize('strategy', ['X', 'Y', 'Z'])
    def test_angle_embedding_exception_fewer_rotations(self, strategy):
        """Verifies that pennylane.templates.embeddings.AngleEmbedding() raises an exception if there are fewer
           rotation gates than features."""

        features = [0, 0, 0, 0]
        n_subsystems = 1
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation=strategy)
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation=strategy)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        with pytest.raises(ValueError, match='Number of features to embed cannot be larger than number of '
                                             'wires, which is 1; got 4.'):
            circuit(x=features)

    def test_angle_embedding_exception_wrongrot(self):
        """Verifies that pennylane.templates.embeddings.AngleEmbedding() raises an exception if the
        rotation strategy is unknown."""

        n_subsystems = 1
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation='A')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        with pytest.raises(ValueError, match='Rotation has to be `X`, `Y` or `Z`; got A.'):
            circuit(x=[1])

    def test_angle_embedding_exception_wiresnolist(self):
        """Verifies that pennylane.templates.embeddings.AngleEmbedding() raises an exception if ``wires`` is not
        iterable."""

        n_subsystems = 5
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=3, rotation='A')
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match='Wires must be passed as a list of integers; got 3.'):
            circuit(x=[1])


class TestBasisEmbedding:
    """ Tests the pennylane.templates.embeddings.BasisEmbedding method."""

    def test_basis_embedding_state(self):
        """Checks the state produced by pennylane.templates.embeddings.BasisEmbedding()."""

        state = np.array([0, 1])
        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        res = circuit(x=state)
        assert np.allclose(res, [1, -1])

    def test_basis_embedding_exception_subsystems(self):
        """Verifies that pennylane.templates.embeddings.BasisEmbedding() throws exception
        if number of subsystems wrong."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match='Number of bits to embed cannot be larger than '
                                             'number of wires, which is 2; got 3.'):
            circuit(x=np.array([0, 1, 1]))

    def test_basis_embedding_exception_wiresnolist(self):
        """Verifies that pennylane.templates.embeddings.BasisEmbedding() raises an exception if ``wires`` is not
        iterable."""

        n_subsystems = 5
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires=3)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match='Wires must be passed as a list of integers; got 3.'):
            circuit(x=[1])


class TestAmplitudeEmbedding:
    """ Tests the pennylane.templates.embeddings.AmplitudeEmbedding method."""

    @pytest.mark.parametrize("inpt", [np.array([0, 1, 0, 0]),
                                      1/np.sqrt(4)*np.array([1, 1, 1, 1]),
                                      np.array([np.complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3),
                                                np.complex(0, -np.sqrt(0.1)), np.sqrt(0.5)])])
    def test_amplitude_embedding_prepares_state(self, inpt):
        """Checks the state produced by pennylane.templates.embeddings.AmplitudeEmbedding() for real and complex
        inputs."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=False, normalize=False)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        circuit(x=inpt)
        state = dev._state
        assert np.allclose(state, inpt)

    def test_amplitude_embedding_throws_exception_if_wrong_number_of_subsystems(self):
        """Verifies that pennylane.templates.embeddings.AmplitudeEmbedding() throws exception
        if number of subsystems is wrong."""

        n_qubits = 2
        dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=False, normalize=False)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[np.sqrt(0.2), np.sqrt(0.8), 0, 0, 0])
        assert excinfo.value.args[0] == "AmplitudeEmbedding requires the size of feature vector to " \
                                        "be smaller than or equal to 2**len(wires), which is 4; got 5."

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[np.sqrt(0.2), np.sqrt(0.8)])
        assert excinfo.value.args[0] == 'AmplitudeEmbedding must get a feature vector ' \
                                        'of size 2**len(wires), which is 4; got 2. Use ``pad=True`` to ' \
                                        'automatically pad the features with zeros.'

    def test_amplitude_embedding_throws_exception_notnormalized(self):
        """Verifies that pennylane.templates.embeddings.AmplitudeEmbedding() raises an exception if input is not
        normalized and 'normalization' is set to False."""

        n_subsystems = 5
        dev = qml.device('default.qubit', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=1, pad=False, normalize=False)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match='AmplitudeEmbedding requires a normalized feature vector. '
                                             'Set ``normalize=True`` to automatically normalize it.'):
            circuit(x=[1, 1])

    @pytest.mark.parametrize("inpt, expected", [(np.array([0, 2]), [0, 1]),
                                                (1/np.sqrt(2)*np.array([2, 2]), [1/np.sqrt(2), 1/np.sqrt(2)]),
                                                ([np.complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3)], [-0.5, 0.8660254]),
                                                ([0, 2], [0, 1])])
    def test_amplitude_embedding_normalizes_keywordargs(self, inpt, expected):
        """Verifies that pennylane.templates.embeddings.AmplitudeEmbedding() normalizes an input passed as a keyword
         argument if `normalize` is set to `True`."""

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=0, pad=False, normalize=True)
            return qml.expval(qml.PauliZ(0))

        circuit(x=inpt)
        state = dev._state
        assert np.allclose(state, expected)

    @pytest.mark.parametrize("inpt, expected", [(np.array([0, 2]), [0, 1]),
                                                (1/np.sqrt(2)*np.array([2, 2]), [1/np.sqrt(2), 1/np.sqrt(2)]),
                                                ([np.complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3)], [-0.5, 0.8660254]),
                                                ([0, 2], [0, 1])])
    def test_amplitude_embedding_normalizes_positionalargs(self, inpt, expected):
        """Verifies that pennylane.templates.embeddings.AmplitudeEmbedding() normalizes an input passed as a positional
        argument if `normalize` is set to `True`."""

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(x):
            AmplitudeEmbedding(features=x, wires=0, pad=False, normalize=True)
            return qml.expval(qml.PauliZ(0))

        circuit(inpt)
        state = dev._state
        assert np.allclose(state, expected)


class TestSqueezingEmbedding:
    """ Tests the pennylane.templates.embeddings.SqueezingEmbedding method."""

    def test_squeezing_embedding_state_execution_amplitude(self):
        """Checks the state produced by pennylane.templates.embeddings.SqueezingEmbedding()
        using the amplitude execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method='amplitude', c=1)
            return [qml.expval(qml.NumberOperator(wires=0)), qml.expval(qml.NumberOperator(wires=1))]

        assert np.allclose(circuit(x=features), [2.2784, 0.09273], atol=0.001)

    def test_squeezing_embedding_state_execution_phase(self):
        """Checks the state produced by pennylane.templates.embeddings.SqueezingEmbedding()
        using the phase execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method='phase', c=1)
            Beamsplitter(pi/2, 0, wires=[0, 1])
            SqueezingEmbedding(features=[0, 0], wires=range(n_wires), method='phase', c=1)
            return [qml.expval(qml.NumberOperator(wires=0)), qml.expval(qml.NumberOperator(wires=1))]

        assert np.allclose(circuit(x=features), [12.86036, 8.960306], atol=0.001)

    def test_squeezing_embedding_exception_subsystems(self):
        """Verifies that pennylane.templates.embeddings.SqueezingEmbedding() throws exception
        if number of subsystems wrong."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method='phase')
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match='Number of features to embed cannot be larger than number of wires, '
                                             'which is 2; got 3.'):
            circuit(x=[0.2, 0.3, 0.4])

    def test_squeezing_embedding_exception_strategy(self):
        """Verifies that pennylane.templates.embeddings.SqueezingEmbedding() throws exception
        if strategy unknown."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method='A')
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[1, 2])
        assert excinfo.value.args[0] == 'Execution method \'A\' not known. Has to be \'phase\' or \'amplitude\'.'

    def test_squeezing_embedding_exception_wiresnolist(self):
        """Verifies that pennylane.templates.embeddings.SqueezingEmbedding() raises an exception if ``wires`` is not
        iterable."""

        n_subsystems = 5
        dev = qml.device('default.gaussian', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=3, method='A')
            return qml.expval(qml.X(0))

        with pytest.raises(ValueError, match='Wires must be passed as a list of integers; got 3.'):
            circuit(x=[1])


class TestDisplacementEmbedding:
    """ Tests the pennylane.templates.embeddings.DisplacementEmbedding method."""

    def test_displacement_embedding_state_execution_amplitude(self):
        """Checks the state produced by pennylane.templates.embeddings.DisplacementEmbedding()
        using the amplitude execution method."""

        features = np.array([0.1, 1.2])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method='amplitude', c=1.)
            return [qml.expval(qml.NumberOperator(wires=0)), qml.expval(qml.NumberOperator(wires=1))]

        assert np.allclose(circuit(x=features), [0.01, 1.44], atol=0.001)

    def test_displacement_embedding_state_execution_phase(self):
        """Checks the state produced by pennylane.templates.embeddings.DisplacementEmbedding()
        using the phase execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method='phase', c=1.)
            Beamsplitter(pi/2, 0, wires=[0, 1])
            DisplacementEmbedding(features=[0, 0], wires=range(n_wires), method='phase', c=1.)
            return [qml.expval(qml.NumberOperator(wires=0)), qml.expval(qml.NumberOperator(wires=1))]

        assert np.allclose(circuit(x=features), [0.089327, 2.724715], atol=0.01)

    def test_displacement_embedding_exception_subsystems(self):
        """Verifies that pennylane.templates.embeddings.DisplacementEmbedding() throws exception
        if number of subsystems wrong."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method='phase')
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match='Number of features to embed cannot be larger than number of wires, '
                                             'which is 2; got 3.'):
            circuit(x=[0.2, 0.3, 0.4])

    def test_displacement_embedding_exception_strategy(self):
        """Verifies that pennylane.templates.embeddings.DisplacementEmbedding() throws exception
        if strategy unknown."""

        n_wires = 2
        dev = qml.device('default.gaussian', wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method='A')
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError) as excinfo:
            circuit(x=[1, 2])
        assert excinfo.value.args[0] == 'Execution method \'A\' not known. Has to be \'phase\' or \'amplitude\'.'

    def test_displacement_embedding_exception_wiresnolist(self):
        """Verifies that pennylane.templates.embeddings.DisplacementEmbedding() raises an exception if ``wires`` is not
        iterable."""

        n_subsystems = 5
        dev = qml.device('default.gaussian', wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=3, method='A')
            return qml.expval(qml.X(0))

        with pytest.raises(ValueError, match='Wires must be passed as a list of integers; got 3.'):
            circuit(x=[1])
