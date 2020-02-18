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
"""
Unit tests for the :mod:`pennylane.template.embeddings` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from math import pi
import numpy as np
import pennylane as qml
from pennylane.ops import RY
from pennylane.templates.embeddings import (
    AngleEmbedding,
    BasisEmbedding,
    AmplitudeEmbedding,
    QAOAEmbedding,
    DisplacementEmbedding,
    SqueezingEmbedding,
)
from pennylane import Beamsplitter


class TestAmplitudeEmbedding:
    """ Tests the AmplitudeEmbedding method."""

    INPT = [
        np.array([0, 1, 0, 0]),
        1 / np.sqrt(4) * np.array([1, 1, 1, 1]),
        np.array(
            [
                np.complex(-np.sqrt(0.1), 0.0),
                np.sqrt(0.3),
                np.complex(0, -np.sqrt(0.1)),
                np.sqrt(0.5),
            ]
        ),
    ]

    NOT_ENOUGH_FEATURES = [
        np.array([0, 1, 0]),
        1 / np.sqrt(3) * np.array([1, 1, 1]),
        np.array([np.complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), np.complex(0, -np.sqrt(0.6))]),
    ]

    TOO_MANY_FEATURES = [
        [0, 0, 0, 1, 0],
        1 / np.sqrt(8) * np.array([1] * 8),
        [np.complex(-np.sqrt(0.1), 0.0), np.sqrt(0.3), np.complex(0, -np.sqrt(0.6)), 0.0, 0.0,],
    ]

    @pytest.mark.parametrize("inpt", INPT)
    def test_amplitude_embedding_prepares_state(self, inpt):
        """Checks the state produced by AmplitudeEmbedding() for real and complex
        inputs."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=None, normalize=False)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        circuit(x=inpt)
        state = dev._state
        assert np.allclose(state, inpt)

    @pytest.mark.parametrize("inpt", NOT_ENOUGH_FEATURES)
    @pytest.mark.parametrize("pad", [complex(0.1, 0.1), 0.0, 1.0])
    def test_amplitude_embedding_prepares_padded_state(self, inpt, pad):
        """Checks the state produced by AmplitudeEmbedding() for real and complex padding constants."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=pad, normalize=False)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        circuit(x=inpt)
        state = dev._state
        assert len(set(state[len(inpt) :])) == 1

    @pytest.mark.parametrize("inpt", INPT)
    def test_amplitude_embedding_throws_exception_if_not_normalized(self, inpt):
        """Checks that AmplitudeEmbedding() throws exception when state is not normalized and `normalize=False`."""
        not_nrmlzd = 2 * inpt
        n_qubits = 2
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=None, normalize=False)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        with pytest.raises(ValueError, match="'features' must be a vector of length"):
            circuit(x=not_nrmlzd)

    @pytest.mark.parametrize("inpt", NOT_ENOUGH_FEATURES)
    def test_amplitude_embedding_throws_exception_if_fewer_features_than_amplitudes(self, inpt):
        """Verifies that AmplitudeEmbedding() throws exception
        if the number of features is fewer than the number of amplitudes, and
        no automatic padding is chosen."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=None, normalize=False)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'features' must be of shape"):
            circuit(x=inpt)

    @pytest.mark.parametrize("inpt", TOO_MANY_FEATURES)
    def test_amplitude_embedding_throws_exception_if_more_features_than_amplitudes(self, inpt):
        """Verifies that AmplitudeEmbedding() throws exception
        if the number of features is larger than the number of amplitudes, and
        no automatic padding is chosen."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=None, normalize=False)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'features' must be of shape"):
            circuit(x=inpt)

    @pytest.mark.parametrize("inpt", TOO_MANY_FEATURES)
    def test_amplitude_embedding_with_padding_throws_exception_if_more_features_than_amplitudes(
        self, inpt
    ):
        """Verifies that AmplitudeEmbedding() throws exception
        if the number of features is larger than the number of amplitudes, and
        automatic padding is chosen."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            AmplitudeEmbedding(features=x, wires=range(n_qubits), pad=0.0, normalize=False)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'features' must be of shape"):
            circuit(x=inpt)


class TestAngleEmbedding:
    """ Tests the AngleEmbedding method."""

    def test_angle_embedding_state_rotx(self, qubit_device, n_subsystems):
        """Checks the state produced by AngleEmbedding()
           using the rotation='X' strategy."""

        features = [pi / 2, pi / 2, pi / 4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation="X")
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation="X")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [1, -1, 0, 1, 1]

        assert np.allclose(res, target[:n_subsystems])

    def test_angle_embedding_state_roty(self, qubit_device, n_subsystems):
        """Checks the state produced by AngleEmbedding()
           using the rotation='Y' strategy."""

        features = [pi / 2, pi / 2, pi / 4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Y")
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Y")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, -1, 0, 1, 1]
        assert np.allclose(res, target[:n_subsystems])

    def test_angle_embedding_state_rotz(self, qubit_device, n_subsystems):
        """Checks the state produced by AngleEmbedding()
           using the rotation='Z' strategy."""

        features = [pi / 2, pi / 2, pi / 4, 0]

        @qml.qnode(qubit_device)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Z")
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation="Z")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [-1, 1, 1, 1, 1]
        assert np.allclose(res, target[:n_subsystems])

    @pytest.mark.parametrize("strategy", ["X", "Y", "Z"])
    def test_angle_embedding_exception_fewer_rotations(self, strategy):
        """Verifies that AngleEmbedding() raises an exception if there are fewer
           rotation gates than features."""

        features = [0, 0, 0, 0]
        n_subsystems = 1
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation=strategy)
            qml.PauliX(wires=0)
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation=strategy)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        with pytest.raises(ValueError, match="'features' must be of shape"):
            circuit(x=features)

    def test_angle_embedding_exception_wrongrot(self):
        """Verifies that AngleEmbedding() raises an exception if the
        rotation strategy is unknown."""

        n_subsystems = 1
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, wires=range(n_subsystems), rotation="A")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        with pytest.raises(ValueError, match="did not recognize option"):
            circuit(x=[1])

    def test_angle_embedding_exception_wires_not_valid(self):
        """Verifies that AngleEmbedding() raises an exception if ``wires`` is not
        a valid list of indices."""

        n_subsystems = 5
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            AngleEmbedding(features=x, wires="a")
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit(x=[1])


class TestBasisEmbedding:
    """ Tests the BasisEmbedding method."""

    def test_basis_embedding_state(self):
        """Checks the state produced by BasisEmbedding()."""

        state = np.array([0, 1])
        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires=range(2))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        res = circuit(x=state)
        assert np.allclose(res, [1, -1])

    def test_basis_embedding_too_many_input_bits_exception(self):
        """Verifies that BasisEmbedding() throws exception if there are more features than qubits."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError):
            circuit(x=np.array([0, 1, 1]))

    def test_basis_embedding_not_enough_input_bits_exception(self):
        """Verifies that BasisEmbedding() throws exception if there are less features than qubits."""

        n_qubits = 2
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires=range(2))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError):
            circuit(x=np.array([0]))

    def test_basis_embedding_illegal_wires_exception(self):
        """Verifies that BasisEmbedding() raises an exception if ``wires`` has incorrect format."""

        n_subsystems = 2
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires="a")
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit(x=[1, 0])

    def test_basis_embedding_input_not_binary_exception(self):
        """Verifies that BasisEmbedding() raises an exception if the features contain
        values other than zero and one."""

        n_subsystems = 2
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            BasisEmbedding(features=x, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'basis_state' must only consist of"):
            circuit(x=[2, 3])


class TestQAOAEmbedding:
    """ Tests the QAOAEmbedding method."""

    def test_qaoa_embedding_state_zero_weights(self, qubit_device, n_subsystems, tol):
        """Checks the state produced by QAOAEmbedding() is correct if the weights are zero."""

        features = [pi, pi / 2, pi / 4, 0]
        if n_subsystems == 1:
            shp = (1, 1)
        elif n_subsystems == 2:
            shp = (1, 3)
        else:
            shp = (1, 2 * n_subsystems)

        weights = np.zeros(shape=shp)

        @qml.qnode(qubit_device)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        target = [1, -1, 0, 1, 1]
        assert np.allclose(res, target[:n_subsystems], atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "n_subsystems, weights, target",
        [
            (1, [[pi / 2]], [0]),
            (2, [[1, pi / 2, pi / 4]], [0, 1 / np.sqrt(2)]),
            (3, [[0, 0, 0, pi, pi / 2, pi / 4]], [-1, 0, 1 / np.sqrt(2)]),
        ],
    )
    def test_qaoa_embedding_state_zero_features_local_field_ry(
        self, n_subsystems, weights, target, tol
    ):
        """Checks the state produced by QAOAEmbedding() is correct if the features are zero. Uses RY local fields."""

        features = np.zeros(shape=(n_subsystems,))
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems), local_field="Y")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "n_subsystems, weights, target",
        [
            (1, [[pi / 2]], [0]),
            (2, [[1, pi / 2, pi / 4]], [0, 1 / np.sqrt(2)]),
            (3, [[0, 0, 0, pi, pi / 2, pi / 4]], [-1, 0, 1 / np.sqrt(2)]),
        ],
    )
    def test_qaoa_embedding_state_zero_features_local_field_rx(
        self, n_subsystems, weights, target, tol
    ):
        """Checks the state produced by QAOAEmbedding() is correct if the features are zero. Uses RX local fields."""

        features = np.zeros(shape=(n_subsystems,))
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems), local_field="X")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "n_subsystems, weights, target",
        [
            (1, [[pi / 2]], [1]),
            (2, [[1, pi / 2, pi / 4]], [1, 1]),
            (3, [[0, 0, 0, pi, pi / 2, pi / 4]], [1, 1, 1]),
        ],
    )
    def test_qaoa_embedding_state_zero_features_local_field_rz(
        self, n_subsystems, weights, target, tol
    ):
        """Checks the state produced by QAOAEmbedding() is correct if the features are zero. Uses RZ local fields."""

        features = np.zeros(shape=(n_subsystems,))
        dev = qml.device("default.qubit", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(n_subsystems), local_field="Z")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_subsystems)]

        res = circuit(x=features[:n_subsystems])
        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "features, weights, target",
        [
            ([np.pi / 2, 0], [[np.pi, 0, 0]], [-1, 1]),
            ([0, np.pi / 2], [[np.pi, 0, 0]], [1, -1]),
            ([np.pi / 2, np.pi / 2], [[np.pi, 0, 0]], [-1, -1]),
        ],
    )
    def test_qaoa_embedding_state_features_and_zz(self, features, weights, target, tol):
        """Checks the state produced by QAOAEmbedding() is correct if the features and weights are nonzero.
        Uses RY local fields."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(2), local_field="Y")
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        res = circuit(x=features)
        assert np.allclose(res, target, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "n_wires, features, weights, target",
        [
            (2, [0], [[0, 0, np.pi / 2]], [1, 0]),
            (3, [0, 0], [[0, 0, 0, 0, 0, np.pi / 2]], [1, 1, 0]),
        ],
    )
    def test_qaoa_embedding_state_more_qubits_than_features(
        self, n_wires, features, weights, target, tol
    ):
        """Checks the state is correct if there are more qubits than features."""

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(n_wires), local_field="Z")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        res = circuit(x=features)
        assert np.allclose(res, target, atol=tol, rtol=0)

    def test_qaoa_embedding_exception_fewer_wires_than_features(self,):
        """Verifies that QAOAEmbedding() raises an exception if there are fewer
           wires than features."""

        features = [0, 0, 0, 0]
        n_wires = 1
        weights = np.zeros(shape=(1, 2 * n_wires))
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="'features' must be of shape"):
            circuit(x=features)

    def test_qaoa_embedding_exception_wrongrot(self):
        """Verifies that QAOAEmbedding() raises an exception if the
        rotation strategy is unknown."""

        n_wires = 1
        weights = np.zeros(shape=(1, 2 * n_wires))
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires=range(n_wires), local_field="A")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="did not recognize option"):
            circuit(x=[1])

    def test_qaoa_embedding_exception_wires_not_valid(self):
        """Verifies that QAOAEmbedding() raises an exception if ``wires`` is not
        a valid list of indices."""

        n_wires = 5
        weights = np.zeros(shape=(1, 2 * n_wires))
        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            QAOAEmbedding(features=x, weights=weights, wires="a")
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit(x=[1])


class TestDisplacementEmbedding:
    """ Tests the DisplacementEmbedding method."""

    def test_displacement_embedding_state_execution_amplitude(self):
        """Checks the state produced by DisplacementEmbedding()
        using the amplitude execution method."""

        features = np.array([0.1, 1.2])
        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method="amplitude", c=1.0)
            return [
                qml.expval(qml.NumberOperator(wires=0)),
                qml.expval(qml.NumberOperator(wires=1)),
            ]

        assert np.allclose(circuit(x=features), [0.01, 1.44], atol=0.001)

    def test_displacement_embedding_state_execution_phase(self):
        """Checks the state produced by DisplacementEmbedding()
        using the phase execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method="phase", c=1.0)
            Beamsplitter(pi / 2, 0, wires=[0, 1])
            DisplacementEmbedding(features=[0, 0], wires=range(n_wires), method="phase", c=1.0)
            return [
                qml.expval(qml.NumberOperator(wires=0)),
                qml.expval(qml.NumberOperator(wires=1)),
            ]

        assert np.allclose(circuit(x=features), [0.089327, 2.724715], atol=0.01)

    def test_squeezing_embedding_exception_for_wrong_num_wires(self):
        """Verifies that DisplacementEmbedding() throws exception
        if number of subsystems wrong."""

        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method="phase")
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="'features' must be of shape"):
            circuit(x=[0.2, 0.3, 0.4])

    def test_displacement_embedding_strategy_not_recognized_exception(self):
        """Verifies that DisplacementEmbedding() throws exception
        if strategy unknown."""

        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires=range(n_wires), method="A")
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="did not recognize option"):
            circuit(x=[1, 2])

    def test_displacement_embedding_wires_not_valid_exception(self):
        """Verifies that DisplacementEmbedding() raises an exception if ``wires`` is not
        a list of indices."""

        n_subsystems = 5
        dev = qml.device("default.gaussian", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires="a")
            return qml.expval(qml.X(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit(x=[1])


class TestSqueezingEmbedding:
    """ Tests the SqueezingEmbedding method."""

    def test_squeezing_embedding_state_execution_amplitude(self):
        """Checks the state produced by SqueezingEmbedding()
        using the amplitude execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method="amplitude", c=1)
            return [
                qml.expval(qml.NumberOperator(wires=0)),
                qml.expval(qml.NumberOperator(wires=1)),
            ]

        assert np.allclose(circuit(x=features), [2.2784, 0.09273], atol=0.001)

    def test_squeezing_embedding_state_execution_phase(self):
        """Checks the state produced by SqueezingEmbedding()
        using the phase execution method."""

        features = np.array([1.2, 0.3])
        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method="phase", c=1)
            Beamsplitter(pi / 2, 0, wires=[0, 1])
            SqueezingEmbedding(features=[0, 0], wires=range(n_wires), method="phase", c=1)
            return [
                qml.expval(qml.NumberOperator(wires=0)),
                qml.expval(qml.NumberOperator(wires=1)),
            ]

        assert np.allclose(circuit(x=features), [12.86036, 8.960306], atol=0.001)

    def test_squeezing_embedding_exception_for_wrong_num_wires(self):
        """Verifies that SqueezingEmbedding() throws exception if number of modes is wrong."""

        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method="phase")
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="'features' must be of shape"):
            circuit(x=[0.2, 0.3, 0.4])

    def test_squeezing_embedding_strategy_not_recognized_exception(self):
        """Verifies that SqueezingEmbedding() throws exception
        if strategy unknown."""

        n_wires = 2
        dev = qml.device("default.gaussian", wires=n_wires)

        @qml.qnode(dev)
        def circuit(x=None):
            SqueezingEmbedding(features=x, wires=range(n_wires), method="A")
            return [qml.expval(qml.X(i)) for i in range(n_wires)]

        with pytest.raises(ValueError, match="did not recognize option"):
            circuit(x=[1, 2])

    def test_squeezing_embedding_wires_not_valid_exception(self):
        """Verifies that SqueezingtEmbedding() raises an exception if ``wires`` is not
        a list of indices."""

        n_subsystems = 5
        dev = qml.device("default.gaussian", wires=n_subsystems)

        @qml.qnode(dev)
        def circuit(x=None):
            DisplacementEmbedding(features=x, wires="a")
            return qml.expval(qml.X(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit(x=[1])
