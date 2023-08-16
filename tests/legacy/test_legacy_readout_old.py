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
Unit tests for the :mod:`pennylane.devices.DefaultMixed` device.
"""
# pylint: disable=protected-access

import pytest
import numpy as np

import pennylane as qml

INV_SQRT2 = 1 / np.sqrt(2)


class TestReadoutError:
    """Tests for measurement readout error"""

    prob_and_expected_expval = [
        (0, np.array([1, 1])),
        (0.5, np.array([0, 0])),
        (1, np.array([-1, -1])),
    ]

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", prob_and_expected_expval)
    def test_readout_expval_pauliz(self, nr_wires, prob, expected):
        """Tests the measurement results for expval of PauliZ"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", prob_and_expected_expval)
    def test_readout_expval_paulix(self, nr_wires, prob, expected):
        """Tests the measurement results for expval of PauliX"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("prob", [0, 0.5, 1])
    @pytest.mark.parametrize(
        "nr_wires, expected", [(1, np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]]))]
    )
    def test_readout_state(self, nr_wires, prob, expected):
        """Tests the state output is not affected by readout error"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.state()

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob", [0, 0.5, 1])
    def test_readout_density_matrix(self, nr_wires, prob):
        """Tests the density matrix output is not affected by readout error"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.density_matrix(wires=1)

        res = circuit()
        expected = np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]])
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("prob", [0, 0.5, 1])
    @pytest.mark.parametrize("nr_wires", [2, 3])
    def test_readout_vnentropy_and_mutualinfo(self, nr_wires, prob):
        """Tests the output of qml.vn_entropy and qml.mutual_info is not affected by readout error"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.vn_entropy(wires=0, log_base=2), qml.mutual_info(
                wires0=[0], wires1=[1], log_base=2
            )

        res = circuit()
        expected = np.array([0, 0])
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize(
        "prob, expected", [(0, [np.zeros(2), np.zeros(2)]), (1, [np.ones(2), np.ones(2)])]
    )
    def test_readout_sample(self, nr_wires, prob, expected):
        """Tests the sample output with readout error"""
        dev = qml.device("default.mixed", shots=2, wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(wires=[0, 1])

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", [(0, {"00": 100}), (1, {"11": 100})])
    def test_readout_counts(self, nr_wires, prob, expected):
        """Tests the counts output with readout error"""
        dev = qml.device("default.mixed", shots=100, wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.counts(wires=[0, 1])

        res = circuit()
        assert res == expected

    prob_and_expected_probs = [
        (0, np.array([1, 0])),
        (0.5, np.array([0.5, 0.5])),
        (1, np.array([0, 1])),
    ]

    @pytest.mark.parametrize("nr_wires", [2, 3])
    @pytest.mark.parametrize("prob, expected", prob_and_expected_probs)
    def test_readout_probs(self, nr_wires, prob, expected):
        """Tests the measurement results for probs"""
        dev = qml.device("default.mixed", wires=nr_wires, readout_prob=prob)

        @qml.qnode(dev)
        def circuit():
            return qml.probs(wires=0)

        res = circuit()
        assert np.allclose(res, expected)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    def test_prob_out_of_range(self, nr_wires):
        """Tests that an error is raised when readout error probability is outside [0,1]"""
        with pytest.raises(ValueError, match="should be in the range"):
            qml.device("default.mixed", wires=nr_wires, readout_prob=2)

    @pytest.mark.parametrize("nr_wires", [2, 3])
    def test_prob_type(self, nr_wires):
        """Tests that an error is raised for wrong data type of readout error probability"""
        with pytest.raises(TypeError, match="should be an integer or a floating-point number"):
            qml.device("default.mixed", wires=nr_wires, readout_prob="RandomNum")
