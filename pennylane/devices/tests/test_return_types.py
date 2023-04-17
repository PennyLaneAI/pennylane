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
"""Tests that a device gives the same output as the default device."""
# pylint: disable=no-self-use,no-member,redefined-outer-name
import pytest

import pennylane as qml
from pennylane import numpy as np  # Import from PennyLane to mirror the standard approach in demos

pytestmark = pytest.mark.skip_unsupported

wires = [2, 3, 4]


def qubit_ansatz(x):
    """Qfunc ansatz"""
    qml.Hadamard(wires=[0])
    qml.CRX(x, wires=[0, 1])


class TestIntegrationMultipleReturns:
    """Test the new return types for multiple measurements, it should always return a tuple containing the single
    measurements.
    """

    def test_multiple_expval(self, device):
        """Return multiple expvals."""
        dev = qml.device(device, wires=2)

        obs1 = qml.Projector([0], wires=0)
        obs2 = qml.PauliZ(wires=1)
        func = qubit_ansatz

        def circuit(x):
            func(x)
            return qml.expval(obs1), qml.expval(obs2)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

    def test_multiple_var(self, device):
        """Return multiple vars."""
        dev = qml.device(device, wires=2)

        obs1 = qml.Projector([0], wires=0)
        obs2 = qml.PauliZ(wires=1)
        func = qubit_ansatz

        def circuit(x):
            func(x)
            return qml.var(obs1), qml.var(obs2)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

    # op1, wires1, op2, wires2
    multi_probs_data = [
        (None, [0], None, [0]),
        (None, [0], None, [0, 1]),
        (None, [0, 1], None, [0]),
        (None, [0, 1], None, [0, 1]),
        (qml.PauliZ(0), None, qml.PauliZ(1), None),
        (None, [0], qml.PauliZ(1), None),
        (qml.PauliZ(0), None, None, [0]),
        (qml.PauliZ(1), None, qml.PauliZ(0), None),
    ]

    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    def test_multiple_prob(self, op1, op2, wires1, wires2, device):  # pylint: disable=too-many-arguments
        """Return multiple probs."""

        dev = qml.device(device, wires=2)

        def circuit(x):
            qubit_ansatz(x)
            return qml.probs(op=op1, wires=wires1), qml.probs(op=op2, wires=wires2)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)
        assert len(res) == 2

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2 ** len(wires1),)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (2 ** len(wires2),)

    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    @pytest.mark.parametrize("wires3, wires4", wires)
    def test_mix_meas(self, op1, wires1, op2, wires2, wires3, wires4, device):  # pylint: disable=too-many-arguments
        """Return multiple different measurements."""
        if device == "default.qutrit":
            pytest.skip("Different test for DefaultQutrit.")

        dev = qml.device(device, wires=2)

        def circuit(x):
            qubit_ansatz(x)
            return (
                qml.probs(op=op1, wires=wires1),
                qml.vn_entropy(wires=wires3),
                qml.probs(op=op2, wires=wires2),
                qml.expval(qml.PauliZ(wires=wires4)),
            )

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        assert isinstance(res, tuple)
        assert len(res) == 4

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2 ** len(wires1),)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

        assert isinstance(res[2], np.ndarray)
        assert res[2].shape == (2 ** len(wires2),)

        assert isinstance(res[3], np.ndarray)
        assert res[3].shape == ()

    @pytest.mark.parametrize(
        "measurement",
        [qml.sample(qml.PauliZ(0)), qml.sample(wires=[0]), qml.sample(qml.GellMann(0, 3))],
    )
    def test_expval_sample(self, measurement, device, shots=100):
        """Test the expval and sample measurements together."""

        dev = qml.device(device, wires=2, shots=shots)
        func = qubit_ansatz
        obs = qml.PauliZ(1)

        def circuit(x):
            func(x)
            return qml.expval(obs), qml.apply(measurement)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        # Expval
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        # Sample
        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (shots,)

    @pytest.mark.parametrize(
        "measurement",
        [qml.counts(qml.PauliZ(0)), qml.counts(wires=[0]), qml.counts(qml.GellMann(0, 3))],
    )
    def test_expval_counts(self, measurement, device, shots=100):
        """Test the expval and counts measurements together."""

        dev = qml.device(device, wires=2, shots=shots)
        func = qubit_ansatz
        obs = qml.PauliZ(1)

        def circuit(x):
            func(x)
            return qml.expval(obs), qml.apply(measurement)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        # Expval
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        # Counts
        assert isinstance(res[1], dict)
        assert sum(res[1].values()) == shots

    wires = [2, 3, 4, 5]

    @pytest.mark.parametrize("wires", wires)
    def test_list_one_expval(self, wires, device):
        """Return a comprehension list of one expvals."""
        dev = qml.device(device, wires=wires)
        func = qubit_ansatz
        obs = qml.PauliZ(0)

        def circuit(x):
            func(x)
            return [qml.expval(obs)]

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, list)
        assert len(res) == 1
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

    shot_vectors = [None, [10, 1000], [1, 10, 10, 1000], [1, (10, 2), 1000]]

    @pytest.mark.parametrize("wires", wires)
    @pytest.mark.parametrize("shot_vector", shot_vectors)
    def test_list_multiple_expval(self, wires, device, shot_vector):
        """Return a comprehension list of multiple expvals."""
        dev = qml.device(device, wires=wires, shots=shot_vector)
        func = qubit_ansatz
        obs = qml.PauliZ

        def circuit(x):
            func(x)
            return [qml.expval(obs(wires=i)) for i in range(0, wires)]

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        if shot_vector is None:
            assert isinstance(res, list)
            assert len(res) == wires
            for r in res:
                assert isinstance(r, np.ndarray)
                assert r.shape == ()

        else:
            for r in res:
                assert isinstance(r, list)
                assert len(r) == wires

                for t in r:
                    assert isinstance(t, np.ndarray)
                    assert t.shape == ()

    def test_array_multiple(self, device):
        """Return PennyLane array of multiple measurements"""
        dev = qml.device(device, wires=2)
        func = qubit_ansatz
        obs = qml.PauliZ(1)

        def circuit(x):
            func(x)
            return qml.numpy.array([qml.expval(obs), qml.probs(wires=[0, 1])])

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, qml.numpy.ndarray)
        assert res[0].shape == ()
        assert res[1].shape == (4,)

    @pytest.mark.parametrize("comp_basis_sampling", [qml.sample(), qml.counts()])
    def test_sample_counts_no_obs(self, device, comp_basis_sampling):
        """Measuring qml.sample()/qml.counts() works with other measurements even with the same wire being measured."""

        shot_num = 1000
        num_wires = 2
        dev = qml.device(device, wires=num_wires, shots=shot_num)
        func = qubit_ansatz
        obs = qml.PauliZ(1)

        def circuit(x):
            func(x)
            return qml.apply(comp_basis_sampling), qml.expval(obs), qml.probs(wires=[0])

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)

        if comp_basis_sampling.return_type == qml.measurements.Sample:
            assert res[0].shape == (shot_num, num_wires)
        else:
            assert isinstance(res[0], dict)

        assert isinstance(res[1], qml.numpy.ndarray)
        assert res[1].shape == ()
        assert isinstance(res[2], qml.numpy.ndarray)
        assert res[2].shape == (2,)
