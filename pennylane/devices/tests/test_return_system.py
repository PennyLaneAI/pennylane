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
"""Tests that a device gives the correct output for multiple measurement."""
# pylint: disable=no-self-use
import pytest

import pennylane as qp
from pennylane import numpy as np  # Import from PennyLane to mirror the standard approach in demos

pytestmark = pytest.mark.skip_unsupported

wires = [2, 3, 4]


def qubit_ansatz(x):
    """Qfunc ansatz"""
    qp.Hadamard(wires=[0])
    qp.CRX(x, wires=[0, 1])


class TestIntegrationMultipleReturns:
    """Test the new return types for multiple measurements, it should always return a tuple containing the single
    measurements.
    """

    def test_multiple_expval(self, device):
        """Return multiple expvals."""
        n_wires = 2
        dev = device(n_wires)

        if hasattr(dev, "observables") and "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        obs1 = qp.Projector([0], wires=0)
        obs2 = qp.Z(1)
        func = qubit_ansatz

        def circuit(x):
            func(x)
            return qp.expval(obs1), qp.expval(obs2)

        qnode = qp.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], (float, np.ndarray))

        assert isinstance(res[1], (float, np.ndarray))

    def test_multiple_var(self, device):
        """Return multiple vars."""

        n_wires = 2
        dev = device(n_wires)

        if hasattr(dev, "observables") and "Projector" not in dev.observables:
            pytest.skip("Skipped because device does not support the Projector observable.")

        obs1 = qp.Projector([0], wires=0)
        obs2 = qp.Z(1)
        func = qubit_ansatz

        def circuit(x):
            func(x)
            return qp.var(obs1), qp.var(obs2)

        qnode = qp.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], (float, np.ndarray))

        assert isinstance(res[1], (float, np.ndarray))

    def test_multiple_prob(self, device):
        """Return multiple probs."""

        n_wires = 2
        dev = device(n_wires)

        def circuit(x):
            qubit_ansatz(x)
            return qp.probs(op=qp.Z(0)), qp.probs(op=qp.Y(1))

        qnode = qp.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2**1,)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (2**1,)

    def test_mix_meas(self, device):
        """Return multiple different measurements."""
        n_wires = 2
        dev = device(n_wires)

        def circuit(x):
            qubit_ansatz(x)
            return (
                qp.probs(wires=0),
                qp.expval(qp.Z(0)),
                qp.probs(op=qp.Y(1)),
                qp.expval(qp.Y(1)),
            )

        qnode = qp.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        assert isinstance(res, tuple)
        assert len(res) == 4

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2**1,)

        assert isinstance(res[1], (float, np.ndarray))

        assert isinstance(res[2], np.ndarray)
        assert res[2].shape == (2**1,)

        assert isinstance(res[3], (float, np.ndarray))
