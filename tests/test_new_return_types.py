# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the new return types.
"""
import pytest

import numpy as np
import pennylane as qml

wires = [2, 3, 4]

devices = ["default.qubit", "default.mixed"]


class TestSingleReturnExecute:
    """Test that single measurements return behavior does not change."""

    @pytest.mark.parametrize("wires", wires)
    def test_state_default(self, wires):
        """Return state with default.qubit."""
        dev = qml.device("default.qubit", wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (2**wires,)
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("wires", wires)
    def test_state_mixed(self, wires):
        """Return state with default.mixed."""
        dev = qml.device("default.mixed", wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (2**wires, 2**wires)
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("d_wires", wires)
    def test_density_matrix_default(self, d_wires, device):
        """Return density matrix."""
        dev = qml.device(device, wires=4)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=range(0, d_wires))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (2**d_wires, 2**d_wires)
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_expval(self, device):
        """Return a single expval."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_var(self, device):
        """Return a single var."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.var(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_vn_entropy(self, device):
        """Return a single vn entropy."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.vn_entropy(wires=0)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_mutual_info(self, device):
        """Return a single mutual information."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    herm = np.diag([1, 2, 3, 4])
    probs_data = [
        (None, [0]),
        (None, [0, 1]),
        (qml.PauliZ(0), None),
        (qml.Hermitian(herm, wires=[1, 0]), None),
    ]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("op,wires", probs_data)
    def test_probs(self, op, wires, device):
        """Return a single prob."""
        dev = qml.device(device, wires=3)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op, wires=wires)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        if wires is None:
            wires = op.wires

        assert res[0].shape == (2 ** len(wires),)
        assert isinstance(res[0], np.ndarray)

    # Samples and counts


wires = [([0], [1]), ([1], [0]), ([0], [0]), ([1], [1])]


class TestMultipleReturns:
    """Test the new return types for multiple measurements, it should always return a tuple containing the single
    measurements.
    """

    @pytest.mark.parametrize("device", devices)
    def test_multiple_expval(self, device):
        """Return multiple expvals."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == ()

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

    @pytest.mark.parametrize("device", devices)
    def test_multiple_var(self, device):
        """Return multiple vars."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.var(qml.PauliZ(wires=0)), qml.var(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == ()

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

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

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    def test_multiple_prob(self, op1, op2, wires1, wires2, device):
        """Return multiple probs."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op1, wires=wires1), qml.probs(op=op2, wires=wires2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == (2 ** len(wires1),)

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == (2 ** len(wires2),)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    @pytest.mark.parametrize("wires3, wires4", wires)
    def test_mix_meas(self, op1, wires1, op2, wires2, wires3, wires4, device):
        """Return multiple different measurements."""
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return (
                qml.probs(op=op1, wires=wires1),
                qml.vn_entropy(wires=wires3),
                qml.probs(op=op2, wires=wires2),
                qml.expval(qml.PauliZ(wires=wires4)),
            )

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 4

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == (2 ** len(wires1),)

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

        assert isinstance(res[0][2], np.ndarray)
        assert res[0][2].shape == (2 ** len(wires2),)

        assert isinstance(res[0][3], np.ndarray)
        assert res[0][3].shape == ()

    wires = [2, 3, 4, 5]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("wires", wires)
    def test_list_multiple_expval(self, wires, device):
        """Return a comprehension list of multiple expvals."""
        dev = qml.device(device, wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(0, wires)]

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == wires

        for i in range(0, wires):
            assert isinstance(res[0][i], np.ndarray)
            assert res[0][i].shape == ()
