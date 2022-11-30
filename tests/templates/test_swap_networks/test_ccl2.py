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
Tests for the TwoLocalSwapNetwork template.
"""

import pytest
from pennylane import numpy as np
import pennylane as qml


class TestDecomposition:
    """Test that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        ("wires", "acquaintances", "weights", "fermionic", "shift"),
        [
            (4, None, None, True, False),
            (5, lambda index, wires, param: (), None, True, False),
            (5, lambda index, wires, param: qml.CNOT(index), None, False, False),
            (6, lambda index, wires, param: qml.CRX(param, index), np.random.rand(18), True, False),
            (6, lambda index, wires, param: qml.CRY(param, index), np.random.rand(18), True, True),
        ],
    )
    def test_ccl2_operations(self, wires, acquaintances, weights, fermionic, shift):
        """Test the correctness of the TwoLocalSwapNetwork template including the
        gate count and order, the wires the operation acts on and the correct use
        of parameters in the circuit."""

        wire_order = range(wires)
        itrweights = iter([]) if weights is None else iter(weights)

        pass

        # number of gates

        # order of gates

        # gate parameter

        # gate wires

    def test_custom_wire_labels(self, tol=1e-8):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        acquaintances = lambda index, wires, param=None: qml.CNOT(index)
        weights = np.random.random(size=(10))

        dev = qml.device("default.qubit", wires=5)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "e", "y"])

        @qml.qnode(dev)
        def circuit():
            qml.templates.TwoLocalSwapNetwork(
                dev.wires, acquaintances, weights, fermionic=True, shift=False
            )
            return qml.expval(qml.Identity(0))

        @qml.qnode(dev2)
        def circuit2():
            qml.templates.TwoLocalSwapNetwork(
                dev2.wires, acquaintances, weights, fermionic=True, shift=False
            )
            return qml.expval(qml.Identity("z"))

        circuit()
        circuit2()

        assert np.allclose(dev.state, dev2.state, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        ("wires", "acquaintances", "weights", "fermionic", "shift", "exp_state"),
        [
            (
                4,
                None,
                None,
                True,
                False,
                qml.math.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                ),
            ),
        ],
    )
    def test_ccl2(self, wires, acquaintances, weights, fermionic, shift, exp_state):
        """Test that the TwoLocalSwapNetwork template with multiple layers works correctly asserting the prepared state."""

        wires = range(wires)
        pass


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        ("wires", "acquaintances", "weights", "fermionic", "shift", "msg_match"),
        [
            (
                1,
                None,
                None,
                True,
                False,
                "TwoLocalSwapNetwork requires at least 2 qubits",
            ),
            (
                6,
                qml.CNOT(wires=[0, 1]),
                np.random.rand(18),
                True,
                False,
                "Acquaintances must either be a callable or None",
            ),
            (
                6,
                lambda index, wires, param: qml.CRX(param, index),
                np.random.rand(12),
                True,
                False,
                "Weight tensor must be of size",
            ),
        ],
    )
    def test_ccl2_exceptions(self, wires, acquaintances, weights, fermionic, shift, msg_match):
        """Test that TwoLocalSwapNetwork throws an exception if the parameters have illegal
        shapes, types or values."""

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit():
            qml.templates.TwoLocalSwapNetwork(
                dev.wires, acquaintances, weights, fermionic=True, shift=False
            )
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(circuit, dev)

        with pytest.raises(ValueError, match=msg_match):
            circuit()

    def test_id(self):
        """Test that the id attribute can be set."""
        template = qml.templates.TwoLocalSwapNetwork(
            wires=range(4),
            acquaintances=None,
            weights=None,
            fermionic=True,
            shif=False,
            id="a",
        )
        assert template.id == "a"


class TestAttributes:
    """Test additional methods and attributes"""

    @pytest.mark.parametrize(
        "n_wires, expected_shape",
        [
            (2, (1,)),
            (4, (6,)),
            (5, (10,)),
            (6, (15,)),
        ],
    )
    def test_shape(self, n_wires, expected_shape):
        """Test that the shape method returns the correct shape of the weights tensor."""

        shape = qml.templates.TwoLocalSwapNetwork.shape(n_wires)
        assert shape == expected_shape

    def test_shape_exception_not_enough_qubits(self):
        """Test that the shape function warns if there are not enough qubits."""

        with pytest.raises(ValueError, match="TwoLocalSwapNetwork requires at least 2 qubits"):
            qml.templates.TwoLocalSwapNetwork.shape(n_wires=1)
