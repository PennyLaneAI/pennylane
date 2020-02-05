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
Unit tests for the :mod:`pennylane.template.base` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from math import pi
import numpy as np
import pennylane as qml
from pennylane.templates.base import Single
from pennylane.ops import RX, RY, RZ, Squeezing, Displacement, T, Rot


class TestBaseSingle:
    """ Tests the Single base template."""

    dev_4_qubits = qml.device('default.qubit', wires=4)
    dev_4_qumodes = qml.device('default.gaussian', wires=4)

    PARAMS_UNITARY_OUTPUT_DEVICE = [([pi / 2, pi / 2, pi / 4, 0], RX, [1, -1, 0, 1, 1], dev_4_qubits),
                                    ([pi / 2, pi / 2, pi / 4, 0], RY, [-1, -1, 0, 1, 1], dev_4_qubits),
                                    ([pi / 2, pi / 2, pi / 4, 0], RZ, [-1, 1, 1, 1, 1], dev_4_qubits),
                                    ([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], Displacement,
                                     [0.01, 0.01, 0.01, 0.01], dev_4_qumodes),
                                    ([[1.2, 0.1], [1.2, 0.1], [1.2, 0.1], [1.2, 0.1]], Squeezing,
                                     [2.2784, 2.2784, 2.2784, 2.2784], dev_4_qumodes)]

    MISMATCH_PARAMS_WIRES = [(np.array([0]), 2),
                             ([0, 0, 0, 1, 0], 3)]

    GATE_UNITARY_PARAMETERS = [(RX, [0.1, 0.2]),
                               (Rot, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]),
                               (T, [[], []]),
                               # (T, None),
                               # (T, [])
                               ]

    # @pytest.mark.parametrize("unitary, parameters", GATE_UNITARY_PARAMETERS)
    # def test_single_correct_queue_for_gate_unitary(self, unitary, parameters):
    #     """Checks that correct gate queue is created when 'unitary' is a single gate."""
    #
    #     with qml.utils.OperationRecorder() as rec:
    #         Single(unitary=unitary, wires=range(2), parameters=parameters)
    #
    #     for gate, p in zip(rec.queue, parameters):
    #         assert isinstance(gate, unitary)
    #         assert gate.parameters == p

    @pytest.mark.parametrize("parameters, unitary, output, dev", PARAMS_UNITARY_OUTPUT_DEVICE)
    def test_single_prepares_state(self, parameters, unitary, output, dev):
        """Checks the state produced by different unitaries."""

        @qml.qnode(dev)
        def circuit():
            Single(unitary=unitary, wires=range(4), parameters=parameters)
            return [qml.expval(qml.Identity(wires=w)) for w in range(4)]

        res = circuit()
        assert np.allclose(res, output)

    @pytest.mark.parametrize("parameters, n_wires", MISMATCH_PARAMS_WIRES)
    def test_single_throws_error_when_mismatch_params_wires(self, parameters, n_wires):
        """Checks that error thrown when 'parameters' does not contain one set
           of parameters for each wire."""

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            Single(unitary=RX, wires=range(n_wires), parameters=parameters)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must contain one entry for each"):
            circuit()
