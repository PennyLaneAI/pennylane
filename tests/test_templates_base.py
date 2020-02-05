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
from pennylane.templates import template
from pennylane.templates.base import Single
from pennylane.ops import RX, RY, RZ, Displacement, T, S, Rot

dev_4_qubits = qml.device('default.qubit', wires=4)
dev_4_qumodes = qml.device('default.gaussian', wires=4)


@template
def ConstantTemplate(wires):
    T(wires=wires)
    S(wires=wires)


@template
def ParametrizedTemplate(par1, par2, wires):
    RX(par1, wires=wires)
    RY(par2, wires=wires)


@template
def KwargTemplate(par, wires, a=True):
    if a:
        T(wires=wires)
    RY(par, wires=wires)


class TestBaseSingle:
    """ Tests the Single base template."""

    PARAMS_UNITARY_TARGET_DEVICE_OBS = [([pi, pi, pi / 2, 0], RX, [-1, -1, 0, 1], dev_4_qubits, qml.PauliZ),
                                        ([pi, pi, pi / 2, 0], RY, [-1, -1, 0, 1], dev_4_qubits, qml.PauliZ),
                                        ([pi / 2, pi / 2, pi / 4, 0], RZ, [1, 1, 1, 1], dev_4_qubits, qml.PauliZ),
                                        ([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]], Displacement,
                                         [0.2, 0.4, 0.6, 0.8], dev_4_qumodes, qml.X)]

    GATE_PARAMETERS = [(RX, [[0.1], [0.2]]),
                       (Rot, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]),
                       (T, [[], []])]

    @pytest.mark.parametrize("unitary, parameters", GATE_PARAMETERS)
    def test_single_correct_queue_for_gate_unitary(self, unitary, parameters):
        """Tests that correct gate queue is created when 'unitary' is a single gate."""

        with qml.utils.OperationRecorder() as rec:
            Single(unitary=unitary, wires=range(2), parameters=parameters)

        for gate in rec.queue:
            assert isinstance(gate, unitary)

    @pytest.mark.parametrize("unitary, gates, parameters", [(ParametrizedTemplate, [RX, RY], [[0.1, 1], [0.2, 1]]),
                                                            (ConstantTemplate, [T, S], [[], []])])
    def test_single_correct_queue_for_template_unitary(self, unitary, gates, parameters):
        """Tests that correct gate queue is created when 'unitary' is a template."""

        with qml.utils.OperationRecorder() as rec:
            Single(unitary=unitary, wires=range(2), parameters=parameters)

        for idx, gate in enumerate(rec.queue):
            i = idx % 2
            assert isinstance(gate, gates[i])

    @pytest.mark.parametrize("kwarg, target_queue", [(True, [T, RY, T, RY]),
                                                     (False, [RY, RY])])
    def test_single_correct_queue_for_template_unitary_with_keyword(self, kwarg, target_queue):
        """Tests that correct gate queue is created when 'unitary' is a template that uses a keyword."""

        with qml.utils.OperationRecorder() as rec:
            Single(unitary=KwargTemplate, wires=range(2), parameters=[[1], [2]], kwargs={'a': kwarg})

        for gate, target_gate in zip(rec.queue, target_queue):
            assert isinstance(gate, target_gate)

    @pytest.mark.parametrize("gate, parameters", GATE_PARAMETERS)
    def test_single_correct_parameters_in_queue(self, gate, parameters):
        """Tests that gate queue has correct parameters."""

        with qml.utils.OperationRecorder() as rec:
            Single(unitary=gate, wires=range(2), parameters=parameters)

        for target_par, g in zip(parameters, rec.queue):
            assert g.parameters == target_par

    @pytest.mark.parametrize("pars1, pars2, gate", [([[], []], None, T),
                                                    ([1, 2], [[1], [2]], RX)])
    def test_single_correct_queue_for_gate_unitary(self, pars1, pars2, gate):
        """Tests that specific parameter inputs have the same output."""

        with qml.utils.OperationRecorder() as rec1:
            Single(unitary=gate, wires=range(2), parameters=pars1)

        with qml.utils.OperationRecorder() as rec2:
            Single(unitary=gate, wires=range(2), parameters=pars2)

        for g1, g2 in zip(rec1.queue, rec2.queue):
            assert g1.parameters == g2.parameters

    @pytest.mark.parametrize("parameters, unitary, target, dev, observable", PARAMS_UNITARY_TARGET_DEVICE_OBS)
    def test_single_prepares_state(self, parameters, unitary, target, dev, observable):
        """Tests the state produced by different unitaries."""

        @qml.qnode(dev)
        def circuit():
            Single(unitary=unitary, wires=range(4), parameters=parameters)
            return [qml.expval(observable(wires=w)) for w in range(4)]

        res = circuit()
        assert np.allclose(res, target)

    @pytest.mark.parametrize("parameters, n_wires", [(np.array([0]), 2),
                                                     ([0, 0, 0, 1, 0], 3)])
    def test_single_throws_error_when_mismatch_params_wires(self, parameters, n_wires):
        """Tests that error thrown when 'parameters' does not contain one set
           of parameters for each wire."""

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            Single(unitary=RX, wires=range(n_wires), parameters=parameters)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must contain one entry for each"):
            circuit()

    def test_single_exception_wires_not_valid(self):
        """Tests that an exception is raised if 'wires' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            Single(unitary=RX, wires='a', parameters=[[1], [2]])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit()

    def test_single_exception_parameters_not_valid(self):
        """Tests that an exception is raised if 'parameters' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            Single(unitary=RX, wires=[0, 1], parameters=RX)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must be either of type None or "):
            circuit()