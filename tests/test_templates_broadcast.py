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
Unit tests for the :func:`pennylane.template.broadcast` function.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from math import pi
import numpy as np
import pennylane as qml
from pennylane.templates import template, broadcast
from pennylane.ops import RX, RY, T, S, Rot, CRX, CRot, CNOT
from pennylane.templates.broadcast import wires_pyramid, wires_all_to_all, wires_ring


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


@template
def ConstantTemplateDouble(wires):
    T(wires=wires[0])
    CNOT(wires=wires)


@template
def ParametrizedTemplateDouble(par1, par2, wires):
    CRX(par1, wires=wires)
    RY(par2, wires=wires[0])


@template
def KwargTemplateDouble(par, wires, a=True):
    if a:
        T(wires=wires[0])
    CRX(par, wires=wires)


TARGET_OUTPUTS = [("single", 4, [pi, pi, pi / 2, 0], RX, [1, 1, 0, -1]),
                  ("double", 4, [pi / 2, pi / 2], CRX, [-1, 0, -1, 0]),
                  ("double", 4, None, CNOT, [-1, 1, -1, 1]),
                  ("double_odd", 4, [pi / 2], CRX, [-1, -1, 0, -1]),
                  ("chain", 4, [pi, pi, pi / 2], CRX, [-1, 1, -1, 0]),
                  ("ring", 4, [pi, pi, pi / 2, pi], CRX, [0, 1, -1, 0]),
                  ("pyramid", 4, [0, pi, pi / 2], CRX, [-1, -1, 0, 1]),
                  ("all_to_all", 4, [pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, pi / 2], CRX, [-1, 0, 1 / 2, 3 / 4])
                  ]

GATE_PARAMETERS = [("single", 0, T, []),
                   ("single", 1, T, [[]]),
                   ("single", 2, T, [[], []]),
                   ("single", 3, T, [[], [], []]),
                   ("single", 3, RX, [[0.1], [0.2], [0.3]]),
                   ("single", 3, Rot, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.3, 0.2, -0.1]]),
                   ("double", 0, CNOT, []),
                   ("double", 1, CNOT, []),
                   ("double", 3, CNOT, [[]]),
                   ("double", 2, CNOT, [[]]),
                   ("double", 3, CRX, [[0.1]]),
                   ("double", 3, CRot, [[0.1, 0.2, 0.3]]),
                   ("double_odd", 0, CNOT, []),
                   ("double_odd", 1, CNOT, []),
                   ("double_odd", 2, CNOT, []),
                   ("double_odd", 3, CNOT, [[]]),
                   ("double_odd", 3, CRX, [[0.1]]),
                   ("double_odd", 3, CRot, [[0.3, 0.2, 0.1]]),
                   ("chain", 0, CNOT, []),
                   ("chain", 1, CNOT, []),
                   ("chain", 2, CNOT, [[]]),
                   ("chain", 3, CNOT, [[], []]),
                   ("chain", 3, CRX, [[0.1], [0.1]]),
                   ("chain", 3, CRot, [[0.3, 0.2, 0.1], [0.3, 0.2, 0.1]]),
                   ("ring", 0, CNOT, []),
                   ("ring", 1, CNOT, []),
                   ("ring", 2, CNOT, [[]]),
                   ("ring", 3, CNOT, [[], [], []]),
                   ("ring", 3, CRX, [[0.1], [0.1], [0.1]]),
                   ("ring", 3, CRot, [[0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1]]),
                   ("pyramid", 0, CNOT, []),
                   ("pyramid", 1, CNOT, []),
                   ("pyramid", 2, CNOT, [[]]),
                   ("pyramid", 4, CNOT, [[], [], []]),
                   ("pyramid", 3, CRX, [[0.1]]),
                   ("pyramid", 4, CRX, [[0.1], [0.1], [0.1]]),
                   ("pyramid", 4, CRot, [[0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1]]),
                   ("all_to_all", 0, CNOT, []),
                   ("all_to_all", 1, CNOT, []),
                   ("all_to_all", 2, CNOT, [[]]),
                   ("all_to_all", 4, CNOT, [[], [], [], [], [], []]),
                   ("all_to_all", 3, CRX, [[0.1], [0.1], [0.1]]),
                   ("all_to_all", 4, CRX, [[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]]),
                   ("all_to_all", 4, CRot, [[0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1],
                                            [0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1]]),
                   ]


class TestConstructorBroadcast:
    """Tests the broadcast template constructor."""

    @pytest.mark.parametrize("pattern, unitary, parameters", [("single", RX, [[0.1], [0.2], [0.3]]),
                                                              ("single", Rot,
                                                               [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1],
                                                                [0.3, 0.2, -0.1]]),
                                                              ("single", T, [[], [], []]),
                                                              ])
    def test_correct_queue_for_gate_unitary(self, pattern, unitary, parameters):
        """Tests that correct gate queue is created when 'block' is a single gate."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=unitary, pattern=pattern, wires=range(3), parameters=parameters)

        for gate in rec.queue:
            assert isinstance(gate, unitary)

    @pytest.mark.parametrize("unitary, gates, parameters",
                             [(ParametrizedTemplate, [RX, RY], [[0.1, 1], [0.2, 1], [0.1, 1]]),
                              (ConstantTemplate, [T, S], [[], [], []]),
                              ])
    def test_correct_queue_for_template_block(self, unitary, gates, parameters):
        """Tests that correct gate queue is created when 'block' is a template."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=unitary, pattern="single", wires=range(3), parameters=parameters)

        first_gate = gates[0]
        second_gate = gates[1]
        for idx, gate in enumerate(rec.queue):
            if idx % 2 == 0:
                assert isinstance(gate, first_gate)
            else:
                assert isinstance(gate, second_gate)

    @pytest.mark.parametrize("template, kwarg, target_queue, parameters",
                             [(KwargTemplate, True, [T, RY, T, RY], [[1], [2]]),
                              (KwargTemplate, False, [RY, RY], [[1], [2]]),
                              ])
    def test_correct_queue_for_template_block_with_keyword(self, template, kwarg, target_queue, parameters):
        """Tests that correct gate queue is created when 'block' is a template that uses a keyword."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=template, pattern="single", wires=range(2),
                      parameters=parameters, kwargs={'a': kwarg})

        for gate, target_gate in zip(rec.queue, target_queue):
            assert isinstance(gate, target_gate)

    @pytest.mark.parametrize("pattern, pars1, pars2, gate", [("single", [[], [], []], None, T),
                                                             ("single", [1, 2, 3], [[1], [2], [3]], RX),
                                                             ])
    def test_correct_queue_same_gate_block_different_parameter_formats(self, pattern, pars1, pars2, gate):
        """Tests that specific parameter inputs have the same output."""

        with qml.utils.OperationRecorder() as rec1:
            broadcast(block=gate, pattern=pattern, wires=range(3), parameters=pars1)

        with qml.utils.OperationRecorder() as rec2:
            broadcast(block=gate, pattern=pattern, wires=range(3), parameters=pars2)

        for g1, g2 in zip(rec1.queue, rec2.queue):
            assert g1.parameters == g2.parameters

    @pytest.mark.parametrize("pattern, n_wires, gate, parameters", GATE_PARAMETERS)
    def test_correct_parameters_in_queue(self, pattern, n_wires, gate, parameters):
        """Tests that gate queue has correct parameters."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=gate, pattern=pattern, wires=range(n_wires), parameters=parameters)

        for target_par, g in zip(parameters, rec.queue):
            assert g.parameters == target_par

    @pytest.mark.parametrize("pattern, n_wires, parameters, unitary, target", TARGET_OUTPUTS)
    def test_prepares_correct_state(self, pattern, n_wires, parameters, unitary, target):
        """Tests the state produced by different unitaries."""

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            for w in range(4):
                qml.PauliX(wires=w)
            broadcast(block=unitary, pattern=pattern, wires=range(4), parameters=parameters)
            return [qml.expval(qml.PauliZ(wires=w)) for w in range(4)]

        res = circuit()
        assert np.allclose(res, target)

    @pytest.mark.parametrize("parameters, n_wires", [(np.array([0]), 2),
                                                     ([0, 0, 0, 1, 0], 3)])
    def test_throws_error_when_mismatch_params_wires(self, parameters, n_wires):
        """Tests that error thrown when 'parameters' does not contain one set
           of parameters for each wire."""

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            broadcast(block=RX, wires=range(n_wires), pattern="single", parameters=parameters)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must contain entries for"):
            circuit()

    def test_throws_special_error_for_ring_pattern_2_wires(self):
        """Tests that the special error is thrown when 'parameters' does not contain one set
           of parameters for a two-wire ring pattern."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(pars):
            broadcast(block=RX, wires=range(2), pattern="ring", parameters=pars)
            return qml.expval(qml.PauliZ(0))

        pars = [[1.6], [2.1]]

        with pytest.raises(ValueError, match="the ring pattern with 2 wires is an exception"):
            circuit(pars)

    def test_exception_wires_not_valid(self):
        """Tests that an exception is raised if 'wires' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            broadcast(block=RX, wires='a', pattern="single", parameters=[[1], [2]])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit()

    def test_exception_parameters_not_valid(self):
        """Tests that an exception is raised if 'parameters' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            broadcast(block=RX, wires=[0, 1], pattern="single", parameters=RX)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must be either of type None or "):
            circuit()

    @pytest.mark.parametrize("function, wires, target", [(wires_pyramid, [8, 2, 0, 4, 6, 2],
                                                          [[8, 2], [0, 4], [6, 2], [2, 0], [4, 6], [0, 4]]),
                                                         (wires_pyramid, [5, 10, 1, 0, 3, 4, 6],
                                                          [[5, 10], [1, 0], [3, 4], [10, 1], [0, 3], [1, 0]]),
                                                         (wires_pyramid, [0], []),
                                                         (wires_ring, [8, 2, 0, 4, 6, 2],
                                                          [[8, 2], [2, 0], [0, 4], [4, 6], [6, 2], [2, 8]]),
                                                         (wires_ring, [0], []),
                                                         (wires_ring, [4, 2], [[4, 2]]),
                                                         (wires_all_to_all, [8, 2, 0, 4],
                                                          [[8, 2], [8, 0], [8, 4], [2, 0], [2, 4], [0, 4]]),
                                                         (wires_all_to_all, [0], []),
                                                         ])
    def test_wire_sequence_generating_functions(self, function, wires, target):
        """Tests that the wire list generating functions for different patterns create the correct sequence."""

        sequence = function(wires)
        assert sequence == target
