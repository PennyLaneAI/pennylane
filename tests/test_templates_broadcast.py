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
from pennylane.ops import RX, RY, Displacement, Beamsplitter, T, S, Rot, CRX, CRot, CNOT

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


TARGET_OUTPUTS = [("single", [pi, pi, pi / 2, 0], RX, [1, 1, 0, -1]),
                  ("double", [pi / 2, pi / 2], CRX, [-1, 0, -1, 0]),
                  ("double", None, CNOT, [-1, 1, -1, 1]),
                  ("double_odd", [pi / 2], CRX, [-1, -1, 0, -1]),
                  ]

CV_TARGET_OUTPUTS = [("single", [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]], Displacement, [2.2, 2.4, 2.6, 2.8],
                      qml.X),
                     ("double", [[pi / 4, 0.0], [pi / 4, 0.0]], Beamsplitter, [0, 2, 0, 2],
                      qml.NumberOperator),
                     ]

GATE_PARAMETERS = [("single", RX, [[0.1], [0.2], [0.3]]),
                   ("single", Rot, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.3, 0.2, -0.1]]),
                   ("single", T, [[], [], []]),
                   ("double", CRX, [[0.1]]),
                   ("double", CRot, [[0.1, 0.2, 0.3]]),
                   ("double", CNOT, [[]]),
                   ("double_odd", CRX, [[0.1]]),
                   ("double_odd", CRot, [[0.3, 0.2, 0.1]]),
                   ("double_odd", CNOT, [[]]),
                   ]


class TestConstructorBroadcast:
    """Tests the broadcast template constructor."""

    @pytest.mark.parametrize("structure, unitary, parameters", GATE_PARAMETERS)
    def test_correct_queue_for_gate_unitary(self, structure, unitary, parameters):
        """Tests that correct gate queue is created when 'block' is a single gate."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=unitary, structure=structure, wires=range(3), parameters=parameters)

        for gate in rec.queue:
            assert isinstance(gate, unitary)

    @pytest.mark.parametrize("structure, unitary, gates, parameters",
                             [("single", ParametrizedTemplate, [RX, RY], [[0.1, 1], [0.2, 1], [0.1, 1]]),
                              ("single", ConstantTemplate, [T, S], [[], [], []]),
                              ("double", ParametrizedTemplateDouble, [CRX, RY], [[0.1, 1]]),
                              ("double", ConstantTemplateDouble, [T, CNOT], [[]]),
                              ])
    def test_correct_queue_for_template_block(self, structure, unitary, gates, parameters):
        """Tests that correct gate queue is created when 'block' is a template."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=unitary, structure=structure, wires=range(3), parameters=parameters)

        first_gate = gates[0]
        second_gate = gates[1]
        for idx, gate in enumerate(rec.queue):
            if idx % 2 == 0:
                assert isinstance(gate, first_gate)
            else:
                assert isinstance(gate, second_gate)

    @pytest.mark.parametrize("structure, template, kwarg, target_queue, parameters",
                             [("single", KwargTemplate, True, [T, RY, T, RY], [[1], [2]]),
                              ("single", KwargTemplate, False, [RY, RY], [[1], [2]]),
                              ("double", KwargTemplateDouble, True, [T, CRX], [[1]]),
                              ("double", KwargTemplateDouble, False, [CRX], [[1]])])
    def test_correct_queue_for_template_block_with_keyword(self, structure, template, kwarg, target_queue, parameters):
        """Tests that correct gate queue is created when 'block' is a template that uses a keyword."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=template, structure=structure, wires=range(2),
                      parameters=parameters, kwargs={'a': kwarg})

        for gate, target_gate in zip(rec.queue, target_queue):
            assert isinstance(gate, target_gate)

    @pytest.mark.parametrize("structure, pars1, pars2, gate", [("single", [[], [], []], None, T),
                                                               ("single", [1, 2, 3], [[1], [2], [3]], RX),
                                                               ])
    def test_correct_queue_for_gate_block(self, structure, pars1, pars2, gate):
        """Tests that specific parameter inputs have the same output."""

        with qml.utils.OperationRecorder() as rec1:
            broadcast(block=gate, structure=structure, wires=range(3), parameters=pars1)

        with qml.utils.OperationRecorder() as rec2:
            broadcast(block=gate, structure=structure, wires=range(3), parameters=pars2)

        for g1, g2 in zip(rec1.queue, rec2.queue):
            assert g1.parameters == g2.parameters

    @pytest.mark.parametrize("structure, gate, parameters", GATE_PARAMETERS)
    def test_correct_parameters_in_queue(self, structure, gate, parameters):
        """Tests that gate queue has correct parameters."""

        with qml.utils.OperationRecorder() as rec:
            broadcast(block=gate, structure=structure, wires=range(3), parameters=parameters)

        for target_par, g in zip(parameters, rec.queue):
            assert g.parameters == target_par

    @pytest.mark.parametrize("structure, pars1, pars2, gate", [("double", [[], []], None, CNOT),
                                                               ("double", [1, 2], [[1], [2]], CRX)])
    def test_double_correct_queue_for_different_parameter_formats(self, structure, pars1, pars2, gate):
        """Tests that specific parameter formats have the same output."""

        with qml.utils.OperationRecorder() as rec1:
            broadcast(block=gate, structure=structure, wires=range(4), parameters=pars1)

        with qml.utils.OperationRecorder() as rec2:
            broadcast(block=gate, structure=structure, wires=range(4), parameters=pars2)

        for g1, g2 in zip(rec1.queue, rec2.queue):
            assert g1.parameters == g2.parameters

    @pytest.mark.parametrize("structure, parameters, unitary, target", TARGET_OUTPUTS)
    def test_prepares_correct_state(self, structure, parameters, unitary, target):
        """Tests the state produced by different unitaries."""

        @qml.qnode(dev_4_qubits)
        def circuit():
            for w in range(4):
                qml.PauliX(wires=w)
            broadcast(block=unitary, structure=structure, wires=range(4), parameters=parameters)
            return [qml.expval(qml.PauliZ(wires=w)) for w in range(4)]

        res = circuit()
        assert np.allclose(res, target)

    @pytest.mark.parametrize("structure, parameters, unitary, target, observable", CV_TARGET_OUTPUTS)
    def test_prepares_correct_state_cv(self, structure, parameters, unitary, target, observable):
        """Tests the state produced by different unitaries."""

        @qml.qnode(dev_4_qumodes)
        def circuit():
            for w in range(4):
                Displacement(1, 0, wires=w)
            broadcast(block=unitary, structure=structure, wires=range(4), parameters=parameters)
            return [qml.expval(observable(wires=w)) for w in range(4)]

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
            broadcast(block=RX, wires=range(n_wires), structure="single", parameters=parameters)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must contain entries for"):
            circuit()

    def test_exception_wires_not_valid(self):
        """Tests that an exception is raised if 'wires' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            broadcast(block=RX, wires='a', structure="single", parameters=[[1], [2]])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit()

    def test_exception_parameters_not_valid(self):
        """Tests that an exception is raised if 'parameters' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            broadcast(block=RX, wires=[0, 1], parameters=RX)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must be either of type None or "):
            circuit()

    def test_exception_default_case_not_valid(self):
        """Tests that an exception is raised if 'structure=None' for a 'block' acting on more than
        two wires."""

        @template
        def ThreeWireTemplate(wires):
            qml.Hadamard(wires[0])
            qml.Hadamard(wires[1])
            qml.Hadamard(wires[2])

        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit():
            broadcast(block=ThreeWireTemplate, wires=range(3))
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="if block acts on more than 2 wires, a valid "):
            circuit()
