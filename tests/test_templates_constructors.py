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
Unit tests for the :mod:`pennylane.template.constructors` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from math import pi
import numpy as np
import pennylane as qml
from pennylane.templates import template
from pennylane.templates.constructors import Broadcast, broadcast_double
from pennylane.ops import RX, RY, RZ, Displacement, Beamsplitter, T, S, Rot, CRX, CRot, CNOT

dev_4_qubits = qml.device('default.qubit', wires=4)
dev_4_qumodes = qml.device('default.gaussian', wires=4)


# templates for TestConstructorBroadcast

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


class TestConstructorBroadcast:
    """ Tests the broadcast template constructor."""

    PARAMS_UNITARY_TARGET_DEVICE_OBS = [([pi, pi, pi / 2, 0], RX, [-1, -1, 0, 1], dev_4_qubits, qml.PauliZ),
                                        ([pi, pi, pi / 2, 0], RY, [-1, -1, 0, 1], dev_4_qubits, qml.PauliZ),
                                        ([pi / 2, pi / 2, pi / 4, 0], RZ, [1, 1, 1, 1], dev_4_qubits, qml.PauliZ),
                                        ([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]], Displacement,
                                         [0.2, 0.4, 0.6, 0.8], dev_4_qumodes, qml.X)]

    GATE_PARAMETERS = [(RX, [[0.1], [0.2]]),
                       (Rot, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]),
                       (T, [[], []])]

    @pytest.mark.parametrize("unitary, parameters", GATE_PARAMETERS)
    def test_broadcast_correct_queue_for_gate_unitary(self, unitary, parameters):
        """Tests that correct gate queue is created when 'block' is a single gate."""

        with qml.utils.OperationRecorder() as rec:
            Broadcast(block=unitary, wires=range(2), parameters=parameters)

        for gate in rec.queue:
            assert isinstance(gate, unitary)

    @pytest.mark.parametrize("unitary, gates, parameters", [(ParametrizedTemplate, [RX, RY], [[0.1, 1], [0.2, 1]]),
                                                            (ConstantTemplate, [T, S], [[], []])])
    def test_broadcast_correct_queue_for_template_unitary(self, unitary, gates, parameters):
        """Tests that correct gate queue is created when 'block' is a template."""

        with qml.utils.OperationRecorder() as rec:
            Broadcast(block=unitary, wires=range(2), parameters=parameters)

        for idx, gate in enumerate(rec.queue):
            i = idx % 2
            assert isinstance(gate, gates[i])

    @pytest.mark.parametrize("kwarg, target_queue", [(True, [T, RY, T, RY]),
                                                     (False, [RY, RY])])
    def test_broadcast_correct_queue_for_template_unitary_with_keyword(self, kwarg, target_queue):
        """Tests that correct gate queue is created when 'block' is a template that uses a keyword."""

        with qml.utils.OperationRecorder() as rec:
            Broadcast(block=KwargTemplate, wires=range(2), parameters=[[1], [2]], kwargs={'a': kwarg})

        for gate, target_gate in zip(rec.queue, target_queue):
            assert isinstance(gate, target_gate)

    @pytest.mark.parametrize("gate, parameters", GATE_PARAMETERS)
    def test_broadcast_correct_parameters_in_queue(self, gate, parameters):
        """Tests that gate queue has correct parameters."""

        with qml.utils.OperationRecorder() as rec:
            Broadcast(block=gate, wires=range(2), parameters=parameters)

        for target_par, g in zip(parameters, rec.queue):
            assert g.parameters == target_par

    @pytest.mark.parametrize("pars1, pars2, gate", [([[], []], None, T),
                                                    ([1, 2], [[1], [2]], RX)])
    def test_broadcast_correct_queue_for_gate_unitary(self, pars1, pars2, gate):
        """Tests that specific parameter inputs have the same output."""

        with qml.utils.OperationRecorder() as rec1:
            Broadcast(block=gate, wires=range(2), parameters=pars1)

        with qml.utils.OperationRecorder() as rec2:
            Broadcast(block=gate, wires=range(2), parameters=pars2)

        for g1, g2 in zip(rec1.queue, rec2.queue):
            assert g1.parameters == g2.parameters

    @pytest.mark.parametrize("parameters, unitary, target, dev, observable", PARAMS_UNITARY_TARGET_DEVICE_OBS)
    def test_broadcast_prepares_state(self, parameters, unitary, target, dev, observable):
        """Tests the state produced by different unitaries."""

        @qml.qnode(dev)
        def circuit():
            Broadcast(block=unitary, wires=range(4), parameters=parameters)
            return [qml.expval(observable(wires=w)) for w in range(4)]

        res = circuit()
        assert np.allclose(res, target)

    @pytest.mark.parametrize("parameters, n_wires", [(np.array([0]), 2),
                                                     ([0, 0, 0, 1, 0], 3)])
    def test_broadcast_throws_error_when_mismatch_params_wires(self, parameters, n_wires):
        """Tests that error thrown when 'parameters' does not contain one set
           of parameters for each wire."""

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            Broadcast(block=RX, wires=range(n_wires), parameters=parameters)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must contain one entry for each"):
            circuit()

    def test_broadcast_exception_wires_not_valid(self):
        """Tests that an exception is raised if 'wires' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            Broadcast(block=RX, wires='a', parameters=[[1], [2]])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit()

    def test_broadcast_exception_parameters_not_valid(self):
        """Tests that an exception is raised if 'parameters' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            Broadcast(block=RX, wires=[0, 1], parameters=RX)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must be either of type None or "):
            circuit()


# Templates for TestConstructorBroadcastDouble

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


class TestConstructorBroadcastDouble:
    """ Tests the broadcast_double template constructor."""

    EVEN_GATE_PARAMETERS = [(CRX, [[0.1], [0.2]]),
                            (CRot, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]),
                            (CNOT, [[], []])]

    ODD_GATE_PARAMETERS = [(CRX, [[0.1]]),
                           (CRot, [[0.3, 0.2, 0.1]]),
                           (CNOT, [[]])]

    @pytest.mark.parametrize("unitary, parameters", EVEN_GATE_PARAMETERS)
    def test_broadcast_double_correct_queue_for_gate_unitary(self, unitary, parameters):
        """Tests that correct gate queue is created when 'block' is a single gate and the template
        is used with even=True."""

        with qml.utils.OperationRecorder() as rec:
            broadcast_double(block=unitary, wires=range(4), even=True, parameters=parameters)

        # check that correct number of blocks applied
        assert len(rec.queue) == 2

        # check that the queue only contains the gate
        for gate in rec.queue:
            assert isinstance(gate, unitary)

    @pytest.mark.parametrize("unitary, parameters", ODD_GATE_PARAMETERS)
    def test_broadcast_double_correct_queue_for_gate_unitary(self, unitary, parameters):
        """Tests that correct gate queue is created when 'block' is a single gate and the template
        is used with even=False."""

        with qml.utils.OperationRecorder() as rec:
            broadcast_double(block=unitary, wires=range(4), even=False, parameters=parameters)

        # check that correct number of blocks applied
        assert len(rec.queue) == 1

        # check that the queue only contains the gate
        for gate in rec.queue:
            assert isinstance(gate, unitary)

    @pytest.mark.parametrize("unitary, gates, parameters", [(ParametrizedTemplateDouble, [CRX, RY],
                                                             [[0.1, 1], [0.2, 1]]),
                                                            (ConstantTemplateDouble, [T, CNOT], [[], []])])
    def test_broadcast_double_correct_queue_for_template_unitary(self, unitary, gates, parameters):
        """Tests that correct gate queue is created when 'block' is a template."""

        with qml.utils.OperationRecorder() as rec:
            broadcast_double(block=unitary, wires=range(4), even=True, parameters=parameters)

        for idx, gate in enumerate(rec.queue):
            i = idx % 2
            assert isinstance(gate, gates[i])

    @pytest.mark.parametrize("kwarg, target_queue", [(True, [T, CRX, T, CRX]),
                                                     (False, [CRX, CRX])])
    def test_broadcast_double_correct_queue_for_template_unitary_with_keyword(self, kwarg, target_queue):
        """Tests that correct gate queue is created when 'block' is a template that uses a keyword."""

        with qml.utils.OperationRecorder() as rec:
            broadcast_double(block=KwargTemplateDouble, wires=range(4), even=True,
                             parameters=[[1], [2]], kwargs={'a': kwarg})

        for gate, target_gate in zip(rec.queue, target_queue):
            assert isinstance(gate, target_gate)

    @pytest.mark.parametrize("gate, parameters", EVEN_GATE_PARAMETERS)
    def test_broadcast_double_correct_parameters_in_queue_even(self, gate, parameters):
        """Tests that gate queue has correct parameters for even implementation."""

        with qml.utils.OperationRecorder() as rec:
            broadcast_double(block=gate, wires=range(4), even=True, parameters=parameters)

        for target_par, g in zip(parameters, rec.queue):
            assert g.parameters == target_par

    @pytest.mark.parametrize("gate, parameters", ODD_GATE_PARAMETERS)
    def test_broadcast_double_correct_parameters_in_queue_odd(self, gate, parameters):
        """Tests that gate queue has correct parameters for odd implementation."""

        with qml.utils.OperationRecorder() as rec:
            broadcast_double(block=gate, wires=range(4), even=False, parameters=parameters)

        for target_par, g in zip(parameters, rec.queue):
            assert g.parameters == target_par

    @pytest.mark.parametrize("pars1, pars2, gate", [([[], []], None, CNOT),
                                                    ([1, 2], [[1], [2]], CRX)])
    def test_broadcast_double_correct_queue_for_different_parameter_formats(self, pars1, pars2, gate):
        """Tests that specific parameter formats have the same output."""

        with qml.utils.OperationRecorder() as rec1:
            broadcast_double(block=gate, wires=range(4), even=True, parameters=pars1)

        with qml.utils.OperationRecorder() as rec2:
            broadcast_double(block=gate, wires=range(4), even=True, parameters=pars2)

        for g1, g2 in zip(rec1.queue, rec2.queue):
            assert g1.parameters == g2.parameters

    @pytest.mark.parametrize("parameters, unitary, even, target", [([pi/2, pi/2], CRX, True, [-1, 0, -1, 0]),
                                                                   ([pi / 2], CRX, False, [-1, -1, 0, -1]),
                                                                   (None, CNOT, True, [-1, 1, -1, 1])])
    def test_broadcast_double_prepares_state(self, parameters, unitary, even, target):
        """Tests the state produced by different unitaries for a qubit device."""

        @qml.qnode(dev_4_qubits)
        def circuit():
            for w in range(4):
                qml.PauliX(wires=w)
            broadcast_double(block=unitary, wires=range(4), even=even, parameters=parameters)
            return [qml.expval(qml.PauliZ(wires=w)) for w in range(4)]

        res = circuit()
        assert np.allclose(res, target)

    @pytest.mark.parametrize("parameters, unitary, even, target", [([[pi/4, 0.0], [pi/4, 0.0]], Beamsplitter, True,
                                                                    [0, 2, 0, 2]),
                                                                   #([[pi / 4, 0.0]], Beamsplitter, False,
                                                                   # [0, 0, 2, 0])
                                                                   ])
    def test_broadcast_double_prepares_state_cv(self, parameters, unitary, even, target):
        """Tests the state produced by different unitaries for a cv device."""

        @qml.qnode(dev_4_qumodes)
        def circuit():
            for w in range(4):
                qml.Displacement(1, 0, wires=w)
            broadcast_double(block=unitary, wires=range(4), even=even, parameters=parameters)
            return [qml.expval(qml.NumberOperator(wires=w)) for w in range(4)]

        res = circuit()
        assert np.allclose(res, target)

    @pytest.mark.parametrize("parameters, even, n_wires", [([0, 0], True, 6),
                                                           ([0, 0, 0], False, 6),
                                                           ([0, 0, 0], True, 4)])
    def test_broadcast_double_throws_error_when_mismatch_params_wires(self, parameters, even, n_wires):
        """Tests that error thrown when 'parameters' does not contain one set
           of parameters for each wire."""

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            broadcast_double(block=CRX, wires=range(n_wires), even=even, parameters=parameters)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must contain one entry for each"):
            circuit()

    def test_broadcast_double_exception_wires_not_valid(self):
        """Tests that an exception is raised if 'wires' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            broadcast_double(block=RX, wires='a', parameters=[[1], [2]])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="wires must be a positive"):
            circuit()

    def test_broadcast_double_exception_parameters_not_valid(self):
        """Tests that an exception is raised if 'parameters' argument has invalid format."""

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            broadcast_double(block=RX, wires=[0, 1], parameters=RX)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match="'parameters' must be either of type None or "):
            circuit()
