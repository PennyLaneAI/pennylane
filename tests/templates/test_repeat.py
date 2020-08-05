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
Unit tests for the :func:`pennylane.template.repeat` function.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from pennylane import repeat

@template
def ConstantCircuit(wires, **kwargs):

    qml.PauliX(wires=wires[0])
    qml.Hadamard(wires=wires[0])
    qml.PauliY(wires=wires[1])

@template
def BigConstantCircuit(wires, **kwargs):

    qml.CNOT(wires=[wires[3], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.PauliY(wires=wires[2])

@template
def ConstantKwargCircuit(wires, **kwargs):

    qml.CNOT(wires=[wires[3], wires[1]])
    qml.Hadamard(wires=wires[1])
    qml.PauliY(wires=wires[2])

    if kwargs['var'] == True:
        qml.Hadamard(wires=wires[0])

@template
def ParamCircuit(parameters, wires, **kwargs):

    for i, w in enumerate(wires):
        qml.RX(parameters[0][i], wires=w)

    qml.MultiRZ(parameters[1], wires=wires)

@template
def TemplateParamCircuit(parameters, wires, **kwargs):

    for i, w in enumerate(wires):
        qml.RY(parameters[0][i], wires=w)

    qml.templates.BasicEntanglerLayers([parameters[1]], wires=wires)

UNITARIES = [
    ConstantCircuit,
    BigConstantCircuit,
    ConstantKwargCircuit,
    ParamCircuit,
    TemplateParamCircuit
]


########################


class TestRepeat:
    """Tests the repetation function"""

    def test_depth_error(self):
        """Tests that the correct error is thrown when depth is not an integer"""

        depth = 1.5

        def unitary(wires, **kwargs):
            qml.PauliX(wires=wires)

        with pytest.raises(ValueError, match=r"'depth' must be of type int"):
            repeat(unitary, [0], depth)

    def test_parameter_error(self):
        """Tests that the correct error is thrown when parameters are not None or Iterable"""

        params = 1.0

        def unitary(parameters, wires, **kwargs):
            qml.RX(parameters, wires=wires)

        with pytest.raises(ValueError, match=r"'parameters' must be either of type None or Iterable"):
            repeat(unitary, [0], 2, parameters=params)

    def test_kwargs_error(self):
        """Tests that the correct error is thrown when kwargs is not a list"""

        kwargs = {'var': True}

        def unitary(wires, **kwargs):
            qml.PauliX(wires=wires)
            if kwargs['var'] == True:
                qml.Hadamard(wires=wires)

        with pytest.raises(ValueError, match=r"'kwargs' must be a list"):
            repeat(unitary, [0], 2, kwargs=kwargs)

    def test_kwarg_elements(self):
        """Tests that the correct error is thrown when the elements of 'kwargs' are not dictionaries"""

        kwargs = [('var', True), ('var', True)]

        def unitary(wires, **kwargs):
            qml.PauliX(wires=wires)
            if kwargs['var'] == True:
                qml.Hadamard(wires=wires)

        with pytest.raises(ValueError, match=r"Elements of 'kwargs' must be dictionaries"):
            repeat(unitary, [0], 2, kwargs=kwargs)

    def test_kwargs_length(self):
        """Tests that the correct error is thrown when the length of kwargs is incorrect"""

        kwargs = [{'var': True}]

        def unitary(wires, **kwargs):
            qml.PauliX(wires=wires)
            if kwargs['var'] == True:
                qml.Hadamard(wires=wires)

        with pytest.raises(ValueError, match=r"Expected length of 'kwargs' to be 2"):
            repeat(unitary, [0], 2, kwargs=kwargs)

    def test_dim(self):
        """Tests that the correct error is thrown when the dimension of the parameters in wrong"""

        params = [1, 1, 1]

        def unitary(parameters, wires, **kwargs):
            qml.RX(parameters, wires=wires)

        with pytest.raises(ValueError, match=r"Expected first dimension of 'parameters' to be 2"):
            repeat(unitary, [0], 2, parameters=params)


    WIRES = [range(i) for i in [2, 5, 4, 3, 3, 3]]
    DEPTH = [2, 1, 2, 2, 1, 1]
    PARAMS = [
        None,
        None,
        None,
        [[[0.5, 0.5, 0.5], 0.3], [[0.3, 0.3, 0.3], 0.5]],
        [[0.5, 0.4, 0.5], [0.4, 0.4, 0.4]]
    ]
    KWARGS = [None, None, [{'var': True}, {'var': False}], None, None]

    GATES = [
        [qml.PauliX(wires=0), qml.Hadamard(wires=0), qml.PauliY(wires=1), qml.PauliX(wires=0), qml.Hadamard(wires=0), qml.PauliY(wires=1)],
        [qml.CNOT(wires=[3, 1]), qml.PauliX(wires=0), qml.Hadamard(wires=1), qml.CZ(wires=[0, 1]), qml.PauliY(wires=2)],
        [qml.CNOT(wires=[3, 1]), qml.Hadamard(wires=1), qml.PauliY(wires=2), qml.Hadamard(wires=0), qml.CNOT(wires=[3, 1]), qml.Hadamard(wires=1), qml.PauliY(wires=2)],
        [qml.RX(0.5, wires=0), qml.RX(0.5, wires=1), qml.RX(0.5, wires=2), qml.CNOT(wires=[0, 2]), qml.RZ(0.3, wires=2) qml.CNOT(wires=[0, 2]),
         qml.RX(0.3, wires=0), qml.RX(0.3, wires=1), qml.RX(0.3, wires=2), qml.CNOT(wires=[0, 2]), qml.RZ(0.5, wires=2) qml.CNOT(wires=[0, 2])],
        [qml.RY(0.5, wires=0), qml.RY(0.4, wires=1), qml.RY(0.5, wires=2), qml.RX(0.4, wires=0), qml.RX(0.4, wires=1), qml.RX(0.4, wires=2),
         qml.CNOT(wires=[0, 1]), qml.CNOT(wires=[1, 2]), qml.CNOT(wires=[2, 0])],

    ]


    REPEAT = zip(UNITARIES, WIRES, DEPTH, PARAMS, KWARGS, GATES)

    @pytest.mark.parametrize(("unitary", "wires", "depth", "params", "kwargs", "gates"), REPEAT)
    def test_repeat(self, unitary, wires, depth, params, kwargs, gates):
        """Tests that the repetition function is yielding the correct sequence of gates"""

        with qml._queuing.OperationRecorder() as rec:
            repeat(unitary, wires, depth, parameters=params, kwargs=kwargs)

        for i, gate in enumerate(rec.operations):
            prep = [gate.name, gate.parameters, gate.wires]
            target = [gates[i].name, gates[i].parameters, gates[i].wires]

        assert prep == target
