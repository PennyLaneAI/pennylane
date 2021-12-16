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
Unit tests for the :func:`pennylane.template.layer` function.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import pennylane as qml
from pennylane import layer


def ConstantCircuit():

    qml.PauliX(wires=[0])
    qml.Hadamard(wires=[0])
    qml.PauliY(wires=[1])


def StaticCircuit(wires, var):

    qml.CNOT(wires=[wires[3], wires[1]])
    qml.Hadamard(wires=wires[1])
    qml.PauliY(wires=wires[2])

    if var == True:
        qml.Hadamard(wires=wires[0])


def KwargCircuit(wires, **kwargs):

    qml.CNOT(wires=[wires[3], wires[1]])
    qml.Hadamard(wires=wires[1])
    qml.PauliY(wires=wires[2])

    if kwargs["var"] == True:
        qml.Hadamard(wires=wires[0])


def DynamicCircuit(parameters):

    for i in range(2):
        qml.RX(parameters[0][i], wires=i)

    qml.MultiRZ(parameters[1], wires=[0, 1])


def MultiCircuit(parameters1, parameters2, var1, wires, var2):

    if var2 == True:
        for i, w in enumerate(wires):
            qml.RY(parameters1[i], wires=w)

    if var1 == True:
        qml.templates.BasicEntanglerLayers([parameters2], wires=wires)


UNITARIES = [ConstantCircuit, StaticCircuit, KwargCircuit, DynamicCircuit, MultiCircuit]

DEPTH = [2, 1, 2, 1, 2]

GATES = [
    [
        qml.PauliX(wires=0),
        qml.Hadamard(wires=0),
        qml.PauliY(wires=1),
        qml.PauliX(wires=0),
        qml.Hadamard(wires=0),
        qml.PauliY(wires=1),
    ],
    [qml.CNOT(wires=[3, 1]), qml.Hadamard(wires=1), qml.PauliY(wires=2), qml.Hadamard(wires=0)],
    [
        qml.CNOT(wires=[3, 1]),
        qml.Hadamard(wires=1),
        qml.PauliY(wires=2),
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[3, 1]),
        qml.Hadamard(wires=1),
        qml.PauliY(wires=2),
        qml.Hadamard(wires=[0]),
    ],
    [qml.RX(0.5, wires=0), qml.RX(0.5, wires=1), qml.MultiRZ(0.3, wires=[0, 1])],
    [
        qml.RY(0.5, wires=0),
        qml.RY(0.4, wires=1),
        qml.templates.BasicEntanglerLayers([[0.5, 0.4]], wires=[0, 1]),
        qml.RY(0.5, wires=0),
        qml.RY(0.4, wires=1),
        qml.templates.BasicEntanglerLayers([[0.5, 0.4]], wires=[0, 1]),
    ],
]

ARGS = [
    [],
    [],
    [],
    [[[[0.5, 0.5], 0.3]]],
    [[[0.5, 0.4], [0.5, 0.4]], [[0.4, 0.4], []], [True, False]],
]
KWARGS = [
    {},
    {"wires": range(4), "var": True},
    {"wires": range(4), "var": True},
    {},
    {"wires": range(2), "var2": True},
]

REPEAT = zip(UNITARIES, DEPTH, ARGS, KWARGS, GATES)

########################


class TestLayer:
    """Tests the layering function"""

    def test_args_length(self):
        """Tests that the correct error is thrown when the length of an argument is incorrect"""

        params = [1, 1]

        def unitary(param, wire):
            qml.RX(param, wires=wire)

        with pytest.raises(
            ValueError,
            match=r"Each positional argument must have length matching 'depth'; expected 3",
        ):
            layer(unitary, 3, params, wires=[0])

    @pytest.mark.parametrize(("unitary", "depth", "arguments", "keywords", "gates"), REPEAT)
    def test_layer(self, unitary, depth, arguments, keywords, gates):
        """Tests that the layering function is yielding the correct sequence of gates"""

        with qml.tape.OperationRecorder() as rec:
            layer(unitary, depth, *arguments, **keywords)

        for i, gate in enumerate(rec.operations):
            prep = [gate.name, gate.parameters, gate.wires]
            target = [gates[i].name, gates[i].parameters, gates[i].wires]

        assert prep == target

    def test_layer_tf(self):
        """Tests that the layering function accepts Tensorflow variables."""

        tf = pytest.importorskip("tensorflow")

        def unitary(param):
            qml.RX(param, wires=0)

        x = tf.Variable([0.1, 0.2, 0.3])

        with qml.tape.OperationRecorder() as rec:
            layer(unitary, 3, x)

        assert len(rec.operations) == 3

        for ii, op in enumerate(rec.operations):
            assert qml.math.allclose(op.parameters[0], x[ii])
            assert isinstance(op, qml.RX)
