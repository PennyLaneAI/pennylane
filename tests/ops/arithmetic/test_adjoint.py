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
"""Tests for the Adjoint operator wrapper."""

import pytest

import pennylane as qml
from pennylane.ops.arithmetic import Adjoint

class TestInitialization:

    def test_nonparametric_ops(self):

        base = qml.PauliX("a")

        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters['base'] is base
        assert op.name == "Adjoint(PauliX)"

        assert op.num_params == 0
        assert op.parameters == []
        assert op.data == []

        assert op.wires == qml.wires.Wires("a")

    def test_parametric_ops(self):

        params = [1.2345, 2.3456, 3.4567]
        base = qml.Rot(*params, wires="b")

        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters['base'] is base
        assert op.name == "Adjoint(Rot)"

        assert op.num_params == 3
        assert qml.math.allclose(params, op.parameters)
        assert qml.math.allclose(params, op.data)

        assert op.wires == qml.wires.Wires("b")

    def test_hamiltonian_base(self):

        base = 2.0*qml.PauliX(0) @ qml.PauliY(0) + qml.PauliZ("b")

        op = Adjoint(base)

        assert op.base is base
        assert op.hyperparameters['base'] is base
        assert op.name == "Adjoint(Hamiltonian)"

        assert op.num_params == 2
        assert qml.math.allclose(op.parameters, [2.0, 1.0])
        assert qml.math.allclose(op.data, [2.0, 1.0])

        assert op.wires == qml.wires.Wires([0, "b"])


class TestQueueing:

    def test_queueing(self):

        with qml.tape.QuantumTape() as tape:
            base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
            op = Adjoint(base)

        assert tape._queue[base]['owner'] is op
        assert tape._queue[op]['owns'] is base
        assert tape.operations == [op]

    def test_queueing_base_defined_outside(self):

        base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
        with qml.tape.QuantumTape() as tape:
            op = Adjoint(base)

        assert tape._queue[base]['owner'] is op
        assert tape._queue[op]['owns'] is base
        assert tape.operations == [op]

def test_label():
    base = qml.Rot(1.2345, 2.3456, 3.4567, wires="b")
    op = Adjoint(base)
    assert op.label(decimals=2) == 'Rot\n(1.23,\n2.35,\n3.46)†'

class TestMatrix:

    def test_parametrized_gate(self):

        base = qml.RX(1.234, wires=0)
        base_matrix = base.get_matrix()
        expected = qml.math.conjugate(qml.math.transpose(base_matrix))

        op = Adjoint(base)

        assert qml.math.allclose(expected, op.get_matrix())

    def test_matrix_jax(self):

        jnp = pytest.importorskip("jax.numpy")

        base = qml.RX(jnp.array(1.2345), wires=0)
        expected = qml.math.conjugate(qml.math.transpose(base.get_matrix()))

        op = Adjoint(base)
        mat = op.get_matrix()

        assert qml.math.allclose(expected, op.get_matrix())
        assert qml.math.get_interface(mat) == "jax"

    def test_matrix_tensorflow(self)