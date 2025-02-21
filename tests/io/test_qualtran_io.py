# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.io.qualtran_io` module.
"""
import pytest

import pennylane as qml


class TestFromBloq:
    """Test that FromBloq accurately wraps around Bloqs."""

    def test_attributes(self):
        pass

    def test_composite_bloq(self):
        """Tests that a simple composite bloq has the correct decomposition after wrapped with `FromBloq`"""
        from qualtran import BloqBuilder
        from qualtran.bloqs.basic_gates import Hadamard, CNOT, Toffoli

        bb = BloqBuilder()  # bb is the circuit like object

        w1 = bb.add_register('wire1', 1)
        w2 = bb.add_register('wire2', 1)
        aux = bb.add_register('aux_wires', 2)  # add wires

        aux_wires = bb.split(aux)

        w1 = bb.add(Hadamard(), q=w1)
        w2 = bb.add(Hadamard(), q=w2)

        w1, aux1 = bb.add(CNOT(), ctrl=w1, target=aux_wires[0])
        w2, aux2 = bb.add(CNOT(), ctrl=w2, target=aux_wires[1])

        ctrl_aux, w1 = bb.add(Toffoli(), ctrl=(aux1, aux2), target=w1)
        ctrl_aux, w2 = bb.add(Toffoli(), ctrl=ctrl_aux, target=w2)
        aux_wires = bb.join(ctrl_aux)

        circuit_bloq = bb.finalize(wire1=w1, wire2=w2, aux_wires=aux_wires)

        decomp = qml.FromBloq(circuit_bloq, wires=list(range(4))).decomposition()
        expected_decomp = [qml.H(0), qml.H(1), qml.CNOT([0, 2]), qml.CNOT([1, 3]), qml.Toffoli([2, 3, 0]), qml.Toffoli([2, 3, 1])]
        assert decomp == expected_decomp
        
        mapped_decomp = qml.FromBloq(circuit_bloq, wires=[3, 0, 1, 2]).decomposition()
        mapped_expected_decomp = [qml.H(3), qml.H(0), qml.CNOT([3, 1]), qml.CNOT([0, 2]), qml.Toffoli([1, 2, 3]), qml.Toffoli([1, 2, 0])]
        assert mapped_decomp == mapped_expected_decomp
