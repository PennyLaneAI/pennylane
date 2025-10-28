# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for capturing Pauli product measurements."""

# pylint: disable=wrong-import-order,wrong-import-position,ungrouped-imports

import pytest

import pennylane as qml
from pennylane.ops import MeasurementValue, PauliMeasure
from pennylane.wires import Wires

jax = pytest.importorskip("jax")

import jax.numpy as jnp

from pennylane.tape.plxpr_conversion import plxpr_to_tape

pytestmark = [pytest.mark.capture, pytest.mark.jax]


@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("pauli_word", ["XY", "Z"])
class TestPauliMeasure:
    """Tests that the Pauli product measurement can be capture."""

    @pytest.mark.unit
    def test_pauli_measure(self, pauli_word, postselect):
        """Tests that a basic qml.pauli_measure can be captured."""

        def f(wires):
            m0 = qml.pauli_measure(pauli_word, wires=wires, postselect=postselect)
            return m0

        wires = jnp.array(range(len(pauli_word)))
        jaxpr = jax.make_jaxpr(f)(wires)
        assert jaxpr.eqns[-1].primitive.name == "pauli_measure"
        invars = jaxpr.eqns[-1].invars
        outvars = jaxpr.eqns[-1].outvars
        expected_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
        assert len(invars) == len(wires)
        for invar in invars:
            assert invar.aval == jax.core.ShapedArray((), expected_dtype)
        assert len(outvars) == 1
        assert outvars[0].aval == jax.core.ShapedArray((), expected_dtype)
        assert set(jaxpr.eqns[-1].params.keys()) == {"pauli_word", "postselect"}

    @pytest.mark.integration
    def test_pauli_measure_bind(self, pauli_word, postselect):
        """Tests the pauli_measure primitive can be converted back."""

        wires = [0, 1] if len(pauli_word) == 2 else 1

        @qml.qnode(qml.device("default.qubit", wires=2))
        def f():
            m0 = qml.pauli_measure(pauli_word, wires=wires, postselect=postselect)
            return qml.expval(m0)

        jaxpr = jax.make_jaxpr(f)()
        tape = plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)
        assert len(tape.operations) == 1
        assert isinstance(tape.operations[0], PauliMeasure)
        assert tape.operations[0].pauli_word == pauli_word
        assert tape.operations[0].postselect == postselect
        assert tape.operations[0].wires == Wires(wires)
        assert len(tape.measurements) == 1
        assert tape.measurements[0].mv.measurements[0] is tape.operations[0]
        assert isinstance(tape.measurements[0].mv, MeasurementValue)
