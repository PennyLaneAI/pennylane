# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the fabricate operation."""

import pytest

import pennylane as qp
from pennylane.core import queuing
from pennylane.drawer import tape_text
from pennylane.ops.mid_measure import PauliMeasure
from pennylane.ops.qubit.fabricate import Fabricate, fabricate


class TestFabricate:
    """Tests for the Fabricate operator and fabricate function."""

    @pytest.mark.parametrize("init_state", ("plus_i", "minus_i", "magic", "magic_conj"))
    def test_fabricate_operator(self, init_state):
        """Tests that Fabricate stores the init_state correctly."""
        op = Fabricate(init_state)
        assert op.init_state == init_state
        assert len(op.wires) == 0
        assert op.label() == f"Fabricate({init_state})"
        assert repr(op) == f"Fabricate('{init_state}')"

    def test_invalid_init_state(self):
        """Tests that invalid init states raise an error."""
        with pytest.raises(ValueError, match='The init_state "invalid" is not allowed'):
            Fabricate("invalid")

        with pytest.raises(ValueError, match='The init_state "invalid" is not allowed'):
            fabricate("invalid")

    def test_fabricate_without_capture_or_qjit(self):
        """Tests that fabricate raises outside capture and QJIT contexts."""
        with pytest.raises(
            NotImplementedError,
            match="fabricate is only supported with program capture or Catalyst QJIT",
        ):
            fabricate("magic")

    @pytest.mark.jax
    @pytest.mark.capture
    def test_fabricate_capture(self):
        """Tests that fabricate is captured as fabricate_prim."""
        from pennylane.capture.primitives import fabricate_prim
        from pennylane.wires import AbstractQubit

        jax = pytest.importorskip("jax")

        def circuit():
            return fabricate("magic_conj")

        jaxpr = jax.make_jaxpr(circuit)()
        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == fabricate_prim
        assert jaxpr.eqns[0].params == {"init_state": "magic_conj"}
        assert len(jaxpr.eqns[0].outvars) == 1
        assert isinstance(jaxpr.eqns[0].outvars[0].aval, AbstractQubit)

        with pytest.raises(
            NotImplementedError, match="jaxpr containing fabricate cannot be executed"
        ):
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts)

    @pytest.mark.jax
    @pytest.mark.capture
    def test_fabricate_plxpr_conversion(self):
        """Tests that captured fabricate is converted to a Fabricate op on the tape."""
        jax = pytest.importorskip("jax")

        def circuit():
            fabricate("magic")

        jaxpr = jax.make_jaxpr(circuit)()
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)
        assert len(tape.operations) == 1
        assert isinstance(tape.operations[0], Fabricate)
        assert tape.operations[0].init_state == "magic"
        assert len(tape.operations[0].wires) == 1

    @pytest.mark.jax
    @pytest.mark.capture
    def test_fabricate_downstream_pauli_measure(self):
        """Tests that a fabricated wire can be used by downstream operations."""
        jax = pytest.importorskip("jax")

        def circuit():
            magic = fabricate("magic")
            qp.pauli_measure("ZZ", wires=[0, magic])

        jaxpr = jax.make_jaxpr(circuit)()
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)
        assert len(tape.operations) == 2
        assert isinstance(tape.operations[0], Fabricate)
        assert isinstance(tape.operations[1], PauliMeasure)
        assert tape.operations[1].wires[1] == tape.operations[0].wires[0]

    @pytest.mark.jax
    @pytest.mark.capture
    def test_fabricate_draw(self):
        """Tests that tapes containing fabricated wires can be drawn."""
        jax = pytest.importorskip("jax")

        def circuit():
            magic = fabricate("magic")
            qp.pauli_measure("ZZ", wires=[0, magic])

        jaxpr = jax.make_jaxpr(circuit)()
        tape = qp.tape.plxpr_to_tape(jaxpr.jaxpr, jaxpr.consts)
        drawing = tape_text(tape)
        assert "|m>├" in drawing
        assert "↗" in drawing

    def test_fabricate_queuing(self):
        """Tests that Fabricate can be queued on a tape."""
        with queuing.AnnotatedQueue() as q:
            Fabricate("plus_i")

        assert len(q.queue) == 1
        assert isinstance(q.queue[0], Fabricate)
        assert q.queue[0].init_state == "plus_i"


@pytest.mark.external
@pytest.mark.catalyst
def test_fabricate_catalyst_dispatch():
    """Verify Catalyst exposes fabricate when the companion release is installed."""
    pytest.importorskip("catalyst")
    from pennylane.compiler import compiler

    ops_loader = compiler.AvailableCompilers.names_entrypoints["catalyst"]["ops"].load()
    if not hasattr(ops_loader, "fabricate"):
        pytest.skip("Installed Catalyst does not yet support fabricate")
