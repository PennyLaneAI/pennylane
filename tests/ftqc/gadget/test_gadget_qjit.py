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
"""End-to-end tests for gadgets compiled with :func:`~pennylane.qjit` and Catalyst's QEC
pipeline."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.ftqc import gadget
from pennylane.ftqc.gadget.library import steane_code, steane_memory

pytest.importorskip("xdsl")
pytest.importorskip("catalyst")
ftqc = pytest.importorskip("catalyst.ftqc")
python_interface = pytest.importorskip("catalyst.python_interface")
qecl_passes = pytest.importorskip("catalyst.python_interface.transforms.qecl")
qecp_passes = pytest.importorskip("catalyst.python_interface.transforms.qecp")
xdsl_passes = pytest.importorskip("xdsl.passes")
CompileError = pytest.importorskip("catalyst.utils.exceptions").CompileError

pytestmark = pytest.mark.catalyst


class _CountGadgetCycles(xdsl_passes.ModulePass):
    """Records how many ``qecl.qec`` cycles each inlined gadget contributed."""

    name = "test-count-gadget-cycles"
    seen: list[dict[str, int]] = []

    def apply(self, _ctx, op):
        counts: dict[str, int] = {}
        for inner in op.walk():
            if inner.name == "qecl.qec" and "gadget.name" in inner.attributes:
                name = inner.attributes["gadget.name"].data
                counts[name] = counts.get(name, 0) + 1
        type(self).seen.append(counts)


count_gadget_cycles = python_interface.compiler_transform(_CountGadgetCycles)


def _qec_circuit(traced, prepare_one=True, shots=5):
    """A one-logical-qubit QNode that applies ``traced`` in the Steane QEC pipeline."""

    @qp.qjit(capture=True, collect_decomp_rules=False, pipelines=ftqc.qec_pipeline())
    @qecp_passes.convert_qecp_to_quantum_pass
    @qecp_passes.convert_qecl_to_qecp_pass(qec_code="Steane", number_errors=0)
    @count_gadget_cycles
    @qecl_passes.convert_quantum_to_qecl_pass(k=1)
    @qp.set_shots(shots)
    @qp.qnode(qp.device("lightning.qubit", wires=1), mcm_method="one-shot")
    def circuit():
        if prepare_one:
            qp.X(0)
        gadget.apply(traced, wires=0)
        return qp.sample(wires=[0])

    return circuit


class TestMemoryGadget:
    """Tests for a memory gadget compiled and executed through the QEC pipeline."""

    @pytest.mark.parametrize("prepare_one, expected", [(True, 1), (False, 0)])
    def test_preserves_logical_state(self, prepare_one, expected):
        """Test that the memory gadget preserves the logical computational-basis state."""
        _, _, memory = steane_memory(rounds=3)
        samples = np.ravel(_qec_circuit(memory, prepare_one)())
        assert samples.tolist() == [expected] * 5

    def test_rounds_are_inserted(self):
        """Test that every round of the gadget becomes a qecl.qec cycle on the codeblock."""
        _, _, memory = steane_memory(rounds=4)
        _CountGadgetCycles.seen.clear()
        _qec_circuit(memory)()
        assert _CountGadgetCycles.seen == [{"steane_memory": 4}]


class TestRejectedGadgets:
    """Tests that gadgets the pipeline cannot compile faithfully are rejected."""

    def test_code_mismatch(self):
        """Test that a gadget written for different checks than the pipeline code is rejected,
        rather than compiled with the pipeline code's cycles."""
        h = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]])
        ones = np.ones((1, 7), dtype=np.uint8)
        other = gadget.CSSCode("reordered", h.astype(np.uint8), h.astype(np.uint8), ones, ones)

        @gadget.define(
            action=gadget.Action.idle(),
            code=other,
            phases=(gadget.Phase.from_code("s", other),),
        )
        def reordered_memory(handle):
            handle, _ = gadget.rounds(handle, 2, record="m")
            return handle

        with pytest.raises(CompileError, match="written for code reordered, whose checks differ"):
            _qec_circuit(reordered_memory)()

    def test_deformation(self):
        """Test that a gadget changing the measured stabilizer group reports the gap."""
        code = steane_code()
        a, b = gadget.Phase.from_code("a", code), gadget.Phase.from_code("b", code)

        @gadget.define(action=gadget.Action.idle(), code=code, phases=(a, b))
        def hop(handle):
            handle, _ = gadget.rounds(handle, 1, record="r0")
            handle = gadget.deform(handle, to="b")
            handle, _ = gadget.rounds(handle, 1, record="r1")
            return handle

        with pytest.raises(CompileError, match="changes the measured stabilizer group"):
            _qec_circuit(hop)()

    def test_outcome(self):
        """Test that a gadget with a measurement outcome reports the gap for outcomes."""
        code = steane_code()
        meas = gadget.Phase("meas", code.hx, np.vstack([code.hz, code.lz]), np.ones(7, bool))

        @gadget.define(action=gadget.Action.measure(("z", (0,))), code=code, phases=(meas,))
        def measure_z(handle):
            handle, r = gadget.rounds(handle, 1, record="m")
            return handle, gadget.observe(r.product((6,)), index=0)

        with pytest.raises(CompileError, match="non-destructive logical product measurement"):
            _qec_circuit(measure_z)()
