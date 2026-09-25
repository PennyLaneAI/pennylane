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

"""Tests for emission into Catalyst IR and lowering to ``qecl``, ``pennylane.ftqc.gadget.lowering``."""

import numpy as np
import pytest

from pennylane.ftqc import gadget
from pennylane.ftqc.gadget.library import steane_code
from pennylane.ftqc.gadget.support import CATALYST_EVIDENCE

func = pytest.importorskip("xdsl.dialects.func")
xdsl_ir = pytest.importorskip("xdsl.ir")
builtin = pytest.importorskip("xdsl.dialects.builtin")
Block = xdsl_ir.Block
ModuleOp = builtin.ModuleOp
xdsl_parser = pytest.importorskip("xdsl.parser")
pytest.importorskip("catalyst")
lowering = pytest.importorskip("pennylane.ftqc.gadget.lowering")

pytestmark = pytest.mark.catalyst


def _ints(array_attr):
    return [a.value.data for a in array_attr.data]


def _parity(attr):
    terms, syndrome = attr.data
    return [_ints(t) for t in terms.data], _ints(syndrome)


def _func(module):
    (fn,) = [op for op in module.body.block.ops if isinstance(op, func.FuncOp)]
    return fn


def _body_names(module):
    return [op.name for op in _func(module).body.block.ops]


def _detectors_op(module):
    (op,) = [op for op in module.body.block.ops if isinstance(op, lowering.DetectorsOp)]
    return op


class TestDialect:
    """Tests for the proposed ``gadget`` dialect."""

    def test_records_type(self):
        """Test the printed form and shape accessors of the records type."""
        records = lowering.RecordsType(3, 5)
        assert str(records) == "!gadget.records<3 x 5>"
        assert (records.n_rounds, records.n_width) == (3, 5)

    @pytest.mark.parametrize("fixture", ["steane_mem", "rep_zz"])
    def test_text_round_trip(self, fixture, request):
        """Test that emitted IR parses back to an identical, verified module."""
        text = lowering.emit(request.getfixturevalue(fixture)[2].program).text()
        parsed = xdsl_parser.Parser(lowering.context(), text).parse_module()
        parsed.verify()
        assert str(parsed) == text

    def test_dialect_module_is_cached(self):
        """Test that Catalyst's dialect module is loaded once, so its types stay identical."""
        assert lowering.load_dialect_module("qecl") is lowering.load_dialect_module("qecl")


class TestEmit:
    """Tests for emission of traced gadgets."""

    def test_memory(self, steane_mem):
        """Test the emitted module of the Steane memory gadget."""
        prog = steane_mem[2].program
        em = lowering.emit(prog)
        ops = list(em.module.body.block.ops)
        assert [op.name for op in ops] == ["gadget.phase", "func.func", "gadget.detectors"]
        assert isinstance(ops[0], lowering.PhaseOp)
        assert _body_names(em.module) == ["gadget.rounds", "func.return"]
        fn = _func(em.module)
        assert str(fn.function_type.inputs.data[0]) == "!qecl.codeblock<1>"
        assert fn.attributes["gadget.action"].data == "idle"
        assert fn.attributes["gadget.fingerprint"].data == prog.fingerprint()
        layout = _detectors_op(em.module)
        assert layout.detectors.get_type().get_shape() == (18, 18)
        assert _ints(layout.undetermined) == []

    def test_surgery(self, rep_zz):
        """Test the emitted module of the ZZ merge."""
        prog = rep_zz[2].program
        em = lowering.emit(prog)
        assert [
            op.sym_name.data for op in em.module.body.block.ops if isinstance(op, lowering.PhaseOp)
        ] == [
            "base",
            "merged",
        ]
        assert _body_names(em.module) == [
            "gadget.rounds",
            "gadget.deform",
            "gadget.rounds",
            "gadget.observable",
            "gadget.deform",
            "gadget.rounds",
            "gadget.frame_update",
            "func.return",
        ]
        assert str(_func(em.module).function_type.inputs.data[0]) == "!qecl.codeblock<2>"

    def test_observable_carries_completed_parity(self, rep_zz):
        """Test that the emitted outcome is the layout's completed parity, not the single
        record the author wrote."""
        em = lowering.emit(rep_zz[2].program)
        (obs,) = [
            op for op in _func(em.module).body.block.ops if isinstance(op, lowering.ObservableOp)
        ]
        assert _parity(obs.parity) == ([[0, 2, 0], [0, 2, 1], [0, 2, 4]], [])
        assert str(obs.records.type) == "!gadget.records<3 x 5>"

    def test_detectors_op(self, rep_zz):
        """Test that the layout lists the undetermined record by flat index."""
        em = lowering.emit(rep_zz[2].program)
        layout = _detectors_op(em.module)
        assert layout.detectors.get_type().get_shape() == (22, 23)
        assert _ints(layout.undetermined) == [8]
        assert layout.entry_width.value.data == 4
        observables = np.array(layout.observables.get_values()).reshape(1, 23)
        assert np.nonzero(observables[0])[0].tolist() == [14, 15, 18]

    def test_deform_and_detach(self, aux_merge):
        """Test that init and readout bases are encoded as ``[qubit, axis]`` pairs."""
        em = lowering.emit(aux_merge.program)
        body = list(_func(em.module).body.block.ops)
        (deform,) = [op for op in body if isinstance(op, lowering.DeformOp)]
        (detach,) = [op for op in body if isinstance(op, lowering.DetachOp)]
        assert [_ints(p) for p in deform.init.data] == [[6, 1]]
        assert [_ints(p) for p in detach.measure_out.data] == [[6, 1]]
        assert str(detach.records.type) == "!gadget.records<1 x 1>"

    def test_only_exposed_outcomes_are_emitted(self, rep_zz):
        """Test that a callee outcome the caller does not expose emits no observable."""
        code, phases, measure_zz = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def keep_last(handle):
            handle, _ = measure_zz(handle)
            handle, (second,) = measure_zz(handle)
            return handle, gadget.observe(second, index=0)

        em = lowering.emit(keep_last.program)
        names = _body_names(em.module)
        assert names.count("gadget.observable") == 1
        assert names.count("gadget.frame_update") == 2
        assert _detectors_op(em.module).observables.get_type().get_shape()[0] == 1

    def test_multi_block_parity_is_explicit(self, rep_zz):
        """Test that a parity spanning record blocks raises instead of dropping terms."""
        code, phases, _ = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def span(handle):
            handle, pre = gadget.rounds(handle, 1, record="pre")
            handle = gadget.deform(handle, to="merged")
            handle, merged = gadget.rounds(handle, 3, record="merged")
            return handle, gadget.observe(merged.at(2, 4) ^ pre.at(0, 0), index=0)

        with pytest.raises(NotImplementedError, match=r"spans record blocks \['merged', 'pre'\]"):
            lowering.emit(span.program)


class TestLower:
    """Tests for lowering to ``qecl``."""

    def test_memory_lowers(self, steane_mem):
        """Test that a single-phase, k=1 memory gadget lowers to one ``qecl.qec`` per round."""
        em = lowering.emit(steane_mem[2].program)
        result = lowering.lower_gadget_to_qecl(em.module)
        assert result.summary() == "lowered to qecl: 3 qecl.qec, 0 qecl.measure"
        assert _body_names(result.module) == ["qecl.qec"] * 3 + ["func.return"]
        result.module.verify()
        assert "gadget.detectors" in result.text()

    def test_multi_logical_codeblock(self, rep_zz):
        """Test that a k=2 gadget stops with the evidence for the limit."""
        em = lowering.emit(rep_zz[2].program)
        with pytest.raises(lowering.LoweringGap) as info:
            lowering.lower_gadget_to_qecl(em.module)
        assert isinstance(info.value, NotImplementedError)
        assert info.value.capability == "codeblocks with k > 1 (this gadget uses k=2)"
        assert info.value.op == "func @measure_zz"
        assert info.value.evidence == CATALYST_EVIDENCE["max_k"]

    def test_deformation(self):
        """Test that a k=1 deformation stops rather than being approximated by QEC cycles."""
        code = steane_code()
        a, b = gadget.Phase.from_code("a", code), gadget.Phase.from_code("b", code)

        @gadget.define(action=gadget.Action.idle(), code=code, phases=(a, b))
        def hop(handle):
            handle, _ = gadget.rounds(handle, 1, record="r0")
            handle = gadget.deform(handle, to="b")
            handle, _ = gadget.rounds(handle, 1, record="r1")
            return handle

        with pytest.raises(
            lowering.LoweringGap, match="changes the measured stabilizer group"
        ) as info:
            lowering.lower_gadget_to_qecl(lowering.emit(hop.program).module)
        assert info.value.op == "gadget.deform to @b"

    def test_detach(self):
        """Test that measuring out part of a k=1 codeblock stops."""
        code = steane_code()
        n = 8
        pad = np.zeros((3, 1), dtype=np.uint8)
        z7 = np.zeros((1, n), dtype=np.uint8)
        z7[0, 7] = 1
        wide = gadget.Phase(
            "wide",
            np.hstack([code.hx, pad]),
            np.vstack([np.hstack([code.hz, pad]), z7]),
            np.ones(n, dtype=bool),
        )
        base = gadget.Phase.from_code("base", code, n_frame=n)

        @gadget.define(action=gadget.Action.idle(), code=code, phases=(wide, base), n_data=7)
        def drop(handle):
            handle, _ = gadget.rounds(handle, 1, record="r0")
            handle, _ = gadget.detach(handle, to="base", measure_out={7: "z"}, record="out")
            handle, _ = gadget.rounds(handle, 1, record="r1")
            return handle

        with pytest.raises(lowering.LoweringGap, match="mid-circuit readout") as info:
            lowering.lower_gadget_to_qecl(lowering.emit(drop.program).module)
        assert info.value.op == "gadget.detach to @base"

    def test_outcome(self):
        """Test that a k=1 gadget with an outcome reports the gap for outcomes, not the
        generic gap for records used by a later op."""
        code = steane_code()
        meas = gadget.Phase("meas", code.hx, np.vstack([code.hz, code.lz]), np.ones(7, bool))

        @gadget.define(action=gadget.Action.measure(("z", (0,))), code=code, phases=(meas,))
        def measure_z(handle):
            handle, r = gadget.rounds(handle, 1, record="m")
            return handle, gadget.observe(r.product((6,)), index=0)

        with pytest.raises(lowering.LoweringGap, match="non-destructive logical product"):
            lowering.lower_gadget_to_qecl(lowering.emit(measure_z.program).module)

    def test_frame_update(self):
        """Test that a k=1 gadget with a conditioned frame update reports that gap."""
        code = steane_code()
        phase = gadget.Phase.from_code("s", code)

        @gadget.define(action=gadget.Action.idle(), code=code, phases=(phase,))
        def conditioned(handle):
            handle, r = gadget.rounds(handle, 1, record="m")
            return gadget.frame(handle, r.product((3,)))

        with pytest.raises(lowering.LoweringGap, match="classically conditioned Pauli frame"):
            lowering.lower_gadget_to_qecl(lowering.emit(conditioned.program).module)


class TestGadgetCalls:
    """Tests for the hooks Catalyst's QEC passes use to compile ``gadget.apply`` calls."""

    def test_inline(self, steane_mem):
        """Test that a memory gadget becomes a chain of tagged qecl.qec cycles on the given
        codeblock."""
        _, _, memory = steane_mem
        block = Block(arg_types=[lowering.codeblock_type(memory.program)])
        payload = lowering.emit(memory.program).text()
        ops, out = lowering.inline_gadget_call(payload, block.args[0])
        assert [op.name for op in ops] == ["qecl.qec"] * 3
        assert ops[0].operands[0] is block.args[0]
        assert ops[1].operands[0] is ops[0].results[0]
        assert out is ops[-1].results[0]
        assert ops[0].attributes["gadget.name"].data == "steane_memory"

    def test_inline_rejects_other_codeblock_types(self, steane_mem):
        """Test that a gadget cannot be inlined on a codeblock of a different type."""
        _, _, memory = steane_mem
        qecl = lowering.load_dialect_module("qecl")
        block = Block(arg_types=[qecl.LogicalCodeblockType(2)])
        with pytest.raises(gadget.GadgetError, match="is applied to !qecl.codeblock<2>"):
            lowering.inline_gadget_call(lowering.emit(memory.program).text(), block.args[0])

    def test_check_pipeline_code(self, steane_mem):
        """Test that inlined cycles are accepted for the same checks, in any row order, and
        rejected for checks on a different qubit order."""
        code, _, memory = steane_mem
        block = Block(arg_types=[lowering.codeblock_type(memory.program)])
        ops, _ = lowering.inline_gadget_call(lowering.emit(memory.program).text(), block.args[0])
        module = ModuleOp([op.clone() for op in ops[:1]])
        lowering.check_pipeline_code(module, code.hx[::-1], code.hz, "Steane")
        with pytest.raises(gadget.GadgetError, match="whose checks differ from those of"):
            lowering.check_pipeline_code(module, code.hx[:, ::-1], code.hz, "reversed")
