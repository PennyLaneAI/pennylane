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

"""Unit tests for authoring gadgets with ``pennylane.gadget.define``."""

import numpy as np
import pytest

from pennylane import gadget
from pennylane.gadget.ir import Deform, Detach, Frame, Observe
from pennylane.gadget.library import steane_code


def _idle(code, phases, **kwargs):
    return gadget.define(action=gadget.Action.idle(), code=code, phases=phases, **kwargs)


def _measure_zz(code, phases, **kwargs):
    return gadget.define(
        action=gadget.Action.measure(("z", (0, 1))),
        code=code,
        phases=phases,
        **kwargs,
    )


class TestDefine:
    """Tests for the ``define`` decorator."""

    def test_returns_definition(self, rep_zz):
        """Test that decoration traces the body once into a definition."""
        _, _, defn = rep_zz
        assert isinstance(defn, gadget.TracedGadget)
        assert defn.name == "measure_zz"
        assert repr(defn) == "<gadget measure_zz action=measure(Z_0_1)>"
        assert defn.records == defn.program.records

    def test_defaults(self, rep_zz):
        """Test that notes come from the docstring and the frame update defaults to zero."""
        prog = rep_zz[2].program
        assert prog.notes == "Measure logical Z tensor Z by merging two repetition-code blocks."
        assert prog.frame_update.shape == (1, 6)
        assert not prog.frame_update.any()
        assert prog.n_data == 6

    def test_explicit_notes(self, steane_mem):
        """Test that explicit notes override the docstring."""
        code, phases, _ = steane_mem

        @_idle(code, phases, notes="custom")
        def mem(handle):
            """Docstring."""
            handle, _ = gadget.rounds(handle, 1, record="r")
            return handle

        assert mem.program.notes == "custom"

    def test_entry_phase(self):
        """Test that the input handle starts in the declared entry phase."""
        code = steane_code()
        a, b = gadget.Phase.from_code("a", code), gadget.Phase.from_code("b", code)

        @_idle(code, (a, b), entry_phase="b")
        def start_in_b(handle):
            handle, _ = gadget.rounds(handle, 1, record="r")
            return handle

        assert start_in_b.program.ops[0].phase == "b"

    def test_unknown_entry_phase(self, steane_mem):
        """Test that the entry phase must be declared."""
        code, phases, _ = steane_mem
        with pytest.raises(gadget.GadgetError, match="no phase named 'nope'; declared phases are"):

            @_idle(code, phases, entry_phase="nope")
            def body(handle):
                return handle

    def test_needs_a_phase(self, steane_mem):
        """Test that at least one phase must be declared."""
        code, _, _ = steane_mem
        with pytest.raises(gadget.GadgetError, match="at least one phase must be declared"):

            @_idle(code, ())
            def body(handle):
                return handle

    def test_phases_share_a_frame(self, steane_mem):
        """Test that every phase must live on the same qubit frame."""
        code, (steane,), _ = steane_mem
        wide = gadget.Phase.from_code("wide", steane_code(), n_frame=8)
        with pytest.raises(gadget.GadgetError, match=r"share one qubit frame, got sizes \[7, 8\]"):

            @_idle(code, (steane, wide))
            def body(handle):
                return handle

    def test_logical_qubits_must_exist(self, rep_zz):
        """Test that a declared measurement can only refer to logical qubits of the code."""
        code, phases, _ = rep_zz
        with pytest.raises(
            gadget.GadgetError, match=r"logical qubit\(s\) \[2\] out of range for code rep3x2"
        ):

            @gadget.define(action=gadget.Action.measure(("z", (0, 2))), code=code, phases=phases)
            def body(handle):
                return handle


class TestOwnership:
    """Tests for single use of handles."""

    def test_consumed_patch_cannot_be_reused(self, rep_zz):
        """Test that a handle cannot be used after it was consumed."""
        code, (base, _), _ = rep_zz
        with pytest.raises(
            gadget.OwnershipError,
            match=r"rounds: handle %0 was already consumed by rounds \(op 0\)",
        ):

            @_idle(code, (base,))
            def reuse(handle):
                gadget.rounds(handle, 1, record="a")
                again, _ = gadget.rounds(handle, 1, record="b")
                return again

    def test_returned_patch_must_be_live(self, steane_mem):
        """Test that the body cannot return a consumed handle."""
        code, phases, _ = steane_mem
        with pytest.raises(gadget.OwnershipError, match="returned handle %0 was already consumed"):

            @_idle(code, phases)
            def stale(handle):
                gadget.rounds(handle, 1, record="r")
                return handle

    def test_every_patch_reaches_the_boundary(self, steane_mem):
        """Test that a handle produced in the body must be passed on to the end of the body."""
        code, phases, _ = steane_mem
        with pytest.raises(
            gadget.OwnershipError, match=r"handle value\(s\) \[1\] are produced but never consumed"
        ):

            @_idle(code, phases)
            def drop(handle):
                gadget.rounds(handle, 1, record="r")
                return gadget.Handle(value_id=99, code=code, phase="steane")

    def test_non_patch_argument(self, steane_mem):
        """Test that traced ops require a handle handle."""
        code, phases, _ = steane_mem
        with pytest.raises(gadget.GadgetError, match="frame: expected a Handle, got str"):

            @_idle(code, phases)
            def body(handle):
                gadget.frame("block", gadget.RecordExpr())
                return handle

    @pytest.mark.parametrize(
        "call",
        [
            lambda: gadget.rounds(None, 1, record="r"),
            lambda: gadget.deform(None, to="p"),
            lambda: gadget.detach(None, to="p", record="r"),
            lambda: gadget.observe(gadget.RecordExpr(), index=0),
            lambda: gadget.frame(None, gadget.RecordExpr()),
        ],
    )
    def test_ops_outside_a_body(self, call):
        """Test that traced ops can only be called while tracing."""
        with pytest.raises(gadget.GadgetError, match="can only be called inside"):
            call()


class TestRounds:
    """Tests for ``rounds``."""

    @pytest.mark.parametrize("count", [0, -1, 1.0, True, "3"])
    def test_count_must_be_static_positive_int(self, steane_mem, count):
        """Test that round counts must be positive Python ints fixed at trace time."""
        code, phases, _ = steane_mem
        with pytest.raises(gadget.GadgetError, match="count must be a positive Python int"):

            @_idle(code, phases)
            def body(handle):
                handle, _ = gadget.rounds(handle, count, record="r")
                return handle

    def test_duplicate_record_name(self, steane_mem):
        """Test that record block names are unique within a gadget."""
        code, phases, _ = steane_mem
        with pytest.raises(gadget.GadgetError, match="duplicate record block name 'r'"):

            @_idle(code, phases)
            def body(handle):
                handle, _ = gadget.rounds(handle, 1, record="r")
                handle, _ = gadget.rounds(handle, 1, record="r")
                return handle

    def test_record_family(self, rep_zz):
        """Test that the record block is shaped by the phase and names its producing op."""
        fam = rep_zz[2].program.record("merged")
        assert (fam.rounds, fam.width, fam.op_index) == (3, 5, 2)
        assert fam.axes == ("z",) * 5
        assert fam.phase == "merged"


class TestDeformAndDetach:
    """Tests for ``deform`` and ``detach``."""

    def test_activated_qubit_needs_init(self, aux_pair):
        """Test that a qubit activated by a deformation must be given an initial basis."""
        code, base, merged = aux_pair
        with pytest.raises(
            gadget.GadgetError,
            match=r"deform to merged: qubits \[6\] become active but no init basis was given",
        ):

            @_measure_zz(code, (base, merged), n_data=6)
            def uninitialized(handle):
                handle, _ = gadget.rounds(handle, 1, record="pre")
                handle = gadget.deform(handle, to="merged")
                handle, checks = gadget.rounds(handle, 3, record="merged")
                return handle, gadget.observe(checks.product((4, 5)), index=0)

    def test_init_on_inactive_qubit(self, aux_pair):
        """Test that only qubits live in the target phase can be initialized."""
        code, base, merged = aux_pair
        with pytest.raises(gadget.GadgetError, match="qubit 6 is initialized but is not active"):

            @_idle(code, (base, merged), n_data=6)
            def body(handle):
                return gadget.deform(handle, to="base", init={6: "z"})

    def test_init_is_normalized(self, aux_merge):
        """Test that init bases are lower-cased and recorded on the op."""
        (deform,) = [op for op in aux_merge.program.ops if isinstance(op, Deform)]
        assert deform.init == ((6, "z"),)

    def test_unknown_target_phase(self, rep_zz):
        """Test that deformations can only target declared phases."""
        code, phases, _ = rep_zz
        with pytest.raises(gadget.GadgetError, match="no phase named 'nowhere'"):

            @_idle(code, phases)
            def body(handle):
                return gadget.deform(handle, to="nowhere")

    def test_detach_needs_readout_basis(self, aux_pair):
        """Test that qubits leaving the frame must be read out."""
        code, base, merged = aux_pair
        with pytest.raises(
            gadget.GadgetError,
            match=r"detach to base: qubits \[6\] leave the frame but no readout basis",
        ):

            @_idle(code, (base, merged), n_data=6)
            def body(handle):
                handle = gadget.deform(handle, to="merged", init={6: "z"})
                handle, _ = gadget.detach(handle, to="base", record="out")
                return handle

    def test_detach_records(self, aux_merge):
        """Test that the readout block has one record per measured-out qubit."""
        prog = aux_merge.program
        (detach,) = [op for op in prog.ops if isinstance(op, Detach)]
        assert detach.measure_out == ((6, "z"),)
        fam = prog.record("out")
        assert (fam.rounds, fam.width, fam.axes, fam.phase) == (1, 1, ("z",), "merged")


class TestObserveAndFrame:
    """Tests for ``observe``, ``frame`` and outcome validation."""

    def test_observe_needs_a_parity(self, rep_zz):
        """Test that an empty parity cannot be an outcome."""
        code, phases, _ = rep_zz
        with pytest.raises(gadget.GadgetError, match="expected a non-empty record parity"):

            @_measure_zz(code, phases)
            def body(handle):
                return handle, gadget.observe(gadget.RecordExpr(), index=0)

    def test_outcome_xor(self):
        """Test that outcomes combine with outcomes and parities."""
        fam = gadget.RecordBlock("m", 0, "p", 1, 2, ("z", "z"))
        a = gadget.Outcome(0, fam.at(0, 0))
        b = gadget.Outcome(1, fam.at(0, 1))
        assert (a ^ b) == fam.at(0, 0) ^ fam.at(0, 1)
        assert (a ^ fam.at(0, 0)) == gadget.RecordExpr()

    def test_frame_records_condition(self, rep_zz):
        """Test that a frame update records its condition and declared row."""
        prog = rep_zz[2].program
        (frame,) = [op for op in prog.ops if isinstance(op, Frame)]
        assert frame.expr == prog.observables[0].expr
        assert frame.update_index == 0

    def test_body_must_return_patch(self, steane_mem):
        """Test that the body returns the handle, optionally followed by outcomes."""
        code, phases, _ = steane_mem
        with pytest.raises(gadget.GadgetError, match="must return the handle.*got int"):

            @_idle(code, phases)
            def body(_block):
                return 1

    def test_outcome_count(self, rep_zz):
        """Test that the number of observed outcomes must match the declared action."""
        code, phases, _ = rep_zz
        with pytest.raises(gadget.GadgetError, match=r"has 1 outcome\(s\) but the body exposes 0"):

            @_measure_zz(code, phases)
            def body(handle):
                handle, _ = gadget.rounds(handle, 1, record="r")
                return handle

    def test_outcome_indices(self, rep_zz):
        """Test that outcome indices must be exactly 0..t-1."""
        code, phases, _ = rep_zz
        with pytest.raises(gadget.GadgetError, match=r"must be exactly 0\.\.0, got \[1\]"):

            @_measure_zz(code, phases)
            def body(handle):
                handle, r = gadget.rounds(handle, 1, record="r")
                return handle, gadget.observe(r.product(), index=1)

    def test_frame_update_rows(self, rep_zz):
        """Test that the frame update has one row per outcome."""
        code, phases, _ = rep_zz
        with pytest.raises(gadget.GadgetError, match=r"frame_update has 2 rows for 1 outcome\(s\)"):

            @_measure_zz(code, phases, frame_update=np.zeros((2, 6), dtype=np.uint8))
            def body(handle):
                handle, r = gadget.rounds(handle, 1, record="r")
                return handle, gadget.observe(r.product(), index=0)

    def test_outcomes_in_a_sequence(self, rep_zz):
        """Test that outcomes may be returned as a sequence after the handle."""
        code, phases, _ = rep_zz

        @_measure_zz(code, phases)
        def body(handle):
            handle, r = gadget.rounds(handle, 1, record="r")
            return handle, [gadget.observe(r.product(), index=0)]

        assert len(body.program.observables) == 1


class TestProtocol:
    """Tests for composing gadgets inside an enclosing gadget."""

    def test_call_inlines_callee(self, rep_zz):
        """Test that an enclosing gadget has the same type as a gadget and inlines its callees with
        invocation-qualified record names."""
        code, phases, measure_zz = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def once(handle):
            handle, (outcome,) = measure_zz(handle)
            return handle, gadget.observe(outcome, index=0)

        assert isinstance(once, gadget.TracedGadget)
        assert [r.name for r in once.records] == [
            "measure_zz#0/pre",
            "measure_zz#0/merged",
            "measure_zz#0/post",
        ]
        assert once.program.observables[0].expr.describe() == "measure_zz#0/merged[r2,c4]"

    def test_invocations_are_numbered(self, steane_mem):
        """Test that the n-th invocation of a gadget is qualified with ``#n``, regardless of
        how many ops the gadget has."""
        code, phases, _ = steane_mem

        @_idle(code, phases)
        def two_windows(handle):
            handle, _ = gadget.rounds(handle, 1, record="a")
            handle, _ = gadget.rounds(handle, 1, record="b")
            return handle

        @gadget.define(action=gadget.Action.idle(), code=code, phases=phases)
        def twice(handle):
            handle, _ = two_windows(handle)
            handle, _ = two_windows(handle)
            return handle

        assert [r.name for r in twice.records] == [
            "two_windows#0/a",
            "two_windows#0/b",
            "two_windows#1/a",
            "two_windows#1/b",
        ]

    def test_callee_phases_are_added(self, rep_zz):
        """Test that phases used by a callee join the caller's phases."""
        code, (base, _), measure_zz = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=(base,))
        def once(handle):
            handle, (outcome,) = measure_zz(handle)
            return handle, gadget.observe(outcome, index=0)

        assert [p.name for p in once.program.phases] == ["base", "merged"]

    def test_code_mismatch(self, rep_zz, steane_mem):
        """Test that a gadget cannot be called on a handle of a different code."""
        code, phases, _ = rep_zz
        _, _, memory = steane_mem
        with pytest.raises(gadget.GadgetError, match="code mismatch, gadget expects Steane"):

            @gadget.define(action=gadget.Action.idle(), code=code, phases=phases)
            def body(handle):
                handle, _ = memory(handle)
                return handle

    def test_entry_phase_mismatch(self, rep_zz):
        """Test that a gadget must be called on a handle in its entry phase."""
        code, phases, measure_zz = rep_zz
        with pytest.raises(
            gadget.GadgetError, match="gadget expects a handle in phase 'base', got 'merged'"
        ):

            @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
            def body(handle):
                handle = gadget.deform(handle, to="merged")
                return measure_zz(handle)

    def test_phase_name_conflict(self, steane_mem):
        """Test that a callee phase cannot silently shadow a different caller phase."""
        code, (steane,), memory = steane_mem
        impostor = gadget.Phase(
            "steane", steane.hx, np.zeros((0, 7), dtype=np.uint8), steane.active
        )
        with pytest.raises(gadget.GadgetError, match="'steane' already refers to different checks"):

            @gadget.define(action=gadget.Action.idle(), code=code, phases=(impostor,))
            def body(handle):
                handle, _ = memory(handle)
                return handle

    def test_call_consumes_patch(self, steane_mem):
        """Test that calling a gadget consumes the handle it is given."""
        code, phases, memory = steane_mem
        with pytest.raises(gadget.OwnershipError, match="call steane_memory"):

            @gadget.define(action=gadget.Action.idle(), code=code, phases=phases)
            def body(handle):
                memory(handle)
                gadget.rounds(handle, 1, record="r")
                return handle


def _twice(code, phases, callee, expose):
    """A gadget calling ``callee`` twice; ``expose(first, second)`` returns its outcomes."""

    @gadget.define(action=gadget.Action.measure(*[("z", (0, 1))] * 2), code=code, phases=phases)
    def twice(handle):
        handle, (first,) = callee(handle)
        handle, (second,) = callee(handle)
        return handle, expose(first, second)

    return twice


class TestProtocolOutcomes:
    """Tests that an enclosing gadget decides explicitly which outcomes of its callees it exposes."""

    def test_callee_outcomes_are_internal(self, rep_zz):
        """Test that an outcome of a called gadget is not an outcome of the caller."""
        code, phases, measure_zz = rep_zz
        seen = []

        @gadget.define(action=gadget.Action.idle(), code=code, phases=phases)
        def discard(handle):
            handle, outcomes = measure_zz(handle)
            seen.extend(outcomes)
            return handle

        assert len(seen) == 1
        assert seen[0].index is None
        assert discard.program.observables == ()
        assert gadget.derive_detectors(discard.program).observables == ()
        (internal,) = [op for op in discard.program.ops if isinstance(op, Observe)]
        assert internal.observable_index is None

    def test_returning_is_not_exposing(self, rep_zz):
        """Test that returning a callee's outcome without observing it does not expose it."""
        code, phases, measure_zz = rep_zz
        with pytest.raises(gadget.GadgetError, match=r"has 1 outcome\(s\) but the body exposes 0"):

            @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
            def forgot(handle):
                return measure_zz(handle)

    def test_exposed_in_place(self, rep_zz):
        """Test that exposing an outcome relabels the callee's observe op where it stands,
        so it is completed exactly as in the callee."""
        code, phases, measure_zz = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def once(handle):
            handle, (outcome,) = measure_zz(handle)
            return handle, gadget.observe(outcome, index=0)

        prog = once.program
        assert [type(op) for op in prog.ops] == [type(op) for op in measure_zz.program.ops]
        (obs,) = gadget.derive_detectors(prog).observables
        assert obs.expr.describe() == (
            "measure_zz#0/merged[r2,c0] ^ measure_zz#0/merged[r2,c1] ^ measure_zz#0/merged[r2,c4]"
        )

    def test_expose_in_any_order(self, rep_zz):
        """Test that the caller chooses the outcome order."""
        code, phases, measure_zz = rep_zz
        twice = _twice(
            code,
            phases,
            measure_zz,
            lambda a, b: (gadget.observe(b, index=0), gadget.observe(a, index=1)),
        )
        prog = twice.program
        assert [op.observable_index for op in prog.observables] == [1, 0]
        by_index = {o.index: o for o in gadget.derive_detectors(prog).observables}
        assert by_index[0].author_expr.describe() == "measure_zz#1/merged[r2,c4]"

    def test_expose_a_subset(self, rep_zz):
        """Test that the caller can expose some outcomes of its callees and not others."""
        code, phases, measure_zz = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def keep_last(handle):
            handle, _ = measure_zz(handle)
            handle, (second,) = measure_zz(handle)
            return handle, gadget.observe(second, index=0)

        (obs,) = gadget.derive_detectors(keep_last.program).observables
        assert obs.author_expr.describe() == "measure_zz#1/merged[r2,c4]"

    def test_expose_twice(self, rep_zz):
        """Test that an outcome can be exposed only once."""
        code, phases, measure_zz = rep_zz
        with pytest.raises(gadget.GadgetError, match="already exposed as outcome 0"):
            _twice(
                code,
                phases,
                measure_zz,
                lambda a, b: (gadget.observe(a, index=0), gadget.observe(a, index=1)),
            )

    def test_foreign_outcome(self, rep_zz):
        """Test that an outcome that was not produced in this body cannot be exposed."""
        code, phases, _ = rep_zz
        stray = gadget.Outcome(index=None, expr=gadget.entry_syndrome((0,)), op_index=0)
        with pytest.raises(gadget.GadgetError, match="was not produced in this body"):

            @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
            def body(handle):
                handle, _ = gadget.rounds(handle, 1, record="r")
                return handle, gadget.observe(stray, index=0)

    def test_combined_outcomes_are_a_new_parity(self, rep_zz):
        """Test that a parity of several outcomes is a new outcome at the point it is
        observed, not a relabelling of either callee outcome."""
        code, phases, measure_zz = rep_zz

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def combined(handle):
            handle, (first,) = measure_zz(handle)
            handle, (second,) = measure_zz(handle)
            return handle, gadget.observe(first ^ second, index=0)

        ops = combined.program.ops
        assert [op.observable_index for op in ops if isinstance(op, Observe)] == [None, None, 0]
        assert ops[-1].expr == combined.program.observables[0].expr

    def test_nested_calls(self, rep_zz):
        """Test that a caller's outcomes are returned in its declared order when it is
        itself called, and are internal to the caller."""
        code, phases, measure_zz = rep_zz
        inner = _twice(
            code,
            phases,
            measure_zz,
            lambda a, b: (gadget.observe(b, index=0), gadget.observe(a, index=1)),
        )

        @gadget.define(action=gadget.Action.measure(("z", (0, 1))), code=code, phases=phases)
        def outer(handle):
            handle, (zero, _) = inner(handle)
            return handle, gadget.observe(zero, index=0)

        (obs,) = outer.program.observables
        assert obs.expr.describe() == "twice#0/measure_zz#1/merged[r2,c4]"

    def test_frame_updates_are_carried(self, rep_zz):
        """Test that a callee's declared byproduct is carried into the caller rather than
        dropped, with each call's frame op pointing at its own carried row."""
        code, phases, _ = rep_zz
        byproduct = np.zeros((1, 6), dtype=np.uint8)
        byproduct[0, 0] = 1

        @_measure_zz(code, phases, frame_update=byproduct)
        def with_byproduct(handle):
            handle, _ = gadget.rounds(handle, 1, record="pre")
            handle = gadget.deform(handle, to="merged")
            handle, checks = gadget.rounds(handle, 3, record="merged")
            outcome = gadget.observe(checks.product((4,)), index=0)
            handle = gadget.deform(handle, to="base")
            return gadget.frame(handle, outcome), outcome

        twice = _twice(
            code,
            phases,
            with_byproduct,
            lambda a, b: (gadget.observe(a, index=0), gadget.observe(b, index=1)),
        )
        prog = twice.program
        assert prog.frame_update.tolist() == [[0] * 6, [0] * 6, [1] + [0] * 5, [1] + [0] * 5]
        assert [op.update_index for op in prog.ops if isinstance(op, Frame)] == [2, 3]

    def test_frame_update_width_mismatch(self, aux_pair):
        """Test that a callee's frame updates must act on the caller's data block."""
        code, base, merged = aux_pair

        @_idle(code, (base, merged), n_data=7)
        def wide(handle):
            handle, _ = gadget.rounds(handle, 1, record="r")
            return handle

        with pytest.raises(
            gadget.GadgetError, match="over 7 data qubits but the enclosing gadget uses 6"
        ):

            @gadget.define(action=gadget.Action.idle(), code=code, phases=(base, merged))
            def body(handle):
                handle, _ = wide(handle)
                return handle


class TestUnroll:
    """Tests for ``unroll``, which expands a loop at trace time."""

    def test_unrolls(self, steane_mem):
        """Test that every iteration contributes its own record block."""
        code, phases, memory = steane_mem

        @gadget.define(action=gadget.Action.idle(), code=code, phases=phases)
        def hold(handle):
            return gadget.unroll(3, handle, lambda b: memory(b)[0])

        assert [r.name for r in hold.records] == [f"steane_memory#{i}/mem" for i in range(3)]
        assert hold.program.total_rounds == 9

    @pytest.mark.parametrize("count", [0, 2.0])
    def test_count_must_be_positive_int(self, count):
        """Test that the unroll count is a positive Python int."""
        with pytest.raises(gadget.GadgetError, match="unroll: count must be a positive Python int"):
            gadget.unroll(count, None, lambda b: b)

    def test_body_ops_are_ordinary_ops(self, steane_mem):
        """Test that a repeated body is indistinguishable from writing it out by hand."""
        code, phases, _ = steane_mem
        names = iter(("r0", "r1"))

        @_idle(code, phases)
        def unrolled(handle):
            return gadget.unroll(2, handle, lambda b: gadget.rounds(b, 1, record=next(names))[0])

        assert not any(isinstance(op, Observe) for op in unrolled.program.ops)
        assert [r.name for r in unrolled.records] == ["r0", "r1"]
