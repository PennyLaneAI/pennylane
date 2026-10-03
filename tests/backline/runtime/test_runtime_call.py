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

"""Tests for :mod:`pennylane.backline.runtime`.

Declared symbols land in a process-wide registry that is never cleared, so every symbol declared
here has a name of its own.
"""

# pylint: disable=too-few-public-methods

import numpy as np
import pytest

import pennylane as qp
from pennylane.backline.runtime import CSignature, CType, operands


@pytest.fixture(name="x64")
def x64_fixture():
    """Run a test with 64-bit values available, as Catalyst configures JAX."""
    jax = pytest.importorskip("jax")
    with jax.experimental.enable_x64():
        yield jax


@pytest.mark.all_interfaces
class TestCSignature:
    """Parsing and inspecting a signature."""

    @pytest.mark.parametrize(
        "spec, params, result",
        [
            ("(ptr, u32) -> i32", (CType.PTR, CType.U32), CType.I32),
            ("()", (), CType.VOID),
            ("(void) -> void", (), CType.VOID),
            ("( str , buf , u64 )->u64", (CType.STR, CType.BUF, CType.U64), CType.U64),
            ("(ptr, out, u64) -> i32", (CType.PTR, CType.OUT, CType.U64), CType.I32),
        ],
    )
    def test_parse(self, spec, params, result):
        """A spec is parsed into its parameter and result types."""
        signature = CSignature.parse("sym", spec)
        assert signature.params == params
        assert signature.result == result

    @pytest.mark.parametrize(
        "spec, message",
        [
            ("(ptr, nope) -> i32", "unknown C type 'nope'"),
            ("ptr -> i32", "malformed signature"),
            ("(ptr) -> str", "not a valid result type"),
            ("(ptr) -> out", "not a valid result type"),
            ("(ptr, void) -> i32", "void is not a valid parameter type"),
        ],
    )
    def test_parse_rejects_a_bad_spec(self, spec, message):
        """A spec that cannot describe a call is rejected where it is written."""
        with pytest.raises(ValueError, match=message):
            CSignature.parse("sym", spec)

    def test_a_signature_is_hashable(self):
        """A signature travels as a primitive parameter, so it has to be hashable."""
        signature = CSignature.parse("sym", "(ptr) -> i32")
        assert {signature: 1}[signature] == 1

    def test_repr_names_the_symbol_and_shows_the_signature(self):
        """The repr carries enough information to identify the symbol and its ABI."""
        signature = CSignature.parse("connect", "(ptr, u32) -> i32")
        assert repr(signature) == "CSignature('connect', (ptr, u32) -> i32)"

    def test_the_wrong_number_of_arguments_is_refused(self):
        """A call with too few arguments is reported, not truncated."""
        signature = CSignature.parse("sym", "(ptr, u32) -> i32")
        with pytest.raises(TypeError, match=r"takes 2 argument\(s\) \(ptr, u32\)"):
            signature.check_arity((1,))

    def test_out_buffers_are_not_caller_arguments(self):
        """A buffer the callee fills is counted separately from the caller's arguments."""
        signature = CSignature.parse("sym", "(ptr, out, u64) -> i32")
        assert signature.caller_params == (CType.PTR, CType.U64)
        assert signature.out_params == (1,)

        signature.check_arity((0, 8))  # the caller passes two, not three
        with pytest.raises(TypeError, match="out buffer"):
            signature.check_arity((0, None, 8))


@pytest.mark.all_interfaces
class TestDeclare:
    """Declaring a symbol's signature."""

    def test_declare_then_look_up(self):
        """A declared signature is retrievable by symbol."""
        signature = qp.runtime_declare("declared_symbol", "(ptr) -> i64")
        assert qp.backline.runtime.signature_of("declared_symbol") is signature
        assert "declared_symbol" in qp.backline.runtime.declared_symbols()

    def test_redeclaring_the_same_signature_is_allowed(self):
        """The same declaration twice is bookkeeping, not a conflict."""
        first = qp.runtime_declare("redeclared_symbol", "(ptr) -> i64")
        assert qp.runtime_declare("redeclared_symbol", "(ptr) -> i64") == first

    def test_declare_records_the_library(self):
        """A local symbol's backing library is carried on its signature."""
        signature = qp.runtime_declare("xor_reduce", "(buf, u64) -> i32", library="/opt/libx.so")
        assert signature.library == "/opt/libx.so"

    def test_declare_without_a_library(self):
        """Without one, the library is None (a dispatched or already-loaded symbol)."""
        assert qp.runtime_declare("unbound_symbol", "(ptr) -> i64").library is None

    def test_conflicting_libraries_are_refused(self):
        """Two declarations putting the same symbol in different libraries disagree."""
        qp.runtime_declare("clashing_library", "(buf, u64) -> i32", library="/opt/a.so")
        with pytest.raises(ValueError, match="already declared"):
            qp.runtime_declare("clashing_library", "(buf, u64) -> i32", library="/opt/b.so")

    def test_conflicting_signatures_are_refused(self):
        """Two call sites disagreeing on an ABI is a bug, so it is refused."""
        qp.runtime_declare("clashing_result", "(ptr) -> i64")
        with pytest.raises(ValueError, match="already declared"):
            qp.runtime_declare("clashing_result", "(ptr) -> i32")

    def test_signature_of_an_undeclared_symbol_points_at_runtime_declare(self):
        """Looking up a symbol that was never declared raises with an actionable message."""
        with pytest.raises(KeyError, match="has no declared signature"):
            qp.backline.runtime.signature_of("never_declared_symbol")


@pytest.mark.all_interfaces
class TestRecordedCalls:
    """Recording a dispatched call, and the operands it becomes."""

    @pytest.mark.usefixtures("x64")
    def test_one_operand_per_parameter(self):
        """Each argument becomes its own operand, shaped the way the callee reads it."""
        signature = CSignature.parse("connect", "(ptr, str, u16) -> i32")
        built = operands.operands_for(signature, (0x7FAB1234, "10.0.0.1", 18560))

        assert [tuple(o.shape) for o in built] == [(1,), (operands.STR_OPERAND_BYTES,), (1,)]
        assert [str(o.dtype) for o in built] == ["uint64", "uint8", "uint16"]
        assert int(built[0][0]) == 0x7FAB1234

    @pytest.mark.usefixtures("x64")
    def test_a_string_is_padded_to_a_fixed_field(self):
        """A str is padded to a fixed width, so a flat buffer needs no framing to delimit it."""
        signature = CSignature.parse("connect", "(ptr, str, u16) -> i32")
        built = operands.operands_for(signature, (0, "10.0.0.1", 0))
        field = bytes(np.asarray(built[1]))

        assert len(field) == operands.STR_OPERAND_BYTES
        assert field.startswith(b"10.0.0.1\x00")
        assert field == b"10.0.0.1".ljust(operands.STR_OPERAND_BYTES, b"\x00")

    @pytest.mark.usefixtures("x64")
    def test_a_string_that_does_not_fit_is_refused(self):
        """A fixed field means there is a limit, and it is said rather than truncated."""
        signature = CSignature.parse("connect", "(ptr, str, u16) -> i32")
        with pytest.raises(ValueError, match="does not fit"):
            operands.operands_for(signature, (0, "x" * operands.STR_OPERAND_BYTES, 0))

    @pytest.mark.usefixtures("x64")
    def test_a_buffer_cannot_be_dispatched(self):
        """A buf's length is not implied by its type, so it cannot cross in a flat buffer."""
        signature = CSignature.parse("write_bytes", "(ptr, buf, u64) -> i32")
        with pytest.raises(TypeError, match="cannot be read out of the flat buffer"):
            operands.operands_for(signature, (0, np.arange(4, dtype=np.uint8), 4))

    @pytest.mark.usefixtures("x64")
    def test_an_out_buffer_comes_back_as_a_result(self):
        """A buffer the callee fills is returned, so it is never an operand."""
        signature = CSignature.parse("read_bytes", "(ptr, out, u64) -> i32")
        assert len(operands.operands_for(signature, (1, 4))) == 2

        avals = operands.result_avals(signature, 4)
        assert [tuple(a.shape) for a in avals] == [(1,), (4,)]
        assert [str(a.dtype) for a in avals] == ["int32", "uint8"]

    def test_a_traced_string_is_refused(self, x64):
        """A str becomes a constant in the program, so a traced value cannot be one."""
        signature = CSignature.parse("connect", "(ptr, str, u16) -> i32")

        def program(peer):
            return operands.operands_for(signature, (0, peer, 1))

        with pytest.raises(TypeError, match="has to be a Python string"):
            x64.make_jaxpr(program)(np.uint8(3))

    def test_a_64_bit_value_needs_x64(self):
        """A narrowed pointer would be a different address, so it is refused, not warned about."""
        jax = pytest.importorskip("jax")
        signature = CSignature.parse("some_symbol", "(ptr, u32) -> i32")
        with jax.experimental.disable_x64():
            with pytest.raises(TypeError, match="jax_enable_x64 is off"):
                operands.operands_for(signature, (0x7FAB1234, 0))

    def test_a_dispatched_call_records_its_symbol_and_address(self, x64):
        """The recorded call names the C symbol itself, and where to run it."""
        qp.runtime_declare("run_rounds", "(ptr, u64) -> i32")

        jaxpr = x64.make_jaxpr(lambda h: qp.runtime_call("run_rounds", h, 4, address="h:1"))(
            np.uint64(7)
        )
        calls = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]
        assert len(calls) == 1
        assert calls[0].params["symbol"] == "run_rounds"
        assert calls[0].params["dispatch"] == "h:1"

    def test_out_bytes_adds_a_returned_buffer(self, x64):
        """An out buffer is returned alongside the declared result."""
        qp.runtime_declare("read_counters", "(out, u64) -> i32")

        jaxpr = x64.make_jaxpr(
            lambda: qp.runtime_call("read_counters", 8, out_bytes=8, address="h:1")
        )()
        avals = [v.aval for v in jaxpr.jaxpr.outvars]
        assert [tuple(a.shape) for a in avals] == [(), (8,)]

    def test_several_out_buffers_come_back_in_order(self, x64):
        """Returns several out buffers in the order they are declared."""
        qp.runtime_declare("read_two_regions", "(out, out, u64) -> i32")

        jaxpr = x64.make_jaxpr(
            lambda: qp.runtime_call("read_two_regions", 96, out_bytes=(32, 64), address="h:1")
        )()
        avals = [v.aval for v in jaxpr.jaxpr.outvars]

        assert [tuple(a.shape) for a in avals] == [(), (32,), (64,)]
        assert [str(a.dtype) for a in avals] == ["int32", "uint8", "uint8"]

    @pytest.mark.parametrize("sizes", [32, (32,), (32, 64, 8)])
    def test_one_size_per_out_buffer_is_required(self, sizes):
        """The number of sizes must match the number of out buffers."""
        signature = CSignature.parse("read_two_regions", "(out, out, u64) -> i32")
        with pytest.raises(ValueError, match=r"writes 2 out buffer\(s\)"):
            operands.out_sizes(signature, sizes)

    def test_the_library_path_is_not_validated(self, x64):
        """The path is data for the linker. A cross-compiled program's library only exists on the
        target, so checking it here would reject the case the flow is built for."""
        qp.runtime_declare("board_kernel", "(u64) -> i32")
        jaxpr = x64.make_jaxpr(
            lambda: qp.runtime_call("board_kernel", 1, library="/on/the/board/libx.so")
        )()
        call = [e for e in jaxpr.eqns if str(e.primitive) == "runtime_call"][0]
        assert call.params["library"] == "/on/the/board/libx.so"

    def test_out_bytes_cannot_be_traced(self, x64):
        """The compiler allocates the buffer, so a value computed while running cannot size it."""
        qp.runtime_declare("sized_reader", "(out, u64) -> i32")
        with pytest.raises(TypeError, match="known when the program is compiled"):
            x64.make_jaxpr(
                lambda n: qp.runtime_call("sized_reader", n, out_bytes=n, address="h:1")
            )(np.uint64(8))

    def test_a_dispatched_void_call_is_refused(self, x64):
        """A dispatched call the program gets nothing back from is refused."""
        qp.runtime_declare("fire_and_forget", "(u64)")
        with pytest.raises(TypeError, match="has no result"):
            x64.make_jaxpr(lambda: qp.runtime_call("fire_and_forget", 0, address="h:1"))()


@pytest.mark.all_interfaces
class TestLocalCalls:
    """A call with no address, invoked in-process instead of dispatched to an executor."""

    @pytest.mark.usefixtures("x64")
    def test_a_local_call_can_pass_a_buffer(self):
        """A buf is refused for a dispatched call but fine locally: it crosses as its own pointer."""
        signature = CSignature.parse("sum_bytes", "(buf, u64) -> i32")
        built = operands.operands_for(signature, (np.arange(4, dtype=np.uint8), 4), local=True)
        assert len(built) == 2
        assert tuple(np.asarray(built[0]).shape) == (4,)

    @pytest.mark.usefixtures("x64")
    def test_a_64_bit_buffer_keeps_its_width(self):
        """The buffer's width is preserved."""
        signature = CSignature.parse("sum_doubles", "(buf, u32) -> i32")
        built = operands.operands_for(signature, (np.arange(4, dtype=np.float64), 32), local=True)
        buffer = built[0]

        assert buffer.dtype == np.float64
        assert buffer.size * buffer.dtype.itemsize == 32

    @pytest.mark.parametrize("array", [np.arange(4, dtype=np.uint64), [1.0, 2.0, 3.0]])
    def test_a_64_bit_buffer_needs_x64(self, array):
        """64-bit buffers are refused if JAX would narrow them to 32 bits."""
        jax = pytest.importorskip("jax")
        signature = CSignature.parse("sum_bytes_only", "(buf, u32) -> i32")
        with jax.experimental.disable_x64():
            with pytest.raises(TypeError, match="is a buf of .*narrowed to 32 bits"):
                operands.operands_for(signature, (array, 32), local=True)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.float32])
    def test_a_narrow_buffer_is_unaffected(self, dtype):
        """32-bit buffers are accepted if JAX would narrow them to 32 bits."""
        jax = pytest.importorskip("jax")
        signature = CSignature.parse("sum_narrow", "(buf, u32) -> i32")
        with jax.experimental.disable_x64():
            built = operands.operands_for(signature, (np.arange(4, dtype=dtype), 4), local=True)

        assert built[0].dtype == dtype

    def test_a_local_call_is_not_dispatched(self, x64):
        """With no address the recorded call carries dispatch=None and the declared library."""
        qp.runtime_declare("local_xor", "(buf, u64) -> i32", library="/opt/libx.so")
        jaxpr = x64.make_jaxpr(lambda d: qp.runtime_call("local_xor", d, 4))(
            np.arange(4, dtype=np.uint8)
        )
        calls = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]
        assert len(calls) == 1
        assert calls[0].params["dispatch"] is None
        assert calls[0].params["library"] == "/opt/libx.so"

    def test_the_call_site_library_wins(self, x64):
        """library= at the call site overrides the one set at declare time."""
        qp.runtime_declare("local_override", "(buf, u64) -> i32", library="/opt/declared.so")
        jaxpr = x64.make_jaxpr(
            lambda d: qp.runtime_call("local_override", d, 4, library="/opt/called.so")
        )(np.arange(4, dtype=np.uint8))
        calls = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]
        assert calls[0].params["library"] == "/opt/called.so"

    def test_a_local_void_call_is_allowed(self, x64):
        """A void symbol has no result, but a local call to it is kept, not refused."""
        qp.runtime_declare("local_reset", "(u64)", library="/opt/libx.so")

        def program():
            qp.runtime_call("local_reset", 0)  # returns None; must not raise
            return np.int32(0)

        jaxpr = x64.make_jaxpr(program)()
        calls = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]
        assert len(calls) == 1
        assert calls[0].params["dispatch"] is None


@pytest.mark.all_interfaces
class TestResolveSignature:
    """Which signature :func:`~.runtime_call` calls a symbol with."""

    def test_a_csignature_target_needs_no_signature_kwarg(self, x64):
        """Passing the signature as the target is enough to record the call."""
        signature = CSignature.parse("resolved_by_target", "(u64) -> i32")
        jaxpr = x64.make_jaxpr(lambda: qp.runtime_call(signature, 1, address="h:1"))()
        calls = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]
        assert calls[0].params["symbol"] == "resolved_by_target"

    def test_both_target_and_signature_is_refused(self):
        """Two ways to say the same thing invite disagreement, so both is refused."""
        pytest.importorskip("jax")
        signature = CSignature.parse("clashing_target", "(u64) -> i32")
        with pytest.raises(ValueError, match="either a CSignature or signature="):
            qp.runtime_call(signature, 1, signature=signature, address="h:1")

    def test_a_csignature_kwarg_is_used_as_given(self, x64):
        """A signature= kwarg with a CSignature is used verbatim; no declaration is inferred."""
        signature = CSignature.parse("resolved_by_kwarg", "(u64) -> i32")

        def program():
            return qp.runtime_call("resolved_by_kwarg", 1, signature=signature, address="h:1")

        jaxpr = x64.make_jaxpr(program)()
        calls = [eqn for eqn in jaxpr.eqns if str(eqn.primitive) == "runtime_call"]
        assert calls[0].params["signature"] is signature

    def test_a_signature_spec_string_is_declared_on_first_call(self, x64):
        """A signature= kwarg as a string declares the symbol and uses that declaration."""

        def program():
            return qp.runtime_call("declared_via_call", 1, signature="(u64) -> i32", address="h:1")

        x64.make_jaxpr(program)()
        assert qp.backline.runtime.signature_of("declared_via_call").symbol == "declared_via_call"


@pytest.mark.all_interfaces
class TestOutsideATrace:
    """A recorded call belongs in a compiled program."""

    @pytest.mark.parametrize(
        ("symbol", "declare_kwargs", "call_kwargs"),
        [
            ("outside_trace_local", {"library": "/opt/libx.so"}, {}),
            ("outside_trace_dispatched", {}, {"address": "h:1"}),
        ],
    )
    def test_a_call_outside_a_trace_is_refused(self, symbol, declare_kwargs, call_kwargs):
        """Local and dispatched calls both refuse eagerly outside a JAX trace."""
        pytest.importorskip("jax")
        qp.runtime_declare(symbol, "(u64) -> i32", **declare_kwargs)
        with pytest.raises(RuntimeError, match="outside a compiled program"):
            qp.runtime_call(symbol, 0, **call_kwargs)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
