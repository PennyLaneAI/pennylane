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

"""Edge-case tests for :mod:`pennylane.backline.runtime.operands`.

The happy paths are covered by ``test_runtime_call.py``; this file targets the branches that
only fire on odd inputs and would otherwise stay uncovered.
"""

# pylint: disable=too-few-public-methods

import numpy as np
import pytest

from pennylane.backline.runtime import CSignature, CType, operands


@pytest.fixture(name="x64")
def x64_fixture():
    """Run with 64-bit values available, as Catalyst configures JAX."""
    jax = pytest.importorskip("jax")
    with jax.experimental.enable_x64():
        yield jax


class TestTextBytes:
    """Building the fixed-width byte field for a ``str`` argument."""

    @pytest.mark.usefixtures("x64")
    def test_bytes_input_is_accepted(self):
        """A ``bytes`` value is padded like a ``str``."""
        raw = operands.text_bytes(CType.STR, b"hello", "sym", 0)
        assert raw == b"hello".ljust(operands.STR_OPERAND_BYTES, b"\x00")

    @pytest.mark.usefixtures("x64")
    def test_bytearray_input_is_accepted(self):
        """A ``bytearray`` value is copied to bytes and padded."""
        raw = operands.text_bytes(CType.STR, bytearray(b"hello"), "sym", 0)
        assert raw == b"hello".ljust(operands.STR_OPERAND_BYTES, b"\x00")

    @pytest.mark.usefixtures("x64")
    def test_wrong_type_is_refused(self):
        """A non-string, non-bytes value is refused rather than coerced."""
        with pytest.raises(TypeError, match=r"argument 0 is a str, got int"):
            operands.text_bytes(CType.STR, 42, "sym", 0)


class TestOperandFor:
    """Building the single-parameter operand."""

    @pytest.mark.usefixtures("x64")
    def test_a_type_with_no_dtype_is_refused(self):
        """VOID has no dtype, so ``operand_for`` cannot build an operand for it."""
        with pytest.raises(TypeError, match=r"argument 3 of type void cannot be passed"):
            operands.operand_for(CType.VOID, 0, "sym", 3)


class TestOutSizes:
    """The size table for a call's ``out`` buffers."""

    def _sig(self, spec):
        return CSignature.parse("sym", spec)

    def test_a_non_int_scalar_is_refused(self):
        """A float, being neither an int nor a valid iterable of sizes, is refused."""
        signature = self._sig("(out, u64) -> i32")
        with pytest.raises(TypeError, match="must be a size known when the program is compiled"):
            operands.out_sizes(signature, 3.5)

    def test_a_bool_is_refused(self):
        """``bool`` is a subclass of ``int`` but does not describe a byte count."""
        signature = self._sig("(out, u64) -> i32")
        with pytest.raises(TypeError, match="must be a size known when the program is compiled"):
            operands.out_sizes(signature, True)

    def test_a_non_int_in_a_sequence_is_refused(self):
        """A float inside the size tuple is refused too."""
        signature = self._sig("(out, out, u64) -> i32")
        with pytest.raises(TypeError, match="must be a size known when the program is compiled"):
            operands.out_sizes(signature, (8, 3.5))

    def test_a_numpy_integer_scalar_is_accepted(self):
        """A ``numpy.integer`` scalar is a size, and is coerced to a Python int."""
        signature = self._sig("(out, u64) -> i32")
        (size,) = operands.out_sizes(signature, np.uint32(16))
        assert isinstance(size, int)
        assert size == 16

    def test_a_zero_out_signature_with_nonzero_out_bytes_is_refused(self):
        """A signature that writes no buffer has no ``out_bytes`` to consume."""
        signature = self._sig("(u64) -> i32")
        with pytest.raises(ValueError, match="writes no out buffer"):
            operands.out_sizes(signature, 16)

    def test_a_zero_out_signature_with_zero_out_bytes_is_ok(self):
        """A signature that writes no buffer accepts a bookkeeping ``out_bytes=0``."""
        signature = self._sig("(u64) -> i32")
        assert operands.out_sizes(signature, 0) == ()

    def test_a_zero_size_is_refused(self):
        """A buffer the callee fills has to be at least one byte."""
        signature = self._sig("(out, u64) -> i32")
        with pytest.raises(ValueError, match="must say how big it is"):
            operands.out_sizes(signature, 0)

    def test_a_negative_size_is_refused(self):
        """A negative size cannot describe a byte count either."""
        signature = self._sig("(out, u64) -> i32")
        with pytest.raises(ValueError, match="must say how big it is"):
            operands.out_sizes(signature, -4)


class TestResultAvals:
    """The shapes a call returns."""

    @pytest.mark.usefixtures("x64")
    def test_a_void_signature_returns_only_out_buffers(self):
        """A void-returning signature has no scalar aval, only the buffer avals."""
        signature = CSignature.parse("sym", "(out, u64)")
        avals = operands.result_avals(signature, 4)
        assert [tuple(a.shape) for a in avals] == [(4,)]
        assert [str(a.dtype) for a in avals] == ["uint8"]

    @pytest.mark.usefixtures("x64")
    def test_a_void_signature_with_no_out_returns_nothing(self):
        """A void, no-out signature has nothing to describe."""
        signature = CSignature.parse("sym", "(u64)")
        assert operands.result_avals(signature, 0) == ()


class TestCheckBufferWidth:
    """The 64-bit narrowing check for buf arguments."""

    def test_a_python_list_is_measured_via_asarray(self):
        """A list has no ``.dtype``, so its dtype is measured by wrapping it in an ndarray."""
        jax = pytest.importorskip("jax")
        with jax.experimental.disable_x64():
            # A list of Python ints becomes int64 on 64-bit platforms, which is narrowed.
            with pytest.raises(TypeError, match="narrowed to 32 bits"):
                operands.check_buffer_width([1, 2, 3], "sym", 0)

    def test_a_narrow_scalar_needs_no_x64(self):
        """A value whose dtype is already 32-bit passes without ``x64``."""
        jax = pytest.importorskip("jax")
        with jax.experimental.disable_x64():
            # Does not raise; nothing to narrow.
            operands.check_buffer_width(np.zeros(4, dtype=np.uint8), "sym", 0)
