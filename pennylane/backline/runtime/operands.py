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

"""How a recorded :func:`~.runtime_call`'s arguments reach the entry point.

Local and dispatched calls are recorded the same way: scalar operands stay scalar, ``str``
becomes a compile-time constant rather than an operand, and a ``buf`` or ``out`` stays an array.
The compiler decides what that means. Locally, ``ptr`` is lowered to ``!llvm.ptr`` and a
``buf``/``out`` is bufferized only so its data pointer can be passed, so the external function
receives plain C arguments rather than a memref descriptor or wrapper ABI. For a dispatched call
the compiler marshals the same operands into the executor's flat transport buffer.
"""

from __future__ import annotations

import numpy as np

from .signature import CType

SCALAR_SHAPE = (1,)

# Width of the fixed NUL-padded field a dispatched ``str`` argument travels in. The compiler
# does the padding; this is here to reject an oversized string while the program is still being
# traced. Must match CATALYST_TRANSPORT_STR_BYTES in runtime/include/TransportABI.h.
STR_OPERAND_BYTES = 256


def _narrows_64_bit() -> bool:
    """Check whether JAX would narrow 64-bit values to 32-bit."""
    # pylint: disable=import-outside-toplevel
    import jax

    return not jax.config.jax_enable_x64


def check_width(ctype: CType, symbol: str, what: str) -> None:
    """Refuse a 64-bit value that JAX would quietly narrow to 32 bits.

    Args:
        ctype (CType): the type being passed or returned
        symbol (str): the entry point, for the error message
        what (str): what is being described, for the error message

    Raises:
        TypeError: if the type needs 64 bits and JAX is configured without them
    """
    dtype = ctype.dtype
    if dtype is None or dtype.itemsize < 8:
        return
    if _narrows_64_bit():
        raise TypeError(
            f"{symbol}: {what} is a {ctype}, which JAX would narrow to 32 bits because "
            f"jax_enable_x64 is off. Turn it on with "
            f"jax.config.update('jax_enable_x64', True), as Catalyst does."
        )


def check_buffer_width(value, symbol: str, position: int) -> None:
    """Refuse a ``buf`` whose elements would be narrowed by JAX to 32 bits.

    Args:
        value: the buffer being passed
        symbol (str): the symbol being called
        position (int): the argument's position

    Raises:
        TypeError: if the buffer's elements would be narrowed by JAX to 32 bits
    """
    dtype = getattr(value, "dtype", None)
    dtype = np.dtype(dtype) if dtype is not None else np.asarray(value).dtype
    if dtype.itemsize < 8:
        return
    if _narrows_64_bit():
        raise TypeError(
            f"{symbol}: argument {position} is a buf of {dtype}, whose elements would be narrowed "
            f"to 32 bits by JAX because jax_enable_x64 is off."
        )


def _is_tracer(value) -> bool:
    """Whether a value only exists while the program is being traced."""
    # pylint: disable=import-outside-toplevel
    try:
        import jax
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, jax.core.Tracer)


def c_string_bytes(
    ctype: CType, value, symbol: str, position: int, *, max_bytes: int | None = None
) -> bytes:
    """Encode one compile-time ``str`` argument as NUL-terminated bytes.

    Args:
        ctype (CType): the parameter type, used only for error messages
        value (str | bytes): the string
        symbol (str): the entry point being called, for error messages
        position (int): the argument's position, for error messages
        max_bytes (int | None): maximum width of the fixed field the string passes through

    Returns:
        bytes: the string followed by a single NUL terminator

    Raises:
        TypeError: if the value is not known yet, or is not a string
        ValueError: if the string is not valid UTF-8, contains a NUL, or does not fit its field
    """
    if _is_tracer(value):
        raise TypeError(
            f"{symbol}: argument {position} is a {ctype} and becomes a constant in the compiled "
            f"program, so it has to be a Python string, not a traced value"
        )
    if isinstance(value, str):
        raw = value.encode()
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        try:
            raw.decode()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{symbol}: argument {position} contains bytes that are not valid UTF-8"
            ) from exc
    else:
        raise TypeError(
            f"{symbol}: argument {position} is a {ctype}, got {type(value).__name__}"
        )
    if b"\x00" in raw:
        raise ValueError(f"{symbol}: argument {position} contains an embedded NUL byte")
    if max_bytes is not None and len(raw) >= max_bytes:
        raise ValueError(
            f"{symbol}: argument {position} is {len(raw)} bytes, which does not fit a {ctype}'s "
            f"{max_bytes}-byte field (one byte goes to the NUL terminator)"
        )
    return raw + b"\x00"


def operand_for(ctype: CType, value, symbol: str, position: int):
    """Build the operand one C parameter is passed as.

    Args:
        ctype (CType): the parameter type
        value: the argument
        symbol (str): the entry point being called, for error messages
        position (int): the argument's position, for error messages

    Returns:
        A ``jax`` array holding the argument as the entry point will read it
    """
    # pylint: disable=import-outside-toplevel
    import jax.numpy as jnp

    if ctype is CType.BUF:
        # Local calls only; `operands_for` rejects buf for a dispatched call.
        check_buffer_width(value, symbol, position)
        return jnp.asarray(value)
    if ctype.dtype is None:
        raise TypeError(f"{symbol}: argument {position} of type {ctype} cannot be passed")
    check_width(ctype, symbol, f"argument {position}")
    return jnp.asarray(value, dtype=ctype.dtype).reshape(SCALAR_SHAPE)


def operands_for(
    signature, args, *, dispatched: bool = False
) -> tuple[list, tuple[bytes, ...]]:
    """Build every operand a recorded call passes.

    Args:
        signature (CSignature): the signature being called
        args (Sequence): the caller's arguments, ``out`` buffers excluded
        dispatched (bool): whether the call is addressed to an executor

    Returns:
        tuple: the dynamic operands

    Raises:
        TypeError: if an argument cannot be passed the way its type requires
        ValueError: if a compile-time string is not usable
    """
    signature.check_arity(args)

    dynamic = []
    strings = []
    for position, (ctype, value) in enumerate(
        zip(signature.caller_params, args, strict=True)
    ):
        if ctype is CType.BUF and dispatched:
            raise TypeError(
                f"{signature.symbol}: argument {position} is a {ctype}, which cannot be read "
                f"out of the flat buffer"
            )
        if ctype is CType.STR:
            strings.append(
                c_string_bytes(
                    ctype,
                    value,
                    signature.symbol,
                    position,
                    max_bytes=STR_OPERAND_BYTES if dispatched else None,
                )
            )
        else:
            dynamic.append(operand_for(ctype, value, signature.symbol, position))
    return dynamic, tuple(strings)


def out_sizes(signature, out_bytes) -> tuple[int, ...]:
    """How big each ``out`` buffer of a call is.

    Args:
        signature (CSignature): the signature being called
        out_bytes (int | Sequence[int]): the size of the one ``out`` buffer, or one size per
            buffer for a signature declaring several

    Returns:
        tuple[int]: one size per ``out`` parameter

    Raises:
        ValueError: if the sizes do not account for the declared buffers
    """
    wanted = len(signature.out_params)

    if isinstance(out_bytes, int):
        sizes = (out_bytes,)
    else:
        try:
            sizes = tuple(out_bytes)
        except TypeError:
            sizes = (out_bytes,)
    for size in sizes:
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
            raise TypeError(
                f"{signature.symbol}: out_bytes must be a size known when the program is compiled, "
                f"not a {type(size).__name__} computed while it runs"
            )
    sizes = tuple(int(size) for size in sizes)

    if not wanted:
        if any(size for size in sizes):
            raise ValueError(
                f"{signature.symbol}{signature} writes no out buffer, so out_bytes does not "
                f"apply to it"
            )
        return ()

    if len(sizes) != wanted:
        raise ValueError(
            f"{signature.symbol}{signature} writes {wanted} out buffer(s); out_bytes gave "
            f"{len(sizes)} size(s)"
        )
    for size in sizes:
        if int(size) <= 0:
            raise ValueError(
                f"{signature.symbol}{signature} writes an out buffer, so out_bytes must say how "
                f"big it is; got {size}"
            )
    return tuple(int(size) for size in sizes)


def result_avals(signature, out_bytes):
    """The shapes a recorded call returns: the declared result, then each ``out`` buffer.

    Args:
        signature (CSignature): the signature being called
        out_bytes (int | Sequence[int]): the size of each ``out`` buffer

    Returns:
        tuple: one ``jax.core.ShapedArray`` per returned value
    """
    # pylint: disable=import-outside-toplevel
    import jax

    avals = []
    if signature.result is not CType.VOID:
        check_width(signature.result, signature.symbol, "the result")
        avals.append(jax.core.ShapedArray(SCALAR_SHAPE, signature.result.dtype))
    for size in out_sizes(signature, out_bytes):
        avals.append(jax.core.ShapedArray((size,), np.dtype(np.uint8)))
    return tuple(avals)
