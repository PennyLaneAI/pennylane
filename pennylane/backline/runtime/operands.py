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

Each C parameter becomes one array operand, in declaration order, and the compiler passes it as a
pointer to its data:

* a scalar is a one-element array of its own type
* a ``str`` is its NUL-terminated bytes, padded to ``STR_OPERAND_BYTES``
* a ``buf`` is the whole array, and only works for a local call
* an ``out`` buffer is not an operand at all - the compiler allocates it and it comes back as a
  result

A call returns the declared result first, then one buffer per ``out`` parameter.
"""

from __future__ import annotations

import numpy as np

from .signature import CType

SCALAR_SHAPE = (1,)

# Fixed-width NUL-padded field for a ``str`` argument. Must match
# CATALYST_TRANSPORT_STR_BYTES in runtime/include/TransportABI.h.
STR_OPERAND_BYTES = 256


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
    # pylint: disable=import-outside-toplevel
    import jax

    if not jax.config.jax_enable_x64:
        raise TypeError(
            f"{symbol}: {what} is a {ctype}, which JAX would narrow to 32 bits because "
            f"jax_enable_x64 is off. Turn it on with "
            f"jax.config.update('jax_enable_x64', True), as Catalyst does."
        )


def _is_tracer(value) -> bool:
    """Whether a value only exists while the program is being traced."""
    # pylint: disable=import-outside-toplevel
    try:
        import jax
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, jax.core.Tracer)


def text_bytes(ctype: CType, value, symbol: str, position: int) -> bytes:
    """The bytes of a ``str`` argument, NUL-padded to ``STR_OPERAND_BYTES``.

    Args:
        ctype (CType): the parameter type, used only for the error message
        value (str | bytes): the string
        symbol (str): the entry point being called, for the error message
        position (int): the argument's position, for the error message

    Returns:
        bytes: ``STR_OPERAND_BYTES`` bytes, the string followed by NULs

    Raises:
        TypeError: if the value is not known yet, or is not a string
        ValueError: if the string does not fit its field
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
    else:
        raise TypeError(f"{symbol}: argument {position} is a {ctype}, got {type(value).__name__}")

    if len(raw) >= STR_OPERAND_BYTES:
        raise ValueError(
            f"{symbol}: argument {position} is {len(raw)} bytes, which does not fit a {ctype}'s "
            f"{STR_OPERAND_BYTES}-byte field (one byte goes to the NUL terminator)"
        )
    return raw.ljust(STR_OPERAND_BYTES, b"\x00")


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

    if ctype is CType.STR:
        return jnp.frombuffer(text_bytes(ctype, value, symbol, position), dtype=jnp.uint8)
    if ctype is CType.BUF:
        # Local calls only; `operands_for` rejects buf for a dispatched call.
        return jnp.asarray(value)
    if ctype.dtype is None:
        raise TypeError(f"{symbol}: argument {position} of type {ctype} cannot be passed")
    check_width(ctype, symbol, f"argument {position}")
    return jnp.asarray(value, dtype=ctype.dtype).reshape(SCALAR_SHAPE)


def operands_for(signature, args, *, local: bool = False) -> list:
    """Build every operand a recorded call passes.

    Args:
        signature (CSignature): the signature being called
        args (Sequence): the caller's arguments, ``out`` buffers excluded
        local (bool): whether the call is local (in-process). Only a local call may pass a
            ``buf``.

    Returns:
        list: one array per operand, in the order the entry point takes them

    Raises:
        TypeError: if an argument's size does not follow from its type
    """
    signature.check_arity(args)
    if not local:
        for i, ctype in enumerate(signature.caller_params):
            if ctype is CType.BUF:
                raise TypeError(
                    f"{signature.symbol}: argument {i} is a {ctype}, whose length is not implied by "
                    f"its type, so it cannot be read out of the flat buffer a dispatched call "
                    f"arrives in. Declare the data as a fixed-width argument, or call the symbol "
                    f"locally (no address=)."
                )
    return [
        operand_for(ctype, value, signature.symbol, i)
        for i, (ctype, value) in enumerate(zip(signature.caller_params, args, strict=True))
    ]


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
