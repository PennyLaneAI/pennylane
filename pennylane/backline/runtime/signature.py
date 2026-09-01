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

"""C types and signatures for runtime symbols."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


class CType(Enum):
    """A C type in a :class:`CSignature`.

    Types can be specified in two ways. When constructing a :class:`CSignature` directly, pass
    uppercase enum members such as ``CType.PTR`` and ``CType.U32``. When declaring a symbol with
    :func:`~pennylane.runtime_declare`, write their lowercase values in a string signature:

    >>> import pennylane as qp
    >>> from pennylane.backline.runtime import CSignature, CType
    >>> CType.U32.value
    'u32'
    >>> direct = CSignature("example_run", (CType.PTR, CType.U32), result=CType.I32)
    >>> declared = qp.runtime_declare("example_run", "(ptr, u32) -> i32")
    >>> direct == declared
    True

    In ``"(ptr, u32) -> i32"``, the parentheses contain the parameter types in order: ``ptr``
    followed by ``u32``. The arrow separates those parameters from the ``i32`` result type.

    The supported types are:

    .. list-table::
       :header-rows: 1
       :widths: 24 20 56

       * - Enum member
         - String token
         - Use
       * - ``VOID``
         - ``"void"``
         - No scalar result. It is valid only as a result type; ``(void)`` also denotes no
           parameters.
       * - ``I1``
         - ``"i1"``
         - A Python, NumPy, or JAX boolean scalar.
       * - ``I8``, ``I16``, ``I32``, ``I64``
         - ``"i8"``, ``"i16"``, ``"i32"``, ``"i64"``
         - A signed integer scalar of the stated width.
       * - ``U8``, ``U16``, ``U32``, ``U64``
         - ``"u8"``, ``"u16"``, ``"u32"``, ``"u64"``
         - An unsigned integer scalar of the stated width.
       * - ``F32``, ``F64``
         - ``"f32"``, ``"f64"``
         - A floating-point scalar of the stated width.
       * - ``PTR``
         - ``"ptr"``
         - An integer address the runtime hands out and takes back, such as a session handle.
       * - ``STR``
         - ``"str"``
         - A Python string or bytes-like value known at trace time. It becomes a NUL-terminated
           constant in the compiled module.
       * - ``BUF``
         - ``"buf"``
         - Array data for a local call. Its length is not implied; declare a separate scalar
           parameter when the entry point needs a byte count.
       * - ``OUT``
         - ``"out"``
         - A writable buffer. The caller gives its size with ``out_bytes=`` instead of passing an
           argument, then receives the filled buffer alongside the result.

    The special types change how arguments and results are supplied. For example:

    .. code-block:: python

        qp.runtime_declare("example_log", "(str) -> void")
        qp.runtime_declare("example_transform", "(buf, u64, out) -> i32")

        @qp.qjit
        def program(data):
            qp.runtime_call("example_log", "starting")
            status, output = qp.runtime_call(
                "example_transform", data, data.size, out_bytes=64
            )
            return status, output

    In ``example_log``, ``"starting"`` supplies the trace-time constant ``STR`` parameter and
    ``VOID`` means the call has no scalar result. In ``example_transform``, ``data`` supplies the
    local ``BUF`` and ``data.size`` supplies its explicit element count. The caller passes no
    argument for ``OUT``; ``out_bytes=64`` allocates it, and the filled buffer is returned as
    ``output`` after the ``I32`` status result.

    .. note::

        The 64-bit types like :attr:`PTR`, :attr:`I64`, :attr:`U64`, :attr:`F64`, and a
        :attr:`BUF` of 64-bit elements need ``jax_enable_x64``, which Catalyst turns on when it is
        imported.
    """

    VOID = "void"
    I1 = "i1"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    F32 = "f32"
    F64 = "f64"
    PTR = "ptr"
    STR = "str"
    BUF = "buf"
    OUT = "out"

    def __str__(self) -> str:
        return self.value

    @property
    def dtype(self) -> np.dtype | None:
        """The numpy dtype a value of this type has in a traced program.

        ``None`` for the types that never show up as a traced scalar (:attr:`VOID`, :attr:`STR`,
        :attr:`BUF`).
        """
        return _DTYPES.get(self)

    @property
    def is_constant(self) -> bool:
        """Whether this type has to be a compile-time constant."""
        return self is CType.STR

    @property
    def is_buffer(self) -> bool:
        """Whether this type is passed as an array's data pointer."""
        return self in (CType.BUF, CType.OUT)

    @property
    def is_out(self) -> bool:
        """Whether the entry point writes this argument instead of reading it."""
        return self is CType.OUT


_DTYPES = {
    CType.I1: np.dtype(np.bool_),
    CType.I8: np.dtype(np.int8),
    CType.I16: np.dtype(np.int16),
    CType.I32: np.dtype(np.int32),
    CType.I64: np.dtype(np.int64),
    CType.U8: np.dtype(np.uint8),
    CType.U16: np.dtype(np.uint16),
    CType.U32: np.dtype(np.uint32),
    CType.U64: np.dtype(np.uint64),
    CType.F32: np.dtype(np.float32),
    CType.F64: np.dtype(np.float64),
    CType.PTR: np.dtype(np.uint64),
}


def _parse_type(token: str) -> CType:
    """Parse one type token, listing the known types if it isn't one."""
    try:
        return CType(token.strip())
    except ValueError as exc:
        known = ", ".join(t.value for t in CType)
        raise ValueError(f"unknown C type {token.strip()!r}; expected one of: {known}") from exc


@dataclass(frozen=True)
class CSignature:
    """The C signature of a runtime symbol.

    Args:
        symbol (str): the C symbol name, as exported by the runtime library
        params (tuple[CType]): parameter types, in the order the C entry point takes them
        result (CType): the return type; :attr:`CType.VOID` if the entry point returns nothing
        library (str | None): the shared library exporting this symbol, for a local in-process
            call. It is recorded on the compiled module so the driver links against it. ``None``
            for a dispatched symbol, or one already loaded in the calling process.

    **Example**

    >>> from pennylane.backline.runtime import CSignature, CType
    >>> CSignature("example_run", (CType.PTR, CType.U32), result=CType.I32)
    CSignature('example_run', (ptr, u32) -> i32)

    Alternatively, :meth:`parse` builds the same object from ``"(ptr, u32) -> i32"``.
    """

    symbol: str
    params: tuple[CType, ...]
    result: CType = CType.VOID
    library: str | None = None

    @classmethod
    def parse(cls, symbol: str, spec: str) -> CSignature:
        """Build a signature from a ``"(t1, t2, ...) -> tr"`` spec string.

        The arrow and result are optional; without them the result is ``void``.

        Args:
            symbol (str): the C symbol name
            spec (str): the signature spec, e.g. ``"(ptr, buf, u64) -> i32"``

        Returns:
            CSignature: the parsed signature
        """
        text = spec.strip()
        result = CType.VOID
        if "->" in text:
            text, _, result_text = text.partition("->")
            result = _parse_type(result_text)
            if result.is_constant or result.is_buffer:
                raise ValueError(f"{result} is not a valid result type for {symbol!r}")

        text = text.strip()
        if not (text.startswith("(") and text.endswith(")")):
            raise ValueError(
                f"malformed signature {spec!r} for {symbol!r}: "
                'expected parameters in parentheses, e.g. "(ptr, u32) -> i32"'
            )

        inner = text[1:-1].strip()
        params: tuple[CType, ...] = ()
        if inner and inner != CType.VOID.value:
            params = tuple(_parse_type(token) for token in inner.split(","))
        if any(p is CType.VOID for p in params):
            raise ValueError(f"void is not a valid parameter type in {spec!r}")

        return cls(symbol=symbol, params=params, result=result)

    def __str__(self) -> str:
        args = ", ".join(str(p) for p in self.params)
        return f"({args}) -> {self.result}"

    def __repr__(self) -> str:
        return f"CSignature({self.symbol!r}, {self})"

    @property
    def out_params(self) -> tuple[int, ...]:
        """Positions of the parameters the entry point writes instead of reading."""
        return tuple(i for i, p in enumerate(self.params) if p.is_out)

    @property
    def caller_params(self) -> tuple[CType, ...]:
        """The parameters a caller passes: all of them, minus the out buffers."""
        return tuple(p for p in self.params if not p.is_out)

    def check_arity(self, args) -> None:
        """Raise a :exc:`TypeError` if ``args`` doesn't match :attr:`caller_params`."""
        expected = self.caller_params
        if len(args) != len(expected):
            spelled = ", ".join(str(p) for p in expected)
            extra = (
                ""
                if not self.out_params
                else f" ({len(self.out_params)} out buffer(s) come from out_bytes=)"
            )
            raise TypeError(
                f"{self.symbol}{self} takes {len(expected)} argument(s) ({spelled}){extra}, "
                f"got {len(args)}"
            )


_REGISTRY: dict[str, CSignature] = {}


def declare(symbol: str, spec: str, library: str | None = None) -> CSignature:
    """Declare the signature of a runtime symbol so it can be called by name.

    ``spec`` lists parameter types in runtime order, followed by an optional result. An ``out``
    parameter is declared in ``spec`` but omitted from the arguments to
    :func:`~pennylane.runtime_call`.

    Args:
        symbol (str): the C symbol name, as exported by the runtime library
        spec (str): the signature spec, e.g. ``"(ptr, buf, u64) -> i32"``. See :class:`CType`
            for the available types.
        library (str | None): the shared library (``.so`` / ``.dylib``) exporting ``symbol``, for a
            local in-process call. It is recorded on the compiled module so the driver links
            against it. Leave it ``None`` for a dispatched symbol, or one already loaded in the
            calling process.

    Returns:
        CSignature: the declared signature, which :func:`~pennylane.runtime_call` also takes
            directly

    **Example**

    .. code-block:: python

        import pennylane as qp

        qp.runtime_declare("example_declared_rounds", "(ptr, u32) -> i32")

        # ...and a local symbol, from a shared library the program is linked against:
        qp.runtime_declare("example_local", "(buf, u64) -> i32", library="/path/liblocal.so")

        def program(session, data):
            qp.runtime_call("example_declared_rounds", session, 1000, address="board:9000")
            return qp.runtime_call("example_local", data, data.size)

    .. seealso:: :func:`~pennylane.runtime_call`
    """
    signature = CSignature.parse(symbol, spec)
    if library is not None:
        signature = replace(signature, library=library)
    previous = _REGISTRY.get(symbol)
    if previous is not None and previous != signature:
        raise ValueError(
            f"{symbol!r} is already declared as {previous}, which conflicts with {signature}"
        )
    _REGISTRY[symbol] = signature
    return signature


def signature_of(symbol: str) -> CSignature:
    """Look up a signature declared with :func:`declare`.

    Args:
        symbol (str): the C symbol name

    Returns:
        CSignature: the declared signature
    """
    try:
        return _REGISTRY[symbol]
    except KeyError as exc:
        raise KeyError(
            f"{symbol!r} has no declared signature; declare it with "
            f'qp.runtime_declare({symbol!r}, "(...) -> ..."), or pass signature= to '
            f"qp.runtime_call"
        ) from exc


def declared_symbols() -> tuple[str, ...]:
    """The symbols declared so far, in declaration order."""
    return tuple(_REGISTRY)
