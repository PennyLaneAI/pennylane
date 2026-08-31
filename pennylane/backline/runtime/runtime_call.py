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

"""Calling a runtime, either as an operation on a backend or as a bare symbol."""

from __future__ import annotations

import functools

from . import operands
from .signature import CSignature, CType, declare, signature_of


def _tracing() -> bool:
    """Whether a JAX trace is open, which is where a recorded call belongs."""
    # pylint: disable=import-outside-toplevel
    try:
        from jax._src.core import trace_state_clean

        return not trace_state_clean()
    except ImportError:  # pragma: no cover
        return True


@functools.lru_cache(maxsize=1)
def get_runtime_call_prim():
    """Get the primitive that backs :func:`runtime_call`, creating it on the first call."""

    # pylint: disable=import-outside-toplevel
    from pennylane.capture.custom_primitives import QpPrimitive

    runtime_call_prim = QpPrimitive("runtime_call")
    runtime_call_prim.multiple_results = True

    # pylint: disable=unused-argument
    @runtime_call_prim.def_abstract_eval
    def _(*args, signature, symbol, out_bytes, dispatch, library):
        return operands.result_avals(signature, out_bytes)

    return runtime_call_prim


def runtime_call(target, *args, signature=None, out_bytes=0, address=None, library=None):
    r"""Call a declared runtime symbol, in-process or on the executor it is addressed to.

    A symbol is declared once with :func:`~pennylane.runtime_declare`, then called by name from
    inside a :func:`~pennylane.qjit` program. For a one-off call, pass its signature with
    ``signature=`` or pass a :class:`~pennylane.backline.runtime.CSignature` as ``target``.

    Passing ``address`` dispatches the call: it is recorded into the program and sent to that
    executor, which invokes the symbol on the machine the runtime lives on. Omitting ``address``
    makes the call **local**: the symbol is invoked in the process running the compiled program,
    through the ordinary in-process C ABI.

    **Dispatched** (``address`` set):
    A dispatched call is run by the Catalyst executor. The `dispatch-executor-targets` pass turns it
    into an ``executor.call``, which reaches ``__catalyst__executor__call_wrapper`` on the addressed
    machine. That calls the symbol through LLVM ORC's wrapper convention. You need to have a
    catalyst executor running on the addressed machine, which can be enabled from within a
    backline-controlled QJIT function.

    **Local** (``address`` is ``None``):
    A local call is run in the process running the compiled program. The symbol is resolved and
    invoked through the ordinary C ABI. ``library`` is recorded on the compiled module so the driver
    links the shared library that exports it.

    Args:
        target (str, CSignature): the declared symbol name, or its complete signature
        *args: one argument per declared parameter except ``out``, in declaration order
        signature (str | CSignature | None): the signature for a symbol not yet declared, using
            the ``"(parameter, ...) -> result"`` form or a :class:`CSignature`
        out_bytes (int | Sequence[int]): the compile-time size of each ``out`` buffer
        address (str | None): the executor to dispatch to, as ``"host:port"``. ``None`` makes the
                              call local (in-process).
        library (str | None): for a local call, the shared library exporting the symbol, recorded so
                              the driver links it. Overrides the library set at
                              :func:`~pennylane.runtime_declare` time. Ignored for a dispatched
                              call.

    Returns:
        The symbol's declared result. A symbol with one ``out`` parameter returns
        ``(result, buffer)``. With several, the buffers follow the result in the order they are
        declared. A local call to a ``void`` symbol returns ``None``.

    **Example**

    .. code-block:: python

        import pennylane as qp

        qp.runtime_declare("example_call_rounds", "(ptr, u32) -> u64")

        def program(session):  # the body of a qjit program
            return qp.runtime_call("example_call_rounds", session, 100000, address="board:9000")

    A symbol that fills a buffer declares it as an ``out`` parameter. The caller does not pass one:
    it asks for ``out_bytes=`` and gets the filled buffer back alongside the result.

    .. code-block:: python

        qp.runtime_declare("example_call_collect", "(ptr, out, u64) -> i32")

        def collect(session):
            status, reply = qp.runtime_call("example_call_collect", session, 64, out_bytes=64)
            return reply

    For example, the above call to this symbol declared ``(ptr, out, u64) -> i32``:

    * ``session`` is the ``ptr``.
    * The ``out`` takes no argument. The symbol writes that buffer, so there is nothing to pass in.
    * ``64`` is the ``u64``: this symbol's own argument for how many bytes it may write.
    * ``out_bytes=64`` says how big a buffer to give it. The number shows up twice because one is
      what you tell the symbol and the other is what you hand it, so keep the two in step.
    * The call returns two results: ``status``, the ``i32`` the symbol returned, and ``reply``,
      which corresponds to the ``out`` buffer it filled in, as a ``uint8`` array of ``out_bytes``
      bytes.

    A symbol can declare more than one ``out``. Give ``out_bytes`` one size per buffer, and they
    return in the order they are declared, for example:

    .. code-block:: python

        qp.runtime_declare("example_call_two_regions", "(ptr, out, out, u64) -> i32")

        def collect_both(session):
            status, first, second = qp.runtime_call(
                "example_call_two_regions", session, 96, out_bytes=(32, 64)
            )
            return first, second

    .. seealso:: :func:`~pennylane.runtime_declare`
    """
    resolved = _resolve_signature(target, signature)
    sizes = operands.out_sizes(resolved, out_bytes)
    lib = library if library is not None else resolved.library

    return _record(resolved, args, sizes, address, lib)


def _record(signature: CSignature, args, sizes, address, library):
    """Record a call on a declared symbol, dispatched to an executor or invoked in-process.

    A dispatched call (``address`` set) becomes an ``executor.call``, which flattens the arguments
    into one buffer and invokes the symbol through LLVM ORC's wrapper convention.

    A local call (``address`` is ``None``) becomes an ordinary ``catalyst.custom_call``, which
    passes each argument as a descriptor pointer (the in-process C ABI) and reaches the symbol
    resolved at load time.
    """
    if not _tracing():
        raise RuntimeError(
            f"{signature.symbol} is being called outside a compiled program. A recorded call is "
            f"only valid inside a qjit function."
        )
    if address is not None and signature.result is CType.VOID and not sizes:
        raise TypeError(
            f"{signature.symbol}{signature} returns nothing, a dispatched call to it has no result."
        )
    results = get_runtime_call_prim().bind(
        *operands.operands_for(signature, args, local=address is None),
        signature=signature,
        symbol=signature.symbol,
        out_bytes=sizes,
        dispatch=address,
        library=library,
    )
    value = None if signature.result is CType.VOID else results[0][0]
    buffers = results[0 if signature.result is CType.VOID else 1 :]
    if not sizes:
        return value
    return (value, *buffers) if len(buffers) > 1 else (value, buffers[0])


def _resolve_signature(symbol, signature) -> CSignature:
    """Work out which signature a symbol call site means."""
    if isinstance(symbol, CSignature):
        if signature is not None:
            raise ValueError("pass either a CSignature or signature=, not both")
        return symbol
    if signature is None:
        return signature_of(symbol)
    if isinstance(signature, CSignature):
        return signature
    return declare(symbol, signature)
