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

"""``qp.backline.decode`` -- the explicit per-round syndrome->correction decode.

:func:`decode` emits one transport round from inside a captured QNode: resolve the controller's
session, stage the syndrome, post the round, collect the reply.

Every step is a local :func:`~pennylane.runtime_call`, so the round runs in the controller's own
process through the transport C ABI that ``librt_transport`` exports. Session bring-up and teardown
come from the ``inject-transport-session`` pass; :func:`decode` only drives the round.
"""

from __future__ import annotations

import numpy as np

from pennylane import math

from .device import active_placement
from .placement import DEFAULT_MESSAGE_BYTES
from .runtime import runtime_call

# In-process ``__call`` adapters from TransportCAPI.h, named verbatim by a local runtime_call.
_PREFIX = "__catalyst__transport__"
_GET_SESSION = f"{_PREFIX}get_session__call"
_STAGE_PAYLOAD = f"{_PREFIX}stage_payload__call"
_POST = f"{_PREFIX}post__call"
_COLLECT = f"{_PREFIX}collect__call"


_SIG_GET_SESSION = "(i32, str) -> ptr"  # role, key
_SIG_STAGE_PAYLOAD = "(ptr, buf, u64, u32) -> i32"  # session, src, bytes, decoder_id
_SIG_POST = "(ptr, u32) -> i32"  # session, work_item_idx
_SIG_COLLECT = "(ptr, out, u64) -> i32"  # session, reply(out), reply_bytes

# Mirrors catalyst::transport::Role.
ROLE_CONTROLLER = 0
ROLE_COPROCESSOR = 1

_DEFAULT_WORK_ITEM = 0

_PACKED_U64_BITS = 64
_PACKED_U64_BYTES = DEFAULT_MESSAGE_BYTES


def _session_key(coprocessor, coprocessors=()) -> str:
    """The key the controller's session for this round is registered under.

    ``inject-transport-session`` keys one controller session per coprocessor by *that coprocessor's*
    ``name``, falling back to ``"coprocessor.0"``; a placement with no coprocessor is
    ``"controller"``. With several coprocessors that fallback would send every unnamed one to the
    same session, so a name is required.

    Raises:
        ValueError: if the round targets an unnamed coprocessor and the placement holds more
            than one, which no key can tell apart
    """
    if coprocessor is None:
        return "controller"
    name = getattr(coprocessor, "name", None)
    if name:
        return name
    if len(coprocessors) > 1:
        raise ValueError(
            f"decode: this round targets a coprocessor with no name, and the placement has "
            f"{len(coprocessors)}. Pass coprocessor= to choose one directly."
        )
    return "coprocessor.0"


def _byte_count(array) -> int:
    """The number of bytes ``array`` occupies, from its shape and dtype.

    Works for a JAX tracer as well as a plain array: both carry a static shape and dtype.
    """
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    if shape is None or dtype is None:
        return int(np.asarray(array).nbytes)
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * np.dtype(dtype).itemsize


def _resolve_out_bytes(controller, out_bytes) -> int:
    """How many bytes the correction reply occupies.

    Explicit ``out_bytes`` wins; otherwise use the controller's committed reply size.
    """
    if out_bytes is not None:
        return int(out_bytes)
    return int(getattr(controller, "out_bytes", DEFAULT_MESSAGE_BYTES))


def _resolve_nodes(controller, coprocessor):
    """Fill in whichever node the caller left out, from the built placement. An explicit node always wins."""
    if controller is not None and coprocessor is not None:
        return controller, coprocessor

    placement = active_placement()
    if placement is None:
        raise ValueError(
            "decode: the nodes are resolved from the placement of the device being traced, and "
            "this trace has none. Build a placement, or pass the nodes explicitly: "
            "decode(syndrome, controller=ctrl, coprocessor=coproc)."
        )

    if controller is None:
        controller = placement.controller
    if coprocessor is None:
        coprocs = placement.coprocessors
        if len(coprocs) != 1:
            raise ValueError(
                "decode: with multiple coprocessors, pass coprocessor= explicitly to choose the "
                "transport session. decoder_id only selects the decoder inside that coprocessor."
            )
        coprocessor = coprocs[0]
    return controller, coprocessor


def _validate_packed(syndrome, in_bytes, out_bytes):
    """Validate packed decode inputs and transport sizes."""
    if in_bytes is not None and int(in_bytes) != _PACKED_U64_BYTES:
        raise ValueError(
            f"decode: bitpack=True requires in_bytes={_PACKED_U64_BYTES}, got {int(in_bytes)}"
        )
    if out_bytes is not None and int(out_bytes) != _PACKED_U64_BYTES:
        raise ValueError(
            f"decode: bitpack=True requires out_bytes={_PACKED_U64_BYTES}, got {int(out_bytes)}"
        )

    interface = (
        math.get_interface(*syndrome)
        if isinstance(syndrome, (tuple, list))
        else math.get_interface(syndrome)
    )
    if interface == "jax":
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

        syndrome = jnp.asarray(syndrome, dtype=jnp.uint8)
    else:
        syndrome = np.asarray(syndrome, dtype=np.uint8)

    if syndrome.ndim != 1:
        raise ValueError("decode: packed syndromes must be a 1D bit vector")
    if int(syndrome.shape[0]) > _PACKED_U64_BITS:
        raise ValueError(
            f"decode: packed syndromes support at most {_PACKED_U64_BITS} bits, got {int(syndrome.shape[0])}"
        )
    return syndrome


def _pack(syndrome):
    """Pack a syndrome bit vector into 8 little-endian bytes."""
    if math.get_interface(syndrome) == "jax":
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

        xp = jnp
    else:
        xp = np
    packed = xp.packbits(syndrome, bitorder="little")
    # HACK: ``packbits`` returns the minimal byte width, but for bitpack=True transport payloads
    # are always exactly 8 bytes, so pad and slice to get the right size
    packed = xp.pad(packed, (0, _PACKED_U64_BYTES))[:_PACKED_U64_BITS]
    return packed


def _unpack(correction):
    """Unpack 8 little-endian bytes into a 64-entry boolean bit vector."""
    if math.get_interface(correction) == "jax":
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

        xp = jnp
    else:
        xp = np
    return math.cast(xp.unpackbits(correction, bitorder="little", count=_PACKED_U64_BITS), bool)


def decode(  # pylint: disable=too-many-arguments
    syndrome,
    *,
    controller=None,
    coprocessor=None,
    out_bytes=None,
    in_bytes=None,
    decoder_id=0,
    work_item=_DEFAULT_WORK_ITEM,
    bitpack=True,
    library=None,
):
    r"""Offload one syndrome to a coprocessor and return its correction (post & collect).

    Records a single transport round inside a captured QNode. Every step is a local
    :func:`~pennylane.runtime_call`, so the round is driven from the controller's own process over
    the session that ``inject-transport-session`` brought up from the device's placement.

    Args:
        syndrome: The syndrome to send. Passed by data pointer, so its byte length comes from its
            shape and dtype at compile time. With ``bitpack=True``, this must be a 1D bit vector
            with at most 64 entries.
        controller (Controller): The :class:`~.Controller` whose session drives the round, and whose
            :attr:`~.Controller.out_bytes` supplies the default reply size.
        coprocessor (Coprocessor | None): The :class:`~.Coprocessor` the round targets. Selects the
            session key; which coprocessor serves the round is otherwise fixed by the session's
            configuration.
        out_bytes (int | None): The correction reply size in bytes. Defaults to the controller's
            :attr:`~.Controller.out_bytes`.
        in_bytes (int | None): How many bytes of ``syndrome`` to send, at most what the round was
            committed to carry. Defaults to ``syndrome``'s full byte length.
        decoder_id (int): Which coprocessor-side decoder handles this round.
        work_item (int): The committed work-item index to post.
        bitpack (bool): Pack syndrome bits into 8-byte payload and unpack reply to 64-bit vector.
        library (str | None): Shared library exporting the transport symbols, recorded so the
            compiler links it. Defaults to ``None``, relying on ``librt_transport`` already being
            loaded.

    Returns:
        The correction reply, as a ``uint8`` buffer of ``out_bytes`` bytes, or a 64-entry boolean
        bit vector when ``bitpack=True``.

    Raises:
        ValueError: if the round targets an unnamed coprocessor while the placement holds more
        than one.

    .. warning::

        Backline is experimental and only usable through the Catalyst compiler. :func:`decode` must
        be called inside a ``@qjit`` program; calling it eagerly raises.

    .. seealso:: :class:`~.Controller`, :class:`~.Coprocessor`,
        :class:`~pennylane.Backline`

    **Example**

    The nodes come from the placement of the device being traced, so a round on a built backline is
    just ``qp.backline.decode(syndrome)``. Here the controller commits to an 8-byte correction, and
    the coprocessor runs the decoder that produces it:

    .. code-block:: python

        import pennylane as qp

        con = qp.Controller(
            device=qp.device("null.qubit", wires=2),
            executor_options={"host": "192.168.3.15", "port": 7810},
        )
        coproc = qp.Coprocessor(
            coprocessor_fn="decoder",
            name="decoder-0",
            hardware="gpu",
            endpoint=qp.Endpoint("192.168.1.3", 18590),
        )
        dev = qp.Backline(controller=con, coprocessors=[coproc], transport="rdma")

        @qp.qjit(capture=True)
        @qp.qnode(dev)
        def circuit(syndrome):
            # one round: stage the syndrome, post it, wait for the correction
            correction = qp.backline.decode(syndrome)
            qp.cond(correction[0] == 1, qp.X)(0)
            return qp.expval(qp.Z(0))

    The round is resolved from the device being traced, so the program has to be captured
    (``qp.qjit(capture=True)``). ``syndrome`` is sent by data pointer, so its byte length is fixed
    by its shape and dtype at compile time; ``correction`` comes back as a ``uint8`` buffer of
    ``out_bytes`` bytes. Pass ``controller=`` / ``coprocessor=`` to choose the nodes explicitly, and
    ``decoder_id=`` to select which coprocessor-side decoder handles the round.
    """

    controller, coprocessor = _resolve_nodes(controller, coprocessor)
    placement = active_placement()
    key = _session_key(coprocessor, placement.coprocessors if placement is not None else ())
    if bitpack:
        syndrome = _pack(_validate_packed(syndrome, in_bytes, out_bytes))
        nbytes = _PACKED_U64_BYTES
        reply_bytes = _PACKED_U64_BYTES
    else:
        nbytes = _byte_count(syndrome) if in_bytes is None else int(in_bytes)
        reply_bytes = _resolve_out_bytes(controller, out_bytes)

    # The live controller session the setup pass registered under `key`.
    session = runtime_call(
        _GET_SESSION, ROLE_CONTROLLER, key, signature=_SIG_GET_SESSION, library=library
    )

    # Copy `nbytes` into the outbound slot and stamp `decoder_id`, then start the round trip.
    _ = runtime_call(
        _STAGE_PAYLOAD,
        session,
        syndrome,
        nbytes,
        decoder_id,
        signature=_SIG_STAGE_PAYLOAD,
        library=library,
    )

    # Start the round trip
    _ = runtime_call(_POST, session, work_item, signature=_SIG_POST, library=library)

    # Wait for the reply
    _, correction = runtime_call(
        _COLLECT,
        session,
        reply_bytes,
        signature=_SIG_COLLECT,
        out_bytes=reply_bytes,
        library=library,
    )
    return _unpack(correction) if bitpack else correction
