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

"""Persistent Triton kernel for backline decoder dispatch."""

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError("Triton decoders require installed `triton` Python package.") from exc

K_RING_SLOTS = tl.constexpr(256)  # can store 256 elements
PAYLOAD_SLOT_WORDS = tl.constexpr(8)  # sizeof(PayloadSlot) / sizeof(u64)
HANDOFF_SLOT_WORDS = tl.constexpr(2)  # sizeof(HandoffSlot) / sizeof(u64)


@triton.jit
def _persistent_decoder_kernel(
    ring_u64_ptr,
    handoff_u64_ptr,
    stop_u32_ptr,
    total,
    decoder_fns: tl.constexpr,
):
    """Poll decoder requests from a ring buffer and write packed corrections.

    Args:
        ring_u64_ptr (*u64): Pointer to the request ring buffer. Each slot stores
            a packed syndrome in bytes ``0-7`` and packed metadata in bytes
            ``8-15``. The metadata layout is little-endian, with
            ``decoder_id`` in the low 32 bits and the sequence number in the
            high 32 bits.
        handoff_u64_ptr (*u64): Pointer to the response buffer. Bytes ``0-7``
            of each slot store the packed correction mask and bytes ``8-15``
            store the completion sequence number.
        stop_u32_ptr (*u32): Pointer to a stop flag polled while waiting for the
            next request.
        total (u64): Number of requests to process. A value of ``0`` means keep
            running until ``stop_u32_ptr`` becomes nonzero.
        decoder_fns (tuple[Callable]): Compile-time tuple of Triton decoder
            functions selected by ``decoder_id``.
    """
    stop_poll_iters = tl.full((), 1024, tl.uint32)

    # cursor/total are u64; slot indices and seq numbers are u32 by wire layout.
    cursor = tl.zeros((), dtype=tl.uint64)
    halt = tl.zeros((), dtype=tl.int1)
    while ((total == 0) or (cursor < total)) and (halt == 0):
        # cursor % K_RING_SLOTS
        idx = tl.cast(cursor & (K_RING_SLOTS - 1), tl.uint32)
        expect = tl.cast(cursor + 1, tl.uint32)

        # PayloadSlot = 8 * uint64
        req = ring_u64_ptr + idx * PAYLOAD_SLOT_WORDS
        # little-endian: low32=decoder_id, high32=seq
        metadata = tl.load(req + 1, volatile=True)
        seq = tl.cast(metadata >> 32, tl.uint32)

        nspins = tl.zeros((), dtype=tl.uint32)
        # loop until expected seq or stop
        while (seq != expect) and (halt == 0):
            if (nspins % stop_poll_iters) == 0:
                # check the stop flag only every stop_poll_iters iters
                halt = tl.load(stop_u32_ptr, volatile=True) != 0
            nspins += 1
            metadata = tl.load(req + 1, volatile=True)
            seq = tl.cast(metadata >> 32, tl.uint32)
        decoder_id = tl.cast(metadata, tl.uint32)

        # return statements are unsupported so we need to check halt again
        if halt == 0:
            syndrome = tl.load(req, volatile=True)
            correction = tl.cast(0, tl.uint64)
            # dispatch to the right decoder e.g., X/Z CSS code
            for i in tl.static_range(len(decoder_fns)):
                if decoder_id == i:
                    correction = decoder_fns[i](syndrome)

            out = handoff_u64_ptr + idx * HANDOFF_SLOT_WORDS
            tl.store(out, correction, cache_modifier=".wt")
            tl.atomic_xchg(
                out + 1,
                tl.cast(expect, tl.uint64),  # sets seq assuming little-endian
                sem="release",
                scope="sys",
            )

            cursor += 1
