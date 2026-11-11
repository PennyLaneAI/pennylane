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


@triton.jit
def _persistent_decoder_kernel(  # pylint: disable=too-many-arguments  # pragma: no cover
    ring_u64_ptr,
    handoff_u64_ptr,
    stop_u32_ptr,
    ring_slots,
    slot_words,
    total,
    decoder_fns: tl.constexpr,
    cache_mod: tl.constexpr,
):
    """Poll decoder requests from a ring buffer and write packed corrections.

    Both rings hold Catalyst's 16 B frames, one at the start of each slot: the payload in bytes
    ``0-7``, then little-endian metadata in bytes ``8-15`` with ``decoder_id`` in the low 32 bits
    and the sequence number in the high 32 bits.

    Args:
        ring_u64_ptr (*u64): Pointer to the request ring buffer, whose frames carry a packed
            syndrome.
        handoff_u64_ptr (*u64): Pointer to the reply ring buffer. Each reply frame carries the
            packed correction mask and the request's ``decoder_id``, and its sequence number,
            written last, marks it complete.
        stop_u32_ptr (*u32): Pointer to a stop flag polled while waiting for the
            next request.
        ring_slots (u32): Number of request and handoff slots. Must be a power
            of two so the kernel can mask ``cursor`` into a slot index.
        slot_words (u32): Distance from one slot to the next, in 8-byte words.
        total (u64): Number of requests to process. A value of ``0`` means keep
            running until ``stop_u32_ptr`` becomes nonzero.
        decoder_fns (tuple[Callable]): Compile-time tuple of Triton decoder
            functions selected by ``decoder_id``.
        cache_mod (str): Cache modifier for the polling loads, chosen per backend by
            :func:`~.decoder_frontend._build_triton_decoder`.
    """
    stop_poll_iters = tl.full((), 1024, tl.uint32)

    # cursor/total are u64; slot indices and seq numbers are u32 by wire layout.
    cursor = tl.zeros((), dtype=tl.uint64)
    halt = tl.zeros((), dtype=tl.int1)
    ring_slot_mask = tl.cast(ring_slots - 1, tl.uint64)
    while ((total == 0) or (cursor < total)) and (halt == 0):
        # ring_slots is a power of two, so modulo is a mask.
        idx = tl.cast(cursor & ring_slot_mask, tl.uint32)
        expect = tl.cast(cursor + 1, tl.uint32)

        req = ring_u64_ptr + idx * slot_words
        # little-endian: low32=decoder_id, high32=seq
        metadata = tl.load(req + 1, volatile=True, cache_modifier=cache_mod)
        seq = tl.cast(metadata >> 32, tl.uint32)

        nspins = tl.zeros((), dtype=tl.uint32)
        # loop until expected seq or stop
        while (seq != expect) and (halt == 0):
            if (nspins % stop_poll_iters) == 0:
                # check the stop flag only every stop_poll_iters iters
                halt = tl.load(stop_u32_ptr, volatile=True, cache_modifier=cache_mod) != 0
            nspins += 1
            metadata = tl.load(req + 1, volatile=True, cache_modifier=cache_mod)
            seq = tl.cast(metadata >> 32, tl.uint32)
        decoder_id = tl.cast(metadata, tl.uint32)

        # return statements are unsupported so we need to check halt again
        if halt == 0:
            syndrome = tl.load(req, volatile=True, cache_modifier=cache_mod)
            correction = tl.cast(0, tl.uint64)
            # dispatch to the right decoder e.g., X/Z CSS code
            for i in tl.static_range(len(decoder_fns)):
                if decoder_id == i:
                    correction = decoder_fns[i](syndrome)

            out = handoff_u64_ptr + idx * slot_words
            tl.store(out, correction, cache_modifier=".wt")
            # The reply's decoder_id and sequence number in one release store, so the sequence
            # number is never seen before the correction it completes.
            tl.atomic_xchg(
                out + 1,
                (tl.cast(expect, tl.uint64) << 32) | tl.cast(decoder_id, tl.uint64),
                sem="release",
                scope="sys",
            )

            cursor += 1
