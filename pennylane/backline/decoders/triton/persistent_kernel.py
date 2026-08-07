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

import triton
import triton.language as tl

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
    """Decode ring-buffer requests until completion or shutdown.

    ring_u64_ptr: PayloadSlot (64 bytes) = 8 u64 words
        word 0: syndrome packed into one u64, so it cannot exceed 64 bits/checks
        word 1: low32 = decoder_id, high32 = seq
    handoff_u64_ptr: HandoffSlot (16 bytes) = 2 u64 words
        word 0: correction packed into one u64, so it cannot exceed 64 bits/qubits
        word 1: low32 = seq, high32 = pad
    stop_u32_ptr: single u32 scalar

    ``decoder_fns`` is a constexpr tuple of Triton decoder functions. ``decoder_id``
    selects the corresponding tuple index.

    Note: low32/high32 field layout assumes little-endian targets.
    """
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
            if (nspins & 1023) == 0:
                # check the stop flag only every 1024 iters
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


_persistent_css_decoder_kernel = _persistent_decoder_kernel
