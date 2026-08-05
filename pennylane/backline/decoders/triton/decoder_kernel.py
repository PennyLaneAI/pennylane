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

from .algorithms import _decode_one

K_RING_SLOTS = tl.constexpr(256)  # can store 256 elements
PAYLOAD_SLOT_WORDS = tl.constexpr(8)  # sizeof(PayloadSlot) / sizeof(u64)
HANDOFF_SLOT_WORDS = tl.constexpr(2)  # sizeof(HandoffSlot) / sizeof(u64)


@triton.jit
def _persistent_css_decoder_kernel(
    ring_u64_ptr,
    handoff_u64_ptr,
    stop_u32_ptr,
    total,
    Hx: tl.constexpr,
    Hz: tl.constexpr,
    BP_VARIANT: tl.constexpr,
    POSTPROCESS: tl.constexpr,
    PROB: tl.constexpr,
    NITER: tl.constexpr,
    ALPHA: tl.constexpr,
):
    """Decode ring-buffer requests until completion or shutdown.

    ring_u64_ptr: PayloadSlot (64 bytes) = 8 u64 words
        word 0: syndrome
        word 1: low32 = decoder_id (0 -> X, 1 -> Z), high32 = seq
    handoff_u64_ptr: HandoffSlot (16 bytes) = 2 u64 words
        word 0: correction
        word 1: low32 = seq, high32 = pad
    stop_u32_ptr: single u32 scalar

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
            if (nspins & 0x3FF) == 0:
                # check the stop flag only every 1024 iters
                halt = tl.load(stop_u32_ptr, volatile=True) != 0
            nspins += 1
            metadata = tl.load(req + 1, volatile=True)
            seq = tl.cast(metadata >> 32, tl.uint32)
        decoder_id = tl.cast(metadata, tl.uint32)

        # return statements are unsupported so we need to check halt again
        if halt == 0:
            syndrome = tl.load(req, volatile=True)
            if decoder_id == 0:
                correction = _decode_one(
                    syndrome,
                    Hx,
                    bp_variant=BP_VARIANT,
                    postprocess=POSTPROCESS,
                    prob=PROB,
                    NITER=NITER,
                    ALPHA=ALPHA,
                )
            elif decoder_id == 1:
                correction = _decode_one(
                    syndrome,
                    Hz,
                    bp_variant=BP_VARIANT,
                    postprocess=POSTPROCESS,
                    prob=PROB,
                    NITER=NITER,
                    ALPHA=ALPHA,
                )
            else:
                # NOTE: unrecognized decoder_id -> no correction
                correction = tl.cast(0, tl.uint64)

            out = handoff_u64_ptr + idx * HANDOFF_SLOT_WORDS
            tl.store(out, correction, cache_modifier=".wt")
            tl.atomic_xchg(
                out + 1,
                tl.cast(expect, tl.uint64),  # sets seq assuming little-endian
                sem="release",
                scope="sys",
            )

            cursor += 1
