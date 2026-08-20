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

"""Regression tests for Triton shared-library cache keys."""

# pylint: disable=protected-access,wrong-import-position,too-few-public-methods

import numpy as np
import pytest

triton = pytest.importorskip("triton")
pytestmark = [pytest.mark.gpu]

from pennylane.backline.decoders.triton import triton_so_builder as builder
from pennylane.backline.decoders.triton.decoder_frontend import _make_css_decoder
from pennylane.backline.decoders.triton.persistent_kernel import _persistent_decoder_kernel


class TestConstexprCacheKeys:
    """Tests for constexpr hashing used by the shared-library builder."""

    def test_wrap_constexpr_distinguishes_decoder_tuples(self):
        """It should hash distinct decoder tuples differently."""
        hx = np.array([[1, 0], [0, 1]], dtype=int)
        hz = np.array([[1, 1], [0, 1]], dtype=int)
        decode_x = _make_css_decoder(hx, postprocess="hard", num_iters=5, prob=0.1)
        decode_z = _make_css_decoder(hz, postprocess="hard", num_iters=5, prob=0.1)
        signature = {
            "ring_u64_ptr": "*u64",
            "handoff_u64_ptr": "*u64",
            "stop_u32_ptr": "*u32",
            "ring_slots": "u32",
            "total": "u64",
        }

        _persistent_decoder_kernel.create_binder()

        def ast_hash(decoder_fns):
            wrapped = builder._wrap_constexpr(decoder_fns)
            src = _persistent_decoder_kernel.ASTSource(
                fn=_persistent_decoder_kernel,
                constexprs={"decoder_fns": wrapped},
                signature=signature,
                attrs={},
            )
            return src.hash()

        assert decode_x.cache_key != decode_z.cache_key
        assert ast_hash((decode_x, decode_z)) != ast_hash((decode_x, decode_x))
        assert ast_hash((decode_x, decode_z)) != ast_hash((decode_z, decode_z))
