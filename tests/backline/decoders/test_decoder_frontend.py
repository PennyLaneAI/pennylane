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

"""Tests for the Triton decoder frontend."""

# pylint: disable=protected-access,wrong-import-position,broad-exception-caught

from pathlib import Path

import numpy as np
import pytest

triton = pytest.importorskip("triton")


def _has_cuda_target() -> bool:
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except Exception:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _has_cuda_target(), reason="Triton decoder tests require a CUDA device"),
]

from pennylane.backline.decoders.triton import decoder_frontend as frontend


class TestBuildTritonDecoder:
    """Tests for build_triton_decoder."""

    def test_build_triton_decoder_passes_options_to_builder(self, monkeypatch, tmp_path):
        """It should forward the normalized build options to build_so."""
        calls = {}

        def fake_build_so(kernel, **kwargs):
            calls["kernel"] = kernel
            calls.update(kwargs)
            return Path(kwargs["out"]), "decoder_symbol"

        monkeypatch.setattr(frontend.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
        monkeypatch.setattr(frontend, "build_so", fake_build_so)

        decoder_fns = (object(), object())
        so_path, symbol_name = frontend.build_triton_decoder(
            decoder_fns,
            platform=" cuda:80:32 ",
            grid=(2, 3, 4),
            num_warps=2,
            num_stages=3,
            compiler="cc",
            cflags=("-g",),
        )

        assert so_path == (tmp_path / "librdma_triton_decoder.so").resolve()
        assert symbol_name == "decoder_symbol"
        assert calls == {
            "kernel": frontend._persistent_decoder_kernel,
            "signature": {
                "ring_u64_ptr": "*u64",
                "handoff_u64_ptr": "*u64",
                "stop_u32_ptr": "*u32",
                "total": "u64",
            },
            "constexpr": {"decoder_fns": decoder_fns},
            "grid": (2, 3, 4),
            "target": "cuda:80:32",
            "out": str((tmp_path / "librdma_triton_decoder.so").resolve()),
            "num_warps": 2,
            "num_stages": 3,
            "compiler": "cc",
            "cflags": ("-g",),
        }

    def test_build_triton_decoder_cleans_tmpdir_on_failure(self, monkeypatch, tmp_path):
        """It should delete the temporary output directory when the build fails."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        monkeypatch.setattr(frontend.tempfile, "mkdtemp", lambda prefix: str(scratch))
        monkeypatch.setattr(
            frontend,
            "build_so",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError, match="boom"):
            frontend.build_triton_decoder((object(),), platform="cuda:80:32")

        assert not scratch.exists()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"decoder_fns": (), "platform": "cuda:80:32"}, "non-empty tuple"),
            (
                {"decoder_fns": (object(),), "platform": "cuda:80:32", "num_warps": 0},
                "num_warps must be > 0",
            ),
            (
                {"decoder_fns": (object(),), "platform": "cuda:80:32", "num_stages": 0},
                "num_stages must be > 0",
            ),
            ({"decoder_fns": (object(),), "platform": "cuda:80"}, "raw Triton target"),
        ],
    )
    def test_build_triton_decoder_validates_options(self, kwargs, message):
        """It should reject invalid build options before starting a build."""
        with pytest.raises(ValueError, match=message):
            frontend.build_triton_decoder(**kwargs)


class TestBuildCssBpDecoder:
    """Tests for build_css_bp_decoder."""

    def test_build_css_bp_decoder_builds_two_specialized_decoders(self, monkeypatch):
        """It should specialize one decoder per parity-check matrix and forward them."""
        decoder_calls = []
        built_decoders = [object(), object()]
        forwarded = {}

        def fake_make_css_decoder(hx, *, postprocess, num_iters, prob):
            decoder_calls.append((hx.copy(), postprocess, num_iters, prob))
            return built_decoders[len(decoder_calls) - 1]

        def fake_build_triton_decoder(decoder_fns, **kwargs):
            forwarded["decoder_fns"] = decoder_fns
            forwarded.update(kwargs)
            return Path("/tmp/decoder.so"), "decode_symbol"

        monkeypatch.setattr(frontend, "_make_css_decoder", fake_make_css_decoder)
        monkeypatch.setattr(frontend, "build_triton_decoder", fake_build_triton_decoder)

        hx = [[1, 0], [0, 1]]
        hz = [[1, 1], [0, 1]]

        so_path, symbol_name = frontend.build_css_bp_decoder(
            hx,
            hz,
            postprocess="hard",
            num_iters=7,
            prob=0.2,
            platform="cuda:80:32",
            grid=(5, 1, 1),
            num_warps=4,
            num_stages=2,
            compiler="cc",
            cflags=("-g",),
        )

        assert so_path == Path("/tmp/decoder.so")
        assert symbol_name == "decode_symbol"
        assert len(decoder_calls) == 2
        np.testing.assert_array_equal(decoder_calls[0][0], np.asarray(hx))
        np.testing.assert_array_equal(decoder_calls[1][0], np.asarray(hz))
        assert decoder_calls[0][1:] == ("hard", 7, 0.2)
        assert decoder_calls[1][1:] == ("hard", 7, 0.2)
        assert forwarded == {
            "decoder_fns": tuple(built_decoders),
            "platform": "cuda:80:32",
            "grid": (5, 1, 1),
            "num_warps": 4,
            "num_stages": 2,
            "compiler": "cc",
            "cflags": ("-g",),
        }

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"postprocess": "bad"}, "postprocess must be 'hard' or 'osd'"),
            ({"num_iters": 0}, "num_iters must be > 0"),
            ({"prob": 0.0}, r"prob must be in \(0, 1\)"),
            ({"prob": 1.0}, r"prob must be in \(0, 1\)"),
        ],
    )
    def test_build_css_bp_decoder_validates_decoder_options(self, kwargs, message):
        """It should reject invalid BP decoder options."""
        H = np.eye(2, dtype=int)

        decoder_kwargs = {
            "postprocess": "osd",
            "num_iters": 10,
            "prob": 0.1,
            "platform": "cuda:80:32",
        } | kwargs
        with pytest.raises(ValueError, match=message):
            frontend.build_css_bp_decoder(H, H, **decoder_kwargs)

    @pytest.mark.parametrize(
        ("H", "message"),
        [
            (np.array([1, 0]), "2D array"),
            (np.zeros((0, 1), dtype=int), "non-empty"),
            (np.array([[0, 2]]), "binary entries"),
            (np.ones((65, 1), dtype=int), "at most 64"),
            (np.ones((1, 65), dtype=int), "at most 64"),
        ],
    )
    def test_to_numpy_rejects_invalid_matrices(self, H, message):
        """It should validate parity-check matrices before specialization."""
        with pytest.raises(ValueError, match=message):
            frontend._to_numpy(H)

    def test_triton_module_exports_bp_builder(self):
        """The Triton decoder package should export the BP builder by name."""
        from pennylane.backline.decoders import triton as triton_module

        assert triton_module.build_css_bp_decoder is frontend.build_css_bp_decoder
