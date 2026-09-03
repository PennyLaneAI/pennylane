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

# pylint: disable=protected-access,wrong-import-position,broad-exception-caught,import-outside-toplevel

import ctypes
import shutil

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

from pennylane.backline import CoprocessorFunction, css_bp_decoder, triton_decoder
from pennylane.backline.decoders.triton import decoder_frontend as frontend


def _echo_decoder(syndrome):
    return syndrome


def _cuda_platform() -> str:
    target = triton.runtime.driver.active.get_current_target()
    return f"{target.backend}:{target.arch}:{target.warp_size}"


class TestPollingCacheModifier:
    """The polling loads need a per-backend cache modifier.

    ``volatile=True`` is kept on both backends, but the cache modifier is not portable:
    on HIP ``".cv"`` is what lowers the load to an LLVM ``volatile`` load, while on CUDA
    ``volatile=True`` already emits ``ld.volatile.global`` and adding ``".cv"`` makes
    ``ptxas`` reject the combination from Triton 3.8 onwards.
    """

    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            ("hip:gfx90a:64", ".cv"),
            ("hip:gfx942:64", ".cv"),
            ("cuda:80:32", ""),
            ("cuda:120:32", ""),
        ],
    )
    def test_cache_mod_is_selected_per_backend(self, platform, expected, monkeypatch, tmp_path):
        """The backend of the platform string decides the cache modifier, nothing else does."""
        pytest.importorskip("triton")
        captured = {}

        def fake_build_so(*_args, **kwargs):
            captured["cache_mod"] = kwargs["constexpr"]["cache_mod"]
            return tmp_path / "fake.so", "fake_symbol"

        monkeypatch.setattr(frontend, "_build_so", fake_build_so)
        frontend._build_triton_decoder((_echo_decoder,), platform=platform)

        assert captured["cache_mod"] == expected


class TestBuildTritonDecoder:
    """Tests for _build_triton_decoder."""

    @pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc compiler not available")
    def test_build_triton_decoder_compiles_shared_library(self):
        """It should compile a Triton decoder shared library end to end."""
        so_path, symbol_name = frontend._build_triton_decoder(
            (_echo_decoder,),
            platform=_cuda_platform(),
            compiler=shutil.which("nvcc") or "nvcc",
        )

        assert so_path.exists()
        assert symbol_name.endswith("_catalyst")
        lib = ctypes.CDLL(str(so_path))
        assert getattr(lib, symbol_name)

    def test_build_triton_decoder_cleans_tmpdir_on_failure(self, monkeypatch, tmp_path):
        """It should delete the temporary output directory when the build fails."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        real_mkdtemp = frontend.tempfile.mkdtemp

        def fake_mkdtemp(*args, **kwargs):
            prefix = kwargs.get("prefix")
            if prefix is None and len(args) >= 2:
                prefix = args[1]
            return str(scratch) if prefix == "pl_triton_decoder_" else real_mkdtemp(*args, **kwargs)

        monkeypatch.setattr(frontend.tempfile, "mkdtemp", fake_mkdtemp)

        with pytest.raises(FileNotFoundError, match="missing-nvcc"):
            frontend._build_triton_decoder(
                (_echo_decoder,),
                platform=_cuda_platform(),
                compiler="/missing-nvcc",
            )

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
            ({"decoder_fns": (object(),), "platform": "cuda:80"}, "raw Triton platform"),
            (
                {"decoder_fns": (object(),), "platform": "cpu:80:32"},
                "backend must be 'cuda' or 'hip'",
            ),
            (
                {"decoder_fns": (object(),), "platform": "cuda:80:64"},
                "warp size must be 32 for cuda",
            ),
            (
                {"decoder_fns": (object(),), "platform": "hip:gfx942:32"},
                "warp size must be 64 for hip",
            ),
        ],
    )
    def test_build_triton_decoder_validates_options(self, kwargs, message):
        """It should reject invalid build options before starting a build."""
        with pytest.raises(ValueError, match=message):
            frontend._build_triton_decoder(**kwargs)


class TestBuildCssBpDecoder:
    """Tests for _build_css_bp_decoder."""

    @pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc compiler not available")
    def test_build_css_bp_decoder_compiles_shared_library(self):
        """It should compile a CSS BP decoder shared library end to end."""
        hx = np.array([[1, 0], [0, 1]], dtype=int)
        hz = np.array([[1, 1], [0, 1]], dtype=int)

        so_path, symbol_name = frontend._build_css_bp_decoder(
            hx,
            hz,
            postprocess="hard",
            num_iters=7,
            prob=0.2,
            platform=_cuda_platform(),
            grid=(5, 1, 1),
            num_warps=4,
            num_stages=2,
            compiler=shutil.which("nvcc") or "nvcc",
            cflags=("-g",),
        )

        assert so_path.exists()
        assert symbol_name.endswith("_catalyst")
        lib = ctypes.CDLL(str(so_path))
        assert getattr(lib, symbol_name)

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
            frontend._build_css_bp_decoder(H, H, **decoder_kwargs)

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


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc compiler not available")
class TestPublicDecoderWrappers:
    """The public ``triton_decoder`` / ``css_bp_decoder`` entry points wrap the internal builders
    and return a :class:`~.CoprocessorFunction` handle.
    """

    def test_triton_decoder_returns_a_coprocessor_function(self):
        """``qp.backline.triton_decoder`` compiles and wraps the result in a CoprocessorFunction."""
        fn = triton_decoder(
            (_echo_decoder,),
            platform=_cuda_platform(),
            compiler=shutil.which("nvcc") or "nvcc",
        )

        assert isinstance(fn, CoprocessorFunction)
        assert fn.name.endswith("_catalyst")
        assert fn.lib_path is not None
        assert fn.lib_path.endswith(".so")

    def test_css_bp_decoder_returns_a_coprocessor_function(self):
        """``qp.backline.css_bp_decoder`` compiles and wraps the result in a CoprocessorFunction."""
        hx = np.array([[1, 0], [0, 1]], dtype=int)
        hz = np.array([[1, 1], [0, 1]], dtype=int)

        fn = css_bp_decoder(
            hx,
            hz,
            postprocess="hard",
            num_iters=5,
            prob=0.1,
            platform=_cuda_platform(),
            grid=(1, 1, 1),
            num_warps=1,
            num_stages=1,
            compiler=shutil.which("nvcc") or "nvcc",
        )

        assert isinstance(fn, CoprocessorFunction)
        assert fn.name.endswith("_catalyst")
        assert fn.lib_path is not None
        assert fn.lib_path.endswith(".so")
