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

"""Tests for the Triton shared-library builder."""

# pylint: disable=protected-access,wrong-import-position,broad-exception-caught

import ctypes
import shutil
from types import SimpleNamespace

import pytest

triton = pytest.importorskip("triton")


def _current_target():
    """Return the active Triton target when available."""
    try:
        return triton.runtime.driver.active.get_current_target()
    except Exception:
        return None


def _has_backend_target(backend: str) -> bool:
    """Return whether the active Triton target matches ``backend``."""
    target = _current_target()
    return target is not None and target.backend == backend


def _platform_for_backend(backend: str, fallback_arch: str, fallback_warp_size: int) -> str:
    """Build a platform string for the active target or a fallback value."""
    target = _current_target()
    if target is not None and target.backend == backend:
        return f"{backend}:{target.arch}:{target.warp_size}"
    return f"{backend}:{fallback_arch}:{fallback_warp_size}"


pytestmark = [pytest.mark.gpu]

from pennylane.backline.decoders.triton import triton_so_builder as builder
from pennylane.backline.decoders.triton.persistent_kernel import _persistent_decoder_kernel


@triton.jit
def _echo_decoder(syndrome):
    return syndrome


class TestBuildSo:
    """Tests for _build_so."""

    @pytest.mark.parametrize(
        ("grid", "platform", "message"),
        [
            ((1, 1), "hip:gfx942:64", "exactly 3 dimensions"),
            ((1, 1, 1), "hip:gfx942", "backend:arch:warp"),
            ((1, 1, 1), "cpu:80:32", "backend must be 'cuda' or 'hip'"),
            ((1, 1, 1), "cuda::32", "arch must be non-empty"),
            ((1, 1, 1), "cuda:80:abc", "warp size must be an integer"),
            ((1, 1, 1), "cuda:80:64", "warp size must be 32 for cuda"),
            ((1, 1, 1), "hip:gfx942:32", "warp size must be 64 for hip"),
        ],
    )
    def test_build_so_validates_shape_inputs(self, tmp_path, grid, platform, message):
        """It should reject malformed grid and platform values."""
        with pytest.raises(ValueError, match=message):
            builder._build_so(
                object(),
                signature={},
                constexpr={},
                grid=grid,
                platform=platform,
                num_warps=1,
                num_stages=1,
                out=str(tmp_path / "decoder.so"),
                compiler="",
                cflags=(),
            )

    @pytest.mark.skipif(
        not _has_backend_target("hip"), reason="Triton decoder tests require a HIP device"
    )
    @pytest.mark.skipif(shutil.which("hipcc") is None, reason="hipcc compiler not available")
    def test_build_so_compiles_hip_shared_library(self, tmp_path):
        """It should compile a HIP shared library with a Catalyst ABI wrapper."""
        out = tmp_path / "decoder.so"
        so_path, symbol_name = builder._build_so(
            _persistent_decoder_kernel,
            signature={
                "ring_u64_ptr": "*u64",
                "handoff_u64_ptr": "*u64",
                "stop_u32_ptr": "*u32",
                "ring_slots": "u32",
                "total": "u64",
            },
            constexpr={"decoder_fns": (_echo_decoder,)},
            grid=(1, 1, 1),
            platform=_platform_for_backend("hip", fallback_arch="gfx942", fallback_warp_size=64),
            num_warps=1,
            num_stages=1,
            out=str(out),
            compiler=shutil.which("hipcc") or "hipcc",
            cflags=(),
        )

        assert so_path == out.resolve()
        assert so_path.exists()
        assert symbol_name.endswith("_catalyst")
        lib = ctypes.CDLL(str(so_path))
        assert getattr(lib, symbol_name)

    @pytest.mark.skipif(
        not _has_backend_target("cuda"), reason="Triton decoder CUDA tests require a CUDA device"
    )
    @pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc compiler not available")
    def test_build_so_compiles_cuda_shared_library(self, tmp_path):
        """It should compile a CUDA shared library with a Catalyst ABI wrapper."""
        out = tmp_path / "decoder.so"
        so_path, symbol_name = builder._build_so(
            _persistent_decoder_kernel,
            signature={
                "ring_u64_ptr": "*u64",
                "handoff_u64_ptr": "*u64",
                "stop_u32_ptr": "*u32",
                "ring_slots": "u32",
                "total": "u64",
            },
            constexpr={"decoder_fns": (_echo_decoder,)},
            grid=(1, 1, 1),
            platform=_platform_for_backend("cuda", fallback_arch="80", fallback_warp_size=32),
            num_warps=1,
            num_stages=1,
            out=str(out),
            compiler=shutil.which("nvcc") or "nvcc",
            cflags=(),
        )

        assert so_path == out.resolve()
        assert so_path.exists()
        assert symbol_name.endswith("_catalyst")
        lib = ctypes.CDLL(str(so_path))
        assert getattr(lib, symbol_name)

    def test_compile_kernel_suffixes_generated_symbol_with_ast_hash(self, monkeypatch):
        """It should derive the launcher symbol from the Triton AST hash."""

        def _ast_source_init(self, fn, constexprs, signature, attrs):
            self.fn = fn
            self.constants = constexprs
            self.signature = signature
            self.attrs = attrs

        def _ast_hash(_self):
            return "abcdef1234567890"

        def _create_binder(_self):
            return None

        def _parse_options(_self, _):
            return SimpleNamespace()

        fake_ast_source = type(
            "FakeASTSource",
            (),
            {"__init__": _ast_source_init, "hash": _ast_hash},
        )
        fake_kernel = type(
            "FakeKernel",
            (),
            {
                "__name__": "decoder_kernel",
                "arg_names": ["ring_u64_ptr"],
                "ASTSource": fake_ast_source,
                "create_binder": _create_binder,
            },
        )()
        fake_backend = type(
            "FakeBackend",
            (),
            {"binary_ext": "cubin", "parse_options": _parse_options},
        )()
        compile_result = SimpleNamespace(
            metadata=SimpleNamespace(shared=0, profile_scratch_size=0, global_scratch_size=0),
            asm={"cubin": b"\x00\x01"},
        )

        monkeypatch.setattr(builder.triton.compiler, "make_backend", lambda _: fake_backend)
        monkeypatch.setattr(builder.triton, "compile", lambda *args, **kwargs: compile_result)

        backend, func_name, generated_c = builder._compile_kernel(
            fake_kernel,
            signature={"ring_u64_ptr": "*u64"},
            constexpr={},
            grid=(1, 1, 1),
            platform="cuda:80:32",
            num_warps=1,
            num_stages=1,
        )

        assert backend == "cuda"
        # Verify that _compile_kernel uses the AST hash in the generated symbol,
        # rather than only confirming that distinct kernels produce distinct hashes.
        assert func_name == "decoder_kernel_abcdef123456"
        assert func_name in generated_c.read_text(encoding="utf-8")
        generated_c.unlink(missing_ok=True)


class TestCatalystWrapperSource:
    """The wrapper source builder for the Catalyst launcher ABI."""

    def test_an_unknown_backend_is_refused(self):
        """A backend outside {cuda, hip} does not have a known Catalyst ABI mapping."""
        with pytest.raises(ValueError, match="unsupported backend for Catalyst wrapper: cpu"):
            builder._make_catalyst_wrapper_source("cpu", "sym")


class TestCompileKernelTemplateMissing:
    """The generated launcher source has to come from Triton's compile templates."""

    def test_missing_c_template_is_reported(self, tmp_path, monkeypatch):
        """Without a ``.c`` template on disk, the missing launcher source is flagged.

        This defends against a broken or stripped-down triton install: the loop over
        ``template_dir.glob("compile.*")`` finds nothing and ``generated_c`` stays ``None``.
        """
        # Point ``triton_compile_tool.__file__`` at a fake tools module whose ``extra/<backend>``
        # directory holds no ``compile.c`` template.
        fake_tools = tmp_path / "tools" / "__init__.py"
        fake_tools.parent.mkdir(parents=True)
        fake_tools.write_text("", encoding="utf-8")
        (tmp_path / "tools" / "extra" / "cuda").mkdir(parents=True)

        monkeypatch.setattr(builder.triton_compile_tool, "__file__", str(fake_tools), raising=False)

        # Reuse the mocking approach from the AST-hash test above so triton.compile is stubbed.
        def _ast_source_init(self, fn, constexprs, signature, attrs):
            self.fn = fn
            self.constants = constexprs
            self.signature = signature
            self.attrs = attrs

        fake_ast_source = type(
            "FakeASTSource",
            (),
            {"__init__": _ast_source_init, "hash": lambda _self: "0" * 16},
        )
        fake_kernel = type(
            "FakeKernel",
            (),
            {
                "__name__": "decoder_kernel",
                "arg_names": ["ring_u64_ptr"],
                "ASTSource": fake_ast_source,
                "create_binder": lambda _self: None,
            },
        )()
        fake_backend = type(
            "FakeBackend",
            (),
            {"binary_ext": "cubin", "parse_options": lambda _self, _: SimpleNamespace()},
        )()
        compile_result = SimpleNamespace(
            metadata=SimpleNamespace(shared=0, profile_scratch_size=0, global_scratch_size=0),
            asm={"cubin": b"\x00\x01"},
        )

        monkeypatch.setattr(builder.triton.compiler, "make_backend", lambda _: fake_backend)
        monkeypatch.setattr(builder.triton, "compile", lambda *args, **kwargs: compile_result)

        with pytest.raises(RuntimeError, match="expected Triton compile templates to generate .c"):
            builder._compile_kernel(
                fake_kernel,
                signature={"ring_u64_ptr": "*u64"},
                constexpr={},
                grid=(1, 1, 1),
                platform="cuda:80:32",
                num_warps=1,
                num_stages=1,
            )


class TestCompileKernelScratchLimits:
    """AOT compilation refuses kernels that need scratch space.

    Global- and profile-scratch requirements are reported by Triton on the compile result. The
    launcher wrapper has no place to allocate that memory, so ``_compile_kernel`` refuses the
    kernel rather than emit a launcher that would trap on use.
    """

    @pytest.mark.parametrize(
        ("scratch_field", "message"),
        [
            ("global_scratch_size", "global scratch requirements"),
            ("profile_scratch_size", "profile scratch requirements"),
        ],
    )
    def test_a_kernel_with_scratch_is_refused(self, monkeypatch, scratch_field, message):
        """A non-zero scratch size on the compile result is reported, not silently dropped."""

        def _ast_source_init(self, fn, constexprs, signature, attrs):
            self.fn = fn
            self.constants = constexprs
            self.signature = signature
            self.attrs = attrs

        fake_ast_source = type(
            "FakeASTSource",
            (),
            {"__init__": _ast_source_init, "hash": lambda _self: "0" * 16},
        )
        fake_kernel = type(
            "FakeKernel",
            (),
            {
                "__name__": "decoder_kernel",
                "arg_names": ["ring_u64_ptr"],
                "ASTSource": fake_ast_source,
                "create_binder": lambda _self: None,
            },
        )()
        fake_backend = type(
            "FakeBackend",
            (),
            {"binary_ext": "cubin", "parse_options": lambda _self, _: SimpleNamespace()},
        )()
        metadata_kwargs = {"shared": 0, "profile_scratch_size": 0, "global_scratch_size": 0}
        metadata_kwargs[scratch_field] = 1
        compile_result = SimpleNamespace(
            metadata=SimpleNamespace(**metadata_kwargs),
            asm={"cubin": b"\x00\x01"},
        )

        monkeypatch.setattr(builder.triton.compiler, "make_backend", lambda _: fake_backend)
        monkeypatch.setattr(builder.triton, "compile", lambda *args, **kwargs: compile_result)

        with pytest.raises(RuntimeError, match=message):
            builder._compile_kernel(
                fake_kernel,
                signature={"ring_u64_ptr": "*u64"},
                constexpr={},
                grid=(1, 1, 1),
                platform="cuda:80:32",
                num_warps=1,
                num_stages=1,
            )
