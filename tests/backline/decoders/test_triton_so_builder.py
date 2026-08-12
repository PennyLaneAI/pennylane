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

from pathlib import Path
from types import SimpleNamespace

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

from pennylane.backline.decoders.triton import triton_so_builder as builder


class TestBuildSo:
    """Tests for _build_so."""

    @pytest.mark.parametrize(
        ("grid", "platform", "message"),
        [
            ((1, 1), "hip:gfx942:64", "exactly 3 dimensions"),
            ((1, 1, 1), "hip:gfx942", "backend:arch:warp"),
            ((1, 1, 1), "cpu:80:32", "backend must be 'cuda' or 'hip'"),
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

    def test_build_so_builds_hip_command_and_appends_catalyst_wrapper(self, monkeypatch, tmp_path):
        """It should add stdlib.h and a Catalyst ABI wrapper to the HIP source."""
        generated_c = tmp_path / "generated.c"
        generated_c.write_text("#include <stdio.h>\nint kernel(void) { return 0; }\n")

        calls = {}

        monkeypatch.setattr(
            builder,
            "_compile_kernel",
            lambda *args, **kwargs: ("hip", "decoder_symbol", generated_c),
        )

        def fake_run(cmd, check):
            calls["cmd"] = cmd
            calls["check"] = check
            calls["source"] = Path(cmd[6]).read_text(encoding="utf-8")

        monkeypatch.setattr(builder.subprocess, "run", fake_run)

        out = tmp_path / "decoder.so"
        so_path, symbol_name = builder._build_so(
            object(),
            signature={},
            constexpr={},
            grid=(1, 1, 1),
            platform="hip:gfx942:64",
            num_warps=1,
            num_stages=1,
            out=str(out),
            compiler="hipcc-custom",
            cflags=("-Wall",),
        )

        assert so_path == out.resolve()
        assert symbol_name == "decoder_symbol_catalyst"
        assert calls["check"] is True
        assert calls["cmd"][0] == "hipcc-custom"
        assert calls["cmd"][1:6] == ["-fPIC", "-shared", "-O3", "-o", str(out.resolve())]
        assert calls["cmd"][-1] == "-Wall"
        assert calls["source"].startswith("#include <stdlib.h>\n#include <stdio.h>\n")
        assert "typedef struct {" in calls["source"]
        assert (
            "int decoder_symbol_catalyst(const CoprocLaunchDescCompat *desc, void *ctx)"
            in calls["source"]
        )
        assert "desc->ring_slots" in calls["source"]
        assert "(hipStream_t)desc->stream" in calls["source"]
        assert "int rc = decoder_symbol(" in calls["source"]

    def test_build_so_patches_source_at_most_once(self, monkeypatch, tmp_path):
        """It should not duplicate stdlib.h if the generated source already has it."""
        generated_c = tmp_path / "generated.c"
        generated_c.write_text("#include <stdlib.h>\n#include <stdio.h>\n")

        seen = {}
        monkeypatch.setattr(
            builder,
            "_compile_kernel",
            lambda *args, **kwargs: ("hip", "decoder_symbol", generated_c),
        )

        def fake_run(cmd, check):
            assert check is True
            seen["source"] = Path(cmd[6]).read_text(encoding="utf-8")

        monkeypatch.setattr(builder.subprocess, "run", fake_run)

        builder._build_so(
            object(),
            signature={},
            constexpr={},
            grid=(1, 1, 1),
            platform="hip:gfx942:64",
            num_warps=1,
            num_stages=1,
            out=str(tmp_path / "decoder.so"),
            compiler="hipcc-custom",
            cflags=(),
        )

        assert seen["source"].count("#include <stdlib.h>") == 1
        assert seen["source"].startswith("#include <stdlib.h>\n#include <stdio.h>\n")

    def test_build_so_adds_nvcc_wrapper_and_cuda_link_flag(self, monkeypatch, tmp_path):
        """It should wrap fPIC, add libcuda, and append a CUDA Catalyst wrapper."""
        generated_c = tmp_path / "generated.c"
        generated_c.write_text("#include <stdio.h>\nint kernel(void) { return 0; }\n")

        seen = {}
        monkeypatch.setattr(
            builder,
            "_compile_kernel",
            lambda *args, **kwargs: ("cuda", "decoder_symbol", generated_c),
        )

        def fake_run(cmd, check):
            assert check is True
            seen["cmd"] = cmd
            seen["source"] = Path(cmd[7]).read_text(encoding="utf-8")

        monkeypatch.setattr(builder.subprocess, "run", fake_run)

        builder._build_so(
            object(),
            signature={},
            constexpr={},
            grid=(1, 1, 1),
            platform="cuda:80:32",
            num_warps=1,
            num_stages=1,
            out=str(tmp_path / "decoder.so"),
            compiler="/usr/local/cuda/bin/nvcc",
            cflags=(),
        )

        assert seen["cmd"][:4] == ["/usr/local/cuda/bin/nvcc", "-Xcompiler", "-fPIC", "-shared"]
        assert "-lcuda" in seen["cmd"]
        assert (
            "int decoder_symbol_catalyst(const CoprocLaunchDescCompat *desc, void *ctx)"
            in seen["source"]
        )
        assert "int rc = decoder_symbol(" in seen["source"]

    def test_build_so_uses_backend_default_compiler(self, monkeypatch, tmp_path):
        """It should fall back to HIPCC when no compiler override is provided."""
        generated_c = tmp_path / "generated.c"
        generated_c.write_text("int kernel(void) { return 0; }\n")

        monkeypatch.setattr(
            builder,
            "_compile_kernel",
            lambda *args, **kwargs: ("hip", "decoder_symbol", generated_c),
        )
        monkeypatch.setenv("HIPCC", "hipcc-from-env")

        seen = {}

        def fake_run(cmd, check):
            assert check is True
            seen["cmd"] = cmd

        monkeypatch.setattr(builder.subprocess, "run", fake_run)

        builder._build_so(
            object(),
            signature={},
            constexpr={},
            grid=(1, 1, 1),
            platform="hip:gfx942:64",
            num_warps=1,
            num_stages=1,
            out=str(tmp_path / "decoder.so"),
            compiler="",
            cflags=(),
        )

        assert seen["cmd"][0] == "hipcc-from-env"

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
        assert func_name == "decoder_kernel_abcdef123456"
        assert func_name in generated_c.read_text(encoding="utf-8")
        generated_c.unlink(missing_ok=True)
