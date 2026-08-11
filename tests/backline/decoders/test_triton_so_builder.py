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
    """Tests for build_so."""

    @pytest.mark.parametrize(
        ("grid", "platform", "message"),
        [
            ((1, 1), "hip:gfx942:64", "exactly 3 dimensions"),
            ((1, 1, 1), "hip:gfx942", "backend:arch:warp"),
        ],
    )
    def test_build_so_validates_shape_inputs(self, tmp_path, grid, platform, message):
        """It should reject malformed grid and platform values."""
        with pytest.raises(ValueError, match=message):
            builder.build_so(
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

    def test_build_so_builds_hip_command_and_patches_source(self, monkeypatch, tmp_path):
        """It should add stdlib.h and pass the expected HIP compile command."""
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
        so_path, symbol_name = builder.build_so(
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
        assert symbol_name == "decoder_symbol"
        assert calls["check"] is True
        assert calls["cmd"][0] == "hipcc-custom"
        assert calls["cmd"][1:6] == ["-fPIC", "-shared", "-O3", "-o", str(out.resolve())]
        assert calls["cmd"][-1] == "-Wall"
        assert calls["source"].startswith("#include <stdlib.h>\n#include <stdio.h>\n")

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

        builder.build_so(
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

        assert seen["source"] == "#include <stdlib.h>\n#include <stdio.h>\n"

    def test_build_so_adds_nvcc_wrapper_and_cuda_link_flag(self, monkeypatch, tmp_path):
        """It should wrap fPIC for nvcc and link against libcuda for CUDA launchers."""
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

        monkeypatch.setattr(builder.subprocess, "run", fake_run)

        builder.build_so(
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

        builder.build_so(
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
