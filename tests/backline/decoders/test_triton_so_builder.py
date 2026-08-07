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
        ("grid", "target", "message"),
        [
            ((1, 1), "hip:gfx90a:64", "exactly 3 dimensions"),
            ((1, 1, 1), "hip:gfx90a", "backend:arch:warp"),
        ],
    )
    def test_build_so_validates_shape_inputs(self, tmp_path, grid, target, message):
        """It should reject malformed grid and target values."""
        with pytest.raises(ValueError, match=message):
            builder.build_so(
                object(),
                signature={},
                constexpr={},
                grid=grid,
                target=target,
                out=str(tmp_path / "decoder.so"),
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
            target="hip:gfx90a:64",
            out=str(out),
            compiler="hipcc-custom",
            cflags=("-Wall",),
        )

        assert so_path == out.resolve()
        assert symbol_name == "decoder_symbol"
        assert calls["check"] is True
        assert calls["cmd"][0] == "hipcc-custom"
        assert calls["cmd"][1:6] == ["-fPIC", "-shared", "-O2", "-o", str(out.resolve())]
        assert calls["cmd"][-1] == "-Wall"
        assert calls["source"].startswith("#include <stdlib.h>\n#include <stdio.h>\n")

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
            target="hip:gfx90a:64",
            out=str(tmp_path / "decoder.so"),
        )

        assert seen["cmd"][0] == "hipcc-from-env"


class TestHelpers:
    """Tests for helper functions in triton_so_builder."""

    def test_make_target_parses_cuda_numeric_arch(self):
        """CUDA targets should parse the architecture as an integer."""
        target = builder._make_target("cuda:80:32")

        assert target.backend == "cuda"
        assert target.arch == 80
        assert target.warp_size == 32

    def test_make_target_parses_hip_string_arch(self):
        """HIP targets should keep the architecture string as-is."""
        target = builder._make_target("hip:gfx90a:64")

        assert target.backend == "hip"
        assert target.arch == "gfx90a"
        assert target.warp_size == 64

    def test_copy_generated_source_is_idempotent(self, tmp_path):
        """It should prepend stdlib.h once and leave it alone after that."""
        source = tmp_path / "source.c"
        dest = tmp_path / "dest.c"
        source.write_text("#include <stdio.h>\n")

        builder._copy_generated_source(source, dest)
        first = dest.read_text()
        assert first == "#include <stdlib.h>\n#include <stdio.h>\n"

        builder._copy_generated_source(dest, dest)
        assert dest.read_text() == first

    def test_backend_c_type_for_signature_rejects_unknown_backend(self):
        """Only CUDA and HIP backends are supported."""
        with pytest.raises(ValueError, match="unsupported backend"):
            builder._backend_c_type_for_signature("cpu", "u64")

    def test_backend_c_type_for_signature_strips_layout_suffix(self):
        """Signature layout suffixes should be ignored before backend mapping."""
        assert builder._backend_c_type_for_signature("cuda", "u64:16") == "uint64_t"
