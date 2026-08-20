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

"""Tests for backline coprocessor functions."""

# pylint: disable=too-few-public-methods

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

import pennylane as qp
from pennylane.backline import CoprocessorFunction, css_bp_decoder, triton_decoder

_DECODER_FRONTEND = "pennylane.backline.decoders.triton.decoder_frontend"


class TestCoprocessorFunction:
    """Tests for the CoprocessorFunction handle."""

    def test_symbol_name_defaults_to_name(self):
        fn = CoprocessorFunction("decode")
        assert fn.name == "decode"
        assert fn.lib_path is None
        assert fn.symbol_name == "decode"

    def test_lib_path(self):
        fn = CoprocessorFunction("decode", lib_path="/opt/lib/libdecode.so")
        assert fn.lib_path == "/opt/lib/libdecode.so"

    def test_the_dataclass_is_frozen(self):
        """Attribute assignment on a coprocessor function is refused."""
        fn = CoprocessorFunction("decode")
        with pytest.raises(Exception):
            fn.name = "renamed"  # type: ignore[misc]

    def test_two_equal_handles_compare_equal(self):
        """Same name and library means same handle."""
        assert CoprocessorFunction("decode", lib_path="/a.so") == CoprocessorFunction(
            "decode", lib_path="/a.so"
        )


class TestTritonDecoder:
    """The Triton decoder compilation entry point."""

    def test_missing_triton_raises_import_error(self, monkeypatch):
        """The message points the user at installing triton, and wraps the original cause."""
        monkeypatch.setitem(sys.modules, _DECODER_FRONTEND, None)
        with pytest.raises(ImportError, match="Triton decoders require installed"):
            triton_decoder((object(),))

    def test_the_wrapper_reexports_from_backline(self):
        """The public name is exported from pennylane.backline."""
        assert qp.backline.triton_decoder is triton_decoder

    def test_accepts_plain_python_functions_and_unique_names_them(self, monkeypatch):
        """Un-jitted Triton functions are jitted internally under unique generated names."""
        pytest.importorskip("triton")
        from pennylane.backline.decoders.triton import decoder_frontend as frontend

        captured = {}

        def fake_build_so(*_args, **kwargs):
            captured["names"] = [fn.fn.__name__ for fn in kwargs["constexpr"]["decoder_fns"]]
            return Path("/tmp/fake.so"), "fake_symbol"

        monkeypatch.setattr(frontend, "_build_so", fake_build_so)

        def make_decoder():
            def decode(syndrome):
                return syndrome

            return decode

        fn = triton_decoder((make_decoder(), make_decoder()), platform="cuda:80:32")

        assert isinstance(fn, CoprocessorFunction)
        assert captured["names"] == ["decode_0", "decode_1"]

    def test_rejects_already_jitted_functions(self):
        """The public API owns jitting and rejects pre-jitted kernels."""
        triton = pytest.importorskip("triton")
        import triton.language as tl

        @triton.jit
        def decode(syndrome):
            return tl.where(syndrome != 0, 1 << (syndrome - 1), 0)

        with pytest.raises(TypeError, match="already-jitted Triton functions"):
            triton_decoder((decode,), platform="cuda:80:32")


class TestCssBpDecoder:
    """The CSS belief-propagation decoder compilation entry point."""

    def test_missing_triton_raises_import_error(self, monkeypatch):
        """The message points the user at installing triton, and wraps the original cause."""
        monkeypatch.setitem(sys.modules, _DECODER_FRONTEND, None)
        Hx = Hz = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        with pytest.raises(ImportError, match="Triton decoders require installed"):
            css_bp_decoder(Hx, Hz)

    def test_the_wrapper_reexports_from_backline(self):
        """The public name is exported from pennylane.backline."""
        assert qp.backline.css_bp_decoder is css_bp_decoder

    def test_same_shape_matrices_get_distinct_decoder_names(self, monkeypatch):
        """Hx and Hz specializations stay distinct even when their shapes match."""
        pytest.importorskip("triton")
        from pennylane.backline.decoders.triton import decoder_frontend as frontend

        captured = {}

        def fake_build_so(*_args, **kwargs):
            captured["names"] = [fn.fn.__name__ for fn in kwargs["constexpr"]["decoder_fns"]]
            return Path("/tmp/fake.so"), "fake_symbol"

        monkeypatch.setattr(frontend, "_build_so", fake_build_so)

        hx = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        hz = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8)

        fn = css_bp_decoder(hx, hz, platform="cuda:80:32")

        assert isinstance(fn, CoprocessorFunction)
        assert len(set(captured["names"])) == 2


class TestTritonSubmoduleImportGuards:
    """Each triton submodule raises a helpful ImportError when triton is missing.

    Every submodule under ``pennylane.backline.decoders.triton`` wraps its
    ``import triton`` in a try/except that re-raises with a message directing the user to install
    the package. On a system without triton, an accidental import should hit that branch.
    """

    @pytest.mark.parametrize(
        "module_name",
        [
            "pennylane.backline.decoders.triton.algorithms",
            "pennylane.backline.decoders.triton.bp_iters",
            "pennylane.backline.decoders.triton.decoder_frontend",
            "pennylane.backline.decoders.triton.persistent_kernel",
            "pennylane.backline.decoders.triton.triton_so_builder",
        ],
    )
    def test_missing_triton_re_raises_with_install_hint(self, monkeypatch, module_name):
        """Importing the module without triton points at installing it."""
        # Force ``import triton`` to fail from a fresh import of the target submodule.
        monkeypatch.setitem(sys.modules, "triton", None)
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        with pytest.raises(ImportError, match="Triton decoders require installed"):
            importlib.import_module(module_name)
