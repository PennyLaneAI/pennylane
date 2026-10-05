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

import base64
import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import pennylane as qp
from pennylane.backline import CoprocessorFunction, css_bp_decoder, onnx_decoder, triton_decoder
from pennylane.backline.functions import _onnx_message_bytes

_DECODER_FRONTEND = "pennylane.backline.decoders.triton.decoder_frontend"

# An ONNX model whose single uint8[1, 8] input passes through Identity to its uint8[1, 8] output.
_IDENTITY_U8X8_ONNX = (
    "CAgSDmNhdGFseXN0LXRlc3RzOksKEAoBeBIBeSIISWRlbnRpdHkSDWlkZW50aXR5X3U4eDhaEwoBeBIOCgwIAhIICgIIAQoC"
    "CAhiEwoBeRIOCgwIAhIICgIIAQoCCAhCBAoAEA0="
)


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

    def test_a_function_declares_no_files_by_default(self):
        """A CoprocessorFunction built by hand names no local files in its config."""
        assert CoprocessorFunction("fn").files == ()

    def test_files_given_as_a_list_are_stored_as_a_tuple(self):
        """The declared file keys are kept as a tuple, like the other sequence fields."""
        assert CoprocessorFunction("fn", config="table=t.cfg", files=["table"]).files == ("table",)

    def test_extra_files_are_stored_as_a_tuple_of_paths(self):
        """Extra files default to none, and are kept as a tuple of path strings."""
        assert CoprocessorFunction("fn").extra_files == ()
        extra = CoprocessorFunction("fn", extra_files=[Path("/opt/a.so")]).extra_files
        assert extra == ("/opt/a.so",)

    def test_message_bytes_given_as_a_list_is_stored_as_a_tuple(self):
        """A declared size pair is kept as a tuple, so placements can compare and hash it."""
        assert CoprocessorFunction("fn", message_bytes=[120, 121]).message_bytes == (120, 121)


class TestOnnxDecoder:
    """The ONNX coprocessor function and the config it carries."""

    @pytest.fixture(autouse=True)
    def model_sizes(self, monkeypatch):
        """Stand in for reading a model, whose files here are empty, as uint8[120] to uint8[121]."""
        monkeypatch.setattr(
            "pennylane.backline.functions._onnx_message_bytes", lambda model: (120, 121)
        )

    @pytest.fixture
    def model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pennylane.backline.functions._onnxruntime_library", lambda: "/opt/libonnxruntime.so"
        )
        path = tmp_path / "model.onnx"
        path.write_bytes(b"")
        return path

    def test_the_function_declares_the_model_tensor_sizes(self, model):
        """The function declares the model's input and output sizes as its message sizes."""
        assert onnx_decoder(model).message_bytes == (120, 121)

    def test_the_onnx_function_declares_its_model_and_onnxruntime_as_files(self, model):
        """The model and the onnxruntime library are the files a remote coprocessor needs."""
        assert onnx_decoder(model).files == ("model", "ort_lib")

    def test_the_model_tensor_sizes_are_read_with_onnxruntime(self, tmp_path):
        """A uint8[1, 8] to uint8[1, 8] identity model declares 8 B in and 8 B out."""
        pytest.importorskip("onnxruntime")
        path = tmp_path / "identity.onnx"
        path.write_bytes(base64.b64decode(_IDENTITY_U8X8_ONNX))
        assert _onnx_message_bytes(path) == (8, 8)

    def test_config_defaults_to_empty(self):
        """A CoprocessorFunction built by hand carries no config."""
        assert CoprocessorFunction("fn").config == ""

    def test_the_function_is_catalysts_own(self, model):
        """The function is Catalyst's ONNX coprocessor function, so it needs no lib_path."""
        fn = onnx_decoder(model)
        assert fn.name == "catalyst_onnx_coprocessor"
        assert fn.lib_path is None

    def test_the_function_runs_per_message(self, model):
        """The ONNX function is a host function, so a GPU coprocessor calls it per message."""
        assert onnx_decoder(model).per_message
        assert not CoprocessorFunction("fn").per_message

    def test_the_provider_defaults_to_auto(self, model):
        """With no provider given, onnxruntime picks the GPU it has, or the CPU."""
        assert (
            f"model={model.resolve()};ort_lib=/opt/libonnxruntime.so;provider=auto;device=0;threads=1"
            == onnx_decoder(model).config
        )

    @pytest.mark.parametrize("provider", ["cpu", "migraphx", "cuda", "tensorrt", "rocm"])
    def test_a_named_provider_is_passed_on(self, model, provider):
        assert (
            f"provider={provider};device=1"
            in onnx_decoder(model, provider=provider, device=1).config
        )

    def test_threads_are_passed_on(self, model):
        """The intra-op thread count reaches the function's config."""
        assert onnx_decoder(model, threads=8).config.endswith(";threads=8")

    @pytest.mark.parametrize("name", ["device", "threads"])
    @pytest.mark.parametrize("value", [True, 1.0, "1"])
    def test_a_count_that_is_not_an_int_raises(self, model, name, value):
        """device and threads must be ints, and a bool is not accepted as one."""
        with pytest.raises(TypeError, match=f"{name} must be an int"):
            onnx_decoder(model, **{name: value})

    @pytest.mark.parametrize("name, value", [("device", -1), ("threads", 0)])
    def test_a_count_out_of_range_raises(self, model, name, value):
        """device starts at 0, and threads at 1."""
        with pytest.raises(ValueError, match=f"{name} must be at least"):
            onnx_decoder(model, **{name: value})

    def test_the_installed_onnxruntime_is_found_on_macos(self, monkeypatch, tmp_path):
        """On macOS the library is the package's .dylib."""
        package = tmp_path / "onnxruntime"
        (package / "capi").mkdir(parents=True)
        (package / "capi" / "libonnxruntime.1.2.3.dylib").write_bytes(b"")
        (package / "capi" / "libonnxruntime_providers_shared.dylib").write_bytes(b"")
        spec = importlib.machinery.ModuleSpec("onnxruntime", None, is_package=True)
        spec.submodule_search_locations = [str(package)]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        monkeypatch.setattr(sys, "platform", "darwin")
        model = tmp_path / "model.onnx"
        model.write_bytes(b"")
        assert f"ort_lib={package / 'capi' / 'libonnxruntime.1.2.3.dylib'}" in (
            onnx_decoder(model).config
        )

    def test_an_onnxruntime_without_its_library_raises(self, monkeypatch, tmp_path):
        """An onnxruntime package whose capi directory has no shared library is an error."""
        package = tmp_path / "onnxruntime"
        (package / "capi").mkdir(parents=True)
        spec = importlib.machinery.ModuleSpec("onnxruntime", None, is_package=True)
        spec.submodule_search_locations = [str(package)]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        model = tmp_path / "model.onnx"
        model.write_bytes(b"")
        with pytest.raises(ImportError, match="no onnxruntime shared library found"):
            onnx_decoder(model)

    def test_the_installed_onnxruntime_is_found(self, monkeypatch, tmp_path):
        """The library is the one in the installed onnxruntime package, and its execution provider
        libraries beside it travel with it as extra files."""
        package = tmp_path / "onnxruntime"
        (package / "capi").mkdir(parents=True)
        (package / "capi" / "libonnxruntime.so.1.2.3").write_bytes(b"")
        (package / "capi" / "libonnxruntime_providers_shared.so").write_bytes(b"")
        (package / "capi" / "onnxruntime_pybind11_state.so").write_bytes(b"")
        spec = importlib.machinery.ModuleSpec("onnxruntime", None, is_package=True)
        spec.submodule_search_locations = [str(package)]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        model = tmp_path / "model.onnx"
        model.write_bytes(b"")
        fn = onnx_decoder(model)
        assert f"ort_lib={package / 'capi' / 'libonnxruntime.so.1.2.3'}" in fn.config
        assert fn.extra_files == (str(package / "capi" / "libonnxruntime_providers_shared.so"),)

    def test_missing_onnxruntime_raises_import_error(self, monkeypatch, tmp_path):
        """With no onnxruntime installed, the error says what to install."""
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        model = tmp_path / "model.onnx"
        model.write_bytes(b"")
        with pytest.raises(ImportError, match="onnxruntime-migraphx"):
            onnx_decoder(model)

    @pytest.mark.usefixtures("model")
    def test_missing_model_raises(self, tmp_path):
        """The model is checked when the function is built, not when the coprocessor starts."""
        with pytest.raises(FileNotFoundError, match="no model"):
            onnx_decoder(tmp_path / "absent.onnx")

    def test_unknown_provider_raises(self, model):
        with pytest.raises(ValueError, match="provider must be one of"):
            onnx_decoder(model, provider="tpu")

    @pytest.mark.usefixtures("model")
    def test_a_path_with_the_separator_raises(self, tmp_path):
        """A ';' in a path would split the config, so it is rejected."""
        odd = tmp_path / "a;b.onnx"
        odd.write_bytes(b"")
        with pytest.raises(ValueError, match="must not contain ';'"):
            onnx_decoder(odd)


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

    def test_accepts_plain_python_functions_and_unique_names_them(self, monkeypatch, tmp_path):
        """Un-jitted Triton functions are jitted internally under unique generated names."""
        pytest.importorskip("triton")
        from pennylane.backline.decoders.triton import decoder_frontend as frontend

        captured = {}
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        real_mkdtemp = frontend.tempfile.mkdtemp

        def fake_mkdtemp(*args, **kwargs):
            prefix = kwargs.get("prefix")
            if prefix is None and len(args) >= 2:
                prefix = args[1]
            return str(scratch) if prefix == "pl_triton_decoder_" else real_mkdtemp(*args, **kwargs)

        def fake_build_so(*_args, **kwargs):
            captured["qualnames"] = [
                fn.fn.__qualname__ for fn in kwargs["constexpr"]["decoder_fns"]
            ]
            return scratch / "fake.so", "fake_symbol"

        monkeypatch.setattr(frontend.tempfile, "mkdtemp", fake_mkdtemp)
        monkeypatch.setattr(frontend, "_build_so", fake_build_so)

        def make_decoder():
            def decode(syndrome):
                return syndrome

            return decode

        fn = triton_decoder((make_decoder(), make_decoder()), platform="cuda:80:32")

        assert isinstance(fn, CoprocessorFunction)
        assert captured["qualnames"] == ["decode_0", "decode_1"]

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

    def test_same_shape_matrices_get_distinct_decoder_names(self, monkeypatch, tmp_path):
        """Hx and Hz specializations stay distinct even when their shapes match."""
        pytest.importorskip("triton")
        from pennylane.backline.decoders.triton import decoder_frontend as frontend

        captured = {}
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        real_mkdtemp = frontend.tempfile.mkdtemp

        def fake_mkdtemp(*args, **kwargs):
            prefix = kwargs.get("prefix")
            if prefix is None and len(args) >= 2:
                prefix = args[1]
            return str(scratch) if prefix == "pl_triton_decoder_" else real_mkdtemp(*args, **kwargs)

        def fake_build_so(*_args, **kwargs):
            captured["qualnames"] = [
                fn.fn.__qualname__ for fn in kwargs["constexpr"]["decoder_fns"]
            ]
            return scratch / "fake.so", "fake_symbol"

        monkeypatch.setattr(frontend.tempfile, "mkdtemp", fake_mkdtemp)
        monkeypatch.setattr(frontend, "_build_so", fake_build_so)

        hx = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        hz = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.uint8)

        fn = css_bp_decoder(hx, hz, platform="cuda:80:32")

        assert isinstance(fn, CoprocessorFunction)
        assert len(set(captured["qualnames"])) == 2


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
