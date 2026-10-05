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


"""Tests for the ONNX coprocessor function, onnx_decoder."""

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

from pennylane.backline import onnx_decoder
from pennylane.backline.onnx import _onnx_message_bytes

# An ONNX model whose uint8[1, 8] input passes through Identity to its uint8[1, 8] output.
IDENTITY_U8X8_MODEL = Path(__file__).parent / "data" / "identity_u8x8.onnx"


class TestOnnxDecoder:
    """The ONNX coprocessor function and the config it carries."""

    @pytest.fixture(autouse=True)
    def model_sizes(self, monkeypatch):
        """Stand in for reading a model, whose files here are empty, as uint8[120] to uint8[121]."""
        monkeypatch.setattr("pennylane.backline.onnx._onnx_message_bytes", lambda model: (120, 121))

    @pytest.fixture
    def model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pennylane.backline.onnx._onnxruntime_library", lambda: "/opt/libonnxruntime.so"
        )
        path = tmp_path / "model.onnx"
        path.write_bytes(b"")
        return path

    def test_the_function_declares_the_model_tensor_sizes(self, model):
        """The function declares the model's input and output sizes as its message sizes."""
        assert onnx_decoder(model).message_bytes == (120, 121)

    def test_the_model_tensor_sizes_are_read_with_onnxruntime(self):
        """A uint8[1, 8] to uint8[1, 8] identity model declares 8 B in and 8 B out."""
        pytest.importorskip("onnxruntime")
        assert _onnx_message_bytes(IDENTITY_U8X8_MODEL) == (8, 8)

    def test_the_function_is_catalysts_own(self, model):
        """The function is Catalyst's ONNX coprocessor function, so it needs no lib_path."""
        fn = onnx_decoder(model)
        assert fn.name == "catalyst_onnx_coprocessor"
        assert fn.lib_path is None

    def test_the_function_runs_per_message(self, model):
        """The ONNX function is a host function, so a GPU coprocessor calls it per message."""
        assert onnx_decoder(model).per_message

    def test_the_provider_defaults_to_auto(self, model):
        """With no provider given, onnxruntime picks the GPU it has, or the CPU."""
        assert (
            f"model={model.resolve()};ort_lib=/opt/libonnxruntime.so;provider=auto;device=0;threads=1"
            == onnx_decoder(model).config
        )

    @pytest.mark.parametrize("provider", ["cpu", "migraphx", "cuda", "tensorrt"])
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
        """The library is the one in the installed onnxruntime package."""
        package = tmp_path / "onnxruntime"
        (package / "capi").mkdir(parents=True)
        (package / "capi" / "libonnxruntime.so.1.2.3").write_bytes(b"")
        spec = importlib.machinery.ModuleSpec("onnxruntime", None, is_package=True)
        spec.submodule_search_locations = [str(package)]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
        model = tmp_path / "model.onnx"
        model.write_bytes(b"")
        assert (
            f"ort_lib={package / 'capi' / 'libonnxruntime.so.1.2.3'}" in onnx_decoder(model).config
        )

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
