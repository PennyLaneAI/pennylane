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

import importlib.util
import types
from pathlib import Path

import pytest

from pennylane.backline import onnx_decoder
from pennylane.backline.onnx import (
    _onnx_function,
    _onnxruntime_library,
    _onnxruntime_library_in,
    _tensor_message_bytes,
)

# An ONNX model whose uint8[1, 8] input passes through Identity to its uint8[1, 8] output.
IDENTITY_U8X8_MODEL = Path(__file__).parent / "data" / "identity_u8x8.onnx"


_ORT_LIB = "/opt/libonnxruntime.so"


def _function(model=Path("/models/model.onnx"), provider="auto", device=0, threads=1):
    """The ONNX function for a model taking 120 bytes and returning 121."""
    return _onnx_function(
        model, _ORT_LIB, (120, 121), provider=provider, device=device, threads=threads
    )


class TestOnnxFunction:
    """The ONNX coprocessor function and the config it carries."""

    def test_the_function_declares_the_model_tensor_sizes(self):
        """The function declares the model's input and output sizes as its message sizes."""
        assert _function().message_bytes == (120, 121)

    def test_the_function_is_catalysts_own(self):
        """The function is Catalyst's ONNX coprocessor function, so it needs no lib_path."""
        fn = _function()
        assert fn.name == "catalyst_onnx_coprocessor"
        assert fn.lib_path is None

    def test_the_function_runs_per_message(self):
        """The ONNX function is a host function, so a GPU coprocessor calls it per message."""
        assert _function().per_message

    def test_the_config_names_the_model_the_library_and_the_options(self):
        assert _function().config == (
            f"model=/models/model.onnx;ort_lib={_ORT_LIB};provider=auto;device=0;threads=1"
        )

    @pytest.mark.parametrize("provider", ["cpu", "migraphx", "cuda", "tensorrt"])
    def test_a_named_provider_is_passed_on(self, provider):
        assert f"provider={provider};device=1" in _function(provider=provider, device=1).config

    def test_threads_are_passed_on(self):
        """The intra-op thread count reaches the function's config."""
        assert _function(threads=8).config.endswith(";threads=8")

    def test_a_path_with_the_separator_raises(self):
        """A ';' in a path would split the config, so it is rejected."""
        with pytest.raises(ValueError, match="must not contain ';'"):
            _function(model=Path("/models/a;b.onnx"))


class TestOnnxDecoder:
    """The arguments onnx_decoder checks before it reads the model, and a model it reads."""

    @pytest.fixture
    def model(self, tmp_path):
        path = tmp_path / "model.onnx"
        path.write_bytes(b"")
        return path

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

    def test_unknown_provider_raises(self, model):
        with pytest.raises(ValueError, match="provider must be one of"):
            onnx_decoder(model, provider="tpu")

    def test_missing_model_raises(self, tmp_path):
        """The model is checked when the function is built, not when the coprocessor starts."""
        with pytest.raises(FileNotFoundError, match="no model"):
            onnx_decoder(tmp_path / "absent.onnx")

    def test_a_model_is_read_with_the_installed_onnxruntime(self):
        """A uint8[1, 8] to uint8[1, 8] identity model declares 8 B in and 8 B out, and the
        function runs it through the installed onnxruntime with the default options. With no
        onnxruntime installed, the error says what to install."""
        if importlib.util.find_spec("onnxruntime") is None:
            with pytest.raises(ImportError, match="onnxruntime-migraphx"):
                onnx_decoder(IDENTITY_U8X8_MODEL)
            return
        fn = onnx_decoder(IDENTITY_U8X8_MODEL)
        assert fn.message_bytes == (8, 8)
        assert fn.config.endswith(
            f";ort_lib={_onnxruntime_library()};provider=auto;device=0;threads=1"
        )


class TestOnnxruntimeLibrary:
    """The shared library of an onnxruntime package."""

    @pytest.fixture
    def capi(self, tmp_path):
        capi = tmp_path / "onnxruntime" / "capi"
        capi.mkdir(parents=True)
        return capi

    def test_the_library_is_found(self, capi):
        (capi / "libonnxruntime.so.1.2.3").write_bytes(b"")
        library = _onnxruntime_library_in(capi.parent, "linux")
        assert library == str(capi / "libonnxruntime.so.1.2.3")

    def test_on_macos_the_library_is_the_dylib(self, capi):
        (capi / "libonnxruntime.1.2.3.dylib").write_bytes(b"")
        (capi / "libonnxruntime_providers_shared.dylib").write_bytes(b"")
        library = _onnxruntime_library_in(capi.parent, "darwin")
        assert library == str(capi / "libonnxruntime.1.2.3.dylib")

    def test_a_package_without_its_library_raises(self, capi):
        with pytest.raises(ImportError, match="no onnxruntime shared library found"):
            _onnxruntime_library_in(capi.parent, "linux")

    def test_no_package_raises_naming_what_to_install(self):
        with pytest.raises(ImportError, match="onnxruntime-migraphx"):
            _onnxruntime_library_in(None, "linux")

    def test_the_installed_package_is_looked_up(self):
        """The installed onnxruntime's library is found, and with none installed the error says
        what to install."""
        if importlib.util.find_spec("onnxruntime") is None:
            with pytest.raises(ImportError, match="onnxruntime-migraphx"):
                _onnxruntime_library()
        else:
            assert Path(_onnxruntime_library()).is_file()


def _tensor(onnx_type, shape, name="x"):
    """An onnxruntime tensor description, as a session's inputs and outputs report it."""
    return types.SimpleNamespace(name=name, type=onnx_type, shape=shape)


class TestTensorMessageBytes:
    """The message sizes of a model's input and output tensors."""

    def test_a_size_is_the_element_count_times_the_element_size(self):
        """A float[2, 3] input is 24 bytes, and an int64[4] output 32."""
        inputs, outputs = [_tensor("tensor(float)", [2, 3])], [_tensor("tensor(int64)", [4])]
        assert _tensor_message_bytes(inputs, outputs) == (24, 32)

    def test_a_dynamic_dimension_counts_as_one(self):
        """A named, unknown or negative dimension counts as 1."""
        inputs = [_tensor("tensor(uint8)", ["batch", None, -1, 8])]
        assert _tensor_message_bytes(inputs, [_tensor("tensor(uint8)", [8])]) == (8, 8)

    def test_a_model_with_two_inputs_raises(self):
        inputs = [_tensor("tensor(uint8)", [8]), _tensor("tensor(uint8)", [8])]
        with pytest.raises(ValueError, match="one input and one output, it has 2 and 1"):
            _tensor_message_bytes(inputs, [_tensor("tensor(uint8)", [8])])

    def test_an_unsupported_tensor_type_raises(self):
        inputs = [_tensor("tensor(string)", [8], name="words")]
        with pytest.raises(
            ValueError, match=r"unsupported tensor type tensor\(string\) for 'words'"
        ):
            _tensor_message_bytes(inputs, [_tensor("tensor(uint8)", [8])])
