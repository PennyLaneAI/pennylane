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

import sys

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
