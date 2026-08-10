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

"""Tests for backline coprocessor helper functions."""

import sys
import types
from pathlib import Path

from pennylane.backline import functions, triton_decoder


def _install_fake_triton_frontend(monkeypatch, **builders):
    """Install a fake Triton frontend module into ``sys.modules``."""
    frontend_module = types.ModuleType("pennylane.backline.decoders.triton.decoder_frontend")
    for name, builder in builders.items():
        setattr(frontend_module, name, builder)

    triton_package = types.ModuleType("pennylane.backline.decoders.triton")
    triton_package.__path__ = []
    triton_package.decoder_frontend = frontend_module

    monkeypatch.setitem(sys.modules, "pennylane.backline.decoders.triton", triton_package)
    monkeypatch.setitem(
        sys.modules,
        "pennylane.backline.decoders.triton.decoder_frontend",
        frontend_module,
    )


def test_triton_decoder_forwards_build_options(monkeypatch):
    """It should forward build options and wrap the compiled Triton decoder."""
    calls = {}

    def fake_build_triton_decoder(decoder_fns, **kwargs):
        calls["decoder_fns"] = decoder_fns
        calls.update(kwargs)
        return Path("/tmp/triton_decoder.so"), "decode_symbol"

    _install_fake_triton_frontend(monkeypatch, build_triton_decoder=fake_build_triton_decoder)

    decoder_fns = (object(), object())
    decoder = functions.triton_decoder(
        decoder_fns,
        platform="cuda:80:32",
        num_warps=4,
        num_stages=2,
        compiler="cc",
        cflags=("-g",),
    )

    assert decoder == functions.CoprocessorFunction(
        name="decode_symbol", lib_path="/tmp/triton_decoder.so"
    )
    assert calls == {
        "decoder_fns": decoder_fns,
        "platform": "cuda:80:32",
        "num_warps": 4,
        "num_stages": 2,
        "compiler": "cc",
        "cflags": ("-g",),
    }


def test_css_decoder_forwards_build_options(monkeypatch):
    """It should forward decoder build options and wrap the result."""
    calls = {}

    def fake_build_css_bp_decoder(Hx, Hz, **kwargs):
        calls["Hx"] = Hx
        calls["Hz"] = Hz
        calls.update(kwargs)
        return Path("/tmp/decoder.so"), "decode_symbol"

    _install_fake_triton_frontend(monkeypatch, build_css_bp_decoder=fake_build_css_bp_decoder)

    decoder = functions.css_decoder(
        [[1, 0], [0, 1]],
        [[1, 1], [0, 1]],
        postprocess="hard",
        num_iters=7,
        prob=0.2,
        platform="cuda:80:32",
        num_warps=4,
        num_stages=2,
        compiler="cc",
        cflags=("-g",),
    )

    assert decoder == functions.CoprocessorFunction(
        name="decode_symbol", lib_path="/tmp/decoder.so"
    )
    assert calls == {
        "Hx": [[1, 0], [0, 1]],
        "Hz": [[1, 1], [0, 1]],
        "postprocess": "hard",
        "num_iters": 7,
        "prob": 0.2,
        "platform": "cuda:80:32",
        "num_warps": 4,
        "num_stages": 2,
        "compiler": "cc",
        "cflags": ("-g",),
    }


def test_triton_decoder_is_exported():
    """The package should export the generic Triton decoder helper."""
    assert triton_decoder is functions.triton_decoder
