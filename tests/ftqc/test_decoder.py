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

"""Unit tests for the FTQC Decoder contract."""

import pytest

from pennylane.ftqc import Decoder, DecoderSchema, get_decoder, register_decoder


class TestDecoderSchema:
    """The syndrome -> correction I/O contract."""

    def test_stores_io(self):
        schema = DecoderSchema(syndrome="uint8[N]", correction="uint8[N]")
        assert schema.syndrome == "uint8[N]"
        assert schema.correction == "uint8[N]"


class TestDecoder:
    """The Decoder value type."""

    def test_stores_name_and_schema(self):
        schema = DecoderSchema(syndrome="uint8[N]", correction="uint8[N]")
        dec = Decoder(name="steane", schema=schema)
        assert dec.name == "steane"
        assert dec.schema is schema
        assert dec.kernel is None

    def test_kernel_is_optional(self):
        schema = DecoderSchema("uint8[N]", "uint8[N]")
        dec = Decoder(name="steane", schema=schema, kernel="artifact-handle")
        assert dec.kernel == "artifact-handle"


class TestDecoderRegistry:
    """Named decoder registration (mirrors the transport registry)."""

    def test_register_and_retrieve(self):
        from pennylane.ftqc import decoder as decoder_mod

        before = dict(decoder_mod._DECODERS)
        try:

            @register_decoder("test-steane")
            def _factory():
                return Decoder(name="test-steane", schema=DecoderSchema("uint8[N]", "uint8[N]"))

            assert get_decoder("test-steane").name == "test-steane"
        finally:
            decoder_mod._DECODERS.clear()
            decoder_mod._DECODERS.update(before)

    def test_unknown_decoder_raises(self):
        with pytest.raises(ValueError, match="unknown decoder"):
            get_decoder("does-not-exist")
