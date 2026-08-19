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

from pennylane.backline import CoprocessorFunction


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
