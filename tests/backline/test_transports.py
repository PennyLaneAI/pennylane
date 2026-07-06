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

"""Unit tests for the backline transport registry."""

import pytest

from pennylane.backline import Transport, get_transport, register_transport


class TestTransportRegistry:
    """The named transport registry."""

    def test_roce_is_builtin(self):
        assert get_transport("roce").name == "roce"

    def test_unknown_transport_raises(self):
        with pytest.raises(ValueError, match="unknown transport"):
            get_transport("infiniband")

    def test_register_and_retrieve_custom_transport(self):
        from pennylane.backline import transports

        before = dict(transports._TRANSPORTS)
        try:

            @register_transport("test-xport")
            def _factory():
                return Transport("test-xport")

            assert get_transport("test-xport").name == "test-xport"
        finally:
            transports._TRANSPORTS.clear()
            transports._TRANSPORTS.update(before)
