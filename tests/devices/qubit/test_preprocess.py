# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for preprocess in devices/qubit."""
import pytest

import pennylane as qml
from pennylane.devices.qubit.preprocess import (
    _stopping_condition,
    _supports_observable,
    expand_fn,
    check_validity,
    batch_transform,
    preprocess,
)
from pennylane import DeviceError


class TestPreprocess:
    """Test that functions in qml.devices.qubit.preprocess work as expected"""

    def test_stopping_condition(self):
        pass

    def test_supports_observable(self):
        pass

    def test_batch_transform(self):
        pass

    def test_expand_fn(self):
        pass

    def test_check_validity_fails(self):
        pass

    def test_check_validity_passes(self):
        pass

    def test_preprocess(self):
        pass
