# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the custom capture primitives for the ftqc module"""
import pytest

pytest.importorskip("jax")

# pylint: disable=wrong-import-position
from pennylane.capture.custom_primitives import QmlPrimitive
from pennylane.ftqc.primitives import measure_in_basis_prim


def test_importing_primitive():
    """Test that the measure_in_basis_prim is accessible from pennylane.ftqc.primitives.
    This is mostly for CodeCov."""

    assert isinstance(measure_in_basis_prim, QmlPrimitive)
