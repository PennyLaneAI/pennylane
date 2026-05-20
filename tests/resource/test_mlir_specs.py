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
"""Unit tests for the MLIR helpers for the specs transform"""

import pytest

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.resource import SpecsResources
from pennylane.resource.specs import (
    _make_level_name_unique,
)


def test_make_level_name_unique():
    existing_levels = {"foo", "foo-2", "bar"}

    assert _make_level_name_unique("foo", existing_levels) == "foo-3"
    assert _make_level_name_unique("bar", existing_levels) == "bar-2"
    assert _make_level_name_unique("baz", existing_levels) == "baz"
