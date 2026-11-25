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
"""Tests for the PennyLane module namespace."""

import importlib

import pytest

import pennylane as qp


def test_estimator_is_not_exported():
    """The removed estimator module is absent from the top-level namespace."""
    assert "estimator" not in dir(qp)


def test_estimator_is_not_importable():
    """Importing the removed estimator module raises ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("pennylane.estimator")

    assert exc_info.value.name == "pennylane.estimator"
