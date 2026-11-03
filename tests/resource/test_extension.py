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
"""Tests for resource extension aggregation and parsing."""

import copy
from numbers import Real
from typing import cast

import pytest

import pennylane as qp
from pennylane.resource.extension import _RESOURCE_EXTENSIONS, weighted_sum
from pennylane.resource.parsing import parse_resources_json


@pytest.fixture(autouse=True)
def restore_registry():
    """Keep callback registrations local to each test."""
    original = _RESOURCE_EXTENSIONS.copy()
    yield
    _RESOURCE_EXTENSIONS.clear()
    _RESOURCE_EXTENSIONS.update(original)


@pytest.fixture(name="resource_data", params=[2, 3, 5])
def resource_data_fixture(request):
    """Data for extension parsing tests."""
    return {
        "main": {
            "metadata": {"qnode": True, "has_branches": False},
            "num_qubits": {"total": 2},
            "quantum_operations": {},
            "measurement_processes": {},
            "function_calls": {
                "static": {"helper": request.param},
                "dynamic": {"helper": "loop_id"},
            },
            "extended_fields": {"example_count": {"count": 2}},
        },
        "helper": {
            "metadata": {"qnode": True, "has_branches": False},
            "num_qubits": {"total": 2},
            "quantum_operations": {},
            "measurement_processes": {},
            "function_calls": {"static": {}, "dynamic": {}},
            "extended_fields": {"example_count": {"count": 3}},
        },
    }


def test_weighted_sum():
    """Test weighted sum aggregation."""
    incoming = {"Hadamard": 4}
    total = {"CNOT": 2}
    assert weighted_sum({}, incoming, 3) == {"Hadamard": 12}
    assert weighted_sum(total, incoming, 3) == {"CNOT": 2, "Hadamard": 12}
    assert total == {"CNOT": 2}
    assert incoming == {"Hadamard": 4}


def test_aggregation_and_substitution(resource_data):
    """Test aggregation and substitution of extended fields."""
    qp.resource.register_resource_extension(
        "example_count", aggregate=weighted_sum, display_name="Example resources"
    )
    calls = resource_data["main"]["function_calls"]["static"]["helper"]
    original = copy.deepcopy(resource_data)
    resource, helper = parse_resources_json(resource_data)
    a = qp.resource.Expression({("a",): cast(Real, 1)})
    assert resource.extra["example_count"]["count"] == 2 + 3 * calls + 3 * a
    assert resource.subs(a=2).extra["example_count"]["count"] == 8 + 3 * calls
    assert helper.extra["example_count"] == {"count": 3}
    assert resource_data == original  # Ensure inputs are not mutated


def test_unregistered_fields(resource_data):
    """Unknown extended fields survive with warning but not aggregated."""
    with pytest.warns(UserWarning, match="unknown extended fields"):
        resource, _ = parse_resources_json(resource_data)
    assert resource.extra == {"example_count": {"count": 2}}
