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
"""Registered resource callbacks compose with the original parser and reports."""

import copy
from operator import methodcaller

import pytest

import pennylane as qp
from pennylane.resource.extension import _RESOURCE_EXTENSIONS
from pennylane.resource.parsing import parse_resources_json


@pytest.fixture(autouse=True)
def restore_registry():
    """Keep callback registrations local to each test."""
    original = _RESOURCE_EXTENSIONS.copy()
    yield
    _RESOURCE_EXTENSIONS.clear()
    _RESOURCE_EXTENSIONS.update(original)


def aggregate_count(total, incoming, calls):
    """Combine counts without modifying either input."""
    total = {} if total is None else total
    return {"count": total.get("count", 0) + calls * incoming["count"]}


def display_count(data):
    """Include groups, text, unavailable values and a label needing Markdown escaping."""
    return {
        "Example resources": {
            "Custom | count": data["count"],
            "Unavailable": None,
            "Description": "first\nsecond",
        }
    }


def register_count():
    qp.resource.register_resource_extension(
        "example_count", aggregate=aggregate_count, display=display_count
    )


def function_record(extra, *, qnode=False, static=None, dynamic=None):
    return {
        "metadata": {"qnode": qnode, "has_branches": False},
        "num_qubits": {"total": 0},
        "quantum_operations": {},
        "measurement_processes": {},
        "function_calls": {"static": static or {}, "dynamic": dynamic or {}},
        "extended_fields": extra,
    }


def test_registration():
    """Repeat registration is safe, while a conflicting implementation is rejected."""
    register_count()
    register_count()
    with pytest.raises(ValueError, match="already registered"):
        qp.resource.register_resource_extension(
            "example_count", aggregate=aggregate_count, display=lambda data: data
        )


def test_calls_pbc_and_substitution():
    """Independent extensions and PBC share static and symbolic call multipliers."""
    register_count()
    data = {
        "main": function_record(
            {"example_count": {"count": 1}},
            qnode=True,
            static={"helper": 2},
            dynamic={"helper": "loop_id"},
        ),
        "helper": function_record(
            {
                "example_count": {"count": 3},
                "pbc_depth": {"any_commuting_depth": 4, "qubit_disjoint_depth": 6},
            },
            qnode=True,
        ),
    }
    original = copy.deepcopy(data)
    resource, helper = parse_resources_json(data)
    a = qp.resource.Expression({("a",): 1})
    assert isinstance(resource, qp.resource.PBCSpecsResources)
    assert resource.extra["example_count"]["count"] == 7 + 3 * a
    assert resource.any_commuting_depth == 8 + 4 * a
    assert resource.qubit_disjoint_depth == 12 + 6 * a
    assert resource.subs(a=2).extra["example_count"]["count"] == 13
    assert resource.vars == {"a"}
    assert helper.extra["example_count"] == {"count": 3}
    assert helper.any_commuting_depth == 4
    assert helper.qubit_disjoint_depth == 6
    assert data == original


@pytest.mark.parametrize("calls", [0, 0.25, 3])
def test_fractional_and_zero_calls(calls):
    """Extension values avoid the rounding applied to the standard operation counts."""
    register_count()
    data = {
        "main": function_record({}, qnode=True, static={"helper": calls}),
        "helper": function_record({"example_count": {"count": 3}}),
    }
    (resource,) = parse_resources_json(data)
    assert resource.extra == {"example_count": {"count": 3 * calls}}
    assert isinstance(resource, qp.resource.SpecsResources)


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (None, {"any_commuting_depth": 0.75, "qubit_disjoint_depth": 1.5}),
        (
            {"any_commuting_depth": 1, "qubit_disjoint_depth": 2},
            {"any_commuting_depth": 1.75, "qubit_disjoint_depth": 3.5},
        ),
    ],
)
def test_pbc_callback_preserves_inputs(total, expected):
    """The built-in PBC callback accepts missing totals and does not mutate shared data."""
    incoming = {"any_commuting_depth": 3, "qubit_disjoint_depth": 6}
    before = copy.deepcopy((total, incoming))
    result = _RESOURCE_EXTENSIONS["pbc_depth"].aggregate(total, incoming, 0.25)
    assert result == expected
    assert (total, incoming) == before


def test_pbc_rounds_after_nested_calls():
    """Fractional PBC depths are rounded only after all call multipliers are applied."""
    depths = {"any_commuting_depth": 3, "qubit_disjoint_depth": 6}
    data = {
        "main": function_record({}, qnode=True, static={"middle": 4}),
        "middle": function_record({}, static={"helper": 0.25}),
        "helper": function_record({"pbc_depth": depths}),
    }
    (resource,) = parse_resources_json(data)
    assert isinstance(resource, qp.resource.PBCSpecsResources)
    assert resource.any_commuting_depth == 3
    assert resource.qubit_disjoint_depth == 6


@pytest.mark.parametrize("multi_level", [False, True])
def test_pbc_extension_display(multi_level):
    """PBC data in extra uses the registered display callback in each report format."""
    a = qp.resource.Expression({("a",): 1})
    resource = qp.resource.SpecsResources(
        counts={},
        measurement_processes={},
        num_wires=0,
        extra={"pbc_depth": {"any_commuting_depth": a, "qubit_disjoint_depth": 2 * a}},
    )
    before = copy.deepcopy(resource.extra)
    concrete = resource.subs(a=3)
    report = (
        qp.resource.CircuitSpecs(
            level={0: "symbolic", 1: "concrete"},
            resources={0: resource, 1: concrete},
        )
        if multi_level
        else concrete
    )
    for render in (str, methodcaller("_repr_markdown_")):
        text = render(report)
        assert text.count("PBC Depths") == 1
        assert "Any commuting depth" in text
        assert "Qubit disjoint depth" in text
        assert "3" in text and "6" in text
    assert resource.extra == before


def test_unregistered_fields():
    """Unknown local data survives, while unknown callee data keeps its warning."""
    data = {
        "main": function_record({"unknown": {"count": 1}}, qnode=True, static={"helper": 2}),
        "helper": function_record({"unknown": {"count": 3}}),
    }
    with pytest.warns(UserWarning, match="unknown extended fields"):
        (resource,) = parse_resources_json(data)
    assert resource.extra == {"unknown": {"count": 1}}
    assert "unknown" not in resource.to_pretty_str()


@pytest.mark.parametrize("multi_level", [False, True])
def test_display_and_substitution(multi_level):
    """Every report path renders the callbacks after substitution, including batches."""
    register_count()
    a = qp.resource.Expression({("a",): 1})
    resource = qp.resource.SpecsResources(
        counts={},
        measurement_processes={},
        num_wires=0,
        extra={"example_count": {"count": 1 + 3 * a}},
    )
    empty = qp.resource.SpecsResources(counts={}, measurement_processes={}, num_wires=0)
    report = qp.resource.CircuitSpecs(
        level={0: "before", 1: "after"} if multi_level else "after",
        resources={"before": empty, "after": [resource, resource]} if multi_level else resource,
    )
    before = copy.deepcopy(resource.extra)
    renderers = (
        str,
        methodcaller("to_pretty_str", tabular=False),
        methodcaller("_repr_markdown_"),
    )
    for render in renderers:
        text = render(report)
        assert "Example resources" in text
        assert "Not computed" in text
        assert "3" in text and "a" in text
    markdown = renderers[-1](report)
    assert r"Custom \| count" in markdown
    assert "first<br>second" in markdown
    if multi_level:
        assert "N/A" in markdown
        assert "1-a" in markdown and "1-b" in markdown
    concrete = resource.subs(a=2)
    assert "7" in concrete.to_pretty_str()
    assert "7" in renderers[-1](concrete)
    assert resource.extra == before


def test_same_label_in_independent_extensions():
    """The row identity includes the extension key, not just its display label."""
    register_count()
    qp.resource.register_resource_extension(
        "other_count", aggregate=aggregate_count, display=display_count
    )
    resource = qp.resource.SpecsResources(
        counts={},
        measurement_processes={},
        num_wires=0,
        extra={"example_count": {"count": 3}, "other_count": {"count": 5}},
    )
    text = resource.to_pretty_str()
    assert "Custom | count: 3" in text
    assert "Custom | count: 5" in text


def test_new_rows_stay_in_their_group():
    """A metric appearing at a later level stays under its original group heading."""
    qp.resource.register_resource_extension(
        "changing_rows", aggregate=aggregate_count, display=lambda data: data
    )
    resources = {
        name: qp.resource.SpecsResources(
            counts={},
            measurement_processes={},
            num_wires=0,
            extra={"changing_rows": data},
        )
        for name, data in (
            ("before", {"Group one": {"First metric": 1}, "Group two": {"Other metric": 2}}),
            ("after", {"Group one": {"Later metric": 3}, "Group two": {"Other metric": 4}}),
        )
    }
    report = qp.resource.CircuitSpecs(level={0: "before", 1: "after"}, resources=resources)
    for render in (str, methodcaller("_repr_markdown_")):
        text = render(report)
        assert text.index("Group one") < text.index("Later metric") < text.index("Group two")
