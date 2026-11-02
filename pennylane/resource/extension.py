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
"""Callbacks for compiler resource extensions stored in ``SpecsResources.extra``."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .expression import Expression


@dataclass(frozen=True)
class _ResourceHandler:
    """Aggregation and display rules for one backend extension key."""

    aggregate: Callable[[Any, Any, int | float | Expression], Any]
    display: Callable[[Any], dict[str, Any]]


_RESOURCE_EXTENSIONS: dict[str, _ResourceHandler] = {}


def register_resource_extension(
    name: str,
    aggregate: Callable[[Any, Any, int | float | Expression], Any],
    display: Callable[[Any], dict[str, Any]],
) -> None:
    """Register plugin-owned rules for a compiler resource extension.

    .. warning::

        The ``aggregate`` and ``display`` callbacks must not modify their input data,
        including nested values. Inputs are passed without copying, so mutations can
        affect subsequent aggregations or reports.

    Args:
        name (str): Key in the compiler's ``extended_fields`` and in
            :attr:`~pennylane.resource.SpecsResources.extra`.
        aggregate (Callable): Called with the caller's data (``None`` if absent),
            the callee's data, and its numeric or symbolic call count. Return the
            combined data without rounding fractional estimates.
        display (Callable): Called with the aggregated data after any substitutions.
            Return an ordered mapping of labels to values; nested mappings introduce
            group headings.

    Register before requesting compiled resource analysis. Data remains in ``extra``;
    no result subclass or conversion is required. Existing symbolic discovery and
    substitution work with expressions nested in dictionaries, lists, and tuples.
    The display callback is shared by individual and multi-level text/Markdown output.

    Registering the same callbacks again is harmless. Registering different callbacks
    for an existing name raises ``ValueError``. Unregistered callee extensions retain
    the parser's warning and are not aggregated.
    """
    handler = _ResourceHandler(aggregate, display)
    if name in _RESOURCE_EXTENSIONS and _RESOURCE_EXTENSIONS[name] != handler:
        raise ValueError(f"A resource extension is already registered for '{name}'.")
    _RESOURCE_EXTENSIONS[name] = handler


# Register PBC depth aggregation and display functions

def _aggregate_pbc_depth(total, incoming, call_count):
    """Combine PBC depths without modifying inputs or rounding intermediate values."""
    result = {} if total is None else total.copy()
    for name in ("any_commuting_depth", "qubit_disjoint_depth"):
        result[name] = result.get(name, 0) + call_count * incoming[name]
    return result


def _display_pbc_depth(data):
    """Group the PBC depth metrics for resource reports."""
    return {
        "PBC Depths": {
            "Any commuting depth": data["any_commuting_depth"],
            "Qubit disjoint depth": data["qubit_disjoint_depth"],
        }
    }


register_resource_extension("pbc_depth", aggregate=_aggregate_pbc_depth, display=_display_pbc_depth)
