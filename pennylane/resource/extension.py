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
"""Aggregation and headings for resource extensions stored in ``SpecsResources.extra``."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from .expression import Expression

Count = int | float | Expression
ExtensionData = dict[str, Count]
AggregateCallback = Callable[[ExtensionData, ExtensionData, Count], ExtensionData]


@dataclass(frozen=True)
class _ResourceHandler:
    """Aggregation and a report heading for one backend extension key."""

    aggregate: AggregateCallback
    display_name: str


_RESOURCE_EXTENSIONS: dict[str, _ResourceHandler] = {}


def weighted_sum(
    total: ExtensionData,
    incoming: ExtensionData,
    call_count: Count,
) -> ExtensionData:
    """
    Helper for aggregating dictionaries of using weighted sums.

    Each incoming count is multiplied by ``call_count`` and added to the total.
    Missing counts start at zero. Fractional and symbolic values are preserved.

    Args:
        total (Mapping[str, Count]): Accumulated counts, or an empty dictionary
            when the caller has no data for this extension.
        incoming (Mapping[str, Count]): Counts from a called function.
        call_count (Count): Number of calls, possibly estimated or symbolic.

    Returns:
        ExtensionData: Combined counts in a new dictionary.

    **Example**

    >>> from pennylane.resource.extension import weighted_sum
    >>> weighted_sum({}, {"Hadamard": 4}, 3)
    {'Hadamard': 12}
    >>> weighted_sum({"Hadamard": 1}, {"Hadamard": 4, "CNOT": 2}, 3)
    {'Hadamard': 13, 'CNOT': 6}
    """
    result = dict(total)
    for name, value in incoming.items():
        result[name] = cast(Count, result.get(name, 0) + call_count * value)
    return result


def register_resource_extension(
    name: str,
    aggregate: AggregateCallback,
    *,
    display_name: str,
) -> None:
    """Register aggregation and a report heading for a compiler resource extension.

    .. warning::

        The ``aggregate`` callback must not modify its input data, including nested values.
        Inputs are passed without copying, so mutations can affect subsequent aggregations
        or reports.

        Currently, aggregation only supports flat dictionaries of numeric values (``int``, 
        ``float``, or :class:`~.resource.Expression`). Boolean and other metadata types are 
        not supported.

    Args:
        name (str): Key in the compiler's ``extended_fields`` and in
            :attr:`~pennylane.resource.SpecsResources.extra`.
        aggregate (AggregateCallback): ``aggregate(total, incoming, call_count)``
            receives the caller's data (an empty dictionary if absent), the callee's data, and
            its numeric or symbolic call count. Return combined data with the same
            structure.
        display_name (str): Heading shown above the extension's fields in reports.
    Raises:
        ValueError: If the name is already registered with different rules.

    Register before requesting compiled resource analysis. Unregistered callee extensions
    retain the parser's warning and are not aggregated.

    **Example**

    Register a count dictionary without writing callbacks:

    >>> from pennylane.resource import register_resource_extension
    >>> from pennylane.resource.extension import weighted_sum
    >>> register_resource_extension(
    ...     "example_gates",
    ...     aggregate=weighted_sum,
    ...     display_name="Example gates",
    ... )

    A payload such as ``{"Hadamard": 7}`` appears as a ``Hadamard`` row under
    ``Example gates`` in individual and comparison reports.

    Symbolic call counts remain expressions that can be substituted later:

    >>> from pennylane.resource import Expression
    >>> a = Expression({("a",): 1})
    >>> combined = weighted_sum({"count": 1}, {"count": 2}, a)
    >>> combined["count"].subs(a=0)
    1
    >>> combined["count"].subs(a=3)
    7
    """
    handler = _ResourceHandler(aggregate, display_name)
    if name in _RESOURCE_EXTENSIONS and _RESOURCE_EXTENSIONS[name] != handler:
        raise ValueError(f"A resource extension is already registered for '{name}'.")
    _RESOURCE_EXTENSIONS[name] = handler


# Register the PBC depth extension
register_resource_extension("pbc_depth", aggregate=weighted_sum, display_name="PBC Depths")
