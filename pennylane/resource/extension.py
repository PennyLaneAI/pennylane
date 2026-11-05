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
"""Independent extensions stored in compiler resource results."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from .expression import Expression


class ResourceExtension(ABC):
    """Interface for independent resource groups in :class:`~.resource.SpecsResources`.

    Implement extensions as dataclasses and register them with
    :func:`~pennylane.resource.register_resource_extension`. Each backend key identifies
    one extension in ``resources.extra``; several extensions may coexist.

    ``aggregate`` combines raw backend data while PennyLane resolves calls and loops.
    ``from_dict`` constructs the extension after aggregation. ``display`` supplies
    ordered row labels and values for both individual results and multi-level tables.
    Values may be numbers, :class:`~.resource.Expression` objects, booleans, strings,
    or ``None`` for unavailable estimates. Formatting is applied after substitution.

    Registration also enables symbolic discovery and substitution in dataclass fields.
    ``display_name`` is the heading for the group's display rows.
    """

    display_name: ClassVar[str] = "Extra fields"

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceExtension":
        """Construct an extension from backend fields; override for a different schema."""
        return cls(**data)

    @staticmethod
    @abstractmethod
    def aggregate(total: Any, incoming: Any, call_count: int | float | Expression) -> Any:
        """Combine independent copies of caller and callee data.

        ``total`` is ``None`` when the caller has no value. The extension owns all
        aggregation rules, including zero, fractional, and symbolic call counts.
        """

    @abstractmethod
    def display(self) -> dict[str, Any]:
        """Return ordered metric labels and display values for this extension."""

    def flatten(self):
        """Expose dataclass fields to PennyLane's existing symbolic traversal."""
        names = tuple(item.name for item in fields(self) if item.init)
        return tuple(getattr(self, name) for name in names), names

    @classmethod
    def unflatten(cls, values, names):
        """Reconstruct an extension after symbolic substitution."""
        return cls(**dict(zip(names, values, strict=True)))


@dataclass(frozen=True, slots=True, kw_only=True)
class PBCDepthResourceExtension(ResourceExtension):
    """PBC depth metrics, stored under ``resources.extra["pbc_depth"]``."""

    display_name: ClassVar[str] = "PBC Depths"
    any_commuting_depth: int | Expression
    qubit_disjoint_depth: int | Expression

    @classmethod
    def from_dict(cls, data):
        """Round the final PBC depths, after weighted and symbolic calls are combined."""
        return cls(
            any_commuting_depth=math.ceil(round(data["any_commuting_depth"], 4)),
            qubit_disjoint_depth=math.ceil(round(data["qubit_disjoint_depth"], 4)),
        )

    @staticmethod
    def aggregate(total, incoming, call_count):
        """Add PBC depth counts."""
        total = {} if total is None else total
        return {
            name: total.get(name, 0) + call_count * incoming[name]
            for name in ("any_commuting_depth", "qubit_disjoint_depth")
        }

    def display(self):
        """Return the two PBC depth rows."""
        return {
            "Any commuting depth": self.any_commuting_depth,
            "Qubit disjoint depth": self.qubit_disjoint_depth,
        }
