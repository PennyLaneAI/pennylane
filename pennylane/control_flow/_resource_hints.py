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
"""Validation helpers for Catalyst resource-estimation hints on control flow."""

from collections.abc import Sequence

ESTIMATED_ITERATIONS_ATTR = "catalyst.estimated_iterations"
ESTIMATED_PROBABILITY_ATTR = "catalyst.estimated_probability"
ESTIMATED_PROBABILITIES_ATTR = "catalyst.estimated_probabilities"


def validate_estimated_iterations(value: int | float) -> float:
    """Validate a loop trip-count hint for ``scf.for`` / ``scf.while``."""
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"'estimated_iterations' must be a non-negative number, but got {type(value).__name__}."
        )
    value = float(value)
    if value < 0:
        raise ValueError(f"'estimated_iterations' must be non-negative, but got {value}.")
    return value


def validate_estimated_probability(value: float) -> float:
    """Validate a branch probability hint for ``scf.if``."""
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"'estimated_probability' must be a float in [0, 1], but got {type(value).__name__}."
        )
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"'estimated_probability' must be in [0, 1], but got {value}.")
    return value


def validate_estimated_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    """Validate branch probability hints for multi-branch conditionals.

    The values represent the expected unconditional probability of each non-default
    branch (in branch order). The default branch probability is ``1 - sum(values)``.
    """
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(
            "'estimated_probabilities' must be a sequence of floats in [0, 1], "
            f"but got {type(values).__name__}."
        )
    probs = tuple(validate_estimated_probability(v) for v in values)
    if sum(probs) > 1.0 + 1e-10:
        raise ValueError(
            f"'estimated_probabilities' entries must sum to at most 1, but got {sum(probs)}."
        )
    return probs


def normalize_estimated_probabilities(
    value: float | Sequence[float] | None,
) -> tuple[float, ...] | None:
    """Normalize user-provided probability hint(s) to a validated tuple."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return validate_estimated_probabilities((float(value),))
    return validate_estimated_probabilities(value)


def validate_estimated_probabilities_count(
    probs: tuple[float, ...], num_branches: int, *, arg_name: str = "estimated_probabilities"
) -> None:
    """Ensure there is one probability hint per non-default branch."""
    if len(probs) != num_branches:
        raise ValueError(
            f"'{arg_name}' must have one entry per non-default branch, but got "
            f"{len(probs)} probabilities for {num_branches} branch(es)."
        )
