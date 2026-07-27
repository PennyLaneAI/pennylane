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
    """Validate a loop trip-count hint for ``scf.for`` / ``scf.while``.

    Args:
        value (int | float): estimated trip-count. Can be fractional because it is an estimate
            that may arise from averaging or a statistical analysis.

    Returns:
        float: Validated estimate, cast to float.

    """
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"'estimated_iterations' must be a non-negative number, but got {type(value).__name__}."
        )
    value = float(value)
    if value < 0:
        raise ValueError(f"'estimated_iterations' must be non-negative, but got {value}.")
    return value


def validate_estimated_probability(value: float) -> float:
    """Validate a branch probability hint for ``scf.if``.

    Args:
        value (float): estimated branch probability.

    Returns:
        float: Validated probability.

    """
    if not isinstance(value, float):
        raise TypeError(
            f"'estimated_probability' must be a float in [0, 1], but got {type(value).__name__}."
        )
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"'estimated_probability' must be in [0, 1], but got {value}.")
    return value


def validate_estimated_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    """Validate branch probability hints for multi-branch conditionals.

    Args:
        values (Sequence[float]): Probability values per branch

    Returns:
        tuple[float]: Validated probabilities, cast to ``tuple``.

    The values represent the expected unconditional probability of each non-default
    branch (in branch order). The default branch probability is ``1 - sum(values)``.
    """
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(
            "'estimated_probability' must be a float in [0, 1], "
            f"but got {type(values).__name__}."
        )
    probs = tuple(validate_estimated_probability(v) for v in values)
    if sum(probs) > 1.0 + 1e-10:
        raise ValueError(
            f"'estimated_probability' entries must sum to at most 1, but got {sum(probs)}."
        )
    return probs


def collect_estimated_probabilities(
    branch_probs: Sequence[float | None],
) -> tuple[float, ...] | None:
    """Collect per-branch probability hints into a validated tuple.
    Args:
        branch_probs (Sequence[float | None]): Probability values per branch. Must all be ``None``
            if any is ``None``.

    Returns:
        tuple[float] | None: Validated probabilities, cast to ``tuple``, or ``None`` if all entries
        of ``branch_probs`` were ``None``.

    """
    if all(p is None for p in branch_probs):
        return None
    if any(p is None for p in branch_probs):
        raise ValueError(
            "'estimated_probability' must be provided for every non-default branch when "
            "using resource-estimation hints."
        )
    return validate_estimated_probabilities(branch_probs)
