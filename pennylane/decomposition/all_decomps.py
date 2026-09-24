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
"""
Defines all_decomps, a recursive tool for collecting all downstream operators
and rules.
"""

from pennylane.core.operator import Operator2, abstractify

from .decomposition_rule import DecompositionRule, list_decomps


def _recursive_all_decomps(op: Operator2, rules_map: dict, skip_ops):
    """Collect all the rules for op, putting them into rules_map by in-place mutation.

    As opposed to _pure_recursive_add_decomps, it also adds variants of the operator
    specified by adj, n_ctrls, and adj_n_ctrls
    """

    if op in rules_map or op in skip_ops:
        return
    decomps = list_decomps(op)
    applicable_rules = [r for r in decomps if r.is_applicable(**op.arguments)]
    rules_map[op] = applicable_rules
    for rule in applicable_rules:
        resources = rule.compute_resources(**op.arguments)
        for r in resources.gate_counts:
            _recursive_all_decomps(r, rules_map, skip_ops)


def all_decomps(
    op: Operator2,
    skip_ops: set[Operator2] | None = None,
) -> dict[Operator2, list[DecompositionRule]]:
    """Collect all decomposition rules downstream of an Operator.

    Args:
        op (Operator2): an operator that we will want to decompose:
        skip_ops (set[Operator2] | None): abstract operators that should not be included in the results

    Returns:
        dict[Operator2, list[DecompositionRule]]: A map from abstract operators to their applicable rules

    >>> from pennylane.typing import Wire, Float

    >>> rules_map = all_decomps(qp.X(0))
    >>> len(rules_map)
    46
    >>> rules_map[qp.RX(Float, Wire[1])]
    [DecompositionRule(name=_rx_to_rot),
     DecompositionRule(name=_rx_to_rz_ry),
     DecompositionRule(name=_rx_to_ppr),
     DecompositionRule(name=_rx_to_ry_cliff),
     DecompositionRule(name=_rx_to_rz_cliff)]

    The keys from a rule map can be reused on successive calls to avoid recollecting the same nodes.
    Note that if an op is skipped, it's children will also not be added.

    >>> qp.decomposition.all_decomps(qp.Y(0), skip_ops=rules_map)
    {}

    """
    skip_ops = skip_ops or set()
    rules_map = {}
    _recursive_all_decomps(
        abstractify(op),
        rules_map,
        skip_ops=skip_ops,
    )
    return rules_map
