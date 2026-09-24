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
Defines add_decomps, a recursive tool for collecting all downstream operators
and rules.
"""

from typing import TypedDict, Unpack

from pennylane.core.operator import Operator2, abstractify
from pennylane.typing import Bool, Wire

from .decomposition_rule import DecompositionRule, list_decomps

SKIP_MODIFICATION_OP_NAMES = {"PauliMeasure", "MidMeasureMP"}
"""
The mid circuit measurements we shouldn't create modified versions of.
"""

def unwrap(
    op: Operator2, is_adjoint: bool = False, n_ctrls: int = 0
) -> tuple[Operator2, bool, int]:
    """A simple utility for extracting out the adjoint and control information from an operator."""
    from pennylane.ops import Adjoint, Controlled # here for circular dependency
    if not isinstance(op, (Adjoint, Controlled)):
        return op, is_adjoint, n_ctrls
    if isinstance(op, Adjoint):
        return unwrap(op.base, not is_adjoint, n_ctrls)
    # is controlled
    return unwrap(op.base, is_adjoint, n_ctrls + len(op.control_wires))




class Modifiers(TypedDict):
    """The keyword arguments specifing all the variants of an operator we also
    want decomposition rules for."""

    adj: bool
    n_ctrls: int
    adj_n_ctrls: int
    skip_ops: set[Operator2]


def _pure_recursive_all_decomps(op: Operator2, rules_map: dict, **kwargs: Unpack[Modifiers]):
    """
    This helper only collect the rules for the provided operator, and not any modified versions
    of it.
    """
    if op in rules_map or op in kwargs["skip_ops"]:
        return
    decomps = list_decomps(op)
    applicable_rules = [r for r in decomps if r.is_applicable(**op.arguments)]
    rules_map[op] = applicable_rules
    for rule in applicable_rules:
        resources = rule.compute_resources(**op.arguments)
        for r in resources.gate_counts:
            _recursive_all_decomps(r, rules_map, **kwargs)


def _recursive_all_decomps(op: Operator2, rules_map: dict, **kwargs: Unpack[Modifiers]):
    """Collect all the rules for op, putting them into rules_map by in-place mutation.

    As opposed to _pure_recursive_add_decomps, it also adds variants of the operator
    specified by adj, n_ctrls, and adj_n_ctrls
    """
    from pennylane.ops import adjoint, ctrl # here for circular dependency
    if op in rules_map:
        return

    base, is_adj, has_n_ctrls = unwrap(op)

    _pure_recursive_all_decomps(op, rules_map, **kwargs)

    if base.name in SKIP_MODIFICATION_OP_NAMES:
        return

    if kwargs["adj"] and not has_n_ctrls:
        # if adj op appears from resources, make sure to also include target
        target = base if is_adj else adjoint(op)
        _pure_recursive_all_decomps(target, rules_map, **kwargs)

    if not has_n_ctrls and not is_adj:
        for n in range(1, kwargs['n_ctrls']+1):
            target = ctrl(base, Wire[n], Bool[n])
            _pure_recursive_all_decomps(target, rules_map, **kwargs)
        for n in range(1, kwargs['adj_n_ctrls']+1):
            target = ctrl(adjoint(base), Wire[n], Bool[n])
            _pure_recursive_all_decomps(target, rules_map, **kwargs)


def all_decomps(
    op: Operator2,
    adj: bool = True,
    n_ctrls: int = 1,
    adj_n_ctrls: int = 1,
    skip_ops: set[Operator2] | None = None,
) -> dict[Operator2, list[DecompositionRule]]:
    """Collect all decomposition rules downstream of an Operator.

    Args:
        op (Operator2): an operator that we will want to decompose:
        adj=True (bool): whether or not to also include the rules for handling the adjoint of every
            operator
        n_ctrls=1 (int): How many controlled versions to include for each base operator
        adj_n_ctrls=1 (int): How many controlled version to include for the adjoint of each base operator
        skip_ops (set[Operator2] | None): abstract operators that should not be included in the results

    Returns:
        dict[Operator2, list[DecompositionRule]]: A map from abstract operators to their applicable rules

    >>> from pennylane.typing import Wire, Float

    >>> rules_map = all_decomps(qp.X(0))
    >>> len(rules_map)
    406
    >>> rules_map[qp.CZ(Wire[2])]
    [DecompositionRule(name=_cz_to_cps),
    DecompositionRule(name=_cz_to_cnot),
    DecompositionRule(name=_cz_to_ppr),
    DecompositionRule(name=_cz_lattice_surgery_ppm)]
    >>> rules_map[qp.adjoint(qp.PhaseShift(Float, Wire[1]))]
    [DecompositionRule(name=adjoint_rotation),
    DecompositionRule(name=adjoint(_phaseshift_to_rz_gp))]

    The keys from a rule map can be reused on successive calls to avoid recollecting the same nodes.

    >>> qp.decomposition.all_decomps(qp.Y(0), skip_ops=rules_map)
    {}
    
    ``adj``, ``n_ctrls``, and ``adj_n_ctrls`` can be used to customize the modified versions of operators
    that should also be included. Note that pauli measure and measure never have modified versions.

    >>> rules_map_no_mod = qp.decomposition.all_decomps(qp.X(0), adj=False, n_ctrls=0, adj_n_ctrls=0)
    >>> len(rules_map_no_mod)
    46

    We can see that even though ``adjoint(X)`` is in the default map, it is no longer included with ``adj=False``:

    >>> qp.adjoint(qp.X(Wire[1])) in rules_map
    True
    >>> qp.adjoint(qp.X(Wire[1])) in rules_map_no_mod
    False

    More control variants can also be include with ``n_ctrls`` and ``adj_n_ctrls``:

    >>> rules_map_3C = qp.decomposition.all_decomps(qp.X(0), n_ctrls=3)
    >>> len(rules_map_3C)
    759

    Triply controlled are now included:

    >>> qp.ctrl(qp.RX(Float, Wire[1]), Wire[3], Bool[3]) in rules_map_3C
    True

    """
    skip_ops = skip_ops or set()
    rules_map = {}
    _recursive_all_decomps(
        abstractify(op),
        rules_map,
        adj=adj,
        n_ctrls=n_ctrls,
        adj_n_ctrls=adj_n_ctrls,
        skip_ops=skip_ops,
    )
    return rules_map
