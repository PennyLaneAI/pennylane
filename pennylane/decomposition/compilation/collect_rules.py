from pennylane.core.operator import Operator2, abstractify
from pennylane.ops import Adjoint, Controlled, adjoint, ctrl
from pennylane.typing import Bool, Wire

from ..decomposition_rule import DecompositionRule, list_decomps


def _unwrap(op, is_adjoint=False, n_ctrls=0):
    if not isinstance(op, (Adjoint, Controlled)):
        return op, is_adjoint, n_ctrls
    if isinstance(op, Adjoint):
        return _unwrap(op.base, not is_adjoint, n_ctrls)
    # is controlled
    return _unwrap(op.base, is_adjoint, n_ctrls + len(op.control_wires))


def _pure_recursive_add_decomps(op: Operator2, rules_map: dict, **kwargs):
    if op in rules_map:
        return
    decomps = list_decomps(op)
    applicable_rules = [r for r in decomps if r.is_applicable(**op.arguments)]
    rules_map[op] = applicable_rules
    for rule in applicable_rules:
        resources = rule.compute_resources(**op.arguments)
        for r in resources.gate_counts:
            _recursive_all_decomps(r, rules_map, **kwargs)


def _recursive_all_decomps(op: Operator2, rules_map: dict, **kwargs):
    if op in rules_map:
        return

    base, is_adj, has_n_ctrls = _unwrap(op)

    _pure_recursive_add_decomps(op, rules_map, **kwargs)

    if kwargs["adj"] and not has_n_ctrls:
        # if adj op appears from resources, make sure to also include target
        target = base if is_adj else op
        _pure_recursive_add_decomps(target, rules_map, **kwargs)

    if not has_n_ctrls and not is_adj:
        if n_ctrls := kwargs["n_ctrls"]:
            target = ctrl(base, Wire[n_ctrls], Bool[n_ctrls])
            _pure_recursive_add_decomps(target, rules_map, **kwargs)
        if adj_n_ctrls := kwargs["adj_n_ctrls"]:
            target = ctrl(adjoint(base), Wire[adj_n_ctrls], Bool[adj_n_ctrls])
            _pure_recursive_add_decomps(target, rules_map, **kwargs)


def all_decomps(
    op: Operator2, adj: bool = True, n_ctrls: int = 1, adj_n_ctrls: int = 1
) -> dict[Operator2, list[DecompositionRule]]:
    """Collect all decomposition rules downstream of an Operator.

    Args:
        op (Operator2): an operator that we will want to decompose:
        adj=True (bool): whether or not to also include the rules for handling the adjoint of every
            operator
        n_ctrls=1 (int): How many controlled versions to include for each base operator
        adj_n_ctrls=1 (int): How many controlled version to include for the adjoint of each base operator

    Returns:
        dict[Operator2, list[DecompositionRule]]: A map from abstract operators to their applicable rules

    >>> from pennylane.typing import Wire, Float

    >>> rule_map = all_decomps(qp.X(0))
    >>> len(rule_map)
    256
    >>> rule_map[qp.CZ(Wire[2])]
    [DecompositionRule(name=_cz_to_cps),
    DecompositionRule(name=_cz_to_cnot),
    DecompositionRule(name=_cz_to_ppr),
    DecompositionRule(name=_cz_lattice_surgery_ppm)]
    >>> rule_map[qp.adjoint(qp.PhaseShift(Float, Wire[1]))]
    [DecompositionRule(name=adjoint_rotation),
    DecompositionRule(name=adjoint(_phaseshift_to_rz_gp))]

    """
    if n_ctrls > 1:
        raise NotImplementedError
    if adj_n_ctrls > 1:
        raise NotImplementedError

    rules_map = {}
    _recursive_all_decomps(
        abstractify(op), rules_map, adj=adj, n_ctrls=n_ctrls, adj_n_ctrls=adj_n_ctrls
    )
    return rules_map
