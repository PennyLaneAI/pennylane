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

"""Tests that the resource functions of decomposition rules registered for ``Operator2``
operators accept the rule's arguments passed positionally, mirroring the rule's signature
instead of relying on a keyword-only catch-all."""

import inspect
import re

import pytest

from pennylane.core.operator import Operator2
from pennylane.decomposition.decomposition_rule import _decompositions_var
from pennylane.decomposition.utils import translate_op_alias

_SYMBOLIC_NAME = re.compile(r"(?:Adjoint|C|Pow)\((\w+)\)")


def _operator2_names():
    """The registry names of all ``Operator2`` subclasses."""
    names, seen, stack = set(), set(), [Operator2]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            stack.append(sub)
            if not sub.__name__.startswith("_"):
                names.add(translate_op_alias(sub.__name__))
    return names


def _is_operator2_entry(op_name, op2_names):
    """Whether a registry entry belongs to an ``Operator2``, directly or symbolically.

    Legacy operators are excluded because their resource functions follow the unordered
    ``resource_keys`` convention, for which positional calling is not defined.
    """
    if match := _SYMBOLIC_NAME.fullmatch(op_name):
        return match.group(1) in op2_names
    return op_name in op2_names


def _rule_cases():
    """All (op name, rule) pairs in the global registry relevant to ``Operator2``."""
    op2_names = _operator2_names()
    return [
        pytest.param(op_name, rule, id=f"{op_name}-{rule.name}")
        for op_name, rules in _decompositions_var.get().items()
        if _is_operator2_entry(op_name, op2_names)
        for rule in rules
    ]


def _num_positional_params(fn):
    """The number of parameters of ``fn`` that can be passed positionally."""
    params = inspect.signature(fn).parameters.values()
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)


_CASES = _rule_cases()


@pytest.mark.unit
def test_sweep_is_not_empty():
    """Test that the sweep finds the registered Operator2 rules, guarding against the
    entry filter silently excluding everything."""
    assert len(_CASES) > 50


@pytest.mark.unit
@pytest.mark.parametrize("op_name, rule", _CASES)
def test_resource_fn_accepts_positional_arguments(op_name, rule):
    """Test that a rule's resource function accepts the rule's arguments positionally."""
    # pylint: disable=protected-access
    resource_fn = rule._compute_resources
    if resource_fn is None:
        pytest.skip("Rule does not define a resource function.")
    num_args = _num_positional_params(rule._impl)
    try:
        inspect.signature(resource_fn).bind(*([None] * num_args))
    except TypeError as e:
        pytest.fail(
            f"The resource function of decomposition rule '{rule.name}' for '{op_name}' does "
            f"not accept the rule's {num_args} argument(s) passed positionally. Give the "
            f"resource function the same signature as the rule instead of a keyword-only "
            f"catch-all. ({e})"
        )
