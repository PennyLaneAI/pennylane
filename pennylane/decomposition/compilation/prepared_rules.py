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
Defines a tool for gathering and preparing all rules that will needed
for the mlir graph decomposition.
"""

import functools

from pennylane.capture import pause, subroutine, symbolic_array
from pennylane.core.operator import Operator2
from pennylane.pytrees import flatten, unflatten

from .all_decomps import all_decomps
from .goid import graph_op_id


def _to_array(arr):
    if not hasattr(arr, "shape"):
        return arr
    if arr.shape == (0,):
        return ()
    if type(arr).__name__ == "AbstractQubit":
        return symbolic_array((), int)
    return symbolic_array(arr.shape, arr.dtype)


class PreparedRule:
    """A contained for a rule that is ready to be called in a qjit context for lowering to mlir.
    This object simply needs to be called at the right time.

    Args:
        op (Operator2): an abstractified operator 2 instance
        rule (DecompositionRule): the decomposition rule
        new_rule_name (str): the name the rule should get in mlir.

    .. code-block:: python
        rule = PreparedRule(qp.X(Wire[1]), qp.list_decomps(qp.X)[0], "X_decomp1")

        @qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
        @qp.qnode(qp.device('null.qubit', wires=10))
        def c():
            rule()
            return qp.state()

        print(c.mlir)

    .. code-block::

        ...
        func.func private @X_decomp1(%arg0: !qref.reg<10>, %arg1: tensor<i64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
            %cst = arith.constant -1.5707963267948966 : f64
            %cst_0 = arith.constant 3.1415926535897931 : f64
            %extracted = tensor.extract %arg1[] : tensor<i64>
            %0 = qref.get %arg0[%extracted] : !qref.reg<10>, i64 -> !qref.bit
            qref.custom "RX"(%cst_0) %0 : !qref.bit
            qref.gphase(%cst)
            return
        }

    """

    def __init__(self, op, rule, new_rule_name):
        self.op = op
        self.rule = rule
        self.new_rule_name = new_rule_name

    def __repr__(self):
        return f"<PreparedRule: {self.op}, {self.rule.name}>"

    def __call__(self):
        leaves, tree = flatten(self.op.arguments)
        new_leaves = (_to_array(l) for l in leaves)
        new_args = unflatten(new_leaves, tree)
        with pause():
            # performs processing like sticking into Wires object
            repacked = type(self.op)(**new_args)

        @functools.wraps(self.rule)
        def wrapper_with_new_name(**arguments):
            return self.rule(**arguments)

        wrapper_with_new_name.__name__ = self.new_rule_name
        f = subroutine(
            wrapper_with_new_name,
            static_argnames=self.op.static_argnames + self.op.compilable_argnames,
        )
        f(**repacked.arguments)


def all_prepared_decomps(
    op: Operator2, adj: bool = True, n_ctrls: int = 1, adj_n_ctrls: int = 1
) -> tuple[list[PreparedRule], dict]:
    """Collect and prepare decomposition rules for lowering to mlir.

    Args:
        op (Operator2): an operator that we will want to decompose:
        adj=True (bool): whether or not to also include the rules for handling the adjoint of every
            operator
        n_ctrls=1 (int): How many controlled versions to include for each base operator
        adj_n_ctrls=1 (int): How many controlled version to include for the adjoint of each base operator

    Returns:
        list[PreparedRule], dict: a list of objects that simply need to be called in a qjit context, and a dictionary
            of the metadata that will need to be added to each rule after lowering

    >>> rules, metadata = all_prepared_decomps(qp.X(0))
    >>> rules[0]
    <PreparedRule: PauliX, _paulix_to_rx>
    >>> metadata["_paulix_to_rx_PauliX{}{wires:1}{}"]
    {'target_gate': 'PauliX{}{wires:1}{}',
      'resources': {'operations': {'GlobalPhase{phi:f64}{}{}': 1,
        'RX{0:f64}{wires:1}{}': 1}}}

    .. code-block:: python

        @qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
        @qp.qnode(qp.device('null.qubit', wires=10))
        def c():
            rules[0]()
            return qp.state()

        print(c.mlir)

    .. code-block::

        ...
        func.func private @"_paulix_to_rx_PauliX{}{wires:1}{}"(%arg0: !qref.reg<10>, %arg1: tensor<i64>) attributes {llvm.linkage = #llvm.linkage<internal>} {
            %cst = arith.constant -1.5707963267948966 : f64
            %cst_0 = arith.constant 3.1415926535897931 : f64
            %extracted = tensor.extract %arg1[] : tensor<i64>
            %0 = qref.get %arg0[%extracted] : !qref.reg<10>, i64 -> !qref.bit
            qref.custom "RX"(%cst_0) %0 : !qref.bit
            qref.gphase(%cst)
            return
            }
        }
        ...


    """
    rules_map = all_decomps(op, adj=adj, n_ctrls=n_ctrls, adj_n_ctrls=adj_n_ctrls)

    prepared_rules = []
    metadata_map = {}

    for target, rules in rules_map.items():
        for rule in rules:
            goid = graph_op_id(target)
            rule_name = f"{rule.name}_{goid}"

            prepared_rules.append(PreparedRule(target, rule, rule_name))

            resources = rule.compute_resources(**target.arguments).gate_counts
            prepared_resources = {graph_op_id(r): c for r, c in resources.items()}
            metadata_map[rule_name] = {
                "target_gate": goid,
                "resources": {"operations": prepared_resources},
            }

    return prepared_rules, metadata_map
