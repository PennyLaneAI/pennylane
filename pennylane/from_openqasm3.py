# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains a converted for executing openqasm3."""


from copy import deepcopy
from functools import singledispatch
from typing import Any, Callable

import numpy as np
import openqasm3
from openqasm3 import ast as oq3_ast

import pennylane as qml

std_gates = {
    "h": qml.H,
    "x": qml.X,
    "y": qml.Y,
    "z": qml.Z,
    "s": qml.S,
    "t": qml.T,
    "tdg": lambda *args, **kwargs: qml.adjoint(qml.T(*args, **kwargs)),
    "sx": qml.SX,
    "rx": qml.RX,
    "ry": qml.RY,
    "rz": qml.RZ,
    "p": qml.PhaseShift,
}


_special_chars = {getattr(oq3_ast.BinaryOperator, "&&"), getattr(oq3_ast.BinaryOperator, "||")}
Binary_operator_map: dict[oq3_ast.BinaryOperator, Callable[[Any, Any], Any]] = {
    op_enum: eval(f"lambda a, b: a {op_enum.name} b")
    for op_enum in oq3_ast.BinaryOperator
    if op_enum not in _special_chars
}
Binary_operator_map[getattr(oq3_ast.BinaryOperator, "&&")] = (lambda a, b: a and b,)
Binary_operator_map[getattr(oq3_ast.BinaryOperator, "||")] = (lambda a, b: a or b,)

_numpy_funcs = {
    "arccos",
    "arcsin",
    "arctan",
    "cos",
    "exp",
    "floor",
    "log",
    "mod",
    "pow",
    "sin",
    "sqrt",
    "tan",
}
_numpy_func_map = {name: getattr(np, name) for name in _numpy_funcs}


class Environment:

    def __init__(self, kwargs):
        self.num_qubits = 0
        self._reg = kwargs
        self._reg.update(
            {
                "pi": np.pi,
                "tau": 2 * np.pi,
                "e": np.e,
                "u": qml.U3,
                "gphase": qml.GlobalPhase,
            }
        )
        self._reg.update(_numpy_func_map)
        self.output_identifiers = []

    def __getitem__(self, key):
        return self._reg[key]

    def __setitem__(self, key, value):
        self._reg[key] = value


@singledispatch
def visit(node: oq3_ast.QASMNode, env: Environment) -> Any:
    """A single dispatch function interpreting a ``openqasm3.ast.QASMNode``."""
    raise NotImplementedError(f"no registration for {node}")


@visit.register
def _(node: oq3_ast.Program, env):
    for statement in node.statements:
        visit(statement, env)


@visit.register
def _(node: oq3_ast.Include, env):
    # or something else?
    if node.filename == "stdgates.inc":
        env._reg.update(std_gates)
    else:
        raise NotImplementedError(
            f"unknown filename {node.filename}. Currently only support stdgates.inc"
        )


@visit.register
def _(node: oq3_ast.IODeclaration, env: Environment):
    if node.io_identifier == oq3_ast.IOKeyword.input:
        if node.identifier.name not in env._reg:
            raise ValueError(f"need value for {node.identifier.name} passed.")
    else:
        env.output_identifiers.append(node.identifier.name)
        env[node.identifier.name] = None


@visit.register
def _(node: oq3_ast.Identifier, env):
    try:
        return env[node.name]
    except KeyError as e:
        raise KeyError(f"variable {node.name} has no set value.") from e


@visit.register
def _(node: oq3_ast.Concatenation, env):
    return visit(node.lhs, env) + visit(node.rhs, env)


@visit.register
def _(node: oq3_ast.DiscreteSet, env):
    return {visit(v, env) for v in node.values}


@visit.register
def _(node: oq3_ast.IndexedIdentifier, env):
    target = visit(node.name, env)
    # pretty sure this is quite wrong
    indices = []
    for idx in node.indices:
        for iidx in idx:
            indices.append(visit(iidx, env))
    return target[indices[0]]


@visit.register
def _(node: oq3_ast.RangeDefinition, env):
    start = visit(node.start, env) if node.start else None
    end = visit(node.end, env) if node.end else None
    step = visit(node.step, env) if node.step else None
    return range(start, end, step)


@visit.register
def _(node: oq3_ast.QuantumGate, env):
    num_control_wires = 0
    control_values = []
    for mod in node.modifiers:
        if mod.modifier in {oq3_ast.GateModifierName.ctrl, oq3_ast.GateModifierName.negctrl}:
            n = visit(mod.argument, env) if mod.argument else 1
            num_control_wires += n
            control_values.extend([mod.modifier == oq3_ast.GateModifierName.ctrl] * n)

    gate_type = visit(node.name, env)
    args = [visit(sub_node, env) for sub_node in node.arguments]
    qubits = [visit(sub_node, env) for sub_node in node.qubits]
    controls = qubits[:num_control_wires]
    targets = qubits[num_control_wires:]
    op = gate_type(*args, wires=targets)

    for mod in node.modifiers:
        if mod.modifier == oq3_ast.GateModifierName.inv:
            op = qml.adjoint(op)
        elif mod.modifier == oq3_ast.GateModifierName.pow:
            z = visit(mod.argument, env)
            op = qml.pow(op, z)
    if controls:
        op = qml.ctrl(op, controls, control_values=control_values)


@visit.register
def _(node: oq3_ast.QuantumMeasurement, env):
    return qml.measure(visit(node.qubit, env))


@visit.register
def _(node: oq3_ast.QuantumMeasurementStatement, env):
    mv = qml.measure(visit(node.measure.qubit, env))
    env[node.target.name] = mv


@visit.register
def _(node: oq3_ast.QuantumReset, env):
    return qml.measure(visit(node.qubits, env), reset=True)


@visit.register
def _(node: oq3_ast.ClassicalDeclaration, env):
    if node.init_expression:
        init_value = visit(node.init_expression, env)
        env[node.identifier.name] = init_value
    else:
        raise NotImplementedError


@visit.register
def _(node: oq3_ast.QubitDeclaration, env: Environment):
    if not node.size:
        env[node.qubit.name] = env.num_qubits
        env.num_qubits += 1
    else:
        s = visit(node.size, env)
        env[node.qubit.name] = list(range(env.num_qubits, env.num_qubits + s))
        env.num_qubits += s


@visit.register
def _(node: oq3_ast.ClassicalAssignment, env):
    if node.op == getattr(oq3_ast.AssignmentOperator, "="):
        env[node.lvalue.name] = visit(node.rvalue, env)
    if node.op == getattr(oq3_ast.AssignmentOperator, "+="):
        env[node.lvalue.name] += visit(node.rvalue, env)
    else:
        raise NotImplementedError(f"{node}")


@visit.register
def _(node: oq3_ast.ConstantDeclaration, env):
    if not node.init_expression:
        raise ValueError("constants must be initialized")
    init_value = visit(node.init_expression, env)
    env[node.identifier.name] = init_value


@visit.register(oq3_ast.BooleanLiteral)
@visit.register(oq3_ast.IntegerLiteral)
@visit.register(oq3_ast.FloatLiteral)
def _(node: oq3_ast.BooleanLiteral, env):
    return node.value


@visit.register
def _(node: oq3_ast.BinaryExpression, env):
    fn = Binary_operator_map.get(node.op)
    if fn is None:
        raise NotImplementedError(f"no implemented function for {node.op}")
    lhs = visit(node.lhs, env)
    rhs = visit(node.rhs, env)
    return fn(lhs, rhs)


@visit.register
def _(node: oq3_ast.UnaryExpression, env):
    if node.op == getattr(oq3_ast.UnaryOperator, "~"):
        return ~visit(node.expression, env)
    if node.op == getattr(oq3_ast.UnaryOperator, "-"):
        return -visit(node.expression, env)
    raise NotImplementedError(f"{node.op} is not yet implemented")


@visit.register
def _(node: oq3_ast.FunctionCall, env):
    return visit(node.name, env)(*(visit(identifier, env) for identifier in node.arguments))


@visit.register
def _(node: oq3_ast.ExpressionStatement, env):
    visit(node.expression, env)


@visit.register
def _(node: oq3_ast.BranchingStatement, env):
    condition = visit(node.condition, env)

    def true_fn():
        for sub_node in node.if_block:
            visit(sub_node, env)

    def false_fn():
        for sub_node in node.else_block:
            visit(sub_node, env)

    qml.cond(condition, true_fn, false_fn)()


@visit.register
def _(node: oq3_ast.WhileLoop, env):
    def cond_fn():
        return visit(node.while_condition, env)

    def body_fn():
        for stmt in node.block:
            visit(stmt, env)

    while cond_fn():
        body_fn()


@visit.register
def _(node: oq3_ast.ForInLoop, env):
    for i in visit(node.set_declaration, env):
        env[node.identifier.name] = i
        for stmt in node.block:
            visit(stmt, env)


@visit.register
def _(node: oq3_ast.SubroutineDefinition, env):
    name = node.name.name

    def f(*args):
        new_env = deepcopy(env)
        for a, target in zip(args, node.arguments, strict=True):
            new_env[target.name.name] = a
        for stmt in node.body:
            if isinstance(stmt, oq3_ast.ReturnStatement):
                return visit(stmt.expression, new_env)
            visit(stmt, new_env)

    env[name] = f


@visit.register
def _(node: oq3_ast.AliasStatement, env: Environment):
    env[node.target.name] = visit(node.value, env)


def from_openqasm3(source: str, **inputs):
    r"""Interpret a string of openqasm3 using pennylane.

    Args:
        source (str): a string containing valid openqasm3
        **inputs (dict): Any variable inputs. Names must match those declared
            in the openqasm.

    .. code-block::

        OPENQASM 3;
        include "stdgates.inc";

        qubit[4] q;

        // loop over a discrete set of values
        for int[32] i in {1, 2, 3} {
            bit b = measure q[i];
            if (b == 0) ctrl @ inv @ s q[i-1], q[i];
        }

    >>> def f():
    ...     from_openqasm3(example)()
    >>> print(qml.draw(f)())
    0: ──────╭●────────────────────┤
    1: ──┤↗├─╰S†──────╭●───────────┤
    2: ───║───║───┤↗├─╰S†──────╭●──┤
    3: ───║───║────║───║───┤↗├─╰S†─┤
          ╚═══╝    ║   ║    ║   ║
                   ╚═══╝    ║   ║
                            ╚═══╝

    """
    node = openqasm3.parse(source)
    env = Environment(inputs)
    visit(node, env)
    return [env[output_expr] for output_expr in env.output_identifiers]
