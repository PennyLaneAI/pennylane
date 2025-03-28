from functools import singledispatch
from typing import Any, Callable

import numpy as np
import openqasm3
from openqasm3 import ast as oq3_ast

import pennylane as qml

gate_map = {
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
    "u": qml.U3,
    "gphase": qml.GlobalPhase,
}

Binary_operator_map: dict[oq3_ast.BinaryOperator, Callable[[Any, Any], Any]] = {
    getattr(oq3_ast.BinaryOperator, ">"): lambda a, b: a > b,
    getattr(oq3_ast.BinaryOperator, "<"): lambda a, b: a < b,
    getattr(oq3_ast.BinaryOperator, ">="): lambda a, b: a >= b,
    getattr(oq3_ast.BinaryOperator, "<="): lambda a, b: a <= b,
    getattr(oq3_ast.BinaryOperator, "=="): lambda a, b: a == b,
    getattr(oq3_ast.BinaryOperator, "!="): lambda a, b: a != b,
    getattr(oq3_ast.BinaryOperator, "*"): lambda a, b: a * b,
    getattr(oq3_ast.BinaryOperator, "+"): lambda a, b: a + b,
    getattr(oq3_ast.BinaryOperator, "-"): lambda a, b: a - b,
    getattr(oq3_ast.BinaryOperator, "/"): lambda a, b: a / b,
    getattr(oq3_ast.BinaryOperator, "%"): lambda a, b: a % b,
}


numpy_funcs = {
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


class Environment:

    def __init__(self):
        self.qubits = {}
        self.c_reg = {"pi": np.pi, "tau": 2 * np.pi, "e": np.e}
        self.inclusions = []


@singledispatch
def visit(node, env: Environment) -> Any:
    raise NotImplementedError(f"no registration for {node}")


@visit.register
def _(node: oq3_ast.Program, env):
    for statement in node.statements:
        visit(statement, env)


@visit.register
def _(node: oq3_ast.Include, env):
    # or something else?
    env.inclusions.append(node.filename)


@visit.register
def _(node: oq3_ast.Identifier, env):
    if node.name in env.qubits:
        return env.qubits[node.name]
    elif node.name in gate_map:
        return gate_map[node.name]
    return env.c_reg[node.name]


@visit.register
def _(node: oq3_ast.DiscreteSet, env):
    return [visit(v, env) for v in node.values]


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
def _(node: oq3_ast.QubitDeclaration, env):
    if not node.size:
        env.qubits[node.qubit.name] = len(env.qubits)
    else:
        cur_num = len(env.qubits)
        s = visit(node.size, env)
        env.qubits[node.qubit.name] = list(range(cur_num, cur_num + s))


@visit.register
def _(node: oq3_ast.QuantumGate, env):
    gate_type = visit(node.name, env)
    args = [visit(sub_node, env) for sub_node in node.arguments]
    num_control_wires = 0
    control_values = []
    for mod in node.modifiers:
        if mod.modifier == oq3_ast.GateModifierName.ctrl:
            if mod.argument:
                n = visit(mod.argument, env)
                num_control_wires += n
                control_values.extend([1] * n)
            else:
                num_control_wires += 1
                control_values.append(1)
        if mod.modifier == oq3_ast.GateModifierName.negctrl:
            if mod.argument:
                n = visit(mod.argument, env)
                num_control_wires += n
                control_values.extend([0] * n)
            else:
                num_control_wires += 1
                control_values.append(0)

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
    return qml.measure(env.qubits[node.qubit.name])


@visit.register
def _(node: oq3_ast.ClassicalDeclaration, env):
    if node.init_expression:
        init_value = visit(node.init_expression, env)
        env.c_reg[node.identifier.name] = init_value
    else:
        raise NotImplementedError


@visit.register
def _(node: oq3_ast.ClassicalAssignment, env):
    rvalue = visit(node.rvalue, env)
    name = node.lvalue.name
    if node.op == getattr(oq3_ast.AssignmentOperator, "="):
        env.c_reg[name] = rvalue
    if node.op == getattr(oq3_ast.AssignmentOperator, "+="):
        env.c_reg[name] += rvalue
    else:
        raise NotImplementedError(f"{node}")


@visit.register
def _(node: oq3_ast.ConstantDeclaration, env):
    if not node.init_expression:
        raise ValueError("constants must be initialized")
    init_value = visit(node.init_expression, env)
    env.c_reg[node.identifier.name] = init_value


@visit.register(oq3_ast.BooleanLiteral)
@visit.register(oq3_ast.IntegerLiteral)
@visit.register(oq3_ast.FloatLiteral)
def _(node: oq3_ast.BooleanLiteral, env):
    return node.value


@visit.register
def _(node: oq3_ast.BinaryExpression, env):
    fn = Binary_operator_map.get(node.op)
    if fn is None:
        raise NotImplementedError
    lhs = visit(node.lhs, env)
    rhs = visit(node.rhs, env)
    return fn(lhs, rhs)


@visit.register
def _(node: oq3_ast.FunctionCall, env):
    args = [visit(identifier, env) for identifier in node.arguments]
    func_name = node.name.name
    if func_name in env.c_reg:
        return env.c_reg[func_name](*args)
    if func_name in numpy_funcs:
        return getattr(np, func_name)(*args)
    raise NotImplementedError(f"unable to identify {func_name}")


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
        env.c_reg[node.identifier.name] = i
        for stmt in node.block:
            visit(stmt, env)


@visit.register
def _(node: oq3_ast.SubroutineDefinition, env):
    name = node.name.name

    def f(*args):
        new_env = Environment()
        for a, target in zip(args, node.arguments, strict=True):
            new_env.c_reg[target.name.name] = a
        for stmt in node.body:
            if isinstance(stmt, oq3_ast.ReturnStatement):
                return visit(stmt.expression, new_env)
            visit(stmt, new_env)

    env.c_reg[name] = f


def from_openqasm3(source: str):
    node = openqasm3.parse(source)
    env = Environment()
    return visit(node, env)
