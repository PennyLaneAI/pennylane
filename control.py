import inspect
import ast

import astunparse

import torch

import pennylane as qml
from pennylane.tape import QuantumTape

def transform_top_level_function(node: ast.FunctionDef):

    tape_name = ast.Name()
    tape_name.id = "FunctionTape"

    tape = ast.Call()
    tape.func = tape_name
    arg_names = list((map(lambda arg: arg.arg, node.args.args)))
    tape.args = list((map(lambda arg: ast.Str(arg), arg_names)))
    tape.keywords = {}

    v = ast.withitem()
    v.context_expr = tape
    v.optional_vars = ast.Name(node.name)

    with_block = ast.With()
    with_block.body = node.body
    with_block.items = [v]
    return with_block, arg_names


class ControlFlowTransformer(ast.NodeTransformer):

    def __init__(self, arg_names, *args, **kwargs):
        self.arg_names = arg_names
        super().__init__(*args, **kwargs)

    def visit_If(self, node):
        self.generic_visit(node)

        if_tape_name = ast.Name()
        if_tape_name.id = "IfTape"

        if_tape = ast.Call()
        if_tape.func = if_tape_name
        if_tape.args = [node.test]
        if_tape.keywords = {}

        with_block = ast.With()
        with_block.body = node.body
        with_block.items = [if_tape]
        return with_block

    def visit_While(self, node):
        self.generic_visit(node)

        tape_name = ast.Name()
        tape_name.id = "WhileTape"

        tape = ast.Call()
        tape.func = tape_name
        tape.args = [node.test]
        tape.keywords = {}

        with_block = ast.With()
        with_block.body = node.body
        with_block.items = [tape]
        return with_block

    def visit_For(self, node):
        self.generic_visit(node)

        tape_name = ast.Name()
        tape_name.id = "ForTape"

        tape = ast.Call()
        tape.func = tape_name
        tape.args = [node.test]
        tape.keywords = {}

        with_block = ast.With()
        with_block.body = node.body
        with_block.items = [tape]
        return with_block

    def visit_Name(self, node):

        if node.id in self.arg_names:
            attr = ast.Attribute()
            attr.attr = node.id
            attr.value = ast.Name("circuit")
            return attr
        return node


def script(fn):
    fn_source = inspect.getsource(fn)
    print(fn_source)
    fn_ast = ast.parse(fn_source)
    trimmed_ast = fn_ast.body[0]
    tape_ast, arg_names = transform_top_level_function(trimmed_ast)
    cft = ControlFlowTransformer(arg_names)
    transformed_ast = cft.visit(tape_ast)
    print(astunparse.unparse(transformed_ast))


@script
def circuit(x, y):
    while y > 0 and x == 0:
        if x > 3:
            if y < 10:
                qml.RX(x + 10, wires=0)
    return qml.expval(qml.PauliX(wires=0))