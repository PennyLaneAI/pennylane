import inspect
import ast

import astunparse

import pennylane as qml
from pennylane.tape import QuantumTape


class ControlFlowTransformer(ast.NodeTransformer):

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

cft = ControlFlowTransformer()

def script(fn):
    fn_source = inspect.getsource(fn)
    fn_ast = ast.parse(fn_source)
    transformed_ast = cft.visit(fn_ast)
    print(transformed_ast)
    print(astunparse.unparse(transformed_ast))


@script
def circuit(x, y):
    while y > 0 and x == 0:
        with QuantumTape():
            print("hi")
        if x > 3:
            qml.RX(x, wires=0)
    return qml.expval(qml.PauliX(wires=0))