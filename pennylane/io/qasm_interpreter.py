from openqasm3.visitor import QASMVisitor, QASMNode
from typing import Optional, TypeVar
import re

T = TypeVar("T")

class QasmInterpreter(QASMVisitor):
    """
    Inherits generic_visit(self, node: QASMNode, context: Optional[T]) which takes the
    top level node of the AST as a parameter and recursively descends the AST, calling the
    user-defined visitor function on each node.
    """

    def visit(self, node: QASMNode, context: Optional[T] = None):
        """
        Applied to each node in the AST.
        """
        if hasattr(node, "name"):
            if re.search("Identifier", node.name) is not None:
                self.identifier(node, context)
            # TODO: call appropriate handler methods here
            else:
                raise Warning(f"An unrecognized QASM instruction was encountered: {node.name}")

    def identifier(self, node: QASMNode, context: Optional[T] = None):
        """
        Registerss an identifier in the current context.
        """
        context.identifiers.append(node.name.name)
