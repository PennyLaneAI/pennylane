"""AST for product formula expressions"""

from __future__ import annotations
from abc import abstractmethod
from abstract_fragment import Fragment

class PF_Node:
    """An AST expressing product formulas"""

    @staticmethod
    def add_node(l_child: PF_Node, r_child: PF_Node) -> PF_Add:
        """Construct an ADD node"""

        return PF_Add(l_child, r_child)

    @staticmethod
    def multiply_node(l_child: PF_Node, r_child: PF_Node) -> PF_Multiply:
        """Construct a MULTIPLY node"""

        return PF_Multiply(l_child, r_child)

    @staticmethod
    def commutator_node(l_child: PF_Node, r_child: PF_Node) -> PF_Commutator:
        """Construct a COMMUTATOR node"""

        return PF_Commutator(l_child, r_child)

    @staticmethod
    def scalar_node(child: PF_Node, scalar: float) -> PF_Scalar:
        """Construct a SCALAR node"""

        return PF_Scalar(child, scalar)

    @staticmethod
    def fragment_node(fragment: Fragment) -> PF_Fragment:
        """Construct a FRAGMENT node"""

        return PF_Fragment(fragment)

    def __add__(self, other: PF_Node) -> PF_Node:
        return PF_Node.add_node(self, other)

    def __sub__(self, other: PF_Node) -> PF_Node:
        return (-1)*(self + other)

    def __mul__(self, scalar: float) -> PF_Node:
        return PF_Node.scalar_node(self, scalar)

    def __matmul__(self, other: PF_Node) -> PF_Node:
        return PF_Node.multiply_node(self, other)

    def commutator(self, other: PF_Node) -> PF_Node:
        """Return the commutator [self, other]"""
        return PF_Node.commutator_node(self, other)

    @abstractmethod
    def eval(self) -> Fragment:
        """Evaluate the expression"""
        raise NotImplementedError

class PF_Add(PF_Node):
    """Product formula ADD node"""

    def __init__(self, l_child: PF_Node, r_child: PF_Node):
        self.l_child = l_child
        self.r_child = r_child

    def eval(self) -> Fragment:
        return self.l_child.eval() + self.r_child.eval()

class PF_Multiply(PF_Node):
    """Product formula MULTIPLY node"""

    def __init__(self, l_child: PF_Node, r_child: PF_Node):
        self.l_child = l_child
        self.r_child = r_child

    def eval(self) -> Fragment:
        return self.l_child.eval() @ self.r_child.eval()

class PF_Commutator(PF_Node):
    """Product formula COMMUTATOR node"""

    def __init__(self, l_child: PF_Node, r_child: PF_Node):
        self.l_child = l_child
        self.r_child = r_child

    def eval(self) -> Fragment:
        return self.l_child.eval().commutator(self.r_child.eval())

class PF_Scalar(PF_Node):
    """Product formula SCALAR node"""

    def __init__(self, child: PF_Node, scalar: float):
        self.child = child
        self.scalar = scalar

    def eval(self) -> Fragment:
        return self.scalar * self.child.eval()

class PF_Fragment(PF_Node):
    """Product formula FRAGMENT node"""

    def __init__(self, fragment: Fragment):
        self.fragment = fragment

    def eval(self) -> Fragment:
        return self.fragment
