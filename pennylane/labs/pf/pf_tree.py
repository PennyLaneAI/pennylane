"""AST for product formula expressions"""

from __future__ import annotations
from abc import abstractmethod
from typing import Dict, List

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
    def fragment_node(label: str) -> PF_Fragment:
        """Construct a FRAGMENT node"""

        return PF_Fragment(label)

    @staticmethod
    def nested_commutator_node(l_child: PF_Node, m_child: PF_Node, r_child: PF_Node) -> PF_Nested_Commutator:
        """Construct a NESTED_COMMUTATOR node"""

        return PF_Nested_Commutator(l_child, m_child, r_child)

    @staticmethod
    def empty_node() -> PF_Empty:
        """ Construt an EMPTY node"""

        return PF_Empty()

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
    def eval(self, fragments: Dict[str, Fragment]) -> Fragment:
        """Evaluate the expression"""
        raise NotImplementedError

    @classmethod
    def second_order_trotter(cls, labels: List[str]) -> PF_Node:
        """Returns an AST representation of the second order trotter product formula"""
        #scalar = -(delta**2) / 24
        #epsilon = VibronicMatrix(self.states, self.modes, sparse=self.sparse)

        #for i in range(self.states):
        #    for j in range(i + 1, self.states + 1):
        #        epsilon += self._commute_fragments(i, i, j)
        #        for k in range(i + 1, self.states + 1):
        #            epsilon += 2 * self._commute_fragments(k, i, j)

        #epsilon *= scalar

        n_fragments = len(labels)
        fragments = [cls.fragment_node(label) for label in labels]
        ast = cls.empty_node()

        for i in range(n_fragments):
            for j in range(i + 1, n_fragments + 1):
                pass


class PF_Add(PF_Node):
    """Product formula ADD node"""

    def __init__(self, l_child: PF_Node, r_child: PF_Node):
        self.l_child = l_child
        self.r_child = r_child

        self.l_fragments = l_child.fragments
        self.r_fragments = r_child.fragments
        self.fragments = set.union(self.l_fragments, self.r_fragments)

    def eval(self, fragments: Dict[str, Fragment]) -> Fragment:
        l_dict = {label: fragments[label] for label in self.l_fragments}
        r_dict = {label: fragments[label] for label in self.r_fragments}

        return self.l_child.eval(l_dict) + self.r_child.eval(r_dict)

class PF_Multiply(PF_Node):
    """Product formula MULTIPLY node"""

    def __init__(self, l_child: PF_Node, r_child: PF_Node):
        self.l_child = l_child
        self.r_child = r_child

        self.l_fragments = l_child.fragments
        self.r_fragments = r_child.fragments
        self.fragments = set.union(self.l_fragments, self.r_fragments)

    def eval(self, fragments: Dict[str, Fragment]) -> Fragment:
        l_dict = {label: fragments[label] for label in self.l_fragments}
        r_dict = {label: fragments[label] for label in self.r_fragments}

        return self.l_child.eval(l_dict) @ self.r_child.eval(r_dict)

class PF_Commutator(PF_Node):
    """Product formula COMMUTATOR node"""

    def __init__(self, l_child: PF_Node, r_child: PF_Node):
        self.l_child = l_child
        self.r_child = r_child

        self.l_fragments = l_child.fragments
        self.r_fragments = r_child.fragments
        self.fragments = set.union(self.l_fragments, self.r_fragments)

    def eval(self, fragments: Dict[str, Fragment]) -> Fragment:
        l_dict = {label: fragments[label] for label in self.l_fragments}
        r_dict = {label: fragments[label] for label in self.r_fragments}

        return self.l_child.eval(l_dict).commutator(self.r_child.eval(r_dict))

class PF_Nested_Commutator(PF_Node):
    """Producted formula NESTED_COMMUTATOR node"""
    def eval(self, fragments: Dict[str, Fragment]):
        raise NotImplementedError

class PF_Scalar(PF_Node):
    """Product formula SCALAR node"""

    def __init__(self, child: PF_Node, scalar: float):
        self.child = child
        self.scalar = scalar
        self.fragments = child.fragments

    def eval(self, fragments: Dict[str, Fragment]) -> Fragment:
        return self.scalar * self.child.eval(fragments)

class PF_Fragment(PF_Node):
    """Product formula FRAGMENT node"""

    def __init__(self, label: str):
        self.label = label
        self.fragments = set(label)

    def eval(self, fragments: Dict[str, Fragment]) -> Fragment:
        return fragments[self.label]

class PF_Empty(PF_Node):
    """Product formula EMPTY node"""

    def eval(self, fragments: Dict[str, Fragment]):
        raise NotImplementedError
