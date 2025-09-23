class Node:
    """Abstract base class for all nodes in the commutator tree."""

    pass


class LeafNode(Node):
    """
    Represents a leaf node in the commutator tree, holding a base value.

    Think of this as the 'A', 'B', or 'C' in an expression like [[A, B], C].
    In other words, this is a symbol.
    """

    def __init__(self, value: any):
        self.value = value

    def __str__(self) -> str:
        """Returns the string representation of the leaf's value."""
        return str(self.value)


class CommutatorNode(Node):
    """
    Represents an internal commutator node [left, right] in the tree.

    This node contains the method to search for and replace child nodes.
    """

    def __init__(self, left: Node, right: Node):
        if not isinstance(left, Node) or not isinstance(right, Node):
            raise TypeError("Both left and right children must be Node instances.")
        self.left = left
        self.right = right

    def __str__(self) -> str:
        """Returns the standard mathematical string representation of the commutator."""
        return f"[{self.left}, {self.right}]"

    def replace_node(self, target_node: Node, new_node: Node) -> bool:
        """
        Recursively finds a target child node and replaces it.
        This method traverses the tree starting from the current node's children.

        Args:
            target_node: The specific node object to be replaced.
            new_node: The new node object to insert in its place.

        Returns:
            True if the replacement was successful, False otherwise.
        """
        if not isinstance(new_node, Node):
            raise TypeError("The replacement node must be a Node instance.")

        if self.left is target_node:
            self.left = new_node
            return True

        if self.right is target_node:
            self.right = new_node
            return True

        # If not a direct child, recurse into children that are CommutatorNodes
        if isinstance(self.left, CommutatorNode) and self.left.replace_node(target_node, new_node):
            return True

        if isinstance(self.right, CommutatorNode) and self.right.replace_node(
            target_node, new_node
        ):
            return True

        # Target node was not found in this branch of the tree
        return False


A = LeafNode("A")
B = LeafNode("B")
C = LeafNode("C")
D = LeafNode("D")
comm_AB = CommutatorNode(A, B)
expr = CommutatorNode(comm_AB, C)
print(f"Original Expression: {expr}")
new_comm = CommutatorNode(C, D)
expr.replace_node(target_node=A, new_node=new_comm)
print(f"New Expression: {expr}")


comm_AB = CommutatorNode(A, B)
expr = CommutatorNode(comm_AB, C)
print(f"\nOriginal Expression: {expr}")
X = LeafNode("X")
Y = LeafNode("Y")
comm_XY = CommutatorNode(X, Y)
# comm_XY = D
expr.replace_node(target_node=comm_AB, new_node=comm_XY)
print(f"New Expression: {expr}")
