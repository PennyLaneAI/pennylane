from .commutator import *


def test_replacement():
    A = LeafNode([1, 2])
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    X = LeafNode("X")
    Y = LeafNode("Y")
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y), E]
    big_comm.replace_node(A, replacements)
    assert str(big_comm) == "[[[X, Y], [B, C]], [[E, D], E]]"


def test_replacement_too_many():
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    X = LeafNode("X")
    Y = LeafNode("Y")
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y), E, X]
    with pytest.raises(RuntimeError, match="Got more replacement nodes"):
        big_comm.replace_node(A, replacements)


def test_replacement_too_few():
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    X = LeafNode("X")
    Y = LeafNode("Y")
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y)]
    with pytest.raises(RuntimeError, match="Got fewer replacement nodes"):
        big_comm.replace_node(A, replacements)


@pytest.mark.parametrize("value", ["A", 12, [1, 2, 3], {"X", "Y", 1.2}])
def test_mergeability(value):
    A = LeafNode(value)
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    comm1 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, B), D)
    )

    comm2 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), D)
    )

    comm3 = CommutatorNode(
        CommutatorNode(CommutatorNode(A, B), C), CommutatorNode(CommutatorNode(A, D), D)
    )

    assert is_mergeable(comm1, comm2, 0) == False
    assert is_mergeable(comm1, comm2, 4) == True
    assert is_mergeable(comm1, comm3, 0) == False


def test_merge():
    A = LeafNode({"A"})
    B = LeafNode({"B"})
    C = LeafNode({"C"})
    D = LeafNode({"D"})
    comm1 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, B), D)
    )

    comm2 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), D)
    )
    sum_comm = merge(comm1, comm2, 4)

    merged_node = LeafNode({"B", "D"})
    expected = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, merged_node), D)
    )
    assert sum_comm == expected


def test_expand_commutator():
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")

    expected_A = {("A",): 1}
    assert A.expand() == expected_A

    comm_AB = CommutatorNode(A, B)
    expected_AB = {("A", "B"): 1, ("B", "A"): -1}
    assert comm_AB.expand() == expected_AB

    comm_ABC = CommutatorNode(CommutatorNode(A, B), C)
    expected_ABC = {
        ("A", "B", "C"): 1,
        ("B", "A", "C"): -1,
        ("C", "A", "B"): -1,
        ("C", "B", "A"): 1,
    }
    assert comm_ABC.expand() == expected_ABC
