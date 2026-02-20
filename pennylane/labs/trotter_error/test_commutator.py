from collections import defaultdict

import pytest

from .commutator import *


def test_replacement():
    A = LeafNode({1, 2})
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
    replaced_comm = replace_node(big_comm, A, replacements)
    assert str(replaced_comm) == "[[[X, Y], [B, C]], [[E, D], E]]"
    assert str(big_comm) == "[[{1, 2}, [B, C]], [[{1, 2}, D], E]]"

    replacements = [E, CommutatorNode(X, Y)]
    replaced_comm = replace_node(big_comm, A, replacements)
    assert str(replaced_comm) == "[[E, [B, C]], [[[X, Y], D], E]]"
    assert str(big_comm) == "[[{1, 2}, [B, C]], [[{1, 2}, D], E]]"


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
        _ = replace_node(big_comm, A, replacements)


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
        _ = replace_node(big_comm, A, replacements)


@pytest.mark.parametrize("value", ["A", 12, (1, 2, 3), {"X", "Y", 1.2}])
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


def test_bilinear_expansion():
    """Tests the substitution of BCH expansions into a commutator tree."""
    X = LeafNode("X")
    Y = LeafNode("Y")
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")

    commutator_to_evaluate = CommutatorNode(X, Y)
    bch_coefficient = 0.5
    max_order = 3

    # Expansion for 'X' is A (order 1)
    # Expansion for 'Y' is B (order 1) + 0.5 * [B, C] (order 2)
    terms_expansions = {
        X: [
            defaultdict(complex, {A: 1.0}),
            defaultdict(complex),
            defaultdict(complex),
        ],
        Y: [
            defaultdict(complex, {B: 1.0}),
            defaultdict(complex, {CommutatorNode(B, C): 0.5}),
            defaultdict(complex),
        ],
    }

    result = bilinear_expansion(
        commutator_node=commutator_to_evaluate,
        terms=terms_expansions,
        max_order=max_order,
        bch_coeff=bch_coefficient,
    )

    # Expected: 0.5 * [A, B] (order 2) + 0.25 * [A, [B, C]] (order 3)
    expected_order_2_term = CommutatorNode(A, B)
    expected_order_3_term = CommutatorNode(A, CommutatorNode(B, C))

    expected_result = [
        defaultdict(complex),  # Order 1
        defaultdict(complex, {expected_order_2_term: 0.5}),  # Order 2
        defaultdict(complex, {expected_order_3_term: 0.25}),  # Order 3
    ]

    assert len(result) == len(expected_result)
    for result_dict, expected_dict in zip(result, expected_result):
        assert result_dict == expected_dict


def test_bilinear_expansion_complex():
    """Tests the substitution with more complex, multi-term BCH expansions."""
    X = LeafNode("X")
    Y = LeafNode("Y")
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    F = LeafNode("F")

    commutator_to_evaluate = CommutatorNode(X, Y)
    bch_coefficient = 1.0
    max_order = 4

    # X = A + 0.2 * [B, C]
    # Y = D + 0.5 * [E, F] + 0.1 * [E, [E, F]]
    terms_expansions = {
        X: [
            defaultdict(complex, {A: 1.0}),
            defaultdict(complex, {CommutatorNode(B, C): 0.2}),
            defaultdict(complex),
            defaultdict(complex),
        ],
        Y: [
            defaultdict(complex, {D: 1.0}),
            defaultdict(complex, {CommutatorNode(E, F): 0.5}),
            defaultdict(complex, {CommutatorNode(E, CommutatorNode(E, F)): 0.1}),
            defaultdict(complex),
        ],
    }

    result = bilinear_expansion(
        commutator_node=commutator_to_evaluate,
        terms=terms_expansions,
        max_order=max_order,
        bch_coeff=bch_coefficient,
    )

    # Order 2: [A, D] (coeff: 1*1*1 = 1.0)
    # Order 3: [A, 0.5[E,F]] + [0.2[B,C], D] (coeffs: 0.5, 0.2)
    # Order 4: [A, 0.1[E,[E,F]]] + [0.2[B,C], 0.5[E,F]] (coeffs: 0.1, 0.1)
    expected_result = [
        defaultdict(complex),  # Order 1
        defaultdict(complex, {CommutatorNode(A, D): 1.0}),  # Order 2
        defaultdict(
            complex,
            {
                CommutatorNode(A, CommutatorNode(E, F)): 0.5,
                CommutatorNode(CommutatorNode(B, C), D): 0.2,
            },
        ),  # Order 3
        defaultdict(
            complex,
            {
                CommutatorNode(A, CommutatorNode(E, CommutatorNode(E, F))): 0.1,
                CommutatorNode(CommutatorNode(B, C), CommutatorNode(E, F)): 0.1,
            },
        ),  # Order 4
    ]

    assert len(result) == len(expected_result)
    for result_dict, expected_dict in zip(result, expected_result):
        assert result_dict.keys() == expected_dict.keys()
        for key in expected_dict:
            assert result_dict[key] == expected_dict[key]


def test_bilinear_expansion_nested_structure():
    """Tests substitution into a more complex, nested commutator structure."""
    X = LeafNode("X")
    Y = LeafNode("Y")
    Z = LeafNode("Z")
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")

    commutator_to_evaluate = CommutatorNode(X, CommutatorNode(Y, Z))
    bch_coefficient = 1.0
    max_order = 5

    # X = A
    # Y = B
    # Z = C + 0.5 * [D, E]
    terms_expansions = {
        X: [defaultdict(complex, {A: 1.0})],
        Y: [defaultdict(complex, {B: 1.0})],
        Z: [defaultdict(complex, {C: 1.0}), defaultdict(complex, {CommutatorNode(D, E): 0.5})],
    }

    result = bilinear_expansion(
        commutator_node=commutator_to_evaluate,
        terms=terms_expansions,
        max_order=max_order,
        bch_coeff=bch_coefficient,
    )

    # Expected: [A, [B, C + 0.5*[D, E]]]
    # = [A, [B, C]] (Order 3) + 0.5 * [A, [B, [D, E]]] (Order 4)
    expected_result = [
        defaultdict(complex),  # Order 1
        defaultdict(complex),  # Order 2
        defaultdict(complex, {CommutatorNode(A, CommutatorNode(B, C)): 1.0}),  # Order 3
        defaultdict(
            complex, {CommutatorNode(A, CommutatorNode(B, CommutatorNode(D, E))): 0.5}
        ),  # Order 4
        defaultdict(complex),  # Order 5
    ]

    assert len(result) == len(expected_result)
    for result_dict, expected_dict in zip(result, expected_result):
        assert result_dict.keys() == expected_dict.keys()
        for key in expected_dict:
            assert result_dict[key] == expected_dict[key]


def test_is_tree_isomorphic():
    """Tests the tree isomorphism check."""
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    F = LeafNode("F")

    # Structure 1: [x, y]
    comm_ab = CommutatorNode(A, B)
    comm_cd = CommutatorNode(C, D)

    # Structure 2: [[x, y], z] (left-nested)
    comm_abc_left = CommutatorNode(CommutatorNode(A, B), C)
    comm_def_left = CommutatorNode(CommutatorNode(D, E), F)

    # Structure 3: [x, [y, z]] (right-nested)
    comm_abc_right = CommutatorNode(A, CommutatorNode(B, C))

    assert is_tree_isomorphic(A, B) is True

    assert is_tree_isomorphic(comm_ab, comm_cd) is True
    assert is_tree_isomorphic(comm_abc_left, comm_def_left) is True

    assert is_tree_isomorphic(comm_ab, comm_abc_left) is False
    assert is_tree_isomorphic(comm_abc_left, comm_abc_right) is False

    assert is_tree_isomorphic(A, comm_ab) is False
    assert is_tree_isomorphic(comm_abc_left, B) is False

    assert is_tree_isomorphic(comm_ab, comm_ab) is True


def test_commutators_partitioner():
    """Tests the CommutatorsPartitioner class for correctly grouping nodes."""
    A = LeafNode("A")
    B = LeafNode("B")
    C = LeafNode("C")
    D = LeafNode("D")
    E = LeafNode("E")
    F = LeafNode("F")

    # Class 1: Leaf nodes
    node_leaf_A = A
    node_leaf_B = B

    # Class 2: Simple commutators [x, y]
    comm_simple_1 = CommutatorNode(A, B)
    comm_simple_2 = CommutatorNode(C, D)

    # Class 3: Left-nested commutators [[x, y], z]
    comm_left_1 = CommutatorNode(CommutatorNode(A, B), C)
    comm_left_2 = CommutatorNode(CommutatorNode(D, E), F)

    # Class 4: Right-nested commutator [x, [y, z]]
    comm_right_1 = CommutatorNode(A, CommutatorNode(B, C))

    nodes_to_add = [
        node_leaf_A,
        comm_simple_1,
        comm_left_1,
        comm_right_1,
        node_leaf_B,  # Should join class 1
        comm_simple_2,  # Should join class 2
        comm_left_2,  # Should join class 3
    ]

    partitioner = CommutatorsPartitioner()
    for node in nodes_to_add:
        partitioner.add_to_partition(node)

    assert len(partitioner.partitions) == 4

    expected_class_1 = {node_leaf_A, node_leaf_B}
    expected_class_2 = {comm_simple_1, comm_simple_2}
    expected_class_3 = {comm_left_1, comm_left_2}
    expected_class_4 = {comm_right_1}

    actual_partition_sets = {frozenset(s) for s in partitioner.partitions}
    expected_partition_sets = {
        frozenset(expected_class_1),
        frozenset(expected_class_2),
        frozenset(expected_class_3),
        frozenset(expected_class_4),
    }

    assert actual_partition_sets == expected_partition_sets
