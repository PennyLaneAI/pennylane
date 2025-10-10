from collections import defaultdict

import pytest

from pennylane.labs.trotter_error.product_formulas.commutator import (
    CommutatorNode,
    SymbolNode,
    bilinear_expansion,
    is_mergeable,
    is_tree_isomorphic,
    merge,
    replace_node,
)


def test_replacement():
    A = SymbolNode((1, 2), (2, 3))
    B = SymbolNode("B", 1)
    C = SymbolNode("C", 1)
    D = SymbolNode("D", 1)
    E = SymbolNode("E", 1)
    X = SymbolNode("X", 1)
    Y = SymbolNode("Y", 1)
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y), E]
    replaced_comm = replace_node(big_comm, A, replacements)

    assert replaced_comm == CommutatorNode(
        CommutatorNode(CommutatorNode(X, Y), CommutatorNode(B, C)),
        CommutatorNode(CommutatorNode(E, D), E),
    )

    assert big_comm == CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [E, CommutatorNode(X, Y)]
    replaced_comm = replace_node(big_comm, A, replacements)

    assert replaced_comm == CommutatorNode(
        CommutatorNode(E, CommutatorNode(B, C)),
        CommutatorNode(CommutatorNode(CommutatorNode(X, Y), D), E),
    )

    assert big_comm == CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )


def test_replacement_too_many():
    A = SymbolNode("A", 1)
    B = SymbolNode("B", 1)
    C = SymbolNode("C", 1)
    D = SymbolNode("D", 1)
    E = SymbolNode("E", 1)
    X = SymbolNode("X", 1)
    Y = SymbolNode("Y", 1)
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y), E, X]
    with pytest.raises(RuntimeError, match="Got more replacement nodes"):
        _ = replace_node(big_comm, A, replacements)


def test_replacement_too_few():
    A = SymbolNode("A", 1)
    B = SymbolNode("B", 1)
    C = SymbolNode("C", 1)
    D = SymbolNode("D", 1)
    E = SymbolNode("E", 1)
    X = SymbolNode("X", 1)
    Y = SymbolNode("Y", 1)
    big_comm = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), E)
    )

    replacements = [CommutatorNode(X, Y)]
    with pytest.raises(RuntimeError, match="Got fewer replacement nodes"):
        _ = replace_node(big_comm, A, replacements)


def test_mergeability():
    A = SymbolNode("A", 1)
    B = SymbolNode("B", 1)
    C = SymbolNode("C", 1)
    D = SymbolNode("D", 1)
    comm1 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, B), D)
    )

    comm2 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), D)
    )

    comm3 = CommutatorNode(
        CommutatorNode(CommutatorNode(A, B), C), CommutatorNode(CommutatorNode(A, D), D)
    )

    assert is_mergeable(comm1, comm2, 0) is False
    assert is_mergeable(comm1, comm2, 4) is True
    assert is_mergeable(comm1, comm3, 0) is False


@pytest.mark.parametrize(
    "B, D, B_plus_D",
    [
        (SymbolNode("B"), SymbolNode("D"), SymbolNode(("B", "D"), (1, 1))),
        (
            SymbolNode(("X", "Y"), (1, 1)),
            SymbolNode(("X", "Y", "Z"), (1, 1, 1)),
            SymbolNode(("X", "Y", "Z"), (2, 2, 1)),
        ),
    ],
)
def test_merge(B, D, B_plus_D):
    A = SymbolNode("A")
    C = SymbolNode("C")
    comm1 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, B), D)
    )

    comm2 = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, D), D)
    )
    sum_comm = merge(comm1, comm2, 4)

    expected = CommutatorNode(
        CommutatorNode(A, CommutatorNode(B, C)), CommutatorNode(CommutatorNode(A, B_plus_D), D)
    )
    assert sum_comm == expected


def test_expand_commutator():
    A = SymbolNode("A")
    B = SymbolNode("B")
    C = SymbolNode("C")

    expected_A = {(SymbolNode("A"),): 1}
    assert A.expand() == expected_A

    comm_AB = CommutatorNode(A, B)
    expected_AB = {(SymbolNode("A"), SymbolNode("B")): 1, (SymbolNode("B"), SymbolNode("A")): -1}
    assert comm_AB.expand() == expected_AB

    comm_ABC = CommutatorNode(CommutatorNode(A, B), C)
    expected_ABC = {
        (SymbolNode("A"), SymbolNode("B"), SymbolNode("C")): 1,
        (SymbolNode("B"), SymbolNode("A"), SymbolNode("C")): -1,
        (SymbolNode("C"), SymbolNode("A"), SymbolNode("B")): -1,
        (SymbolNode("C"), SymbolNode("B"), SymbolNode("A")): 1,
    }
    assert comm_ABC.expand() == expected_ABC


def test_bilinear_expansion():
    """Tests the substitution of BCH expansions into a commutator tree."""
    X = SymbolNode("X")
    Y = SymbolNode("Y")
    A = SymbolNode("A")
    B = SymbolNode("B")
    C = SymbolNode("C")

    commutator_to_evaluate = CommutatorNode(X, Y)
    bch_coefficient = 0.5
    max_order = 3

    # Expansion for 'X' is A (order 1)
    # Expansion for 'Y' is B (order 1) + 0.5 * [B, C] (order 2)
    terms_expansions = {
        X: {A: 1.0},
        Y: {B: 1.0},
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

    expected_result = {
        expected_order_2_term: 0.5,
        expected_order_3_term: 0.25,
    }

    for result_dict, expected_dict in zip(result, expected_result):
        assert result_dict == expected_dict


def test_bilinear_expansion_complex():
    """Tests the substitution with more complex, multi-term BCH expansions."""
    X = SymbolNode("X")
    Y = SymbolNode("Y")
    A = SymbolNode("A")
    B = SymbolNode("B")
    C = SymbolNode("C")
    D = SymbolNode("D")
    E = SymbolNode("E")
    F = SymbolNode("F")

    commutator_to_evaluate = CommutatorNode(X, Y)
    bch_coefficient = 1.0
    max_order = 4

    # X = A + 0.2 * [B, C]
    # Y = D + 0.5 * [E, F] + 0.1 * [E, [E, F]]
    terms_expansions = {
        X: {A: 1.0, CommutatorNode(B, C): 0.2},
        Y: {D: 1.0, CommutatorNode(E, F): 0.5, CommutatorNode(E, CommutatorNode(E, F)): 0.1},
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
    expected_result = {
        CommutatorNode(A, D): 1.0,
        CommutatorNode(A, CommutatorNode(E, F)): 0.5,
        CommutatorNode(CommutatorNode(B, C), D): 0.2,
        CommutatorNode(A, CommutatorNode(E, CommutatorNode(E, F))): 0.1,
        CommutatorNode(CommutatorNode(B, C), CommutatorNode(E, F)): 0.1,
    }

    assert result.keys() == expected_result.keys()
    for key in expected_result:
        assert result[key] == expected_result[key]


def test_bilinear_expansion_nested_structure():
    """Tests substitution into a more complex, nested commutator structure."""
    X = SymbolNode("X")
    Y = SymbolNode("Y")
    Z = SymbolNode("Z")
    A = SymbolNode("A")
    B = SymbolNode("B")
    C = SymbolNode("C")
    D = SymbolNode("D")
    E = SymbolNode("E")

    commutator_to_evaluate = CommutatorNode(X, CommutatorNode(Y, Z))
    bch_coefficient = 1.0
    max_order = 5

    # X = A
    # Y = B
    # Z = C + 0.5 * [D, E]
    terms_expansions = {
        X: {A: 1.0},
        Y: {B: 1.0},
        Z: {C: 1.0, CommutatorNode(D, E): 0.5},
    }

    result = bilinear_expansion(
        commutator_node=commutator_to_evaluate,
        terms=terms_expansions,
        max_order=max_order,
        bch_coeff=bch_coefficient,
    )

    # Expected: [A, [B, C + 0.5*[D, E]]]
    # = [A, [B, C]] (Order 3) + 0.5 * [A, [B, [D, E]]] (Order 4)
    expected_result = {
        CommutatorNode(A, CommutatorNode(B, C)): 1.0,
        CommutatorNode(A, CommutatorNode(B, CommutatorNode(D, E))): 0.5,
    }

    assert result.keys() == expected_result.keys()
    for key in expected_result:
        assert result[key] == expected_result[key]


def test_is_tree_isomorphic():
    """Tests the tree isomorphism check."""
    A = SymbolNode("A")
    B = SymbolNode("B")
    C = SymbolNode("C")
    D = SymbolNode("D")
    E = SymbolNode("E")
    F = SymbolNode("F")

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
