import pytest

from itertools import product

import numpy as np
from scipy.linalg import expm

from pennylane.labs.trotter_error import ProductFormula
from pennylane.labs.trotter_error.abstract import nested_commutator

fragments_list = [
    {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3)), 2: np.zeros(shape=(3, 3))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3)), 2: np.random.random(size=(3, 3))}
]

@pytest.mark.parametrize("fragments", fragments_list)
def test_second_order(fragments):
    """Test against the second order Trotter error formula"""

    n_frags = len(fragments)
    frag_labels = list(range(n_frags)) + list(range(n_frags -1, -1, -1))
    coeffs = [1/2]*len(frag_labels)

    pf = ProductFormula(coeffs, frag_labels)
    ast = pf.bch_ast(max_order=3)
    actual = ast.evaluate(fragments)

    expected = np.zeros(shape=(3, 3))

    for i in range(n_frags - 1):
        for j in range(i + 1, n_frags):
            expected += nested_commutator([fragments[i], fragments[i], fragments[j]])
            for k in range(i + 1, n_frags):
                expected += 2 * nested_commutator([fragments[k], fragments[i], fragments[j]])

    expected = -(1/24)*expected

    for fragment in fragments.values():
        expected += fragment

    assert np.allclose(actual, expected)

fragments_list = [
    {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
]

@pytest.mark.parametrize("fragments", fragments_list)
def test_second_order_against_log(fragments):
    """Test against the matrix logarithm of the product formula"""

    t = 0.1

    n_frags = len(fragments)
    frag_labels = list(range(n_frags)) + list(range(n_frags -1, -1, -1))
    coeffs = [1/2]*len(frag_labels)

    pf = ProductFormula(coeffs, frag_labels)
    ast = pf.bch_ast(max_order=3)
    actual = t * ast.evaluate(fragments)

    pf = expm(0.5j * t * fragments[0]) @ expm(1j * t * fragments[1]) @ expm(0.5j * t * fragments[0])
    evals, evecs = np.linalg.eig(pf)
    theta = np.angle(evals)
    expected = evecs @ np.diag(theta) @ np.linalg.inv(evecs)

    print(actual)
    print(np.real(expected))

    assert np.allclose(actual, np.real(expected))

fragments_list = [
    {0: np.zeros(shape=(3, 3)), 1: np.zeros(shape=(3, 3))},
    {0: np.random.random(size=(3, 3)), 1: np.random.random(size=(3, 3))},
]

@pytest.mark.parametrize("fragments", fragments_list)
def test_fourth_order_two_fragments(fragments):
    coeffs = {
        (0, 0, 0, 1, 0): 0.0047,
        (0, 0, 1, 1, 0): 0.0057,
        (0, 1, 0, 1, 0): 0.0046,
        (0, 1, 1, 1, 0): 0.0074,
        (1, 0, 0, 1, 0): 0.0097,
        (1, 0, 1, 1, 0): 0.0097,
        (1, 1, 0, 1, 0): 0.0173,
        (1, 1, 1, 1, 0): 0.0284,
    }

    #expected = fragments[0] + fragments[1]
    expected = np.zeros(shape=(3, 3))

    for index in product((0, 1), repeat=5):
        expected += coeffs.get(index, 0) * nested_commutator([fragments[i] for i in index])

    u = 1 / (4 - 4**(1/3))
    frag_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    frag_coeff = [u/2, u, u, u, (1-3*u)/2, 1 - 4*u, (1-3*u)/2, u, u, u, u/2]

    pf = ProductFormula(frag_coeff, frag_labels)
    ast = pf.bch_ast(max_order=5)
    actual = ast.evaluate(fragments) - fragments[0] - fragments[1]

    assert np.allclose(actual, expected)

@pytest.mark.parametrize("fragments", fragments_list)
def test_fourth_order_against_log(fragments):

    t = 0.1
    u = 1 / (4 - 4**(1/3))
    frag_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    frag_coeff = [u/2, u, u, u, (1-3*u)/2, 1 - 4*u, (1-3*u)/2, u, u, u, u/2]

    pf = ProductFormula(frag_coeff, frag_labels)
    ast = pf.bch_ast(max_order=5)
    actual = t * ast.evaluate(fragments)

    pf = expm(0.5j * t * fragments[0]) @ expm(1j * t * fragments[1]) @ expm(0.5j * t * fragments[0])
    evals, evecs = np.linalg.eig(pf)
    theta = np.angle(evals)
    expected = np.real(evecs @ np.diag(theta) @ np.linalg.inv(evecs))

    print(actual)
    print(expected)

    assert np.allclose(actual, expected)
