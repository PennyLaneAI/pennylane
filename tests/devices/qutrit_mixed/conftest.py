import numpy as np
from scipy.stats import unitary_group
import pytest

SEED = 4774
def get_random_mixed_state(num_qutrits):
    dim = 3**num_qutrits

    np.random.seed(seed=SEED)
    basis = unitary_group.rvs(dim)
    Schmidt_weights = np.random.dirichlet(np.ones(dim), size=1).astype(complex)[0]
    mixed_state = np.zeros((dim, dim)).astype(complex)
    for i in range(dim):
        mixed_state += Schmidt_weights[i] * np.outer(np.conj(basis[i]), basis[i])

    return mixed_state.reshape([3] * (2 * num_qutrits))


# 1 qutrit states
@pytest.fixture(scope="package")
def one_qutrit_state():
    return get_random_mixed_state(1)


@pytest.fixture(scope="package")
def one_qutrit_batched_state():
    return np.array([get_random_mixed_state(1) for _ in range(2)])


# 2 qutrit states
@pytest.fixture(scope="package")
def two_qutrit_state():
    return get_random_mixed_state(2)


@pytest.fixture(scope="package")
def two_qutrit_batched_state():
    return np.array([get_random_mixed_state(2) for _ in range(2)])


# 3 qutrit states
@pytest.fixture(scope="package")
def three_qutrit_state():
    return get_random_mixed_state(3)


@pytest.fixture(scope="package")
def three_qutrit_batched_state():
    return np.array([get_random_mixed_state(3) for _ in range(2)])