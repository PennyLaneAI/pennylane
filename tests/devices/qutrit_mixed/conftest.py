# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pytest configuration file for PennyLane qutrit mixed state test suite."""
import numpy as np
import pytest
from scipy.stats import unitary_group


def get_random_mixed_state(num_qutrits):
    dim = 3**num_qutrits

    rng = np.random.default_rng(seed=4774)
    basis = unitary_group(dim=dim, seed=584545).rvs()
    schmidt_weights = rng.dirichlet(np.ones(dim), size=1).astype(complex)[0]
    mixed_state = np.zeros((dim, dim)).astype(complex)
    for i in range(dim):
        mixed_state += schmidt_weights[i] * np.outer(np.conj(basis[i]), basis[i])

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
