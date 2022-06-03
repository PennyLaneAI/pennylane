# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for functions needed for computing the Hamiltonian.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np


@pytest.mark.parametrize(
    ("factors", "eigvals", "rank_r_ref", "rank_m_ref"),
    [
        (
            np.array(
                [
                    [[1.06723440e-01, 6.31328996e-15], [4.82191426e-15, -1.04898533e-01]],
                    [[-1.37560572e-13, -4.25688222e-01], [-4.25688222e-01, -1.85111072e-13]],
                    [[-8.14472856e-01, 3.11914551e-13], [3.11687682e-13, -8.28642140e-01]],
                ]
            ),
            np.array(
                [[-0.10489853, 0.10672344], [-0.42568822, 0.42568822], [-0.82864214, -0.81447286]]
            ),
            3,
            2,
        ),
    ],
)
def test_rank(factors, eigvals, rank_r_ref, rank_m_ref):
    r"""Test that rank function returns the correct ranks."""
    rank_r, rank_m = qml.resources.rank(factors, eigvals)

    assert rank_r == rank_r_ref
    assert rank_m == rank_m_ref
