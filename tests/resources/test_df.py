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
Unit tests for functions needed for resource estimation with double factorization method.
"""
import pytest
import pennylane as qml
from pennylane import numpy as np


@pytest.mark.parametrize(
    ("one", "two", "eigvals", "lamb_ref"),
    [
        (
            np.array([[-1.25330961e00, 4.01900735e-14], [4.01900735e-14, -4.75069041e-01]]),
            # two-electron integral is arranged in the chemist notation [11|22]
            np.array(
                [
                    [
                        [[6.74755872e-01, -4.60742555e-14], [-4.60742555e-14, 6.63711349e-01]],
                        [[-4.61020111e-14, 1.81210478e-01], [1.81210478e-01, -4.26325641e-14]],
                    ],
                    [
                        [[-4.60464999e-14, 1.81210478e-01], [1.81210478e-01, -4.25215418e-14]],
                        [[6.63711349e-01, -4.28546088e-14], [-4.24105195e-14, 6.97651447e-01]],
                    ],
                ]
            ),
            np.tensor(
                [[-0.10489852, 0.10672343], [-0.42568824, 0.42568824], [-0.82864211, -0.81447282]]
            ),
            1.6570518796336895,  # lambda value obtained from openfermion
        )
    ],
)
def test_df_norm(one, two, eigvals, lamb_ref):
    r"""Test that the norm function returns the correct 1-norm."""
    lamb = qml.resources.norm(one, two, eigvals)

    assert np.allclose(lamb, lamb_ref)
