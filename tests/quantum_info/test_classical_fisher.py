# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Tests for the classical fisher information matrix in the pennylane.qinfo
"""
from xml.dom.minidom import Element
import pytest

import pennylane as qml
import numpy as np
import pennylane.numpy as pnp

from pennylane.quantum_info import _compute_cfim


@pytest.mark.parametrize("n_params", np.arange(1, 10))
@pytest.mark.parametrize("n_wires", np.arange(1, 5))
def test_construction_of_compute_cfim(n_params, n_wires):
    """Ensuring the construction in _compute_cfim is correct"""
    dp = np.arange(2**n_wires * n_params, dtype=float).reshape(2**n_wires, n_params)
    p = np.ones(2**n_wires)

    res = _compute_cfim(p, dp, None)

    assert np.allclose(res, res.T)
    assert all(
        [
            res[i, j] == np.sum(dp[:, i] * dp[:, j] / p)
            for i in range(n_params)
            for j in range(n_params)
        ]
    )


@pytest.mark.parametrize("n_params", np.arange(1, 10))
@pytest.mark.parametrize("n_wires", np.arange(1, 5))
def test_compute_cfim_trivial_distribution(n_params, n_wires):
    """Test that the classical fisher information matrix (CFIM) is
    constructed correctly for the trivial distirbution p=(1,0,0,...)
    and some dummy gradient dp"""
    # implicitly also does the unit test for the correct output shape

    dp = np.ones(2**n_wires * n_params, dtype=float).reshape(2**n_wires, n_params)
    p = np.zeros(2**n_wires)
    p[0] = 1
    res = _compute_cfim(p, dp, None)
    assert np.allclose(res, np.ones((n_params, n_params)))
