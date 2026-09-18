# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the qubitized tensor hypercontraction walk operator ``QubitizationTHC``."""

import numpy as np

import pennylane as qp
from pennylane.ops.functions.assert_valid import assert_valid


class TestQubitizationTHC:
    """Test the QubitizationTHC class."""

    # pylint: disable=too-few-public-methods

    def test_standard_validity(self, seed):
        """Test standard validity of the QubitizationTHC operator with assert_valid."""

        M, N, aleph, beth = 6, 2, 2, 2
        rng = np.random.default_rng(seed)
        zeta = rng.standard_normal((M, M))
        zeta = tuple(map(tuple, (zeta + zeta.T) / 2))
        chi = tuple(map(tuple, rng.standard_normal((M, N // 2))))
        t_ell = tuple(rng.standard_normal(N // 2))
        t_eigenvectors = tuple(map(tuple, np.eye(N // 2)))

        sizes = qp.qubitization_thc_wires(M, N, aleph, beth)
        wires = qp.registers(sizes)

        op = qp.QubitizationTHC(zeta, t_ell, chi, t_eigenvectors, aleph, beth, **wires)
        assert_valid(op)
