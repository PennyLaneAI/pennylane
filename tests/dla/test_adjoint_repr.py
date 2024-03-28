# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/dla/adjoint_repr.py functionality"""
import pytest
import numpy as np

import pennylane as qml

from pennylane.dla import adjoint_repr
from pennylane.pauli import PauliWord, PauliSentence

n = 3
gens = [PauliSentence({PauliWord({i: "X", i + 1: "X"}): 1.0}) for i in range(n - 1)]
gens += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(n)]
Ising3 = qml.dla.lie_closure(gens)


class TestAdjointRepr:
    """Tests for adjoint_repr"""

    def test_adjoint_repr_dim(self):
        """Test the dimension of the adjoint repr"""
        d = len(Ising3)
        adjoint = adjoint_repr(Ising3)
        assert adjoint.shape == (d, d, d)
        assert adjoint.dtype == float

    @pytest.mark.parametrize("dla", [Ising3])
    def test_adjoint_repr_elements(self, dla):
        r"""Test relation :math:`[i G_\alpha, i G_\beta] = \sum_{\gamma = 0}^{\mathfrak{d}-1} f^\gamma_{\alpha, \beta} iG_\gamma`."""
        d = len(dla)
        ad_rep = adjoint_repr(dla)
        for i in range(d):
            for j in range(d):
                comm_res = -1j * dla[i].commutator(dla[j]) / 2
                res = sum(
                    np.array(c, dtype=complex) * dla[gamma]
                    for gamma, c in enumerate(ad_rep[:, i, j])
                )
                res.simplify()
                assert comm_res == res

    @pytest.mark.parametrize("dla", [Ising3])
    def test_use_operators(self, dla):
        """Test that operators can be passed and lead to the same result"""
        ad_rep_true = adjoint_repr(dla)

        ops = [op.operation() for op in dla]
        ad_rep = adjoint_repr(ops)
        assert qml.math.allclose(ad_rep, ad_rep_true)

    def test_raise_error_for_non_paulis(self):
        """Test that an error is raised when passing operators that do not have a pauli_rep"""
        generators = [qml.Hadamard(0), qml.X(0)]
        with pytest.raises(
            ValueError, match="Cannot compute adjoint representation of non-pauli operators"
        ):
            qml.dla.adjoint_repr(generators)
