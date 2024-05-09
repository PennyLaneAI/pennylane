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
"""Tests for pennylane/pauli/dla/structure_constants.py functionality"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.pauli import PauliSentence, PauliWord, structure_constants

## Construct some example DLAs
# TFIM
gens = [PauliSentence({PauliWord({i: "X", i + 1: "X"}): 1.0}) for i in range(2)]
gens += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(3)]
Ising3 = qml.pauli.lie_closure(gens, pauli=True)

# XXZ-type DLA, i.e. with true PauliSentences
gens2 = [
    PauliSentence(
        {
            PauliWord({i: "X", i + 1: "X"}): 1.0,
            PauliWord({i: "Y", i + 1: "Y"}): 1.0,
        }
    )
    for i in range(2)
]
gens2 += [PauliSentence({PauliWord({i: "Z"}): 1.0}) for i in range(3)]
XXZ3 = qml.pauli.lie_closure(gens2, pauli=True)


class TestAdjointRepr:
    """Tests for structure_constants"""

    def test_structure_constants_dim(self):
        """Test the dimension of the adjoint repr"""
        d = len(Ising3)
        adjoint = structure_constants(Ising3, pauli=True)
        assert adjoint.shape == (d, d, d)
        assert adjoint.dtype == float

    @pytest.mark.parametrize("dla", [Ising3, XXZ3])
    def test_structure_constants_elements(self, dla):
        r"""Test relation :math:`[i G_\alpha, i G_\beta] = \sum_{\gamma = 0}^{\mathfrak{d}-1} f^\gamma_{\alpha, \beta} iG_\gamma`."""

        d = len(dla)
        ad_rep = structure_constants(dla, pauli=True)
        for i in range(d):
            for j in range(d):

                comm_res = 1j * dla[i].commutator(dla[j])

                res = sum(
                    np.array(c, dtype=complex) * dla[gamma]
                    for gamma, c in enumerate(ad_rep[:, i, j])
                )
                res.simplify()
                assert comm_res == res

    @pytest.mark.parametrize("dla", [Ising3, XXZ3])
    def test_use_operators(self, dla):
        """Test that operators can be passed and lead to the same result"""
        ad_rep_true = structure_constants(dla, pauli=True)

        ops = [op.operation() for op in dla]
        ad_rep = structure_constants(ops, pauli=False)
        assert qml.math.allclose(ad_rep, ad_rep_true)

    def test_raise_error_for_non_paulis(self):
        """Test that an error is raised when passing operators that do not have a pauli_rep"""
        generators = [qml.Hadamard(0), qml.X(0)]
        with pytest.raises(
            ValueError, match="Cannot compute adjoint representation of non-pauli operators"
        ):
            qml.pauli.structure_constants(generators)
