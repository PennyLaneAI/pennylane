# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the evolve function
"""
from typing import Callable

import pennylane as qml
from pennylane.ops import Evolution, ParametrizedEvolution, ParametrizedHamiltonian


class TestEvolve:
    """Unit tests for the evolve function"""

    def test_evolve_returns_evolution_op(self):
        """Test that the evolve function returns the `Evolution` operator when the input is
        a generic operator."""
        op = qml.s_prod(2, qml.PauliX(0))
        final_op = qml.evolve(op)
        assert isinstance(final_op, Evolution)

    def test_matrix(self):
        """Test that the matrix of the evolved function is correct."""
        op = qml.s_prod(2, qml.PauliX(0))
        final_op = qml.evolve(op)
        mat = qml.math.expm(1j * qml.matrix(op))
        assert qml.math.allequal(qml.matrix(final_op), mat)

    def test_evolve_returns_callable(self):
        """Test that the evolve function returns a callable when the input is a
        ParametrizedHamiltonian."""
        coeffs = [1, 2, 3]
        ops = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]
        H = ParametrizedHamiltonian(coeffs=coeffs, observables=ops)
        final_op = qml.evolve(H)
        assert isinstance(  # pylint: disable=isinstance-second-argument-not-valid-type
            final_op, Callable
        )
        param_evolution = final_op(params=[], t=1)
        assert isinstance(param_evolution, ParametrizedEvolution)
        assert param_evolution.H is H
