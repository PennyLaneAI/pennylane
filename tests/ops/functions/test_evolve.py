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
"""Unit tests for the ``evolve`` function."""

import warnings

import pytest

import pennylane as qp
from pennylane.ops import Evolution


def test_error_for_unsupported_input():
    """Test an error is raised for an unsupported input type."""

    with pytest.raises(ValueError, match="No dispatch rule for first argument of type"):
        qp.evolve(0.5)


class TestEvolveConstructor:
    """Unit tests for the evolve function"""

    def test_evolve_doesnt_raise_any_warning(self):
        """Test that using `qp.evolve`, the warning inside `Evolution.__init__` is not raised."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            qp.evolve(qp.PauliX(0))

    def test_evolve_returns_evolution_op(self):
        """Test that the evolve function returns the `Evolution` operator when the input is
        a generic operator."""
        op = qp.s_prod(2, qp.PauliX(0))
        final_op = qp.evolve(op)
        assert isinstance(final_op, Evolution)

    def test_matrix(self):
        """Test that the matrix of the evolved function is correct."""
        final_op = qp.evolve(qp.PauliX(0), coeff=2)
        mat = qp.math.expm(-1j * qp.matrix(2 * qp.PauliX(0)))
        assert qp.math.allequal(qp.matrix(final_op), mat)
