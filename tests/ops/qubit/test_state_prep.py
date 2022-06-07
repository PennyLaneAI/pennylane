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
Unit tests for the available qubit state preparation operations.
"""
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.wires import Wires


densitymat0 = np.array([[1.0, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize(
    "op",
    [
        qml.BasisState(np.array([0, 1]), wires=0),
        qml.QubitStateVector(np.array([1.0, 0.0]), wires=0),
        qml.QubitDensityMatrix(densitymat0, wires=0),
    ],
)
def test_adjoint_error_exception(op):
    with pytest.raises(qml.operation.AdjointUndefinedError):
        op.adjoint()


@pytest.mark.parametrize(
    "op, mat, base",
    [
        (qml.BasisState(np.array([0, 1]), wires=0), [0, 1], "BasisState"),
        (qml.QubitStateVector(np.array([1.0, 0.0]), wires=0), [1.0, 0.0], "QubitStateVector"),
        (qml.QubitDensityMatrix(densitymat0, wires=0), densitymat0, "QubitDensityMatrix"),
    ],
)
def test_labelling_matrix_cache(op, mat, base):
    """Test state prep matrix parameters interact with labelling matrix cache"""

    assert op.label() == base

    cache = {"matrices": []}
    assert op.label(cache=cache) == base + "(M0)"
    assert qml.math.allclose(cache["matrices"][0], mat)

    cache = {"matrices": [0, mat, 0]}
    assert op.label(cache=cache) == base + "(M1)"
    assert len(cache["matrices"]) == 3


class TestDecomposition:
    def test_BasisState_decomposition(self):
        """Test the decomposition for BasisState"""

        n = np.array([0, 1, 0])
        wires = (0, 1, 2)
        ops1 = qml.BasisState.compute_decomposition(n, wires)
        ops2 = qml.BasisState(n, wires=wires).decomposition()

        assert len(ops1) == len(ops2) == 1
        assert isinstance(ops1[0], qml.BasisStatePreparation)
        assert isinstance(ops2[0], qml.BasisStatePreparation)

    def test_QubitStateVector_decomposition(self):
        """Test the decomposition for QubitStateVector."""

        U = np.array([1, 0, 0, 0])
        wires = (0, 1)

        ops1 = qml.QubitStateVector.compute_decomposition(U, wires)
        ops2 = qml.QubitStateVector(U, wires=wires).decomposition()

        assert len(ops1) == len(ops2) == 1
        assert isinstance(ops1[0], qml.MottonenStatePreparation)
        assert isinstance(ops2[0], qml.MottonenStatePreparation)
