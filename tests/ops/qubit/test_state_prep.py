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

densitymat0 = np.array([[1.0, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize(
    "op",
    [
        qml.BasisState(np.array([0, 1]), wires=0),
        qml.QubitStateVector(np.array([1.0, 0.0]), wires=0),
        qml.QubitDensityMatrix(denistymat0, wires=0),
    ],
)
def test_adjoint_error_exception(op):
    with pytest.raises(qml.ops.AdjointError):
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
