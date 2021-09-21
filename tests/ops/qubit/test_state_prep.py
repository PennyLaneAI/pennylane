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
import itertools
import re
import pytest
import functools
import copy
import numpy as np
from numpy.linalg import multi_dot
from scipy.stats import unitary_group
from scipy.linalg import expm
from pennylane import numpy as npp

import pennylane as qml
from pennylane.wires import Wires

from gate_data import (
    I,
    X,
    Y,
    Z,
    H,
    StateZeroProjector,
    StateOneProjector,
    CNOT,
    SWAP,
    ISWAP,
    SISWAP,
    CZ,
    S,
    T,
    CSWAP,
    Toffoli,
    QFT,
    ControlledPhaseShift,
    SingleExcitation,
    SingleExcitationPlus,
    SingleExcitationMinus,
    DoubleExcitation,
    DoubleExcitationPlus,
    DoubleExcitationMinus,
)


class TestOperations:
    @pytest.mark.parametrize(
        "op",
        [
            qml.BasisState(np.array([0, 1]), wires=0),
            qml.QubitStateVector(np.array([1.0, 0.0]), wires=0),
        ],
    )
    def test_adjoint_error_exception(self, op, tol):
        with pytest.raises(qml.ops.AdjointError):
            op.adjoint()
