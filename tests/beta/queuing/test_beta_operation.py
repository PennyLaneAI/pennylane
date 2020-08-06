# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for :mod:`pennylane.operation`.
"""
import itertools
import functools
from unittest.mock import patch

import pytest
import numpy as np
from numpy.linalg import multi_dot

import pennylane as qml
import pennylane._queuing

from gate_data import I, X, Y, Rotx, Roty, Rotz, CRotx, CRoty, CRotz, CNOT, Rot3, Rphi
from pennylane.wires import Wires

# --------------------
# Beta related imports
# --------------------

# Fixture for importing beta files
from conftest import import_beta

# The BetaTensor class
from pennylane.beta.queuing.operation import BetaTensor

# pylint: disable=no-self-use, no-member, protected-access, pointless-statement

class TestBetaTensor:
    """Unit tests for the BetaTensor class"""

    def test_append_annotating_object(self):
        """Test appending an object that writes annotations when queuing itself"""

        with qml._queuing.AnnotatedQueue() as q:
            A = qml.PauliZ(0)
            B = qml.PauliY(1)
            tensor_op = BetaTensor(A, B)

        assert q.queue == [A, B, tensor_op]
        assert q.get_info(A) == {"owner": tensor_op}
        assert q.get_info(B) == {"owner": tensor_op}
        assert q.get_info(tensor_op) == {"owns": (A, B)}
