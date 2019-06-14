# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.output`
"""
# pylint: disable=protected-access,cell-var-from-loop
import pennylane as qml
from pennylane.ops import PauliX, Hadamard

import numpy as np


dev = qml.device('default.qubit', wires=2)


def test_expval(tol):
    """expval: Tests the `expval` function.
    """
    def circuit():
        qml.PauliZ(wires=0)
        return qml.expval(Hadamard(wires=[0]))
    
    qcircuit = qml.QNode(circuit, dev)

    assert np.allclose(qcircuit(), 1/np.sqrt(2), atol=tol, rtol=0)
