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
Unit tests for the :mod:`pennylane.template.subroutines` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from scipy.stats import unitary_group
import pennylane as qml
from pennylane import numpy as np
from pennylane.wires import Wires

from pennylane.templates.subroutines import (
    Interferometer,
    ArbitraryUnitary,
    SingleExcitationUnitary,
    DoubleExcitationUnitary,
    UCCSD,
    Permute,
    QuantumPhaseEstimation,
)



class TestUCCSDUnitary:
    """Tests for the UCCSD template from the pennylane.templates.subroutine module."""




    @pytest.mark.parametrize(
        ("weights", "s_wires", "d_wires", "expected"),
        [
            (
                np.array([3.90575761, -1.89772083, -1.36689032]),
                [[0, 1, 2], [1, 2, 3]],
                [[[0, 1], [2, 3]]],
                [-0.14619406, -0.06502792, 0.14619406, 0.06502792],
            )
        ],
    )
    def test_integration(self, weights, s_wires, d_wires, expected, tol):
        """Test integration with PennyLane and gradient calculations"""

        N = 4
        wires = range(N)
        dev = qml.device("default.qubit", wires=N)

        w0 = weights[0]
        w1 = weights[1]
        w2 = weights[2]

        @qml.qnode(dev)
        def circuit(w0, w1, w2):
            UCCSD(
                [w0, w1, w2],
                wires,
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=np.array([1, 1, 0, 0]),
            )
            return [qml.expval(qml.PauliZ(w)) for w in range(N)]

        res = circuit(w0, w1, w2)
        assert np.allclose(res, np.array(expected), atol=tol)


