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
Unit tests for the :mod:`pennylane.template.state_preparations` module.
Integration tests should be placed into ``test_templates.py``.
"""
# pylint: disable=protected-access,cell-var-from-loop

import math
from unittest.mock import patch
from pennylane import numpy as np
import pytest
import pennylane as qml
from pennylane.templates.state_preparations import (
    BasisStatePreparation,
    MottonenStatePreparation,
    ArbitraryStatePreparation,
)
from pennylane.templates.state_preparations.mottonen import gray_code
from pennylane.templates.state_preparations.arbitrary_state_preparation import (
    _state_preparation_pauli_words,
)
from pennylane.templates.state_preparations.mottonen import _get_alpha_y
from pennylane.wires import Wires


class TestMottonenStatePreparation:
    """Tests the template MottonenStatePreparation."""





    @pytest.mark.parametrize(
        "state_vector", [np.array([0.70710678, 0.70710678]), np.array([0.70710678, 0.70710678j])]
    )
    def test_gradient_evaluated(self, state_vector):
        """Test that the gradient is successfully calculated for a simple example. This test only
        checks that the gradient is calculated without an error."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(state_vector):
            MottonenStatePreparation(state_vector, wires=range(1))
            return qml.expval(qml.PauliZ(0))

        qml.grad(circuit)(state_vector)

