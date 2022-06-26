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
Tests for the FlipSign template.
"""
import pytest
from pennylane import numpy as np
import pennylane as qml


class TestFlipSign:
    """Tests that the template defines the correct sign flip."""

    @pytest.mark.parametrize(
        ("n_status, n_wires"),
        [
            (2, 2),
            (6, 3),
            (8, 4),
            ([1, 0], 2),
            ([1, 1, 0], 3),
            ([1, 0, 0, 0], 4),
        ],
    )
    def test_eval(self, n_status, n_wires):

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            for wire in list(range(n_wires)):
                qml.Hadamard(wires=wire)

            qml.FlipSign(n_status, wires=list(range(n_wires)))

            return qml.state()

        def to_number(status):
            return sum([status[i] * 2 ** (len(status) - i - 1) for i in range(len(status))])

        if isinstance(n_status, list):
            n_status = to_number(n_status)

        # we check that only the indicated value has been changed
        statuses = []
        for ind, x in enumerate(circuit()):
            if ind == n_status:
                statuses.append(bool(np.sign(x) == -1))
            else:
                statuses.append(bool(np.sign(x) == 1))

        assert np.all(np.array(statuses))
