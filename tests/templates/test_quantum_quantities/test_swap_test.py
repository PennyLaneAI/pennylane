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
r"""
Contains the unit tests for the SwapTest template.
"""
import pytest
import numpy as np
import pennylane as qml


class TestSwapTest:
    """Testing that the result of the swap test matches theory"""
    np.random.seed(1)  # set random seed

    def test_identical_states(self):
        dev = qml.device("default.qubit", wires=5, shots=100)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=1)
            qml.PauliX(wires=3)
            return qml.SwapTest(0, [1, 2], [3, 4])()

        expected_res = 1.0
        swap_res = circuit()
        assert np.isclose(expected_res, swap_res, )

    def test_orthogonal_states(self):
        dev = qml.device("default.qubit", wires=5, shots=200)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=1)
            qml.PauliX(wires=2)
            return qml.SwapTest(0, [1, 2], [3, 4])()

        expected_res = 0.0
        swap_res = circuit()
        assert np.isclose(expected_res, swap_res, atol=1e-02)
