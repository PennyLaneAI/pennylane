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
Unit tests for the :mod:`pennylane.visualize` submodule.
"""
import pytest
import pennylane as qml
import numpy as np
from pennylane.visualize import *

"""A variational quantum circuit for the tests"""

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def circuit(gamma):
    qml.RX(gamma, wires=0)
    qml.RY(gamma, wires=1)
    qml.RX(gamma, wires=2)
    qml.CNOT(wires=[0, 2])
    qml.RY(gamma, wires=1)

    return [qml.expval(qml.PauliZ(i)) for i in range(3)]


def cost(gamma):
    return sum(circuit(gamma[0]))


class TestVisualize:
    """Tests the Visualize class"""

    def test_init_errors(self):
        """Tests that the correct errors are thrown in the initialization of Visualize"""

        with pytest.raises(ValueError, match="'step_size' must be of type int"):
            with Visualize(step_size=1.5) as viz:
                pass

    def test_cost_data(self):
        """Tests that the correct cost function data is being recorded"""

        cost_values = [
            0.29001330299410394,
            0.16958978666464564,
            0.0554891459073546,
            -0.05179578982297689,
            -0.151952780085496,
        ]

        optimizer = qml.GradientDescentOptimizer()
        params = np.array([1.0])
        steps = 5

        with Visualize(cost_fn=cost) as viz:
            for i in range(steps):
                params = optimizer.step(cost, params)
                viz.update(params=params)

        assert cost_values == viz.cost_data[1]

    def test_param_data(self):
        """Tests that the correct parameter data is being recorded"""

        param_values = [1.03569363, 1.07061477, 1.10463974, 1.13766278, 1.16959683]

        optimizer = qml.GradientDescentOptimizer()
        params = np.array([1.0])
        steps = 5

        with Visualize(cost_fn=cost) as viz:
            for i in range(steps):
                params = optimizer.step(cost, params)
                viz.update(params=params)

        assert np.allclose(param_values, [i[0] for i in viz.param_data[1]])

    def test_text_errors(self):
        """Tests that the correct errors are thrown in the text method"""

        with pytest.raises(ValueError, match="'step' must be of type bool"):
            with Visualize() as viz:
                viz.text(step=1)

        with pytest.raises(ValueError, match="'cost' must be of type bool"):
            with Visualize() as viz:
                viz.text(cost=1)

        with pytest.raises(ValueError, match="'params' must be of type bool"):
            with Visualize() as viz:
                viz.text(params=1)

    def test_graph_errors(self):
        """Tests that the correct errors are thrown in the graph method"""

        with pytest.raises(ValueError, match="'color' must be of type str"):
            with Visualize() as viz:
                viz.graph(color=1)
