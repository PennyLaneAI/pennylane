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
Unit tests for the :mod:`pennylane.templates` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import numpy as np
import pennylane as qml
from pennylane.plugins import DefaultGaussian
from pennylane.templates.layers import (CVNeuralNetLayers, CVNeuralNetLayer,
                                        StronglyEntanglingLayers, StronglyEntanglingLayer,
                                        RandomLayers, RandomLayer)
from pennylane.init import (strong_ent_layers_uniform, strong_ent_layer_uniform,
                            random_layers_uniform, random_layer_uniform,
                            cvqnn_layers_uniform, cvqnn_layer_uniform)


class TestParameterIntegration:
    """ Integration tests for the parameter generation methods from pennylane.init
    and pennylane.templates.layers."""

    def test_integration_cvqnn_layers(self, gaussian_device, n_subsystems, n_layers):
        """Checks that pennylane.init.cvqnn_layers_uniform() integrates
        with pennnylane.templates.layers.CVNeuralNetLayers()."""

        p = cvqnn_layers_uniform(n_layers=n_layers, n_wires=n_subsystems)

        @qml.qnode(gaussian_device)
        def circuit(weights):
            CVNeuralNetLayers(*weights, wires=range(n_subsystems))
            return qml.expval.Identity(0)

        circuit(weights=p)

    def test_integration_cvqnn_layer(self, gaussian_device, n_subsystems):
        """Checks that pennylane.init.cvqnn_layer_uniform() integrates
        with pennnylane.templates.layers.CVNeuralNetLayer()."""

        p = cvqnn_layer_uniform(n_wires=n_subsystems)

        @qml.qnode(gaussian_device)
        def circuit(weights):
            CVNeuralNetLayer(*weights, wires=range(n_subsystems))
            return qml.expval.Identity(0)

        circuit(weights=p)

    def test_integration_stronglyentangling_layers(self, qubit_device, n_subsystems, n_layers):
        """Checks that the pennylane.init.strong_ent_layers_uniform() integrates
        with pennnylane.templates.layers.StronglyEntanglingLayers()."""

        p = strong_ent_layers_uniform(n_layers=n_layers, n_wires=n_subsystems)

        @qml.qnode(qubit_device)
        def circuit(weights):
            StronglyEntanglingLayers(*weights, wires=range(n_subsystems))
            return qml.expval.Identity(0)

        circuit(weights=p)

    def test_integration_stronglyentangling_layer(self, qubit_device, n_subsystems):
        """Checks that the pennylane.init.strong_ent_layer_uniform() integrates
        with pennnylane.templates.layers.StronglyEntanglingLayer()."""

        p = strong_ent_layer_uniform(n_wires=n_subsystems)

        @qml.qnode(qubit_device)
        def circuit(weights):
            StronglyEntanglingLayer(*weights, wires=range(n_subsystems))
            return qml.expval.Identity(0)

        circuit(weights=p)

    def test_integration_random_layers(self, qubit_device, n_subsystems, n_layers):
        """Checks that the pennylane.init.random_layers_uniform() integrates
        with pennnylane.templates.layers.RandomLayers()."""

        p = random_layers_uniform(n_layers=n_layers, n_wires=n_subsystems)

        @qml.qnode(qubit_device)
        def circuit(weights):
            RandomLayers(*weights, wires=range(n_subsystems))
            return qml.expval.Identity(0)

        circuit(weights=p)

    def test_integration_random_layer(self, qubit_device, n_subsystems):
        """Checks that the pennylane.init.random_layer_uniform() integrates
        with pennnylane.templates.layers.RandomLayer()."""

        p = random_layer_uniform(n_wires=n_subsystems)

        @qml.qnode(qubit_device)
        def circuit(weights):
            RandomLayer(*weights, wires=range(n_subsystems))
            return qml.expval.Identity(0)

        circuit(weights=p)
