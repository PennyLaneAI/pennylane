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
Integration tests for templates, including integration of passing outputs of initialization functions
in :mod:`pennylane.init`, and running templates in larger circuits.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import torch
import tensorflow as tf
import numpy as np
import pennylane as qml
from pennylane.templates.layers import Interferometer
from pennylane.templates.layers import (CVNeuralNetLayers,
                                        StronglyEntanglingLayers,
                                        RandomLayers)
from pennylane.templates.embeddings import (AmplitudeEmbedding,
                                            BasisEmbedding,
                                            AngleEmbedding,
                                            SqueezingEmbedding,
                                            DisplacementEmbedding)
from pennylane.init import (strong_ent_layers_uniform,
                            strong_ent_layers_normal,
                            random_layers_uniform,
                            random_layers_normal,
                            cvqnn_layers_uniform,
                            cvqnn_layers_normal)

# When adding a new template,
# extend the appropriate lists with the template function
# and the desired inputs (i.e., features and/or weights) that need to be tested

# Constant input fixtures for qubit templates
qubit_func = [(StronglyEntanglingLayers, strong_ent_layers_uniform),
              (StronglyEntanglingLayers, strong_ent_layers_normal),
              (RandomLayers, random_layers_uniform),
              (RandomLayers, random_layers_normal)]

# Constant input fixtures for continuous-variable templates
cv_func = [(CVNeuralNetLayers, cvqnn_layers_uniform),
           (CVNeuralNetLayers, cvqnn_layers_normal)]

# Constant input fixtures for qubit templates
qubit_const = [(RandomLayers, [[[0.53479316, 5.88709314], [2.21352321, 4.28468607]]]),
               (AmplitudeEmbedding, [[1 / 2, 1 / 2, 1 / 2, 1 / 2]]),
               (BasisEmbedding, [[1, 0]]),
               (AngleEmbedding, [[1., 2.]])]

# Constant input fixtures for continuous-variable templates
cv_const = [(DisplacementEmbedding, [[1., 2.]]),
            (SqueezingEmbedding, [[1., 2.]]),
            (CVNeuralNetLayers, [[[2.33312851], [1.20670562]],
                                 [[3.49488327], [2.01683706]],
                                 [[0.9868003, 1.58798724], [5.06301407, 4.83852562]],
                                 [[0.21358641,  0.120304], [-0.00724019, 0.01996744]],
                                 [[4.62040076, 6.08773452], [6.09056998, 6.22395862]],
                                 [[4.10336783], [1.70001985]],
                                 [[4.74112903], [5.31462729]],
                                 [[0.89758198, 0.41604762], [1.09680782, 3.08223802]],
                                 [[-0.0807571, -0.00908855], [0.06051908, -0.1667079]],
                                 [[1.87210909, 3.59695024], [1.42759279, 3.84330071]],
                                 [[0.00389139,  0.05125553], [-0.12120044,  0.03111934]]
                                 ]),
            (Interferometer, [[2.33312851], [3.49488327], [0.9868003, 1.58798724]])
            ]


class TestInitializationIntegration:
    """Tests integration with the parameter initialization functions from pennylane.init"""

    @pytest.mark.parametrize("template, inpts", qubit_func)
    def test_integration_qubit_init(self, template, inpts, qubit_device, n_subsystems, n_layers):
        """Checks parameter initialization compatible with qubit templates."""

        inp = inpts(n_layers=n_layers, n_wires=n_subsystems)

        @qml.qnode(qubit_device)
        def circuit(inp_):
            template(*inp_, wires=range(n_subsystems))
            return qml.expval(qml.Identity(0))

        circuit(inp)

    @pytest.mark.parametrize("template, inpts", cv_func)
    def test_integration_cv_init(self, template, inpts, gaussian_device, n_subsystems, n_layers):
        """Checks parameter initialization compatible with continuous-variable templates."""

        inp = inpts(n_layers=n_layers, n_wires=n_subsystems)

        @qml.qnode(gaussian_device)
        def circuit(inp_):
            template(*inp_, wires=range(n_subsystems))
            return qml.expval(qml.Identity(0))

        circuit(inp)


class TestIntegrationCircuit:
    """Tests the integration of templates into circuits using the NumPy interface. """

    @pytest.mark.parametrize("template, inpts", qubit_const)
    def test_integration_qubit_positional_arg(self, template, inpts):
        """Checks integration of qubit templates using positional arguments to qnode."""

        inpts = [np.array(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(inp_):
            qml.PauliX(wires=0)
            template(*inp_, wires=range(2))
            template(*inp_, wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        circuit(inpts)

    @pytest.mark.parametrize("template, inpts", qubit_const)
    def test_integration_qubit_keyword_arg(self, template, inpts):
        """Checks integration of qubit templates using keyword arguments to qnode."""

        inpts = [np.array(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(inp_=None):
            qml.PauliX(wires=0)
            template(*inp_, wires=range(2))
            template(*inp_, wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        circuit(inp_=inpts)

    @pytest.mark.parametrize("template, inpts", cv_const)
    def test_integration_cv_positional_args(self, gaussian_device_2_wires, template, inpts):
        """Checks integration of continuous-variable templates using positional arguments to qnode."""

        inpts = [np.array(i) for i in inpts]

        @qml.qnode(gaussian_device_2_wires)
        def circuit(inp_):
            qml.Displacement(1., 1., wires=0)
            template(*inp_, wires=range(2))
            template(*inp_, wires=range(2))
            qml.Displacement(1., 1., wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

        circuit(inpts)

    @pytest.mark.parametrize("template, inpts", cv_const)
    def test_integration_cv_keyword_args(self, gaussian_device_2_wires, template, inpts):
        """Checks integration of continuous-variable templates using keyword arguments to qnode."""

        inpts = {"w"+str(i): np.array(inpts[i]) for i in range(len(inpts))}

        @qml.qnode(gaussian_device_2_wires)
        def circuit(**inp_):
            qml.Displacement(1., 1., wires=0)
            template(*inp_.values(), wires=range(2))
            template(*inp_.values(), wires=range(2))
            qml.Displacement(1., 1., wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

        circuit(**inpts)


class TestIntegrationCircuitTorch:
    """Tests the integration of templates into circuits using the Torch interface."""

    @pytest.mark.parametrize("template, inpts", qubit_const)
    def test_integration_qubit_positional_arg(self, template, inpts):
        """Checks integration of qubit templates using positional arguments to qnode."""

        inpts = [torch.tensor(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, interface='torch')
        def circuit(*inp_):
            qml.PauliX(wires=0)
            template(*inp_, wires=range(2))
            template(*inp_, wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        circuit(*inpts)

    @pytest.mark.parametrize("template, inpts", qubit_const)
    def test_integration_qubit_keyword_arg(self, template, inpts):
        """Checks integration of qubit templates using keyword arguments to qnode."""

        inpts = {"w"+str(1): torch.tensor(inpts[i]) for i in range(len(inpts))}
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, interface='torch')
        def circuit(**inp_):
            qml.PauliX(wires=0)
            template(*inp_.values(), wires=range(2))
            template(*inp_.values(), wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        circuit(**inpts)

    @pytest.mark.parametrize("template, inpts", cv_const)
    def test_integration_cv_positional_args(self, gaussian_device_2_wires, template, inpts):
        """Checks integration of continuous-variable templates using positional arguments to qnode."""

        inpts = [torch.tensor(i) for i in inpts]

        @qml.qnode(gaussian_device_2_wires, interface='torch')
        def circuit(*inp_):
            qml.Displacement(1., 1., wires=0)
            template(*inp_, wires=range(2))
            template(*inp_, wires=range(2))
            qml.Displacement(1., 1., wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

        circuit(*inpts)

    @pytest.mark.parametrize("template, inpts", cv_const)
    def test_integration_cv_keyword_args(self, gaussian_device_2_wires, template, inpts):
        """Checks integration of continuous-variable templates using keyword arguments to qnode."""

        inpts = {"w"+str(i): torch.tensor(inpts[i]) for i in range(len(inpts))}

        @qml.qnode(gaussian_device_2_wires, interface='torch')
        def circuit(**inp_):
            qml.Displacement(1., 1., wires=0)
            template(*inp_.values(), wires=range(2))
            template(*inp_.values(), wires=range(2))
            qml.Displacement(1., 1., wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

        circuit(**inpts)


class TestCircuitTfIntegration:
    """Tests the integration of templates into circuits using the TensorFlow interface."""

    @pytest.mark.parametrize("template, inpts", qubit_const)
    def test_integration_qubit_positional_arg(self, template, inpts):
        """Checks integration of qubit templates using positional arguments to qnode."""

        inpts = [tf.Variable(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, interface='tf')
        def circuit(*inp_):
            qml.PauliX(wires=0)
            template(*inp_, wires=range(2))
            template(*inp_, wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        circuit(*inpts)

    @pytest.mark.parametrize("template, inpts", qubit_const)
    def test_integration_qubit_keyword_arg(self, template, inpts):
        """Checks integration of qubit templates using keyword arguments to qnode."""

        inpts = {"w"+str(1): tf.Variable(inpts[i]) for i in range(len(inpts))}
        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev, interface='tf')
        def circuit(**inp_):
            qml.PauliX(wires=0)
            template(*inp_.values(), wires=range(2))
            template(*inp_.values(), wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        circuit(**inpts)

    @pytest.mark.parametrize("template, inpts", cv_const)
    def test_integration_cv_positional_args(self, gaussian_device_2_wires, template, inpts):
        """Checks integration of continuous-variable templates using positional arguments to qnode."""

        inpts = [tf.Variable(i) for i in inpts]

        @qml.qnode(gaussian_device_2_wires, interface='tf')
        def circuit(*inp_):
            qml.Displacement(1., 1., wires=0)
            template(*inp_, wires=range(2))
            template(*inp_, wires=range(2))
            qml.Displacement(1., 1., wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

        circuit(*inpts)

    @pytest.mark.parametrize("template, inpts", cv_const)
    def test_integration_cv_keyword_args(self, gaussian_device_2_wires, template, inpts):
        """Checks integration of continuous-variable templates using keyword arguments to qnode."""

        inpts = {"w"+str(i): tf.Variable(inpts[i]) for i in range(len(inpts))}

        @qml.qnode(gaussian_device_2_wires, interface='tf')
        def circuit(**inp_):
            qml.Displacement(1., 1., wires=0)
            template(*inp_.values(), wires=range(2))
            template(*inp_.values(), wires=range(2))
            qml.Displacement(1., 1., wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

        circuit(**inpts)
