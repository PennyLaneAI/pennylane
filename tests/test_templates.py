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
qubit_const = [#(StronglyEntanglingLayers, [[[4.54, 4.79, 2.98], [4.93, 4.11, 5.58]],
               #                            [[6.08, 5.94, 0.05], [2.44, 5.07, 0.95]]]),
               (RandomLayers, [[[0.56, 5.14], [2.21, 4.27]]]),
               (AmplitudeEmbedding, [[1 / 2, 1 / 2, 1 / 2, 1 / 2]]),
               (BasisEmbedding, [[1, 0]]),
               (AngleEmbedding, [[1., 2.]])]

# Constant input fixtures for continuous-variable templates
cv_const = [(DisplacementEmbedding, [[1., 2.]]),
            (SqueezingEmbedding, [[1., 2.]]),
            (CVNeuralNetLayers, [[[2.31], [1.22]],
                                 [[3.47], [2.01]],
                                 [[0.93, 1.58], [5.07, 4.82]],
                                 [[0.21,  0.12], [-0.09, 0.04]],
                                 [[4.76, 6.08], [6.09, 6.22]],
                                 [[4.83], [1.70]],
                                 [[4.74], [5.39]],
                                 [[0.88, 0.62], [1.09, 3.02]],
                                 [[-0.01, -0.05], [0.08, -0.19]],
                                 [[1.89, 3.59], [1.49, 3.71]],
                                 [[0.09,  0.03], [-0.14,  0.04]]
                                 ]),
            (Interferometer, [[2.31], [3.49], [0.98, 1.54]])
            ]


def qfunc_qubit_args(template1, template2, n):
    """Qubit integration circuit using positional arguments"""
    def circuit(*inp_):
        inp1_ = inp_[:n]
        inp2_ = inp_[n:]
        qml.PauliX(wires=0)
        template1(*inp1_, wires=range(2))
        template2(*inp2_, wires=range(2))
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]
    return circuit


def qfunc_qubit_kwargs(template1, template2, n):
    """Qubit integration circuit using keyword arguments"""
    def circuit(**inp_):
        vals = list(inp_.values())
        inp1_ = vals[:n]
        inp2_ = vals[n:]
        qml.PauliX(wires=0)
        template1(*inp1_, wires=range(2))
        template2(*inp2_, wires=range(2))
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]
    return circuit


def qfunc_cv_args(template1, template2, n):
    """CV integration circuit using positional arguments"""
    def circuit(*inp_):
        inp1_ = inp_[:n]
        inp2_ = inp_[n:]
        qml.Displacement(1., 1., wires=0)
        template1(*inp1_, wires=range(2))
        template2(*inp2_, wires=range(2))
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]
    return circuit


def qfunc_cv_kwargs(template1, template2, n):
    """CV integration circuit using keyword arguments"""
    def circuit(**inp_):
        vals = list(inp_.values())
        inp1 = vals[:n]
        inp2 = vals[n:]
        qml.Displacement(1., 1., wires=0)
        template1(*inp1, wires=range(2))
        template2(*inp2, wires=range(2))
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]
    return circuit


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

    @pytest.mark.parametrize("template1, inpts1", qubit_const)
    @pytest.mark.parametrize("template2, inpts2", qubit_const)
    def test_integration_qubit_args(self, template1, inpts1, template2, inpts2):
        """Checks integration of qubit templates using positional arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with *
        inpts = [np.array(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)
        qfunc = qfunc_qubit_args(template1, template2, len(inpts1))
        circuit = qml.QNode(qfunc, dev)
        circuit(*inpts)

    @pytest.mark.parametrize("template1, inpts1", qubit_const)
    @pytest.mark.parametrize("template2, inpts2", qubit_const)
    def test_integration_qubit_kwargs(self, template1, inpts1, template2, inpts2):
        """Checks integration of qubit templates using keyword arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with **
        inpts = {str(i): np.array(inp) for i, inp in enumerate(inpts)}
        dev = qml.device('default.qubit', wires=2)
        qfunc = qfunc_qubit_kwargs(template1, template2, len(inpts1))
        circuit = qml.QNode(qfunc, dev)
        circuit(**inpts)

    @pytest.mark.parametrize("template1, inpts1", cv_const)
    @pytest.mark.parametrize("template2, inpts2", cv_const)
    def test_integration_cv_args(self, gaussian_device_2_wires, template1, inpts1, template2, inpts2):
        """Checks integration of continuous-variable templates using positional arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with *
        inpts = [np.array(i) for i in inpts]
        qfunc = qfunc_cv_args(template1, template2, len(inpts1))
        circuit = qml.QNode(qfunc, gaussian_device_2_wires)
        circuit(*inpts)

    @pytest.mark.parametrize("template1, inpts1", cv_const)
    @pytest.mark.parametrize("template2, inpts2", cv_const)
    def test_integration_cv_kwargs(self, gaussian_device_2_wires, template1, inpts1, template2, inpts2):
        """Checks integration of continuous-variable templates using keyword arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with **
        inpts = {str(i): np.array(inp) for i, inp in enumerate(inpts)}
        qfunc = qfunc_cv_args(template1, template2, len(inpts1))
        circuit = qml.QNode(qfunc, gaussian_device_2_wires)
        circuit(**inpts)


class TestIntegrationCircuitTorch:
    """Tests the integration of templates into circuits using the Torch interface."""

    @pytest.mark.parametrize("template1, inpts1", qubit_const)
    @pytest.mark.parametrize("template2, inpts2", qubit_const)
    def test_integration_qubit_args(self, template1, inpts1, template2, inpts2):
        """Checks integration of qubit templates using positional arguments and the Torch interface."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with *
        inpts = [torch.tensor(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)
        qfunc = qfunc_qubit_args(template1, template2, len(inpts1))
        circuit = qml.QNode(qfunc, dev)
        circuit.to_torch()
        circuit(*inpts)

    # @pytest.mark.parametrize("template1, inpts1", qubit_const)
    # @pytest.mark.parametrize("template2, inpts2", qubit_const)
    # def test_integration_qubit_kwargs(self, template1, inpts1, template2, inpts2):
    #     """Checks integration of qubit templates using keyword arguments and the Torch interface."""
    #     inpts = inpts1 + inpts2  # Combine inputs to allow passing with **
    #     inpts = {str(i): torch.tensor(inp) for i, inp in enumerate(inpts)}
    #     dev = qml.device('default.qubit', wires=2)
    #     qfunc = qfunc_qubit_kwargs(template1, template2, len(inpts1))
    #     circuit = qml.QNode(qfunc, dev)
    #     circuit.to_torch()
    #     circuit(**inpts)
    #
    # @pytest.mark.parametrize("template1, inpts1", cv_const)
    # @pytest.mark.parametrize("template2, inpts2", cv_const)
    # def test_integration_cv_args(self, gaussian_device_2_wires, template1, inpts1, template2, inpts2):
    #     """Checks integration of continuous-variable templates using positional arguments and the Torch interface."""
    #     inpts = inpts1 + inpts2  # Combine inputs to allow passing with *
    #     inpts = [torch.tensor(i) for i in inpts]
    #     qfunc = qfunc_cv_args(template1, template2, len(inpts1))
    #     circuit = qml.QNode(qfunc, gaussian_device_2_wires)
    #     circuit.to_torch()
    #     circuit(*inpts)
    #
    # @pytest.mark.parametrize("template1, inpts1", cv_const)
    # @pytest.mark.parametrize("template2, inpts2", cv_const)
    # def test_integration_cv_kwargs(self, gaussian_device_2_wires, template1, inpts1, template2, inpts2):
    #     """Checks integration of continuous-variable templates using keyword arguments and the Torch interface."""
    #     inpts = inpts1 + inpts2  # Combine inputs to allow passing with **
    #     inpts = {str(i): torch.tensor(inp) for i, inp in enumerate(inpts)}
    #     qfunc = qfunc_cv_args(template1, template2, len(inpts1))
    #     circuit = qml.QNode(qfunc, gaussian_device_2_wires)
    #     circuit.to_torch()
    #     circuit(**inpts)


class TestIntegrationCircuitTf:
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
