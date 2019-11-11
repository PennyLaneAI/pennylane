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
Integration tests for templates, including integration with initialization functions
in :mod:`pennylane.init`, running templates in larger circuits,
combining templates, using positional and keyword arguments, and using different interfaces.

New tests are added as follows:

* When adding a new interface, try to import it and extend the fixture ``interfaces``.

* When adding a new template, extend the fixtures ``qubit_const`` or ``cv_const`` by a *list* of arguments to the
template. Note: Even if the template takes only one argument, it has to be wrapped in a list (i.e. [weights]).

* When adding a new parameter initialization function, extend the fixtures ``qubit_func`` or ``cv_func`` by the
function.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
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

interfaces = [('numpy', np.array)]

try:
    import torch
    interfaces.append(('torch', torch.tensor))
except ImportError as e:
    pass

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        print(tf.__version__)
        import tensorflow.contrib.eager as tfe
        tf.enable_eager_execution()
        Variable = tfe.Variable
    else:
        from tensorflow import Variable
    interfaces.append(('tf', Variable))
except ImportError as e:
    pass

#########################################
# Templates

qubit_func = [(StronglyEntanglingLayers, strong_ent_layers_uniform),
              (StronglyEntanglingLayers, strong_ent_layers_normal),
              (RandomLayers, random_layers_uniform),
              (RandomLayers, random_layers_normal)]

cv_func = [(CVNeuralNetLayers, cvqnn_layers_uniform),
           (CVNeuralNetLayers, cvqnn_layers_normal)]

qubit_const = [(StronglyEntanglingLayers, [[[[4.54, 4.79, 2.98], [4.93, 4.11, 5.58]],
                                           [[6.08, 5.94, 0.05], [2.44, 5.07, 0.95]]]]),
               (RandomLayers, [[[0.56, 5.14], [2.21, 4.27]]]),
               (AmplitudeEmbedding, [[1 / 2, 1 / 2, 1 / 2, 1 / 2]]),
               (BasisEmbedding, [[1, 0]]),
               (AngleEmbedding, [[1., 2.]])]

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

#########################################
# Circuits


def qnode_qubit_args(dev, intrfc, templ1, templ2, n):
    """QNode for qubit integration circuit using positional arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(*inp_):
        inp1_ = inp_[:n]
        inp2_ = inp_[n:]
        qml.PauliX(wires=0)
        templ1(*inp1_, wires=range(2))
        templ2(*inp2_, wires=range(2))
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]
    return circuit


def qnode_qubit_kwargs(dev, intrfc, templ1, templ2, n):
    """QNode for qubit integration circuit using keyword arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(**inp_):
        vals = list(inp_.values())
        inp1_ = vals[:n]
        inp2_ = vals[n:]
        qml.PauliX(wires=0)
        templ1(*inp1_, wires=range(2))
        templ2(*inp2_, wires=range(2))
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]
    return circuit


def qnode_cv_args(dev, intrfc, templ1, templ2, n):
    """QNode for CV integration circuit using positional arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(*inp_):
        inp1_ = inp_[:n]
        inp2_ = inp_[n:]
        qml.Displacement(1., 1., wires=0)
        templ1(*inp1_, wires=range(2))
        templ2(*inp2_, wires=range(2))
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]
    return circuit


def qnode_cv_kwargs(dev, intrfc, templ1, templ2, n):
    """QNode for CV integration circuit using keyword arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(**inp_):
        vals = list(inp_.values())
        inp1 = vals[:n]
        inp2 = vals[n:]
        qml.Displacement(1., 1., wires=0)
        templ1(*inp1, wires=range(2))
        templ2(*inp2, wires=range(2))
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]
    return circuit


class TestIntegrationCircuit:
    """Tests the integration of templates into circuits using the NumPy interface. """

    @pytest.mark.parametrize("template1, inpts1", qubit_const)
    @pytest.mark.parametrize("template2, inpts2", qubit_const)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_qubit_args(self, template1, inpts1, template2, inpts2,
                                    intrfc, to_var):
        """Checks integration of qubit templates using positional arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with *
        inpts = [to_var(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)
        circuit = qnode_qubit_args(dev, intrfc, template1, template2, len(inpts1))
        circuit(*inpts)

    @pytest.mark.parametrize("template1, inpts1", qubit_const)
    @pytest.mark.parametrize("template2, inpts2", qubit_const)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_qubit_kwargs(self, template1, inpts1, template2, inpts2,
                                      intrfc, to_var):
        """Checks integration of qubit templates using keyword arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with **
        inpts = {str(i): to_var(inp) for i, inp in enumerate(inpts)}
        dev = qml.device('default.qubit', wires=2)
        circuit = qnode_qubit_kwargs(dev, intrfc, template1, template2, len(inpts1))
        circuit(**inpts)

    @pytest.mark.parametrize("template1, inpts1", cv_const)
    @pytest.mark.parametrize("template2, inpts2", cv_const)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_cv_args(self, gaussian_device_2_wires, template1, inpts1, template2, inpts2,
                                 intrfc, to_var):
        """Checks integration of continuous-variable templates using positional arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with *
        inpts = [to_var(i) for i in inpts]
        dev = gaussian_device_2_wires
        circuit = qnode_cv_args(dev, intrfc, template1, template2, len(inpts1))
        circuit(*inpts)

    @pytest.mark.parametrize("template1, inpts1", cv_const)
    @pytest.mark.parametrize("template2, inpts2", cv_const)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_cv_kwargs(self, gaussian_device_2_wires, template1, inpts1, template2, inpts2,
                                   intrfc, to_var):
        """Checks integration of continuous-variable templates using keyword arguments."""
        inpts = inpts1 + inpts2  # Combine inputs to allow passing with **
        inpts = {str(i): to_var(inp) for i, inp in enumerate(inpts)}
        dev = gaussian_device_2_wires
        circuit = qnode_cv_kwargs(dev, intrfc, template1, template2, len(inpts1))
        circuit(**inpts)


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
