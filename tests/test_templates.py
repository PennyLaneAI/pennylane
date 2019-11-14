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

* When adding a new interface, try to import it and extend the fixture ``interfaces``. Add interface to
  TestGradientIntegration tests.

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
    from torch.autograd import Variable as TorchVariable
    interfaces.append(('torch', torch.tensor))
except ImportError as e:
    pass

try:
    import tensorflow as tf

    if tf.__version__[0] == "1":
        import tensorflow.contrib.eager as tfe
        tf.enable_eager_execution()
        TFVariable = tfe.Variable
    else:
        from tensorflow import Variable as TFVariable
    interfaces.append(('tf', TFVariable))
except ImportError as e:
    pass

#########################################
# Templates

# qubit templates and intialization functions
qubit_func = [(StronglyEntanglingLayers, strong_ent_layers_uniform),
              (StronglyEntanglingLayers, strong_ent_layers_normal),
              (RandomLayers, random_layers_uniform),
              (RandomLayers, random_layers_normal)]

# cv templates and intialization functions
cv_func = [(CVNeuralNetLayers, cvqnn_layers_uniform),
           (CVNeuralNetLayers, cvqnn_layers_normal)]

# qubit templates and constant inputs
qubit_const = [(StronglyEntanglingLayers, [[[[4.54, 4.79, 2.98], [4.93, 4.11, 5.58]],
                                           [[6.08, 5.94, 0.05], [2.44, 5.07, 0.95]]]]),
               (RandomLayers, [[[0.56, 5.14], [2.21, 4.27]]]),
               (AngleEmbedding, [[1., 2.]]),
               ]

# cv templates and constant inputs
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

# qubit templates, constant inputs, and ``argnum`` argument of qml.grad
qubit_grad = [(StronglyEntanglingLayers, [[[[4.54, 4.79, 2.98], [4.93, 4.11, 5.58]],
                                           [[6.08, 5.94, 0.05], [2.44, 5.07, 0.95]]]], [0]),
               (RandomLayers, [[[0.56, 5.14], [2.21, 4.27]]], [0]),
               (AngleEmbedding, [[1., 2.]], [0])
              ]

# cv templates, constant inputs, and ``argnum`` argument of qml.grad
cv_grad = [(DisplacementEmbedding, [[1., 2.]], [0]),
            (SqueezingEmbedding, [[1., 2.]], [0]),
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
                                 ], list(range(11))),
            (Interferometer, [[2.31], [3.49], [0.98, 1.54]], [0, 1, 2])
            ]

#########################################
# Circuits


def qnode_qubit_args(dev, intrfc, templ1, templ2, n):
    """QNode for qubit integration circuit using positional arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(*inp):
        # Split inputs again
        inp1 = inp[:n]
        inp2 = inp[n:]
        # Circuit
        qml.PauliX(wires=0)
        templ1(*inp1, wires=range(2))
        templ2(*inp2, wires=range(2))
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]
    return circuit


def qnode_qubit_kwargs(dev, intrfc, templ1, templ2, n):
    """QNode for qubit integration circuit using keyword arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(**inp):
        # Split inputs again
        ks = [int(k) for k in inp.keys()]
        vs = inp.values()
        inp = [x for _, x in sorted(zip(ks, vs))]
        inp1 = inp[:n]
        inp2 = inp[n:]
        # Circuit
        qml.PauliX(wires=0)
        templ1(*inp1, wires=range(2))
        templ2(*inp2, wires=range(2))
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]
    return circuit


def qnode_cv_args(dev, intrfc, templ1, templ2, n):
    """QNode for CV integration circuit using positional arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(*inp):
        # Split inputs again
        inp1 = inp[:n]
        inp2 = inp[n:]
        # Circuit
        qml.Displacement(1., 1., wires=0)
        templ1(*inp1, wires=range(2))
        templ2(*inp2, wires=range(2))
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]
    return circuit


def qnode_cv_kwargs(dev, intrfc, templ1, templ2, n):
    """QNode for CV integration circuit using keyword arguments"""
    @qml.qnode(dev, interface=intrfc)
    def circuit(**inp_):
        # Split inputs again
        ks = [int(k) for k in inp_.keys()]
        vs = inp_.values()
        inp = [x for _, x in sorted(zip(ks, vs))]
        inp1 = inp[:n]
        inp2 = inp[n:]
        # Circuit
        qml.Displacement(1., 1., wires=0)
        templ1(*inp1, wires=range(2))
        templ2(*inp2, wires=range(2))
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]
    return circuit

######################


class TestIntegrationCircuit:
    """Tests the integration of templates into circuits using different interfaces. """

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
        # Check that execution does not throw error
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
        # Check that execution does not throw error
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
        # Check that execution does not throw error
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
        # Check that execution does not throw error
        circuit(**inpts)


class TestIntegrationCircuitSpecialCases:
    """Tests the integration of templates with special requirements into circuits. """

    first_templ = [(AmplitudeEmbedding, [[1 / 2, 1 / 2, 1 / 2, 1 / 2]]),
                   (BasisEmbedding, [[1, 0]])]

    def qnode_first_op_args(self, dev, intrfc, templ1, templ2, n):
        """QNode for qubit integration circuit using positional arguments"""

        @qml.qnode(dev, interface=intrfc)
        def circuit(*inp):
            # Split inputs again
            inp1 = inp[:n]
            inp2 = inp[n:]
            # Circuit
            templ1(*inp1, wires=range(2))
            templ2(*inp2, wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        return circuit

    def qnode_first_op_kwargs(self, dev, intrfc, templ1, templ2, n):
        """QNode for qubit integration circuit using positional arguments"""

        @qml.qnode(dev, interface=intrfc)
        def circuit(**inp):
            # Split inputs again
            ks = [int(k) for k in inp.keys()]
            vs = inp.values()
            inp = [x for _, x in sorted(zip(ks, vs))]
            inp1 = inp[:n]
            inp2 = inp[n:]
            # Circuit
            templ1(*inp1, wires=range(2))
            templ2(*inp2, wires=range(2))
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        return circuit

    @pytest.mark.parametrize("first_tmpl, first_inpts", first_templ)
    @pytest.mark.parametrize("template, inpts", qubit_const)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_first_template_args(self, first_tmpl, first_inpts, template, inpts, intrfc, to_var):
        """Checks integration of templates that must be the first operation in the circuit
        , using positional arguments."""
        inpts = first_inpts + inpts  # Combine inputs to allow passing with *
        inpts = [to_var(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)
        circuit = self.qnode_first_op_args(dev, intrfc, first_tmpl, template, len(first_inpts))
        # Check that execution does not throw error
        circuit(*inpts)

    @pytest.mark.parametrize("first_tmpl, first_inpts", first_templ)
    @pytest.mark.parametrize("template, inpts", qubit_const)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_first_template_kwargs(self, first_tmpl, first_inpts, template, inpts, intrfc, to_var):
        """Checks integration of templates that must be the first operation in the circuit
        , using keyword arguments."""
        inpts = first_inpts + inpts  # Combine inputs to allow passing with *
        inpts = {str(i): to_var(inp) for i, inp in enumerate(inpts)}
        dev = qml.device('default.qubit', wires=2)
        circuit = self.qnode_first_op_kwargs(dev, intrfc, first_tmpl, template, len(first_inpts))
        # Check that execution does not throw error
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
        # Check that execution does not throw error
        circuit(inp)

    @pytest.mark.parametrize("template, inpts", cv_func)
    def test_integration_cv_init(self, template, inpts, gaussian_device, n_subsystems, n_layers):
        """Checks parameter initialization compatible with continuous-variable templates."""
        inp = inpts(n_layers=n_layers, n_wires=n_subsystems)
        @qml.qnode(gaussian_device)
        def circuit(inp_):
            template(*inp_, wires=range(n_subsystems))
            return qml.expval(qml.Identity(0))
        # Check that execution does not throw error
        circuit(inp)


class TestGradientIntegration:
    """Tests that gradients of circuits with templates can be computed."""

    @pytest.mark.parametrize("template, inpts, argnm", qubit_grad)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_qubit_grad(self, template, inpts, argnm, intrfc, to_var):
        """Checks that gradient calculations of qubit templates execute without error."""
        inpts = [to_var(i) for i in inpts]
        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev, interface=intrfc)
        def circuit(*inp_):
            template(*inp_, wires=range(2))
            return qml.expval(qml.Identity(0))

        # Check gradients in numpy interface
        if intrfc == 'numpy':
            grd = qml.grad(circuit, argnum=argnm)
            grd(*inpts)

        # Check gradients in torch interface
        if intrfc == 'torch':
            for a in argnm:
                inpts[a] = TorchVariable(inpts[a], requires_grad=True)
            res = circuit(*inpts)
            res.backward()
            for a in argnm:
                inpts[a].grad.numpy()

        # Check gradients in tf interface
        if intrfc == 'tf':
            grad_inpts = [inpts[a] for a in argnm]
            with tf.GradientTape() as tape:
                loss = circuit(*inpts)
                tape.gradient(loss, grad_inpts)

    @pytest.mark.parametrize("template, inpts, argnm", cv_grad)
    @pytest.mark.parametrize("intrfc, to_var", interfaces)
    def test_integration_cv_grad(self, gaussian_device_2_wires, template, inpts, argnm, intrfc, to_var):
        """Checks that gradient calculations of cv templates execute without error."""
        inpts = [to_var(i) for i in inpts]
        @qml.qnode(gaussian_device_2_wires, interface=intrfc)
        def circuit(*inp_):
            template(*inp_, wires=range(2))
            return qml.expval(qml.Identity(0))

        # Check gradients in numpy interface
        if intrfc == 'numpy':
            grd = qml.grad(circuit, argnum=argnm)
            assert grd(*inpts) is not None

        # Check gradients in torch interface
        if intrfc == 'torch':
            for a in argnm:
                inpts[a] = TorchVariable(inpts[a], requires_grad=True)
            res = circuit(*inpts)
            res.backward()
            for a in argnm:
                assert inpts[a].grad.numpy() is not None

        # Check gradients in tf interface
        if intrfc == 'tf':
            grad_inpts = [inpts[a] for a in argnm]
            with tf.GradientTape() as tape:
                loss = circuit(*inpts)
                assert tape.gradient(loss, grad_inpts) is not None

