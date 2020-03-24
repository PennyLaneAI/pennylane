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
Integration tests for templates, including integration with initialization functions
in :mod:`pennylane.init`, running templates in larger circuits,
combining templates, feeding positional and keyword arguments of qnodes into templates,
and using different interfaces.

New tests are added as follows:

* When adding a new interface, try to import it and extend the fixture ``interfaces``. Also add the interface
  gradient computation to the TestGradientIntegration tests.

* When adding a new template, extend the fixtures ``QUBIT_DIFFABLE_NONDIFFABLE`` or ``CV_DIFFABLE_NONDIFFABLE``
  by a tuple of three entries: an instance of the template, a *dict* of arguments that are differentiable,
  as well as a dict of arguments that are not differentiable. The tests will pass the differentiable arguments
  as positional AND keyword arguments to a qnode, while the nondifferentiable arguments are only passed as
  keyword arguments.

* When adding a new parameter initialization function, extend the fixtures ``QUBIT_INIT`` or
``CV_INIT``.

"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import numpy as np
import pennylane as qml
from pennylane.templates import (Interferometer,
                                 CVNeuralNetLayers,
                                 StronglyEntanglingLayers,
                                 RandomLayers,
                                 AmplitudeEmbedding,
                                 BasisEmbedding,
                                 AngleEmbedding,
                                 SqueezingEmbedding,
                                 DisplacementEmbedding,
                                 BasisStatePreparation,
                                 MottonenStatePreparation,
                                 QAOAEmbedding,
                                 SimplifiedTwoDesign,
                                 BasicEntanglerLayers)


from pennylane.templates import broadcast

from pennylane.init import (strong_ent_layers_uniform,
                            strong_ent_layers_normal,
                            random_layers_uniform,
                            random_layers_normal,
                            cvqnn_layers_a_normal,
                            cvqnn_layers_a_uniform,
                            cvqnn_layers_kappa_normal,
                            cvqnn_layers_kappa_uniform,
                            cvqnn_layers_phi_a_normal,
                            cvqnn_layers_phi_a_uniform,
                            cvqnn_layers_phi_normal,
                            cvqnn_layers_phi_r_normal,
                            cvqnn_layers_phi_r_uniform,
                            cvqnn_layers_phi_uniform,
                            cvqnn_layers_r_normal,
                            cvqnn_layers_r_uniform,
                            cvqnn_layers_theta_normal,
                            cvqnn_layers_theta_uniform,
                            cvqnn_layers_varphi_normal,
                            cvqnn_layers_varphi_uniform,
                            interferometer_phi_normal,
                            interferometer_phi_uniform,
                            interferometer_varphi_normal,
                            interferometer_varphi_uniform,
                            interferometer_theta_normal,
                            interferometer_theta_uniform,
                            qaoa_embedding_uniform,
                            qaoa_embedding_normal,
                            simplified_two_design_initial_layer_normal,
                            simplified_two_design_initial_layer_uniform,
                            simplified_two_design_weights_normal,
                            simplified_two_design_weights_uniform,
                            basic_entangler_layers_normal,
                            basic_entangler_layers_uniform)

#######################################
# Interfaces

INTERFACES = [('numpy', np.array)]

try:
    import torch
    from torch.autograd import Variable as TorchVariable

    INTERFACES.append(('torch', torch.tensor))
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
    INTERFACES.append(('tf', TFVariable))

except ImportError as e:
    pass

#########################################
# Parameters shared between test classes

# qubit templates for 2 wires, dict of differentiable arguments, dict of non-differentiable arguments
QUBIT_DIFFABLE_NONDIFFABLE = [(StronglyEntanglingLayers,
                               {'weights': [[[4.54, 4.79, 2.98], [4.93, 4.11, 5.58]],
                                            [[6.08, 5.94, 0.05], [2.44, 5.07, 0.95]]]},
                               {}),
                              (RandomLayers,
                               {'weights': [[0.56, 5.14], [2.21, 4.27]]},
                               {}),
                              (AngleEmbedding,
                               {'features': [1., 2.]},
                               {}),
                              (QAOAEmbedding,
                               {'features': [1., 2.],
                                'weights': [[0.1, 0.1, 0.1]]},
                               {}),
                              (broadcast,
                               {'parameters': [[1.], [1.]]},
                               {'unitary': qml.RX,
                                'pattern': 'single'}),
                              (SimplifiedTwoDesign,
                               {'initial_layer': [1., 1.],
                                'weights': [[[1., 1.]]]},
                               {}),
                              (BasicEntanglerLayers,
                               {'weights': [[1., 1.]]},
                               {'rotation': qml.RX}),
                              ]

# cv templates for 2 wires, dict of differentiable arguments, dict of non-differentiable arguments
CV_DIFFABLE_NONDIFFABLE = [(DisplacementEmbedding,
                            {'features': [1., 2.]},
                            {}),
                           (SqueezingEmbedding,
                            {'features': [1., 2.]},
                            {}),
                           (CVNeuralNetLayers,
                            {'theta_1': [[2.31], [1.22]],
                             'phi_1': [[3.47], [2.01]],
                             'varphi_1': [[0.93, 1.58], [5.07, 4.82]],
                             'r': [[0.21, 0.12], [-0.09, 0.04]],
                             'phi_r': [[4.76, 6.08], [6.09, 6.22]],
                             'theta_2': [[4.83], [1.70]],
                             'phi_2': [[4.74], [5.39]],
                             'varphi_2': [[0.88, 0.62], [1.09, 3.02]],
                             'a': [[-0.01, -0.05], [0.08, -0.19]],
                             'phi_a': [[1.89, 3.59], [1.49, 3.71]],
                             'k': [[0.09, 0.03], [-0.14, 0.04]]},
                            {}),
                           (Interferometer,
                            {'theta': [2.31],
                             'phi': [3.49],
                             'varphi': [0.98, 1.54]},
                            {}),
                           ]

#########################################
# Circuits shared by test classes


def qnode_qubit_args(dev, interface, template1, template2, n_args1):
    """Qubit qnode factory passing differentiable parameters as positional arguments"""

    # Signature to pass diffable arguments as single positional arg, but keep track of input names
    # in the 'keys_diffable' arguments
    @qml.qnode(dev, interface=interface)
    def circuit(*diffable, keys_diffable1=None, keys_diffable2=None, nondiffable1=None, nondiffable2=None):
        # Separate differentiable arguments
        diffable1 = diffable[:n_args1]
        diffable2 = diffable[n_args1:]

        # Turn diffables back into dictionaries
        dict1 = {key: item for key, item in zip(keys_diffable1, diffable1)}
        dict2 = {key: item for key, item in zip(keys_diffable2, diffable2)}

        # Merge with nondiffables
        dict1.update(nondiffable1)
        dict2.update(nondiffable2)

        # Add number of wires
        dict1['wires'] = range(2)
        dict2['wires'] = range(2)

        # Actual circuit
        qml.PauliX(wires=0)
        template1(**dict1)
        template2(**dict2)
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

    return circuit


def qnode_cv_args(dev, interface, template1, template2, n_args1):
    """CV qnode factory passing differentiable parameters as positional arguments"""

    # Signature juggling to pass diffable as single positional arg, but keep track of input names

    @qml.qnode(dev, interface=interface)
    def circuit(*diffable, keys_diffable1=None, keys_diffable2=None, nondiffable1=None, nondiffable2=None):
        # Separate differentiable arguments
        diffable1 = diffable[:n_args1]
        diffable2 = diffable[n_args1:]

        # Turn diffables back into dictionaries
        dict1 = {key: item for key, item in zip(keys_diffable1, diffable1)}
        dict2 = {key: item for key, item in zip(keys_diffable2, diffable2)}

        # Merge with nondiffables
        dict1.update(nondiffable1)
        dict2.update(nondiffable2)

        # Add number of wires
        dict1['wires'] = range(2)
        dict2['wires'] = range(2)

        # Actual circuit
        qml.Displacement(1., 1., wires=0)
        template1(**dict1)
        template2(**dict2)
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

    return circuit


def qnode_qubit_kwargs(dev, interface, template1, template2):
    """Qubit qnode factory passing differentiable parameters as keyword arguments"""

    @qml.qnode(dev, interface=interface)
    def circuit(nondiffable1=None, nondiffable2=None):
        # Add wires
        nondiffable1['wires'] = range(2)
        nondiffable2['wires'] = range(2)

        # Circuit
        qml.PauliX(wires=0)
        template1(**nondiffable1)
        template2(**nondiffable2)
        qml.PauliX(wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

    return circuit


def qnode_cv_kwargs(dev, interface, template1, template2):
    """CV qnode factory passing differentiable parameters as keyword arguments"""

    @qml.qnode(dev, interface=interface)
    def circuit(nondiffable1=None, nondiffable2=None):
        # Add wires
        nondiffable1['wires'] = range(2)
        nondiffable2['wires'] = range(2)

        # Circuit
        qml.Displacement(1., 1., wires=0)
        template1(**nondiffable1)
        template2(**nondiffable2)
        qml.Displacement(1., 1., wires=1)
        return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

    return circuit


######################


class TestIntegrationCircuit:
    """Tests the integration of templates into circuits using different interfaces. """

    @pytest.mark.parametrize("template1, diffable1, nondiffable1", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("template2, diffable2, nondiffable2", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_qubit_diffable(self, template1, diffable1, nondiffable1,
                                        template2, diffable2, nondiffable2,
                                        interface, to_var):
        """Tests integration of qubit templates passing differentiable arguments as positional arguments to qnode."""
        #TODO: rewrite test to avoid quadratic growth of test cases with the number of templates

        # Extract keys and items
        keys_diffable1 = [*diffable1]
        diffable1 = list(diffable1.values())
        keys_diffable2 = [*diffable2]
        diffable2 = list(diffable2.values())

        # Combine diffable inputs to allow passing with *
        diffable = diffable1 + diffable2

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Generate qnode
        dev = qml.device('default.qubit', wires=2)
        circuit = qnode_qubit_args(dev, interface, template1, template2, len(diffable1))

        # Check that execution does not throw error
        circuit(*diffable, keys_diffable1=keys_diffable1, keys_diffable2=keys_diffable2,
                nondiffable1=nondiffable1, nondiffable2=nondiffable2)

    @pytest.mark.parametrize("template1, diffable1, nondiffable1", CV_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("template2, diffable2, nondiffable2", CV_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_cv_diffable(self, template1, diffable1, nondiffable1,
                                     template2, diffable2, nondiffable2,
                                     interface, to_var, gaussian_device_2_wires):
        """Tests integration of cv templates passing differentiable arguments as positional arguments to qnode."""

        # Extract keys and items
        keys_diffable1 = [*diffable1]
        diffable1 = list(diffable1.values())
        keys_diffable2 = [*diffable2]
        diffable2 = list(diffable2.values())

        # Combine diffable inputs to allow passing with *
        diffable = diffable1 + diffable2

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Generate qnode
        circuit = qnode_cv_args(gaussian_device_2_wires, interface, template1, template2, len(diffable1))

        # Check that execution does not throw error
        circuit(*diffable, keys_diffable1=keys_diffable1, keys_diffable2=keys_diffable2,
                nondiffable1=nondiffable1, nondiffable2=nondiffable2)

    @pytest.mark.parametrize("template1, diffable1, nondiffable1", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("template2, diffable2, nondiffable2", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_qubit_nondiffable(self, template1, diffable1, nondiffable1,
                                           template2, diffable2, nondiffable2,
                                           interface, to_var):
        """Tests integration of qubit templates passing differentiable arguments as keyword arguments to qnode."""

        # Change type of differentiable arguments
        # TODO: templates should all take arrays AND lists, at the moment this is not the case
        diffable1 = {k: np.array(v) for k, v in diffable1.items()}
        diffable2 = {k: np.array(v) for k, v in diffable2.items()}

        # Merge differentiable and non-differentiable arguments
        nondiffable1.update(diffable1)
        nondiffable2.update(diffable2)

        # Generate qnode
        dev = qml.device('default.qubit', wires=2)
        circuit = qnode_qubit_kwargs(dev, interface, template1, template2)

        # Check that execution does not throw error
        circuit(nondiffable1=nondiffable1, nondiffable2=nondiffable2)

    @pytest.mark.parametrize("template1, diffable1, nondiffable1", CV_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("template2, diffable2, nondiffable2", CV_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_cv_nondiffable(self, template1, diffable1, nondiffable1,
                                        template2, diffable2, nondiffable2,
                                        interface, to_var, gaussian_device_2_wires):
        """Tests integration of cv templates passing differentiable arguments as keyword arguments to qnode."""

        # Change type of differentiable arguments
        # TODO: templates should all take arrays AND lists, at the moment this is not the case
        diffable1 = {k: np.array(v) for k, v in diffable1.items()}
        diffable2 = {k: np.array(v) for k, v in diffable2.items()}

        # Merge differentiable and non-differentiable arguments
        nondiffable1.update(diffable1)
        nondiffable2.update(diffable2)

        # Generate qnode
        circuit = qnode_cv_kwargs(gaussian_device_2_wires, interface, template1, template2)

        # Check that execution does not throw error
        circuit(nondiffable1=nondiffable1, nondiffable2=nondiffable2)


class TestIntegrationCircuitSpecialCases:
    """Tests the integration of templates with special requirements into circuits. """

    FIRST_QUBIT_DIFFABLE_NONDIFFABLE = [(AmplitudeEmbedding,
                                         {'features': [1 / 2, 1 / 2, 1 / 2, 1 / 2]},
                                         {'normalize': False}),
                                        (AmplitudeEmbedding,
                                         {'features': [1 / 2, 1 / 2, 1 / 2, 1 / 2]},
                                         {'normalize': True}),
                                        (BasisEmbedding,
                                         {},
                                         {'features': [1, 0]}),
                                        (MottonenStatePreparation,
                                         {'state_vector': np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2])},
                                         {}),
                                        (BasisStatePreparation,
                                         {},
                                         {'basis_state': np.array([1, 0])})]

    def qnode_first_qubit_args(self, dev, interface, template1, template2, n_args1):
        """Qubit qnode factory passing differentiable parameters as positional arguments, and using
        the template on"""

        # Signature juggling to pass diffable as single positional arg, but keep track of input names
        @qml.qnode(dev, interface=interface)
        def circuit(*diffable, keys_diffable1=None, keys_diffable2=None, nondiffable1=None, nondiffable2=None):
            # Separate differentiable arguments
            diffable1 = diffable[:n_args1]
            diffable2 = diffable[n_args1:]

            # Turn diffables back into dictionaries
            dict1 = {key: item for key, item in zip(keys_diffable1, diffable1)}
            dict2 = {key: item for key, item in zip(keys_diffable2, diffable2)}

            # Merge with nondiffables
            dict1.update(nondiffable1)
            dict2.update(nondiffable2)

            # Add number of wires
            dict1['wires'] = range(2)
            dict2['wires'] = range(2)

            # Actual circuit
            template1(**dict1)
            template2(**dict2)
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        return circuit

    def qnode_first_qubit_kwargs(self, dev, interface, template1, template2):
        """Qubit qnode factory passing differentiable parameters as keyword arguments"""

        @qml.qnode(dev, interface=interface)
        def circuit(nondiffable1=None, nondiffable2=None):
            # Add wires
            nondiffable1['wires'] = range(2)
            nondiffable2['wires'] = range(2)

            # Circuit
            template1(**nondiffable1)
            template2(**nondiffable2)
            qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        return circuit

    @pytest.mark.parametrize("template1, diffable1, nondiffable1", FIRST_QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("template2, diffable2, nondiffable2", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_qubit_diffable(self, template1, diffable1, nondiffable1,
                                        template2, diffable2, nondiffable2,
                                        interface, to_var):
        """Tests integration of qubit templates passing differentiable arguments as positional arguments to qnode."""

        # Extract keys and items
        keys_diffable1 = [*diffable1]
        diffable1 = list(diffable1.values())
        keys_diffable2 = [*diffable2]
        diffable2 = list(diffable2.values())

        # Combine diffable inputs to allow passing with *
        diffable = diffable1 + diffable2

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Generate qnode
        dev = qml.device('default.qubit', wires=2)
        circuit = self.qnode_first_qubit_args(dev, interface, template1, template2, len(diffable1))

        # Check that execution does not throw error
        circuit(*diffable, keys_diffable1=keys_diffable1, keys_diffable2=keys_diffable2,
                nondiffable1=nondiffable1, nondiffable2=nondiffable2)

    @pytest.mark.parametrize("template1, diffable1, nondiffable1", FIRST_QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("template2, diffable2, nondiffable2", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_qubit_nondiffable(self, template1, diffable1, nondiffable1,
                                           template2, diffable2, nondiffable2,
                                           interface, to_var):
        """Tests integration of qubit templates passing differentiable arguments as keyword arguments to qnode."""

        # Change type of differentiable arguments
        # TODO: templates should all take arrays AND lists, at the moment this is not the case
        diffable1 = {k: np.array(v) for k, v in diffable1.items()}
        diffable2 = {k: np.array(v) for k, v in diffable2.items()}

        # Merge differentiable and non-differentiable arguments
        nondiffable1.update(diffable1)
        nondiffable2.update(diffable2)

        # Generate qnode
        dev = qml.device('default.qubit', wires=2)
        circuit = self.qnode_first_qubit_kwargs(dev, interface, template1, template2)

        # Check that execution does not throw error
        circuit(nondiffable1=nondiffable1, nondiffable2=nondiffable2)


class TestInitializationIntegration:
    """Tests integration with the parameter initialization functions from pennylane.init"""

    # TODO: Combine CV and Qubit tests, since the only difference is the device

    def make_n_features(self, n):
        """Helper to prepare dummy feature inputs for templates that have
        as many features as number of wires."""
        return [i for i in range(n)]

    QUBIT_INIT = [(StronglyEntanglingLayers,
                   {'weights': strong_ent_layers_uniform(n_layers=3, n_wires=2), 'wires': range(2)}),
                  (StronglyEntanglingLayers,
                   {'weights': strong_ent_layers_uniform(n_layers=2, n_wires=3), 'wires': range(3)}),
                  (StronglyEntanglingLayers,
                   {'weights': strong_ent_layers_normal(n_layers=3, n_wires=2), 'wires': range(2)}),
                  (StronglyEntanglingLayers,
                   {'weights': strong_ent_layers_normal(n_layers=2, n_wires=3), 'wires': range(3)}),
                  (RandomLayers,
                   {'weights': random_layers_uniform(n_layers=3, n_rots=2, n_wires=2), 'wires': range(2)}),
                  (RandomLayers,
                   {'weights': random_layers_uniform(n_layers=3, n_rots=2, n_wires=2), 'wires': range(2)}),
                  (RandomLayers,
                   {'weights': random_layers_normal(n_layers=2, n_rots=2, n_wires=3), 'wires': range(3)}),
                  (RandomLayers,
                   {'weights': random_layers_normal(n_layers=2, n_rots=2, n_wires=3), 'wires': range(3)}),
                  (QAOAEmbedding,
                   {'features': [1., 2.], 'weights': qaoa_embedding_uniform(n_layers=3, n_wires=2), 'wires': range(2)}),
                  (QAOAEmbedding,
                   {'features': [1., 2.], 'weights': qaoa_embedding_uniform(n_layers=3, n_wires=2), 'wires': range(2)}),
                  (QAOAEmbedding,
                   {'features': [1., 2.], 'weights': qaoa_embedding_normal(n_layers=2, n_wires=3), 'wires': range(3)}),
                  (QAOAEmbedding,
                   {'features': [1., 2.], 'weights': qaoa_embedding_normal(n_layers=2, n_wires=3), 'wires': range(3)}),
                  (QAOAEmbedding,
                   {'features': [1., 2.], 'weights': qaoa_embedding_normal(n_layers=2, n_wires=1), 'wires': range(1)}),
                  (QAOAEmbedding,
                   {'features': [1., 2.], 'weights': qaoa_embedding_uniform(n_layers=2, n_wires=1), 'wires': range(1)}),
                  (SimplifiedTwoDesign,
                   {'initial_layer': simplified_two_design_initial_layer_uniform(n_wires=4),
                    'weights': simplified_two_design_weights_uniform(n_layers=3, n_wires=4),
                    'wires': range(4)}),
                  (SimplifiedTwoDesign,
                   {'initial_layer': simplified_two_design_initial_layer_normal(n_wires=4),
                    'weights': simplified_two_design_weights_normal(n_layers=3, n_wires=4),
                    'wires': range(4)}),
                  (BasicEntanglerLayers,
                   {'weights': basic_entangler_layers_uniform(n_layers=1, n_wires=1), 'wires': range(1)}),
                  (BasicEntanglerLayers,
                   {'weights': basic_entangler_layers_uniform(n_layers=3, n_wires=1), 'wires': range(1)}),
                  (BasicEntanglerLayers,
                   {'weights': basic_entangler_layers_uniform(n_layers=3, n_wires=2), 'wires': range(2)}),
                  (BasicEntanglerLayers,
                   {'weights': basic_entangler_layers_uniform(n_layers=3, n_wires=3), 'wires': range(3)}),
                  (BasicEntanglerLayers,
                   {'weights': basic_entangler_layers_normal(n_layers=3, n_wires=1), 'wires': range(1)}),
                  (BasicEntanglerLayers,
                   {'weights': basic_entangler_layers_normal(n_layers=3, n_wires=2), 'wires': range(2)}),
                  (BasicEntanglerLayers,
                   {'weights': basic_entangler_layers_normal(n_layers=3, n_wires=3), 'wires': range(3)}),
                  ]

    CV_INIT = [(CVNeuralNetLayers,
                {'theta_1': cvqnn_layers_theta_uniform(n_layers=3, n_wires=2),
                 'phi_1': cvqnn_layers_phi_uniform(n_layers=3, n_wires=2),
                 'varphi_1': cvqnn_layers_varphi_uniform(n_layers=3, n_wires=2),
                 'r': cvqnn_layers_r_uniform(n_layers=3, n_wires=2),
                 'phi_r': cvqnn_layers_phi_r_uniform(n_layers=3, n_wires=2),
                 'theta_2': cvqnn_layers_theta_uniform(n_layers=3, n_wires=2),
                 'phi_2': cvqnn_layers_phi_uniform(n_layers=3, n_wires=2),
                 'varphi_2': cvqnn_layers_varphi_uniform(n_layers=3, n_wires=2),
                 'a': cvqnn_layers_a_uniform(n_layers=3, n_wires=2),
                 'phi_a': cvqnn_layers_phi_a_uniform(n_layers=3, n_wires=2),
                 'k': cvqnn_layers_kappa_uniform(n_layers=3, n_wires=2),
                 'wires': range(2)}),
               (CVNeuralNetLayers,
                {'theta_1': cvqnn_layers_theta_normal(n_layers=3, n_wires=2),
                 'phi_1': cvqnn_layers_phi_normal(n_layers=3, n_wires=2),
                 'varphi_1': cvqnn_layers_varphi_normal(n_layers=3, n_wires=2),
                 'r': cvqnn_layers_r_normal(n_layers=3, n_wires=2),
                 'phi_r': cvqnn_layers_phi_r_normal(n_layers=3, n_wires=2),
                 'theta_2': cvqnn_layers_theta_normal(n_layers=3, n_wires=2),
                 'phi_2': cvqnn_layers_phi_normal(n_layers=3, n_wires=2),
                 'varphi_2': cvqnn_layers_varphi_normal(n_layers=3, n_wires=2),
                 'a': cvqnn_layers_a_normal(n_layers=3, n_wires=2),
                 'phi_a': cvqnn_layers_phi_a_normal(n_layers=3, n_wires=2),
                 'k': cvqnn_layers_kappa_normal(n_layers=3, n_wires=2),
                 'wires': range(2)}),
               (Interferometer,
                {'phi': interferometer_phi_uniform(n_wires=2), 'varphi': interferometer_varphi_uniform(n_wires=2),
                 'theta': interferometer_theta_uniform(n_wires=2), 'wires': range(2)}),
               (Interferometer,
                {'phi': interferometer_phi_normal(n_wires=2), 'varphi': interferometer_varphi_normal(n_wires=2),
                 'theta': interferometer_theta_normal(n_wires=2), 'wires': range(2)}),
               (Interferometer,
                {'phi': interferometer_phi_uniform(n_wires=3), 'varphi': interferometer_varphi_uniform(n_wires=3),
                 'theta': interferometer_theta_uniform(n_wires=3), 'wires': range(3)}),
               (Interferometer,
                {'phi': interferometer_phi_normal(n_wires=3), 'varphi': interferometer_varphi_normal(n_wires=3),
                 'theta': interferometer_theta_normal(n_wires=3), 'wires': range(3)})
               ]

    @pytest.mark.parametrize("template, dict", QUBIT_INIT)
    def test_integration_qubit_init(self, template, dict):
        """Checks parameter initialization compatible with qubit templates."""

        n_wires = len(dict['wires'])
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            template(**dict)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit()

    @pytest.mark.parametrize("template, dict", CV_INIT)
    def test_integration_qubit_init(self, template, dict, gaussian_dummy):
        """Checks parameter initialization compatible with qubit templates."""

        n_wires = len(dict['wires'])
        dev = gaussian_dummy(n_wires)

        @qml.qnode(dev)
        def circuit():
            template(**dict)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit()


class TestGradientIntegration:
    """Tests that gradients of circuits with templates can be computed."""

    QUBIT_DIFFABLE_NONDIFFABLE_ARGNUM = [(StronglyEntanglingLayers,
                                          {'weights': [[[4.54, 4.79, 2.98], [4.93, 4.11, 5.58]],
                                                       [[6.08, 5.94, 0.05], [2.44, 5.07, 0.95]]]},
                                          {'wires': range(2)},
                                          [0]),
                                         (RandomLayers,
                                          {'weights': [[0.56, 5.14], [2.21, 4.27]]},
                                          {'wires': range(2)},
                                          [0]),
                                         (AngleEmbedding,
                                          {'features': [1., 2.]},
                                          {'wires': range(2)},
                                          [0]),
                                         (QAOAEmbedding,
                                          {'features': [1., 2.], 'weights': [[0.1, 0.1, 0.1]]},
                                          {'wires': range(2)},
                                          [0]),
                                         (QAOAEmbedding,
                                          {'features': [1., 2.], 'weights': [[0.1, 0.1, 0.1]]},
                                          {'wires': range(2)},
                                          [1]),
                                         (broadcast,
                                          {'parameters': [[1.], [1.]]},
                                          {'unitary': qml.RX,
                                           'pattern': 'single',
                                           'wires': [0, 1]},
                                          [0]),
                                         (SimplifiedTwoDesign,
                                          {'initial_layer': [1., 1.],
                                           'weights': [[[1., 1.]]]},
                                          {'wires': [0, 1]},
                                          [0, 1]),
                                         (BasicEntanglerLayers,
                                          {'weights': [[1., 1.]]},
                                          {'wires': [0, 1]},
                                          [0]),
                                         ]

    CV_DIFFABLE_NONDIFFABLE_ARGNUM = [(DisplacementEmbedding,
                                       {'features': [1., 2.]},
                                       {'wires': range(2)},
                                       [0]),
                                      (SqueezingEmbedding,
                                       {'features': [1., 2.]},
                                       {'wires': range(2)},
                                       [0]),
                                      (CVNeuralNetLayers,
                                       {'theta_1': [[2.31], [1.22]],
                                        'phi_1': [[3.47], [2.01]],
                                        'varphi_1': [[0.93, 1.58], [5.07, 4.82]],
                                        'r': [[0.21, 0.12], [-0.09, 0.04]],
                                        'phi_r': [[4.76, 6.08], [6.09, 6.22]],
                                        'theta_2': [[4.83], [1.70]],
                                        'phi_2': [[4.74], [5.39]],
                                        'varphi_2': [[0.88, 0.62], [1.09, 3.02]],
                                        'a': [[-0.01, -0.05], [0.08, -0.19]],
                                        'phi_a': [[1.89, 3.59], [1.49, 3.71]],
                                        'k': [[0.09, 0.03], [-0.14, 0.04]]},
                                       {'wires': range(2)},
                                       list(range(11))),
                                      (Interferometer,
                                       {'theta': [2.31],
                                        'phi': [3.49],
                                        'varphi': [0.98, 1.54]},
                                       {'wires': range(2)},
                                       [0, 1, 2])
                                      ]

    @pytest.mark.parametrize("template, diffable, nondiffable, argnum", QUBIT_DIFFABLE_NONDIFFABLE_ARGNUM)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_qubit_grad(self, template, diffable, nondiffable, argnum, interface, to_var):
        """Tests that gradient calculations of qubit templates execute without error."""

        # Extract keys and items
        keys_diffable = [*diffable]
        diffable = list(diffable.values())

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Make qnode
        n_wires = len(nondiffable['wires'])
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev, interface=interface)
        def circuit(*diffable):

            # Turn diffables back into dictionaries
            dict = {key: item for key, item in zip(keys_diffable, diffable)}

            # Merge diffables and nondiffables
            dict.update(nondiffable)

            # Circuit
            template(**dict)
            return qml.expval(qml.Identity(0))

        # Check gradients in numpy interface
        if interface == 'numpy':
            grd = qml.grad(circuit, argnum=argnum)
            grd(*diffable)

        # Check gradients in torch interface
        if interface == 'torch':
            for a in argnum:
                diffable[a] = TorchVariable(diffable[a], requires_grad=True)
            res = circuit(*diffable)
            res.backward()
            for a in argnum:
                diffable[a].grad.numpy()

        # Check gradients in tf interface
        if interface == 'tf':
            grad_inpts = [diffable[a] for a in argnum]
            with tf.GradientTape() as tape:
                loss = circuit(*diffable)
                tape.gradient(loss, grad_inpts)

    @pytest.mark.parametrize("template, diffable, nondiffable, argnum", CV_DIFFABLE_NONDIFFABLE_ARGNUM)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_cv_grad(self, template, diffable, nondiffable, argnum, interface, to_var, gaussian_dummy):
        """Tests that gradient calculations of cv templates execute without error."""

        # Extract keys and items
        keys_diffable = [*diffable]
        diffable = list(diffable.values())

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Make qnode
        n_wires = len(nondiffable['wires'])
        dev = gaussian_dummy(n_wires)

        @qml.qnode(dev, interface=interface)
        def circuit(*diffable):

            # Turn diffables back into dictionaries
            dict = {key: item for key, item in zip(keys_diffable, diffable)}

            # Merge diffables and nondiffables
            dict.update(nondiffable)

            # Circuit
            template(**dict)
            return qml.expval(qml.Identity(0))

        # Check gradients in numpy interface
        if interface == 'numpy':
            grd = qml.grad(circuit, argnum=argnum)
            grd(*diffable)

        # Check gradients in torch interface
        if interface == 'torch':
            for a in argnum:
                diffable[a] = TorchVariable(diffable[a], requires_grad=True)
            res = circuit(*diffable)
            res.backward()
            for a in argnum:
                diffable[a].grad.numpy()

        # Check gradients in tf interface
        if interface == 'tf':
            grad_inpts = [diffable[a] for a in argnum]
            with tf.GradientTape() as tape:
                loss = circuit(*diffable)
                tape.gradient(loss, grad_inpts)
