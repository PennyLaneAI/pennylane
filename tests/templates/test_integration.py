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
Integration tests for templates.

New **templates** are added as follows:

* extend the fixtures ``QUBIT_DIFFABLE_NONDIFFABLE`` or ``CV_DIFFABLE_NONDIFFABLE``
  by the new template
* extend the fixtures ``QUBIT_INIT`` or ``CV_INIT`` if you want to test integration with initialization
  functions from the ``pennylane.init`` module.

Note that a template may need to be manually excluded from a test,
as shown for the templates listed in NO_OPS_BEFORE, which do not allow for
operations to be executed before the template is called.

Templates are tested with a range of interfaces. To test templates with an additional interface:

* Try to import the interface and add its variable creation function to INTERFACES
* Extend the fixture ``interfaces``
* Add the interface gradient computation to the TestGradientIntegration tests
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
import numpy as np
import pennylane as qml


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
        tf.enable_eager_execution()

    from tensorflow import Variable as TFVariable
    INTERFACES.append(('tf', TFVariable))

except ImportError as e:
    pass

#########################################
# Fixtures for integration tests
#########################################

# Each entry to QUBIT_DIFFABLE_NONDIFFABLE or CV_DIFFABLE_NONDIFFABLE
# adds a template with specified inputs to the integration tests
# ``TestIntegrationQnode``, ``TestIntegrationOtherOps``, ``TestIntegrationGradient``

# The entries have the following form:
# (template, dict of differentiable arguments, dict of non-differentiable arguments, n_wires)

# "Differentiable arguments" to a template are those that in principle allow a user to compute gradients for,
# while "nondifferentiable arguments" must always be passed as auxiliary (keyword) arguments to a qnode.
# n_wires is the number of wires the device needs.

# Note that the template is called in all tests using 2 wires

QUBIT_DIFFABLE_NONDIFFABLE = [(qml.templates.AmplitudeEmbedding,
                               {'features': [1 / 2, 1 / 2, 1 / 2, 1 / 2]},
                               {'wires': [0, 1], 'normalize': False},
                               2),
                              (qml.templates.AmplitudeEmbedding,
                               {'features': [1 / 2, 1 / 2, 1 / 2, 1 / 2]},
                               {'wires': [0, 1], 'normalize': True},
                               2),
                              (qml.templates.BasisEmbedding,
                               {},
                               {'wires': [0, 1], 'features': [1, 0]},
                               2),
                              (qml.templates.MottonenStatePreparation,
                               {'state_vector': np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2])},
                               {'wires': [0, 1]},
                               2),
                              (qml.templates.BasisStatePreparation,
                               {},
                               {'wires': [0, 1], 'basis_state': np.array([1, 0])},
                               2),
                              (qml.templates.StronglyEntanglingLayers,
                               {'weights': [[[4.54, 4.79, 2.98], [4.93, 4.11, 5.58]],
                                            [[6.08, 5.94, 0.05], [2.44, 5.07, 0.95]]]},
                               {'wires': [0, 1]},
                               2),
                              (qml.templates.RandomLayers,
                               {'weights': [[0.56, 5.14], [2.21, 4.27]]},
                               {'wires': [0, 1]},
                               2),
                              (qml.templates.AngleEmbedding,
                               {'features': [1., 2.]},
                               {'wires': [0, 1]},
                               2),
                              (qml.templates.QAOAEmbedding,
                               {'features': [1., 2.],
                                'weights': [[0.1, 0.1, 0.1]]},
                               {'wires': [0, 1]},
                               2),
                              (qml.templates.broadcast,
                               {'parameters': [[1.], [1.]]},
                               {'wires': [0, 1], 'unitary': qml.RX, 'pattern': 'single'},
                               2),
                              (qml.templates.SimplifiedTwoDesign,
                               {'initial_layer_weights': [1., 1.],
                                'weights': [[[1., 1.]]]},
                               {'wires': [0, 1]},
                               2),
                              (qml.templates.BasicEntanglerLayers,
                               {'weights': [[1., 1.]]},
                               {'wires': [0, 1], 'rotation': qml.RX},
                               2),
                              (qml.templates.IQPEmbedding,
                               {},
                               {'wires': [0, 1], 'features': [1., 1.]},
                               2),
                              (qml.templates.SingleExcitationUnitary,
                               {'weight': 0.56},
                               {'wires': [0, 1]},
                               2),
                              (qml.templates.DoubleExcitationUnitary,
                               {'weight': 0.56},
                               {'wires1': [0, 1],
                                'wires2': [2, 3]},
                               4),
                              (qml.templates.UCCSD,
                               {'weights': [3.90575761, -1.89772083, -1.36689032]},
                               {'wires': [0, 1, 2, 3], 'ph': [[0, 1, 2], [1, 2, 3]],
                                'pphh': [[[0, 1], [2, 3]]], 'init_state':np.array([1, 1, 0, 0])},
                               4),
                              ]

CV_DIFFABLE_NONDIFFABLE = [(qml.templates.DisplacementEmbedding,
                            {'features': [1., 2.]},
                            {'wires': [0, 1]},
                            2),
                           (qml.templates.SqueezingEmbedding,
                            {'features': [1., 2.]},
                            {'wires': [0, 1]},
                            2),
                           (qml.templates.CVNeuralNetLayers,
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
                            {'wires': [0, 1]},
                            2),
                           (qml.templates.Interferometer,
                            {'theta': [2.31],
                             'phi': [3.49],
                             'varphi': [0.98, 1.54]},
                            {'wires': [0, 1]},
                            2),
                           ]

# List templates in NO_OP_BEFORE that do not allow for operations
# before they are called in a quantum function.
# These templates will be skipped in tests of that nature.

NO_OP_BEFORE = ["AmplitudeEmbedding", "UCCSD"]

# Each entry to QUBIT_INIT and CV_INIT adds a template with specified inputs to the
# integration tests ``TestIntegrationInitFunctions``

# The entries have the following form:
#
# (template, dict of arguments)
#
# The dictionary of arguments calls the initialization function, and contains all other arguments that need to
# be defined in the template.

QUBIT_INIT = [(qml.templates.StronglyEntanglingLayers,
               {'weights': qml.init.strong_ent_layers_uniform(n_layers=3, n_wires=2), 'wires': range(2)}),
              (qml.templates.StronglyEntanglingLayers,
               {'weights': qml.init.strong_ent_layers_uniform(n_layers=2, n_wires=3), 'wires': range(3)}),
              (qml.templates.StronglyEntanglingLayers,
               {'weights': qml.init.strong_ent_layers_normal(n_layers=3, n_wires=2), 'wires': range(2)}),
              (qml.templates.StronglyEntanglingLayers,
               {'weights': qml.init.strong_ent_layers_normal(n_layers=2, n_wires=3), 'wires': range(3)}),
              (qml.templates.RandomLayers,
               {'weights': qml.init.random_layers_uniform(n_layers=3, n_rots=2, n_wires=1), 'wires': range(1)}),
              (qml.templates.RandomLayers,
               {'weights': qml.init.random_layers_uniform(n_layers=3, n_rots=2, n_wires=2), 'wires': range(2)}),
              (qml.templates.RandomLayers,
               {'weights': qml.init.random_layers_normal(n_layers=2, n_rots=2, n_wires=1), 'wires': range(1)}),
              (qml.templates.RandomLayers,
               {'weights': qml.init.random_layers_normal(n_layers=2, n_rots=2, n_wires=2), 'wires': range(2)}),
              (qml.templates.QAOAEmbedding,
               {'features': [1.], 'weights': qml.init.qaoa_embedding_uniform(n_layers=3, n_wires=1),
                'wires': range(1)}),
              (qml.templates.QAOAEmbedding,
               {'features': [1., 2.], 'weights': qml.init.qaoa_embedding_uniform(n_layers=3, n_wires=2),
                'wires': range(2)}),
              (qml.templates.QAOAEmbedding,
               {'features': [1., 2., 3.], 'weights': qml.init.qaoa_embedding_uniform(n_layers=2, n_wires=3),
                'wires': range(3)}),
              (qml.templates.QAOAEmbedding,
               {'features': [1.], 'weights': qml.init.qaoa_embedding_normal(n_layers=3, n_wires=1),
                'wires': range(1)}),
              (qml.templates.QAOAEmbedding,
               {'features': [1., 2.], 'weights': qml.init.qaoa_embedding_normal(n_layers=3, n_wires=2),
                'wires': range(2)}),
              (qml.templates.QAOAEmbedding,
               {'features': [1., 2., 3.], 'weights': qml.init.qaoa_embedding_normal(n_layers=3, n_wires=3),
                'wires': range(3)}),
              (qml.templates.SimplifiedTwoDesign,
               {'initial_layer_weights': qml.init.simplified_two_design_initial_layer_uniform(n_wires=4),
                'weights': qml.init.simplified_two_design_weights_uniform(n_layers=3, n_wires=4),
                'wires': range(4)}),
              (qml.templates.SimplifiedTwoDesign,
               {'initial_layer_weights': qml.init.simplified_two_design_initial_layer_normal(n_wires=4),
                'weights': qml.init.simplified_two_design_weights_normal(n_layers=3, n_wires=4),
                'wires': range(4)}),
              (qml.templates.BasicEntanglerLayers,
               {'weights': qml.init.basic_entangler_layers_uniform(n_layers=1, n_wires=1), 'wires': range(1)}),
              (qml.templates.BasicEntanglerLayers,
               {'weights': qml.init.basic_entangler_layers_uniform(n_layers=3, n_wires=1), 'wires': range(1)}),
              (qml.templates.BasicEntanglerLayers,
               {'weights': qml.init.basic_entangler_layers_uniform(n_layers=3, n_wires=2), 'wires': range(2)}),
              (qml.templates.BasicEntanglerLayers,
               {'weights': qml.init.basic_entangler_layers_uniform(n_layers=3, n_wires=3), 'wires': range(3)}),
              (qml.templates.BasicEntanglerLayers,
               {'weights': qml.init.basic_entangler_layers_normal(n_layers=3, n_wires=1), 'wires': range(1)}),
              (qml.templates.BasicEntanglerLayers,
               {'weights': qml.init.basic_entangler_layers_normal(n_layers=3, n_wires=2), 'wires': range(2)}),
              (qml.templates.BasicEntanglerLayers,
               {'weights': qml.init.basic_entangler_layers_normal(n_layers=3, n_wires=3), 'wires': range(3)}),
              ]

CV_INIT = [(qml.templates.CVNeuralNetLayers,
            {'theta_1': qml.init.cvqnn_layers_theta_uniform(n_layers=3, n_wires=1),
             'phi_1': qml.init.cvqnn_layers_phi_uniform(n_layers=3, n_wires=1),
             'varphi_1': qml.init.cvqnn_layers_varphi_uniform(n_layers=3, n_wires=1),
             'r': qml.init.cvqnn_layers_r_uniform(n_layers=3, n_wires=1),
             'phi_r': qml.init.cvqnn_layers_phi_r_uniform(n_layers=3, n_wires=1),
             'theta_2': qml.init.cvqnn_layers_theta_uniform(n_layers=3, n_wires=1),
             'phi_2': qml.init.cvqnn_layers_phi_uniform(n_layers=3, n_wires=1),
             'varphi_2': qml.init.cvqnn_layers_varphi_uniform(n_layers=3, n_wires=1),
             'a': qml.init.cvqnn_layers_a_uniform(n_layers=3, n_wires=1),
             'phi_a': qml.init.cvqnn_layers_phi_a_uniform(n_layers=3, n_wires=1),
             'k': qml.init.cvqnn_layers_kappa_uniform(n_layers=3, n_wires=1),
             'wires': range(1)}),
           (qml.templates.CVNeuralNetLayers,
            {'theta_1': qml.init.cvqnn_layers_theta_normal(n_layers=3, n_wires=2),
             'phi_1': qml.init.cvqnn_layers_phi_normal(n_layers=3, n_wires=2),
             'varphi_1': qml.init.cvqnn_layers_varphi_normal(n_layers=3, n_wires=2),
             'r': qml.init.cvqnn_layers_r_normal(n_layers=3, n_wires=2),
             'phi_r': qml.init.cvqnn_layers_phi_r_normal(n_layers=3, n_wires=2),
             'theta_2': qml.init.cvqnn_layers_theta_normal(n_layers=3, n_wires=2),
             'phi_2': qml.init.cvqnn_layers_phi_normal(n_layers=3, n_wires=2),
             'varphi_2': qml.init.cvqnn_layers_varphi_normal(n_layers=3, n_wires=2),
             'a': qml.init.cvqnn_layers_a_normal(n_layers=3, n_wires=2),
             'phi_a': qml.init.cvqnn_layers_phi_a_normal(n_layers=3, n_wires=2),
             'k': qml.init.cvqnn_layers_kappa_normal(n_layers=3, n_wires=2),
             'wires': range(2)}),
           (qml.templates.Interferometer,
            {'phi': qml.init.interferometer_phi_uniform(n_wires=1),
             'varphi': qml.init.interferometer_varphi_uniform(n_wires=1),
             'theta': qml.init.interferometer_theta_uniform(n_wires=1),
             'wires': range(1)}),
           (qml.templates.Interferometer,
            {'phi': qml.init.interferometer_phi_normal(n_wires=1),
             'varphi': qml.init.interferometer_varphi_normal(n_wires=1),
             'theta': qml.init.interferometer_theta_normal(n_wires=1),
             'wires': range(1)}),
           (qml.templates.Interferometer,
            {'phi': qml.init.interferometer_phi_uniform(n_wires=3),
             'varphi': qml.init.interferometer_varphi_uniform(n_wires=3),
             'theta': qml.init.interferometer_theta_uniform(n_wires=3),
             'wires': range(3)}),
           (qml.templates.Interferometer,
            {'phi': qml.init.interferometer_phi_normal(n_wires=3),
             'varphi': qml.init.interferometer_varphi_normal(n_wires=3),
             'theta': qml.init.interferometer_theta_normal(n_wires=3),
             'wires': range(3)})
           ]


class TestIntegrationQnode:
    """Tests the integration of templates into qnodes when differentiable arguments are passed as
    primary or auxiliary arguments to the qnode, using different interfaces.

    "Differentiable arguments" to a template are those that in principle allow a user to compute gradients for,
    while "nondifferentiable arguments" must always be passed as auxiliary (keyword) arguments to a qnode.

    The tests are motivated by the fact that the way arguments are passed to the qnode
    influences what shape and/or type the argument has inside the qnode, which is where the template calls it.

    All templates should work no matter how the "differentiable arguments" are passed to the qnode.
    """

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_qubit_qnode_primary_args(self, template, diffable, nondiffable, n_wires, interface, to_var):
        """Tests integration of qubit templates with other operations, passing differentiable arguments
        as primary arguments to qnode."""

        # Extract keys and items
        keys_diffable = [*diffable]
        diffable = list(diffable.values())

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Generate qnode
        dev = qml.device('default.qubit', wires=n_wires)

        # Generate qnode in which differentiable arguments are passed
        # as primary argument
        @qml.qnode(dev, interface=interface)
        def circuit(*diffable, keys_diffable=None, nondiffable=None):
            # Turn diffables back into dictionary
            all_args = {key: item for key, item in zip(keys_diffable, diffable)}

            # Merge with nondiffables
            all_args.update(nondiffable)

            template(**all_args)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit(*diffable, keys_diffable=keys_diffable, nondiffable=nondiffable)

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", CV_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_cv_qnode_primary_args(self, template, diffable, nondiffable, n_wires,
                                   interface, to_var, gaussian_dummy):
        """Tests integration of cv templates passing differentiable arguments as positional arguments to qnode."""

        # Extract keys and items
        keys_diffable = [*diffable]
        diffable = list(diffable.values())

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Generate qnode in which differentiable arguments are passed
        # as primary argument
        dev = gaussian_dummy(n_wires)

        @qml.qnode(dev, interface=interface)
        def circuit(*diffable, keys_diffable=None, nondiffable=None):
            # Turn diffables back into dictionary
            all_args = dict(zip(keys_diffable, diffable))

            # Merge with nondiffables
            all_args.update(nondiffable)

            template(**all_args)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit(*diffable, keys_diffable=keys_diffable, nondiffable=nondiffable)

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_qubit_qnode_auxiliary_args(self, template, diffable, nondiffable, n_wires, interface, to_var):
        """Tests integration of qubit templates passing differentiable arguments as auxiliary arguments to qnode."""

        # Change type of differentiable arguments
        # Fix: templates should all take arrays AND lists, at the moment this is not the case
        diffable = {k: np.array(v) for k, v in diffable.items()}

        # Merge differentiable and non-differentiable arguments
        all_args = {**diffable, **nondiffable}

        # Generate qnode
        dev = qml.device('default.qubit', wires=n_wires)

        # Generate qnode in which differentiable arguments are passed
        # as auxiliary argument
        @qml.qnode(dev, interface=interface)
        def circuit(all_args=None):
            template(**all_args)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit(all_args=all_args)

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", CV_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_qubit_cv_auxiliary_args(self, template, diffable, nondiffable, n_wires,
                                     interface, to_var, gaussian_dummy):
        """Tests integration of cv templates passing differentiable arguments as auxiliary arguments to qnode."""

        # Change type of differentiable arguments
        # Fix: templates should all take arrays AND lists, at the moment this is not the case
        diffable = {k: np.array(v) for k, v in diffable.items()}

        # Merge differentiable and non-differentiable arguments
        all_args = {**diffable, **nondiffable}

        # Generate qnode in which differentiable arguments are passed
        # as primary argument
        dev = gaussian_dummy(n_wires)

        @qml.qnode(dev, interface=interface)
        def circuit(all_args=None):
            template(**all_args)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit(all_args=all_args)


# hand-coded templates for the operation integration test
@qml.template
def QubitTemplate(w):
    qml.PauliX(wires=w)


@qml.template
def CVTemplate(w):
    qml.Displacement(1., 1., wires=w)


class TestIntegrationOtherOps:
    """Tests the integration of templates into qnodes where the template is called
    together with other operations or templates."""

    @pytest.mark.parametrize("op_before_template", [True, False])
    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", QUBIT_DIFFABLE_NONDIFFABLE)
    def test_qubit_template_followed_by_operations(self, template, diffable, nondiffable, n_wires, op_before_template):
        """Tests integration of qubit templates with other operations."""

        # skip this test if template does not allow for operations before
        if template.__name__ in NO_OP_BEFORE and op_before_template:
            pytest.skip("Template does not allow operations before - skipping this test.")

        # Change type of differentiable arguments
        # Fix: templates should all take arrays AND lists, at the moment this is not the case
        diffable = {k: np.array(v) for k, v in diffable.items()}

        # Merge differentiable and non-differentiable arguments
        nondiffable.update(diffable)

        # Generate qnode
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit(nondiffable=None):
            # Circuit with operations before and
            # after the template is called
            if op_before_template:
                QubitTemplate(w=0)
                qml.PauliX(wires=0)
            template(**nondiffable)
            if not op_before_template:
                QubitTemplate(w=0)
                qml.PauliX(wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.PauliX(1))]

        # Check that execution does not throw error
        circuit(nondiffable=nondiffable)

    @pytest.mark.parametrize("op_before_template", [True, False])
    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", CV_DIFFABLE_NONDIFFABLE)
    def test_cv_template_followed_by_operations(self, template, diffable, nondiffable, n_wires, gaussian_dummy,
                                                op_before_template):
        """Tests integration of cv templates passing differentiable arguments as auxiliary arguments to qnode."""

        # Change type of differentiable arguments
        # Fix: templates should all take arrays AND lists, at the moment this is not the case
        diffable = {k: np.array(v) for k, v in diffable.items()}

        # Merge differentiable and non-differentiable arguments
        nondiffable.update(diffable)

        # Make qnode
        dev = gaussian_dummy(n_wires)

        @qml.qnode(dev)
        def circuit(nondiffable=None):

            # Circuit with operations before and
            # after the template is called
            if op_before_template:
                CVTemplate(w=0)
                qml.Displacement(1., 1., wires=0)
            template(**nondiffable)
            if not op_before_template:
                CVTemplate(w=0)
                qml.Displacement(1., 1., wires=1)
            return [qml.expval(qml.Identity(0)), qml.expval(qml.X(1))]

        # Check that execution does not throw error
        circuit(nondiffable=nondiffable)


class TestIntegrationGradient:
    """Tests that gradients of circuits with templates can be computed."""

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", QUBIT_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_qubit_grad(self, template, diffable, nondiffable, n_wires, interface, to_var):
        """Tests that gradient calculations of qubit templates execute without error."""

        # Extract keys and items
        keys_diffable = [*diffable]
        diffable = list(diffable.values())

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Make qnode
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

        # Do gradient check for every differentiable argument
        for argnum in range(len(diffable)):

            # Check gradients in numpy interface
            if interface == 'numpy':
                grd = qml.grad(circuit, argnum=[argnum])
                grd(*diffable)

            # Check gradients in torch interface
            if interface == 'torch':
                diffable[argnum] = TorchVariable(diffable[argnum], requires_grad=True)
                res = circuit(*diffable)
                res.backward()
                diffable[argnum].grad.numpy()

            # Check gradients in tf interface
            if interface == 'tf':
                with tf.GradientTape() as tape:
                    loss = circuit(*diffable)
                    tape.gradient(loss, diffable[argnum])

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", CV_DIFFABLE_NONDIFFABLE)
    @pytest.mark.parametrize("interface, to_var", INTERFACES)
    def test_integration_cv_grad(self, template, diffable, nondiffable, n_wires, interface, to_var, gaussian_dummy):
        """Tests that gradient calculations of cv templates execute without error."""

        # Extract keys and items
        keys_diffable = [*diffable]
        diffable = list(diffable.values())

        # Turn into correct format
        diffable = [to_var(i) for i in diffable]

        # Make qnode
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

        # Do gradient check for every differentiable argument
        for argnum in range(len(diffable)):

            # Check gradients in numpy interface
            if interface == 'numpy':
                grd = qml.grad(circuit, argnum=[argnum])
                grd(*diffable)

            # Check gradients in torch interface
            if interface == 'torch':
                diffable[argnum] = TorchVariable(diffable[argnum], requires_grad=True)
                res = circuit(*diffable)
                res.backward()
                diffable[argnum].grad.numpy()

            # Check gradients in tf interface
            if interface == 'tf':
                with tf.GradientTape() as tape:
                    loss = circuit(*diffable)
                    tape.gradient(loss, diffable[argnum])


class TestInitializationIntegration:
    """Tests integration with the parameter initialization functions from pennylane.init"""

    @pytest.mark.parametrize("template, dict", QUBIT_INIT)
    def test_integration_qubit_init(self, template, dict):
        """Tests that parameter initializations are compatible with qubit templates."""

        n_wires = len(dict['wires'])
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            template(**dict)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit()

    @pytest.mark.parametrize("template, dict", CV_INIT)
    def test_integration_cv_init(self, template, dict, gaussian_dummy):
        """Tests that parameter initializations are compatible with cv templates."""

        n_wires = len(dict['wires'])
        dev = gaussian_dummy(n_wires)

        @qml.qnode(dev)
        def circuit():
            template(**dict)
            return qml.expval(qml.Identity(0))

        # Check that execution does not throw error
        circuit()


class TestNonConsecutiveWires:
    """Tests that a template results in the same state whether we use nonconsecutive wire labels or not.
    """

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", QUBIT_DIFFABLE_NONDIFFABLE)
    def test_qubit_result_is_wire_label_independent(self, template, diffable, nondiffable, n_wires):
        """Tests that qubit templates produce the same state when using two different wire labellings."""

        # merge differentiable and non-differentiable arguments:
        # we don't need them separate here
        kwargs = {**nondiffable, **diffable}

        # construct qnode with consecutive wires
        dev_consec = qml.device('default.qubit', wires=n_wires)
        @qml.qnode(dev_consec)
        def circuit_consec():
            template(**kwargs)
            return qml.expval(qml.Identity(wires=0))

        # construct qnode with nonconsecutive wires
        non_consecutive_strings = ['z', 'b', 'f', 'a', 'k', 'c', 'r', 's', 'd']
        nonconsecutive_wires = non_consecutive_strings[: n_wires]  # make flexible size wires argument
        kwargs2 = kwargs.copy()
        if 'wires' in kwargs2:
            kwargs2['wires'] = nonconsecutive_wires
        # DoubleExcitationLayers does not have a wires kwarg
        if template.__name__ == 'DoubleExcitationUnitary':
            kwargs2['wires1'] = nonconsecutive_wires[:2]
            kwargs2['wires2'] = nonconsecutive_wires[2:]
        # some kwargs in UCSSD need to be manually replaced
        if template.__name__ == 'UCCSD':
             kwargs2['ph'] = [nonconsecutive_wires[:3], nonconsecutive_wires[1:]]
             kwargs2['pphh'] = [[nonconsecutive_wires[:2], nonconsecutive_wires[2:]]]

        dev_nonconsec = qml.device('default.qubit', wires=nonconsecutive_wires)

        @qml.qnode(dev_nonconsec)
        def circuit_nonconsec():
            template(**kwargs2)
            return qml.expval(qml.Identity(wires=nonconsecutive_wires[0]))

        # run circuits
        circuit_consec()
        circuit_nonconsec()

        assert np.allclose(dev_consec.state, dev_nonconsec.state)

    @pytest.mark.parametrize("template, diffable, nondiffable, n_wires", CV_DIFFABLE_NONDIFFABLE)
    def test_cv_result_is_wire_label_independent(self, template, diffable, nondiffable, n_wires, gaussian_dummy):
        """Tests integration of cv templates with non-integer and non-consecutive wires."""

        # merge differentiable and non-differentiable arguments:
        # we don't need them separate here
        kwargs = {**nondiffable, **diffable}

        # Construct qnode with consecutive wires
        dev_consec = gaussian_dummy(wires=kwargs['wires'])

        @qml.qnode(dev_consec)
        def circuit_consec():
            template(**kwargs)
            return qml.expval(qml.Identity(wires=0))

        # Construct qnode with nonconsecutive wires
        kwargs2 = kwargs.copy()
        non_consecutive_strings = ['z', 'b', 'f', 'a', 'k', 'c', 'r', 's', 'd']
        nonconsecutive_wires = non_consecutive_strings[: n_wires]  # make flexible size wires argument
        kwargs2['wires'] = nonconsecutive_wires
        dev_nonconsec = gaussian_dummy(wires=nonconsecutive_wires)

        @qml.qnode(dev_nonconsec)
        def circuit_nonconsec():
            template(**kwargs2)
            return qml.expval(qml.Identity(wires=nonconsecutive_wires[0]))

        circuit_consec()
        circuit_nonconsec()

        assert np.allclose(dev_consec._state[0], dev_nonconsec._state[0])
