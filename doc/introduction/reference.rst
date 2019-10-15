.. role:: html(raw)
   :format: html

.. _intro_reference:

Quick reference
===============

This quick reference links to the :ref:`API <api_qml>` of all :ref:`quantum operations <intro_ref_ops>`,
:ref:`measurements <intro_ref_meas>`, :ref:`templates <intro_ref_temp>` and
:ref:`optimizers <intro_ref_opt>` supported in PennyLane.

.. _intro_ref_ops:

Quantum operations
------------------

.. currentmodule:: pennylane.ops

PennyLane supports a wide variety of quantum operations - such as gates, state preparations and measurement
observables. These operations can be used exclusively in quantum functions. Revisiting the
first example from this section, we find the :class:`RZ <pennylane.ops.qubit.RZ>`,
:class:`CNOT <pennylane.ops.qubit.CNOT>`,
:class:`RY <pennylane.ops.qubit.RY>` :ref:`gates <intro_ref_ops_qgates>` and the
:class:`PauliZ <pennylane.ops.qubit.PauliZ>` :ref:`observable <intro_ref_ops_qobs>`:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))

You find a list of all quantum operations here, as well as in the :ref:`API <api_qml>`.

.. _intro_ref_ops_qubit:

Qubit operations
^^^^^^^^^^^^^^^^

.. _intro_ref_ops_qgates:

Qubit gates
***********
* :ref:`CNOT <pennylane_ops_qubit_CNOT>`
* :ref:`CRot <pennylane_ops_qubit_CRot>`
* :ref:`CRX <pennylane_ops_qubit_CRX>`
* :ref:`CRY <pennylane_ops_qubit_CRY>`
* :ref:`CRZ <pennylane_ops_qubit_CRZ>`
* :ref:`CSWAP <pennylane_ops_qubit_CSWAP>`
* :ref:`CZ <pennylane_ops_qubit_CZ>`
* :ref:`Hadamard <pennylane_ops_qubit_Hadamard>`
* :ref:`PauliX <pennylane_ops_qubit_PauliX>`
* :ref:`PauliY <pennylane_ops_qubit_PauliY>`
* :ref:`PauliZ <pennylane_ops_qubit_PauliZ>`
* :ref:`PhaseShift <pennylane_ops_qubit_PhaseShift>`
* :ref:`QubitUnitary <pennylane_ops_qubit_QubitUnitary>`
* :ref:`Rot <pennylane_ops_qubit_Rot>`
* :ref:`RX <pennylane_ops_qubit_RX>`
* :ref:`RY <pennylane_ops_qubit_RY>`
* :ref:`RZ <pennylane_ops_qubit_RZ>`
* :ref:`S <pennylane_ops_qubit_S>`
* :ref:`SWAP <pennylane_ops_qubit_SWAP>`
* :ref:`T <pennylane_ops_qubit_T>`


Qubit state preparation
***********************

.. toctree::
    :maxdepth: 1

* :ref:`BasisState <pennylane_ops_qubit_BasisState>`
* :ref:`QubitStateVector <pennylane_ops_qubit_QubitStateVector>`


.. _intro_ref_ops_qobs:

Qubit observables
*****************

.. toctree::
    :maxdepth: 1

* :ref:`Hadamard <pennylane_ops_qubit_Hadamard>`
* :ref:`Hermitian <pennylane_ops_qubit_Hermitian>`
* :ref:`PauliX <pennylane_ops_qubit_PauliX>`
* :ref:`PauliY <pennylane_ops_qubit_PauliY>`
* :ref:`PauliZ <pennylane_ops_qubit_PauliZ>`

.. _intro_ref_ops_cv:

Continuous-variable (CV) operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _intro_ref_ops_cvgates:

CV Gates
********

.. toctree::
    :maxdepth: 1

* :ref:`Beamsplitter <pennylane_ops_cv_Beamsplitter>`
* :ref:`ControlledAddition <pennylane_ops_cv_ControlledAddition>`
* :ref:`ControlledPhase <pennylane_ops_cv_ControlledPhase>`
* :ref:`CrossKerr <pennylane_ops_cv_CrossKerr>`
* :ref:`CubicPhase <pennylane_ops_cv_CubicPhase>`
* :ref:`Displacement <pennylane_ops_cv_Displacement>`
* :ref:`Interferometer <pennylane_ops_cv_Interferometer>`
* :ref:`Kerr <pennylane_ops_cv_Kerr>`
* :ref:`Kerr <pennylane_ops_cv_QuadraticPhase>`
* :ref:`QuadraticPhase <pennylane_ops_cv_Rotation>`
* :ref:`Squeezing <pennylane_ops_cv_Squeezing>`
* :ref:`TwoModeSqueezing <pennylane_ops_cv_TwoModeSqueezing>`


CV state preparation
********************

.. toctree::
    :maxdepth: 1

* :ref:`CatState <pennylane_ops_cv_CatState>`
* :ref:`CoherentState <pennylane_ops_cv_CoherentState>`
* :ref:`DisplacedSqueezedState <pennylane_ops_cv_DisplacedSqueezedState>`
* :ref:`FockDensityMatrix <pennylane_ops_cv_FockDensityMatrix>`
* :ref:`FockState <pennylane_ops_cv_FockState>`
* :ref:`FockStateVector <pennylane_ops_cv_FockStateVector>`
* :ref:`GaussianState <pennylane_ops_cv_GaussianState>`
* :ref:`SqueezedState <pennylane_ops_cv_SqueezedState>`
* :ref:`ThermalState <pennylane_ops_cv_ThermalState>`

.. _intro_ref_ops_cvobs:

CV observables
**************

.. toctree::
    :maxdepth: 1

* :ref:`FockStateProjector <pennylane_ops_cv_FockStateProjector>`
* :ref:`NumberOperator <pennylane_ops_cv_NumberOperator>`
* :ref:`P <pennylane_ops_cv_P>`
* :ref:`PolyXP <pennylane_ops_cv_PolyXP>`
* :ref:`QuadOperator <pennylane_ops_cv_QuadOperator>`
* :ref:`X <pennylane_ops_cv_X>`


Shared operations
^^^^^^^^^^^^^^^^^

The only operation shared by both qubit and continouous-variable architectures is the Identity.

.. toctree::
    :maxdepth: 1

* :ref:`Identity <pennylane_ops_Identity>`


.. _intro_ref_meas:

Measurements
------------

.. currentmodule:: pennylane.measure

PennyLane can extract different types of measurement results: The expectation of an observable
over multiple measurements, its variance, or a sample of a single measurement.

The quantum function from above, for example, used the :func:`expval <pennylane.measure.expval>` measurement:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))

The three measurement functions can be found here:

.. toctree::
    :maxdepth: 1

* :ref:`expval <pennylane_measure_expval>`
* :ref:`sample <pennylane_measure_sample>`
* :ref:`var <pennylane_measure_var>`


.. _intro_ref_temp:

Templates
---------

PennyLane provides a growing library of pre-coded templates of common variational circuit architectures
that can be used to easily build, evaluate, and train more complex models. In the
literature, such architectures are commonly known as an *ansatz*.

.. note::

    Templates are constructed out of **structured combinations** of the :mod:`quantum operations <intro_ref_ops>`
    provided by PennyLane. This means that **template functions can only be used within a
    valid** :mod:`pennylane.qnode.QNode`.

PennyLane conceptually distinguishes two types of templates, :ref:`layer architectures <intro_ref_temp_lay>`
and :ref:`input embeddings <intro_ref_temp_emb>`.
Most templates are complemented by functions that provide an array of
random :ref:`initial parameters <intro_ref_temp_params>` .

An example of how to use templates is the following:

.. code-block:: python

    import pennylane as qml
    from pennylane.templates.embeddings import AngleEmbedding
    from pennylane.templates.layers import StronglyEntanglingLayers
    from pennylane.init import strong_ent_layer_uniform

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(weights, x=None):
        AngleEmbedding(x, [0,1])
        StronglyEntanglingLayers(weights=weights, wires=[0,1])
        return qml.expval(qml.PauliZ(0))

    init_weights = strong_ent_layer_uniform(n_wires=2)
    print(circuit(init_weights, x=[1., 2.]))


Here, we used the embedding template :class:`AngleEmbedding <pennylane.templates.embeddings.AngleEmbedding>`
together with the layer template
:class:`StronglyEntanglingLayers <pennylane.templates.layers.StronglyEntanglingLayers>`,
and the uniform parameter initialization strategy
:func:`strong_ent_layer_uniform <pennylane.init.strong_ent_layer_uniform>`.


.. _intro_ref_temp_lay:

Layer templates
^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.templates.layers

Layer architectures, found in the :ref:`templates.layers <api_qml_temp_lay>` module,
define sequences of gates that are repeated like the layers in a neural network.
They usually contain only trainable parameters.

The following layer templates are available:

.. toctree::
    :maxdepth: 1

* :ref:`CVNeuralNetLayer <pennylane_templates_layers_CVNeuralNetLayer>`
* :ref:`CVNeuralNetLayers <pennylane_templates_layers_CVNeuralNetLayers>`
* :ref:`Interferometer <pennylane_templates_layers_Interferometer>`
* :ref:`RandomLayer <pennylane_templates_layers_RandomLayer>`
* :ref:`RandomLayers <pennylane_templates_layers_RandomLayers>`
* :ref:`StronglyEntanglingLayer <pennylane_templates_layers_StronglyEntanglingLayer>`
* :ref:`StronglyEntanglingLayers <pennylane_templates_layers_StronglyEntanglingLayers>`



.. _intro_ref_temp_emb:

Embedding templates
^^^^^^^^^^^^^^^^^^^

Embeddings, found in the :ref:`templates.embeddings <api_qml_temp_emb>` module,
encode input features into the quantum state of the circuit.
Hence, they take a feature vector as an argument. These embeddings can also depend on
trainable parameters, in which case the embedding is learnable.

The following embedding templates are available:


.. toctree::
    :maxdepth: 1

* :ref:`AmplitudeEmbedding <pennylane_templates_embeddings_AmplitudeEmbedding>`
* :ref:`BasisEmbedding <pennylane_templates_embeddings_BasisEmbedding>`
* :ref:`AngleEmbedding <pennylane_templates_embeddings_AngleEmbedding>`
* :ref:`SqueezingEmbedding <pennylane_templates_embeddings_SqueezingEmbedding>`
* :ref:`DisplacementEmbedding <pennylane_templates_embeddings_DisplacementEmbedding>`

.. _intro_ref_temp_params:

Parameter initializations
^^^^^^^^^^^^^^^^^^^^^^^^^

Each trainable template has a dedicated function in the :ref:`init <api_qml_init>` module, which generates a list of
randomly initialized arrays for the trainable parameters.

Strongly entangling circuit
***************************

.. toctree::
    :maxdepth: 1

* :ref:`strong_ent_layers_uniform <pennylane_init_strong_ent_layers_uniform>`
* :ref:`strong_ent_layers_normal <pennylane_init_strong_ent_layers_normal>`
* :ref:`strong_ent_layer_uniform <pennylane_init_strong_ent_layer_uniform>`
* :ref:`strong_ent_layer_normal <pennylane_init_strong_ent_layer_normal>`

Random circuit
**************

.. toctree::
    :maxdepth: 1

* :ref:`random_layers_uniform <pennylane_init_random_layers_uniform>`
* :ref:`random_layers_normal <pennylane_init_random_layers_normal>`
* :ref:`random_layer_uniform <pennylane_init_random_layer_uniform>`
* :ref:`random_layer_normal <pennylane_init_random_layer_normal>`

Continuous-variable quantum neural network
******************************************

.. toctree::
    :maxdepth: 1

* :ref:`cvqnn_layers_uniform <pennylane_init_cvqnn_layers_uniform>`
* :ref:`cvqnn_layers_normal <pennylane_init_cvqnn_layers_normal>`
* :ref:`cvqnn_layer_uniform <pennylane_init_cvqnn_layer_uniform>`
* :ref:`cvqnn_layer_normal <pennylane_init_cvqnn_layer_normal>`

Interferometer
**************

.. toctree::
    :maxdepth: 1

* :ref:`interferometer_uniform <pennylane_init_interferometer_uniform>`
* :ref:`interferometer_normal <pennylane_init_interferometer_normal>`

.. _intro_ref_opt:

.. currentmodule:: pennylane.optimize

Optimizers
----------

When using the standard NumPy interface, PennyLane offers some custom-made optimizers.
Some of these are specific to quantum optimization, such as the :mod:`QNGOptimizer`.

* :ref:`AdagradOptimizer <pennylane_optimize_AdagradOptimizer>`
* :ref:`AdamOptimizer <pennylane_optimize_AdamOptimizer>`
* :ref:`GradientDescentOptimizer <pennylane_optimize_GradientDescentOptimizer>`
* :ref:`MomentumOptimizer <pennylane_optimize_MomentumOptimizer>`
* :ref:`NesterovMomentumOptimizer <pennylane_optimize_NesterovMomentumOptimizer>`
* :ref:`QNGOptimizer <pennylane_optimize_QNGOptimizer>`
* :ref:`RMSPropOptimizer <pennylane_optimize_RMSPropOptimizer>`


.. warning::

  If using the :ref:`PennyLane PyTorch <torch_interf>`
  or the :ref:`PennyLane TensorFlow <tf_interf>` interfaces,
  `PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ and
  TensorFlow optimizers (found in the module ``tf.train``) should be used respectively.

Configuration
-------------

Some important default settings for a device, such as your user credentials for quantum hardware
access, the number of shots, or the cutoff dimension for continuous-variable simulators, are
defined in a configuration file called `config.toml`.

Behaviour
*********

On first import, PennyLane attempts to load the configuration file by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``PENNYLANE_CONF``
3. The default user configuration directory:

   * On Linux: ``~/.config/pennylane``
   * On Windows: ``~C:\Users\USERNAME\AppData\Local\Xanadu\pennylane``
   * On MacOS: ``~/Library/Preferences/pennylane``

If no configuration file is found, a warning message will be displayed in the logs,
and all device parameters will need to be passed as keyword arguments when
loading the device.

The user can access the initialized configuration via `pennylane.config`, view the
loaded configuration filepath, print the configurations options, access and modify
them via keys (i.e., ``pennylane.config['main.shots']``), and save/load new configuration files.

Format
******

The configuration file `config.toml` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [main]
    # Global PennyLane options.
    # Affects every loaded plugin if applicable.
    shots = 1000
    analytic = True

    [strawberryfields.global]
    # Options for the Strawberry Fields plugin
    hbar = 1
    shots = 100

      [strawberryfields.fock]
      # Options for the Strawberry Fields Fock plugin
      cutoff_dim = 10
      hbar = 0.5

      [strawberryfields.gaussian]
      # Indentation doesn't matter in TOML files,
      # but helps provide clarity.

    [projectq.global]
    # Options for the Project Q plugin

      [projectq.simulator]
      gate_fusion = true

      [projectq.ibm]
      user = "johnsmith"
      password = "secret123"
      use_hardware = true
      device = "ibmqx4"
      num_runs = 1024

Main PennyLane options, that are passed to all loaded devices, are provided under the ``[main]``
section. Alternatively, options can be specified on a per-plugin basis, by setting the options under
``[plugin.global]``.

For example, in the above configuration file, the Strawberry Fields
devices will be loaded with a default of ``shots = 100``, rather than ``shots = 1000``. Finally,
you can also specify settings on a device-by-device basis, by placing the options under the
``[plugin.device]`` settings.

