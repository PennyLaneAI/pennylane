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
first example from this section, we find the :class:`RZ <pennylane.RZ>`,
:class:`CNOT <pennylane.CNOT>`,
:class:`RY <pennylane.RY>` :ref:`gates <intro_ref_ops_qgates>` and the
:class:`PauliZ <pennylane.PauliZ>` :ref:`observable <intro_ref_ops_qobs>`:

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

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.CNOT
    ~pennylane.CRot
    ~pennylane.CRX
    ~pennylane.CRY
    ~pennylane.CRZ
    ~pennylane.CSWAP
    ~pennylane.CZ
    ~pennylane.Hadamard
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ
    ~pennylane.PhaseShift
    ~pennylane.QubitUnitary
    ~pennylane.Rot
    ~pennylane.RX
    ~pennylane.RY
    ~pennylane.RZ
    ~pennylane.S
    ~pennylane.SWAP
    ~pennylane.T

:html:`</div>`


Qubit state preparation
***********************


:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.BasisState
    ~pennylane.QubitStateVector

:html:`</div>`


.. _intro_ref_ops_qobs:

Qubit observables
*****************

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Hadamard
    ~pennylane.Hermitian
    ~pennylane.PauliX
    ~pennylane.PauliY
    ~pennylane.PauliZ

:html:`</div>`

.. _intro_ref_ops_cv:

Continuous-variable (CV) operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _intro_ref_ops_cvgates:

CV Gates
********

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Beamsplitter
    ~pennylane.ControlledAddition
    ~pennylane.ControlledPhase
    ~pennylane.CrossKerr
    ~pennylane.CubicPhase
    ~pennylane.Displacement
    ~pennylane.Interferometer
    ~pennylane.Kerr
    ~pennylane.QuadraticPhase
    ~pennylane.Rotation
    ~pennylane.Squeezing
    ~pennylane.TwoModeSqueezing

:html:`</div>`


CV state preparation
********************

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.CatState
    ~pennylane.CoherentState
    ~pennylane.DisplacedSqueezedState
    ~pennylane.FockDensityMatrix
    ~pennylane.FockState
    ~pennylane.FockStateVector
    ~pennylane.GaussianState
    ~pennylane.SqueezedState
    ~pennylane.ThermalState

:html:`</div>`

.. _intro_ref_ops_cvobs:

CV observables
**************

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.FockStateProjector
    ~pennylane.NumberOperator
    ~pennylane.P
    ~pennylane.PolyXP
    ~pennylane.QuadOperator
    ~pennylane.X

:html:`</div>`

Shared operations
^^^^^^^^^^^^^^^^^

The only operation shared by both qubit and continouous-variable architectures is the Identity.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.Identity

:html:`</div>`


.. _intro_ref_meas:

Measurements
------------

.. currentmodule:: pennylane.measure

PennyLane can extract different types of measurement results: The expectation of an observable
over multiple measurements, its variance, or a sample of a single measurement.

The quantum function from above, for example, used the :func:`expval <pennylane.expval>` measurement:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))

The three measurement functions can be found here:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.expval
    ~pennylane.sample
    ~pennylane.var

:html:`</div>`


.. _intro_ref_temp:

Templates
---------

PennyLane provides a growing library of pre-coded templates of common variational circuit architectures
that can be used to easily build, evaluate, and train more complex models. In the
literature, such architectures are commonly known as an *ansatz*.

.. note::

    Templates are constructed out of **structured combinations** of the quantum operations
    provided by PennyLane. This means that **template functions can only be used within a
    valid** :class:`~.QNode`.

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


Here, we used the embedding template :func:`~.AngleEmbedding`
together with the layer template :func:`~.StronglyEntanglingLayers`,
and the uniform parameter initialization strategy
:func:`~.strong_ent_layer_uniform`.


.. _intro_ref_temp_lay:

Layer templates
^^^^^^^^^^^^^^^

.. currentmodule:: pennylane.templates.layers

Layer architectures, found in the :mod:`pennylane.templates.layers` module,
define sequences of gates that are repeated like the layers in a neural network.
They usually contain only trainable parameters.

The following layer templates are available:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.templates.layers.CVNeuralNetLayer
    ~pennylane.templates.layers.CVNeuralNetLayers
    ~pennylane.templates.layers.Interferometer
    ~pennylane.templates.layers.RandomLayer
    ~pennylane.templates.layers.RandomLayers
    ~pennylane.templates.layers.StronglyEntanglingLayer
    ~pennylane.templates.layers.StronglyEntanglingLayers

:html:`</div>`



.. _intro_ref_temp_emb:

Embedding templates
^^^^^^^^^^^^^^^^^^^

Embeddings, found in the :ref:`templates.embeddings <api_qml_temp_emb>` module,
encode input features into the quantum state of the circuit.
Hence, they take a feature vector as an argument. These embeddings can also depend on
trainable parameters, in which case the embedding is learnable.

The following embedding templates are available:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.templates.embeddings.AmplitudeEmbedding
    ~pennylane.templates.embeddings.BasisEmbedding
    ~pennylane.templates.embeddings.AngleEmbedding
    ~pennylane.templates.embeddings.SqueezingEmbedding
    ~pennylane.templates.embeddings.DisplacementEmbedding

:html:`</div>`

.. _intro_ref_temp_params:

Parameter initializations
^^^^^^^^^^^^^^^^^^^^^^^^^

Each trainable template has a dedicated function in the :ref:`init <api_qml_init>` module, which generates a list of
randomly initialized arrays for the trainable parameters.

Strongly entangling circuit
***************************

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.strong_ent_layers_uniform
    ~pennylane.init.strong_ent_layers_normal
    ~pennylane.init.strong_ent_layer_uniform
    ~pennylane.init.strong_ent_layer_normal

:html:`</div>`

Random circuit
**************

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.random_layers_uniform
    ~pennylane.init.random_layers_normal
    ~pennylane.init.random_layer_uniform
    ~pennylane.init.random_layer_normal

:html:`</div>`

Continuous-variable quantum neural network
******************************************

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.cvqnn_layers_uniform
    ~pennylane.init.cvqnn_layers_normal
    ~pennylane.init.cvqnn_layer_uniform
    ~pennylane.init.cvqnn_layer_normal

:html:`</div>`

Interferometer
**************

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.interferometer_uniform
    ~pennylane.init.interferometer_normal

:html:`</div>`

.. _intro_ref_opt:

Optimizers
----------

When using the standard NumPy interface, PennyLane offers some custom-made optimizers.
Some of these are specific to quantum optimization, such as the :mod:`~.QNGOptimizer`.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.AdagradOptimizer
    ~pennylane.AdamOptimizer
    ~pennylane.GradientDescentOptimizer
    ~pennylane.MomentumOptimizer
    ~pennylane.NesterovMomentumOptimizer
    ~pennylane.QNGOptimizer
    ~pennylane.RMSPropOptimizer

:html:`</div>`


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
^^^^^^^^^

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
^^^^^^

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

