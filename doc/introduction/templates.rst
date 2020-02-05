.. role:: html(raw)
   :format: html

.. _intro_ref_temp:

Templates
=========

PennyLane provides a growing library of pre-coded templates of common variational circuit architectures
that can be used to easily build, evaluate, and train more complex models. In the
literature, such architectures are commonly known as an *ansatz*.

.. note::

    Templates are constructed out of **structured combinations** of the quantum operations
    provided by PennyLane. This means that **template functions can only be used within a
    valid** :class:`~.QNode`.

PennyLane conceptually distinguishes different types of templates:

* :ref:`layers <intro_ref_temp_lay>` are circuit sequences that depend on trainable parameters and are repeated
  like the layers of a neural network
* :ref:`embeddings <intro_ref_temp_emb>`.
Most templates are complemented by functions that provide an array of
random :ref:`initial parameters <intro_ref_temp_params>` .

An example of how to use templates is the following:

.. code-block:: python

    import pennylane as qml
    from pennylane.templates.embeddings import AngleEmbedding
    from pennylane.templates.layers import StronglyEntanglingLayers
    from pennylane.init import strong_ent_layers_uniform

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(weights, x=None):
        AngleEmbedding(x, [0,1])
        StronglyEntanglingLayers(weights, wires=[0,1])
        return qml.expval(qml.PauliZ(0))

    init_weights = strong_ent_layers_uniform(n_layers=3, n_wires=2)
    print(circuit(init_weights, x=[1., 2.]))


Here, we used the embedding template :func:`~.AngleEmbedding`
together with the layer template :func:`~.StronglyEntanglingLayers`,
and the uniform parameter initialization strategy
:func:`~.strong_ent_layers_uniform`.

Custom templates
----------------

In addition, custom templates can be created; simply
decorate a Python function that applies quantum gates
with the :func:`~.template` decorator:

.. code-block:: python3

    @qml.template
    def bell_state_preparation(wires):
        qml.Hadamard(wires=wires[0])
        qml.CNOT(wires=wires)

This registers the template with PennyLane, making it compatible with
functions that act on templates, such as :func:`~.pennylane.inv`:

.. code-block:: python3

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.inv(bell_state_preparation(wires=[0, 1]))
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


See below for details on built-in templates provided by PennyLane.


.. _intro_ref_temp_lay:

Layer templates
---------------

.. currentmodule:: pennylane.templates.layers

Layer architectures define sequences of trainable gates that are repeated like the layers in a neural network.

The following layer templates are available:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.templates.layers.CVNeuralNetLayers
    ~pennylane.templates.layers.RandomLayers
    ~pennylane.templates.layers.StronglyEntanglingLayers

:html:`</div>`



.. _intro_ref_temp_emb:

Embedding templates
-------------------

Embeddings encode input features into the quantum state of the circuit.
Hence, they take a feature vector as an argument. Embeddings can also depend on
trainable parameters, and they may consist of repeated layers.

The following embedding templates are available:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.templates.embeddings.AmplitudeEmbedding
    ~pennylane.templates.embeddings.AngleEmbedding
    ~pennylane.templates.embeddings.BasisEmbedding
    ~pennylane.templates.embeddings.DisplacementEmbedding
    ~pennylane.templates.embeddings.QAOAEmbedding
    ~pennylane.templates.embeddings.SqueezingEmbedding


:html:`</div>`

.. _intro_ref_temp_params:

Subroutines
-----------

Subroutines are simply a collection of (trainable) gates.

The following subroutines are available:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.templates.Interferometer

:html:`</div>`


Parameter initializations
-------------------------

Each trainable template has dedicated functions in the :mod:`pennylane.init` module, which generate
randomly initialized arrays for the trainable parameters. For example, :func:`random_layers_uniform` can
be used together with the template :func:`RandomLayers`:

.. code-block:: python

    import pennylane as qml
    from pennylane.templates import RandomLayers
    from pennylane.init import random_layers_uniform

    dev = qml.device('default.qubit', wires=3)

    @qml.qnode(dev)
    def circuit(weights):
        RandomLayers(weights=weights, wires=[0, 2])
        return qml.expval(qml.PauliZ(0))

    init_pars = random_layers_uniform(n_layers=3, n_wires=2)
    circuit(init_pars)

Templates that take more than one parameter
array require several initialization functions:

.. code-block:: python

    from pennylane.templates import Interferometer
    from pennylane.init import (interferometer_theta_uniform,
                                interferometer_phi_uniform,
                                interferometer_varphi_normal)

    dev = qml.device('default.gaussian', wires=3)

    @qml.qnode(dev)
    def circuit(theta, phi, varphi):
        Interferometer(theta=theta, phi=phi, varphi=varphi, wires=[0, 2])
        return qml.expval(qml.X(0))

    init_theta = interferometer_theta_uniform(n_wires=2)
    init_phi = interferometer_phi_uniform(n_wires=2)
    init_varphi = interferometer_varphi_normal(n_wires=2)

    circuit(init_theta, init_phi, init_varphi)


For templates with multiple parameters, initializations that
return a list of all parameter arrays at once are provided, and can
be conveniently used in conjunction with the unpacking operator ``*``:

.. code-block:: python

    from pennylane.templates import Interferometer
    from pennylane.init import interferometer_all

    dev = qml.device('default.gaussian', wires=3)

    @qml.qnode(dev)
    def circuit(*pars):
        Interferometer(*pars, wires=[0, 2])
        return qml.expval(qml.X(0))

    init_pars = interferometer_all(n_wires=2)

    circuit(*init_pars)

Initial parameters can be converted to Torch or TensorFlow tensors, which can be used in the
respective interfaces.

.. code-block:: python

    import torch
    import tensorflow as tf
    from pennylane.init import strong_ent_layers_normal

    init_pars = strong_ent_layers_normal(n_layers=3, n_wires=2)
    init_torch = torch.tensor(init_pars)
    init_tf = tf.Variable(init_pars)

The following initialization functions are available:

.. rubric:: Continuous-variable quantum neural network

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.cvqnn_layers_all
    ~pennylane.init.cvqnn_layers_theta_uniform
    ~pennylane.init.cvqnn_layers_theta_normal
    ~pennylane.init.cvqnn_layers_phi_uniform
    ~pennylane.init.cvqnn_layers_phi_normal
    ~pennylane.init.cvqnn_layers_varphi_uniform
    ~pennylane.init.cvqnn_layers_varphi_normal
    ~pennylane.init.cvqnn_layers_r_uniform
    ~pennylane.init.cvqnn_layers_r_normal
    ~pennylane.init.cvqnn_layers_phi_r_uniform
    ~pennylane.init.cvqnn_layers_phi_r_normal
    ~pennylane.init.cvqnn_layers_a_uniform
    ~pennylane.init.cvqnn_layers_a_normal
    ~pennylane.init.cvqnn_layers_phi_a_uniform
    ~pennylane.init.cvqnn_layers_phi_a_normal
    ~pennylane.init.cvqnn_layers_kappa_uniform
    ~pennylane.init.cvqnn_layers_kappa_normal


:html:`</div>`

.. rubric:: Interferometer

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.interferometer_all
    ~pennylane.init.interferometer_theta_uniform
    ~pennylane.init.interferometer_theta_normal
    ~pennylane.init.interferometer_phi_uniform
    ~pennylane.init.interferometer_phi_normal
    ~pennylane.init.interferometer_varphi_uniform
    ~pennylane.init.interferometer_varphi_normal

:html:`</div>`

.. rubric:: QAOA embedding

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.qaoa_embedding_uniform
    ~pennylane.init.qaoa_embedding_normal

:html:`</div>`

.. rubric:: Random layers

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.random_layers_uniform
    ~pennylane.init.random_layers_normal

:html:`</div>`

.. rubric:: Strongly entangling layers

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.strong_ent_layers_uniform
    ~pennylane.init.strong_ent_layers_normal

:html:`</div>`

