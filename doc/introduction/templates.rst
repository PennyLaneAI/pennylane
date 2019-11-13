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

PennyLane conceptually distinguishes two types of templates, :ref:`layer architectures <intro_ref_temp_lay>`
and :ref:`input embeddings <intro_ref_temp_emb>`.
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
        StronglyEntanglingLayers(*weights, wires=[0,1])
        return qml.expval(qml.PauliZ(0))

    init_weights = strong_ent_layers_uniform(n_layers=3, n_wires=2)
    print(circuit(init_weights, x=[1., 2.]))


Here, we used the embedding template :func:`~.AngleEmbedding`
together with the layer template :func:`~.StronglyEntanglingLayers`,
and the uniform parameter initialization strategy
:func:`~.strong_ent_layers_uniform`.


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
Hence, they take a feature vector as an argument. These embeddings can also depend on
trainable parameters and be repeated.

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

Each trainable template has a dedicated function in the :mod:`pennylane.init` module, which generates a list of
randomly initialized arrays for the trainable parameters.

.. rubric:: Strongly entangling circuit

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.strong_ent_layers_uniform
    ~pennylane.init.strong_ent_layers_normal

:html:`</div>`

.. rubric:: Random circuit

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.random_layers_uniform
    ~pennylane.init.random_layers_normal

:html:`</div>`

.. rubric:: Continuous-variable quantum neural network

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.cvqnn_layers_uniform
    ~pennylane.init.cvqnn_layers_normal

:html:`</div>`

.. rubric:: Interferometer

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.init.interferometer_uniform
    ~pennylane.init.interferometer_normal

:html:`</div>`
