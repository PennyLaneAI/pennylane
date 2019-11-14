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


There are different types of templates, such as :ref:`layer architectures <intro_ref_temp_lay>`
and :ref:`input embeddings <intro_ref_temp_emb>`. Templates that take trainable parameters are
complemented by functions that provide an array of random :ref:`initial parameters <intro_ref_temp_params>` .

An example of how to use templates is the following:

.. code-block:: python

    import pennylane as qml
    from pennylane.templates import AngleEmbedding
    from pennylane.templates import StronglyEntanglingLayers
    from pennylane.init import strong_ent_layers_uniform

    dev = qml.device('default.qubit', wires=3)

    @qml.qnode(dev)
    def circuit(weights, x=None):
        AngleEmbedding(x, wires=[0, 1, 2])
        StronglyEntanglingLayers(weights, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    init_weights = strong_ent_layers_uniform(n_layers=3, n_wires=3)
    print(circuit(init_weights, x=[0.2, 1.2, -1.1]))


Here, we used the embedding template :func:`~.AngleEmbedding`
together with the layer template :func:`~.StronglyEntanglingLayers`,
and the uniform parameter initialization function :func:`~.strong_ent_layers_uniform`.


.. _intro_ref_temp_lay:

Layer templates
---------------

.. currentmodule:: pennylane.templates.layers

Layer architectures define sequences of trainable gates that are repeated like the layers in a neural network.

The following layer templates are provided:

.. toctree::
   :maxdepth: 2

   templates/layers/random
   templates/layers/strongly_entangling
   templates/layers/cvqnn


.. _intro_ref_temp_emb:

Embedding templates
-------------------

Embeddings encode input features into the quantum state of the circuit.
Hence, they take a feature vector as an argument. These embeddings can also depend on
trainable parameters and be repeated.

The following embedding templates are available:

.. toctree::
   :maxdepth: 2

   templates/embeddings/amplitude
   templates/embeddings/angle
   templates/embeddings/basis
   templates/embeddings/displacement
   templates/embeddings/squeezing

.. _intro_ref_temp_params:

Subroutines
-----------

Subroutines are simply a collection of (trainable) gates.

The following subroutine templates are available:

.. toctree::
   :maxdepth: 2

   templates/subroutines/interferometer


State Preparations
------------------

State Preparation templates are special subroutines that have the purpose of preparing a particular
state. They may depend on classical inputs.

Embeddings and state preparations are closely related: One and the same embedding can be realized - and may
therefore call - different state preparation routines. Furthermore, state preparations are typically not
trainable.

PennyLane provides the following state preparation templates:

.. toctree::
   :maxdepth: 2

   templates/state_preparation/basis_state
   templates/state_preparation/motonnen


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
