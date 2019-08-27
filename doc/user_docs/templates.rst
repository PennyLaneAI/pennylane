.. _templates:

Templates
=========

**Module name:** :mod:`pennylane.templates`

.. currentmodule:: pennylane.templates

This module provides a growing library of templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models. In the
quantum machine learning literature, such architectures are commonly known as an
**ansatz**.

.. note::

    Templates are constructed out of **structured combinations** of the :mod:`quantum operations <pennylane.ops>` provided by PennyLane. This means that **template functions can only be used within a valid** :mod:`pennylane.qnode`.

PennyLane conceptually distinguishes two types of templates, **layer architectures** and **input embeddings**.

Layer templates
---------------

Layer architectures, found in :mod:`pennylane.templates.layers`, define sequences of gates that are repeated like the layers in a neural network. They usually contain only trainable parameters.

The following layer templates are available:

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    templates/layers

Embedding templates
-------------------

Embeddings, found in :mod:`pennylane.templates.embeddings`, encode input features into the quantum state of the circuit. Hence, they take a feature vector as an argument. These embeddings can also depend on trainable parameters, in which case the embedding is learnable.

The following embedding templates are available:

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    templates/embeddings

Parameter initialization
------------------------

Each trainable template has a dedicated function in :mod:`pennylane.init` which generates a list of
**randomly initialized** arrays for the trainable **parameters**.

The following parameter initialization functions are available:

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 2

    templates/init_parameters

.. automodule:: pennylane.templates
   :members:
   :private-members:
   :inherited-members: