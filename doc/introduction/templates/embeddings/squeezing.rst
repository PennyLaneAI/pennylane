.. role:: html(raw)
   :format: html

.. _templates_squeezing_embedding:

Squeezing Embedding
===================

Squeezing embedding encodes a feature vector into the squeezing of quantum modes. As a default,
features are encoded into the squeezing *amplitude*, distorting the variance of the ``P`` and ``X``
expectations and leaving the modes unentangled:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np
    from pennylane.templates.embeddings import SqueezingEmbedding

    dev = qml.device('default.gaussian', wires=2)

    features = np.array([1, 1]) / dev.hbar

    @qml.qnode(dev)
    def circuit(features_=None):
        SqueezingEmbedding(features=features_, wires=range(2))
        return [qml.var(qml.P(0)), qml.var(qml.X(1))]

In this example, the variance of the expectation of the ``P`` observable is increased, while the variance
of ``X`` is decreased.

>>> circuit(features_=features)
[2.712 0.373]

For the initial vacuum state these two are equal.

Features can also be encoded into the squeezing phase by using ``method = 'phase'``. The ``c`` option
allows setting either the phase or amplitude (whichever is not used for feature encoding)
to be set to a constant value.

.. note::

    The constant ``c`` can be an interesting hyperparameter to a kernel, see for example
    `arXiv:1803.07128 <https://arxiv.org/abs/1803.07128>`_

The features can also be trained by defining and using them as a positional argument to the quantum node,
``circuit(features)``.

.. code-block:: python

    from pennylane.optimize import GradientDescentOptimizer

    dev = qml.device('default.gaussian', wires=2)

    features = np.array([1.2, -0.4]) / dev.hbar

    @qml.qnode(dev)
    def circuit(features_):
        SqueezingEmbedding(features=features_, wires=range(2))
        return qml.expval(qml.X(0))

    o = GradientDescentOptimizer()
    for i in range(10):
        features = o.step(circuit, features)

>>> features
[0.6, -0.2]