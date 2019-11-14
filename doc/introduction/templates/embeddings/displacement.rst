.. role:: html(raw)
   :format: html

.. _templates_displacement_embedding:

Displacement Embedding
======================

Displacement embedding encodes a feature vector into the displacement of quantum modes. As a default,
features are encoded into the displacement *amplitude*, which shifts the expectation of the ``X`` observable
by the value of the feature (times :math:`\hbar`, which is defined and accessible through the device):

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np
    from pennylane.templates.embeddings import DisplacementEmbedding

    dev = qml.device('default.gaussian', wires=2)

    features = np.array([1.2, -0.4]) / dev.hbar

    @qml.qnode(dev)
    def circuit(features_=None):
        DisplacementEmbedding(features=features_, wires=range(2))
        return [qml.expval(qml.X(0)), qml.expval(qml.X(1))]

The result is close to the original features:

>>> circuit(features_=features)
[ 1.194,   -0.398]

[TODO: explain why not exact].

Features can also be encoded into the displacement phase by using ``method = 'phase'``. The ``c`` option
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
        DisplacementEmbedding(features=features_, wires=range(2))
        return qml.expval(qml.X(0))

    o = GradientDescentOptimizer()
    for i in range(10):
        features = o.step(circuit, features)

>>> features
[ 0.401, -0.2]