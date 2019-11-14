.. role:: html(raw)
   :format: html

.. _templates_angle_embedding:

Angle Embedding
===============

Angle embedding encodes features into the angles of qubit rotation gates. Consequently, the angle of the
Bloch vector of each qubit represents a feature, and the qubits remain unentangled.

As a default, ``pennylane.ops.RX`` rotations are used:

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np
    from pennylane.templates.embeddings import AngleEmbedding

    features = np.array([1., np.pi/2])

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(features_=None):
        AngleEmbedding(features=features_, wires=range(2))
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

The second feature of :math:`\pi/2` rotates the qubit to the state of a uniform superposition,
whith expectation exactly at zero:

>>> circuit(features_=features)
[0.5403, 0.0]

Alternatively, one can use the ``rotation`` argument to choose ``pennylane.ops.RY`` or ``pennylane.ops.RZ`` rotations.

The features can be made trainable if used as a positional argument:

.. code-block:: python

    from pennylane.optimize import GradientDescentOptimizer

    dev = qml.device('default.qubit', wires=2)
    features = np.array([1., np.pi / 2])


    @qml.qnode(dev)
    def circuit(features_):
        AngleEmbedding(features=features_, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    o = GradientDescentOptimizer()
    for i in range(10):
        features = o.step(circuit, features)

>>> features
[1.0861, 1.5707]

.. note::

    This embedding method can also be used to encode to imitate :ref:`BasisEmbedding <templates_basis_embedding>`.
    For example, to prepare basis state :math:`|0,1,1,0\rangle`, choose ``rotation='X'``
    and use the feature vector :math:`[0,\pi/2,\pi/2,0]`.

