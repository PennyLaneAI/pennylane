.. role:: html(raw)
   :format: html

.. _templates_amplitude_embedding:

Amplitude Embedding
===================

Amplitude embedding encodes a normalized :math:`2^n` dimensional feature vector into the state
of :math:`n` qubits:

.. code-block:: python

    import pennylane as qml
    import numpy as np
    from pennylane.templates import AmplitudeEmbedding

    features = np.array([1/2, 1/2, 1/2, 1/2])

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(features_=None):
        AmplitudeEmbedding(features=features_, wires=range(2))
        return qml.expval(qml.PauliZ(0))

    circuit(features_=features)

Checking the final state of the device, we find that it is equivalent to ``features``:

>>> dev._state
[0.5+0.j 0.5+0.j 0.5+0.j 0.5+0.j]

The template will raise an error if the feature input is not normalized. Alternatively,
one can set ``normalize=True`` to automatically normalize it:

.. code-block:: python

    features = np.array([1, 1, 1, 1])

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(features_=None):
        AmplitudeEmbedding(features=features_, wires=range(2), normalize=True)
        return qml.expval(qml.PauliZ(0))

    circuit(features_=features)

Again, the normalized feature vector is encoded into the quantum state vector:

>>> dev._state
[0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.5 + 0.j]

If the dimension of the feature vector is smaller than the number of amplitudes,
one can automatically pad it with a constant for the missing dimensions using the ``pad`` option:


.. code-block:: python

    features = 1/np.sqrt(2)*np.array([1, 1])

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(features_=None):
        AmplitudeEmbedding(features=features_, wires=range(2), pad=0.)
        return qml.expval(qml.PauliZ(0))

    circuit(features_=features)

>>> dev._state
[0.70710678 + 0.j, 0.70710678 + 0.j, 0.0 + 0.j, 0.0 + 0.j]

.. note::

    Padding will be executed *before* normalization.

.. warning::

    On some devices, AmplitudeEmbedding has to be the first operation of a circuit.