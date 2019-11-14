.. role:: html(raw)
   :format: html

.. _templates_basis_embedding:

Basis Embedding
===============

Basis embedding translates a binary sequence to the basis state of a quantum system.
For example :math:`010` gets mapped to :math:`|010\rangle`.

.. code-block:: python

    import pennylane as qml
    import numpy as np
    from pennylane.templates.embeddings import BasisEmbedding

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(features_=None):
        BasisEmbedding(features=features_, wires=range(2))
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

The ``PauliZ`` expectations evaluate to :math:`-1` and :math:`1`, depending on the bit string.

>>> circuit(features_=np.array([1, 0]))
[-1., 1.]

>>> circuit(features_=np.array([1, 1]))
[-1., -1.]

The bitstring is a discrete feature vector and does therefore not have a continuous gradient.
PennyLane therefore raises an error if features are passed to a quantum node via positional arguments, like
in ``circuit(features_)``.

.. warning::

    On some devices, BasisEmbedding has to be the first operation of a circuit.