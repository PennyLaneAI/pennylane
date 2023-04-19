.. _returns:

QNode returns
=============

Version 0.30.0 of PennyLane updated the return type of a :class:`~.pennylane.QNode`. Consider the
following circuit:

.. code-block:: python

    import pennylane as qml

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0)), qml.probs(0)

In version 0.29 and earlier of PennyLane, ``circuit()`` would return a single length-3 array:

.. code-block:: pycon

    >>> circuit(0.5)
    tensor([0.87758256, 0.93879128, 0.06120872], requires_grad=True)

In versions 0.30 and above, ``circuit()`` returns a length-2 tuple containing the expectation value
and probabilities separately:

.. code-block:: pycon

    >>> circuit(0.5)
    (tensor(0.87758256, requires_grad=True),
     tensor([0.93879128, 0.06120872], requires_grad=True))
