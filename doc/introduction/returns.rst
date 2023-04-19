.. _returns:

QNode returns
=============

Version 0.30.0 of PennyLane updated the return type of a :class:`~.pennylane.QNode`. This page
provides additional details about the update and can be used to troubleshoot issues.

.. note::

    If you are looking for a quick fix, jump to the :ref:`Troubleshooting` section!

Summary of the update
---------------------

.. highlights::

    PennyLane QNodes now return exactly what you tell them to! 🎉

Consider the following circuit:

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

Motivation
----------

PennyLane has historically adopted the approach of combining the returned
:ref:`measurements <intro_ref_meas>` of a QNode into a single array. However, this has presented
some challenges:

- The return of a QNode could be different to what is expected, as shown in the example above.
- For measurements of different shapes, ragged arrays were generated internally and then squeezed
  into a single output array. This was incompatible with NumPy's
  `NEP 34 <https://numpy.org/neps/nep-0034-infer-dtype-is-object.html>`__ and constrained the
  `version of NumPy <https://github.com/PennyLaneAI/pennylane/blob/v0.29.1/setup.py#L21>`__ that
  PennyLane was compatible with.
- Use of stacking and squeezing presented performance bottlenecks.

.. _Troubleshooting:

Troubleshooting
---------------

TODO
