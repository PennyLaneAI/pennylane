qml.tape
========

Quantum tapes are a datastructure that can represent quantum circuits and measurement statistics in PennyLane. They are queuing contexts that can record and process quantum operations and measurements.

In addition to being created internally by QNodes, quantum tapes can also be created,
nested, expanded (via :meth:`~.QuantumTape.expand`), and executed manually.

Finally, quantum tapes are fully compatible with autodifferentiating via Autograd, JAX, 
TensorFlow, and PyTorch.

.. warning::

    Unless you are a PennyLane or plugin developer, you likely do not need
    to use these classes directly.

    See the :doc:`quantum circuits <../introduction/circuits>` page for more
    details on creating QNodes, as well as the :func:`~pennylane.qnode` decorator
    and :func:`~pennylane.QNode` constructor.

QuantumTape versus QuantumScript
--------------------------------

A ``QuantumScript`` is purely a representation of a quantum circuit, and can only be constructed
via initialization. Once it is initialized, the contents should then remain immutable throughout its lifetime.

>>> ops = [qml.PauliX(0)]
>>> measurements = [qml.expval(qml.PauliZ(0))]
>>> QuantumScript(ops, measurements, shots=10)
<QuantumScript: wires=[0], params=0>

A ``QuantumTape`` has additional queuing capabilities and also inherits from :class:`~.AnnotatedQueue`.  Its contents
are set on exiting the context, rather than upon initialization. Since queuing requires interaction with the global singleton
:class:`~pennylane.QueuingManager`, the ``QuantumTape`` requires a ``threading.RLock`` which complicates its use in distributed
situations.

>>> with QuantumTape(shots=10) as tape:
...     qml.PauliX(0)
...     qml.expval(qml.PauliZ(0))
>>> tape
<QuantumTape: wires=[0], params=0>

The ``QuantumTape`` also carries around the unprocessed queue in addition to the processed ``operations`` and ``measurements``, 
to a larger memory footprint.

>>> tape.items()
((PauliX(wires=[0]), {}), (expval(PauliZ(wires=[0])), {}))

To capture a quantum function, we instead recommend queuing a quantum function into an :class:`~.AnnotatedQueue`,
and then processing that into a ``QuantumScript``.

>>> with qml.queuing.AnnotatedQueue() as q:
...     qfunc(*args, **kwargs)
>>> QuantumScript.from_queue(q)
<QuantumScript: wires=[0], params=0> 

Since queuing is also sensitive to the "identity" of an operation, not just its contents, an operation has to be copied in
order for it to be used multiple times in a ``QuantumTape``.  A ``QuantumScript`` can allow the same operation to be
used many times in the circuit, potentially reducing its memory footprint.

>>> op = qml.T(0)
>>> QuantumScript([op] * 100, [qml.probs(wires=0)])

Since users are familiar with the term ``QuantumTape``, that term should be used in documentation. For performance
and a reduction in unintended side effects, ``QuantumScript`` is strictly used in PennyLane source code.

.. automodapi:: pennylane.tape
    :no-main-docstr:
    :include-all-objects:
    :skip: QuantumTapeBatch
    :inheritance-diagram: