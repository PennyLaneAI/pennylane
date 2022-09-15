qml.tape
========

Quantum tapes are a datastructure that can represent quantum circuits and measurement statistics in PennyLane. They are queuing contexts that can record and process quantum operations and measurements.

In addition to being created internally by QNodes, quantum tapes can also be created,
nested, expanded (via :meth:`~.QuantumTape.expand`), and executed manually.

Finally, quantum tapes are fully compatible with autodifferentiating via NumPy/Autograd,
TensorFlow, and PyTorch.

.. warning::

    Unless you are a PennyLane or plugin developer, you likely do not need
    to use these classes directly.

    See the :doc:`quantum circuits <../introduction/circuits>` page for more
    details on creating QNodes, as well as the :func:`~pennylane.qnode` decorator
    and :func:`~pennylane.QNode` constructor.

.. automodapi:: pennylane.tape
    :no-main-docstr:
    :include-all-objects:
    :skip: queuing_stop_recording, warn
