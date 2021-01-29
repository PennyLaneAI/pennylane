qml.tape
========

Quantum tapes are responsible for recording quantum operations, executing devices, or computing
gradients.

In addition to being created internally by QNodes, quantum tapes can also be created,
nested, expanded (via :meth:`~.QuantumTape.expand`), and executed manually. Tape subclasses also provide
additional gradient methods:

Finally, quantum tapes are fully compatible with autodifferentiating via NumPy/Autograd,
TensorFlow, and PyTorch.


.. automodapi:: pennylane.tape
    :no-main-docstr:
    :include-all-objects:

