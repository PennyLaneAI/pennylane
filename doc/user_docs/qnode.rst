.. _qnodes:

Quantum Nodes
=============

:ref:`QNodes <quantum_nodes>` form part of the core structure of PennyLane --- they are used
to encapsulate a quantum function that runs quantum circuits on a hardware device or a simulator backend.

By defining QNodes, either via the :mod:`QNode decorator <pennylane.decorator>`
or the :mod:`QNode class <pennylane.qnode>`, dispatching them to devices, and
combining them with classical processing, it is easy to create arbitrary
classical-quantum hybrid computations.

Every interface - NumPy, Pytorch and Tensorflow - uses a different type of QNode.
The basic or default QNode interfaces with NumPy.

.. toctree::
    :maxdepth: 1

    qnodes/numpy_qnode
    qnodes/torch_qnode
    qnodes/tfe_qnode




