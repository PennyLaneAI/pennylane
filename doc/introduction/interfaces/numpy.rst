.. _numpy_interf:

NumPy interface
===============

The NumPy interface executes QNodes with standard Python values and NumPy arrays,
and returns NumPy arrays. It does not provide automatic differentiation.

Set ``interface="numpy"`` or ``interface=None`` when creating a :class:`~pennylane.QNode`.
The default ``interface="auto"`` also selects NumPy for standard Python values and NumPy arrays.

.. code-block:: python

    import numpy as np
    import pennylane as qp

    dev = qp.device("default.qubit", wires=1)

    @qp.qnode(dev, interface="numpy", diff_method=None)
    def circuit(angle):
        qp.RX(angle, wires=0)
        return qp.probs(wires=0)

>>> angle = np.array(0.5)
>>> print(circuit(angle))
[0.93879128 0.06120872]

Use :doc:`JAX <jax>` or :doc:`PyTorch <torch>` to differentiate QNodes and train
hybrid quantum-classical models. The ``"autograd"`` interface is no longer supported.
