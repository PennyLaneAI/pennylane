.. _basic_qnode:

Basic QNode
-----------

**Module name:** :mod:`pennylane.qnode`

.. currentmodule:: pennylane

The :class:`~qnode.QNode` class is used to construct quantum nodes,
encapsulating a quantum function or :ref:`variational circuit <varcirc>` and the computational
device it is executed on.


Important QNode methods
***********************

.. currentmodule:: pennylane.qnode.QNode

.. autosummary::
   __init__
   __call__
   evaluate
   evaluate_obs
   jacobian


Code details
^^^^^^^^^^^^

.. automodule:: pennylane.qnode
   :members:
   :private-members:
   :inherited-members:


Examples
^^^^^^^^
.. code-block:: python

        def qfunc(x):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0,1])
            qml.RY(x, wires=1)
            return qml.expval(qml.PauliZ(0))

        qnode = qml.QNode(qfunc, dev)
        result = qnode(0.543)


