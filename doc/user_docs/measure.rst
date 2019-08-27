Measurement outcomes
====================

**Module name:** :mod:`pennylane.measure`

.. currentmodule:: pennylane.measure

This module contains the functions for computing different types of measurement 
outcomes - expectation values, variances of expectations, and measurement samples of quantum observables.

These are used to indicate to the quantum device how to measure
and return the requested observables. For example, the following
QNode returns the expectation value of observable :class:`~.PauliZ`
on wire 1, and the variance of observable :class:`~.PauliX` on
wire 2.

.. code-block:: python

    import pennylane as qml
    from pennylane import expval, var

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return expval(qml.PauliZ(0)), var(qml.PauliX(1))

Note that *all* returned observables must be within
a measurement function; they cannot be 'bare'.

Types of outcomes
-----------------

.. autosummary::
   expval
   var
   sample

Code details
^^^^^^^^^^^^

.. automodule:: pennylane.measure
   :members: expval, var, sample
