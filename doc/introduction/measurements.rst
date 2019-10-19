.. role:: html(raw)
   :format: html

.. _intro_ref_meas:

Measurements
============

.. currentmodule:: pennylane.measure

PennyLane can extract different types of measurement results: The expectation of an observable
over multiple measurements, its variance, or a sample of a single measurement.

For example, the quantum function shown in the previous section
used the :func:`expval <pennylane.expval>` measurement:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))

The three measurement functions can be found here:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.expval
    ~pennylane.sample
    ~pennylane.var

:html:`</div>`

