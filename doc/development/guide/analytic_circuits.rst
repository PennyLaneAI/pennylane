Analytic Circuit Repository
===========================

When possible, we recommend checking circuits against analytic results in ``pytest`` instead of
results computed via a different route in PennyLane.  So you don't need to calculate anything out
on pen and paper, we provide circuits here.  

.. code-block:: python

    import pennylane as qml
    from pennylane import math

Non-Parameterized
-----------------

GHZ state

.. code-block:: python

    qml.Hadamard(wires=0)
    qml.CNOT(wires=(0,1))
    qml.CNOT(wires=(1,2))

State

.. code-block:: python

    state = math.zeros((8,))
    state[0] = 1/math.sqrt(2)
    state[-1] = 1/math.sqrt(2)

================================================== ==========================
Measurement                                              Value
================================================== ==========================
``qml.expval(qml.PauliZ(wire))``                       ``0``
``qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))``          ``1``
================================================== ==========================

Basic circuit
-------------

Operations

.. code-block:: python

    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    qml.CNOT(wires=(0,1))

State

.. code-block:: python

    state = math.array( [[math.cos(x/2)*math.cos(y/2), math.cos(x/2)*math.sin(y/2)],
                    [-1j*math.sin(x/2)*math.sin(y/2), 1j*math.sin(x/2)*math.cos(y/2)]])


================================================== ==========================
Measurement                                              Value
================================================== ==========================
``qml.expval(qml.PauliZ(0))``                       ``math.cos(x)``
``qml.expval(qml.PauliX(1))``                       ``math.sin(y)``
``qml.expval(qml.PauliZ(0) @ qml.PauliX(1))``       ``math.cos(x)*math.sin(y)``
================================================== ==========================


Gate Tests
----------