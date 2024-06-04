qml.debugging
=============

.. automodapi:: pennylane.debugging
    :no-heading:
    :no-inherited-members:
    :skip: _Debugger
    :skip: contextmanager
    :skip: _measure
    :skip: PLDB
    :skip: pldb_device_manager

Interactive Debugging with breakpoints
--------------------------------------

The ``qml.breakpoint()`` function provides an interface for interacting with and
stepping through a quantum circuit during execution. It allows for faster debugging
by provding access to the internal state of the circuit and the ``QuantumTape`` as 
the circuit operations are applied. 

Consider the following circuit, 

.. code-block:: python

    @qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
    def circuit(x):   
        # Prepare equal super-position over wires (0,1):
        qml.breakpoint()
        qml.Hadamard(wires=0)
        
        # Entangle wires (1,2):
        qml.CNOT(wires=(0,2))

        # Apply a rotation of angle 2x to each wire:
        for w in (0, 1, 2):
            qml.RX(2*x, wires=w)

        # Reset wire 0: 
        qml.breakpoint()
        qml.Hadamard(wires=0)

        return qml.sample()

    circuit(1.2345)

While executing this circuit once a `qml.breakpoint()` is encountered, an interactive
debugging prompt is launched. The prompt specifies the path to the script along with 
the next line to be executed after the breakpoint.

.. code-block:: console

    > /Users/your/path/to/script.py(5)circuit()
    -> qml.Hadamard(wires=0)
    [pldb]:

We can interact with the prompt using the commands: ``list`` , ``next``, 
``continue``, and ``quit``. Additionally, we can also access any variables 
defined in the function.


