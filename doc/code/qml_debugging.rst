.. role:: html(raw)
   :format: html

qml.debugging
=============

This module contains functionality for debugging quantum programs on simulator devices.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.breakpoint
    ~pennylane.debug_expval
    ~pennylane.debug_probs
    ~pennylane.debug_state
    ~pennylane.debug_tape
    ~pennylane.snapshots

:html:`</div>`


Entering the Debugging Context
------------------------------

The :func:`~pennylane.breakpoint` function provides an interface for interacting with and
stepping through a quantum circuit during execution. It allows for faster debugging
by providing access to the internal state of the circuit and the ``QuantumTape`` as 
the circuit operations are applied. The functionality is highlighted by the example 
circuit below.

.. code-block:: python
    :linenos:

    import pennylane as qp
    
    @qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
    def circuit(x):
        qml.breakpoint()

        qml.Hadamard(wires=0)
        qml.CNOT(wires=(0,2))

        for w in (0, 1, 2):
            qml.RX(2*x, wires=w)

        qml.breakpoint()
        qml.RX(-x, wires=1)
        return qml.sample()

    circuit(1.2345)

Running the above python script opens up the interactive ``[pldb]`` prompt in the terminal.
When this code reaches ``qml.breakpoint()`` it will pause and launch an interactive
debugging prompt. The prompt specifies the path to the script and the next line to be 
executed after the breakpoint:

.. code-block:: console

    > /Users/your/path/to/script.py(7)circuit()
    -> qml.Hadamard(wires=0)
    [pldb]

Controlling Code Execution in the Debugging Context
---------------------------------------------------

The Pennylane Debugger (PLDB) is built on top of the native python debugger (PDB). As such, 
it shares a similar interface. We can interact with the debugger using the 
built-in commands such as ``list``, ``longlist``, ``next``, ``continue``, and ``quit``. Any 
variables defined within the scope of the quantum function can also be accessed from the 
debugger.

.. code-block:: console

    [pldb] print(x)
    1.2345

The ``list`` (and ``longlist``) command will print a section of code around the 
breakpoint, highlighting the next line to be executed. This can be used to determine
the location of the execution in the circuit.

.. code-block:: console

    [pldb] longlist
      3  	@qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
      4  	def circuit(x):
      5  	    qml.breakpoint()
      6
      7  ->	    qml.Hadamard(wires=0)
      8  	    qml.CNOT(wires=(0,2))
      9
     10  	    for w in (0, 1, 2):
     11  	        qml.RX(2*x, wires=w)
     12
     13  	    qml.breakpoint()
     14  	    qml.RX(-x, wires=1)
     15  	    return qml.sample()
    
The ``next`` command will execute the next line of code, and print the following 
line to be executed. In this example, the next operation to execute is the ``CNOT``.

.. code-block:: console
    
    [pldb] next
    > /Users/your/path/to/script.py(8)circuit()
    -> qml.CNOT(wires=(0,2))
    [pldb] list
      3  	@qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
      4  	def circuit(x):
      5  	    qml.breakpoint()
      6
      7  	    qml.Hadamard(wires=0)
      8  ->	    qml.CNOT(wires=(0,2))
      9
     10  	    for w in (0, 1, 2):
     11  	        qml.RX(2*x, wires=w)
     12
     13  	    qml.breakpoint()

Alternatively, the ``continue`` command allows for jumping between breakpoints. This command resumes
code execution until the next breakpoint is reached, or termination if there is none. Finally, 
the ``quit`` command ends the debugging prompt and terminates the execution altogether.

.. code-block:: console

    [pldb] continue
    > /Users/your/path/to/script.py(14)circuit()
    -> qml.RX(-x, wires=1)
    [pldb] list
      9
     10  	    for w in (0, 1, 2):
     11  	        qml.RX(2*x, wires=w)
     12
     13  	    qml.breakpoint()
     14  ->	    qml.RX(-x, wires=1)
     15  	    return qml.sample()
     16
     17  	circuit(1.2345)
    [EOF]
    [pldb] quit


Extracting Circuit Information
------------------------------

While in the debugging prompt, we can extract information about the current contents
of the quantum tape using :func:`~pennylane.debug_tape`. We can also perform measurements dynamically
on the quantum circuit using :func:`~pennylane.debug_expval`, :func:`~pennylane.debug_state`, 
and :func:`~pennylane.debug_probs`. 

Consider the circuit from above, 

.. code-block:: console

    > /Users/your/path/to/script.py(7)circuit()
    -> qml.Hadamard(wires=0)
    [pldb] longlist
      3  	@qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
      4  	def circuit(x):
      5  	    qml.breakpoint()
      6
      7  ->	    qml.Hadamard(wires=0)
      8  	    qml.CNOT(wires=(0,2))
      9
     10  	    for w in (0, 1, 2):
     11  	        qml.RX(2*x, wires=w)
     12
     13  	    qml.breakpoint()
     14  	    qml.RX(-x, wires=1)
     15  	    return qml.sample()
    [pldb] next
    > /Users/your/path/to/script.py(8)circuit()
    -> qml.CNOT(wires=(0,2))
    [pldb] next
    > /Users/your/path/to/script.py(10)circuit()
    -> for w in (0, 1, 2):
    [pldb]

All of the operations applied so far are tracked in the circuit's ``QuantumTape`` 
which is accessible using :func:`~pennylane.debug_tape`. This can be used to
*visually* debug the circuit.

.. code-block:: console

    [pldb] qtape = qml.debug_tape()
    [pldb] qtape.operations
    [Hadamard(wires=[0]), CNOT(wires=[0, 2])]
    [pldb] print(qtape.draw())
    0: ──H─╭●─┤
    2: ────╰X─┤

The quantum state of the circuit at this point can be extracted using 
:func:`~pennylane.debug_state`. The associated probability distribution 
for the wires of interest can be probed using :func:`~pennylane.debug_probs`.

.. code-block:: console

    [pldb] qml.debug_state()
    array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
           0.        +0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j])
    [pldb] qml.debug_probs(wires=(0,2))
    array([0.5, 0. , 0. , 0.5])

Another method for probing the system is by measuring observables via 
:func:`~pennylane.debug_expval`.

.. code-block:: console

    [pldb] qml.debug_expval(qml.Z(0))
    0.0
    [pldb] qml.debug_expval(qml.X(0) @ qml.X(2))
    0.9999999999999996

Additionally, the quantum circuit can be dynamically updated by adding gates directly
from the prompt. This allows users to modify the circuit *on-the-fly*!

.. code-block:: console

    [pldb] continue
    > /Users/your/path/to/script.py(14)circuit()
    -> qml.RX(-x, wires=1)
    [pldb] qtape = qml.debug_tape()
    [pldb] print(qtape.draw(wire_order=(0,1,2)))
    0: ──H─╭●──RX─┤
    1: ────│───RX─┤
    2: ────╰X──RX─┤
    [pldb] qml.RZ(0.5*x, wires=0)
    RZ(0.61725, wires=[0])
    [pldb] qml.CZ(wires=(1,2))
    CZ(wires=[1, 2])
    [pldb] qtape = qml.debug_tape()
    [pldb] print(qtape.draw(wire_order=(0,1,2)))
    0: ──H─╭●──RX──RZ─┤
    1: ────│───RX─╭●──┤
    2: ────╰X──RX─╰Z──┤
