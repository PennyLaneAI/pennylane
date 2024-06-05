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
the circuit operations are applied. The functionality is highlighted by debugging 
and example circuit.

Consider the following circuit, 

.. code-block:: python

    import pennylane as qml
    
    @qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
    def circuit(x):   
        # 1) Prepare equal probabilities over wires (0,1):
        qml.breakpoint()
        qml.Hadamard(wires=0)
        
        # 2) Entangle wires (1,2):
        qml.CNOT(wires=(0,2))

        # 3) Apply a rotation of angle 2x to each wire:
        for w in (0, 1, 2):
            qml.RX(2*x, wires=w)

        # 4) Reset wire 1: 
        qml.breakpoint()
        qml.RX(-x, wires=1)

        return qml.sample()

    circuit(1.2345)

While executing this circuit once a `qml.breakpoint()` is encountered, an interactive
debugging prompt is launched. The prompt specifies the path to the script along with
the next line to be executed after the breakpoint.

.. code-block:: console

    > /Users/your/path/to/script.py(7)circuit()
    -> qml.Hadamard(wires=0)
    [pldb]:

We can interact with the prompt using the builtin commands: ``list`` , ``next``, 
``continue``, and ``quit``. Furthermore, we can perform measurements and extract
information about operations or the state of the circuit with 
:func:`~pennylane.debugging.expval`, :func:`~pennylane.debugging.state`, 
:func:`~pennylane.debugging.probs`, and :func:`~pennylane.debugging.tape`. We can 
also access any variables defined in the function.

The ``list`` (and ``longlist``) command will print a section of code around the 
breakpoint, highlighting the next line to be executed. This can be used to determine
where in the circuit execution we are.

.. code-block:: console

    [pldb]: longlist
      3  	@qml.qnode(qml.device('default.qubit', wires=(0,1,2)))
      4  	def circuit(x):
      5  	    # 1) Prepare equal probabilities over wires (0,1):
      6  	    qml.breakpoint()
      7  ->	    qml.Hadamard(wires=0)
      8  	
      9  	    # 2) Entangle wires (1,2):
     10  	    qml.CNOT(wires=(0,2))
     11  	
     12  	    # 3) Apply a rotation of angle 2x to each wire:
     13  	    for w in (0, 1, 2):
     14  	        qml.RX(2*x, wires=w)
     15  	
     16  	    # 4) Reset wire 1:
     17  	    qml.breakpoint()
     18  	    qml.RX(-x, wires=1)
     19  	
     20  	    return qml.sample()
    [pldb]: 

The ``next`` command will execute the next line of code, and print the new line to be
executed. Using this with the ``list`` command, allows us to step through the circuit 
while keeping track of our location in the execution (eg. the next operation to 
execute is ``CNOT``). 

.. code-block:: console

    [pldb]: next
    > /Users/your/path/to/script.py(10)circuit()
    -> qml.CNOT(wires=(0,2))
    [pldb]: list
      5  	    # 1) Prepare equal probabilities over wires (0,1):
      6  	    qml.breakpoint()
      7  	    qml.Hadamard(wires=0)
      8  	
      9  	    # 2) Entangle wires (1,2):
     10  ->	    qml.CNOT(wires=(0,2))
     11  	
     12  	    # 3) Apply a rotation of angle 2x to each wire:
     13  	    for w in (0, 1, 2):
     14  	        qml.RX(2*x, wires=w)
     15  	
    [pldb]: 

In the first section of code aimed to produce an equal probability distribution for 
the first two wires. We can verify this by measuring the probabilities using 
:func:`~pennylane.debugging.probs`. 

.. code-block:: console

    [pldb]: qml.debugging.probs(wires=(0,1))
    array([0.5, 0. , 0.5, 0. ])
    [pldb]: 

We have uncovered a bug in our code! Similarly, we can query the quantum state of the
circuit as we step through using :func:`~pennylane.debugging.state`. For example, we 
can verify if an entangling pair was prepared in the second section of code.

.. code-block:: console

    [pldb]: next
    > /Users/your/path/to/script.py(13)circuit()
    -> for w in (0, 1, 2):
    [pldb]: qml.debugging.state()
    array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
           0.        +0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j])

We have uncovered another bug, qubits ``(0, 2)`` have been entangled instead of 
qubits ``(1, 2)``. Alternative to the step-by-step approach, we can jump between 
breakpoints using ``continue``. This command resumes code execution until the next
breakpoint is reached.

.. code-block:: console

    [pldb]: continue
    > /Users/your/path/to/script.py(18)circuit()
    -> qml.RX(-x, wires=1)
    [pldb]: list
    13  	    for w in (0, 1, 2):
    14  	        qml.RX(2*x, wires=w)
    15  	
    16  	    # 4) Reset wire 1:
    17  	    qml.breakpoint()
    18  ->	    qml.RX(-x, wires=1)
    19  	
    20  	    return qml.sample()
    21  	
    22  	circuit(1.2345)
    [EOF]
    [pldb]:

Additionally, we can *visually* check that our circuit is correct by drawing it! All 
of the operations applied so far are tracked in the circuit's ``QuantumTape`` which 
is accessible using :func:`~pennylane.debugging.tape`.

.. code-block:: console

    [pldb]: tape = qml.debugging.tape()
    [pldb]: print(tape.draw(wire_order=(0, 1, 2)))
    0: ──H─╭●──RX─┤  
    1: ────│───RX─┤  
    2: ────╰X──RX─┤  
    [pldb]:

.. code-block:: console

    [pldb]: next
    > /Users/your/path/to/script.py(20)circuit()
    -> return qml.sample()
    [pldb]: tape = qml.debugging.tape()
    [pldb]: print(tape.draw(wire_order=(0, 1, 2)))
    0: ──H─╭●──RX─────┤  
    1: ────│───RX──RX─┤  
    2: ────╰X──RX─────┤ 
    [pldb]:

Another method for probing the state of the circuit is by measuring observables via 
:func:`~pennylane.debugging.expval`. For example, we use it to verify that the qubit 
on wire 1 was reset.

.. code-block:: console

    [pldb]: qml.debugging.expval(qml.Z(1))
    0.32999315767856763

We have uncovered yet another bug, the qubit was not correctly reset to the 0 state.
We can also dynamically add operations to the quantum circuit from the prompt, allowing us 
to modify the circuit *on-the-fly*! For example, we can apply another ``RX(-x, wires=1)``
gate to fix the qubit.

.. code-block:: console

    [pldb]: x
    1.2345
    [pldb]: qml.RX(-x, wires=1)
    RX(-1.2345, wires=[1])
    [pldb]: tape = qml.debugging.tape()
    [pldb]: print(tape.draw(wire_order=(0,1,2)))
    0: ──H─╭●──RX─────────┤  
    1: ────│───RX──RX──RX─┤  
    2: ────╰X──RX─────────┤ 
    [pldb]: qml.debugging.expval(qml.Z(1))
    0.9999999999999996
    [pldb]: