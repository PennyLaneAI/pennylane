qp.qaoa
========

.. currentmodule:: pennylane.qaoa

.. automodule:: pennylane.qaoa

Solving the MaxCut problem using QAOA
-------------------------------------

We can demonstrate the PennyLane QAOA functionality with a basic application of QAOA:
solving the `MaxCut <https://en.wikipedia.org/wiki/Maximum_cut>`__ problem.
We begin by defining the set of wires on which QAOA is executed, as well as the graph
on which we will perform MaxCut. The node labels of the graph are the index of the wire to which they
correspond:

.. code-block:: python3

    import pennylane as qp
    from pennylane import qaoa
    from networkx import Graph

    # Defines the wires and the graph on which MaxCut is being performed
    wires = range(3)
    graph = Graph([(0, 1), (1, 2), (2, 0)])

We now obtain the QAOA cost and mixer Hamiltonians for MaxCut on the graph that we defined:

.. code-block:: python3

    # Defines the QAOA cost and mixer Hamiltonians
    cost_h, mixer_h = qaoa.maxcut(graph)

These cost and mixer Hamiltonians are then used to define layers of the variational QAOA ansatz,
which we implement as the following function:

.. code-block:: python3

    # Defines a layer of the QAOA ansatz from the cost and mixer Hamiltonians
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

Finally, the full QAOA circuit is built. We begin by initializing the wires in an even superposition over
computational basis states, and then repeatedly apply QAOA layers with the
``qp.layer`` method. In this case we repeat the circuit twice:

.. code-block:: python3

    # Repeatedly applies layers of the QAOA ansatz
    def circuit(params):
        for w in wires:
            qp.Hadamard(wires=w)
        qp.layer(qaoa_layer, 2, params[0], params[1])

With the circuit defined, we call the device on which QAOA will be executed and use ``qp.expval()`` to
create the QAOA cost function: the expected value of the cost Hamiltonian with respect to the parametrized output
of the QAOA circuit.

.. code-block:: python3
    
    # Defines the device and the QAOA cost function
    dev = qp.device('default.qubit', wires=len(wires))
    @qp.qnode(dev)
    def cost_function(params):
        circuit(params)
        return qp.expval(cost_h)

>>> print(cost_function([[1, 1], [1, 1]]))
-1.8260274380964299

The QAOA cost function can then be optimized in the usual way, by calling one of the built-in PennyLane optimizers
and updating the variational parameters until the expected value of the cost Hamiltonian is minimized.
