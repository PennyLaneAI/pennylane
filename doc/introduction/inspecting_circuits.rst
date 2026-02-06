 .. role:: html(raw)
   :format: html


.. _intro_inspecting_circuits:

Inspecting circuits
===================

PennyLane offers functionality to inspect, visualize or analyze quantum circuits.

.. _intro_qtransforms:

Most of these tools are implemented as **transforms**.  Transforms take a :class:`~pennylane.QNode` instance and return a function:

>>> @qp.qnode(dev, diff_method='parameter-shift')
... def my_qnode(x, a=True):
...     # ...
>>> new_func = my_transform(qnode)

This new function accepts the same arguments as the QNode and returns the desired outcome,
such as a dictionary of the QNode's properties, a matplotlib figure drawing the circuit,
or a DAG representing its connectivity structure.

>>> new_func(0.1, a=False)

More information on the concept of transforms can be found in
`Di Matteo et al. (2022) <https://arxiv.org/abs/2202.13414>`_.

Extracting properties of a circuit
----------------------------------

The :func:`~pennylane.specs` transform takes a
QNode and creates a function that returns 
details about the QNode, including depth, number of gates, and number of
gradient executions required.

For example:

.. code-block:: python

    dev = qp.device('default.qubit', wires=4)

    @qp.qnode(dev, diff_method='parameter-shift')
    def circuit(x, y):
        qp.RX(x[0], wires=0)
        qp.Toffoli(wires=(0, 1, 2))
        qp.CRY(x[1], wires=(0, 1))
        qp.Rot(x[2], x[3], y, wires=0)
        return qp.expval(qp.Z(0)), qp.expval(qp.X(1))


We can now use the :func:`~pennylane.specs` transform to generate a function that returns
details and resource information:

>>> x = np.array([0.05, 0.1, 0.2, 0.3], requires_grad=True)
>>> y = np.array(0.4, requires_grad=False)
>>> specs_func = qp.specs(circuit)
>>> specs_func(x, y)
{'resources': Resources(num_wires=3, num_gates=4, gate_types=defaultdict(<class 'int'>, {'RX': 1, 'Toffoli': 1, 'CRY': 1, 'Rot': 1}), depth=4, shots=0),
 'gate_sizes': defaultdict(int, {1: 2, 3: 1, 2: 1}),
 'gate_types': defaultdict(int, {'RX': 1, 'Toffoli': 1, 'CRY': 1, 'Rot': 1}),
 'num_operations': 4,
 'num_observables': 2,
 'num_diagonalizing_gates': 1,
 'num_used_wires': 3,
 'num_trainable_params': 4,
 'depth': 4,
 'num_device_wires': 4,
 'device_name': 'default.qubit',
 'gradient_options': {},
 'interface': 'auto',
 'diff_method': 'parameter-shift',
 'gradient_fn': 'pennylane.gradients.parameter_shift.param_shift',
 'num_gradient_executions': 10}

Circuit drawing
---------------

PennyLane has two built-in circuit drawers, :func:`~pennylane.draw` and
:func:`~pennylane.draw_mpl`.

For example:

.. code-block:: python

    dev = qp.device('default.qubit')

    @qp.qnode(dev)
    def circuit(x, z):
        qp.QFT(wires=(0,1,2,3))
        qp.IsingXX(1.234, wires=(0,2))
        qp.Toffoli(wires=(0,1,2))
        mcm = qp.measure(1)
        mcm_out = qp.measure(2)
        qp.CSWAP(wires=(0,2,3))
        qp.RX(x, wires=0)
        qp.cond(mcm, qp.RY)(np.pi / 4, wires=3)
        qp.CRZ(z, wires=(3,0))
        return qp.expval(qp.Z(0)), qp.probs(op=mcm_out)


    fig, ax = qp.draw_mpl(circuit)(1.2345,1.2345)
    fig.show()

.. image:: ../_static/draw_mpl/main_example.png
    :align: center
    :width: 400px
    :target: javascript:void(0);

>>> print(qp.draw(circuit)(1.2345,1.2345))
0: ─╭QFT─╭IsingXX(1.23)─╭●───────────╭●─────RX(1.23)─╭RZ(1.23)─┤  <Z>
1: ─├QFT─│──────────────├●──┤↗├──────│───────────────│─────────┤
2: ─├QFT─╰IsingXX(1.23)─╰X───║───┤↗├─├SWAP───────────│─────────┤
3: ─╰QFT─────────────────────║────║──╰SWAP──RY(0.79)─╰●────────┤
                             ╚════║═════════╝
                                  ╚════════════════════════════╡  Probs[MCM]

More information, including various fine-tuning options, can be found in
the :doc:`drawing module <../code/qp_drawer>`.

Debugging with mid-circuit snapshots
------------------------------------

When debugging quantum circuits run on simulators, we may want to inspect the current quantum state between gates.

:class:`~pennylane.Snapshot` is an operator like a gate, but it saves the device state at its location in the circuit instead of manipulating the quantum state.

Currently supported devices include:

* ``default.qubit``: each snapshot saves the quantum state vector
* ``default.mixed``: each snapshot saves the density matrix
* ``default.gaussian``: each snapshot saves the covariance matrix and vector of means

A :class:`~pennylane.Snapshot` can be used in a QNode like any other operation:

.. code-block:: python

    dev = qp.device("default.qubit", wires=2)

    @qp.qnode(dev, interface=None)
    def circuit():
        qp.Snapshot(measurement=qp.expval(qp.Z(0)))
        qp.Hadamard(wires=0)
        qp.Snapshot("very_important_state")
        qp.CNOT(wires=[0, 1])
        qp.Snapshot()
        return qp.expval(qp.X(0))

During normal execution, the snapshots are ignored:

>>> circuit()
0.0

However, when using the :func:`~pennylane.snapshots`
transform, intermediate device states will be stored and returned alongside the
results.

>>> qp.snapshots(circuit)()
{0: 1.0,
'very_important_state': array([0.707+0.j, 0.+0.j, 0.707+0.j, 0.+0.j]),
2: array([0.707+0.j, 0.+0.j, 0.+0.j, 0.707+0.j]),
'execution_results': 0.0}

All snapshots are numbered with consecutive integers, and if no tag was provided,
the number of a snapshot is used as a key in the output dictionary instead.

Interactive Debugging on Simulators
-----------------------------------

PennyLane allows for more interactive debugging of quantum circuits in a programmatic 
fashion using quantum breakpoints via :func:`~pennylane.breakpoint`. This feature is 
currently supported on ``default.qubit`` and ``lightning.qubit`` devices. 

Consider the following python script containing the quantum circuit with breakpoints.

.. code-block:: python3
    
    dev = qp.device("default.qubit", wires=2)
    
    @qp.qnode(dev)
    def circuit(x):
        qp.breakpoint()

        qp.RX(x, wires=0)
        qp.Hadamard(wires=1)

        qp.breakpoint()

        qp.CNOT(wires=[0, 1])
        return qp.expval(qp.Z(0))

    circuit(1.23)

Running the circuit above launches an interactive :code:`[pldb]` prompt. Here we can
step through the circuit execution:

.. code-block:: console

    > /Users/your/path/to/script.py(8)circuit()
    -> qp.RX(x, wires=0)
    [pldb] list
      3
      4  	@qp.qnode(dev)
      5  	def circuit(x):
      6  	    qp.breakpoint()
      7
      8  ->	    qp.RX(x, wires=0)
      9  	    qp.Hadamard(wires=1)
     10
     11  	    qp.breakpoint()
     12
     13  	    qp.CNOT(wires=[0, 1])
    [pldb] next
    > /Users/your/path/to/script.py(9)circuit()
    -> qp.Hadamard(wires=1)

We can extract information by making measurements which do not change the state of 
the circuit in execution: 

.. code-block:: console

    [pldb] qp.debug_state()
    array([0.81677345+0.j        , 0.        +0.j        ,
           1.        -0.57695852j, 0.        +0.j        ])
    [pldb] continue
    > /Users/your/path/to/script.py(13)circuit()
    -> qp.CNOT(wires=[0, 1])
    [pldb] next
    > /Users/your/path/to/script.py(14)circuit()
    -> return qp.expval(qp.Z(0))
    [pldb] list
      8  	    qp.RX(x, wires=0)
      9  	    qp.Hadamard(wires=1)
     10  	
     11  	    qp.breakpoint()
     12  	
     13  	    qp.CNOT(wires=[0, 1])
     14  ->	    return qp.expval(qp.Z(0))
     15  	
     16  	circuit(1.23)
    [EOF]

We can also visualize the circuit and dynamically queue operations directly to the circuit:

.. code-block:: console

    [pldb] print(qp.debug_tape().draw())
    0: ──RX─╭●─┤  
    1: ──H──╰X─┤
    [pldb] qp.RZ(-4.56, 1)
    RZ(-4.56, wires=[1])
    [pldb] print(qp.debug_tape().draw())
    0: ──RX─╭●─────┤  
    1: ──H──╰X──RZ─┤

See :doc:`/code/qp_debugging` for more information and detailed examples.

Graph representation
--------------------

PennyLane makes use of several ways to represent a quantum circuit as a Directed Acyclic Graph (DAG).

DAG of causal relations between ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A DAG can be used to represent which operator in a circuit is causally related to another. There are two
options to construct such a DAG:

The :class:`~pennylane.CircuitGraph` class takes a list of gates or channels and hermitian observables
as well as a set of wire labels and constructs a DAG in which the :class:`~.Operator`
instances are the nodes, and each directed edge corresponds to a wire
(or a group of wires) on which the "nodes" act subsequently.

For example, this can be used to compute the effective depth of a circuit,
or to check whether two gates causally influence each other.

.. code-block:: python

    import pennylane as qp
    from pennylane import CircuitGraph
    from pennylane.workflow import construct_tape

    dev = qp.device('lightning.qubit', wires=(0,1,2,3))

    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([1, 2])
        qp.CNOT([2, 3])
        qp.CNOT([3, 1])
        return qp.expval(qp.Z(0))


    circuit()
    tape = construct_tape(circuit)() 
    ops = tape.operations
    obs = tape.observables
    g = CircuitGraph(ops, obs, tape.wires)

Internally, the :class:`~pennylane.CircuitGraph` class constructs a ``rustworkx`` graph object.

>>> type(g.graph)
rustworkx.PyDiGraph

There is no edge between the ``Hadamard`` and the first ``CNOT``, but between consecutive ``CNOT`` gates:

>>> g.has_path(ops[0], ops[1])
False
>>> g.has_path(ops[1], ops[3])
True

The Hadamard is connected to the observable, while the ``CNOT`` operators are not. The observable
does not follow the Hadamard.

>>> g.has_path(ops[0], obs[0])
True
>>> g.has_path(ops[1], obs[0])
False
>>> g.has_path(obs[0], ops[0])
False


Another way to construct the "causal" DAG of a circuit is to use the
:func:`~pennylane.qcut.tape_to_graph` function used by the ``qcut`` module. This
function takes a quantum tape and creates a ``MultiDiGraph`` instance from the ``networkx`` python package.

Using the above example, we get:

>>> g2 = qp.qcut.tape_to_graph(tape)
>>> type(g2)
<class 'networkx.classes.multidigraph.MultiDiGraph'>
>>> for k, v in g2.adjacency():
...    print(k, v)
H(0) {expval(Z(0)): {0: {'wire': 0}}}
CNOT(wires=[1, 2]) {CNOT(wires=[2, 3]): {0: {'wire': 2}}, CNOT(wires=[3, 1]): {0: {'wire': 1}}}
CNOT(wires=[2, 3]) {CNOT(wires=[3, 1]): {0: {'wire': 3}}}
CNOT(wires=[3, 1]) {}
expval(Z(0)) {}

DAG of non-commuting ops
~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~pennylane.commutation_dag` transform can be used to produce an instance of the ``CommutationDAG`` class.
In a commutation DAG, each node represents a quantum operation, and edges represent non-commutation
between two operations.

This transform takes into account that not all operations can be moved next to each other by
pairwise commutation:

>>> def circuit(x, y, z):
...     qp.RX(x, wires=0)
...     qp.RX(y, wires=0)
...     qp.CNOT(wires=[1, 2])
...     qp.RY(y, wires=1)
...     qp.Hadamard(wires=2)
...     qp.CRZ(z, wires=[2, 0])
...     qp.RY(-y, wires=1)
...     return qp.expval(qp.Z(0))
>>> dag_fn = qp.commutation_dag(circuit)
>>> dag = dag_fn(np.pi / 4, np.pi / 3, np.pi / 2)

Nodes in the commutation DAG can be accessed via the ``get_nodes()`` method, returning a list of
the  form ``(ID, CommutationDAGNode)``:

>>> nodes = dag.get_nodes()
>>> nodes
NodeDataView({0: <pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x7f461c4bb580>, ...}, data='node')

Specific nodes in the commutation DAG can be accessed via the ``get_node()`` method:

>>> second_node = dag.get_node(2)
>>> second_node
<pennylane.transforms.commutation_dag.CommutationDAGNode object at 0x136f8c4c0>
>>> second_node.op
CNOT(wires=[1, 2])
>>> second_node.successors
[3, 4, 5, 6]
>>> second_node.predecessors
[]

Fourier representation
----------------------

Parametrized quantum circuits often compute functions in the parameters that
can be represented by Fourier series of a low degree.

The :doc:`../code/qp_fourier` module contains functionality to compute and visualize
properties of such Fourier series.

.. image:: ../_static/fourier_vis_radial_box.png
    :align: center
    :width: 500px
    :target: javascript:void(0);
