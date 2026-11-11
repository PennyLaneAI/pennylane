.. role:: html(raw)
   :format: html

.. _intro_ref_compile_circuits:

Compiling circuits
==================

PennyLane offers multiple tools for compiling circuits. We use the term "compilation"
here in a loose sense as the process of transforming one circuit into one or more
differing circuits. A circuit could be either a quantum function or a sequence of
operators. For example, such a transformation could replace a gate type with another,
fuse gates, exploit mathematical relations that simplify an observable, or replace a
large circuit by a number of smaller circuits.

Compilation functionality is mostly designed as **transforms**; see the :doc:`transforms documentation <../code/qp_transforms>`
for more details, as well as information on how to write your own custom transforms.

In addition to quantum circuit transforms, PennyLane also supports experimental 
just-in-time compilation, via the :func:`~.qjit` decorator and `Catalyst <https://github.com/pennylaneai/catalyst>`__. 
This is more general, and supports full hybrid compilation --- compiling both the 
classical and quantum components of your workflow into a binary that can be run 
close to the accelerators that you are using. More details can be found in :doc:`compiling workflows </introduction/compiling_workflows>`.

Compilation transforms for circuit optimization
-----------------------------------------------

PennyLane includes multiple transforms that can act on ``QNode``'s, quantum functions, 
and multiple other PennyLane objects. See the :ref:`transforms library <transform_library>` 
for a complete list.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.transforms.cancel_inverses
    ~pennylane.transforms.commute_controlled
    ~pennylane.transforms.merge_amplitude_embedding
    ~pennylane.transforms.merge_rotations
    ~pennylane.transforms.pattern_matching
    ~pennylane.transforms.remove_barrier
    ~pennylane.transforms.single_qubit_fusion
    ~pennylane.transforms.undo_swaps
    ~pennylane.transforms.decompose
    ~pennylane.transforms.combine_global_phases

:html:`</div>`

.. note::

    Most compilation transforms support just-in-time compilation with ``jax.jit``.

Transforms can be applied on ``QNodes`` using the decorator syntax:

.. code-block:: python

    dev = qp.device("default.qubit", wires=2)

    @qp.transforms.split_non_commuting(grouping_strategy="wires")
    @qp.qnode(dev)
    def circuit(params):
        qp.RX(params[0], wires=0)
        qp.RZ(params[1], wires=1)
        return [
            qp.expval(qp.X(0)),
            qp.expval(qp.Y(1)),
            qp.expval(qp.Z(0) @ qp.Z(1)),
            qp.expval(qp.X(0) @ qp.Z(1) + 0.5 * qp.Y(1) + qp.Z(0)),
        ]

They can additionally be stacked, allowing for the application of multiple compilation passes on a ``QNode``:

.. code-block:: python

    dev = qp.device("default.qubit", wires=1)

    @qp.transforms.merge_rotations
    @qp.transforms.cancel_inverses(recursive=True)
    @qp.qnode(device=dev)
    def circuit(x, y):
        qp.X(wires=0)
        qp.Hadamard(wires=0)
        qp.Hadamard(wires=0)
        qp.X(wires=0)
        qp.RX(x, wires=0)
        qp.RX(y, wires=0)
        return qp.expval(qp.Z(0))

Alternatively, multiple transforms can be chained together into custom circuit optimization pipelines
via the :class:`~.CompilePipeline` class. The defined compilation program can then be applied on a
``QNode``, which will transform the circuit with each pass therein.

For example, by constructing a pipeline with the sequence of three transforms: :func:`~.pennylane.transforms.commute_controlled`,
:func:`~.pennylane.transforms.cancel_inverses`, and :func:`~.pennylane.transforms.merge_rotations`,
the number of gates in the following quantum circuit can be reduced by almost a half:

.. code-block:: python

    pipeline = qp.CompilePipeline(
        qp.transforms.commute_controlled,
        qp.transforms.cancel_inverses(recursive=True),
        qp.transforms.merge_rotations,
    )

    @pipeline
    @qp.qnode(qp.device("default.qubit"))
    def circuit(x, y, z):
        qp.Hadamard(wires=0)
        qp.Hadamard(wires=1)
        qp.Hadamard(wires=2)
        qp.RZ(z, wires=2)
        qp.CNOT(wires=[2, 1])
        qp.RX(z, wires=0)
        qp.CNOT(wires=[1, 0])
        qp.RX(x, wires=0)
        qp.CNOT(wires=[1, 0])
        qp.RZ(-z, wires=2)
        qp.RX(y, wires=2)
        qp.Y(wires=2)
        qp.CZ(wires=[1, 2])
        return qp.expval(qp.Z(wires=0))

>>> print(qp.draw(circuit)(0.2, 0.3, 0.4))
0: ──H──RX(0.60)─────────────────┤  <Z>
1: ──H─╭X─────────────────────╭●─┤     
2: ──H─╰●─────────RX(0.30)──Y─╰Z─┤     

.. note::

    The :class:`~.pennylane.Barrier` operator can be used to prevent blocks of code from being merged during
    compilation.

For more details on :class:`~.CompilePipeline` and the available compilation transforms, visit
`the compilation documentation
<../code/qp_transforms.html#transforms-for-circuit-compilation>`_.

Gate decompositions
-------------------

When compiling a circuit, it is often beneficial to decompose the circuit into a
set of gates. For tape-based (non-compiled) workflows, use
:func:`~.pennylane.transforms._tape_decompose`, which decomposes circuits into a
set of gates defined either by their name, type, or by a set of rules they must
follow. Under :func:`~.qjit`, use :func:`~.pennylane.transforms.decompose`, which
dispatches to Catalyst's graph-decomposition pass. Both transforms require an explicit
``gate_set``.

Using a gate set
****************

The example below demonstrates how a three-wire circuit can be decomposed using 
a pre-defined set of gates: 

.. code-block:: python

    from pennylane.transforms import _tape_decompose

    dev = qp.device('default.qubit')
    allowed_gates = {qp.Toffoli, qp.RX, qp.RZ, qp.GlobalPhase}

    @_tape_decompose(gate_set=allowed_gates)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=[0])
        qp.Toffoli(wires=[0,1,2])
        return qp.expval(qp.Z(0))
    
With the Hadamard gate not in our gate set, it will be decomposed into allowed rotation 
gate operators.

>>> print(qp.draw(circuit)())
0: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╭GlobalPhase(-1.57)─╭●─┤  <Z>
1: ───────────────────────────────├GlobalPhase(-1.57)─├●─┤
2: ───────────────────────────────╰GlobalPhase(-1.57)─╰X─┤

Using a gate rule
*****************

The example below demonstrates how a three-wire circuit can be decomposed into single 
or two-qubit gates using a rule:

.. code-block:: python

    @_tape_decompose(
        gate_set={"H", "T", "Adjoint(T)", "CNOT"},
        stopping_condition=lambda op: len(op.wires) <= 2,
    )
    @qp.qnode(dev)
    def circuit():
        qp.Toffoli(wires=[0,1,2])
        return qp.expval(qp.Z(0))

>>> print(qp.draw(circuit)())
0: ───────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤     
2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤ 

Decomposition in stages
***********************

You can use the ``max_expansion`` argument to control the number of decomposition 
stages applied to the circuit. By default, the function will decompose the circuit 
until the desired gate set is reached.

The example below shows how the user can visualize the decomposition. We begin 
with creating a :class:`~.pennylane.QuantumPhaseEstimation` circuit: 

.. code-block:: python

    phase = 1 
    unitary = qp.RX(phase, wires=0)
    n_estimation_wires = 3
    estimation_wires = range(1, n_estimation_wires + 1)
    gate_set = qp.gate_sets.ALL_OPS

    @qp.qnode(qp.device('default.qubit'))
    def circuit():
        # Start in the |+> eigenstate of the unitary
        qp.Hadamard(0)
        qp.QuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)

From here, we can iterate through the stages of decomposition:

>>> from pennylane.transforms import _tape_decompose
>>> print(qp.draw(_tape_decompose(circuit, gate_set=gate_set, max_expansion=0))())
0: ──H─╭QuantumPhaseEstimation─┤  
1: ────├QuantumPhaseEstimation─┤  
2: ────├QuantumPhaseEstimation─┤  
3: ────╰QuantumPhaseEstimation─┤  

>>> print(qp.draw(_tape_decompose(circuit, gate_set=gate_set, max_expansion=1))())
0: ──H─╭RX(1.00)⁴─╭RX(1.00)²─╭RX(1.00)¹───────┤
1: ──H─╰●─────────│──────────│──────────╭QFT†─┤
2: ──H────────────╰●─────────│──────────├QFT†─┤
3: ──H───────────────────────╰●─────────╰QFT†─┤

>>> print(qp.draw(_tape_decompose(circuit, gate_set=gate_set, max_expansion=2))())
0: ──H─╭U(M0)─╭U(M1)─╭U(M2)───────────────────────────────────────────────────────────┤
1: ──H─╰●─────│──────│──────╭SWAP†──────────────────────╭(Rϕ(0.79))†─╭(Rϕ(1.57))†──H†─┤
2: ──H────────╰●─────│──────│──────────╭(Rϕ(1.57))†──H†─│────────────╰(Rϕ(1.57))†─────┤
3: ──H───────────────╰●─────╰SWAP†──H†─╰(Rϕ(1.57))†─────╰(Rϕ(0.79))†──────────────────┤
<BLANKLINE>
M0 =
[[-0.41614684+0.j          0.        -0.90929743j]
 [ 0.        -0.90929743j -0.41614684+0.j        ]]
M1 =
[[0.54030231+0.j         0.        -0.84147098j]
 [0.        -0.84147098j 0.54030231+0.j        ]]
M2 =
[[0.87758256+0.j         0.        -0.47942554j]
 [0.        -0.47942554j 0.87758256+0.j        ]]

Custom Operator Decomposition
-----------------------------

When executing QNodes on a device, PennyLane will automatically decompose gates 
that are unsupported by the device using built-in decomposition rules.

You can register additional decomposition rules globally or supply them to an individual
``_tape_decompose`` transform.

Globally registered decompositions
**********************************

For example, suppose we would like to implement the following QNode:

.. code-block:: python

    def circuit(weights):
        qp.BasicEntanglerLayers(weights, wires=[0, 1, 2])
        return qp.expval(qp.Z(0))

    original_dev = qp.device("default.qubit", wires=3)
    original_qnode = qp.QNode(circuit, original_dev)

>>> weights = np.array([[0.4, 0.5, 0.6]])
>>> print(qp.draw(original_qnode, level="device")(weights))
0: ──RX(0.40)─╭●────╭X─┤  <Z>
1: ──RX(0.50)─╰X─╭●─│──┤     
2: ──RX(0.60)────╰X─╰●─┤     

To decompose ``CNOT`` into ``CZ`` and ``Hadamard``, register a quantum function with
its required resources:

.. code-block:: python3

    @qp.decomposition.register_resources({
        qp.H: 2,
        qp.CZ: 1
    })
    def _custom_cnot_decomposition(wires, **_):
        qp.Hadamard(wires=wires[1])
        qp.CZ(wires=[wires[0], wires[1]])
        qp.Hadamard(wires=wires[1])

    qp.decomposition.add_decomps(qp.CNOT, _custom_cnot_decomposition)


    decomp_dev = qp.device("default.qubit", wires=3)
    qnode = qp.QNode(circuit, decomp_dev)
    decomp_qnode = qp.transforms._tape_decompose(qnode, gate_set={qp.CZ, qp.H, qp.RX})

Now when we draw or run a QNode on this device, the gates will be expanded according
to our specifications:

>>> print(qp.draw(decomp_qnode, level="device")(weights))
0: ──RX(0.40)────╭●──H───────╭Z──H─┤  <Z>
1: ──RX(0.50)──H─╰Z──H─╭●────│─────┤
2: ──RX(0.60)──H───────╰Z──H─╰●────┤

.. _custom_decomps_with_graph:

Local decomposition rules
*************************

The :func:`~.pennylane.transforms._tape_decompose` transform offers the ability to inject
custom decompositions via two keyword arguments:

* ``fixed_decomps`` maps an operator type or name to a rule that replaces its registered
  decomposition rules when the operator needs to be decomposed.
* ``alt_decomps`` maps an operator type or name to a list of additional rules the solver
  can consider alongside its registered rules.

Both keyword arguments above require a dictionary mapping PennyLane operator types or names to
custom decompositions. Creating custom decompositions that the graph-based system
can use involves a PennyLane quantum function that represents the decomposition, 
and a declaration of its resource requirements (gate counts) via :func:`~.pennylane.register_resources`.

Consider this example where we add a fixed decomposition to ``CNOT`` gates:

.. code-block:: python

    @qp.register_resources({qp.H: 2, qp.CZ: 1})
    def my_cnot(wires, **__):
        qp.H(wires=wires[1])
        qp.CZ(wires=wires)
        qp.H(wires=wires[1])

The :func:`~.pennylane.register_resources` accepts a dictionary mapping operator 
types within the custom decomposition to the number of times they occur in the decomposition. 
With the resources registered, this can be used with ``fixed_decomps`` or ``alt_decomps``:

.. code-block:: python

    @qp.transforms._tape_decompose(
        fixed_decomps={qp.CNOT: my_cnot},
        gate_set={qp.H, qp.S, qp.T, qp.CZ},
    )
    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        qp.CNOT(wires=[0, 1])
        return qp.state()

>>> print(qp.draw(circuit, level="device")())
0: ────╭●────┤  State
1: ──H─╰Z──H─┤  State

Note that the ``alt_decomps`` argument can handle multiple alternatives per operator
type:

.. code-block:: python

    @qp.register_resources({qp.H: 2, qp.CZ: 1})
    def my_cnot1(wires, **__):
        qp.H(wires=wires[1])
        qp.CZ(wires=wires)
        qp.H(wires=wires[1])

    @qp.register_resources({qp.RY: 2, qp.CZ: 1, qp.Z: 2})
    def my_cnot2(wires, **__):
        qp.RY(np.pi/2, wires[1])
        qp.Z(wires[1])
        qp.CZ(wires=wires)
        qp.RY(np.pi/2, wires[1])
        qp.Z(wires[1])

    @qp.transforms._tape_decompose(
        gate_set={qp.CZ, qp.H, qp.Z, qp.RY},
        alt_decomps={qp.CNOT: [my_cnot1, my_cnot2]},
    )
    @qp.qnode(qp.device("default.qubit"))
    def circuit():
        qp.CNOT(wires=[0, 1])
        return qp.state()

The decomposition that the algorithm chooses internally will be the most resource-efficient.
More details on resource functions and conditional decomposition rules can be found in the
usage details for :func:`~.pennylane.register_resources`.

Alternatively, new decomposition rules can be added to operators *globally* with 
the :func:`~.pennylane.add_decomps` function. This negates having to specify ``alt_decomps``
in every instance of the ``_tape_decompose`` transform. The following example globally
adds the ``my_cnot1`` and ``my_cnot2`` decomposition rules to the ``qp.CNOT`` gate:

>>> qp.add_decomps(qp.CNOT, my_cnot1, my_cnot2)

The newly added rules for the ``qp.CNOT`` operator can be verified or inspected
with the :func:`~.pennylane.list_decomps` function:

>>> my_new_rules = qp.list_decomps(qp.CNOT)[-2:]
>>> print(qp.draw(my_new_rules[1])(wires=[0, 1]))
0: ──────────────╭●──────────────┤  
1: ──RY(1.57)──Z─╰Z──RY(1.57)──Z─┤ 

Simplifying Operators
----------------------

PennyLane provides the :func:`~.pennylane.simplify` function to simplify single operators, quantum
functions, QNodes and tapes. This function has several purposes:

* Reducing the arithmetic depth of the given operators to its minimum.
* Grouping like terms in sums and products.
* Resolving products of Pauli operators.
* Combining identical rotation gates by summing its angles.

Here are some simple simplification routines:

>>> qp.simplify(qp.RX(4*np.pi+0.1, 0 ))
RX(0.09999999999999964, wires=[0])
>>> qp.simplify(qp.adjoint(qp.RX(1.23, 0)))
RX(11.336370614359172, wires=[0])
>>> qp.simplify(qp.ops.Pow(qp.RX(1, 0), 3))
RX(3.0, wires=[0])
>>> qp.simplify(qp.sum(qp.Y(3), qp.Y(3)))
2.0 * Y(3)
>>> qp.simplify(qp.RX(1, 0) @ qp.RX(1, 0))
RX(2.0, wires=[0])
>>> qp.simplify(qp.prod(qp.X(0), qp.Z(0)))
-1j * Y(0)

Now lets simplify a nested operator:

>>> sum_op = qp.RX(1, 0) + qp.X(0)
>>> prod1 = qp.X(0) @ sum_op
>>> nested_op = qp.prod(prod1, qp.RX(1, 0))
>>> qp.simplify(nested_op)
(X(0) @ RX(2.0, wires=[0])) + RX(1.0, wires=[0])

Several simplifications steps are happening here. First of all, the nested products are removed:

.. code-block:: python

    qp.prod(qp.X(0), qp.sum(qp.RX(1, 0), qp.X(0)), qp.RX(1, 0))

Then the product of sums is transformed into a sum of products:

.. code-block:: python

    qp.sum(qp.prod(qp.X(0), qp.RX(1, 0), qp.RX(1, 0)), qp.prod(qp.X(0), qp.X(0), qp.RX(1, 0)))

And finally like terms in the obtained products are grouped together, removing all identities: 

.. code-block:: python

    qp.sum(qp.prod(qp.X(0), qp.RX(2, 0)), qp.RX(1, 0))

As mentioned earlier we can also simplify QNode objects to, for example, group rotation gates:

.. code-block:: python

    dev = qp.device("default.qubit", wires=2)

    @qp.simplify
    @qp.qnode(dev)
    def circuit(x):
        (
            qp.RX(x[0], wires=0)
            @ qp.RY(x[1], wires=1)
            @ qp.RZ(x[2], wires=2)
            @ qp.RX(-1, wires=0)
            @ qp.RY(-2, wires=1)
            @ qp.RZ(2, wires=2)
        )
        return qp.probs([0, 1, 2])

>>> x = [1, 2, 3]
>>> print(qp.draw(circuit)(x))
0: ───────────┤ ╭Probs
1: ───────────┤ ├Probs
2: ──RZ(5.00)─┤ ╰Probs

Groups of commuting Pauli words
-------------------------------

Mutually commuting Pauli words can be measured simultaneously on a quantum computer.
Finding groups of mutually commuting observables can therefore reduce the number of circuit executions,
and is an example of how observables can be "compiled".

PennyLane contains different functionalities for this purpose, ranging from higher-level
transforms acting on QNodes to lower-level functions acting on operators.

An example of a transform manipulating QNodes is :func:`~.pennylane.transforms.split_non_commuting`.
It turns a QNode that measures non-commuting observables into a QNode that internally
uses *multiple* circuit executions with qubit-wise commuting groups. The transform is used
by devices to make such measurements possible.

On a lower level, the :func:`~.pennylane.pauli.group_observables` function can be used to split lists of
observables and coefficients:

>>> obs = [qp.Y(0), qp.X(0) @ qp.X(1), qp.Z(1)]
>>> coeffs = [1.43, 4.21, 0.97]
>>> groupings = qp.pauli.group_observables(obs, coeffs, 'anticommuting', 'lf')
>>> obs_groupings, coeffs_groupings = groupings
>>> obs_groupings
[[Z(1), X(0) @ X(1)], [Y(0)]]
>>> coeffs_groupings
[[0.97, 4.21], [1.43]]

This and more logic to manipulate Pauli observables is found in the :doc:`pauli module <../code/qp_pauli>`.
