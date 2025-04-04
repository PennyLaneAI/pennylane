.. role:: html(raw)
   :format: html

.. _intro_ref_compile_circuits:

Compiling circuits
==================

PennyLane offers multiple tools for compiling circuits. We use the term "compilation"
here in a loose sense as the process of transforming one circuit 
into one or more differing circuits. A circuit could be either a quantum function or a sequence of operators. For
example, such a transformation could
replace a gate type with another, fuse gates, exploit mathematical relations that simplify an observable,
or replace a large circuit by a number of smaller circuits.

Compilation functionality is mostly designed as **transforms**; see the
the :doc:`transforms documentation <../code/qml_transforms>` for more details,
as well as information on how to write your own custom transforms.

In addition to quantum circuit transforms, PennyLane also
supports experimental just-in-time compilation, via the :func:`~.qjit` decorator and
`Catalyst <https://github.com/pennylaneai/catalyst>`__. This is more general, and
supports full hybrid compilation --- compiling both the classical and quantum components
of your workflow into a binary that can be run close to the accelerators.
that you are using. More details can be found in :doc:`compiling workflows </introduction/compiling_workflows>`.

Simplifying Operators
----------------------

PennyLane provides the :func:`~.pennylane.simplify` function to simplify single operators, quantum
functions, QNodes and tapes. This function has several purposes:

* Reducing the arithmetic depth of the given operators to its minimum.
* Grouping like terms in sums and products.
* Resolving products of Pauli operators.
* Combining identical rotation gates by summing its angles.

Here are some simple simplification routines:

>>> qml.simplify(qml.RX(4*np.pi+0.1, 0 ))
RX(0.09999999999999964, wires=[0])
>>> qml.simplify(qml.adjoint(qml.RX(1.23, 0)))
RX(11.336370614359172, wires=[0])
>>> qml.simplify(qml.ops.Pow(qml.RX(1, 0), 3))
RX(3.0, wires=[0])
>>> qml.simplify(qml.sum(qml.Y(3), qml.Y(3)))
2.0 * Y(3)
>>> qml.simplify(qml.RX(1, 0) @ qml.RX(1, 0))
RX(2.0, wires=[0])
>>> qml.simplify(qml.prod(qml.X(0), qml.Z(0)))
-1j * Y(0)

Now lets simplify a nested operator:

>>> sum_op = qml.RX(1, 0) + qml.X(0)
>>> prod1 = qml.X(0) @ sum_op
>>> nested_op = qml.prod(prod1, qml.RX(1, 0))
>>> qml.simplify(nested_op)
(X(0) @ RX(2.0, wires=[0])) + RX(1.0, wires=[0])

Several simplifications steps are happening here. First of all, the nested products are removed:

.. code-block:: python

    qml.prod(qml.X(0), qml.sum(qml.RX(1, 0), qml.X(0)), qml.RX(1, 0))

Then the product of sums is transformed into a sum of products:

.. code-block:: python

    qml.sum(qml.prod(qml.X(0), qml.RX(1, 0), qml.RX(1, 0)), qml.prod(qml.X(0), qml.X(0), qml.RX(1, 0)))

And finally like terms in the obtained products are grouped together, removing all identities: 

.. code-block:: python

    qml.sum(qml.prod(qml.X(0), qml.RX(2, 0)), qml.RX(1, 0))

As mentioned earlier we can also simplify QNode objects to, for example, group rotation gates:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.simplify
    @qml.qnode(dev)
    def circuit(x):
        (
            qml.RX(x[0], wires=0)
            @ qml.RY(x[1], wires=1)
            @ qml.RZ(x[2], wires=2)
            @ qml.RX(-1, wires=0)
            @ qml.RY(-2, wires=1)
            @ qml.RZ(2, wires=2)
        )
        return qml.probs([0, 1, 2])

>>> x = [1, 2, 3]
>>> print(qml.draw(circuit)(x))
0: ───────────┤ ╭Probs
1: ───────────┤ ├Probs
2: ──RZ(5.00)─┤ ╰Probs

Compilation transforms for circuit optimization
-----------------------------------------------

PennyLane includes multiple transforms that take quantum functions and return new
quantum functions of optimized circuits:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.transforms.cancel_inverses
    ~pennylane.transforms.commute_controlled
    ~pennylane.transforms.merge_amplitude_embedding
    ~pennylane.transforms.cancel_inverses
    ~pennylane.transforms.merge_rotations
    ~pennylane.transforms.pattern_matching
    ~pennylane.transforms.remove_barrier
    ~pennylane.transforms.single_qubit_fusion
    ~pennylane.transforms.undo_swaps
    ~pennylane.transforms.decompose

:html:`</div>`

.. note::

    Most compilation transforms support just-in-time compilation with ``jax.jit``.

The :func:`~.pennylane.compile` transform allows you to chain together
sequences of quantum function transforms into custom circuit optimization pipelines.

For example, take the following decorated quantum function:

.. code-block:: python

    dev = qml.device('default.qubit', wires=[0, 1, 2])

    @qml.compile
    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.RZ(z, wires=2)
        qml.CNOT(wires=[2, 1])
        qml.RX(z, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RX(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RZ(-z, wires=2)
        qml.RX(y, wires=2)
        qml.Y(wires=2)
        qml.CZ(wires=[1, 2])
        return qml.expval(qml.Z(wires=0))

The default behaviour of :func:`~.pennylane.compile` applies a sequence of three
transforms: :func:`~.pennylane.transforms.commute_controlled`, :func:`~.pennylane.transforms.cancel_inverses`,
and then :func:`~.pennylane.transforms.merge_rotations`.

>>> print(qml.draw(circuit)(0.2, 0.3, 0.4))
0: ──H──RX(0.60)─────────────────┤  <Z>
1: ──H─╭X─────────────────────╭●─┤     
2: ──H─╰●─────────RX(0.30)──Y─╰Z─┤     


The :func:`~.pennylane.compile` transform is flexible and accepts a custom pipeline
of quantum function transforms (you can even write your own!).
For example, if we wanted to only push single-qubit gates through
controlled gates and cancel adjacent inverses, we could do:

.. code-block:: python

    from pennylane.transforms import commute_controlled, cancel_inverses
    from functools import partial

    pipeline = [commute_controlled, cancel_inverses]

    @partial(qml.compile, pipeline=pipeline)
    @qml.qnode(dev)
    def qfunc(x, y, z):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.RZ(z, wires=2)
        qml.CNOT(wires=[2, 1])
        qml.RX(z, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RX(x, wires=0)
        qml.CNOT(wires=[1, 0])
        qml.RZ(-z, wires=2)
        qml.RX(y, wires=2)
        qml.Y(wires=2)
        qml.CZ(wires=[1, 2])
        return qml.expval(qml.Z(wires=0))

>>> print(qml.draw(qfunc)(0.2, 0.3, 0.4))
0: ──H──RX(0.40)──RX(0.20)────────────────────────────┤  <Z>
1: ──H─╭X──────────────────────────────────────────╭●─┤     
2: ──H─╰●─────────RZ(0.40)──RZ(-0.40)──RX(0.30)──Y─╰Z─┤     

.. note::

    The :class:`~.pennylane.Barrier` operator can be used to prevent blocks of code from being merged during
    compilation.


For more details on :func:`~.pennylane.compile` and the available compilation transforms, visit
`the compilation documentation
<../code/qml_transforms.html#transforms-for-circuit-compilation>`_.

Gate decompositions
-------------------

When compiling a circuit, it is often beneficial to decompose the circuit into a 
set of gates. To do this, we can use the :func:`~.pennylane.transforms.decompose` 
function, which enables decomposition of circuits into a set of gates defined either 
by their name, type, or by a set of rules they must follow.

.. note::

    Using :func:`~.pennylane.decompositions.enable_graph` enables PennyLane's new 
    **experimental** decomposition algorithm (by default, this new system is *not* 
    enabled). This new system uses a graph-based approach, which provides better 
    overall versatility and resource efficiency.

Using a gate set
****************

The example below demonstrates how a three-wire circuit can be decomposed using 
a pre-defined set of gates: 

.. code-block:: python
    
    from pennylane.transforms import decompose
    from functools import partial

    dev = qml.device('default.qubit')
    allowed_gates = {qml.Toffoli, qml.RX, qml.RZ}

    @partial(decompose, gate_set=allowed_gates)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=[0])
        qml.Toffoli(wires=[0,1,2])
        return qml.expval(qml.Z(0))
    
With the Hadamard gate not in our gate set, it will be decomposed into allowed rotation 
gate operators.

>>> print(qml.draw(circuit)())
0: ──RZ(1.57)──RX(1.57)──RZ(1.57)─╭●─┤  <Z>
1: ───────────────────────────────├●─┤     
2: ───────────────────────────────╰X─┤ 

Using a gate rule
*****************

The example below demonstrates how a three-wire circuit can be decomposed into single 
or two-qubit gates using a rule:

.. code-block:: python

    # functions in gate_set can only be used with graph decomposition system disabled
    qml.decompositions.disable_graph()

    @partial(decompose, gate_set=lambda op: len(op.wires) <= 2) 
    @qml.qnode(dev)
    def circuit():
        qml.Toffoli(wires=[0,1,2])
        return qml.expval(qml.Z(0)) 

>>> print(qml.draw(circuit)())
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
    target_wires = [0]
    unitary = qml.RX(phase, wires=0).matrix()
    n_estimation_wires = 3
    estimation_wires = range(1, n_estimation_wires + 1)

    @qml.qnode(qml.device('default.qubit'))
    def circuit():
        # Start in the |+> eigenstate of the unitary
        qml.Hadamard(wires=target_wires)
        qml.QuantumPhaseEstimation(
            unitary,
            target_wires=target_wires,
            estimation_wires=estimation_wires,
        ) 

From here, we can iterate through the stages of decomposition:

>>> print(qml.draw(decompose(circuit, max_expansion=0))())
0: ──H─╭QuantumPhaseEstimation─┤  
1: ────├QuantumPhaseEstimation─┤  
2: ────├QuantumPhaseEstimation─┤  
3: ────╰QuantumPhaseEstimation─┤  

>>> print(qml.draw(decompose(circuit, max_expansion=1))())
0: ──H─╭U(M0)⁴─╭U(M0)²─╭U(M0)¹───────┤  
1: ──H─╰●──────│───────│───────╭QFT†─┤  
2: ──H─────────╰●──────│───────├QFT†─┤  
3: ──H─────────────────╰●──────╰QFT†─┤  

>>> print(qml.draw(decompose(circuit, max_expansion=2))())
0: ──H──RZ(11.00)──RY(1.14)─╭X──RY(-1.14)──RZ(-9.42)─╭X──RZ(-1.57)──RZ(1.57)──RY(1.00)─╭X──RY(-1.00)
1: ──H──────────────────────╰●───────────────────────╰●────────────────────────────────│────────────
2: ──H─────────────────────────────────────────────────────────────────────────────────╰●───────────
3: ──H──────────────────────────────────────────────────────────────────────────────────────────────
───RZ(-6.28)─╭X──RZ(4.71)──RZ(1.57)──RY(0.50)─╭X──RY(-0.50)──RZ(-6.28)─╭X──RZ(4.71)─────────────────
─────────────│────────────────────────────────│────────────────────────│──╭SWAP†────────────────────
─────────────╰●───────────────────────────────│────────────────────────│──│─────────────╭(Rϕ(1.57))†
──────────────────────────────────────────────╰●───────────────────────╰●─╰SWAP†─────H†─╰●──────────
────────────────────────────────────┤  
──────╭(Rϕ(0.79))†─╭(Rϕ(1.57))†──H†─┤  
───H†─│────────────╰●───────────────┤  
──────╰●────────────────────────────┤  

Custom Operator Decomposition
-----------------------------

When executing QNodes on a device, PennyLane will automatically decompose gates 
that are unsupported by the device using built-in decomposition rules.

In addition, you can provide *new* decomposition rules to be used, but the behaviour
and user-interface is different depending on if the graph decompositions system is
enabled.

Default behaviour with custom decompositions
********************************************

For example, suppose we would like to implement the following QNode:

.. code-block:: python

    def circuit(weights):
        qml.BasicEntanglerLayers(weights, wires=[0, 1, 2])
        return qml.expval(qml.Z(0))

    original_dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(circuit, original_dev)

>>> weights = np.array([[0.4, 0.5, 0.6]])
>>> print(qml.draw(original_qnode, level="device")(weights))
0: ──RX(0.40)─╭●────╭X─┤  <Z>
1: ──RX(0.50)─╰X─╭●─│──┤     
2: ──RX(0.60)────╰X─╰●─┤     

Now, let's swap out PennyLane's default decomposition of the ``CNOT`` gate into 
``CZ`` and ``Hadamard``. We define the custom decompositions like so, and pass them 
to a device:

.. code-block:: python

    def custom_cnot(wires, **_):
        return [
            qml.Hadamard(wires=wires[1]),
            qml.CZ(wires=[wires[0], wires[1]]),
            qml.Hadamard(wires=wires[1])
        ]

    custom_decomps = {qml.CNOT: custom_cnot}

    decomp_dev = qml.device("default.qubit", wires=3, custom_decomps=custom_decomps)
    decomp_qnode = qml.QNode(circuit, decomp_dev)

Note that custom decomposition functions should accept keyword arguments even when 
it is not used.

Now when we draw or run a QNode on this device, the gates will be expanded according 
to our specifications:

>>> print(qml.draw(decomp_qnode, level="device")(weights))
0: ──RX(0.40)────╭●──H───────╭Z──H─┤  <Z>
1: ──RX(0.50)──H─╰Z──H─╭●────│─────┤     
2: ──RX(0.60)──H───────╰Z──H─╰●────┤     

If the custom decomposition is only supposed to be used in a specific code context,
a separate context manager :func:`~.pennylane.transforms.set_decomposition` can 
be used.

.. note::

    Device-level custom decompositions **are not applied before other compilation 
    passes (decorators on QNodes)**. For example, the following circuit has ``cancel_inverses`` 
    applied to it, and the device was provided a decomposition for ``qml.CNOT``. 
    The Hadamard gates applied around the ``qml.CNOT`` gate do not get cancelled 
    with those introduced by the custom decomposition.

    .. code-block:: python

        def custom_cnot(wires, **_):
            return [
                qml.H(wires=wires[1]),
                qml.CZ(wires=[wires[0], wires[1]]),
                qml.H(wires=wires[1])
            ]

        dev = qml.device("default.qubit", custom_decomps={qml.CNOT: custom_cnot})

        @qml.transforms.cancel_inverses
        @qml.qnode(dev)
        def circuit():
            qml.H(1)
            qml.CNOT([0, 1])
            qml.H(1)
            return qml.state()

    >>> print(qml.draw(circuit, level="device")())
    0: ───────╭●───────┤  State
    1: ──H──H─╰Z──H──H─┤  State

    To have better control over custom decompositions, consider using the graph 
    decompositions system functionality outlined in the next section.

Custom decompositions with qml.decomposition.enable_graph
*********************************************************

With the graph decompositions system enabled, custom decompositions for operators 
in PennyLane can be added in a few ways depending on the application. 

The :func:`~.pennylane.transforms.decompose` transform offers the ability to inject
custom decompositions via two keyword arguments:

* ``fixed_decomps``: any decomposition for an operator type here will automatically 
  be chosen by the new algorithm, regardless of how resource efficient it may or 
  may not be.
* ``alt_decomps``: any decompositions for an operator type list here are added as 
  a *possible* decomposition rules the algorithm can choose based on its resource 
  efficiency.

Both keyword arguments above require a dictionary mapping PennyLane operator types to 
custom decompositions. Creating custom decompositions that the graph-based system 
can use involves a PennyLane quantum function that represents the decomposition, 
and a declaration of its resource requirements (gate counts) via :func:`~.pennylane.register_resources`.

Consider this example where we add a fixed decomposition to ``CNOT`` gates:

.. code-block:: python

    @qml.register_resources({qml.H: 2, qml.CZ: 1})
    def my_cnot(wires, **__):
        qml.H(wires=wires[1])
        qml.CZ(wires=wires)
        qml.H(wires=wires[1])

The :func:`~.pennylane.register_resources` accepts a dictionary mapping operator 
types within the custom decomposition to the number oftimes they occur in the decomposition. 
With the resources registered, this can be used with ``fixed_decomps`` or ``alt_decomps``:

.. code-block:: python

    @partial(qml.transforms.decompose, fixed_decomps={qml.CNOT: my_cnot})
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.CNOT(wires=[0, 1])
        return qml.state()

>>> print(qml.draw(circuit, level="device")())
0: ────╭●────┤  State
1: ──H─╰Z──H─┤  State

Note that the ``alt_decomps`` argument can handle multiple alternatives per operator
type:

.. code-block:: python

    @qml.register_resources({qml.H: 2, qml.CZ: 1})
    def my_cnot1(wires, **__):
        qml.H(wires=wires[1])
        qml.CZ(wires=wires)
        qml.H(wires=wires[1])

    @qml.register_resources({qml.RY: 2, qml.CZ: 1, qml.Z: 2})
    def my_cnot2(wires, **__):
        qml.RY(np.pi/2, wires[1])
        qml.Z(wires[1])
        qml.CZ(wires=wires)
        qml.RY(np.pi/2, wires[1])
        qml.Z(wires[1])

    @partial(
        qml.transforms.decompose,
        alt_decomps={qml.CNOT: [my_cnot1, my_cnot2]}
    )
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.CNOT(wires=[0, 1])
        return qml.state()

The decomposition that the algorithm chooses internally will be the most resource-efficient.
More details on creating complex decomposition rules that may depend on runtime 
parameters can be found in the usage details for :func:`~.pennylane.register_resources`.

Alternatively, new decomposition rules can be added to operators *globally* with 
the :func:`~.pennylane.add_decomps` function. This negates having to specify ``alt_decomps``
in every instance of the ``decompose`` transform. The following example globally 
adds the ``my_cnot1`` and ``my_cnot2`` decomposition rules to the ``qml.CNOT`` gate:

>>> qml.add_decomps(qml.CNOT, my_cnot1, my_cnot2)

The newly added rules for the ``qml.CNOT`` operator can be verified or inspected 
with the :func:`~.pennylane.list_decomps` function:

>>> my_new_rules = qml.list_decomps(qml.CNOT)[-2:]
>>> print(qml.draw(my_new_rules[1])(wires=[0, 1]))
0: ──────────────╭●──────────────┤  
1: ──RY(1.57)──Z─╰Z──RY(1.57)──Z─┤ 

Circuit cutting
---------------

Circuit cutting allows you to replace a circuit with ``N`` wires by a set of circuits with less than
``N`` wires (see also `Peng et. al <https://arxiv.org/abs/1904.00102>`_). Of course this comes with a cost: The smaller circuits
require a greater number of device executions to be evaluated.

In PennyLane, circuit cutting can be
activated by positioning :class:`~.pennylane.WireCut` operators at the desired cut locations, and
by decorating the QNode with the :func:`~.pennylane.cut_circuit` transform.

The example below shows how a three-wire circuit can be run on a two-wire device:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.cut_circuit
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.RY(0.9, wires=1)
        qml.RX(0.3, wires=2)

        qml.CZ(wires=[0, 1])
        qml.RY(-0.4, wires=0)

        qml.WireCut(wires=1)

        qml.CZ(wires=[1, 2])

        return qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))

Instead of being executed directly, the circuit will be partitioned into
smaller fragments according to the :class:`~.pennylane.WireCut` locations,
and each fragment will be executed multiple times. PennyLane automatically combines the results
of the fragment executions to recover the expected output of the original uncut circuit.

>>> x = np.array(0.531, requires_grad=True)
>>> circuit(0.531)
0.47165198882111165

Circuit cutting support is also differentiable:

>>> qml.grad(circuit)(x)
-0.276982865449393

.. note::

    Simulated quantum circuits that produce samples can be cut using
    the :func:`~.pennylane.cut_circuit_mc`
    transform, which is based on the Monte Carlo method.

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

>>> obs = [qml.Y(0), qml.X(0) @ qml.X(1), qml.Z(1)]
>>> coeffs = [1.43, 4.21, 0.97]
>>> groupings = qml.pauli.group_observables(obs, coeffs, 'anticommuting', 'lf')
>>> obs_groupings, coeffs_groupings = groupings
>>> obs_groupings
[[Z(1), X(0) @ X(1)], [Y(0)]]
>>> coeffs_groupings
[[0.97, 4.21], [1.43]]

This and more logic to manipulate Pauli observables is found in the :doc:`pauli module <../code/qml_pauli>`.
