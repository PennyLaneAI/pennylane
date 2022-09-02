.. role:: html(raw)
   :format: html

.. _intro_ref_compile:

Compiling circuits
==================

PennyLane offers multiple tools for compiling circuits. We use the term "compilation"
here in a loose sense as the process of transforming one circuit 
into one or more differing circuits. A circuit could be either a quantum function or a sequence of operators. For example, such a transformation could
replace a gate type with another, fuse gates, exploit mathematical relations that simplify an observable,
or replace a large circuit by a number of smaller circuits.

Compilation functionality is mostly designed as **transforms**, which you can read up on in the
section on :doc:`inspecting circuits </introduction/inspecting_circuits>`.

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

:html:`</div>`

.. note::

    Most compilation transforms support just-in-time compilation with ``jax.jit``.

The :func:`~.pennylane.compile` transform allows you to chain together
sequences of quantum function transforms into custom circuit optimization pipelines.

For example, take the following decorated quantum function:

.. code-block:: python

    dev = qml.device('default.qubit', wires=[0, 1, 2])

    @qml.qnode(dev)
    @qml.compile()
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
        qml.PauliY(wires=2)
        qml.CZ(wires=[1, 2])
        return qml.expval(qml.PauliZ(wires=0))

The default behaviour of :func:`~.pennylane.compile` applies a sequence of three
transforms: :func:`~.pennylane.transforms.commute_controlled`, :func:`~.pennylane.transforms.cancel_inverses`,
and then :func:`~.pennylane.transforms.merge_rotations`.

>>> print(qml.draw(qfunc)(0.2, 0.3, 0.4))
0: ──H───RX(0.6)──────────────────┤ ⟨Z⟩
1: ──H──╭X────────────────────╭C──┤
2: ──H──╰C────────RX(0.3)──Y──╰Z──┤


The :func:`~.pennylane.compile` transform is flexible and accepts a custom pipeline
of quantum function transforms (you can even write your own!).
For example, if we wanted to only push single-qubit gates through
controlled gates and cancel adjacent inverses, we could do:

.. code-block:: python

    from pennylane.transforms import commute_controlled, cancel_inverses
    pipeline = [commute_controlled, cancel_inverses]

    @qml.qnode(dev)
    @qml.compile(pipeline=pipeline)
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
        qml.PauliY(wires=2)
        qml.CZ(wires=[1, 2])
        return qml.expval(qml.PauliZ(wires=0))

>>> print(qml.draw(qfunc)(0.2, 0.3, 0.4))
0: ──H───RX(0.4)──RX(0.2)────────────────────────────┤ ⟨Z⟩
1: ──H──╭X───────────────────────────────────────╭C──┤
2: ──H──╰C────────RZ(0.4)──RZ(-0.4)──RX(0.3)──Y──╰Z──┤

.. note::

    The :class:`~.pennylane.Barrier` operator can be used to prevent blocks of code from being merged during
    compilation.


For more details on :func:`~.pennylane.compile` and the available compilation transforms, visit
`the compilation documentation
<../code/qml_transforms.html#transforms-for-circuit-compilation>`_.

Custom decompositions
---------------------

PennyLane decomposes gates unknown to the device into other, "lower-level" gates. As a user, you may want to fine-tune this mechanism. For example, you may wish your circuit to use different fundamental gates.

For example, suppose we would like to implement the following QNode:

.. code-block:: python

    def circuit(weights):
        qml.BasicEntanglerLayers(weights, wires=[0, 1, 2])
        return qml.expval(qml.PauliZ(0))

    original_dev = qml.device("default.qubit", wires=3)
    original_qnode = qml.QNode(circuit, original_dev)

>>> weights = np.array([[0.4, 0.5, 0.6]])
>>> print(qml.draw(original_qnode, expansion_strategy="device")(weights))
0: ──RX(0.4)──╭C──────╭X──┤ ⟨Z⟩
1: ──RX(0.5)──╰X──╭C──│───┤
2: ──RX(0.6)──────╰X──╰C──┤

Now, let's swap out PennyLane's default decomposition of the ``CNOT`` gate into ``CZ``
and ``Hadamard``.
We define the custom decompositions like so, and pass them to a device:

.. code-block:: python

    def custom_cnot(wires):
        return [
            qml.Hadamard(wires=wires[1]),
            qml.CZ(wires=[wires[0], wires[1]]),
            qml.Hadamard(wires=wires[1])
        ]

    custom_decomps = {qml.CNOT: custom_cnot}

    decomp_dev = qml.device("default.qubit", wires=3, custom_decomps=custom_decomps)
    decomp_qnode = qml.QNode(circuit, decomp_dev)

Now when we draw or run a QNode on this device, the gates will be expanded
according to our specifications:

>>> print(qml.draw(decomp_qnode, expansion_strategy="device")(weights))
0: ──RX(0.40)────╭C──H───────╭Z──H─┤  <Z>
1: ──RX(0.50)──H─╰Z──H─╭C────│─────┤
2: ──RX(0.60)──H───────╰Z──H─╰C────┤

.. note::
    If the custom decomposition is only supposed to be used in a specific code context,
    a separate context manager :func:`~.pennylane.transforms.set_decomposition` can be used.

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

        return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))

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

On a lower level, the :func:`~.pennylane.grouping.group_observables` function can be used to split lists of
observables and coefficients:

>>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
>>> coeffs = [1.43, 4.21, 0.97]
>>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'anticommuting', 'lf')
>>> obs_groupings
[[PauliZ(wires=[1]), PauliX(wires=[0]) @ PauliX(wires=[1])],
 [PauliY(wires=[0])]]
>>> coeffs_groupings
[[0.97, 4.21], [1.43]]

This and more logic to manipulate Pauli observables is found in the :mod:`~.pennylane.grouping` module.

Simplifying Operators
----------------------

PennyLane offers the :func:`~.pennylane.simplify` function to simplify single operators, quantum
functions, QNodes and tapes. This function reduces the arithemtic depth of all the given operators
to its minimum, groups like terms in sums and products, resolves products of Pauli 
operators and combines identical rotation gates by summing its angles.

For example, lets simplify the following operator:

>>> nested_op = qml.prod(qml.prod(qml.PauliX(0), qml.op_sum(qml.RX(1, 0), qml.PauliX(0))), qml.RX(1, 0))
>>> qml.simplify(nested_op)
PauliX(wires=[0]) @ RX(2.0, wires=[0]) + RX(1.0, wires=[0])

Several simplifications steps are happening here. First of all, the nested products are removed:
`qml.prod(qml.PauliX(0), qml.op_sum(qml.RX(1, 0), qml.PauliX(0)), qml.RX(1, 0))`
Then the product of sums is transformed into a sum of products:
`qml.sum(qml.prod(qml.PauliX(0), qml.RX(1, 0), qml.RX(1, 0)), qml.prod(qml.PauliX(0), qml.PauliX(0), qml.RX(1, 0)))`
And finally like terms in the obtained products are grouped together, removing all identities: 
`qml.sum(qml.prod(qml.PauliX(0), qml.RX(2, 0)), qml.RX(1, 0))`

As mentioned earlier, we can also simplify QNode objects:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.simplify
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=1)
        qml.RZ(x[2], wires=2)
        qml.RX(-0.4, wires=0)
        qml.RZ(-2, wires=2)
        qml.RY(2, wires=1)
        return qml.probs([0, 1, 2])

    x = [1, 2, 3]
    
>>> print(qml.draw(circuit)(x))
0: ─RX(13.17)─┤ Probs
1: ─RY(4.00)─┤ Probs
2: ─RZ(13.57)─┤ Probs