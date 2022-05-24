.. role:: html(raw)
   :format: html

.. _intro_ref_opt:

Compiling circuits
==================

PennyLane offers various functionality to compile circuits. We use the term "compilation"
here in a loose sense as the process of transforming one circuit (or a sequence of operators)
into one or more differing circuits (or sequences of operators). For example, such a transformation could
replace a gate type with another, fuse gates, exploit mathematical relations that simplify an observable,
or replace a large circuit by a number of smaller circuits.

Compilation functionality is mostly designed as **transforms**, which you can read up on in the
section on :doc:`inspecting circuits </introduction/inspecting_circuits>`.

Compilation transforms for circuit optimization
-----------------------------------------------

PennyLane offers a number of transforms that take quantum functions and alter them to represent
quantum functions of optimized circuits:

* :func:`~.pennylane.transforms.optimization.commute_controlled`: pushes commuting single-qubit
  gates through controlled operations.

* :func:`~.pennylane.transforms.optimization.cancel_inverses`: removes adjacent pairs of operations
  that cancel out.

* :func:`~.pennylane.transforms.optimization.merge_rotations`: combines adjacent rotation gates of
  the same type into a single gate, including controlled rotations.

* :func:`~.pennylane.transforms.optimization.single_qubit_fusion`: acts on all sequences of
  single-qubit operations in a quantum function, and converts each
  sequence to a single ``Rot`` gate.

* :func:`~.pennylane.transforms.optimization.pattern_matching`: optimizes a circuit given a list of patterns
  of gates that decompose the identity.

* :func:`~.pennylane.transforms.optimization.undo_swaps`: removes SWAP gates by running from right
  to left through the circuit and changing the position of the qubits accordingly.

.. note::

    Most compilation transforms support just-in-time compilation with ``jax.jit``.

The :func:`~.pennylane.transforms.compile` transform allows you to chain together
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

The default behaviour of :func:`~.pennylane.transforms.compile` is to apply a sequence of three
transforms: ``commute_controlled``, ``cancel_inverses``, and then ``merge_rotations``.

>>> print(qml.draw(qfunc)(0.2, 0.3, 0.4))
0: ──H───RX(0.6)──────────────────┤ ⟨Z⟩
1: ──H──╭X────────────────────╭C──┤
2: ──H──╰C────────RX(0.3)──Y──╰Z──┤


The :func:`~.pennylane.transforms.compile` transform is flexible and accepts a custom pipeline
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


For more details on :func:`~.pennylane.transforms.compile` and the available compilation transforms, visit
`the compilation documentation
<https://pennylane.readthedocs.io/en/stable/code/qml_transforms.html#transforms-for-circuit-compilation>`_.

Groups of commuting Pauli words
-------------------------------

Mutually commuting Pauli words can be measured simultaneously on a quantum computer.
When given an observable that is a linear combination of Pauli words, it can therefore
be useful to find such groups in order to optimize the number of circuit runs.

This can be done with the :func:`~.pennylane.group_observables` function:

>>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
>>> coeffs = [1.43, 4.21, 0.97]
>>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'anticommuting', 'lf')
>>> obs_groupings
[[PauliZ(wires=[1]), PauliX(wires=[0]) @ PauliX(wires=[1])],
 [PauliY(wires=[0])]]
>>> coeffs_groupings
[[0.97, 4.21], [1.43]]

For further details on measurement optimization, grouping observables through
solving the minimum clique cover problem, and other Pauli operator logic, refer to the
:doc:`/code/qml_grouping` subpackage.

.. note::

    PennyLane offers other methods to optimize measurements, such as qubit tapering
    found in the :mod:`~.pennylane.hf.tapering` module,
    which exploits molecular symmetries of Hamiltonians.

Custom decompositions for unknown operators
-------------------------------------------

PennyLane decomposes gates unknown to a particular device into other,
"lower-level" gates. As a user you may want to fine-tune this mechanism,
for example if you want your circuit to only use a certain gate set.

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
and ``Hadamard``, and of ``Hadamard`` into
``RZ`` and ``RY``.
We define the two decompositions like so, and pass them to a device:

.. code-block:: python

    def custom_cnot(wires):
        return [
            qml.Hadamard(wires=wires[1]),
            qml.CZ(wires=[wires[0], wires[1]]),
            qml.Hadamard(wires=wires[1])
        ]

    def custom_hadamard(wires):
        return [
            qml.RZ(np.pi, wires=wires),
            qml.RY(np.pi / 2, wires=wires)
        ]

    # Can pass the operation itself, or a string
    custom_decomps = {qml.CNOT : custom_cnot, "Hadamard" : custom_hadamard}

    decomp_dev = qml.device("default.qubit", wires=3, custom_decomps=custom_decomps)
    decomp_qnode = qml.QNode(circuit, decomp_dev)

Now when we draw or run a QNode on this device, the gates will be expanded
according to our specifications:

>>> print(qml.draw(decomp_qnode, expansion_strategy="device")(weights))
0: ──RX(0.4)──────────────────────╭C──RZ(3.14)──RY(1.57)──────────────────────────╭Z──RZ(3.14)──RY(1.57)──┤ ⟨Z⟩
1: ──RX(0.5)──RZ(3.14)──RY(1.57)──╰Z──RZ(3.14)──RY(1.57)──╭C──────────────────────│───────────────────────┤
2: ──RX(0.6)──RZ(3.14)──RY(1.57)──────────────────────────╰Z──RZ(3.14)──RY(1.57)──╰C──────────────────────┤

If the custom decomposition is only supposed to be used in a specific code context,
a separate context manager :func:`~.pennylane.set_decomposition` can be used:


>>> with qml.transforms.set_decomposition(custom_decomps, original_dev):
...     print(qml.draw(original_qnode, expansion_strategy="device")(weights))
0: ──RX(0.4)──────────────────────╭C──RZ(3.14)──RY(1.57)──────────────────────────╭Z──RZ(3.14)──RY(1.57)──┤ ⟨Z⟩
1: ──RX(0.5)──RZ(3.14)──RY(1.57)──╰Z──RZ(3.14)──RY(1.57)──╭C──────────────────────│───────────────────────┤
2: ──RX(0.6)──RZ(3.14)──RY(1.57)──────────────────────────╰Z──RZ(3.14)──RY(1.57)──╰C──────────────────────┤

Circuit cutting
---------------

Circuit cutting allows you to replace a circuit with ``N`` wires by a set of circuits with less than
``N`` wires (see also `Peng et. al <https://arxiv.org/abs/1904.00102>`_). Of course this comes with a cost: The smaller circuits
require a greater number of device executions to be evaluated.

In PennyLane, circuit cutting can be
activated by positioning :class:`~.pennylane.WireCut` operators at the desired cut locations, and
by decorating the QNode with the :func:`~.pennylane.transforms.cut_circuit` transform.

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

Instead of executing the circuit directly, it will be partitioned into
smaller fragments according to the :class:`~.pennylane.WireCut` locations,
and each fragment executed multiple times. PennyLane automatically combines the results
of the fragment executions to recover the expected output of the original uncut circuit.

>>> x = np.array(0.531, requires_grad=True)
>>> circuit(0.531)
0.47165198882111165

Circuit cutting support is also differentiable:

>>> qml.grad(circuit)(x)
-0.276982865449393

Simulated quantum circuits that produce samples can be cut using
the :func:`~.pennylane.transforms.cut_circuit_mc`
transform based on the Monte Carlo method:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2, shots=1000)

    @qml.cut_circuit_mc
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(0.89, wires=0)
        qml.RY(0.5, wires=1)
        qml.RX(1.3, wires=2)

        qml.CNOT(wires=[0, 1])
        qml.WireCut(wires=1)
        qml.CNOT(wires=[1, 2])

        qml.RX(x, wires=0)
        qml.RY(0.7, wires=1)
        qml.RX(2.3, wires=2)
        return qml.sample(wires=[0, 2])

>>> x = 0.3
>>> circuit(x)
tensor([[1, 1],
        [0, 1],
        [0, 1],
        ...,
        [0, 1],
        [0, 1],
        [0, 1]], requires_grad=True)

The samples are drawn from the same distribution that the original
circuit gives rise to.
