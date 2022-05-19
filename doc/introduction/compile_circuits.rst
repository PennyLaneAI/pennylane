.. role:: html(raw)
   :format: html

.. _intro_ref_opt:

Compiling circuits
==================

PennyLane offers a lot of functionality to "compile" circuits. We use the term "compilation"
here in a loose sense as the process of transforming one circuit (or a sequence of operators)
into one or more differing circuits (or sequences of operators). For example, such a transformation could
replace a gate type with another, fuse gates, exploit mathematical relations that simplify a Hamiltonian
to use fewer Pauli operators, or replace a circuit by a number of smaller circuits to fit on limited
quantum devices.

This functionality is mostly designed as **transforms**, which are also mentioned in the
section on :doc:`inspecting circuits </introduction/inspecting_circuits>`).

Compilation transforms
----------------------

PennyLane offers a number of transforms that take quantum functions and alter them to represent
quantum functions of optimized circuits:

* :func:`~pennylane.transforms.optimization.commute_controlled`: pushes commuting single-qubit
  gates through controlled operations.

* :func:`~pennylane.transforms.optimization.cancel_inverses`: removes adjacent pairs of operations
  that cancel out.

* :func:`~pennylane.transforms.optimization.merge_rotations`: combines adjacent rotation gates of
  the same type into a single gate, including controlled rotations.

* :func:`~pennylane.transforms.optimization.single_qubit_fusion`: acts on all sequences of
  single-qubit operations in a quantum function, and converts each
  sequence to a single ``Rot`` gate.

* :func:`~pennylane.transforms.optimization.pattern_matching`: optimize a circuit given a list of patterns
  of gates that decompose the identity.

* :func:`~pennylane.transforms.optimization.undo_swaps`: remove SWAP gates by running from right
  to left through the circuit changing the position of the qubits accordingly.

.. note::

    Most compilation transforms support just-in-time compilation with jax.jit.

The :func:`~pennylane.transforms.compile` transform allows you to chain together
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

The default behaviour of :func:`~pennylane.transforms.compile` is to apply a sequence of three
transforms: ``commute_controlled``, ``cancel_inverses``, and then ``merge_rotations``.


>>> print(qml.draw(qfunc)(0.2, 0.3, 0.4))
0: ──H───RX(0.6)──────────────────┤ ⟨Z⟩
1: ──H──╭X────────────────────╭C──┤
2: ──H──╰C────────RX(0.3)──Y──╰Z──┤


The :func:`~pennylane.transforms.compile` transform is flexible and accepts a custom pipeline
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

    The :class:`~pennylane.Barrier` operator can be used to prevent blocks of code from being merged during
    compilation.


For more details on :func:`:func:`~pennylane.transforms.compile` and the available compilation transforms, visit
`the compilation documentation
<https://pennylane.readthedocs.io/en/stable/code/qml_transforms.html#transforms-for-circuit-compilation>`_.

Grouping Pauli words
--------------------

Grouping Pauli words can be used for the optimizing the measurement of qubit
Hamiltonians. Along with groups of observables, post-measurement rotations can
also be obtained using :func:`~.optimize_measurements`:

.. code-block:: python

    >>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> post_rotations, diagonalized_groupings, grouped_coeffs = optimize_measurements(obs, coeffs)
    >>> post_rotations
    [[RY(-1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[1])],
     [RX(1.5707963267948966, wires=[0])]]

The post-measurement rotations can be used to diagonalize the partitions of
observables found.

For further details on measurement optimization, grouping observables through
solving the minimum clique cover problem, and auxiliary functions, refer to the
:doc:`/code/qml_grouping` subpackage.

Simplify Hamiltonians
---------------------

Use custom decompositions for unknown gates
-------------------------------------------

Circuit cutting
---------------



general manipulation (insert, merge amplitude gates, )

decompositions (expansion etc.)


