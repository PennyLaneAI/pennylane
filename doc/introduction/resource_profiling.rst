.. role:: html(raw)
   :format: html

.. _intro_resource_profiling:

Resource profiling
==================

Knowing what a quantum algorithm costs, such as how many qubits and how many gates of each type it
needs, is essential long before it can be run. In PennyLane2, :func:`~pennylane.track`,
:func:`~pennylane.analyze` and :func:`~pennylane.estimate` profile the resources of a
:func:`~pennylane.qjit`-compiled QNode, at any stage of its compilation and in terms of any target
gate set.

Choosing a tool
---------------

.. list-table::
    :header-rows: 1
    :widths: 25 40 35

    * - Function
      - What it does
      - Use it to
    * - :func:`~pennylane.track`
      - Executes the circuit on ``null.qubit`` and returns the result alongside the resources.
      - Get exact counts, with control flow unrolled for the given arguments.
    * - :func:`~pennylane.analyze`
      - Analyzes the compiled program without executing it.
      - See how each compilation pass changes the resources.
    * - :func:`~pennylane.estimate`
      - Projects the circuit onto a ``target`` gate set via its decomposition pathways.
      - Cost a circuit in a gate set other than the one it is written in.

Resources of a compiled program
-------------------------------

Consider the following circuit, which is compiled with :func:`~pennylane.qjit` and contains a
:func:`~pennylane.transforms.cancel_inverses` pass that removes the two adjacent ``Toffoli`` gates:

.. code-block:: python

    dev = qp.device("null.qubit", wires=4)

    @qp.qjit(capture=True)
    @qp.transforms.cancel_inverses
    @qp.qnode(dev)
    def circuit(x):
        qp.QFT(wires=range(3))
        qp.Toffoli(wires=[0, 1, 2])
        qp.Toffoli(wires=[0, 1, 2])
        qp.RX(x, wires=3)
        qp.MultiRZ(x, wires=[0, 1, 2, 3])
        return qp.expval(qp.Z(0))

:func:`~pennylane.analyze` inspects the compiled program at a given ``level`` of compilation,
without executing it. ``level="top"`` is the circuit as written, and ``level="user"`` is the
circuit after all user-specified passes:

>>> qp.analyze(circuit, level="top")(0.5).resources.quantum_operations
{'MultiRZ': 1, 'QFT': 1, 'RX': 1, 'Toffoli': 2}
>>> qp.analyze(circuit, level="user")(0.5).resources.quantum_operations
{'MultiRZ': 1, 'QFT': 1, 'RX': 1}

:func:`~pennylane.track` instead executes the circuit on ``null.qubit`` and reports the
resources it used, after the device has also decomposed the circuit into the gates it supports.
It returns the result of the execution alongside the resources:

>>> result, profile = qp.track(circuit)(0.5)
>>> print(profile)
Device: null.qubit
Device wires: 4
Shots: Shots(total=None)
Level: device
<BLANKLINE>
Quantum operations:
- Total: 9
  - MultiRZ: 1
  - RX: 1
  - SWAP: 1
  - ControlledPhaseShift: 3
  - Hadamard: 3
Measurement processes:
- expval(PauliZ): 1
Total wires: 4
Circuit Depth: 7

See :doc:`inspecting_circuits` for more details on these functions, including how
:func:`~pennylane.analyze` and :func:`~pennylane.track` differ in their treatment of control flow.

Projecting onto a target gate set
---------------------------------

The device's gate set is rarely the one an algorithm should be costed in. :func:`~pennylane.estimate`
projects the resources of the circuit as written onto a ``target`` gate set, by analyzing the
:doc:`decomposition pathways <compiling_circuits>` of each of its operations. The circuit is never
executed or fully compiled, so this is fast:

.. code-block:: python

    @qp.qjit(capture=True)
    @qp.transforms.cancel_inverses
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit(x):
        qp.Toffoli(wires=[0, 1, 2])
        qp.Toffoli(wires=[0, 1, 2])
        qp.RX(x, wires=2)
        qp.MultiRZ(x, wires=[0, 1, 2])
        return qp.expval(qp.Z(0))

>>> target = {"RZ", "Hadamard", "CNOT", "T", "Adjoint(T)"}
>>> qp.estimate(circuit, target=target)(0.5).resources.quantum_operations
{'Adjoint(T)': 6, 'CNOT': 16, 'Hadamard': 6, 'RZ': 2, 'T': 8}

Since :func:`~pennylane.estimate` starts from the circuit as written (``level="top"``), both
``Toffoli`` gates are counted. If the compile pipeline contains a :func:`~pennylane.decompose`
transform with ``fixed_decomps`` or ``alt_decomps``, those decomposition rules are used to reach the
target.

.. note::

    :func:`~pennylane.estimate` requires program capture, so the QNode must be compiled with
    ``qp.qjit(capture=True)``.

Neither :func:`~pennylane.analyze` nor :func:`~pennylane.estimate` executes the circuit, so a loop
whose number of iterations depends on a runtime argument is counted *symbolically*. The estimate is
then valid for every value of that argument:

.. code-block:: python

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit(n):
        qp.Hadamard(0)

        @qp.for_loop(n)
        def loop(i):
            qp.Toffoli(wires=[0, 1, 2])

        loop()
        return qp.expval(qp.Z(0))

>>> resources = qp.estimate(circuit, target=target)(5).resources
>>> print(resources)
Symbolic Variables: a
Quantum operations:
- Total: 15*a + 1
  - Adjoint(T): 3*a
  - CNOT: 6*a
  - Hadamard: 2*a + 1
  - T: 4*a
Measurement processes:
- expval(PauliZ): 1
Total wires: 3
Circuit Depth: Not computed

Substitute a value for each symbolic variable with ``subs``:

>>> resources.subs(a=1000).quantum_operations
{'Adjoint(T)': 3000, 'CNOT': 6000, 'Hadamard': 2001, 'T': 4000}

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.track
    ~pennylane.analyze
    ~pennylane.estimate

:html:`</div>`
