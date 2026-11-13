.. role:: html(raw)
   :format: html

.. _intro_resource_profiling:

Resource profiling
==================

PennyLane can profile the resources of a :func:`~pennylane.qjit`-compiled QNode, such as the number
of qubits and the number of gates of each type, without running it on quantum hardware.

Three functions are available, ordered from the roughest estimate to the most precise count:
:func:`~pennylane.estimate` projects the circuit onto a target gate set,
:func:`~pennylane.analyze` inspects the compiled program at any stage of compilation, and
:func:`~pennylane.track` counts the operations of an execution.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.estimate
    ~pennylane.analyze
    ~pennylane.track

:html:`</div>`

The examples on this page use the following circuit, which contains a
:func:`~pennylane.transforms.cancel_inverses` pass that removes the two adjacent ``Toffoli`` gates:

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=4)

    @qp.qjit
    @qp.transforms.cancel_inverses
    @qp.qnode(dev)
    def circuit(x):
        qp.QFT(wires=range(3))
        qp.Toffoli(wires=[0, 1, 2])
        qp.Toffoli(wires=[0, 1, 2])
        qp.RX(x, wires=3)
        qp.MultiRZ(x, wires=[0, 1, 2, 3])
        return qp.expval(qp.Z(0))

Estimating resources
--------------------

The :func:`~pennylane.estimate` function projects the resources of a circuit onto a ``target``
gate set, by analyzing the :doc:`decomposition pathways <compiling_circuits>` of each of its
operations. The circuit is neither executed nor fully compiled, so this is fast:

>>> target = {"RZ", "Hadamard", "CNOT", "T", "Adjoint(T)", "GlobalPhase"}
>>> print(qp.estimate(circuit, target=target)(0.5).resources)
Quantum operations:
- Total: 64
  - Adjoint(T): 6
  - CNOT: 27
  - GlobalPhase: 3
  - Hadamard: 9
  - RZ: 11
  - T: 8
Measurement processes:
- expval(PauliZ): 1
Total wires: 4
Circuit Depth: Not computed

The estimate starts from the circuit as written (``level="top"``), so both ``Toffoli`` gates are
counted, even though :func:`~pennylane.transforms.cancel_inverses` removes them during compilation.
If the compile pipeline contains a :func:`~pennylane.decompose` transform with ``fixed_decomps`` or
``alt_decomps``, those decomposition rules are used to reach the target.

Symbolic resources
^^^^^^^^^^^^^^^^^^

Since the circuit is not executed, a loop whose number of iterations depends on a runtime argument
is counted *symbolically*. The estimate is then valid for every value of that argument:

.. code-block:: python

    @qp.qjit
    @qp.qnode(dev)
    def loop_circuit(n):
        qp.Hadamard(0)

        @qp.for_loop(n)
        def loop(i):
            qp.Toffoli(wires=[0, 1, 2])

        loop()
        return qp.expval(qp.Z(0))

>>> resources = qp.estimate(loop_circuit, target=target)(5).resources
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
Total wires: 4
Circuit Depth: Not computed

A value can be substituted for each symbolic variable with ``subs``:

>>> resources.subs(a=1000).quantum_operations
{'Adjoint(T)': 3000, 'CNOT': 6000, 'Hadamard': 2001, 'T': 4000}

Analyzing compiled programs
---------------------------

The :func:`~pennylane.analyze` function inspects the compiled program at a given stage of
compilation (``level``), without executing it. This makes it possible to see how each compilation
pass changes the resources of the circuit:

* ``level="top"`` (or ``0``): the circuit as written, before any compilation pass.
* ``level="user"``: the circuit after all user-specified passes.
* ``level="device"``: the circuit after the device has decomposed it into the gates it supports.

>>> qp.analyze(circuit, level="top")(0.5).resources.quantum_operations
{'MultiRZ': 1, 'QFT': 1, 'RX': 1, 'Toffoli': 2}
>>> qp.analyze(circuit, level="user")(0.5).resources.quantum_operations
{'MultiRZ': 1, 'QFT': 1, 'RX': 1}
>>> qp.analyze(circuit, level="device")(0.5).resources.quantum_operations
{'ControlledPhaseShift': 3, 'Hadamard': 3, 'MultiRZ': 1, 'RX': 1, 'SWAP': 1}

Like :func:`~pennylane.estimate`, :func:`~pennylane.analyze` counts loops with a runtime-dependent
number of iterations symbolically:

>>> print(qp.analyze(loop_circuit, level="top")(5).resources)
Symbolic Variables: a
Quantum operations:
- Total: a + 1
  - Hadamard: 1
  - Toffoli: a
Measurement processes:
- expval(PauliZ): 1
Total wires: 4
Circuit Depth: Not computed

Tracking executions
-------------------

The :func:`~pennylane.track` function counts the operations of an execution, after the device has
decomposed the circuit into the gates it supports. It returns the result of the execution alongside
the resources:

>>> result, profile = qp.track(circuit)(0.5)
>>> print(profile)
Device: lightning.qubit
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

Resources are tracked by mock-executing the circuit on ``null.qubit``, so ``result`` has the shape
and dtype of the real result, but not its values.

Since the circuit is executed, control flow is unrolled for the given arguments, and every loop
iteration is counted:

>>> _, profile = qp.track(loop_circuit)(5)
>>> profile.resources.quantum_operations
{'Toffoli': 5, 'Hadamard': 1}

Additional resources
--------------------

For other ways to inspect a circuit, such as :func:`~pennylane.specs` and circuit drawing, see
:doc:`inspecting circuits <inspecting_circuits>`. For more details on decompositions and on
specifying a gate set, see :doc:`compiling circuits <compiling_circuits>`.
