.. role:: html(raw)
   :format: html

.. _intro_ref_compile_circuits:

Compiling circuits
==================

The :doc:`qjit` page provides an overview of quantum
just-in-time compilation with :func:`~pennylane.qjit`. Here, "compilation" means
*quantum circuit transformations*, which can include pattern matching, gate
cancellations, gate decomposition, and more.

PennyLane2 includes a growing library of quantum circuit transforms:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.decompose
    ~pennylane.transforms.gridsynth
    ~pennylane.transforms.clifford_t_decomposition
    ~pennylane.transforms.parity_synth
    ~pennylane.transforms.merge_rotations
    ~pennylane.transforms.cancel_inverses
    ~pennylane.transforms.combine_global_phases
    ~pennylane.transforms.disentangle_cnot
    ~pennylane.transforms.disentangle_swap
    ~pennylane.transforms.combine_global_phases
    ~transforms.to_ppr
    ~transforms.ppm_compilation
    ~transforms.ppr_to_ppm
    ~transforms.merge_ppr_ppm
    ~transforms.reduce_t_depth
    ~transforms.commute_ppr
    ~transforms.decompose_arbitrary_ppr

:html:`</div>`

Applying compilation passes
---------------------------

Transforms can be applied on QNodes using the decorator syntax, then decorated
with :func:`~pennylane.qjit`:

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=1)

    @qp.qjit(capture=True)
    @qp.transforms.gridsynth(epsilon=1e-4)
    @qp.qnode(dev)
    def circuit(x):
        qp.Hadamard(0)
        qp.RZ(x, wires=0)
        return qp.expval(qp.X(0))

>>> qp.analyze(circuit)(0.1).resources.quantum_operations
{'Adjoint(S)': 1, 'GlobalPhase': 1, 'Hadamard': 43, 'Identity': 1, 'PauliX': 1, 'PauliY': 1, 'PauliZ': 1, 'S': 21, 'T': 42}

Transforms can be stacked, allowing for the application of multiple compilation
passes on a QNode:

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=1)

    @qp.qjit(capture=True)
    @qp.transforms.merge_rotations
    @qp.transforms.cancel_inverses
    @qp.qnode(device=dev)
    def circuit(x, y):
        qp.X(wires=0)
        qp.Hadamard(wires=0)
        qp.Hadamard(wires=0)
        qp.X(wires=0)
        qp.RX(x, wires=0)
        qp.RX(y, wires=0)
        return qp.expval(qp.Z(0))

>>> qp.analyze(circuit)(0.1, 0.2).resources.quantum_operations
Device: lightning.qubit
Device wires: 1
Shots: Shots(total=None)
Levels:
- 0: Before MLIR Passes
- 1: cancel-inverses
- 2: merge-rotations
<BLANKLINE>
↓Metric         Level→ |  0 |  1 |  2
-------------------------------------
Quantum operations:    |
- Total                |  6 |  2 |  1
  - Hadamard           |  2 |  0 |  0
  - PauliX             |  2 |  0 |  0
  - RX                 |  2 |  2 |  1
Measurement processes: |
- expval(PauliZ)       |  1 |  1 |  1
Total wires            |  1 |  1 |  1

Alternatively, multiple transforms can be chained together into custom
compilation pipelines with :class:`~.CompilePipeline`.

.. code-block:: python

    pipeline = qp.CompilePipeline(
        qp.transforms.cancel_inverses,
        qp.transforms.merge_rotations,
    )

    @qp.qjit(capture=True)
    @pipeline
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit(x, y):
        qp.RX(x, wires=0)
        qp.CNOT(wires=[0, 1])
        qp.Hadamard(wires=1)
        qp.Hadamard(wires=1)
        qp.CNOT(wires=[0, 1])
        qp.RX(y, wires=0)
        return qp.expval(qp.Z(0))

For more details on :class:`~.CompilePipeline` and available compilation passes,
visit `the compilation documentation <../code/qp_transforms.html#transforms-for-circuit-compilation>`_.

Circuit decomposition
---------------------

When compiling a circuit, it is often beneficial to decompose the circuit into a
set of specific operators. Every operator can have multiple different
decompositions into other operators, and the :func:`~pennylane.decompose`
compilation pass finds the most resource-efficient decomposition pathway to the
desired target.

The example below demonstrates how a three-wire circuit can be decomposed using
a pre-defined set of gates:

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=3)
    allowed_gates = {qp.Toffoli, qp.RX, qp.RZ, qp.GlobalPhase}

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=allowed_gates)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=[0])
        qp.Toffoli(wires=[0,1,2])
        return qp.expval(qp.Z(0))

The :class:`~pennylane.Hadamard` will be decomposed into rotation gates and a
global phase, while :class:`~pennylane.Toffoli` remains since it is in the
target gate set.

>>> qp.analyze(circuit)().resources.quantum_operations
{'GlobalPhase': 1, 'RX': 1, 'RZ': 2, 'Toffoli': 1}

Gates in the gate set can also be specified by name. The example below
decomposes a ``Toffoli`` gate into a subset of Clifford + T gates:

.. code-block:: python

    @qp.qjit(capture=True)
    @qp.decompose(gate_set={"H", "T", "Adjoint(T)", "CNOT"})
    @qp.qnode(dev)
    def circuit():
        qp.Toffoli(wires=[0,1,2])
        return qp.expval(qp.Z(0))

>>> qp.analyze(circuit)().resources.quantum_operations
{'Adjoint(T)': 3, 'CNOT': 6, 'Hadamard': 2, 'T': 4}

Influencing decomposition pathways
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Briefly, the :func:`~pennylane.decompose` compilation pass works by constructing
a weighted graph and finding the "shortest" (least resource-intensive) path
between source nodes (original operators in the circuit) and target nodes
(operators in the target gate set). This algorithm can be influenced in several
ways:

* :func:`qp.decompose(..., exclude_decomps={...}) <~pennylane.decompose>`:
  The ``exclude_decomps`` argument excludes given decomposition rules from the
  decomposition pathway—they cannot be explored as a path to the target gate
  set.
* :func:`qp.decompose(..., fixed_decomps={...}) <~pennylane.decompose>`:
  The ``fixed_decomps`` argument fixes the exploration of decomposition pathways
  that only use the given decomposition rules (if the given operator is not in
  the target gate set).
* **Target gate set weights**: It is typical for certain gates to be more costly
  than others (e.g., :class:`~pennylane.T` and :class:`~pennylane.Toffoli` gates
  are more expensive than :class:`~pennylane.H`). Gate-by-gate costs can be
  given to target gates to influence the resource intensiveness of certain
  decomposition pathways. In the example below, the :class:`~pennylane.H` gate
  is decomposed into :class:`~pennylane.RZ` :math:`\otimes` :class:`~pennylane.RY`
  instead of :class:`~pennylane.RZ` :math:`\otimes` :class:`~pennylane.RX`
  :math:`\otimes` :class:`~pennylane.RZ` (modulo global phases) since
  :class:`~pennylane.RX` is 100-times more costly.

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=3)
    allowed_gates = {qp.Toffoli: 1, qp.RX: 100, qp.RY: 1, qp.RZ: 1, qp.GlobalPhase: 1}

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=allowed_gates)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(wires=[0])
        qp.Toffoli(wires=[0,1,2])
        return qp.expval(qp.Z(0))

>>> qp.analyze(circuit)().resources.quantum_operations
{'GlobalPhase': 1, 'RY': 1, 'RZ': 1, 'Toffoli': 1}
