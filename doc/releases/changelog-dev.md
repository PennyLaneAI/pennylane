# Release 0.45.0 (development release)

<h3>New features since last release</h3>

* Decomposition rules are re-written in a `qjit` compatible way so that they can be lowered to Catalyst/MLIR. Rules for the
  following `SymbolicOps` have been re-written.

  - :class:`qp.ops.op_math.Pow` [(#9199)](https://github.com/PennyLaneAI/pennylane/pull/9199) [(#9213)](https://github.com/PennyLaneAI/pennylane/pull/9213)
  - :class:`qp.ops.Adjoint` [(#9190)](https://github.com/PennyLaneAI/pennylane/pull/9190)

* A new angle solver has been added to find QSVT phase angles faster for large-degree polynomials.
  This can be accessed by setting `angle_solver = 'iterative-optax'` in `qp.qsvt` and
  `qp.poly_to_angles` and provides a significant advantage when repeatedly evaluating the
  same-degree polynomial with different coefficients.
  [(#8685)](https://github.com/PennyLaneAI/pennylane/pull/8685)

* Added the function :func:`~.drawer.label` to attach custom labels to operator instances
  for circuit drawing.
  [(#9078)](https://github.com/PennyLaneAI/pennylane/pull/9078)

* Added the function :func:`~.fourier.mark` to mark an operator as an input-encoding gate
  for :func:`~.fourier.circuit_spectrum`, and :func:`~.fourier.qnode_spectrum`.
  [(#9078)](https://github.com/PennyLaneAI/pennylane/pull/9078)

* A new state preparation method called :class:`~.SumOfSlatersPrep` is now available.
  It prepares sparse states using a smaller dense state preparation, :class:`~.QROM`\ s and
  reversible bit encodings.
  [(#8964)](https://github.com/PennyLaneAI/pennylane/pull/8964)
  [(#8997)](https://github.com/PennyLaneAI/pennylane/pull/8997)
  [(#9228)](https://github.com/PennyLaneAI/pennylane/pull/9228)
  [(#9323)](https://github.com/PennyLaneAI/pennylane/pull/9323)

  Consider a sparse state on five qubits, specified by normalized coefficients and statevector
  indices pointing to the populated computational basis states:

  ```python
  import numpy as np
  import pennylane as qp
  coefficients = [0.25, 0.25j, -0.25, 0.5, 0.5, 0.25, -0.25j, 0.25, -0.25, 0.25]
  coefficients = np.array(coefficients)
  indices = (0, 1, 4, 13, 14, 17, 19, 22, 23, 25)
  wires = range(5)
  ```

  And this is all the information we require to create the state
  preparation: ``coefficients``, ``indices``, and ``wires``.

  ```python
  qp.decomposition.enable_graph()
  gate_set = {"QROM", "TemporaryAND", "Adjoint(TemporaryAND)", "StatePrep", "CNOT", "X"}

  @qp.transforms.resolve_dynamic_wires(min_int=max(wires)+1)
  @qp.decompose(gate_set=gate_set, num_work_wires=11)
  @qp.qnode(qp.device("lightning.qubit", wires=16))
  def circuit():
      qp.SumOfSlatersPrep(coefficients, wires, indices)
      return qp.state()
  ```
  ```pycon
  >>> prepared_state = circuit()[::2**11] # Slice the state, as there are eleven work wires
  >>> where = np.where(np.abs(prepared_state) > 1e-12)
  >>> print(where)
  (array([ 0,  1,  4, 13, 14, 17, 19, 22, 23, 25]),)
  >>> with np.printoptions(precision=2, suppress=True): # doctest: +SKIP
  ...   print(prepared_state[where])
  [ 0.25+0.j    0.  +0.25j -0.25+0.j    0.5 +0.j    0.5 +0.j    0.25+0.j
    0.  -0.25j  0.25+0.j   -0.25+0.j    0.25+0.j  ]
  ```

* Moved :func:`~.math.binary_finite_reduced_row_echelon` to a new file and added further
  linear algebraic functionalities over :math:`\mathbb{Z}_2`:
  [(#8982)](https://github.com/PennyLaneAI/pennylane/pull/8982)

  - :func:`~.math.binary_is_independent` computes whether a vector is linear lindependent of
    a basis of binary vectors over :math:`\mathbb{Z}_2`.
  - :func:`~.math.binary_matrix_rank` computes the rank over :math:`\mathbb{Z}_2` of a binary matrix.
  - :func:`~.math.binary_solve_linear_system` solves a linear system of the form :math:`A\cdot x=b`
    with binary matrix :math:`A` and binary coefficient vector :math:`b` over :math:`\mathbb{Z}_2`.
  - :func:`~.math.binary_select_basis` selects linearly independent columns out of a collection
    of binary column vectors. The result forms a basis for the columnspace of the input. The
    columns that are not selected are returned as well.

* Added the Catalyst version to :func:`~.about`.
  [(#9050)](https://github.com/PennyLaneAI/pennylane/pull/9050)

* Added a convenience function :func:`~.math.ceil_log2` that computes the ceiling of the base-2
  logarithm of its input and casts the result to an ``int``. It is equivalent
  to ``int(np.ceil(np.log2(n)))``.
  [(#8972)](https://github.com/PennyLaneAI/pennylane/pull/8972)
  [(#9069)](https://github.com/PennyLaneAI/pennylane/pull/9069)

* Added a ``qp.gate_sets`` that contains pre-defined gate sets such as ``qp.gate_sets.CLIFFORD_T_PLUS_RZ``
  that can be plugged into the ``gate_set`` argument of the :func:`~pennylane.transforms.decompose` transform.
  [(#8915)](https://github.com/PennyLaneAI/pennylane/pull/8915)
  [(#9045)](https://github.com/PennyLaneAI/pennylane/pull/9045)
  [(#9259)](https://github.com/PennyLaneAI/pennylane/pull/9259)

* Adds a new `qp.templates.Subroutine` class for adding a layer of abstraction for
  quantum functions. These objects can now return classical values or mid circuit measurements,
  and are compatible with Program Capture Catalyst. Any `Operator` with a single definition
  in terms of its implementation, a more complicated call signature, and that exists
  at a higher, algorithmic layer of abstraction should switch to using this class instead
  of `Operator`/ `Operation`.
  [(#8929)](https://github.com/PennyLaneAI/pennylane/pull/8929)
  [(#9080)](https://github.com/PennyLaneAI/pennylane/pull/9080)
  [(#9096)](https://github.com/PennyLaneAI/pennylane/pull/9096)
  [(#9070)](https://github.com/PennyLaneAI/pennylane/pull/9070)
  [(#9097)](https://github.com/PennyLaneAI/pennylane/pull/9097)
  [(#9138)](https://github.com/PennyLaneAI/pennylane/pull/9138)
  [(#9119)](https://github.com/PennyLaneAI/pennylane/pull/9119)
  [(#9151)](https://github.com/PennyLaneAI/pennylane/pull/9151)
  [(#9172)](https://github.com/PennyLaneAI/pennylane/pull/9172)
  [(#9180)](https://github.com/PennyLaneAI/pennylane/pull/9180)
  [(#9177)](https://github.com/PennyLaneAI/pennylane/pull/9177)
  [(#9191)](https://github.com/PennyLaneAI/pennylane/pull/9191)
  [(#9176)](https://github.com/PennyLaneAI/pennylane/pull/9176)

  ```python
  from pennylane.templates import Subroutine

  @Subroutine
  def MyTemplate(x, y, wires):
      qp.RX(x, wires[0])
      qp.RY(y, wires[0])

  @qp.qnode(qp.device('default.qubit'))
  def c():
      MyTemplate(0.1, 0.2, 0)
      return qp.state()
  ```

  ```pycon
  >>> print(qp.draw(c)())
  0: ──MyTemplate(0.10,0.20)─┤  State

  ```

The following classes have been ported over:
- `qp.BasisRotation` [(#9026)](https://github.com/PennyLaneAI/pennylane/pull/9026)

* Added a `qp.decomposition.local_decomps` context
  manager that allows one to add decomposition rules to an operator, only taking effect within the context.
  [(#8955)](https://github.com/PennyLaneAI/pennylane/pull/8955)
  [(#8998)](https://github.com/PennyLaneAI/pennylane/pull/8998)

* Added a `qp.workflow.get_compile_pipeline(qnode, level)(*args, **kwargs)` function to extract the
  compile pipeline of a given QNode at a specific level.
  [(#8979)](https://github.com/PennyLaneAI/pennylane/pull/8979)

* Added a `strict` keyword to the :func:`~pennylane.transforms.decompose` transform that, when set to ``False``,
  allows the decomposition graph to treat operators without a decomposition as part of the gate set.
  [(#9025)](https://github.com/PennyLaneAI/pennylane/pull/9025)

* New decomposition rules are added to `Evolution` and `RZ`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)
  [(#9049)](https://github.com/PennyLaneAI/pennylane/pull/9049)

* The custom `adjoint` method of qutrit operators are implemented as decomposition rules compatible with the
  new graph-based decomposition system.
  [(#9056)](https://github.com/PennyLaneAI/pennylane/pull/9056)

* A new :func:`~binary_decimals` function was added to enable easy translation of rotation angles to the binary representation of their decimals.
  This is important for discretization steps, for example via [phase gradient decompositions](https://pennylane.ai/compilation/phase-gradient/).
  [(#9117)](https://github.com/PennyLaneAI/pennylane/pull/9117)

* The :func:`~.transforms.disentangle_cnot` and :func:`~.transforms.disentangle_swap` are now
  available in PennyLane instead of only Catalyst. These compilation passes simplify rendundant
  ``CNOT`` and ``SWAP`` gates.
  [(#9133)](https://github.com/PennyLaneAI/pennylane/pull/9133)

* Decomposition rules can now be assigned custom names using the ``name`` argument in :func:`qp.register_resources <pennylane.decomposition.register_resources>`. This makes it easier to identify specific rules.
  [(#9257)](https://github.com/PennyLaneAI/pennylane/pull/9257)

* Added ``PauliSentence.prune`` and ``FermiSentence.prune`` that removes terms with coefficients below a provided threshold.
  [(#9278)](https://github.com/PennyLaneAI/pennylane/pull/9278)

* Added :func:`~.decomposition.inspect_decomps` that allows users to visualize and inspect the available decomposition rules
  for a concrete operator instance.
  [(#9322)](https://github.com/PennyLaneAI/pennylane/pull/9322)

  ```pycon
  >>> print(qp.inspect_decomps(qp.CRX(0.5, wires=[0, 1])))
  Decomposition 0 (name: _crx_to_rx_cz)
  0: ───────────╭●────────────╭●─┤
  1: ──RX(0.25)─╰Z──RX(-0.25)─╰Z─┤
  Gate Count: {RX: 2, CZ: 2}
  <BLANKLINE>
  Decomposition 1 (name: _crx_to_rz_ry)
  0: ─────────────────────╭●────────────╭●────────────┤
  1: ──RZ(1.57)──RY(0.25)─╰X──RY(-0.25)─╰X──RZ(-1.57)─┤
  Gate Count: {RZ: 2, RY: 2, CNOT: 2}
  <BLANKLINE>
  Decomposition 2 (name: _crx_to_h_crz)
  0: ────╭●───────────┤
  1: ──H─╰RZ(0.50)──H─┤
  Gate Count: {Hadamard: 2, CRZ: 1}
  <BLANKLINE>
  Decomposition 3 (name: _crx_to_ppr)
  0: ───────────╭RZX(-0.25)─┤
  1: ──RX(0.25)─╰RZX(-0.25)─┤
  Gate Count: {PauliRot(pauli_word=ZX): 1, PauliRot(pauli_word=X): 1}

  ```

<h3>Improvements 🛠</h3>

* :func:`~.specs` now supports ``level="user"`` for workflows compiled with :func:`~.qjit`. This returns circuit specifications after all user-specified transforms have been applied.
  [(#9307)](https://github.com/PennyLaneAI/pennylane/pull/9307)

* With program capture and `for_loop` and `while_loop`, const closure variables with dynamic shapes
  can now be combined with explicit inputs with dynamic shapes when they have matching shapes.
  [(#9275)](https://github.com/PennyLaneAI/pennylane/pull/9275)
  [(#9335)](https://github.com/PennyLaneAI/pennylane/pull/9335)

* Added another decomposition to `MultiControlledX` with two control wires and at least one zeroed
  work wire that has been passed explicitly. It decomposes into a pair of `TemporaryAND` and a
  `CNOT`.
  [(#9291)](https://github.com/PennyLaneAI/pennylane/pull/9291)

* Operations using ``FermiWord`` are now much faster due to various performance improvements to the class
  [(#9283)](https://github.com/PennyLaneAI/pennylane/pull/9283)

* Replaced the O(n²) incremental ``@=`` operator chaining in ``qp.pauli.string_to_pauli_word`` and
  ``qp.pauli.binary_to_pauli`` with a single ``qp.prod(*tuple_of_ops)`` call, collecting operators via
  generator expressions. These operators are now much faster for large Pauli strings.
  [(#9271)](https://github.com/PennyLaneAI/pennylane/pull/9271)

* Operations using ``PauliSentence`` are now much faster due to additional memoization in ``PauliWord.__hash__``
  [(#9261)](https://github.com/PennyLaneAI/pennylane/pull/9261)

* The documentation of the QASM interpreter class has been updated to include `Raises` error sections for its methods.
  [(#9244)](https://github.com/PennyLaneAI/pennylane/pull/9244)

* Removed some wire reusage in :class:`~.Select` that is not consistent with the approach to work
  wires elsewhere in PennyLane, and that was not taken into account in the resource functions
  for the graph-based decomposition system (leading to decompositions not being resolved correctly).
  Also simplified the resource calculation of one decomposition of `Select`.
  [(#9222)](https://github.com/PennyLaneAI/pennylane/pull/9222)

* The decomposition of :class:`~.TemporaryAND` is now compatible with traced control values.
  [(#9157)](https://github.com/PennyLaneAI/pennylane/pull/9157)

* The decomposition of :class:`~.MultiRZ` is now compatible with traced wires.
  [(#9157)](https://github.com/PennyLaneAI/pennylane/pull/9157)

* The decomposition of :class:`~.DiagonalQubitUnitary` is now compatible with traced data.
  [(#9157)](https://github.com/PennyLaneAI/pennylane/pull/9157)

* `Callables` defining quantum operations can now be passed to the
  `compute_op`, `target_op` and `uncompute_op` arguments of `qp.change_op_basis`.
  [(#9163)](https://github.com/PennyLaneAI/pennylane/pull/9163)

* The `default.qubit` device now supports parameter-broadcasted global phases.
  [(#9148)](https://github.com/PennyLaneAI/pennylane/pull/9148)

* :class:`~.MottonenStatePreparation` now supports parameter broadcasting in its decomposition.
  [(#9148)](https://github.com/PennyLaneAI/pennylane/pull/9148)

* `qp.math.givens_decomposition` and `qp.BasisRotation` are now compatible with `qjit` when
  `capture` is disabled.
  [(#9155)](https://github.com/PennyLaneAI/pennylane/pull/9155)

* ZX-related transforms are now compatible with `pyzx` v0.10.0.
  [(#9179)](https://github.com/PennyLaneAI/pennylane/pull/9179)

* The :func:`~.transforms.unitary_to_rot` transform now recursively decomposes `QubitUnitary` operations.
  This fixed a bug where two-qubit unitaries would decompose incorrectly to two single-qubit unitaries rather
  than their rotation decomposition.
  [(#9144)](https://github.com/PennyLaneAI/pennylane/pull/9144)

* `qp.value_and_grad` is now available to simultaneously calculate the results and gradients in Catalyst.
  [(#8814)](https://github.com/PennyLaneAI/pennylane/pull/8814)

* The `dynamic_one_shot` and `split_to_single_terms` transforms are now compatible with `qp.qjit`.
  [(#9129)](https://github.com/PennyLaneAI/pennylane/pull/9129)

* When using :func:`~.specs` with Catalyst and with multiple levels,
  the returned :class:`~.resource.CircuitSpecs` will no longer display a
  ``"Before Tape Transforms"`` level if no tape transforms have been applied.
  In particular, for scenarios where no tape transforms are present, the ``"Before MLIR passes"`` level becomes level ``0``.
  In scenarios with at least one tape transform,
  level ``0`` corresponds to ``"Before Tape Transforms"`` and ``"Before MLIR passes"``
  is the level after all tape transforms but before the first MLIR pass.
  [(#9091)](https://github.com/PennyLaneAI/pennylane/pull/9091)
  [(#9166)](https://github.com/PennyLaneAI/pennylane/pull/9166)

* When using :func:`~.specs` with Catalyst and with multiple levels, printing the returned
  :class:`~.resource.CircuitSpecs` object will provide a table detailing relevant information at each requested level,
  for convenient comparison of circuit specifications between compilation passes.
  This display format is enabled by default when using multiple levels in :func:`~.specs` (e.g. in pass-by-pass mode with ``level="all"``):

  ```python
  @qp.qjit
  @qp.transforms.merge_rotations
  @qp.transforms.cancel_inverses
  @qp.qnode(qp.device("lightning.qubit", wires=2))
  def circuit():
      qp.RX(1.23,0)
      qp.RX(1.23,0)
      qp.X(0)
      qp.H(0)
      qp.H(0)
      return qp.probs()
  ```

  ```pycon
  >>> print(qp.specs(circuit, level="all")())
  Device: lightning.qubit
  Device wires: 2
  Shots: Shots(total=None)
  Levels:
  - 0: Before MLIR Passes
  - 1: cancel-inverses
  - 2: merge-rotations
  <BLANKLINE>
  ↓Metric     Level→ |  0 |  1 |  2
  ---------------------------------
  Wire allocations   |  2 |  2 |  2
  Total gates        |  5 |  3 |  2
  Gate counts:       |
  - RX               |  2 |  2 |  1
  - PauliX           |  1 |  1 |  1
  - Hadamard         |  2 |  0 |  0
  Measurements:      |
  - probs(all wires) |  1 |  1 |  1

  ```

  [(#9088)](https://github.com/PennyLaneAI/pennylane/pull/9088)

* `qp.pytrees.PyTreeStructure` is now frozen and hashable. `PyTreeStructure.children` should now
  be a tuple instead of a list.
  [(#9080)](https://github.com/PennyLaneAI/pennylane/pull/9080)

* Allow to pass ``num_work_wires``, ``alt_decomps`` and ``fixed_decomps`` to the device
  preprocessing function :func:`~.devices.preprocess.decompose` , which are then passed through
  to the graph-based decomposition system.
  [(#9094)](https://github.com/PennyLaneAI/pennylane/pull/9094)

* Made the decomposition of :class:`~.BasisState` compatible with ``qjit`` for static wires and
  states, as well as with ``jax.jit`` and static input states. Also changed the parametric
  decomposition for traced states without `qjit` to use powers of `X` rather than `RX`.
  [(#9069)](https://github.com/PennyLaneAI/pennylane/pull/9069)
  [(#9124)](https://github.com/PennyLaneAI/pennylane/pull/9124)
  [(#9339)](https://github.com/PennyLaneAI/pennylane/pull/9339)

* When inspecting a circuit with an integer ``level`` argument in `qp.draw` or `qp.specs`,
  markers in the compilation pipeline are no longer counted towards the level, making inspection more intuitive.
  Integer levels now exclusively refer to transforms, so `level=1` means "after the first transform" regardless
  of how many markers are present.

  Additionally, markers can now be added directly to a :class:`~.CompilePipeline` with the `add_marker` method, and the
  pipeline's string representation now shows both transforms and markers:

  As an example, we now have the following behaviour:

  ```python
  pipeline = qp.CompilePipeline()
  pipeline.add_marker("no-transforms")
  pipeline += qp.transforms.cancel_inverses

  @qp.marker("after-cancel-inverses")
  @pipeline
  @qp.qnode(qp.device("default.qubit"))
  def circuit():
    qp.X(0)
    qp.H(0)
    qp.H(0)
    return qp.probs()
  ```

  The compilation pipeline has a new string representation that can be used to
  inspect the transforms and markers,

  ```pycon
  >>> print(circuit.compile_pipeline)
  CompilePipeline(
     ├─▶ no-transforms
    [1] cancel_inverses()
     └─▶ after-cancel-inverses
  )

  ```

  As usual, marker labels can be used as an argument to `level` in `draw`
  and `specs`, showing the cumulative result of applying transforms up to said marker:

  ```pycon
  >>> print(qp.draw(circuit, level="no-transforms")()) # or level=0
  0: ──X──H──H─┤  Probs
  >>> print(qp.draw(circuit, level="after-cancel-inverses")()) # or level=1
  0: ──X─┤  Probs

  ```
  [(#9007)](https://github.com/PennyLaneAI/pennylane/pull/9007)
  [(#9076)](https://github.com/PennyLaneAI/pennylane/pull/9076)
  [(#9102)](https://github.com/PennyLaneAI/pennylane/pull/9102)

* Catalyst's ``draw_graph`` function is now accessible from PennyLane as :func:`pennylane.draw_graph`.
  [(#9020)](https://github.com/PennyLaneAI/pennylane/pull/9020)

* Raises a more informative error if something that is not a measurement process is returned from a
  QNode when program capture is turned on.
  [(#9072)](https://github.com/PennyLaneAI/pennylane/pull/9072)

* New lightweight representations of the :class:`~.HybridQRAM`, :class:`~.SelectOnlyQRAM`, :class:`~.BasisEmbedding`, and :class:`~.BasisState` templates have
  been added for fast and efficient resource estimation. These operations are available under the `qp.estimator` module as:
  ``qp.estimator.HybridQRAM``, ``qp.estimator.SelectOnlyQRAM``, ``qp.estimator.BasisEmbedding``, and  ``qp.estimator.BasisState``.
  [(#8828)](https://github.com/PennyLaneAI/pennylane/pull/8828)
  [(#8826)](https://github.com/PennyLaneAI/pennylane/pull/8826)

* `qp.transforms.decompose` is now imported top level as `qp.decompose`.
  [(#9011)](https://github.com/PennyLaneAI/pennylane/pull/9011)

* The `CompilePipeline` object now has an improved `__str__`, `__repr__` and `_ipython_display_` allowing improved inspectibility.
  [(#8990)](https://github.com/PennyLaneAI/pennylane/pull/8990)

* `~.specs` now includes PPR and PPM weights in its output, allowing for better categorization of PPMs and PPRs.
  [(#8983)](https://github.com/PennyLaneAI/pennylane/pull/8983)

  ```python
  @qp.qjit(target="mlir")
  @qp.transforms.to_ppr
  @qp.qnode(qp.device("null.qubit", wires=2))
  def circuit():
      qp.H(0)
      qp.CNOT([0, 1])
      m = qp.measure(0)
      qp.T(0)
      return qp.expval(qp.Z(0))
  ```

  ```pycon
  >>> print(qp.specs(circuit, level=1)())
  Device: null.qubit
  Device wires: 2
  Shots: Shots(total=None)
  Level: to-ppr
  <BLANKLINE>
  Wire allocations: 2
  Total gates: 11
  Gate counts:
  - GlobalPhase: 3
  - PPR-pi/4-w1: 5
  - PPR-pi/4-w2: 1
  - PPM-w1: 1
  - PPR-pi/8-w1: 1
  Measurements:
  - expval(PauliZ): 1
  Depth: Not computed

  ```

* :class:`~.BBQRAM`, :class:`~.HybridQRAM`, :class:`SelectOnlyQRAM` and :class:`~.QROM` now accept
  their classical data as a 2-dimensional array data type, which increases compatibility with Catalyst.
  [(#8791)](https://github.com/PennyLaneAI/pennylane/pull/8791)

* :class:`~.CSWAP` is now decomposed more cheaply, using ``change_op_basis`` with
  two ``CNOT`` gates and a single ``Toffoli`` gate.
  [(#8887)](https://github.com/PennyLaneAI/pennylane/pull/8887)

* `qp.vjp` and `qp.jvp` can now be captured into plxpr.
  [(#8736)](https://github.com/PennyLaneAI/pennylane/pull/8736)
  [(#8788)](https://github.com/PennyLaneAI/pennylane/pull/8788)
  [(#9019)](https://github.com/PennyLaneAI/pennylane/pull/9019)

* :func:`~.matrix` can now also be applied to a sequence of operators.
  [(#8861)](https://github.com/PennyLaneAI/pennylane/pull/8861)

* The ``qp.estimator.Resources`` class now has a nice string representation in Jupyter Notebooks.
  [(#8880)](https://github.com/PennyLaneAI/pennylane/pull/8880)

* Adds a `qp.capture.subroutine` for jitting quantum subroutines with program capture.
  [(#8912)](https://github.com/PennyLaneAI/pennylane/pull/8912)

* A function for setting up transform inputs, including setting default values and basic validation,
  can now be provided to `qp.transform` via `setup_inputs`.
  [(#8732)](https://github.com/PennyLaneAI/pennylane/pull/8732)

* Circuits containing `GlobalPhase` are now trainable without removing the `GlobalPhase`.
  [(#8950)](https://github.com/PennyLaneAI/pennylane/pull/8950)

* The decomposition of `QSVT` has been updated to be consistent with or without the graph-based
  decomposition system enabled.
  [(#8994)](https://github.com/PennyLaneAI/pennylane/pull/8994)

* The `to_zx` transform is now compatible with the new graph-based decomposition system.
  [(#8994)](https://github.com/PennyLaneAI/pennylane/pull/8994)

* When the new graph-based decomposition system is enabled, the :func:`~pennylane.transforms.decompose`
  transform no longer raise duplicate warnings about operators that cannot be decomposed.
  [(#9025)](https://github.com/PennyLaneAI/pennylane/pull/9025)

* No unnecessary classical registers will be created now when using `qp.to_openqasm` with `measure_all=False`.
  [(#9033)](https://github.com/PennyLaneAI/pennylane/pull/9033)

* A new `DecompositionWarning` is now raised if the decomposition graph is unable to find a solution
  for an operator, instead of a general `UserWarning`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

* With the new graph-based decomposition system enabled, the `decompose` transform no longer raise
  warnings when the graph is unable to find a decomposition for an operator that does not define a
  decomposition in the following scenarios where operators that does not define a decomposition are
  treated as supported.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

  - When the device is `null.qubit`.
  - With `qp.compile`.
  - Within the `expand_transform` of `hadamard_grad` and `param_shift`.

* Applying `qp.ctrl` on `Snapshot` no longer produces a `Controlled(Snapshot)`. Instead, it now returns the original `Snapshot`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

* When the new graph-based decomposition system is enabled, the `decompose` transform no longer tries to find
  a decomposition for an operator that is not in the statically defined gate set but meets the stopping_condition.
  [(#9036)](https://github.com/PennyLaneAI/pennylane/pull/9036)

* Updated docstring examples in the Pauli-based computation module to reflect the QEC-to-PBC
  dialect rename in Catalyst. References to ``qec.fabricate`` and ``qec.prepare`` are now
  ``pbc.fabricate`` and ``pbc.prepare``.
  [(#9071)](https://github.com/PennyLaneAI/pennylane/pull/9071)

* Ensure `"subroutines"` and `"custom_gates"` are always initialized in the QASM interpreter.
  [(#9201)](https://github.com/PennyLaneAI/pennylane/pull/9201)

* The :func:`~pennylane.ops.sk_decomposition` now accepts `"Adjoint(T)"` and `"Adjoint(S)"` in the `basis_set` as a
  now-preferred alternative to the old `"T*"` and `"S*"` convention for gate adjoints.
  [(#9231)](https://github.com/PennyLaneAI/pennylane/pull/9231)

* The `QROM` decompositions now has a smarter allocation of the work wires achieving better decompositions.
  [(#9131)](https://github.com/PennyLaneAI/pennylane/pull/9131)
  
* The inspectibility of general symbolic decomposition rules is improved. The string representation of a decomposition rule
  is by default its source code. Now for symbolic decomposition rules that wrap a base decomposition rule, the source code
  for the base decomposition rule is also displayed when printing this rule.
  [(#9305)](https://github.com/PennyLaneAI/pennylane/pull/9305)

* The :func:`~pennylane.list_decomps` now returns a new ``DecompCollection`` that allows users to access decomposition rules by index or by name.
  [(#9260)](https://github.com/PennyLaneAI/pennylane/pull/9260)

  ```pycon
  >>> import pennylane as qml
  >>> collection = qml.list_decomps(qml.CRX)
  >>> print(collection)
  Available Decomposition Rules:
  0: _crx_to_rx_cz
  1: _crx_to_rz_ry
  2: _crx_to_h_crz
  3: _crx_to_ppr
  >>> collection[0]
  DecompositionRule(name=_crx_to_rx_cz)
  >>> collection['_crx_to_ppr']
  DecompositionRule(name=_crx_to_ppr)
  >>> print(qml.draw(collection[0])(0.5, wires=[0, 1]))
  0: ───────────╭●────────────╭●─┤
  1: ──RX(0.25)─╰Z──RX(-0.25)─╰Z─┤

  ```

* Applied stricter conditions on some decomposition rules for ``MultiControlledX`` to avoid duplication of equivalent decomposition rules for ``MultiControlledX`` on less than 6 wires.
  [(#9324)](https://github.com/PennyLaneAI/pennylane/pull/9324)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Removed all of the resource estimation functionality from the `labs.resource_estimation`
  module. Users can now directly access a more stable version of this functionality using the
  `estimator` module. All experimental development of resource estimation
  will be added to `.labs.estimator_beta`
  [(#8868)](https://github.com/PennyLaneAI/pennylane/pull/8868)

* The integration test for computing perturbation error of a compressed double-factorized (CDF)
  Hamiltonian in `labs.trotter_error` is upgraded to use a more realistic molecular geometry and
  a more reliable reference error.
  [(#8790)](https://github.com/PennyLaneAI/pennylane/pull/8790)

* Added alternate decompositions for :class:`~.pennylane.labs.estimator_beta.ops.op_math.controlled_ops.CH` and :class:`~.pennylane.labs.estimator_beta.ops.qubit.non_parametric_ops.Hadamard`
  operations in ``labs.estimator_beta`` to get optimal numbers.
  [(#9178)](https://github.com/PennyLaneAI/pennylane/pull/9178)

* Added comparator decompositions for :class:`~.pennylane.labs.estimator_beta.templates.RegisterEquality`
  and :class:`~.pennylane.labs.estimator_beta.templates.OutOfPlaceIntegerComparator` in ``labs.estimator_beta``
  [(#9220)](https://github.com/PennyLaneAI/pennylane/pull/9220)

* Added alternate controlled decompositions for :class:`~.pennylane.labs.estimator_beta.ops.qubit.parametric_ops_multi_qubit.PauliRot` and :class:`~.pennylane.labs.estimator_beta.templates.subroutines.SelectPauliRot`
  operations in ``labs.estimator_beta`` to get optimal numbers.
  [(#9186)](https://github.com/PennyLaneAI/pennylane/pull/9186)

* Added various classes and functions to ``labs.estimator_beta`` to support advanced qubit management
  for resource estimation.
  [(#8996)](https://github.com/PennyLaneAI/pennylane/pull/8996)

  - :class:`~.labs.estimator_beta.Allocate`, allows users to allocate qubits in a resource decomposition.
  - :class:`~.labs.estimator_beta.Deallocate`, allows users to deallocate qubits in a resource decomposition.
  - :class:`~.labs.estimator_beta.MarkClean`, allows users to mark the state of qubits as the zero state in a circuit.
  - :class:`~.labs.estimator_beta.MarkQubits`, allows users to mark the state of qubits in a circuit.
  - :class:`~.labs.estimator_beta.estimate_wires_from_circuit`, estimates the number of additional qubits required
    from a circuit.
  - :class:`~.labs.estimator_beta.estimate_wires_from_resources`, estimates the number of additional qubits required
    from a :class:`~.estimator.Resources` object.

* Created a new ``~.labs.estimator_beta.estimate()`` function which extends the functionality of
  ``qp.estimator.estimate()`` to utilize the advanced qubit management feature for resource estimation.
  [(#9139)](https://github.com/PennyLaneAI/pennylane/pull/9139)

<h4>Other improvements</h4>

* The source code in PennyLane for Pauli-based computation passes was removed, as it is now
  redundant. However, all Pauli-based computation passes can still be accessed from the
  :mod:`pennylane.transforms` module as before (if Catalyst is installed:
  ``pip install pennylane-catalyst``). The reason for the removal is for there to be one single
  source of truth for documentation of a feature if it is desired to be accessible
  from both PennyLane and Catalyst.
  [(#9020)](https://github.com/PennyLaneAI/pennylane/pull/9020)

<h3>Breaking changes 💔</h3>

* ``num_x_wires`` and ``num_work_wires`` were added to the ``resource_keys`` and ``resource_params`` of
  :class:`~.SemiAdder`.
  [(#9293)](https://github.com/PennyLaneAI/pennylane/pull/9293)
  
  With this breaking change, please note the following:
  
   - Decomposition rules for ``SemiAdder`` now require those arguments.
   - When registering a resource function (:func:`qp.register_resources <pennylane.register_resources>`) to a decomposition rule of an operator that contains ``SemiAdder``, the resource representation of ``SemiAdder`` must also receive these new arguments.
   
   These changes are relevant only with :func:`~decomposition.enable_graph`.

* All operator classes are now queued by default, unless they implement a custom ``queue``
  method that changes this behaviour.

  ** Operator math**

  This change also affects operators commonly used for operator math, such as

  - :class:`~.Hermitian`
  - :class:`~.ops.op_math.SProd`
  - :class:`~.ops.op_math.Sum`
  - :class:`~.SparseHamiltonian`
  - :class:`~.Projector`
  - :class:`~.BasisStateProjector`

  All operators are de-queued when used to construct new operators, so the following example
  does *not* show changed behaviour (creating ``B`` removes ``A`` from the queue):

  ```python
  import pennylane as qp
  import numpy as np
  coeff = np.array([0.2, 0.1])

  @qp.qnode(qp.device("lightning.qubit", wires=3))
  def expval(x: float):
      qp.RX(x, 1)
      A = qp.Hamiltonian(coeff, [qp.Y(1), qp.X(0)])
      B = A @ qp.Z(2)
      return qp.expval(B)
  ```

  ```pycon
  >>> print(qp.draw(expval)(0.4))
  0: ───────────┤ ╭<𝓗(0.20,0.10)>
  1: ──RX(0.40)─┤ ├<𝓗(0.20,0.10)>
  2: ───────────┤ ╰<𝓗(0.20,0.10)>

  ```

  However, if we convert an operator ``A`` to numerical data, from which a new
  operator ``B`` is constructed, the chain of operator dependencies is broken and de-queuing will
  not work as expected:

  ```python
  coeff = np.array([0.2, 0.1])

  @qp.qnode(qp.device("lightning.qubit", wires=3))
  def expval(x: float):
      qp.RX(x, 1)
      A = qp.Hamiltonian(coeff, [qp.Y(1), qp.X(0)])
      numerical_data = A.matrix()
      B = qp.Hermitian(numerical_data, wires=[2, 0])
      return qp.expval(B)
  ```

  ```pycon
  >>> print(qp.draw(expval, show_matrices=False)(0.4))
  0: ───────────╭𝓗(0.20,0.10)─┤ ╭<𝓗(M0)>
  1: ──RX(0.40)─╰𝓗(0.20,0.10)─┤ │
  2: ─────────────────────────┤ ╰<𝓗(M0)>

  ```

  As we can see, the ``Hamiltonian`` instance ``A`` remained in the queue.
  In cases where such a conversion to numerical data is unavoidable, perform the conversion
  outside of the quantum circuit.
  [(#8131)](https://github.com/PennyLaneAI/pennylane/pull/8131)
  [(#9029)](https://github.com/PennyLaneAI/pennylane/pull/9029)

* Dropped support for NumPy 1.x following its end-of-life. NumPy 2.0 or higher is now required.
  [(#8914)](https://github.com/PennyLaneAI/pennylane/pull/8914)
  [(#8954)](https://github.com/PennyLaneAI/pennylane/pull/8954)
  [(#9017)](https://github.com/PennyLaneAI/pennylane/pull/9017)
  [(#9167)](https://github.com/PennyLaneAI/pennylane/pull/9167)

* ``compute_qfunc_decomposition`` and ``has_qfunc_decomposition`` have been removed from  :class:`~.Operator`
  and all subclasses that implemented them. The graph decomposition system should be used when capture is enabled.
  [(#8922)](https://github.com/PennyLaneAI/pennylane/pull/8922)

* The :func:`pennylane.devices.preprocess.mid_circuit_measurements` transform is removed. Instead,
  the device should determine which mcm method to use, and explicitly include :func:`~pennylane.transforms.dynamic_one_shot`
  or :func:`~pennylane.transforms.defer_measurements` in its preprocess transforms if necessary. See
  :func:`DefaultQubit.setup_execution_config <pennylane.devices.DefaultQubit.setup_execution_config>` and
  :func:`DefaultQubit.preprocess_transforms <pennylane.devices.DefaultQubit.preprocess_transforms>` for an example.
  [(#8926)](https://github.com/PennyLaneAI/pennylane/pull/8926)

* The ``custom_decomps`` keyword argument to ``qp.device`` has been removed in 0.45. Instead,
  with ``qp.decomposition.enable_graph()``, new decomposition rules can be defined as
  quantum functions with registered resources. See :mod:`pennylane.decomposition` for more details.
  [(#8928)](https://github.com/PennyLaneAI/pennylane/pull/8928)

  As an example, consider the case of running the following circuit on a device that does not natively support ``CNOT`` gates:

  ```python
  def circuit():
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.X(1))
  ```

  Instead of defining the ``CNOT`` decomposition as:

  ```py
  def custom_cnot(wires):
    return [
      qp.Hadamard(wires=wires[1]),
      qp.CZ(wires=[wires[0], wires[1]]),
      qp.Hadamard(wires=wires[1])
    ]

  dev = qp.device('default.qubit', wires=2, custom_decomps={"CNOT" : custom_cnot})
  qnode = qp.QNode(circuit, dev)
  print(qp.draw(qnode, level="device")())
  ```

  The same result would now be obtained using:

  ```python
  @qp.decomposition.register_resources({
    qp.H: 2,
    qp.CZ: 1
  })
  def _custom_cnot_decomposition(wires, **_):
    qp.Hadamard(wires=wires[1])
    qp.CZ(wires=[wires[0], wires[1]])
    qp.Hadamard(wires=wires[1])

  qp.decomposition.add_decomps(qp.CNOT, _custom_cnot_decomposition)

  qp.decomposition.enable_graph()

  @qp.transforms.decompose(gate_set={qp.CZ, qp.H})
  def circuit():
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.X(1))

  dev = qp.device('default.qubit', wires=2)
  qnode = qp.QNode(circuit, dev)
  ```

  ```pycon
  >>> print(qp.draw(qnode, level="device")())
  0: ────╭●────┤
  1: ──H─╰Z──H─┤  <X>

  ```

* The `pennylane.operation.Operator.is_hermitian` property has been removed and replaced
  with `pennylane.operation.Operator.is_verified_hermitian` as it better reflects the functionality of this property.
  Alternatively, consider using the `pennylane.is_hermitian` function instead as it provides a more reliable check for hermiticity.
  Please be aware that it comes with a higher computational cost.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Passing a function to the `gate_set` argument in the `pennylane.transforms.decompose` transform
  is removed. The `gate_set` argument expects a static iterable of operator type and/or operator names,
  and the function should be passed to the `stopping_condition` argument instead.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* `argnum` has been renamed `argnums` in `qp.grad`, `qp.jacobian`, `qp.jvp`, and `qp.vjp`
  to better match Catalyst and JAX.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Access to the following functions and classes from the `~pennylane.resources` module has
  been removed. Instead, these functions must be imported from the `~pennylane.estimator` module.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

    - `qp.estimator.estimate_shots` in favor of `qp.resources.estimate_shots`
    - `qp.estimator.estimate_error` in favor of `qp.resources.estimate_error`
    - `qp.estimator.FirstQuantization` in favor of `qp.resources.FirstQuantization`
    - `qp.estimator.DoubleFactorization` in favor of `qp.resources.DoubleFactorization`

<h3>Deprecations 👋</h3>

* The :func:`~pennylane.workflow.get_transform_program` function has been deprecated and will be removed in v0.46.
  Instead, please use the improved :func:`~pennylane.workflow.get_compile_pipeline` to retrieve the execution pipeline
  of a QNode.
  [(#9077)](https://github.com/PennyLaneAI/pennylane/pull/9077)

* The ``id`` keyword argument to :class:`~.qcut.MeasureNode` and :class:`~.qcut.PrepareNode` has been renamed to `node_uid` and will be removed in v0.46.
  [(#8951)](https://github.com/PennyLaneAI/pennylane/pull/8951)

* The ``id`` keyword argument to :class:`~.ops.MidMeasure` has been renamed to `meas_uid` and will be removed in v0.46.
  [(#8951)](https://github.com/PennyLaneAI/pennylane/pull/8951)

* The ``id`` keyword argument to :class:`~.measurements.MeasurementProcess` has been deprecated and will be removed in v0.46.
  [(#8951)](https://github.com/PennyLaneAI/pennylane/pull/8951)

* The ``id`` keyword argument to :class:`~.Operator` has been deprecated and will be removed in v0.46.
  [(#8951)](https://github.com/PennyLaneAI/pennylane/pull/8951)
  [(#9051)](https://github.com/PennyLaneAI/pennylane/pull/9051)

  The ``id`` argument previously served two purposes: (1) adding custom labels
  to operator instances which were rendered in circuit drawings and (2)
  tagging encoding gates for Fourier spectrum analysis.

  These are now handled by dedicated functions:

  .. warning::
    Neither of these functions are supported in a :func:`~.qjit`-compiled circuit,
    as the original behaviour was never supported.

  - Use :func:`~.drawer.label` to attach a custom label to an operator instance
  for circuit drawing:

      ```python
      # Legacy method (deprecated):
      qp.RX(0.5, wires=0, id="my-rx")

      # New method:
      qp.drawer.label(qp.RX(0.5, wires=0), "my-rx")
      ```

  - Use :func:`~.fourier.mark` to mark an operator as an input-encoding gate
    for :func:`~.fourier.circuit_spectrum`, and :func:`~.fourier.qnode_spectrum`:

      ```py
      # Legacy method (deprecated):
      qp.RX(0.5, wires=0, id="x0")

      # New method:
      qp.fourier.mark(qp.RX(0.5, wires=0), "x0")
      ```

* Setting `_queue_category=None` in an operator class in order to deactivate its instances being
  queued has been deprecated. Implement a custom `queue` method for the respective class instead.
  Operator classes that used to have `_queue_category=None` have been updated
  to `_queue_category="_ops"`, so that they are queued now.
  [(#8131)](https://github.com/PennyLaneAI/pennylane/pull/8131)

* The ``BoundTransform.transform`` property has been deprecated. Use ``BoundTransform.tape_transform`` instead.
  [(#8985)](https://github.com/PennyLaneAI/pennylane/pull/8985)

* :func:`~pennylane.tape.qscript.expand` and the related functions :func:`~pennylane.tape.expand_tape`, :func:`~pennylane.tape.expand_tape_state_prep`, and :func:`~pennylane.tape.create_expand_trainable_multipar`
  have been deprecated and will be removed in v0.46. Instead, please use the :func:`qp.transforms.decompose <.transforms.decompose>`
  function for decomposing circuits.
  [(#8943)](https://github.com/PennyLaneAI/pennylane/pull/8943)

* Providing a value of ``None`` to ``aux_wire`` of ``qp.gradients.hadamard_grad`` in reversed or standard mode has been
  deprecated and will no longer be supported in 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#8905)](https://github.com/PennyLaneAI/pennylane/pull/8905)

* The ``transform_program`` property of ``QNode`` has been renamed to ``compile_pipeline``.
  The deprecated access through ``transform_program`` will be removed in PennyLane v0.46.
  [(#8906)](https://github.com/PennyLaneAI/pennylane/pull/8906)

* Providing a value of ``None`` to ``aux_wire`` of ``qp.gradients.hadamard_grad`` with ``mode="reversed"`` or ``mode="standard"`` has been
  deprecated and will no longer be supported in 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#8905)](https://github.com/PennyLaneAI/pennylane/pull/8905)

* The ``qp.transforms.create_expand_fn`` has been deprecated and will be removed in v0.46.
  Instead, please use the :func:`qp.transforms.decompose <.transforms.decompose>` function for decomposing circuits.
  [(#8941)](https://github.com/PennyLaneAI/pennylane/pull/8941)
  [(#8977)](https://github.com/PennyLaneAI/pennylane/pull/8977)
  [(#9006)](https://github.com/PennyLaneAI/pennylane/pull/9006)

* The ``transform_program`` property of ``QNode`` has been renamed to ``compile_pipeline``.
  The deprecated access through ``transform_program`` will be removed in PennyLane v0.46.
  [(#8906)](https://github.com/PennyLaneAI/pennylane/pull/8906)
  [(#8945)](https://github.com/PennyLaneAI/pennylane/pull/8945)

<h3>Internal changes ⚙️</h3>

* Largely unused PLxPR was recently removed in lightning. Removed tests from PennyLane that are no longer relevant 
  as a result.
  [(#9345)](https://github.com/PennyLaneAI/pennylane/pull/9345)

* During program capture, `qml.cond` converts non-boolean predicates to boolean immediately
  during capture time.
  [(#9336)](https://github.com/PennyLaneAI/pennylane/pull/9336)

* During program, `qml.for_loop` with negative step sizes is now handled immediately during capture time.
  [(#9299)](https://github.com/PennyLaneAI/pennylane/pull/9299)

* With program capture, arrays dynamic shapes with `qp.for_loop` and `qp.while_loop` can now be combined
  after the loop.
  [(#9245)](https://github.com/PennyLaneAI/pennylane/pull/9245)

* Patched `jax._src.pjit._infer_params_internal` for dynamic shapes to correctly handle the concatenation of closure variables and arguments before return.
  [(#9250)](https://github.com/PennyLaneAI/pennylane/pull/9250)

* Removed docker files and workflow.
  [(#9273)](https://github.com/PennyLaneAI/pennylane/pull/9273)

* Remove requirements file from docs folder.
  [(#9242)](https://github.com/PennyLaneAI/pennylane/pull/9242)

* Added the `doctest` group in `pyproject.toml` to easily maintain dependencies of the documentation tests workflow.
  [(#9237)](https://github.com/PennyLaneAI/pennylane/pull/9237)

* `BasisEmbedding` now captures as `BasisState` so it now works with Catalyst and
  program capture.
  [(#9183)](https://github.com/PennyLaneAI/pennylane/pull/9183)

* A transform's `setup_inputs` is no longer called twice when applied on a `QNode`.
  [(#9189)](https://github.com/PennyLaneAI/pennylane/pull/9189)

* Fixed a warning of casting complex values to reals within `qp.math.givens_decomposition`.
  [(#9155)](https://github.com/PennyLaneAI/pennylane/pull/9155)

* The output of the `qp.while_loop` condition is now automatically converted
  to a bool.
  [(#9184)](https://github.com/PennyLaneAI/pennylane/pull/9184)

* When using :func:`~.specs` with Catalyst and with multiple levels,
  with the ``split-non-commuting`` MLIR pass applied,
  the returned :class:`.resource.CircuitSpecs` object will include
  a list of :class:`~.resource.SpecsResources` objects for the associated ``level``.
  [(#9120)](https://github.com/PennyLaneAI/pennylane/pull/9120)

* Upper bound `pyzx<0.10` dependency to ensure compatibility.
  [(#9175)](https://github.com/PennyLaneAI/pennylane/pull/9175)

* Remove usage of `PassPipelineWrapper` due to `removal <https://github.com/PennyLaneAI/catalyst/pull/2525>`) in Catalyst.
  [(#9123)](https://github.com/PennyLaneAI/pennylane/pull/9123)

* Updated the `diastatic-malt` dependency to version v2.15.3.
  [(#9154)](https://github.com/PennyLaneAI/pennylane/pull/9154)

* Workflow created to sync the `main` branch to `master`. Workflow deleted after master branch was deleted.
  [(#9127)](https://github.com/PennyLaneAI/pennylane/pull/9127)
  [(#9316)](https://github.com/PennyLaneAI/pennylane/pull/9316)

* Removed `pytest-benchmark` from the `pyproject.toml` `dev` dependency group. Benchmarking is no longer internally performed in our test suite.
  [(#7900)](https://github.com/PennyLaneAI/pennylane/pull/7900)

* References to the `master` branch are changed to the new default branch `main`.
  [(#9128)](https://github.com/PennyLaneAI/pennylane/pull/9128)

* Update nightly RC builds to not be a schedule triggered in Pennylane anymore. Instead, it will be triggered in the order Lightning —> Catalyst —> Pennylane.
  [(#9092)](https://github.com/PennyLaneAI/pennylane/pull/9092)

* Remove duplicate transforms found in both `ftqc/catalyst_pass_aliases.py` and `transforms/decompositions/pauli_based_computation.py`.
  [(#9090)](https://github.com/PennyLaneAI/pennylane/pull/9090)

* Update pennylane to use a uv lockfile for package dependency tracking. Added `UV_SYSTEM_PYTHON` to the repository's nightly sync workflows. Removed stable dependency folder and files.
  [(#8755)](https://github.com/PennyLaneAI/pennylane/pull/8755)
  [(#9110)](https://github.com/PennyLaneAI/pennylane/pull/9110)
  [(#9218)](https://github.com/PennyLaneAI/pennylane/pull/9218)

* A new AI policy document is now applied across the PennyLaneAI organization for all AI contributions.
  [(#9079)](https://github.com/PennyLaneAI/pennylane/pull/9079)

* Add `sybil` to `dev` dependency group in `pyproject.toml`.
  [(#9060)](https://github.com/PennyLaneAI/pennylane/pull/9060)

* `qp.counts` of mid circuit measurements can now be captured into jaxpr.
  [(#9022)](https://github.com/PennyLaneAI/pennylane/pull/9022)

* Pass-by-pass specs now use ``BoundTransform.tape_transform`` rather than the deprecated ``BoundTransform.transform``.
  Additionally, several internal comments have been updated to bring specs in line with the new ``CompilePipeline`` class.
  [(#9012)](https://github.com/PennyLaneAI/pennylane/pull/9012)

* Specs can now return measurement information for QJIT'd workloads when passed ``level="device"``.
  [(#8988)](https://github.com/PennyLaneAI/pennylane/pull/8988)

* Add documentation tests for various modules.
  [(#9004)](https://github.com/PennyLaneAI/pennylane/pull/9004)
  [(#9206)](https://github.com/PennyLaneAI/pennylane/pull/9206)
  [(#8653)](https://github.com/PennyLaneAI/pennylane/pull/8653)
  [(#9062)](https://github.com/PennyLaneAI/pennylane/pull/9062)
  [(#9236)](https://github.com/PennyLaneAI/pennylane/pull/9236)

* Seeded a test `tests/measurements/test_classical_shadow.py::TestClassicalShadow::test_return_distribution` to fix stochastic failures by adding a `seed` parameter to the circuit helper functions and the test method.
  [(#8981)](https://github.com/PennyLaneAI/pennylane/pull/8981)

* Standardized the tolerances of several stochastic tests to use a 3-sigma rule based on theoretical variance and number of shots, reducing spurious failures. This includes `TestHamiltonianSamples::test_multi_wires`, `TestSampling::test_complex_hamiltonian`, and `TestBroadcastingPRNG::test_nonsample_measure`.
  Bumped `rng_salt` to `v0.45.0`.
  [(#8959)](https://github.com/PennyLaneAI/pennylane/pull/8959)
  [(#8958)](https://github.com/PennyLaneAI/pennylane/pull/8958)
  [(#8938)](https://github.com/PennyLaneAI/pennylane/pull/8938)
  [(#8908)](https://github.com/PennyLaneAI/pennylane/pull/8908)
  [(#8963)](https://github.com/PennyLaneAI/pennylane/pull/8963)

* Updated test helper `get_device` to correctly seed lightning devices.
  [(#8942)](https://github.com/PennyLaneAI/pennylane/pull/8942)

* Updated internal dependencies `autoray` (to 0.8.4), `tach` (to 0.32.2).
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)
  [(#8962)](https://github.com/PennyLaneAI/pennylane/pull/8962)
  [(#9030)](https://github.com/PennyLaneAI/pennylane/pull/9030)

* Relaxed the `torch` dependency from `==2.9.0` to `~=2.9.0` to allow for compatible patch updates.
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)

* Internal calls to the `decompose` transform have been updated to provide a `target_gates` argument so that
  they are compatible with the new graph-based decomposition system.
  [(#8939)](https://github.com/PennyLaneAI/pennylane/pull/8939)

* Added a `qp.decomposition.toggle_graph_ctx` context manager to temporarily enable or disable graph-based
  decompositions in a thread-safe way. The fixtures `"enable_graph_decomposition"`, `"disable_graph_decomposition"`,
  and `"enable_and_disable_graph_decomp"` have been updated to use this method so that they are thread-safe.
  [(#8966)](https://github.com/PennyLaneAI/pennylane/pull/8966)

* Added specialized gate kernels for RX, RY, RZ, and Hadamard in the `default.qubit` device.
  These bypass generic einsum/tensordot dispatches and use direct contractions for NumPy
  states, with correct fallbacks for autodiff interfaces (Autograd, Torch, JAX).
  [(#9075)](https://github.com/PennyLaneAI/pennylane/pull/9075)

* Added a `qp.decomposition.reconstruct` module which implements a method to reconstruct the original
  operator instance from `(*op.data, op.wires, **op.resource_params)`, which enables qjit-compatible
  symbolic decomposition rules that do not need to take an instance of the base operator as input.
  [(#9188)](https://github.com/PennyLaneAI/pennylane/pull/9188)

<h3>Documentation 📝</h3>

* The `qml` alias as in `import pennylane as qml` has been updated to `qp` in our source code and documentation.
  [(#9310)](https://github.com/PennyLaneAI/pennylane/pull/9310)
  [(#9314)](https://github.com/PennyLaneAI/pennylane/pull/9314)
  [(#9319)](https://github.com/PennyLaneAI/pennylane/pull/9319)
  [(#9313)](https://github.com/PennyLaneAI/pennylane/pull/9313)
  [(#9326)](https://github.com/PennyLaneAI/pennylane/pull/9326)
  [(#9329)](https://github.com/PennyLaneAI/pennylane/pull/9329)
  [(#9280)](https://github.com/PennyLaneAI/pennylane/pull/9280)
  [(#9327)](https://github.com/PennyLaneAI/pennylane/pull/9327)

* Documentation has been added to :func:`~.transforms.cancel_inverses` and
  :func:`~.transforms.merge_rotations` that details their usage within a ``qjit`` workflow.
  [(#9134)](https://github.com/PennyLaneAI/pennylane/pull/9134)

* A typo causing a rendering issue in the docstring for :class:`~.QNode` has been fixed.
  [(#8652)](https://github.com/PennyLaneAI/pennylane/pull/8652)

* A typo in the docstring for ``ControlledOp`` was fixed and the ``Controlled`` docstring recommends using ``ctrl`` instead.
  [(#7154)](https://github.com/PennyLaneAI/pennylane/pull/7154)

* Wide-spread changes were made to our documentation to recommend using program capture with ``qjit``
  only, and enabling it via ``qjit(capture=True)`` instead of the global toggle (``qp.capture.enable()``).
  [(#9059)](https://github.com/PennyLaneAI/pennylane/pull/9059)

* Added a note to the documentation of :func:`~.estimator.estimate.estimate` to clarify
  that an error will be raised if a ``ResourceOperator`` is encountered that does not have
  a resource decomposition defined and is not in the provided ``gate_set``.
  [(#9230)](https://github.com/PennyLaneAI/pennylane/pull/9230)

* Updated documentation for :func:`~.transforms.gridsynth` as we now issue a warning when users provide epsilon smaller than ``1e-6``, and simulation of PPRs is now possible.
  [(#9221)](https://github.com/PennyLaneAI/pennylane/pull/9221)

* Refined the documentation of :func:~.shadow_expval measurement for clarity and added instructions
  for achieving reproducible results with the seed keyword argument.
  [(#9216)](https://github.com/PennyLaneAI/pennylane/pull/9216)

* The definition of the ``pipeline`` argument for :func:`~.transforms.compile`
  was clarified in its documentation.
  [(#9159)](https://github.com/PennyLaneAI/pennylane/pull/9159)

* The type of a parameter is fixed in the docstring of :class:`~.templates.layers.BasicEntanglerLayers`.
  [(#9046)](https://github.com/PennyLaneAI/pennylane/pull/9046)

* Infrastructure has been put in place for features that should be accessible from both PennyLane and
  Catalyst to have a single source of truth for documentation, which will provide a better overall
  experience when consulting our documentation.
  [(#9020)](https://github.com/PennyLaneAI/pennylane/pull/9020)

  The process for Catalyst frontend features to be automatically accessible from PennyLane while
  ensuring that such features' documentation is properly sourced from Catalyst and hosted on
  PennyLane's documentation is outlined in the
  :doc:`documentation development guide <../development/guide/documentation>` under the section
  titled "Making Catalyst functionality callable from PennyLane". Related work in Catalyst can be
  found in [(#2409)](https://github.com/PennyLaneAI/catalyst/pull/2409).

* Though the documentation for this function is now solely in the Catalyst repository, a correction was
  made in the output of the code example for :func:`~.transforms.decompose_arbitrary_ppr` while the
  documentation still resided in the PennyLane repository.
  [(#9116)](https://github.com/PennyLaneAI/pennylane/pull/9116)

* Fixed the docstring of ``FermiSentence`` that incorrectly claims that it is immutable.
  [(#9278)](https://github.com/PennyLaneAI/pennylane/pull/9278)

<h3>Bug fixes 🐛</h3>

* Fixes a bug with program capture when a transform is applied to a qnode with a dynamic number of shots
  and return `qml.sample`.
  [(#9342)](https://github.com/PennyLaneAI/pennylane/pull/9342)

* Fixed wire overlap validation in :class:`~.QROM` and :class:`~.Select` to support JAX-traced wires,
  enabling `qml.QROM` to be used with `qjit` when wires are passed as dynamic arguments.
  [(#9282)](https://github.com/PennyLaneAI/pennylane/pull/9282)

* Global phases are now supported in `from_qasm3` so that QASM including the `gphase` instruction 
  can be interpreted.
  [(#9247)](https://github.com/PennyLaneAI/pennylane/pull/9247)

* Fixes an issue with Catalyst and `qp.for_loop` and `qp.while_loop`, where it was defaulting
  to `allow_array_resizing=True` instead of `allow_array_resizing=False`.
  [(#9251)](https://github.com/PennyLaneAI/pennylane/pull/9251)

* Fixed a bug where the ``work_wire_type`` argument of :func:`~.ctrl` was silently dropped when the
  call was delegated to the active compiler (:func:`~.qjit`). The parameter is now forwarded to the
  compiler's ``ctrl`` implementation.
  [(#9328)](https://github.com/PennyLaneAI/pennylane/pull/9328)

* Workflows with program capture that involve dynamic device wires will now raise a `NotImplementedError`
  rather than providing incorrect results.
  [(#9248)](https://github.com/PennyLaneAI/pennylane/pull/9248)

* Fixed a bug in :mod:`~.estimator` where the ``ResourcesUndefinedError``
  was being returned as a class type rather than an instance,
  preventing the intended diagnostic message from being displayed.
  [(#9229)](https://github.com/PennyLaneAI/pennylane/pull/9229)

* Fixed a bug where the data file `transforms/sign_expand/sign_expand_data.json` was not included in
  the source distribution, causing errors when using `qp.transforms.sign_expand` in a production
  environment.
  [(#9197)](https://github.com/PennyLaneAI/pennylane/pull/9197)

* Fixed a bug where `qp.math.givens_decomposition` modified the input in place when using `qjit`.
  [(#9155)](https://github.com/PennyLaneAI/pennylane/pull/9155)

* Fixed a bug where the hashable parameters of a `CompressedResourceOp` in the graph-based
  decomposition system were sensitive to the insertion order of keyword arguments/hyperparameters.
  [(#9137)](https://github.com/PennyLaneAI/pennylane/pull/9137)

* Jacobian-level caching is now unconditionally enabled for `autograd` interface,
  preventing redundant derivative tape executions during the backward pass.
  [(#9081)](https://github.com/PennyLaneAI/pennylane/pull/9081)

* Fixed a bug where `qp.transforms.transpile` would fail when `qp.GlobalPhase` gates
  were present in a circuit.
  [(#9041)](https://github.com/PennyLaneAI/pennylane/pull/9041)

* Fixed a bug where :class:`~.ops.LinearCombination` did not correctly de-queue the constituents
  of an operator product via the dunder method ``__matmul__``.
  [(#9029)](https://github.com/PennyLaneAI/pennylane/pull/9029)

* Fixed :attr:`~.ops.Controlled.map_wires` and :func:`~.equal` with ``Controlled`` instances
  to handle the ``work_wire_type`` correctly within ``map_wires``. Also fixed
  ``Controlled.map_wires`` to preserve ``work_wires``.
  [(#9010)](https://github.com/PennyLaneAI/pennylane/pull/9010)

* Bumps the tolerance used in determining whether the norm of the probabilities is sufficiently close to
  1 in Default Qubit.
  [(#9014)](https://github.com/PennyLaneAI/pennylane/pull/9014)

* Removes automatic unpacking of inner product resources in the resource representation of
  :class:`~.ops.op_math.Prod` for the graph-based decomposition system. This resolves a bug that
  prevents decompositions in this system from using nested operator products while reporting their
  resources accurately.
  [(#8773)](https://github.com/PennyLaneAI/pennylane/pull/8773)

* Decompose integers into powers of two while adhering to standard 64-bit C integer bounds and avoid overflow in the decomposition system.
  [(#8993)](https://github.com/PennyLaneAI/pennylane/pull/8993)

* `CompilePipeline` no longer automatically pushes final transforms to the end of the pipeline as it's being built.
  [(#8995)](https://github.com/PennyLaneAI/pennylane/pull/8995)

* Improves the error messages when the inputs and outputs to a `qp.for_loop` function do not match.
  [(#8984)](https://github.com/PennyLaneAI/pennylane/pull/8984)

* Fixes a bug that `qp.QubitDensityMatrix` was applied in `default.mixed` device using `qp.math.partial_trace` incorrectly.
  This would cause wrong results as described in [this issue](https://github.com/PennyLaneAI/pennylane/pull/8932).
  [(#8933)](https://github.com/PennyLaneAI/pennylane/pull/8933)

* Fixes an issue when binding a transform when the first positional arg
  is a `Sequence`, but not a `Sequence` of tapes.
  [(#8920)](https://github.com/PennyLaneAI/pennylane/pull/8920)

* Fixes a bug with `qp.estimator.templates.QSVT` which allows users to instantiate the class without
  providing wires. This is now consistent with the standard in the estimator module.
  [(#8949)](https://github.com/PennyLaneAI/pennylane/pull/8949)

* Fixes a bug where decomposition raises an error for `Pow` operators when the exponent is batched.
  [(#8969)](https://github.com/PennyLaneAI/pennylane/pull/8969)

* Fixes a bug where the `DecomposeInterpreter` cannot be applied on a `QNode` with the new graph-based decomposition system enabled.
  [(#8965)](https://github.com/PennyLaneAI/pennylane/pull/8965)

* Fixes a bug where `qp.equal` raises an error for `SProd` with abstract scalar parameters and `Exp` with abstract coefficients.
  [(#8965)](https://github.com/PennyLaneAI/pennylane/pull/8965)

* Fixes various issues found with decomposition rules for `QubitUnitary`, `BasisRotation`, `StronglyEntanglingLayers`.
  [(#8965)](https://github.com/PennyLaneAI/pennylane/pull/8965)

* The `decompose` transform no longer warns about being unable to decompose `Barrier` and `Snapshot`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

* When the new graph-based decomposition system is enabled, `Exp` no longer decomposes to nothing when the exponent
  is the identity. Instead, a `PauliRot` is always produced, which in this case decomposes to a `GlobalPhase`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

* Fixes a bug where the graph-based decomposition system is unbale to find a decomposition for a `ControlledQubitUnitary` with more than two target wires.
  [(#9036)](https://github.com/PennyLaneAI/pennylane/pull/9036)

* Fixes a discontinuity in the gradient of the single-qubit unitary decompositions.
  [(#9036)](https://github.com/PennyLaneAI/pennylane/pull/9036)

* Fixes a `MemoryError` in `default.clifford` when preparing a :class:`~.BasisState` with a large number of wires.
  [(#9018)](https://github.com/PennyLaneAI/pennylane/pull/9018)

* Fixes a bug where a controlled `ChangeOpBasis` is sometimes not decomposed optimally when graph is enabled.
  [(#9161)](https://github.com/PennyLaneAI/pennylane/pull/9161)

* Fixes a bug where the decomposition graph is unable to find trivial decompositions of `qp.X(0) ** 1` and `qp.X(0) ** 0`.
  [(#9152)](https://github.com/PennyLaneAI/pennylane/pull/9152)

* Fixed various small bugs within :mod:`pennylane.estimator`.
  [(#9194)](https://github.com/PennyLaneAI/pennylane/pull/9194)

    - Fixed the resource decomposition of `~.estimator.QubitUnitary` to match the results from literature
    - Fixed the resource decomposition of `~.estimator.OutMultiplier` to match the results from literature
    - Added support for mapping `~.Barrier` and `~.SnapShot` to `~.labs.estimator_beta.Identity`
    - Fixed incorrect wire mapping when converting `~.QuantumPhaseEstimation` to `~.estimator.QPE`

* Fixed a bug in the `C(SemiAdder)` decomposition where incorrect results were
  produced for a specific wire configuration.
  [(#9270)](https://github.com/PennyLaneAI/pennylane/pull/9270)

* Fixes a bug where the `DecompositionGraph` underestimates the minimum number of work wires required to solve for a particular operator
  when it has decomposition rules with a lower work wire budget but is unrecheable from the provided gate set.
  [(#9298)](https://github.com/PennyLaneAI/pennylane/pull/9298)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Ali Asadi,
Astral Cai,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Olivia Di Matteo,
Marcus Edwards,
Sengthai Heng,
Jacob Kitchen,
Korbinian Kottmann,
Christina Lee,
Joseph Lee,
Anton Naim Ibrahim,
Oumarou Oumarou,
Mudit Pandey,
Andrija Paurevic,
Omkar Sarkar,
Jay Soni,
Nate Stemen,
David Wierichs,
Fuyuan Xia,
Jake Zaia.
