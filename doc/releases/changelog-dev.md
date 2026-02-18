# Release 0.45.0 (development release)

<h3>New features since last release</h3>

* Added the function :func:`~.drawer.label` to attach custom labels to operator instances
  for circuit drawing.
  [(#9078)](https://github.com/PennyLaneAI/pennylane/pull/9078)  

* Added the function :func:`~.fourier.mark` to mark an operator as an input-encoding gate
  for :func:`~.fourier.circuit_spectrum`, and :func:`~.fourier.qnode_spectrum`.
  [(#9078)](https://github.com/PennyLaneAI/pennylane/pull/9078)  

* Prepared new state preparation template :class:`~.SumOfSlatersStatePrep`.
  It prepares sparse states using a smaller dense state preparation, :class:`~.QROM`\ s and 
  reversible bit encodings. For now, only classical preprocessing required to implement the
  template is added.
  [(#8964)](https://github.com/PennyLaneAI/pennylane/pull/8964)

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
  logarithm of its input and casts the result to an ``int``. It is equivalent to 
  ``int(np.ceil(np.log2(n)))``.
  [(#8972)](https://github.com/PennyLaneAI/pennylane/pull/8972)

* Added a ``qml.gate_sets`` that contains pre-defined gate sets such as ``qml.gate_sets.CLIFFORD_T_PLUS_RZ``
  that can be plugged into the ``gate_set`` argument of the :func:`~pennylane.transforms.decompose` transform.
  [(#8915)](https://github.com/PennyLaneAI/pennylane/pull/8915)
  [(#9045)](https://github.com/PennyLaneAI/pennylane/pull/9045)

* Adds a new `qml.templates.Subroutine` class for adding a layer of abstraction for
  quantum functions. These objects can now return classical values or mid circuit measurements,
  and are compatible with Program Capture Catalyst. Any `Operator` with a single definition
  in terms of its implementation, a more complicated call signature, and that exists
  at a higher, algorithmic layer of abstraction should switch to using this class instead
  of `Operator`/ `Operation`.
  [(#8929)](https://github.com/PennyLaneAI/pennylane/pull/8929)
  [(#9070)](https://github.com/PennyLaneAI/pennylane/pull/9070)

  ```python
  from pennylane.templates import Subroutine

  @Subroutine
  def MyTemplate(x, y, wires):
      qml.RX(x, wires[0])
      qml.RY(y, wires[0])

  @qml.qnode(qml.device('default.qubit'))
  def c():
      MyTemplate(0.1, 0.2, 0)
      return qml.state()
  ```

  ```pycon
  >>> print(qml.draw(c)())
  0: ──MyTemplate(0.10,0.20)─┤  State
  ```

* Added a `qml.decomposition.local_decomps` context
  manager that allows one to add decomposition rules to an operator, only taking effect within the context.
  [(#8955)](https://github.com/PennyLaneAI/pennylane/pull/8955)
  [(#8998)](https://github.com/PennyLaneAI/pennylane/pull/8998)

* Added a `strict` keyword to the :func:`~pennylane.transforms.decompose` transform that, when set to ``False``,
  allows the decomposition graph to treat operators without a decomposition as part of the gate set.
  [(#9025)](https://github.com/PennyLaneAI/pennylane/pull/9025)

* New decomposition rules are added to `Evolution` and `RZ`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)
  [(#9049)](https://github.com/PennyLaneAI/pennylane/pull/9049)

* The custom `adjoint` method of qutrit operators are implemented as decomposition rules compatible with the
  new graph-based decomposition system.
  [(#9056)](https://github.com/PennyLaneAI/pennylane/pull/9056)

<h3>Improvements 🛠</h3>

* When inspecting a circuit with an integer ``level`` argument in `qml.draw` or `qml.specs`,
  markers in the compilation pipeline are no longer counted towards the level, making inspection more intuitive. 
  Integer levels now exclusively refer to transforms, so `level=1` means "after the first transform" regardless 
  of how many markers are present.

  Additionally, markers can now be added directly to a :class:`~.CompilePipeline` with the `add_marker` method, and the
  pipeline's string representation now shows both transforms and markers:

  As an example, we now have the following behaviour:

  ```python
  pipeline = qml.CompilePipeline()
  pipeline.add_marker("no-transforms")
  pipeline += qml.transforms.cancel_inverses

  @qml.marker("after-cancel-inverses")
  @pipeline
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
    qml.X(0)
    qml.H(0)
    qml.H(0)
    return qml.probs()
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
  >>> print(qml.draw(c, level="no-transforms")()) # or level=0
  0: ──X──H──H─┤  Probs
  >>> print(qml.draw(c, level="after-cancel-inverses")()) # or level=1
  0: ──X─┤  Probs
  ```
  [(#9007)](https://github.com/PennyLaneAI/pennylane/pull/9007)
  [(#9076)](https://github.com/PennyLaneAI/pennylane/pull/9076)
  
* Raises a more informative error if something that is not a measurement process is returned from a 
  QNode when program capture is turned on.
  [(#9072)](https://github.com/PennyLaneAI/pennylane/pull/9072)

* New lightweight representations of the :class:`~.HybridQRAM`, :class:`~.SelectOnlyQRAM`, :class:`~.BasisEmbedding`, and :class:`~.BasisState` templates have 
  been added for fast and efficient resource estimation. These operations are available under the `qp.estimator` module as:
  ``qp.estimator.HybridQRAM``, ``qp.estimator.SelectOnlyQRAM``, ``qp.estimator.BasisEmbedding``, and  ``qp.estimator.BasisState``.
  [(#8828)](https://github.com/PennyLaneAI/pennylane/pull/8828)
  [(#8826)](https://github.com/PennyLaneAI/pennylane/pull/8826)

* `qml.transforms.decompose` is now imported top level as `qml.decompose`.
  [(#9011)](https://github.com/PennyLaneAI/pennylane/pull/9011)

* The `CompilePipeline` object now has an improved `__str__`, `__repr__` and `_ipython_display_` allowing improved inspectibility.
  [(#8990)](https://github.com/PennyLaneAI/pennylane/pull/8990)

* `~.specs` now includes PPR and PPM weights in its output, allowing for better categorization of PPMs and PPRs.
  [(#8983)](https://github.com/PennyLaneAI/pennylane/pull/8983)

  ```python
  
  @qml.qjit(target="mlir")
  @qml.transforms.to_ppr
  @qml.qnode(qml.device("null.qubit", wires=2))
  def circuit():
      qml.H(0)
      qml.CNOT([0, 1])
      m = qml.measure(0)
      qml.T(0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.specs(circuit, level=2)())
  Device: null.qubit
  Device wires: 2
  Shots: Shots(total=None)
  Level: 2

  Resource specifications:
      Total wire allocations: 2
      Total gates: 11
      Circuit depth: Not computed

  Gate types:
      GlobalPhase: 3
      PPR-pi/4-w1: 5
      PPR-pi/4-w2: 1
      PPM-w1: 1
      PPR-pi/8-w1: 1

  Measurements:
      expval(PauliZ): 1
  ```

* :class:`~.BBQRAM`, :class:`~.HybridQRAM`, :class:`SelectOnlyQRAM` and :class:`~.QROM` now accept 
  their classical data as a 2-dimensional array data type, which increases compatibility with Catalyst.
  [(#8791)](https://github.com/PennyLaneAI/pennylane/pull/8791)

* :class:`~.CSWAP` is now decomposed more cheaply, using ``change_op_basis`` with
  two ``CNOT`` gates and a single ``Toffoli`` gate.
  [(#8887)](https://github.com/PennyLaneAI/pennylane/pull/8887)

* `qml.vjp` and `qml.jvp` can now be captured into plxpr.
  [(#8736)](https://github.com/PennyLaneAI/pennylane/pull/8736)
  [(#8788)](https://github.com/PennyLaneAI/pennylane/pull/8788)
  [(#9019)](https://github.com/PennyLaneAI/pennylane/pull/9019)

* :func:`~.matrix` can now also be applied to a sequence of operators.
  [(#8861)](https://github.com/PennyLaneAI/pennylane/pull/8861)

* The ``qml.estimator.Resources`` class now has a nice string representation in Jupyter Notebooks.
  [(#8880)](https://github.com/PennyLaneAI/pennylane/pull/8880)

* Adds a `qml.capture.subroutine` for jitting quantum subroutines with program capture.
  [(#8912)](https://github.com/PennyLaneAI/pennylane/pull/8912)

* A function for setting up transform inputs, including setting default values and basic validation,
  can now be provided to `qml.transform` via `setup_inputs`.
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

* A new `DecompositionWarning` is now raised if the decomposition graph is unable to find a solution
  for an operator, instead of a general `UserWarning`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

* With the new graph-based decomposition system enabled, the `decompose` transform no longer raise
  warnings when the graph is unable to find a decomposition for an operator that does not define a
  decomposition in the following scenarios where operators that does not define a decomposition are
  treated as supported.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

  - When the device is `null.qubit`.
  - With `qml.compile`.
  - Within the `expand_transform` of `hadamard_grad` and `param_shift`.

* Applying `qml.ctrl` on `Snapshot` no longer produces a `Controlled(Snapshot)`. Instead, it now returns the original `Snapshot`.
  [(#9001)](https://github.com/PennyLaneAI/pennylane/pull/9001)

* When the new graph-based decomposition system is enabled, the `decompose` transform no longer tries to find
  a decomposition for an operator that is not in the statically defined gate set but meets the stopping_condition.
  [(#9036)](https://github.com/PennyLaneAI/pennylane/pull/9036)

* Updated docstring examples in the Pauli-based computation module to reflect the QEC-to-PBC
  dialect rename in Catalyst. References to ``qec.fabricate`` and ``qec.prepare`` are now
  ``pbc.fabricate`` and ``pbc.prepare``.
  [(#9071)](https://github.com/PennyLaneAI/pennylane/pull/9071)

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

<h3>Breaking changes 💔</h3>

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
  import pennylane as qml
  import numpy as np
  coeff = np.array([0.2, 0.1])

  @qml.qnode(qml.device("lightning.qubit", wires=3))                                                        
  def expval(x: float):
      qml.RX(x, 1)
      A = qml.Hamiltonian(coeff, [qml.Y(1), qml.X(0)])
      B = A @ qml.Z(2)  
      return qml.expval(B)
  ```

  ```pycon
  >>> print(qml.draw(expval)(0.4))
  0: ───────────┤ ╭<𝓗(0.20,0.10)>
  1: ──RX(0.40)─┤ ├<𝓗(0.20,0.10)>
  2: ───────────┤ ╰<𝓗(0.20,0.10)>
  ```

  However, if we convert an operator ``A`` to numerical data, from which a new 
  operator ``B`` is constructed, the chain of operator dependencies is broken and de-queuing will
  not work as expected:
  
  ```python
  coeff = np.array([0.2, 0.1])

  @qml.qnode(qml.device("lightning.qubit", wires=3))                                                        
  def expval(x: float):
      qml.RX(x, 1)
      A = qml.Hamiltonian(coeff, [qml.Y(1), qml.X(0)])
      numerical_data = A.matrix()
      B = qml.Hermitian(numerical_data, wires=[2, 0])
      return qml.expval(B)
  ```

  ```pycon
  >>> print(qp.draw(expval)(0.4))
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

* ``compute_qfunc_decomposition`` and ``has_qfunc_decomposition`` have been removed from  :class:`~.Operator`
  and all subclasses that implemented them. The graph decomposition system should be used when capture is enabled.
  [(#8922)](https://github.com/PennyLaneAI/pennylane/pull/8922)

* The :func:`pennylane.devices.preprocess.mid_circuit_measurements` transform is removed. Instead,
  the device should determine which mcm method to use, and explicitly include :func:`~pennylane.transforms.dynamic_one_shot`
  or :func:`~pennylane.transforms.defer_measurements` in its preprocess transforms if necessary. See
  :func:`DefaultQubit.setup_execution_config <pennylane.devices.DefaultQubit.setup_execution_config>` and 
  :func:`DefaultQubit.preprocess_transforms <pennylane.devices.DefaultQubit.preprocess_transforms>` for an example.
  [(#8926)](https://github.com/PennyLaneAI/pennylane/pull/8926)

* The ``custom_decomps`` keyword argument to ``qml.device`` has been removed in 0.45. Instead, 
  with ``qml.decomposition.enable_graph()``, new decomposition rules can be defined as
  quantum functions with registered resources. See :mod:`pennylane.decomposition` for more details.
  [(#8928)](https://github.com/PennyLaneAI/pennylane/pull/8928)

  As an example, consider the case of running the following circuit on a device that does not natively support ``CNOT`` gates:

  ```python
  def circuit():
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.X(1))
  ```

  Instead of defining the ``CNOT`` decomposition as:

  ```python
  def custom_cnot(wires):
    return [
      qml.Hadamard(wires=wires[1]),
      qml.CZ(wires=[wires[0], wires[1]]),
      qml.Hadamard(wires=wires[1])
    ]

  dev = qml.device('default.qubit', wires=2, custom_decomps={"CNOT" : custom_cnot})
  qnode = qml.QNode(circuit, dev)
  print(qml.draw(qnode, level="device")())
  ```

  The same result would now be obtained using:

  ```python
  @qml.decomposition.register_resources({
    qml.H: 2,
    qml.CZ: 1
  })
  def _custom_cnot_decomposition(wires, **_):
    qml.Hadamard(wires=wires[1])
    qml.CZ(wires=[wires[0], wires[1]])
    qml.Hadamard(wires=wires[1])

  qml.decomposition.add_decomps(qml.CNOT, _custom_cnot_decomposition)

  qml.decomposition.enable_graph()

  @qml.transforms.decompose(gate_set={qml.CZ, qml.H})
  def circuit():
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.X(1))

  dev = qml.device('default.qubit', wires=2)
  qnode = qml.QNode(circuit, dev)
  ```

  ```pycon
  >>> print(qml.draw(qnode, level="device")())
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

* `argnum` has been renamed `argnums` in `qml.grad`, `qml.jacobian`, `qml.jvp`, and `qml.vjp`
  to better match Catalyst and JAX.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Access to the following functions and classes from the `~pennylane.resources` module has 
  been removed. Instead, these functions must be imported from the `~pennylane.estimator` module.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

    - `qml.estimator.estimate_shots` in favor of `qml.resources.estimate_shots`
    - `qml.estimator.estimate_error` in favor of `qml.resources.estimate_error`
    - `qml.estimator.FirstQuantization` in favor of `qml.resources.FirstQuantization`
    - `qml.estimator.DoubleFactorization` in favor of `qml.resources.DoubleFactorization`

<h3>Deprecations 👋</h3>

* The ``id`` keyword argument to :class:`~.Operator` has been deprecated and will be removed in v0.46. 
  [(#8951)](https://github.com/PennyLaneAI/pennylane/pull/8951)
  [(#9051)](https://github.com/PennyLaneAI/pennylane/pull/9051)  

  The ``id`` argument previously served two purposes: (1) adding custom labels
  to operator instances which were rendered in circuit drawings and (2)
  tagging encoding gates for Fourier spectrum analysis.

  These are now handled by dedicated functions:

  > :warning: Neither of these functions are supported in a :func:`~.qjit`-compiled circuit.

  - Use :func:`~.drawer.label` to attach a custom label to an operator instance
  for circuit drawing:

      ```python
      # Legacy method (deprecated):
      qml.RX(0.5, wires=0, id="my-rx")

      # New method:
      qml.drawer.label(qml.RX(0.5, wires=0), "my-rx")
      ```

  - Use :func:`~.fourier.mark` to mark an operator as an input-encoding gate
    for :func:`~.fourier.circuit_spectrum`, and :func:`~.fourier.qnode_spectrum`:

      ```python
      # Legacy method (deprecated):
      qml.RX(0.5, wires=0, id="x0")

      # New method:
      qml.fourier.mark(qml.RX(0.5, wires=0), "x0")
      ```
  
* Setting `_queue_category=None` in an operator class in order to deactivate its instances being
  queued has been deprecated. Implement a custom `queue` method for the respective class instead.
  Operator classes that used to have `_queue_category=None` have been updated
  to `_queue_category="_ops"`, so that they are queued now.
  [(#8131)](https://github.com/PennyLaneAI/pennylane/pull/8131)

* The ``BoundTransform.transform`` property has been deprecated. Use ``BoundTransform.tape_transform`` instead.
  [(#8985)](https://github.com/PennyLaneAI/pennylane/pull/8985)

* :func:`~pennylane.tape.qscript.expand` and the related functions :func:`~pennylane.tape.expand_tape`, :func:`~pennylane.tape.expand_tape_state_prep`, and :func:`~pennylane.tape.create_expand_trainable_multipar` 
  have been deprecated and will be removed in v0.46. Instead, please use the :func:`qml.transforms.decompose <.transforms.decompose>` 
  function for decomposing circuits.
  [(#8943)](https://github.com/PennyLaneAI/pennylane/pull/8943)

* Providing a value of ``None`` to ``aux_wire`` of ``qml.gradients.hadamard_grad`` in reversed or standard mode has been
  deprecated and will no longer be supported in 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#8905)](https://github.com/PennyLaneAI/pennylane/pull/8905)

* The ``transform_program`` property of ``QNode`` has been renamed to ``compile_pipeline``.
  The deprecated access through ``transform_program`` will be removed in PennyLane v0.46.
  [(#8906)](https://github.com/PennyLaneAI/pennylane/pull/8906)

* Providing a value of ``None`` to ``aux_wire`` of ``qml.gradients.hadamard_grad`` with ``mode="reversed"`` or ``mode="standard"`` has been
  deprecated and will no longer be supported in 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#8905)](https://github.com/PennyLaneAI/pennylane/pull/8905)

* The ``qml.transforms.create_expand_fn`` has been deprecated and will be removed in v0.46.
  Instead, please use the :func:`qml.transforms.decompose <.transforms.decompose>` function for decomposing circuits.
  [(#8941)](https://github.com/PennyLaneAI/pennylane/pull/8941)
  [(#8977)](https://github.com/PennyLaneAI/pennylane/pull/8977)
  [(#9006)](https://github.com/PennyLaneAI/pennylane/pull/9006)

* The ``transform_program`` property of ``QNode`` has been renamed to ``compile_pipeline``.
  The deprecated access through ``transform_program`` will be removed in PennyLane v0.46.
  [(#8906)](https://github.com/PennyLaneAI/pennylane/pull/8906)
  [(#8945)](https://github.com/PennyLaneAI/pennylane/pull/8945)

<h3>Internal changes ⚙️</h3>

* Add `sybil` to `dev` dependency group in `pyproject.toml`.
  [(#9060)](https://github.com/PennyLaneAI/pennylane/pull/9060)

* `qml.counts` of mid circuit measurements can now be captured into jaxpr.
  [(#9022)](https://github.com/PennyLaneAI/pennylane/pull/9022)

* Pass-by-pass specs now use ``BoundTransform.tape_transform`` rather than the deprecated ``BoundTransform.transform``.
  Additionally, several internal comments have been updated to bring specs in line with the new ``CompilePipeline`` class.
  [(#9012)](https://github.com/PennyLaneAI/pennylane/pull/9012)

* Specs can now return measurement information for QJIT'd workloads when passed ``level="device"``.
  [(#8988)](https://github.com/PennyLaneAI/pennylane/pull/8988)

* Add documentation tests for the `decomposition` module.
  [(#9004)](https://github.com/PennyLaneAI/pennylane/pull/9004)

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

* Added a `qml.decomposition.toggle_graph_ctx` context manager to temporarily enable or disable graph-based
  decompositions in a thread-safe way. The fixtures `"enable_graph_decomposition"`, `"disable_graph_decomposition"`,
  and `"enable_and_disable_graph_decomp"` have been updated to use this method so that they are thread-safe.
  [(#8966)](https://github.com/PennyLaneAI/pennylane/pull/8966)

<h3>Documentation 📝</h3>

* The type of a parameter is fixed in the docstring of :class:`~.templates.layers.BasicEntanglerLayers`.
  [(#9046)](https://github.com/PennyLaneAI/pennylane/pull/9046)

<h3>Bug fixes 🐛</h3>

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

* Improves the error messages when the inputs and outputs to a `qml.for_loop` function do not match.
  [(#8984)](https://github.com/PennyLaneAI/pennylane/pull/8984)

* Fixes a bug that `qml.QubitDensityMatrix` was applied in `default.mixed` device using `qml.math.partial_trace` incorrectly.
  This would cause wrong results as described in [this issue](https://github.com/PennyLaneAI/pennylane/pull/8932).
  [(#8933)](https://github.com/PennyLaneAI/pennylane/pull/8933)

* Fixes an issue when binding a transform when the first positional arg
  is a `Sequence`, but not a `Sequence` of tapes.
  [(#8920)](https://github.com/PennyLaneAI/pennylane/pull/8920)

* Fixes a bug with `qml.estimator.templates.QSVT` which allows users to instantiate the class without
  providing wires. This is now consistent with the standard in the estimator module.
  [(#8949)](https://github.com/PennyLaneAI/pennylane/pull/8949)

* Fixes a bug where decomposition raises an error for `Pow` operators when the exponent is batched.
  [(#8969)](https://github.com/PennyLaneAI/pennylane/pull/8969)

* Fixes a bug where the `DecomposeInterpreter` cannot be applied on a `QNode` with the new graph-based decomposition system enabled.
  [(#8965)](https://github.com/PennyLaneAI/pennylane/pull/8965)

* Fixes a bug where `qml.equal` raises an error for `SProd` with abstract scalar parameters and `Exp` with abstract coefficients.
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

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi,
Astral Cai,
Yushao Chen,
Marcus Edwards,
Sengthai Heng,
Christina Lee,
Andrija Paurevic,
Omkar Sarkar,
Jay Soni,
David Wierichs,
Jake Zaia.
