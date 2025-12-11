# Release 0.44.0-dev (development release)

<h3>New features since last release</h3>

* :func:`~pennylane.specs` can now be used to analyze arbitrary compilation passes for
  workflows compiled with :func:`~pennylane.qjit`.

  ```python
  @qml.qjit
  @qml.transforms.merge_rotations
  @qml.transforms.cancel_inverses
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      qml.RX(x, wires=0)
      qml.X(0)
      qml.X(0)
      qml.CNOT([0, 1])
      return qml.probs()
  ```

  The supplied levels may be individual `int` values, or an iterable of multiple levels.
  The strings ``"all"`` and ``"all-mlir"`` are also allowed, and return all user-applied transforms
  and MLIR passes, or all user-applied MLIR passes only respectively.

  ```pycon
  >>> print(qml.specs(circuit, level=[1,2])(1.23))
  Device: lightning.qubit
  Device wires: 3
  Shots: Shots(total=None)
  Level: ['Before MLIR Passes (MLIR-0)', 'cancel-inverses (MLIR-1)']
  <BLANKLINE>
  Resource specifications:
  Level = Before MLIR Passes (MLIR-0):
    Total qubit allocations: 3
    Total gates: 5
    Circuit depth: Not computed
  <BLANKLINE>
    Gate types:
      RX: 2
      PauliX: 2
      CNOT: 1
  <BLANKLINE>
    Measurements:
      probs(all wires): 1
  <BLANKLINE>
  ------------------------------------------------------------
  <BLANKLINE>
  Level = cancel-inverses (MLIR-1):
    Total qubit allocations: 3
    Total gates: 3
    Circuit depth: Not computed
  <BLANKLINE>
    Gate types:
      RX: 2
      CNOT: 1
  <BLANKLINE>
    Measurements:
      probs(all wires): 1
  ```
  [(#8606)](https://github.com/PennyLaneAI/pennylane/pull/8606)

* Users can now set precisions for a larger variety of `ResourceOperator`s in
  :mod:`estimator <pennylane.estimator>` using
  :meth:`ResourceConfig.set_precision <pennylane.estimator.resource_config.ResourceConfig.set_precision>`
  thanks to the addition of the `resource_key` keyword argument.
  [(#8561)](https://github.com/PennyLaneAI/pennylane/pull/8561)

* Users can now estimate the resources of Trotterization for Pauli Hamiltonians, using the new
  :class:`estimator.PauliHamiltonian <pennylane.estimator.compact_hamiltonian.PauliHamiltonian>`
  resource Hamiltonian class and the new
  :class:`estimator.TrotterPauli <pennylane.estimator.templates.TrotterPauli>`
  resource operator.
  [(#8546)](https://github.com/PennyLaneAI/pennylane/pull/8546)

* Users can now perform rapid Clifford+T decomposition with QJIT and program capture enabled,
  using the new :func:`~pennylane.transforms.gridsynth` compilation pass.
  This pass discretizes ``RZ`` and ``PhaseShift`` gates to either the Clifford+T basis or to the PPR basis.
  [(#8609)](https://github.com/PennyLaneAI/pennylane/pull/8609)

* Quantum Automatic Differentiation implemented to allow automatic selection of optimal
  Hadamard gradient differentiation methods per [the paper](https://arxiv.org/pdf/2408.05406).
  [(#8640)](https://github.com/PennyLaneAI/pennylane/pull/8640)

* A new decomposition has been added for the Controlled :class:`~.SemiAdder`,
  which is efficient and skips controlling all gates in its decomposition.
  [(#8423)](https://github.com/PennyLaneAI/pennylane/pull/8423)

* Added a :meth:`~pennylane.devices.DeviceCapabilities.gate_set` method to :class:`~pennylane.devices.DeviceCapabilities`
  that produces a set of gate names to be used as the target gate set in decompositions.
  [(#8522)](https://github.com/PennyLaneAI/pennylane/pull/8522)

* The :func:`~pennylane.transforms.decompose` transform now accepts a `minimize_work_wires` argument. With
  the new graph-based decomposition system activated via :func:`~pennylane.decomposition.enable_graph`,
  and `minimize_work_wires` set to `True`, the decomposition system will select decomposition rules that
  minimizes the maximum number of simultaneously allocated work wires.
  [(#8729)](https://github.com/PennyLaneAI/pennylane/pull/8729)
  [(#8734)](https://github.com/PennyLaneAI/pennylane/pull/8734)

<h4>Pauli product measurements</h4>

* Writing circuits in terms of `Pauli product measurements <https://pennylane.ai/compilation/pauli-product-measurement>`_
  (PPMs) in PennyLane is now possible with the new :func:`~.pauli_measure` function.
  Using this function in tandem with :class:`~.PauliRot` to represent Pauli product rotations (PPRs) unlocks surface-code fault-tolerant quantum computing research spurred from `A Game of Surface Codes <http://arxiv.org/abs/1808.02892>`_.
  [(#8461)](https://github.com/PennyLaneAI/pennylane/pull/8461)
  [(#8631)](https://github.com/PennyLaneAI/pennylane/pull/8631)
  [(#8623)](https://github.com/PennyLaneAI/pennylane/pull/8623)
  [(#8663)](https://github.com/PennyLaneAI/pennylane/pull/8663)
  [(#8692)](https://github.com/PennyLaneAI/pennylane/pull/8692)

  The new :func:`~.pauli_measure` function is currently only for analysis on the ``null.qubit`` device, which allows for resource tracking with :func:`~.specs` and circuit inspection with :func:`~.drawer.draw`.

  In the following example, a measurement of the ``XY`` Pauli product on wires ``0`` and ``2`` is performed
  using :func:`~.pauli_measure`, followed by application of a :class:`~.PauliX` gate conditional on
  the outcome of the PPM:

  ```python
  import pennylane as qml
  
  dev = qml.device("null.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(0)
      qml.Hadamard(2)
      qml.PauliRot(np.pi / 4, pauli_word="XYZ", wires=[0, 1, 2])
      ppm = qml.pauli_measure(pauli_word="XY", wires=[0, 2])
      qml.cond(ppm, qml.X)(wires=1)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──H─╭RXYZ(0.79)─╭┤↗X├────┤  <Z>
  1: ────├RXYZ(0.79)─│──────X─┤
  2: ──H─╰RXYZ(0.79)─╰┤↗Y├──║─┤
                       ╚════╝
  ```

  By appliying the :func:`~.specs` function to the circuit above, you can easily determine its resource information.
  In this case, in addition to other gates, we can see that the circuit includes one PPR and one PPM operation (represented
  by the :class:`~.PauliRot` and :class:`~.ops.mid_measure.pauli_measure.PauliMeasure` gate types):

  ```pycon
  >>> print(qml.specs(circuit)()['resources'])
  Total qubit allocations: 3
  Total gates: 5
  Circuit depth: 4

  Gate types:
    Hadamard: 2
    PauliRot: 1
    PauliMeasure: 1
    Conditional(PauliX): 1

  Measurements:
    expval(PauliZ): 1
  ```

<h4> Compile Pipeline and Transforms </h4>

* Arithmetic dunder methods (`__add__`, `__mul__`, `__rmul__`) have been added to 
  :class:`~.transforms.core.TransformDispatcher`, :class:`~.transforms.core.TransformContainer`, 
  and :class:`~.CompilePipeline` (previously known as the `TransformProgram`) to enable intuitive composition of transform programs using `+` and `*` operators.
  [(#8703)](https://github.com/PennyLaneAI/pennylane/pull/8703)

* In the past, calling a transform with only arguments or keyword but no tapes would raise an error.
  Now, two transforms can be concatenated naturally as

  ```python
  decompose(gate_set=gate_set) + merge_rotations(1e-6)
  ```

  [(#8730)](https://github.com/PennyLaneAI/pennylane/pull/8730)

* `@partial` is not needed anymore for using transforms as decorators with arguments.
  Now, the following two usages are equivalent:

  ```python
  @partial(qml.transforms.decompose, gate_set={qml.RX, qml.CNOT})
  @qml.qnode(qml.device('default.qubit', wires=2))
  def circuit():
      qml.Hadamard(wires=0)
      qml.CZ(wires=[0,1])
      return qml.expval(qml.Z(0))
  ```

  ```python
  @qml.transforms.decompose(gate_set={qml.RX, qml.CNOT})
  @qml.qnode(qml.device('default.qubit', wires=2))
  def circuit():
      qml.Hadamard(wires=0)
      qml.CZ(wires=[0,1])
      return qml.expval(qml.Z(0))
  ```

  [(#8730)](https://github.com/PennyLaneAI/pennylane/pull/8730)

* The `TransformProgram` has been renamed to :class:`~pennylane.transforms.core.CompilePipeline`, and uses of
  the term "transform program" has been updated to "compile pipeline" across the codebase. The class is still
  accessible as `TransformProgram` from `pennylane.transforms.core`, but the module `pennylane.transforms.core.transform_program`
  has been renamed to `pennylane.transforms.core.compile_pipeline`, and the old name is no longer available.
  [(#8735)](https://github.com/PennyLaneAI/pennylane/pull/8735)

* The :class:`~pennylane.transforms.core.CompilePipeline` (previously known as `TransformProgram`)
  is available at the top level namespace as `qml.CompilePipeline`.
  [(#8735)](https://github.com/PennyLaneAI/pennylane/pull/8735)

* Now `CompilePipeline` can dispatch to anything individual transforms can dispatch onto, including
  QNodes.
  [(#8731)](https://github.com/PennyLaneAI/pennylane/pull/8731)

* The :class:`~.CompilePipeline` (previously known as the `TransformProgram`) can now be constructed
  more flexibility with a variable number of arguments that are of types `TransformDispatcher`,
  `TransformContainer`, or other `CompilePipeline`s.
  [(#8750)](https://github.com/PennyLaneAI/pennylane/pull/8750)

<h3>Improvements 🛠</h3>

* Improved :mod:`estimator <pennylane.estimator>`'s
  resource decomposition of `PauliRot` to match the optimal resources
  for certain special cases of Pauli strings (e.g. for `XX` and `YY` type Pauli strings).
  [(#8562)](https://github.com/PennyLaneAI/pennylane/pull/8562)

* Added a new decomposition, `_decompose_2_cnots`, for the two-qubit decomposition for `QubitUnitary`.
  It supports the analytical decomposition a two-qubit unitary known to require exactly 2 CNOTs.
  [(#8666)](https://github.com/PennyLaneAI/pennylane/issues/8666)

* Quantum compilation passes in MLIR and XDSL can now be applied using the core PennyLane transform
  infrastructure, instead of using Catalyst-specific tools. This is made possible by a new argument in
  :func:`~pennylane.transform` and `~.TransformDispatcher` called ``pass_name``, which accepts a string
  corresponding to the name of the compilation pass.
  The ``pass_name`` argument ensures that the given compilation pass will be used when qjit'ing a
  workflow, where the pass is performed in MLIR or xDSL.
  [(#8539)](https://github.com/PennyLaneAI/pennylane/pull/8539)

* `Operator.decomposition` will fallback to the first entry in `qml.list_decomps` if the `Operator.compute_decomposition`
  method is not overridden.
  [(#8686)](https://github.com/PennyLaneAI/pennylane/pull/8686)

* A new :func:`~.marker` function allows for easy inspection at particular points in a transform program
  with :func:`~.specs` and :func:`~.drawer.draw` instead of having to increment ``level``
  by integer amounts when not using any Catalyst passes.
  [(#8684)](https://github.com/PennyLaneAI/pennylane/pull/8684)

  The :func:`~.marker` function works like a transform in PennyLane, and can be deployed as
  a decorator on top of QNodes:

  ```
  from functools import partial

  @partial(qml.marker, level="rotations-merged")
  @qml.transforms.merge_rotations
  @partial(qml.marker, level="my-level")
  @qml.transforms.cancel_inverses
  @partial(qml.transforms.decompose, gate_set={qml.RX})
  @qml.qnode(qml.device('lightning.qubit'))
  def circuit():
      qml.RX(0.2,0)
      qml.X(0)
      qml.X(0)
      qml.RX(0.2, 0)
      return qml.state()
  ```

  The string supplied to ``marker`` can then be used as an argument to ``level`` in ``draw``
  and ``specs``, showing the cumulative result of applying transforms up to the marker:

  ```pycon
  >>> print(qml.draw(circuit, level="my-level")())
  0: ──RX(0.20)──RX(3.14)──RX(3.14)──RX(0.20)─┤  State
  >>> print(qml.draw(circuit, level="rotations-merged")())
  0: ──RX(6.68)─┤  State
  ```

* `qml.for_loop` will now fall back to a standard Python `for` loop if capturing a condensed, structured loop fails
  with program capture enabled.
  [(#8615)](https://github.com/PennyLaneAI/pennylane/pull/8615)

* `qml.cond` will now use standard Python logic if all predicates have concrete values. A nested
  control flow primitive will no longer be captured as it is not needed.
  [(#8634)](https://github.com/PennyLaneAI/pennylane/pull/8634)

* The `~.BasisRotation` graph decomposition was re-written in a qjit friendly way with PennyLane control flow.
  [(#8560)](https://github.com/PennyLaneAI/pennylane/pull/8560)
  [(#8608)](https://github.com/PennyLaneAI/pennylane/pull/8608)
  [(#8620)](https://github.com/PennyLaneAI/pennylane/pull/8620)

* The new graph based decompositions system enabled via :func:`~.decomposition.enable_graph` now supports the following
  additional templates.
  [(#8520)](https://github.com/PennyLaneAI/pennylane/pull/8520)
  [(#8515)](https://github.com/PennyLaneAI/pennylane/pull/8515)
  [(#8516)](https://github.com/PennyLaneAI/pennylane/pull/8516)
  [(#8555)](https://github.com/PennyLaneAI/pennylane/pull/8555)
  [(#8558)](https://github.com/PennyLaneAI/pennylane/pull/8558)
  [(#8538)](https://github.com/PennyLaneAI/pennylane/pull/8538)
  [(#8534)](https://github.com/PennyLaneAI/pennylane/pull/8534)
  [(#8582)](https://github.com/PennyLaneAI/pennylane/pull/8582)
  [(#8543)](https://github.com/PennyLaneAI/pennylane/pull/8543)
  [(#8554)](https://github.com/PennyLaneAI/pennylane/pull/8554)
  [(#8616)](https://github.com/PennyLaneAI/pennylane/pull/8616)
  [(#8602)](https://github.com/PennyLaneAI/pennylane/pull/8602)
  [(#8600)](https://github.com/PennyLaneAI/pennylane/pull/8600)
  [(#8601)](https://github.com/PennyLaneAI/pennylane/pull/8601)
  [(#8595)](https://github.com/PennyLaneAI/pennylane/pull/8595)
  [(#8586)](https://github.com/PennyLaneAI/pennylane/pull/8586)
  [(#8614)](https://github.com/PennyLaneAI/pennylane/pull/8614)

  - :class:`~.QSVT`
  - :class:`~.AmplitudeEmbedding`
  - :class:`~.AllSinglesDoubles`
  - :class:`~.SimplifiedTwoDesign`
  - :class:`~.GateFabric`
  - :class:`~.AngleEmbedding`
  - :class:`~.IQPEmbedding`
  - :class:`~.kUpCCGSD`
  - :class:`~.QAOAEmbedding`
  - :class:`~.BasicEntanglerLayers`
  - :class:`~.HilbertSchmidt`
  - :class:`~.LocalHilbertSchmidt`
  - :class:`~.QuantumMonteCarlo`
  - :class:`~.ArbitraryUnitary`
  - :class:`~.ApproxTimeEvolution`
  - :class:`~.ParticleConservingU2`
  - :class:`~.ParticleConservingU1`
  - :class:`~.CommutingEvolution`

* Added a keyword argument ``recursive`` to ``qml.transforms.cancel_inverses`` that enables
  recursive cancellation of nested pairs of mutually inverse gates. This makes the transform
  more powerful, because it can cancel larger blocks of inverse gates without having to scan
  the circuit from scratch. By default, the recursive cancellation is enabled (``recursive=True``).
  To obtain previous behaviour, disable it by setting ``recursive=False``.
  [(#8483)](https://github.com/PennyLaneAI/pennylane/pull/8483)

* `qml.grad` and `qml.jacobian` now lazily dispatch to catalyst and program
  capture, allowing for `qml.qjit(qml.grad(c))` and `qml.qjit(qml.jacobian(c))` to work.
  [(#8382)](https://github.com/PennyLaneAI/pennylane/pull/8382)

* Both the generic and transform-specific application behavior of a `qml.transforms.core.TransformDispatcher`
  can be overwritten with `TransformDispatcher.generic_register` and `my_transform.register`.
  [(#7797)](https://github.com/PennyLaneAI/pennylane/pull/7797)

* With capture enabled, measurements can now be performed on Operator instances passed as closure
  variables from outside the workflow scope.
  [(#8504)](https://github.com/PennyLaneAI/pennylane/pull/8504)

* Users can now estimate the resources for quantum circuits that contain or decompose into
  any of the following symbolic operators: :class:`~.ChangeOpBasis`, :class:`~.Prod`,
  :class:`~.Controlled`, :class:`~.ControlledOp`, :class:`~.Pow`, and :class:`~.Adjoint`.
  [(#8464)](https://github.com/PennyLaneAI/pennylane/pull/8464)

* Wires can be specified via `range` with program capture and autograph.
  [(#8500)](https://github.com/PennyLaneAI/pennylane/pull/8500)

* The :func:`~pennylane.transforms.decompose` transform no longer raises an error if both `gate_set` and
  `stopping_condition` are provided, or if `gate_set` is a dictionary, when the new graph-based decomposition
  system is disabled.
  [(#8532)](https://github.com/PennyLaneAI/pennylane/pull/8532)

* A new decomposition has been added to :class:`pennylane.Toffoli`. This decomposition uses one
  work wire and :class:`pennylane.TemporaryAND` operators to reduce the resources needed.
  [(#8549)](https://github.com/PennyLaneAI/pennylane/pull/8549)

* The :func:`~pennylane.pauli_decompose` now supports decomposing scipy's sparse matrices,
  allowing for efficient decomposition of large matrices that cannot fit in memory when written as
  dense arrays.
  [(#8612)](https://github.com/PennyLaneAI/pennylane/pull/8612)
  
* A decomposition has been added to the adjoint of :class:`pennylane.TemporaryAND`. This decomposition relies on mid-circuit measurments and does not require any T gates.
  [(#8633)](https://github.com/PennyLaneAI/pennylane/pull/8633)

* The graph-based decomposition system now supports decomposition rules that contains mid-circuit measurements.
  [(#8079)](https://github.com/PennyLaneAI/pennylane/pull/8079)

* The `~pennylane.estimator.compact_hamiltonian.CDFHamiltonian`, `~pennylane.estimator.compact_hamiltonian.THCHamiltonian`,
  `~pennylane.estimator.compact_hamiltonian.VibrationalHamiltonian`, and `~pennylane.estimator.compact_hamiltonian.VibronicHamiltonian`
  classes were modified to take the 1-norm of the Hamiltonian as an optional argument.
  [(#8697)](https://github.com/PennyLaneAI/pennylane/pull/8697)

* New decomposition rules that decompose to :class:`~.PauliRot` are added for the following operators.
  [(#8700)](https://github.com/PennyLaneAI/pennylane/pull/8700)
  [(#8704)](https://github.com/PennyLaneAI/pennylane/pull/8704)

  - :class:`~.CRX`, :class:`~.CRY`, :class:`~.CRZ`
  - :class:`~.ControlledPhaseShift`
  - :class:`~.IsingXX`, :class:`~.IsingYY`, :class:`~.IsingZZ`
  - :class:`~.PSWAP`
  - :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`
  - :class:`~.SingleExcitation`, :class:`~.DoubleExcitation`
  - :class:`~.SWAP`, :class:`~.ISWAP`, :class:`~.SISWAP`
  - :class:`~.CY`, :class:`~.CZ`, :class:`~.CSWAP`, :class:`~.CNOT`, :class:`~.Toffoli`

<h3>Breaking changes 💔</h3>

* The output format of `qml.specs` has been restructured into a dataclass to streamline the outputs.
  Some legacy information has been removed from the new output format.
  [(#8713)](https://github.com/PennyLaneAI/pennylane/pull/8713)

* The unified compiler, implemented in the `qml.compiler.python_compiler` submodule, has been removed from PennyLane.
  It has been migrated to Catalyst, available as `catalyst.python_interface`.
  [(#8662)](https://github.com/PennyLaneAI/pennylane/pull/8662)

* `qml.transforms.map_wires` no longer supports plxpr transforms.
  [(#8683)](https://github.com/PennyLaneAI/pennylane/pull/8683)

* ``QuantumScript.to_openqasm`` has been removed. Please use ``qml.to_openqasm`` instead. This removes duplicated
  functionality for converting a circuit to OpenQASM code.
  [(#8499)](https://github.com/PennyLaneAI/pennylane/pull/8499)

* Providing ``num_steps`` to :func:`pennylane.evolve`, :func:`pennylane.exp`, :class:`pennylane.ops.Evolution`,
  and :class:`pennylane.ops.Exp` has been disallowed. Instead, use :class:`~.TrotterProduct` for approximate
  methods, providing the ``n`` parameter to perform the Suzuki-Trotter product approximation of a Hamiltonian
  with the specified number of Trotter steps.
  [(#8474)](https://github.com/PennyLaneAI/pennylane/pull/8474)

  As a concrete example, consider the following case:

  .. code-block:: python

    coeffs = [0.5, -0.6]
    ops = [qml.X(0), qml.X(0) @ qml.Y(1)]
    H_flat = qml.dot(coeffs, ops)

  Instead of computing the Suzuki-Trotter product approximation as:

  ```pycon
  >>> qml.evolve(H_flat, num_steps=2).decomposition()
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```

  The same result can be obtained using :class:`~.TrotterProduct` as follows:

  ```pycon
  >>> decomp_ops = qml.adjoint(qml.TrotterProduct(H_flat, time=1.0, n=2)).decomposition()
  >>> [simp_op for op in decomp_ops for simp_op in map(qml.simplify, op.decomposition())]
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```

* The value ``None`` has been removed as a valid argument to the ``level`` parameter in the
  :func:`pennylane.workflow.get_transform_program`, :func:`pennylane.workflow.construct_batch`,
  :func:`pennylane.draw`, :func:`pennylane.draw_mpl`, and :func:`pennylane.specs` transforms.
  Please use ``level='device'`` instead to apply the transform at the device level.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* Access to ``add_noise``, ``insert`` and noise mitigation transforms from the ``pennylane.transforms`` module is deprecated.
  Instead, these functions should be imported from the ``pennylane.noise`` module.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* ``qml.qnn.cost.SquaredErrorLoss`` has been removed. Instead, this hybrid workflow can be accomplished
  with a function like ``loss = lambda *args: (circuit(*args) - target)**2``.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* Some unnecessary methods of the ``qml.CircuitGraph`` class have been removed:
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

  - ``print_contents`` in favor of ``print(obj)``
  - ``observables_in_order`` in favor of ``observables``
  - ``operations_in_order`` in favor of ``operations``
  - ``ancestors_in_order(obj)`` in favor of ``ancestors(obj, sort=True)``
  - ``descendants_in_order(obj)`` in favor of ``descendants(obj, sort=True)``

* ``pennylane.devices.DefaultExecutionConfig`` has been removed. Instead, use
  ``qml.devices.ExecutionConfig()`` to create a default execution configuration.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* Specifying the ``work_wire_type`` argument in ``qml.ctrl`` and other controlled operators as ``"clean"`` or
  ``"dirty"`` is disallowed. Use ``"zeroed"`` to indicate that the work wires are initially in the :math:`|0\rangle`
  state, and ``"borrowed"`` to indicate that the work wires can be in any arbitrary state. In both cases, the
  work wires are assumed to be restored to their original state upon completing the decomposition.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* `QuantumScript.shape` and `QuantumScript.numeric_type` are removed. The corresponding `MeasurementProcess`
  methods should be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* `MeasurementProcess.expand` is removed.
  `qml.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])`
  can be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* The `qml.QNode.add_transform` method is removed.
  Instead, please use `QNode.transform_program.push_back(transform_container=transform_container)`.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* The `TransformProgram` has been renamed to :class:`~pennylane.transforms.core.CompilePipeline`, and uses of
  the term "transform program" has been updated to "compile pipeline" across the codebase. The class is still
  accessible as `TransformProgram` from `pennylane.transforms.core`, but the module `pennylane.transforms.core.transform_program`
  has been renamed to `pennylane.transforms.core.compile_pipeline`, and the old name is no longer available.
  [(#8735)](https://github.com/PennyLaneAI/pennylane/pull/8735)

<h3>Deprecations 👋</h3>

* Maintenance support of NumPy<2.0 is deprecated as of v0.44 and will be completely dropped in v0.45.
  Future versions of PennyLane will only work with NumPy>=2.0.
  We recommend upgrading your version of NumPy to benefit from enhanced support and features.
  [(#8578)](https://github.com/PennyLaneAI/pennylane/pull/8578)
  [(#8497)](https://github.com/PennyLaneAI/pennylane/pull/8497)

* The ``custom_decomps`` keyword argument to ``qml.device`` has been deprecated and will be removed
  in 0.45. Instead, with ``qml.decomposition.enable_graph()``, new decomposition rules can be defined as
  quantum functions with registered resources. See :mod:`pennylane.decomposition` for more details.

* `qml.measure`, `qml.measurements.MidMeasureMP`, `qml.measurements.MeasurementValue`,
  and `qml.measurements.get_mcm_predicates` are now located in `qml.ops.mid_measure`.
  `MidMeasureMP` is now renamed to `MidMeasure`.
  `qml.measurements.find_post_processed_mcms` is now `qml.devices.qubit.simulate._find_post_processed_mcms`,
  and is being made private, as it is an utility for tree-traversal.
  [(#8466)](https://github.com/PennyLaneAI/pennylane/pull/8466)

* The ``pennylane.operation.Operator.is_hermitian`` property has been deprecated and renamed
  to ``pennylane.operation.Operator.is_verified_hermitian`` as it better reflects the functionality of this property.
  The deprecated access through ``is_hermitian`` will be removed in PennyLane v0.45.
  Alternatively, consider using the ``pennylane.is_hermitian`` function instead as it provides a more reliable check for hermiticity.
  Please be aware that it comes with a higher computational cost.
  [(#8494)](https://github.com/PennyLaneAI/pennylane/pull/8494)

* Access to the follow functions and classes from the ``pennylane.resources`` module are deprecated. Instead, these functions must be imported from the ``pennylane.estimator`` module.
  [(#8484)](https://github.com/PennyLaneAI/pennylane/pull/8484)

  - ``qml.estimator.estimate_shots`` in favor of ``qml.resources.estimate_shots``
  - ``qml.estimator.estimate_error`` in favor of ``qml.resources.estimate_error``
  - ``qml.estimator.FirstQuantization`` in favor of ``qml.resources.FirstQuantization``
  - ``qml.estimator.DoubleFactorization`` in favor of ``qml.resources.DoubleFactorization``

* ``argnum`` has been renamed ``argnums`` for ``qml.grad``, ``qml.jacobian``, ``qml.jvp`` and ``qml.vjp``.
  [(#8496)](https://github.com/PennyLaneAI/pennylane/pull/8496)
  [(#8481)](https://github.com/PennyLaneAI/pennylane/pull/8481)

* The :func:`pennylane.devices.preprocess.mid_circuit_measurements` transform is deprecated. Instead,
  the device should determine which mcm method to use, and explicitly include :func:`~pennylane.transforms.dynamic_one_shot`
  or :func:`~pennylane.transforms.defer_measurements` in its preprocess transforms if necessary.
  [(#8467)](https://github.com/PennyLaneAI/pennylane/pull/8467)

* Passing a function to the ``gate_set`` argument in the :func:`~pennylane.transforms.decompose` transform
  is deprecated. The ``gate_set`` argument expects a static iterable of operator type and/or operator names,
  and the function should be passed to the ``stopping_condition`` argument instead.
  [(#8533)](https://github.com/PennyLaneAI/pennylane/pull/8533)

  The example below illustrates how you can provide a function as the ``stopping_condition`` in addition to providing a
  ``gate_set``. The decomposition of each operator will then stop once it reaches the gates in the ``gate_set`` or the
  ``stopping_condition`` is satisfied.

  ```python
  import pennylane as qml
  from functools import partial

  @partial(qml.transforms.decompose, gate_set={"H", "T", "CNOT"}, stopping_condition=lambda op: len(op.wires) <= 2)
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.Hadamard(wires=[0])
      qml.Toffoli(wires=[0,1,2])
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──H────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
  1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
  2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤
  ```

<h3>Internal changes ⚙️</h3>

* `qml.cond`, the `QNode`, transforms, `qml.grad`, and `qml.jacobian` no longer treat all keyword arguments as static
  arguments. They are instead treated as dynamic, numerical inputs, matching the behaviour of Jax and Catalyst.
  [(#8290)](https://github.com/PennyLaneAI/pennylane/pull/8290)

* To adjust to the Python 3.14, some error messages expectations have been updated in tests; `get_type_str` added a special branch to handle `Union`.
  [(#8568)](https://github.com/PennyLaneAI/pennylane/pull/8568)

* Bump `jax` version to `0.7.1` for `capture` module.
  [(#8715)](https://github.com/PennyLaneAI/pennylane/pull/8715)

* Bump `jax` version to `0.7.0` for `capture` module.
  [(#8701)](https://github.com/PennyLaneAI/pennylane/pull/8701)

* Improve error handling when using PennyLane's experimental program capture functionality with an incompatible JAX version.
  [(#8723)](https://github.com/PennyLaneAI/pennylane/pull/8723)

* Bump `autoray` package version to `0.8.2`.
  [(#8674)](https://github.com/PennyLaneAI/pennylane/pull/8674)

* Update the schedule of nightly TestPyPI uploads to occur at the end rather than the beginning of all week days.
  [(#8672)](https://github.com/PennyLaneAI/pennylane/pull/8672)

* Add workflow to bump Catalyst and Lightning versions in the RC branch, create a new release tag and draft release, tag the RC branch, and create a PR to merge the RC branch into master.
  [(#8352)](https://github.com/PennyLaneAI/pennylane/pull/8352)

* Added `MCM_METHOD` and `POSTSELECT_MODE` `StrEnum` objects to improve validation and handling of `MCMConfig` creation.
  [(#8596)](https://github.com/PennyLaneAI/pennylane/pull/8596)

* Updated various docstrings to be compatible with the new documentation testing approach.
  [(#8635)](https://github.com/PennyLaneAI/pennylane/pull/8635)

* In program capture, transforms now have a single transform primitive that have a `transform` param that stores
  the `TransformDispatcher`. Before, each transform had its own primitive stored on the
  `TransformDispatcher._primitive` private property. It proved difficult to keep maintaining dispatch behaviour
  for every single transform.
  [(#8576)](https://github.com/PennyLaneAI/pennylane/pull/8576)
  [(#8639)](https://github.com/PennyLaneAI/pennylane/pull/8639)

* Updated documentation check workflow to run on pull requests on `v[0-9]+\.[0-9]+\.[0-9]+-docs` branches.
  [(#8590)](https://github.com/PennyLaneAI/pennylane/pull/8590)

* When program capture is enabled, there is no longer caching of the jaxpr on the QNode.
  [(#8629)](https://github.com/PennyLaneAI/pennylane/pull/8629)

* The `grad` and `jacobian` primitives now store the function under `fn`. There is also now a single `jacobian_p`
  primitive for use in program capture.
  [(#8357)](https://github.com/PennyLaneAI/pennylane/pull/8357)

* Update versions for `pylint`, `isort` and `black` in `format.yml`
  [(#8506)](https://github.com/PennyLaneAI/pennylane/pull/8506)

* Reclassifies `registers` as a tertiary module for use with tach.
  [(#8513)](https://github.com/PennyLaneAI/pennylane/pull/8513)

* The :class:`~pennylane.devices.LegacyDeviceFacade` is slightly refactored to implement `setup_execution_config` and `preprocess_transforms`
  separately as opposed to implementing a single `preprocess` method. Additionally, the `mid_circuit_measurements` transform has been removed
  from the preprocess transform program. Instead, the best mcm method is chosen in `setup_execution_config`. By default, the ``_capabilities``
  dictionary is queried for the ``"supports_mid_measure"`` property. If the underlying device defines a TOML file, the ``supported_mcm_methods``
  field in the TOML file is used as the source of truth.
  [(#8469)](https://github.com/PennyLaneAI/pennylane/pull/8469)
  [(#8486)](https://github.com/PennyLaneAI/pennylane/pull/8486)
  [(#8495)](https://github.com/PennyLaneAI/pennylane/pull/8495)

* The various private functions of the :class:`~pennylane.estimator.FirstQuantization` class have
  been modified to avoid using `numpy.matrix` as this function is deprecated.
  [(#8523)](https://github.com/PennyLaneAI/pennylane/pull/8523)

* The `ftqc` module now includes dummy transforms for several Catalyst/MLIR passes (`to-ppr`, `commute-ppr`, `merge-ppr-ppm`,
  `decompose-clifford-ppr`, `decompose-non-clifford-ppr`, `ppr-to-ppm`, `ppr-to-mbqc` and `reduce-t-depth`), to allow them to
  be captured as primitives in PLxPR and mapped to the MLIR passes in Catalyst. This enables using the passes with the unified
  compiler and program capture.
  [(#8519)](https://github.com/PennyLaneAI/pennylane/pull/8519)
  [(#8544)](https://github.com/PennyLaneAI/pennylane/pull/8544)

* The decompositions for several templates have been updated to use
  :class:`~.ops.op_math.ChangeOpBasis`, which makes their decompositions more resource efficient
  by eliminating unnecessary controlled operations. The templates include :class:`~.PhaseAdder`,
  :class:`~.TemporaryAND`, :class:`~.QSVT`, and :class:`~.SelectPauliRot`.
  [(#8490)](https://github.com/PennyLaneAI/pennylane/pull/8490)
  [(#8577)](https://github.com/PennyLaneAI/pennylane/pull/8577)
  [(#8721)](https://github.com/PennyLaneAI/pennylane/issues/8721)

* The constant to convert the length unit Bohr to Angstrom in ``qml.qchem`` is updated to use scipy
  constants.
  [(#8537)](https://github.com/PennyLaneAI/pennylane/pull/8537)

* Solovay-Kitaev decomposition using the :func:`~.clifford_t_decompostion` transform
  with ``method="sk"`` or directly via :func:`~.ops.sk_decomposition` now raises a more
  informative ``RuntimeError`` when used with JAX-JIT or :func:`~.qjit`.
  [(#8489)](https://github.com/PennyLaneAI/pennylane/pull/8489)

* Added a `skip_decomp_matrix_check` argument to :func:`~pennylane.ops.functions.assert_valid` that
  allows the test to skip the matrix check part of testing a decomposition rule but still verify
  that the resource function is correct.
  [(#8687)](https://github.com/PennyLaneAI/pennylane/pull/8687)

<h3>Documentation 📝</h3>

* The documentation of ``qml.transforms.rz_phase_gradient`` has been updated with respect to the
  sign convention of phase gradient states, how it prepares the phase gradient state in the code
  example, and the verification of the code example result.

* The code example in the documentation for ``qml.decomposition.register_resources`` has been
  updated to adhere to renamed keyword arguments and default behaviour of ``max_work_wires``.
  [(#8536)](https://github.com/PennyLaneAI/pennylane/pull/8536)

* The docstring for ``qml.device`` has been updated to include a section on custom decompositions,
  and a warning about the removal of the ``custom_decomps`` kwarg in v0.45. Additionally, the page
  :doc:`Building a plugin <../development/plugins>` now includes instructions on using
  the :func:`~pennylane.devices.preprocess.decompose` transform for device-level decompositions.
  The documentation for :doc:`Compiling circuits <../introduction/compiling_circuits>` has also been
  updated with a warning message about ``custom_decomps`` future removal.
  [(#8492)](https://github.com/PennyLaneAI/pennylane/pull/8492)
  [(#8564)](https://github.com/PennyLaneAI/pennylane/pull/8564)

A warning message has been added to :doc:`Building a plugin <../development/plugins>`
  docstring for ``qml.device`` has been updated to include a section on custom decompositions,
  and a warning about the removal of the ``custom_decomps`` kwarg in v0.44. Additionally, the page
  :doc:`Building a plugin <../development/plugins>` now includes instructions on using
  the :func:`~pennylane.devices.preprocess.decompose` transform for device-level decompositions.
  [(#8492)](https://github.com/PennyLaneAI/pennylane/pull/8492)

* Improves documentation in the transforms module and adds documentation testing for it.
  [(#8557)](https://github.com/PennyLaneAI/pennylane/pull/8557)

<h3>Bug fixes 🐛</h3>

* Handles floating point errors in the norm of the state when applying
  mid circuit measurements.
  [(#8741)](https://github.com/PennyLaneAI/pennylane/pull/8741)

* Update `interface-unit-tests.yml` to use its input parameter `pytest_additional_args` when running pytest.
  [(#8705)](https://github.com/PennyLaneAI/pennylane/pull/8705)

* Fixes a bug where in `resolve_work_wire_type` we incorrectly returned a value of `zeroed` if `both work_wires`
  and `base_work_wires` were empty, causing an incorrect work wire type.
  [(#8718)](https://github.com/PennyLaneAI/pennylane/pull/8718)

* The warnings-as-errors CI action was failing due to an incompatibility between `pytest-xdist` and `pytest-benchmark`.
  Disabling the benchmark package allows the tests to be collected an executed.
  [(#8699)](https://github.com/PennyLaneAI/pennylane/pull/8699)

* Adds an `expand_transform` to `param_shift_hessian` to pre-decompose
  operations till they are supported.
  [(#8698)](https://github.com/PennyLaneAI/pennylane/pull/8698)

* Fixes a bug in `default.mixed` device where certain diagonal operations were incorrectly
  reshaped during application when using broadcasting.
  [(#8593)](https://github.com/PennyLaneAI/pennylane/pull/8593)

* Add an exception to the warning for unsolved operators within the graph-based decomposition
  system if the unsolved operators are :class:`.allocation.Allocate` or :class:`.allocation.Deallocate`.
  [(#8553)](https://github.com/PennyLaneAI/pennylane/pull/8553)

* Fixes a bug in `clifford_t_decomposition` with `method="gridsynth"` and qjit, where using cached decomposition with the same parameter causes an error.
  [(#8535)](https://github.com/PennyLaneAI/pennylane/pull/8535)

* Fixes a bug in :class:`~.SemiAdder` where the results were incorrect when more ``work_wires`` than required were passed.
 [(#8423)](https://github.com/PennyLaneAI/pennylane/pull/8423)

* Fixes a bug where the deferred measurement method is used silently even if ``mcm_method="one-shot"`` is explicitly requested,
  when a device that extends the ``LegacyDevice`` does not declare support for mid-circuit measurements.
  [(#8486)](https://github.com/PennyLaneAI/pennylane/pull/8486)

* Fixes a bug where a `KeyError` is raised when querying the decomposition rule for an operator in the gate set from a :class:`~pennylane.decomposition.DecompGraphSolution`.
  [(#8526)](https://github.com/PennyLaneAI/pennylane/pull/8526)

* Fixes a bug where mid-circuit measurements were generating incomplete QASM.
  [(#8556)](https://github.com/PennyLaneAI/pennylane/pull/8556)

* Fixes a bug where `qml.specs` incorrectly computes the circuit depth when classically controlled operators are involved.
  [(#8668)](https://github.com/PennyLaneAI/pennylane/pull/8668)

* Fixes a bug where an error is raised when trying to decompose a nested composite operator with capture and the new graph system enabled.
  [(#8695)](https://github.com/PennyLaneAI/pennylane/pull/8695)

* Fixes a bug where :func:`~.change_op_basis` cannot be captured when the `uncompute_op` is left out.
  [(#8695)](https://github.com/PennyLaneAI/pennylane/pull/8695)

* Fixes a bug in :func:`~qml.ops.rs_decomposition` where correct solution candidates were being rejected
  due to some incorrect GCD computations.
  [(#8625)](https://github.com/PennyLaneAI/pennylane/pull/8625)

* Fixes a bug where decomposition rules are sometimes incorrectly disregarded by the `DecompositionGraph` when a higher level
  decomposition rule uses dynamically allocated work wires.
  [(#8725)](https://github.com/PennyLaneAI/pennylane/pull/8725)

* Fixes a bug where :class:`~.ops.ChangeOpBasis` is not correctly reconstructed using `qml.pytrees.unflatten(*qml.pytrees.flatten(op))`
  [(#8721)](https://github.com/PennyLaneAI/pennylane/issues/8721)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Utkarsh Azad,
Astral Cai,
Yushao Chen,
Marcus Edwards,
Lillian Frederiksen,
Sengthai Heng,
Soran Jahangiri,
Jacob Kitchen,
Christina Lee,
Joseph Lee,
Lee J. O'Riordan,
Gabriela Sanchez Diaz,
Mudit Pandey,
Shuli Shu,
Jay Soni,
nate stemen,
Theodoros Trochatos,
David Wierichs,
Hongsheng Zheng,
Zinan Zhou
