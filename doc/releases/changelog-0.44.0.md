# Release 0.44.0 (current release)

<h3>New features since last release</h3>

<h4>Pass-by-Pass Circuit Specs </h4>

* Resource tracking with :func:`~pennylane.specs` can now be used to analyze the pass-by-pass impact of arbitrary 
  compilation passes on workflows compiled with :func:`~pennylane.qjit`.
  [(#8606)](https://github.com/PennyLaneAI/pennylane/pull/8606)

  Consider the following :func:`qjit <pennylane.qjit>`'d circuit with two compilation passes applied:

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

  The supplied ``level`` to :func:`pennylane.specs` may be individual `int` values, or an iterable of multiple levels. 
  Additionally, the strings ``"all"`` and ``"all-mlir"`` are allowed, returning circuit resources for all user-applied transforms
  and MLIR passes, or all user-applied MLIR passes only, respectively.

  ```pycon
  >>> print(qml.specs(circuit, level=[1, 2])(1.23))
  Device: lightning.qubit
  Device wires: 3
  Shots: Shots(total=None)
  Level: ['Before MLIR Passes (MLIR-0)', 'cancel-inverses (MLIR-1)']
  <BLANKLINE>
  Resource specifications:
  Level = Before MLIR Passes (MLIR-0):
    Total wire allocations: 3
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
    Total wire allocations: 3
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


<h4>QRAM </h4>

* A lightweight version of Bucket Brigade QRAM :class:`estimator.BBQRAM <pennylane.estimator.templates.BBQRAM>` (based on the :class:`~.estimator.resource_operator.ResourceOperator` class) 
  has been added to rapidly estimate resources used by :class:`~.BBQRAM`.
  [(#8825)](https://github.com/PennyLaneAI/pennylane/pull/8825)

* Bucket Brigade QRAM, a Hybrid QRAM and a Select-Only QRAM variant are implemented as a template :class:`~.BBQRAM`, :class:`~.HybridQRAM` and :class:`~.SelectOnlyQRAM` 
  to allow for selection of bitstrings in superposition.
  [(#8670)](https://github.com/PennyLaneAI/pennylane/pull/8670)
  [(#8679)](https://github.com/PennyLaneAI/pennylane/pull/8679)
  [(#8680)](https://github.com/PennyLaneAI/pennylane/pull/8680)
  [(#8801)](https://github.com/PennyLaneAI/pennylane/pull/8801)

<h4>Quantum Automatic Differentiation </h4>

* Quantum Automatic Differentiation implemented to allow automatic selection of optimal
  Hadamard gradient differentiation methods per [the paper](https://arxiv.org/pdf/2408.05406).
  [(#8640)](https://github.com/PennyLaneAI/pennylane/pull/8640)

<h4>Instantaneous Quantum Polynomial Circuits </h4>

* An efficient expectation value estimator has been added which may be used to train `~.IQP` circuits.
  [(#8749)](https://github.com/PennyLaneAI/pennylane/pull/8749)

* A new template for building an Instantaneous Quantum Polynomial (`~.IQP`) circuit has been added along with a 
  lightweight version (based on the :class:`~.estimator.resource_operator.ResourceOperator` class) to rapidly 
  estimate its resources. This unlocks easily estimating the resources of the IQP circuit introduced in the 
  `Train on classical, deploy on quantum <https://arxiv.org/abs/2503.02934>`_ work for generative quantum machine 
  learning.
  [(#8748)](https://github.com/PennyLaneAI/pennylane/pull/8748)

<h4>Pauli-based computation </h4>

* Users can now perform rapid Clifford+T decomposition with :func:`pennylane.qjit` using the new 
  :func:`~pennylane.transforms.gridsynth` compilation pass.
  This pass discretizes ``RZ`` and ``PhaseShift`` gates to either the Clifford+T basis or to the PPR basis.
  [(#8609)](https://github.com/PennyLaneAI/pennylane/pull/8609)
  [(#8764)](https://github.com/PennyLaneAI/pennylane/pull/8764)

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
  Total wire allocations: 3
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

* Catalyst compilation passes designed for Pauli-based computation are now available in PennyLane, 
  providing accessibility for logical compilation research by directly integrating with 
  :func:`~.pauli_measure` and :class:`~.PauliRot` operations. This includes 
  :func:`pennylane.transforms.to_ppr`, :func:`pennylane.transforms.commute_ppr`, 
  :func:`pennylane.transforms.ppr_to_ppm`, 
  :func:`pennylane.transforms.merge_ppr_ppm`, :func:`pennylane.transforms.ppm_compilation`, 
  :func:`pennylane.transforms.reduce_t_depth`, 
  [(#8762)](https://github.com/PennyLaneAI/pennylane/pull/8762)

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

<h4>Compile Pipeline and Transforms </h4>

* A Catalyst compilation pass implementing ParitySynth to resynthesize phase polynomial
  circuits has been added.
  [(#8810)](https://github.com/PennyLaneAI/pennylane/pull/8810)

* A new :class:`~.CompilePipeline` class (previously known as the `TransformProgram`) is now available
  at the top level as `qml.CompilePipeline`. Using this class you can now define large and complex
  compilation pipelines in an intuitive and flexible way. For backward compatibility, the `TransformProgram`
  class can still be accessed from `pennylane.transforms.core`.
  [(#8735)](https://github.com/PennyLaneAI/pennylane/pull/8735)

* For naming consistency, uses of the term "transform program" have been updated to "compile pipeline" across the codebase.
  Correspondingly, the module `pennylane.transforms.core.transform_program` has been renamed to
  `pennylane.transforms.core.compile_pipeline`, and the old name is no longer available.
  [(#8735)](https://github.com/PennyLaneAI/pennylane/pull/8735)

* The ``TransformDispatcher`` class has been renamed to :class:`~.transforms.core.Transform` and is now
  available at the top level as `qml.transform`. For backward compatibility, `TransformDispatcher`
  can still be accessed from `pennylane.transforms.core`.
  [(#8756)](https://github.com/PennyLaneAI/pennylane/pull/8756)

* The :class:`~.transforms.core.Transform` class (previously known as the `TransformDispatcher`), 
  :class:`~.transforms.core.BoundTransform` class (previously known as the `TransformContainer`), 
  and :class:`~.CompilePipeline` class (previously known as the `TransformProgram`) are updated to
  support intuitive composition of transform programs using `+` and `*` operators.
  [(#8703)](https://github.com/PennyLaneAI/pennylane/pull/8703)
  [(#8730)](https://github.com/PennyLaneAI/pennylane/pull/8730)

  ```pycon
  >>> import pennylane as qml
  >>> qml.transforms.merge_rotations + qml.transforms.cancel_inverses(recursive=True)
  CompilePipeline(merge_rotations, cancel_inverses)
  ```

* The following changes are made to the API of the :class:`~.CompilePipeline` (previously known as the `TransformProgram`)
  [(#8751)](https://github.com/PennyLaneAI/pennylane/pull/8751)
  [(#8774)](https://github.com/PennyLaneAI/pennylane/pull/8774)
  [(#8781)](https://github.com/PennyLaneAI/pennylane/pull/8781)

  - `push_back` is renamed to `append`, and it now accepts both :class:`~.transforms.core.Transform` and :class:`~.trasnforms.core.BoundTransform`.
  - `insert_front` and `insert_front_transform` are removed in favour of a new `insert` method which inserts a transform at any given index.
  - `get_last` is removed, use `pipeline[-1]` to access the last transform instead.
  - `pop_front` is removed in favour of a new `pop` method which removes the transform at any given index.
  - `is_empty` is removed, use `bool(pipeline)` or `len(pipeline) == 0` to check if `pipeline` is empty.
  - Added a `remove` method which removes all matching transforms from the pipeline.
  - The `prune_dynamic_transform` method is removed.

* A :class:`~.CompilePipeline` (previously known as the `TransformProgram`) can now be applied directly on a :class:`~.QNode`.
  [(#8731)](https://github.com/PennyLaneAI/pennylane/pull/8731)

  ```python
  import pennylane as qml

  pipeline = qml.transforms.merge_rotations + qml.transforms.cancel_inverses(recursive=True)

  @pipeline
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
    qml.H(0)
    qml.H(0)
    qml.RX(0.5, 1)
    qml.RX(0.2, 1)
    return qml.expval(qml.Z(0) @ qml.Z(1))
  ```
  ```pycon
  >>> print(qml.draw(circuit)())
  0: ───────────┤ ╭<Z@Z>
  1: ──RX(0.70)─┤ ╰<Z@Z>
  ```

* A :class:`~.CompilePipeline` can be initialized by passing any number of transforms or other ``CompilePipeline``s,
  providing more flexibility than the previous ``TransformProgram`` class.
  [(#8750)](https://github.com/PennyLaneAI/pennylane/pull/8750)

  ```python
  >>> pipeline = qml.CompilePipeline(qml.transforms.commute_controlled, qml.transforms.cancel_inverses)
  >>> qml.CompilePipeline(pipeline, qml.transforms.merge_rotations)
  CompilePipeline(commute_controlled, cancel_inverses, merge_rotations)
  ```

* Quantum compilation passes in MLIR and XDSL can now be applied using the core PennyLane transform
  infrastructure, instead of using Catalyst-specific tools. This is made possible by a new argument in
  :func:`~pennylane.transform` and :class:`~.transforms.core.Transform` called ``pass_name``, which 
  accepts a string corresponding to the name of the compilation pass. The ``pass_name`` argument 
  ensures that the given compilation pass will be used when `qjit` is applied to a workflow, where the 
  pass is performed in MLIR or xDSL.
  [(#8539)](https://github.com/PennyLaneAI/pennylane/pull/8539)

<h4>Analyzing your algorithms quickly and easily with resource estimation</h4>

* Now it's possible to estimate the resources of Trotterization for Pauli Hamiltonians, using the new
  :class:`estimator.PauliHamiltonian <pennylane.estimator.compact_hamiltonian.PauliHamiltonian>`
  resource Hamiltonian class and the new
  :class:`estimator.TrotterPauli <pennylane.estimator.templates.TrotterPauli>`
  resource operator.
  It's possible to access the total number of terms (Pauli words) from the `PauliHamiltonian` object directly,
  using the `PauliHamiltonian.num_terms` property to the ``qml.estimator.PauliHamiltonian`` class.
  [(#8546)](https://github.com/PennyLaneAI/pennylane/pull/8546)
  [(#8761)](https://github.com/PennyLaneAI/pennylane/pull/8761)

* Now it's possible to set precisions for a larger variety of `ResourceOperator`s in
  the :mod:`estimator <pennylane.estimator>` module, using the `resource_key` keyword argument of the
  :meth:`ResourceConfig.set_precision <pennylane.estimator.resource_config.ResourceConfig.set_precision>`
  method.
  [(#8561)](https://github.com/PennyLaneAI/pennylane/pull/8561)

* Users can now estimate the resources for the Generalized Quantum Signal Processing (GQSP)
  algorithm using :class:`estimator.GQSP <pennylane.estimator.templates.qsp.GQSP>` and
  :class:`estimator.GQSPTimeEvolution <pennylane.estimator.templates.qsp.GQSPTimeEvolution>`.
  [(#8675)](https://github.com/PennyLaneAI/pennylane/pull/8675)

* Users can now easily generate the LCU representation of a ``qml.estimator.PauliHamiltonian``
  using the new :class:`estimator.SelectPauli <pennylane.estimator.templates.select.SelectPauli>` operator.
  [(#8675)](https://github.com/PennyLaneAI/pennylane/pull/8675)

* Users can now estimate the resources for the Qubitization algorithm with two new resource
  operators: :class:`estimator.Reflection <pennylane.estimator.templates.subroutines.Reflection>` and
  :class:`estimator.Qubitization <pennylane.estimator.templates.subroutines.Qubitization>`.
  [(#8675)](https://github.com/PennyLaneAI/pennylane/pull/8675)

* Users can now estimate the resources for the Quantum Signal Processing (QSP) and Quantum Singular
  Value Transformation (QSVT) algorithms using two new resource operators: :class:`estimator.QSP <pennylane.estimator.templates.qsp.QSP>` and :class:`estimator.QSVT <pennylane.estimator.templates.qsp.QSVT>`.
  [(#8733)](https://github.com/PennyLaneAI/pennylane/pull/8733)

* Added the :class:`estimator.UnaryIterationQPE <pennylane.estimator.templates.subroutines.UnaryIterationQPE>` subroutine in the :mod:`estimator <pennylane.estimator>`
  module. It is a variant of the Qubitized Quantum Phase Estimation algorithm. This allows for reduced T and Toffoli gate count, in return
  for additional qubits used.
  [(#8708)](https://github.com/PennyLaneAI/pennylane/pull/8708)

* A new :func:`~pennylane.resource.algo_error` function has been added to compute algorithm-specific 
  errors from quantum circuits. This provides a dedicated entry point for retrieving error information 
  that was previously accessible through :func:`~pennylane.specs`. The function works with QNodes and 
  returns a dictionary of error types and their computed values.
  [(#8787)](https://github.com/PennyLaneAI/pennylane/pull/8787)

  ```python
  import pennylane as qml
  from pennylane.resource import SpectralNormError
  from pennylane.resource.error import ErrorOperation
  
  class ApproximateRX(ErrorOperation):
      def __init__(self, phi, wires):
          super().__init__(phi, wires=wires)
      
      def error(self):
          return SpectralNormError(0.01)  # simplified example
  
  dev = qml.device("default.qubit")
  
  @qml.qnode(dev)
  def circuit():
      ApproximateRX(0.5, wires=0)
      return qml.state()
  ```

  ```pycon
  >>> qml.resource.algo_error(circuit)()
  {'SpectralNormError': SpectralNormError(0.01)}
  ```

<h4>Seamless resource tracking and circuit visualization for compiled programs </h4>

* A new :func:`~.marker` function allows for easy inspection at particular points in a transform program
  with :func:`~.specs` and :func:`~.drawer.draw` instead of having to increment ``level``
  by integer amounts when not using any Catalyst passes.
  [(#8684)](https://github.com/PennyLaneAI/pennylane/pull/8684)

  The :func:`~.marker` function works like a transform in PennyLane, and can be deployed as
  a decorator on top of QNodes:

  ```python
  @qml.marker(level="rotations-merged")
  @qml.transforms.merge_rotations
  @qml.marker(level="my-level")
  @qml.transforms.cancel_inverses
  @qml.transforms.decompose(gate_set={qml.RX})
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

<h4> New templates </h4>

* A new template :class:`~.MultiplexerStatePreparation` has been added. This template allows preparing arbitrary states
  using :class:`~.SelectPauliRot` operations.
  [(#8581)](https://github.com/PennyLaneAI/pennylane/pull/8581)

  Using :class:`~.MultiplexerStatePreparation` is analogous to using other state preparation techniques in PennyLane.

  ```python
  probs_vector = np.array([0.5, 0., 0.25, 0.25])

  dev = qml.device("default.qubit", wires = 2)
  wires = [0, 1]

  @qml.qnode(dev)
  def circuit():
    qml.MultiplexerStatePreparation(np.sqrt(probs_vector), wires)
    return qml.probs(wires)
  ```
  
  ```pycon
  >>> np.round(circuit(), 2)
  array([0.5 , 0.  , 0.25, 0.25])
  ```

For theoretical details, see [arXiv:0208112](https://arxiv.org/abs/quant-ph/0208112).

<h3>Improvements 🛠</h3>

* The `ResourcesUndefinedError` has been removed from the `adjoint`, `ctrl`, and `pow` resource
  decomposition methods of `ResourceOperator` to avoid using errors as control flow.
  [(#8598)](https://github.com/PennyLaneAI/pennylane/pull/8598)

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
  [(#8754)](https://github.com/PennyLaneAI/pennylane/pull/8754)

* When program capture is enabled, `qml.adjoint` and `qml.ctrl` can now be called on
  operators that were constructed ahead of time and used as closure variables.
  [(#8816)](https://github.com/PennyLaneAI/pennylane/pull/8816)

* `qml.measure` can now be used as a frontend for `catalyst.measure`.
  [(#8782)](https://github.com/PennyLaneAI/pennylane/pull/8782)

* `qml.while_loop` and `qml.for_loop` can now lazily dispatch to catalyst when called,
  instead of dispatching upon creation.
  [(#8786)](https://github.com/PennyLaneAI/pennylane/pull/8786)

* Improved the documentation and added input validation for various operators and
  functions in the :mod:`estimator <pennylane.estimator>`.
  [(#8827)](https://github.com/PennyLaneAI/pennylane/pull/8827)
  [(#8829)](https://github.com/PennyLaneAI/pennylane/pull/8829)
  [(#8830)](https://github.com/PennyLaneAI/pennylane/pull/8830)
  [(#8832)](https://github.com/PennyLaneAI/pennylane/pull/8832)
  [(#8835)](https://github.com/PennyLaneAI/pennylane/pull/8835)

<h4>Resource estimation</h4>

* Added `Resources.total_wires` and `Resources.total_gates` properties to the 
  ``qml.estimator.Resources`` class. Users can more easily access these quantities from the `Resources` object directly.
  [(#8761)](https://github.com/PennyLaneAI/pennylane/pull/8761)

* Improved the resource decomposition for the :class:`~pennylane.estimator.QROM` class. The cost has
  been reduced in cases when users specify `restored = True` and `sel_swap_depth = 1`.
  [(#8761)](https://github.com/PennyLaneAI/pennylane/pull/8761)

* Improved :mod:`estimator <pennylane.estimator>`'s
  resource decomposition of `PauliRot` to match the optimal resources
  for certain special cases of Pauli strings (e.g. for `XX` and `YY` type Pauli strings).
  [(#8562)](https://github.com/PennyLaneAI/pennylane/pull/8562)

* Users can now estimate the resources for quantum circuits that contain or decompose into
  any of the following symbolic operators: :class:`~.ChangeOpBasis`, :class:`~.Prod`,
  :class:`~.Controlled`, :class:`~.ControlledOp`, :class:`~.Pow`, and :class:`~.Adjoint`.
  [(#8464)](https://github.com/PennyLaneAI/pennylane/pull/8464)

<h4>Decompositions</h4>

* Added decompositions of the ``RX``, ``RY`` and ``RZ`` rotations into one of the other two, as well
  as basis changing Clifford gates, to the graph-based decomposition system.
  [(#8569)](https://github.com/PennyLaneAI/pennylane/pull/8569)

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

* Qualtran call graphs built via :func:`qml.to_bloq <pennylane.to_bloq>` now use PennyLane's resource estimation
  module by default (``call_graph='estimator'``). This provides faster resource counting. 
  To use the previous behaviour based on PennyLane decompositions, set 
  ``call_graph='decomposition'``.
  [(#8390)](https://github.com/PennyLaneAI/pennylane/pull/8390)

* Added a new decomposition, `_decompose_2_cnots`, for the two-qubit decomposition for `QubitUnitary`.
  It supports the analytical decomposition a two-qubit unitary known to require exactly 2 CNOTs.
  [(#8666)](https://github.com/PennyLaneAI/pennylane/issues/8666)

* `Operator.decomposition` will fallback to the first entry in `qml.list_decomps` if the `Operator.compute_decomposition`
  method is not overridden.
  [(#8686)](https://github.com/PennyLaneAI/pennylane/pull/8686)

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

* The decompositions for several templates have been updated to use
  :class:`~.ops.op_math.ChangeOpBasis`, which makes their decompositions more resource efficient
  by eliminating unnecessary controlled operations. The templates include :class:`~.PhaseAdder`,
  :class:`~.TemporaryAND`, :class:`~.QSVT`, and :class:`~.SelectPauliRot`.
  [(#8490)](https://github.com/PennyLaneAI/pennylane/pull/8490)
  [(#8577)](https://github.com/PennyLaneAI/pennylane/pull/8577)
  [(#8721)](https://github.com/PennyLaneAI/pennylane/issues/8721)

<h4>Other improvements</h4>

* The constant to convert the length unit Bohr to Angstrom in ``qml.qchem`` is updated to use scipy constants.
  [(#8537)](https://github.com/PennyLaneAI/pennylane/pull/8537)

* :class:`~.transforms.core.TransformContainer` has been renamed to :class:`~.transforms.core.BoundTransform`.
  The old name is still available in the same location.
  [(#8753)](https://github.com/PennyLaneAI/pennylane/pull/8753)

* `qml.for_loop` will now fall back to a standard Python `for` loop if capturing a condensed, structured loop fails
  with program capture enabled.
  [(#8615)](https://github.com/PennyLaneAI/pennylane/pull/8615)

* `qml.cond` will now use standard Python logic if all predicates have concrete values. A nested
  control flow primitive will no longer be captured as it is not needed.
  [(#8634)](https://github.com/PennyLaneAI/pennylane/pull/8634)

* Added a keyword argument ``recursive`` to ``qml.transforms.cancel_inverses`` that enables
  recursive cancellation of nested pairs of mutually inverse gates. This makes the transform
  more powerful, because it can cancel larger blocks of inverse gates without having to scan
  the circuit from scratch. By default, the recursive cancellation is enabled (``recursive=True``).
  To obtain previous behaviour, disable it by setting ``recursive=False``.
  [(#8483)](https://github.com/PennyLaneAI/pennylane/pull/8483)

* `qml.grad` and `qml.jacobian` now lazily dispatch to catalyst and program
  capture, allowing for `qml.qjit(qml.grad(c))` and `qml.qjit(qml.jacobian(c))` to work.
  [(#8382)](https://github.com/PennyLaneAI/pennylane/pull/8382)

* Both the generic and transform-specific application behavior of a :class:`~.transforms.core.Transform`
  can be overwritten with `Transform.generic_register` and `my_transform.register`.
  [(#7797)](https://github.com/PennyLaneAI/pennylane/pull/7797)

* With capture enabled, measurements can now be performed on Operator instances passed as closure
  variables from outside the workflow scope.
  [(#8504)](https://github.com/PennyLaneAI/pennylane/pull/8504)

* Wires can be specified via `range` with program capture and autograph.
  [(#8500)](https://github.com/PennyLaneAI/pennylane/pull/8500)

* The :func:`~pennylane.transforms.decompose` transform no longer raises an error if both `gate_set` and
  `stopping_condition` are provided, or if `gate_set` is a dictionary, when the new graph-based decomposition
  system is disabled.
  [(#8532)](https://github.com/PennyLaneAI/pennylane/pull/8532)

* The :class:`~.pennylane.estimator.templates.SelectTHC` resource operation is upgraded to allow for a trade-off between the number of qubits and T-gates.
  This provides more flexibility in optimizing algorithms.
  [(#8682)](https://github.com/PennyLaneAI/pennylane/pull/8682)
  
* The `~pennylane.estimator.compact_hamiltonian.CDFHamiltonian`, `~pennylane.estimator.compact_hamiltonian.THCHamiltonian`,
  `~pennylane.estimator.compact_hamiltonian.VibrationalHamiltonian`, and `~pennylane.estimator.compact_hamiltonian.VibronicHamiltonian`
  classes were modified to take the 1-norm of the Hamiltonian as an optional argument.
  [(#8697)](https://github.com/PennyLaneAI/pennylane/pull/8697)

* Added a custom solver to :func:`~.transforms.intermediate_reps.rowcol` for linear systems
  over :math:`\mathbb{Z}_2` based on Gauss-Jordan elimination. This removes the need to install
  the ``galois`` package for this single function and provides a minor performance improvement.
  [(#8771)](https://github.com/PennyLaneAI/pennylane/pull/8771)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* A new transform :func:`~.transforms.select_pauli_rot_phase_gradient` has been added. It allows 
  implementing arbitrary :class:`~.SelectPauliRot` rotations with a phase gradient resource state and 
  semi-in-place addition (:class:`~.SemiAdder`).
  [(#8738)](https://github.com/PennyLaneAI/pennylane/pull/8738)

<h3>Breaking changes 💔</h3>

* The `final_transform` property of the :class:`~.transforms.core.BoundTransform` has been renamed 
  to `is_final_transform` to better follow the naming convention for boolean properties. The `transform` 
  property of the :class:`~.transforms.core.Transform` and :class:`~.transforms.core.BoundTransform` 
  has been renamed to `tape_transform` to avoid ambiguity.
  [(#8756)](https://github.com/PennyLaneAI/pennylane/pull/8756)

* Qualtran call graphs built via :func`:~.to_bloq` now return resource counts via PennyLane's resource estimation module
  instead of via PennyLane decompositions. To restore the previous behaviour, set ``call_graph='decomposition'``.
  [(#8390)](https://github.com/PennyLaneAI/pennylane/pull/8390)

  ```python
  # New default behaviour (estimator mode)
  >>> qml.to_bloq(qml.QFT(wires=range(5)), map_ops=False).call_graph()[1]
  {Hadamard(): 5, CNOT(): 26, TGate(is_adjoint=False): 1320}

  # Previous behaviour (decomposition mode)
  >>> qml.to_bloq(qml.QFT(wires=range(5)), map_ops=False, call_graph='decomposition').call_graph()[1]
  {Hadamard(): 5,
   ZPowGate(exponent=-0.15915494309189535, eps=1e-11): 10,
   ZPowGate(exponent=-0.15915494309189535, eps=5e-12): 10,
   ZPowGate(exponent=0.15915494309189535, eps=5e-12): 10,
   CNOT(): 20,
   TwoBitSwap(): 2
  }
  ```
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

* The ``max_work_wires`` argument of the :func:`~pennylane.transforms.decompose` transform has been renamed to ``num_work_wires``.
  [(#8769)](https://github.com/PennyLaneAI/pennylane/pull/8769)

* ``argnum`` has been renamed ``argnums`` for ``qml.grad``, ``qml.jacobian``, ``qml.jvp`` and ``qml.vjp``.
  [(#8496)](https://github.com/PennyLaneAI/pennylane/pull/8496)
  [(#8481)](https://github.com/PennyLaneAI/pennylane/pull/8481)

* `qml.cond`, the `QNode`, transforms, `qml.grad`, and `qml.jacobian` no longer treat all keyword arguments as static
  arguments. They are instead treated as dynamic, numerical inputs, matching the behaviour of Jax and Catalyst.
  [(#8290)](https://github.com/PennyLaneAI/pennylane/pull/8290)

* `qml.cond` will also accept a partial of an operator type as the true function without a false function
  when capture is enabled.
  [(#8776)](https://github.com/PennyLaneAI/pennylane/pull/8776)

* The :func:`~.dynamic_one_shot` transform can no longer be applied directly on a QNode.
  [(8781)](https://github.com/PennyLaneAI/pennylane/pull/8781)

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

  @qml.transforms.decompose(gate_set={"H", "T", "CNOT"}, stopping_condition=lambda op: len(op.wires) <= 2)
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

* The `_grad.py` file in split into a folder for improved source code organization.
  [(#8800)](https://github.com/PennyLaneAI/pennylane/pull/8800)

* Updated `pyproject.toml` with project dependencies to replace the requirements files. Updated workflows to use installations from `pyproject.toml`.
  [(8702)](https://github.com/PennyLaneAI/pennylane/pull/8702)

* `qml.cond`, the `QNode`, transforms, `qml.grad`, and `qml.jacobian` no longer treat all keyword arguments as static
  arguments. They are instead treated as dynamic, numerical inputs, matching the behaviour of Jax and Catalyst.
  [(#8290)](https://github.com/PennyLaneAI/pennylane/pull/8290)

* To adjust to the Python 3.14, some error messages expectations have been updated in tests; `get_type_str` added a special branch to handle `Union`.
  The import of networkx is softened to not occur on import of pennylane to work around a bug in Python 3.14.1.
  [(#8568)](https://github.com/PennyLaneAI/pennylane/pull/8568)
  [(#8737)](https://github.com/PennyLaneAI/pennylane/pull/8737)

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
  the `Transform`. Before, each transform had its own primitive stored on the
  `Transform._primitive` private property. It proved difficult to keep maintaining dispatch behaviour
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

* Solovay-Kitaev decomposition using the :func:`~.clifford_t_decompostion` transform
  with ``method="sk"`` or directly via :func:`~.ops.sk_decomposition` now raises a more
  informative ``RuntimeError`` when used with JAX-JIT or :func:`~.qjit`.
  [(#8489)](https://github.com/PennyLaneAI/pennylane/pull/8489)

* Added a `skip_decomp_matrix_check` argument to :func:`~pennylane.ops.functions.assert_valid` that
  allows the test to skip the matrix check part of testing a decomposition rule but still verify
  that the resource function is correct.
  [(#8687)](https://github.com/PennyLaneAI/pennylane/pull/8687)

* Simplified the decomposition pipeline for the estimator module. ``qre.estimate`` was updated to call the base class's `symbolic_resource_decomp` method directly.
  [(#8641)](https://github.com/PennyLaneAI/pennylane/pull/8641)
  
* Disabled autograph for the PauliRot decomposition rule as it should not be used with autograph. 
  [(#8765)](https://github.com/PennyLaneAI/pennylane/pull/8765)

<h3>Documentation 📝</h3>

* A note clarifying that the factors of a ``~.ChangeOpBasis`` are iterated in reverse order has been
  added to the documentation of ``~.ChangeOpBasis``.
  [(#8757)](https://github.com/PennyLaneAI/pennylane/pull/8757)

* The documentation of ``qml.transforms.rz_phase_gradient`` has been updated with respect to the
  sign convention of phase gradient states, how it prepares the phase gradient state in the code
  example, and the verification of the code example result.

* The code example in the documentation for ``qml.decomposition.register_resources`` has been
  updated to adhere to renamed keyword arguments and default behaviour of ``num_work_wires``.
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

* The :class:`~.GeneralizedAmplitudeDamping` error channel method has been
  updated to match the literature convention for the definition of the Kraus matrices.
  [(#8707)](https://github.com/PennyLaneAI/pennylane/pull/8707)

<h3>Bug fixes 🐛</h3>

* Fixes a bug where `_double_factorization_compressed` of `pennylane/qchem/factorization.py` used to use `X`
  for `Z` param initialization.
  [(#8689)](https://github.com/PennyLaneAI/pennylane/pull/8689)

* Use a fixed floating number tolerance from `np.finfo` in `_apply_uniform_rotation_dagger`
  to avoid numerical stability issues on some platforms.
  [(#8780)](https://github.com/PennyLaneAI/pennylane/pull/8780)

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

* Fixes a bug where :class:`~.estimator.SelectTHC`, `~.estimator.QubitizeTHC`, `~.estimator.PrepTHC` are not accounting for auxiliary
  wires correctly.
  [(#8719)](https://github.com/PennyLaneAI/pennylane/pull/8719)

* Fixes a bug where the associated `expand_transform` does not stay with the original :class:`~.transforms.core.Transform` in a :class:`~.CompilePipeline`
  during manipulations of the `CompilePipeline`.
  [(#8774)](https://github.com/PennyLaneAI/pennylane/pull/8774)

* Fixes a bug where an error is raised when `to_openqasm` is used with `qml.decomposition.enable_graph()`
  [(#8809)](https://github.com/PennyLaneAI/pennylane/pull/8809)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Runor Agbaire,
Guillermo Alonso,
Utkarsh Azad,
Joseph Bowles,
Astral Cai,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Marcus Edwards,
Lillian Frederiksen,
Diego Guala,
Sengthai Heng,
Austin Huang,
Soran Jahangiri,
Jeffrey Kam,
Jacob Kitchen,
Christina Lee,
Joseph Lee,
Anton Naim Ibrahim,
Lee J. O'Riordan,
Mudit Pandey,
Gabriela Sanchez Diaz,
Shuli Shu,
Jay Soni,
Nate Stemen,
Theodoros Trochatos,
David Wierichs,
Shifan Xu,
Hongsheng Zheng,
Zinan Zhou.
