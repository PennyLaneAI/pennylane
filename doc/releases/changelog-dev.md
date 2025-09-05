
# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

* Dynamic wire allocation with `qml.allocation.allocate` can now be executed on `default.qubit`.
  [(#7718)](https://github.com/PennyLaneAI/pennylane/pull/7718)

  ```python
  @qml.qnode(qml.device('default.qubit'))
  def c():
      with qml.allocate(1) as wires:
          qml.H(wires)
          qml.CNOT((wires[0], 0))
      return qml.probs(wires=0)

  c()
  ```
  ```
  array([0.5, 0.5])
  ```

* A new :func:`~.ops.op_math.change_basis_op` function and :class:`~.ops.op_math.ChangeOpBasis` class were added,
  which allow a compute-uncompute pattern (U V U†) to be represented by a single operator.
  A corresponding decomposition rule has been added to support efficiently controlling the pattern,
  in which only the central (target) operator is controlled, and not U or U†.
  [(#8023)](https://github.com/PennyLaneAI/pennylane/pull/8023)
  [(#8070)](https://github.com/PennyLaneAI/pennylane/pull/8070)

* A new keyword argument ``partial`` has been added to :class:`qml.Select`. It allows for 
  simplifications in the decomposition of ``Select`` under the assumption that the state of the
  control wires has no overlap with computational basis states that are not used by ``Select``.
  [(#7658)](https://github.com/PennyLaneAI/pennylane/pull/7658)

* New ZX calculus-based transforms have been added to access circuit optimization
  passes implemented in [pyzx](https://pyzx.readthedocs.io/en/latest/):

  * :func:`~.transforms.zx.push_hadamards` to optimize a phase-polynomial + Hadamard circuit by pushing
    Hadamard gates as far as possible to one side to create fewer larger phase-polynomial blocks
    (see [pyzx.basic_optimization](https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.basic_optimization)).
    [(#8025)](https://github.com/PennyLaneAI/pennylane/pull/8025)

  * :func:`~.transforms.zx.todd` to optimize a Clifford + T circuit by using the Third Order Duplicate and Destroy (TODD) algorithm
    (see [pyzx.phase_block_optimize](https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.phase_block_optimize)).
    [(#8029)](https://github.com/PennyLaneAI/pennylane/pull/8029)

  * :func:`~.transforms.zx.optimize_t_count` to reduce the number of T gates in a Clifford + T circuit by applying
    a sequence of passes that combine ZX-based commutation and cancellation rules and the TODD algorithm
    (see [pyzx.full_optimize](https://pyzx.readthedocs.io/en/latest/api.html#pyzx.optimize.full_optimize)).
    [(#8088)](https://github.com/PennyLaneAI/pennylane/pull/8088)

  * :func:`~.transforms.zx.reduce_non_clifford` to reduce the number of non-Clifford gates by applying
    a combination of phase gadgetization strategies and Clifford gate simplification rules.
    (see [pyzx.full_reduce](https://pyzx.readthedocs.io/en/latest/api.html#pyzx.simplify.full_reduce)).
    [(#7747)](https://github.com/PennyLaneAI/pennylane/pull/7747)

* The `qml.specs` function now accepts a `compute_depth` keyword argument, which is set to `True` by default.
  This makes the expensive depth computation performed by `qml.specs` optional.
  [(#7998)](https://github.com/PennyLaneAI/pennylane/pull/7998)
  [(#8042)](https://github.com/PennyLaneAI/pennylane/pull/8042)

* New transforms called :func:`~.transforms.match_relative_phase_toffoli` and 
  :func:`~.transforms.match_controlled_iX_gate` have been added to implement passes that make use
  of equivalencies to compile certain patterns to efficient Clifford+T equivalents.
  [(#7748)](https://github.com/PennyLaneAI/pennylane/pull/7748)

* Leveraging quantum just-in-time compilation to optimize parameterized hybrid workflows with the momentum
  quantum natural gradient optimizer is now possible with the new :class:`~.MomentumQNGOptimizerQJIT` optimizer.
  [(#7606)](https://github.com/PennyLaneAI/pennylane/pull/7606)

  Similar to the :class:`~.QNGOptimizerQJIT` optimizer, :class:`~.MomentumQNGOptimizerQJIT` offers a
  `qml.qjit`-compatible analogue to the existing :class:`~.MomentumQNGOptimizer` with an Optax-like interface:

  ```python
  import pennylane as qml
  import jax.numpy as jnp

  dev = qml.device("lightning.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(params):
      qml.RX(params[0], wires=0)
      qml.RY(params[1], wires=1)
      return qml.expval(qml.Z(0) + qml.X(1))

  opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.1, momentum=0.2)

  @qml.qjit
  def update_step_qjit(i, args):
      params, state = args
      return opt.step(circuit, params, state)

  @qml.qjit
  def optimization_qjit(params, iters):
      state = opt.init(params)
      args = (params, state)
      params, state = qml.for_loop(iters)(update_step_qjit)(args)
      return params
  ```

  ```pycon
  >>> params = jnp.array([0.1, 0.2])
  >>> iters = 1000
  >>> optimization_qjit(params=params, iters=iters)
  Array([ 3.14159265, -1.57079633], dtype=float64)
  ```

* The :func:`~.transforms.decompose` transform is now able to decompose classically controlled operations.
  [(#8145)](https://github.com/PennyLaneAI/pennylane/pull/8145)

<h3>Improvements 🛠</h3>

* `allocate` and `deallocate` can now be accessed as `qml.allocate` and `qml.deallocate`.
  [(#8189)](https://github.com/PennyLaneAI/pennylane/pull/8198))

* `allocate` now takes `state: Literal["zero", "any"] = "zero"` instead of `require_zeros=True`.
  [(#8163)](https://github.com/PennyLaneAI/pennylane/pull/8163)

* A `DynamicRegister` can no longer be used as an individual wire itself, as this led to confusing results.
  [(#8151)](https://github.com/PennyLaneAI/pennylane/pull/8151)

* A new keyword argument called ``shot_dist`` has been added to the :func:`~.transforms.split_non_commuting` transform.
  This allows for more customization and efficiency when calculating expectation values across the non-commuting groups
  of observables that make up a ``Hamiltonian``/``LinearCombination``.
  [(#7988)](https://github.com/PennyLaneAI/pennylane/pull/7988)

  Given a QNode that returns a sample-based measurement (e.g., ``expval``) of a ``Hamiltonian``/``LinearCombination``
  with finite ``shots``, the current default behaviour of :func:`~.transforms.split_non_commuting` will perform ``shots``
  executions for each group of commuting terms. With the ``shot_dist`` argument, this behaviour can be changed:

  * ``"uniform"``: evenly distributes the number of ``shots`` across all groups of commuting terms
  * ``"weighted"``: distributes the number of ``shots`` according to weights proportional to the L1 norm of the coefficients in each group
  * ``"weighted_random"``: same as ``"weighted"``, but the numbers of ``shots`` are sampled from a multinomial distribution
  * or a user-defined function implementing a custom shot distribution strategy

  To show an example about how this works, let's start by defining a simple Hamiltonian:

  ```python
  import pennylane as qml

  ham = qml.Hamiltonian(
      coeffs=[10, 0.1, 20, 100, 0.2],
      observables=[
          qml.X(0) @ qml.Y(1),
          qml.Z(0) @ qml.Z(2),
          qml.Y(1),
          qml.X(1) @ qml.X(2),
          qml.Z(0) @ qml.Z(1) @ qml.Z(2)
      ]
  )
  ```

  This Hamiltonian can be split into 3 non-commuting groups of mutually commuting terms.
  With ``shot_dist = "weighted"``, for example, the number of shots will be divided
  according to the L1 norm of each group's coefficients:

  ```python
  from functools import partial
  from pennylane.transforms import split_non_commuting

  dev = qml.device("default.qubit")

  @partial(split_non_commuting, shot_dist="weighted")
  @qml.qnode(dev, shots=10000)
  def circuit():
      return qml.expval(ham)

  with qml.Tracker(dev) as tracker:
      circuit()
  ```

  ```pycon
  >>> print(tracker.history["shots"])
  [2303, 23, 7674]
  ```

* The number of `shots` can now be specified directly in QNodes as a standard keyword argument.
  [(#8073)](https://github.com/PennyLaneAI/pennylane/pull/8073)

  ```python
  @qml.qnode(qml.device("default.qubit"), shots=1000)
  def circuit():
      qml.H(0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> circuit.shots
  Shots(total=1000)
  >>> circuit()
  np.float64(-0.004)
  ```

  Setting the `shots` value in a QNode is equivalent to decorating with :func:`qml.workflow.set_shots`. Note, however, that decorating with :func:`qml.workflow.set_shots` overrides QNode `shots`:

  ```pycon
  >>> new_circ = qml.set_shots(circuit, shots=123)
  >>> new_circ.shots
  Shots(total=123)
  ```

* PennyLane `autograph` supports standard python for updating arrays like `array[i] += x` instead of jax `arr.at[i].add(x)`. 
  Users can now use this when designing quantum circuits with experimental program capture enabled.

  ```python
  import pennylane as qml
  import jax.numpy as jnp

  qml.capture.enable()

  @qml.qnode(qml.device("default.qubit", wires=3))
  def circuit(val):
    angles = jnp.zeros(3)
    angles[0:3] += val

    for i, angle in enumerate(angles):
        qml.RX(angle, i)

    return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2))
  ```

  ```pycon
  >>> circuit(jnp.pi)
  (Array(-1, dtype=float32),
   Array(-1, dtype=float32),
   Array(-1, dtype=float32)) 
  ```

  [(#8076)](https://github.com/PennyLaneAI/pennylane/pull/8076)

* PennyLane `autograph` supports standard python for index assignment (`arr[i] = x`) instead of jax.numpy form (`arr = arr.at[i].set(x)`).
  Users can now use standard python assignment when designing circuits with experimental program capture enabled.

  ```python
  import pennylane as qml
  import jax.numpy as jnp

  qml.capture.enable()

  @qml.qnode(qml.device("default.qubit", wires=3))
  def circuit(val):
    angles = jnp.zeros(3)
    angles[1] = val / 2
    angles[2] = val

    for i, angle in enumerate(angles):
        qml.RX(angle, i)

    return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2))
  ```

  ```pycon
  >>> circuit(jnp.pi)
  (Array(0.99999994, dtype=float32),
   Array(0., dtype=float32),
   Array(-0.99999994, dtype=float32)) 
  ```

  [(#8027)](https://github.com/PennyLaneAI/pennylane/pull/8027)

* Logical operations (`and`, `or` and `not`) are now supported with the `autograph` module. Users can
  now use these logical operations in control flow when designing quantum circuits with experimental
  program capture enabled.

  ```python
  import pennylane as qml

  qml.capture.enable()

  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit(param):
      if param >= 0 and param <= 1:
          qml.H(0)
      return qml.state()
  ```

  ```pycon
  >>> circuit(0.5)
  Array([0.70710677+0.j, 0.70710677+0.j], dtype=complex64)
  ```

  [(#8006)](https://github.com/PennyLaneAI/pennylane/pull/8006)

* The decomposition of :class:`~.BasisRotation` has been optimized to skip redundant phase shift gates
  with angle :math:`\pm \pi` for real-valued, i.e., orthogonal, rotation matrices. This uses the fact that
  no or single :class:`~.PhaseShift` gate is required in case the matrix has a determinant :math:`\pm 1`.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

* Changed how basis states are assigned internally in `qml.Superposition`, improving its
  decomposition slightly both regarding classical computing time and gate decomposition.
  [(#7880)](https://github.com/PennyLaneAI/pennylane/pull/7880)

* The printing and drawing of :class:`~.TemporaryAND`, also known as ``qml.Elbow``, and its adjoint
  have been improved to be more legible and consistent with how it's depicted in circuits in the literature.
  [(#8017)](https://github.com/PennyLaneAI/pennylane/pull/8017)

  ```python
  import pennylane as qml

  @qml.draw
  @qml.qnode(qml.device("lightning.qubit", wires=4))
  def node():
      qml.TemporaryAND([0, 1, 2], control_values=[1, 0])
      qml.CNOT([2, 3])
      qml.adjoint(qml.TemporaryAND([0, 1, 2], control_values=[1, 0]))
      return qml.expval(qml.Z(3))
  ```

  ```pycon
  print(node())
  0: ─╭●─────●╮─┤     
  1: ─├○─────○┤─┤     
  2: ─╰──╭●───╯─┤     
  3: ────╰X─────┤  <Z>
  ```

* Several templates now have decompositions that can be accessed within the graph-based
  decomposition system (:func:`~.decomposition.enable_graph`), allowing workflows
  that include these templates to be decomposed in a resource-efficient and performant
  manner.
  [(#7779)](https://github.com/PennyLaneAI/pennylane/pull/7779)
  [(#7908)](https://github.com/PennyLaneAI/pennylane/pull/7908)
  [(#7385)](https://github.com/PennyLaneAI/pennylane/pull/7385)
  [(#7941)](https://github.com/PennyLaneAI/pennylane/pull/7941)
  [(#7943)](https://github.com/PennyLaneAI/pennylane/pull/7943)
  [(#8075)](https://github.com/PennyLaneAI/pennylane/pull/8075)
  [(#8002)](https://github.com/PennyLaneAI/pennylane/pull/8002)
  
  The included templates are: :class:`~.Adder`, :class:`~.ControlledSequence`, :class:`~.ModExp`, :class:`~.MottonenStatePreparation`, 
  :class:`~.MPSPrep`, :class:`~.Multiplier`, :class:`~.OutAdder`, :class:`~.OutMultiplier`, :class:`~.OutPoly`, :class:`~.PrepSelPrep`,
  :class:`~.ops.Prod`, :class:`~.Reflection`, :class:`~.Select`, :class:`~.StatePrep`, :class:`~.TrotterProduct`, :class:`~.QROM`, 
  :class:`~.GroverOperator`, :class:`~.UCCSD`, :class:`~.StronglyEntanglingLayers`, :class:`~.GQSP`, :class:`~.FermionicSingleExcitation`, 
  :class:`~.FermionicDoubleExcitation`, :class:`~.QROM`, :class:`~.ArbitraryStatePreparation`, :class:`~.CosineWindow`, 
  :class:`~.AmplitudeAmplification`, :class:`~.Permute`, :class:`~.AQFT`, :class:`~.FlipSign`, :class:`~.FABLE`,
  :class:`~.Qubitization`, and :class:`~.Superposition`

* A new function called :func:`~.math.choi_matrix` is available, which computes the [Choi matrix](https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism) of a quantum channel.
  This is a useful tool in quantum information science and to check circuit identities involving non-unitary operations.
  [(#7951)](https://github.com/PennyLaneAI/pennylane/pull/7951)

  ```pycon
  >>> import numpy as np
  >>> Ks = [np.sqrt(0.3) * qml.CNOT((0, 1)), np.sqrt(1-0.3) * qml.X(0)]
  >>> Ks = [qml.matrix(op, wire_order=range(2)) for op in Ks]
  >>> Lambda = qml.math.choi_matrix(Ks)
  >>> np.trace(Lambda), np.trace(Lambda @ Lambda)
  (np.float64(1.0), np.float64(0.58))
  ```

* A new device preprocess transform, `~.devices.preprocess.no_analytic`, is available for hardware devices and hardware-like simulators.
  It validates that all executions are shot-based.
  [(#8037)](https://github.com/PennyLaneAI/pennylane/pull/8037)

* With program capture, the `true_fn` can now be a subclass of `Operator` when no `false_fn` is provided.
  `qml.cond(condition, qml.X)(0)` is now valid code and will return nothing, even though `qml.X` is
  technically a callable that returns an `X` operator.
  [(#8060)](https://github.com/PennyLaneAI/pennylane/pull/8060)
  [(#8101)](https://github.com/PennyLaneAI/pennylane/pull/8101)

* With program capture, an error is now raised if the conditional predicate is not a scalar.
  [(#8066)](https://github.com/PennyLaneAI/pennylane/pull/8066)

* :func:`~.tape.plxpr_to_tape` now supports converting circuits containing dynamic wire allocation.
  [(#8179)](https://github.com/PennyLaneAI/pennylane/pull/8179)

<h4>OpenQASM-PennyLane interoperability</h4>

* The :func:`qml.from_qasm3` function can now convert OpenQASM 3.0 circuits that contain
  subroutines, constants, all remaining stdlib gates, qubit registers, and built-in mathematical functions.
  [(#7651)](https://github.com/PennyLaneAI/pennylane/pull/7651)
  [(#7653)](https://github.com/PennyLaneAI/pennylane/pull/7653)
  [(#7676)](https://github.com/PennyLaneAI/pennylane/pull/7676)
  [(#7679)](https://github.com/PennyLaneAI/pennylane/pull/7679)
  [(#7677)](https://github.com/PennyLaneAI/pennylane/pull/7677)
  [(#7767)](https://github.com/PennyLaneAI/pennylane/pull/7767)
  [(#7690)](https://github.com/PennyLaneAI/pennylane/pull/7690)

<h4>Other improvements</h4>

* Program capture can now handle dynamic shots, shot vectors, and shots set with `qml.set_shots`.
  [(#7652)](https://github.com/PennyLaneAI/pennylane/pull/7652)

* Added a callback mechanism to the `qml.compiler.python_compiler` submodule to inspect the intermediate 
  representation of the program between multiple compilation passes.
  [(#7964)](https://github.com/PennyLaneAI/pennylane/pull/7964)

* The matrix factorization using :func:`~.math.decomposition.givens_decomposition` has
  been optimized to factor out the redundant sign in the diagonal phase matrix for the
  real-valued (orthogonal) rotation matrices. For example, in case the determinant of a matrix is
  :math:`-1`, only a single element of the phase matrix is required.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

* Added the `NumQubitsOp` operation to the `Quantum` dialect of the Python compiler.
[(#8063)](https://github.com/PennyLaneAI/pennylane/pull/8063)

* An error is no longer raised when non-integer wire labels are used in QNodes using `mcm_method="deferred"`.
  [(#7934)](https://github.com/PennyLaneAI/pennylane/pull/7934)
  

  ```python
  @qml.qnode(qml.device("default.qubit"), mcm_method="deferred")
  def circuit():
      m = qml.measure("a")
      qml.cond(m == 0, qml.X)("aux")
      return qml.expval(qml.Z("a"))
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
    a: ──┤↗├────┤  <Z>
  aux: ───║───X─┤     
          ╚═══╝      
  ```

* PennyLane is now compatible with `quimb` 1.11.2 after a bug affecting `default.tensor` was fixed.
  [(#7931)](https://github.com/PennyLaneAI/pennylane/pull/7931)

* The error message raised when using Python compiler transforms with :func:`pennylane.qjit` has been updated
  with suggested fixes.
  [(#7916)](https://github.com/PennyLaneAI/pennylane/pull/7916)

* A new `qml.transforms.resolve_dynamic_wires` transform can allocate concrete wire values for dynamic
  qubit allocation.
  [(#7678)](https://github.com/PennyLaneAI/pennylane/pull/7678)

* The :func:`qml.workflow.set_shots` transform can now be directly applied to a QNode without the need for `functools.partial`, providing a more user-friendly syntax and negating having to import the `functools` package.
  [(#7876)](https://github.com/PennyLaneAI/pennylane/pull/7876)
  [(#7919)](https://github.com/PennyLaneAI/pennylane/pull/7919)

  ```python
  @qml.set_shots(shots=1000)  # or @qml.set_shots(1000)
  @qml.qnode(dev)
  def circuit():
      qml.H(0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> circuit()
  0.002
  ```

* Added a `QuantumParser` class to the `qml.compiler.python_compiler` submodule that automatically loads relevant dialects.
  [(#7888)](https://github.com/PennyLaneAI/pennylane/pull/7888)

* A compilation pass written with xDSL called `qml.compiler.python_compiler.transforms.ConvertToMBQCFormalismPass` has been added for the experimental xDSL Python compiler integration. This pass converts all gates in the MBQC gate set (`Hadamard`, `S`, `RZ`, `RotXZX` and `CNOT`) to the textbook MBQC formalism.
  [(#7870)](https://github.com/PennyLaneAI/pennylane/pull/7870)

* Enforce various modules to follow modular architecture via `tach`.
  [(#7847)](https://github.com/PennyLaneAI/pennylane/pull/7847)

* Users can now specify a relative threshold value for the permissible operator norm error (`epsilon`) that
  triggers rebuilding of the cache in the `qml.clifford_t_transform`, via new `cache_eps_rtol` keyword argument.
  [(#8056)](https://github.com/PennyLaneAI/pennylane/pull/8056)

* A compilation pass written with xDSL called `qml.compiler.python_compiler.transforms.MeasurementsFromSamplesPass`
  has been added for the experimental xDSL Python compiler integration. This pass replaces all
  terminal measurements in a program with a single :func:`pennylane.sample` measurement, and adds
  postprocessing instructions to recover the original measurement.
  [(#7620)](https://github.com/PennyLaneAI/pennylane/pull/7620)

* A combine-global-phase pass has been added to the xDSL Python compiler integration.
  Note that the current implementation can only combine all the global phase operations at
  the last global phase operation in the same region. In other words, global phase operations inside a control flow region can't be combined with those in their parent
  region.
  [(#7675)](https://github.com/PennyLaneAI/pennylane/pull/7675)

* The `mbqc` xDSL dialect has been added to the Python compiler, which is used to represent
  measurement-based quantum-computing instructions in the xDSL framework.
  [(#7815)](https://github.com/PennyLaneAI/pennylane/pull/7815)
  [(#8059)](https://github.com/PennyLaneAI/pennylane/pull/8059)

* The `AllocQubitOp` and `DeallocQubitOp` operations have been added to the `Quantum` dialect in the
  Python compiler.
  [(#7915)](https://github.com/PennyLaneAI/pennylane/pull/7915)

* The :func:`pennylane.ops.rs_decomposition` method now performs exact decomposition and returns
  complete global phase information when used for decomposing a phase gate to Clifford+T basis.
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)

* `default.qubit` will default to the tree-traversal MCM method when `mcm_method="device"`.
  [(#7885)](https://github.com/PennyLaneAI/pennylane/pull/7885)

* The :func:`~.clifford_t_decomposition` transform can now handle circuits with mid-circuit
  measurements including Catalyst's measurements operations. It also now handles `RZ` and `PhaseShift`
  operations where angles are odd multiples of `±pi/4` more efficiently while using `method="gridsynth"`.
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)
  [(#7942)](https://github.com/PennyLaneAI/pennylane/pull/7942)

* The default implementation of `Device.setup_execution_config` now choses `"device"` as the default mcm method if it is available as specified by the device TOML file.
  [(#7968)](https://github.com/PennyLaneAI/pennylane/pull/7968)

<h4>Resource-efficient decompositions 🔎</h4>

* With :func:`~.decomposition.enable_graph()`, dynamically allocated wires are now supported in decomposition rules. This provides a smoother overall experience when decomposing operators in a way that requires auxiliary/work wires.
  [(#7861)](https://github.com/PennyLaneAI/pennylane/pull/7861)

  * The :func:`~.transforms.decompose` transform now accepts a `num_available_work_wires` argument that allows the user to specify the number of work wires available for dynamic allocation.
  [(#7963)](https://github.com/PennyLaneAI/pennylane/pull/7963)
  [(#7980)](https://github.com/PennyLaneAI/pennylane/pull/7980)
  [(#8103)](https://github.com/PennyLaneAI/pennylane/pull/8103)

  * Decomposition rules added for the :class:`~.MultiControlledX` that dynamically allocate work wires if none was explicitly specified via the `work_wires` argument of the operator.
  [(#8024)](https://github.com/PennyLaneAI/pennylane/pull/8024)

* A :class:`~.decomposition.decomposition_graph.DecompGraphSolution` class is added to store the solution of a decomposition graph. An instance of this class is returned from the `solve` method of the :class:`~.decomposition.decomposition_graph.DecompositionGraph`.
  [(#8031)](https://github.com/PennyLaneAI/pennylane/pull/8031)

* With the graph-based decomposition system enabled (:func:`~.decomposition.enable_graph()`), if a decomposition cannot be found for an operator in the circuit, it no longer
  raises an error. Instead, a warning is raised, and `op.decomposition()` (the current default method for decomposing gates) is
  used as a fallback, while the rest of the circuit is still decomposed with
  the new graph-based system. Additionally, a special warning message is
  raised if the circuit contains a `GlobalPhase`, reminding the user that
  `GlobalPhase` is not assumed to have a decomposition under the new system.
  [(#8156)](https://github.com/PennyLaneAI/pennylane/pull/8156)
<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

  * Fixed a queueing issue in `ResourceOperator` tests.
  [(#8204)](https://github.com/PennyLaneAI/pennylane/pull/8204)
  
* The module `qml.labs.zxopt` has been removed as its functionalities are now available in the
  submodule :mod:`~.transforms.zx`. The same functions are available, but their signature
  may have changed.
  - Instead of `qml.labs.zxopt.full_optimize`, use :func:`.transforms.zx.optimize_t_count`
  - Instead of `qml.labs.zxopt.full_reduce`, use :func:`.transforms.zx.reduce_non_clifford`
  - Instead of `qml.labs.zxopt.todd`, use :func:`.transforms.zx.todd`
  - Instead of `qml.labs.zxopt.basic_optimization`, use :func:`.transforms.zx.push_hadamards`
  [(#8177)](https://github.com/PennyLaneAI/pennylane/pull/8177)

* Added state of the art resources for the `ResourceSelectPauliRot` template and the
  `ResourceQubitUnitary` templates.
  [(#7786)](https://github.com/PennyLaneAI/pennylane/pull/7786)

* Added state of the art resources for the `ResourceSingleQubitCompare`, `ResourceTwoQubitCompare`,
  `ResourceIntegerComparator` and `ResourceRegisterComparator` templates.
  [(#7857)](https://github.com/PennyLaneAI/pennylane/pull/7857)

* Added state of the art resources for the `ResourceUniformStatePrep`,
  and `ResourceAliasSampling` templates.
  [(#7883)](https://github.com/PennyLaneAI/pennylane/pull/7883)

* Added state of the art resources for the `ResourceQFT` and `ResourceAQFT` templates.
  [(#7920)](https://github.com/PennyLaneAI/pennylane/pull/7920)

* Added an internal `dequeue()` method to the `ResourceOperator` class to simplify the 
  instantiation of resource operators which require resource operators as input.
  [(#7974)](https://github.com/PennyLaneAI/pennylane/pull/7974)

* The `catalyst` xDSL dialect has been added to the Python compiler, which contains data structures that support core compiler functionality.
  [(#7901)](https://github.com/PennyLaneAI/pennylane/pull/7901)

* New `SparseFragment` and `SparseState` classes have been created that allow to use sparse matrices for the Hamiltonian Fragments when estimating the Trotter error.
  [(#7971)](https://github.com/PennyLaneAI/pennylane/pull/7971)

* The `qec` xDSL dialect has been added to the Python compiler, which contains data structures that support quantum error correction functionality.
  [(#7985)](https://github.com/PennyLaneAI/pennylane/pull/7985)

* The `stablehlo` xDSL dialect has been added to the Python compiler, which extends the existing
  StableHLO dialect with missing upstream operations.
  [(#8036)](https://github.com/PennyLaneAI/pennylane/pull/8036)
  [(#8084)](https://github.com/PennyLaneAI/pennylane/pull/8084)
  [(#8113)](https://github.com/PennyLaneAI/pennylane/pull/8113)
  
* Added more templates with state of the art resource estimates. Users can now use the `ResourceQPE`,
  `ResourceControlledSequence`, and `ResourceIterativeQPE` templates with the resource estimation tool.
  [(#8053)](https://github.com/PennyLaneAI/pennylane/pull/8053)

* Added state of the art resources for the `ResourceTrotterProduct` template.
  [(#7910)](https://github.com/PennyLaneAI/pennylane/pull/7910)

* Updated the symbolic `ResourceOperators` to use hyperparameters from `config` dictionary.
  [(#8181)](https://github.com/PennyLaneAI/pennylane/pull/8181)

<h3>Breaking changes 💔</h3>

* `DefaultQubit.eval_jaxpr` does not use `self.shots` from device anymore; instead, it takes `shots` as a keyword argument,
  and the qnode primitive should process the `shots` and call `eval_jaxpr` accordingly.
  [(#8161)](https://github.com/PennyLaneAI/pennylane/pull/8161)

* The methods :meth:`~.pauli.PauliWord.operation` and :meth:`~.pauli.PauliSentence.operation`
  no longer queue any operators.
  [(#8136)](https://github.com/PennyLaneAI/pennylane/pull/8136)

* `qml.sample` no longer has singleton dimensions squeezed out for single shots or single wires. This cuts
  down on the complexity of post-processing due to having to handle single shot and single wire cases
  separately. The return shape will now *always* be `(shots, num_wires)`.
  [(#7944)](https://github.com/PennyLaneAI/pennylane/pull/7944)
  [(#8118)](https://github.com/PennyLaneAI/pennylane/pull/8118)

  For a simple qnode:

  ```pycon
  >>> @qml.qnode(qml.device('default.qubit'))
  ... def c():
  ...   return qml.sample(wires=0)
  ```

  Before the change, we had:
  
  ```pycon
  >>> qml.set_shots(c, shots=1)()
  0
  ```

  and now we have:

  ```pycon
  >>> qml.set_shots(c, shots=1)()
  array([[0]])
  ```

  Previous behavior can be recovered by squeezing the output:

  ```pycon
  >>> qml.math.squeeze(qml.set_shots(c, shots=1)())
  0
  ```

* `ExecutionConfig` and `MCMConfig` from `pennylane.devices` are now frozen dataclasses whose fields should be updated with `dataclass.replace`. 
  [(#7697)](https://github.com/PennyLaneAI/pennylane/pull/7697)
  [(#8046)](https://github.com/PennyLaneAI/pennylane/pull/8046)

* Functions involving an execution configuration will now default to `None` instead of `pennylane.devices.DefaultExecutionConfig` and have to be handled accordingly. 
  This prevents the potential mutation of a global object. 

  This means that functions like,
  ```python
  ...
    def some_func(..., execution_config = DefaultExecutionConfig):
      ...
  ...
  ```
  should be written as follows,
  ```python
  ...
    def some_func(..., execution_config: ExecutionConfig | None = None):
      if execution_config is None:
          execution_config = ExecutionConfig()
  ...
  ```

  [(#7697)](https://github.com/PennyLaneAI/pennylane/pull/7697)

* The `qml.HilbertSchmidt` and `qml.LocalHilbertSchmidt` templates have been updated and their UI has been remarkably simplified. 
  They now accept an operation or a list of operations as quantum unitaries.
  [(#7933)](https://github.com/PennyLaneAI/pennylane/pull/7933)

  In past versions of PennyLane, these templates required providing the `U` and `V` unitaries as a `qml.tape.QuantumTape` and a quantum function,
  respectively, along with separate parameters and wires.

  With this release, each template has been improved to accept one or more operators as  unitaries. 
  The wires and parameters of the approximate unitary `V` are inferred from the inputs, according to the order provided.

  ```python
  >>> U = qml.Hadamard(0)
  >>> V = qml.RZ(0.1, wires=1)
  >>> qml.HilbertSchmidt(V, U)
  HilbertSchmidt(0.1, wires=[0, 1])
  ```

* Remove support for Python 3.10 and adds support for 3.13.
  [(#7935)](https://github.com/PennyLaneAI/pennylane/pull/7935)

* Move custom exceptions into `exceptions.py` and add a documentation page for them in the internals.
  [(#7856)](https://github.com/PennyLaneAI/pennylane/pull/7856)

* The boolean functions provided in `qml.operation` are deprecated. See the
  :doc:`deprecations page </development/deprecations>` for equivalent code to use instead. These
  include `not_tape`, `has_gen`, `has_grad_method`, `has_multipar`, `has_nopar`, `has_unitary_gen`,
  `is_measurement`, `defines_diagonalizing_gates`, and `gen_is_multi_term_hamiltonian`.
  [(#7924)](https://github.com/PennyLaneAI/pennylane/pull/7924)

* Removed access for `lie_closure`, `structure_constants` and `center` via `qml.pauli`.
  Top level import and usage is advised. The functions now live in the `liealg` module.

  ```python
  import pennylane.liealg
  from pennylane.liealg import lie_closure, structure_constants, center
  ```

  [(#7928)](https://github.com/PennyLaneAI/pennylane/pull/7928)
  [(#7994)](https://github.com/PennyLaneAI/pennylane/pull/7994)

* `qml.operation.Observable` and the corresponding `Observable.compare` have been removed, as
  PennyLane now depends on the more general `Operator` interface instead. The
  `Operator.is_hermitian` property can instead be used to check whether or not it is highly likely
  that the operator instance is Hermitian.
  [(#7927)](https://github.com/PennyLaneAI/pennylane/pull/7927)

* `qml.operation.WiresEnum`, `qml.operation.AllWires`, and `qml.operation.AnyWires` have been removed. Setting `Operator.num_wires = None` (the default)
  should instead indicate that the `Operator` does not need wire validation.
  [(#7911)](https://github.com/PennyLaneAI/pennylane/pull/7911)

* Removed `QNode.get_gradient_fn` method. Instead, use `qml.workflow.get_best_diff_method` to obtain the differentiation method.
  [(#7907)](https://github.com/PennyLaneAI/pennylane/pull/7907)

* Top-level access to ``DeviceError``, ``PennyLaneDeprecationWarning``, ``QuantumFunctionError`` and ``ExperimentalWarning`` has been removed. Please import these objects from the new ``pennylane.exceptions`` module.
  [(#7874)](https://github.com/PennyLaneAI/pennylane/pull/7874)

* `qml.cut_circuit_mc` no longer accepts a `shots` keyword argument. The shots should instead
  be set on the tape itself.
  [(#7882)](https://github.com/PennyLaneAI/pennylane/pull/7882)

<h3>Deprecations 👋</h3>

* Setting shots on a device through the `shots=` kwarg, e.g. `qml.device("default.qubit", wires=2, shots=1000)`, is deprecated. Please use the `set_shots` transform on the `QNode` instead.

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.set_shots(1000)
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return qml.expval(qml.Z(0))
  ```

  [(#7979)](https://github.com/PennyLaneAI/pennylane/pull/7979)
  [(#8161)](https://github.com/PennyLaneAI/pennylane/pull/8161)

* Support for using TensorFlow with PennyLane has been deprecated and will be dropped in Pennylane v0.44.
  Future versions of PennyLane are not guaranteed to work with TensorFlow.
  Instead, we recommend using the :doc:`JAX </introduction/interfaces/jax>` or :doc:`PyTorch </introduction/interfaces/torch>` interface for
  machine learning applications to benefit from enhanced support and features. Please consult the following demos for
  more usage information: 
  [Turning quantum nodes into Torch Layers](https://pennylane.ai/qml/demos/tutorial_qnn_module_torch) and
  [How to optimize a QML model using JAX and Optax](https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax).
  [(#7989)](https://github.com/PennyLaneAI/pennylane/pull/7989)
  [(#8106)](https://github.com/PennyLaneAI/pennylane/pull/8106)

* `pennylane.devices.DefaultExecutionConfig` is deprecated and will be removed in v0.44.
  Instead, use `qml.devices.ExecutionConfig()` to create a default execution configuration.
  [(#7987)](https://github.com/PennyLaneAI/pennylane/pull/7987)

* Specifying the ``work_wire_type`` argument in ``qml.ctrl`` and other controlled operators as ``"clean"`` or 
  ``"dirty"`` is deprecated. Use ``"zeroed"`` to indicate that the work wires are initially in the :math:`|0\rangle`
  state, and ``"borrowed"`` to indicate that the work wires can be in any arbitrary state. In both cases, the
  work wires are restored to their original state upon completing the decomposition.
  [(#7993)](https://github.com/PennyLaneAI/pennylane/pull/7993)

* Providing `num_steps` to :func:`pennylane.evolve`, :func:`pennylane.exp`, :class:`pennylane.ops.Evolution`,
  and :class:`pennylane.ops.Exp` is deprecated and will be removed in a future release. Instead, use
  :class:`~.TrotterProduct` for approximate methods, providing the `n` parameter to perform the Suzuki-Trotter
  product approximation of a Hamiltonian with the specified number of Trotter steps.

  As a concrete example, consider the following case:

  ```python
  coeffs = [0.5, -0.6]
  ops = [qml.X(0), qml.X(0) @ qml.Y(1)]
  H_flat = qml.dot(coeffs, ops)
  ```

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
  [(#7954)](https://github.com/PennyLaneAI/pennylane/pull/7954)
  [(#7977)](https://github.com/PennyLaneAI/pennylane/pull/7977)

* `MeasurementProcess.expand` is deprecated. The relevant method can be replaced with 
  `qml.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])`
  [(#7953)](https://github.com/PennyLaneAI/pennylane/pull/7953)

* `shots=` in `QNode` calls is deprecated and will be removed in v0.44.
  Instead, please use the `qml.workflow.set_shots` transform to set the number of shots for a QNode.
  [(#7906)](https://github.com/PennyLaneAI/pennylane/pull/7906)

* ``QuantumScript.shape`` and ``QuantumScript.numeric_type`` are deprecated and will be removed in version v0.44.
  Instead, the corresponding ``.shape`` or ``.numeric_type`` of the ``MeasurementProcess`` class should be used.
  [(#7950)](https://github.com/PennyLaneAI/pennylane/pull/7950)

* Some unnecessary methods of the `qml.CircuitGraph` class are deprecated and will be removed in version v0.44:
  [(#7904)](https://github.com/PennyLaneAI/pennylane/pull/7904)

    - `print_contents` in favor of `print(obj)`
    - `observables_in_order` in favor of `observables`
    - `operations_in_order` in favor of `operations`
    - `ancestors_in_order` in favor of `ancestors(obj, sort=True)`
    - `descendants_in_order` in favore of `descendants(obj, sort=True)`

* The `QuantumScript.to_openqasm` method is deprecated and will be removed in version v0.44.
  Instead, the `qml.to_openqasm` function should be used.
  [(#7909)](https://github.com/PennyLaneAI/pennylane/pull/7909)

* The `level=None` argument in the :func:`pennylane.workflow.get_transform_program`, :func:`pennylane.workflow.construct_batch`, `qml.draw`, `qml.draw_mpl`, and `qml.specs` transforms is deprecated and will be removed in v0.43.
  Please use `level='device'` instead to apply the noise model at the device level.
  [(#7886)](https://github.com/PennyLaneAI/pennylane/pull/7886)

* `qml.qnn.cost.SquaredErrorLoss` is deprecated and will be removed in version v0.44. Instead, this hybrid workflow can be accomplished
  with a function like `loss = lambda *args: (circuit(*args) - target)**2`.
  [(#7527)](https://github.com/PennyLaneAI/pennylane/pull/7527)

* Access to `add_noise`, `insert` and noise mitigation transforms from the `pennylane.transforms` module is deprecated.
  Instead, these functions should be imported from the `pennylane.noise` module.
  [(#7854)](https://github.com/PennyLaneAI/pennylane/pull/7854)

* The `qml.QNode.add_transform` method is deprecated and will be removed in v0.43.
  Instead, please use `QNode.transform_program.push_back(transform_container=transform_container)`.
  [(#7855)](https://github.com/PennyLaneAI/pennylane/pull/7855)

<h3>Internal changes ⚙️</h3>

* Updated `CompressedResourceOp` class to track the number of wires an operator requires in labs.
  [(#8173)](https://github.com/PennyLaneAI/pennylane/pull/8173)

* Update links in `README.md`.
  [(#8165)](https://github.com/PennyLaneAI/pennylane/pull/8165)

* Update `autograph` guide to reflect new capabilities.
  [(#8132)](https://github.com/PennyLaneAI/pennylane/pull/8132)

* Start using `strict=True` to `zip` usage in source code.
  [(#8164)](https://github.com/PennyLaneAI/pennylane/pull/8164)
  [(#8182)](https://github.com/PennyLaneAI/pennylane/pull/8182)
  [(#8183)](https://github.com/PennyLaneAI/pennylane/pull/8183)

* Unpin `autoray` package in `pyproject.toml` by fixing source code that was broken by release.
  [(#8147)](https://github.com/PennyLaneAI/pennylane/pull/8147)
  [(#8159)](https://github.com/PennyLaneAI/pennylane/pull/8159)
  [(#8160)](https://github.com/PennyLaneAI/pennylane/pull/8160)

* The `autograph` keyword argument has been removed from the `QNode` constructor. 
  To enable autograph conversion, use the `qjit` decorator together with the `qml.capture.disable_autograph` context manager.
  [(#8104)](https://github.com/PennyLaneAI/pennylane/pull/8104)
  
* Add ability to disable autograph conversion using the newly added `qml.capture.disable_autograph` decorator or context manager.
  [(#8102)](https://github.com/PennyLaneAI/pennylane/pull/8102)

* Set `autoray` package upper-bound in `pyproject.toml` CI due to breaking changes in `v0.8.0`.
  [(#8110)](https://github.com/PennyLaneAI/pennylane/pull/8110)

* Add capability for roundtrip testing and module verification to the Python compiler `run_filecheck` and
`run_filecheck_qjit` fixtures.
  [(#8049)](https://github.com/PennyLaneAI/pennylane/pull/8049)

* Improve type hinting internally.
  [(#8086)](https://github.com/PennyLaneAI/pennylane/pull/8086)

* The `cond` primitive with program capture no longer stores missing false branches as `None`, instead storing them
  as jaxprs with no output.
  [(#8080)](https://github.com/PennyLaneAI/pennylane/pull/8080)

* Removed unnecessary execution tests along with accuracy validation in `tests/ops/functions/test_map_wires.py`.
  [(#8032)](https://github.com/PennyLaneAI/pennylane/pull/8032)

* Added a new `all-tests-passed` gatekeeper job to `interface-unit-tests.yml` to ensure all test
  jobs complete successfully before triggering downstream actions. This reduces the need to
  maintain a long list of required checks in GitHub settings. Also added the previously missing
  `capture-jax-tests` job to the list of required test jobs, ensuring this test suite is properly
  enforced in CI.
  [(#7996)](https://github.com/PennyLaneAI/pennylane/pull/7996)

* Equipped `DefaultQubitLegacy` (test suite only) with seeded sampling.
  This allows for reproducible sampling results of legacy classical shadow across CI.
  [(#7903)](https://github.com/PennyLaneAI/pennylane/pull/7903)

* Capture does not block `wires=0` anymore. This allows Catalyst to work with zero-wire devices.
  Note that `wires=None` is still illegal.
  [(#7978)](https://github.com/PennyLaneAI/pennylane/pull/7978)

* Improves readability of `dynamic_one_shot` postprocessing to allow further modification.
  [(#7962)](https://github.com/PennyLaneAI/pennylane/pull/7962)
  [(#8041)](https://github.com/PennyLaneAI/pennylane/pull/8041)

* Update PennyLane's top-level `__init__.py` file imports to improve Python language server support for finding
  PennyLane submodules.
  [(#7959)](https://github.com/PennyLaneAI/pennylane/pull/7959)

* Adds `measurements` as a "core" module in the tach specification.
  [(#7945)](https://github.com/PennyLaneAI/pennylane/pull/7945)

* Improves type hints in the `measurements` module.
  [(#7938)](https://github.com/PennyLaneAI/pennylane/pull/7938)

* Refactored the codebase to adopt modern type hint syntax for Python 3.11+ language features.
  [(#7860)](https://github.com/PennyLaneAI/pennylane/pull/7860)
  [(#7982)](https://github.com/PennyLaneAI/pennylane/pull/7982)

* Improve the pre-commit hook to add gitleaks.
  [(#7922)](https://github.com/PennyLaneAI/pennylane/pull/7922)

* Added a `run_filecheck_qjit` fixture that can be used to run FileCheck on integration tests for the
  `qml.compiler.python_compiler` submodule.
  [(#7888)](https://github.com/PennyLaneAI/pennylane/pull/7888)

* Added a `dialects` submodule to `qml.compiler.python_compiler` which now houses all the xDSL dialects we create.
  Additionally, the `MBQCDialect` and `QuantumDialect` dialects have been renamed to `MBQC` and `Quantum`.
  [(#7897)](https://github.com/PennyLaneAI/pennylane/pull/7897)

* Update minimum supported `pytest` version to `8.4.1`.
  [(#7853)](https://github.com/PennyLaneAI/pennylane/pull/7853)

* `DefaultQubitLegacy` (test suite only) no longer provides a customized classical shadow
  implementation
  [(#7895)](https://github.com/PennyLaneAI/pennylane/pull/7895)

* Make `pennylane.io` a tertiary module.
  [(#7877)](https://github.com/PennyLaneAI/pennylane/pull/7877)

* Seeded tests for the `split_to_single_terms` transformation.
  [(#7851)](https://github.com/PennyLaneAI/pennylane/pull/7851)

* Upgrade `rc_sync.yml` to work with latest `pyproject.toml` changes.
  [(#7808)](https://github.com/PennyLaneAI/pennylane/pull/7808)
  [(#7818)](https://github.com/PennyLaneAI/pennylane/pull/7818)

* `LinearCombination` instances can be created with `_primitive.impl` when
  capture is enabled and tracing is active.
  [(#7893)](https://github.com/PennyLaneAI/pennylane/pull/7893)

* The `TensorLike` type is now compatible with static type checkers.
  [(#7905)](https://github.com/PennyLaneAI/pennylane/pull/7905)

* Update xDSL supported version to `0.49`.
  [(#7923)](https://github.com/PennyLaneAI/pennylane/pull/7923)
  [(#7932)](https://github.com/PennyLaneAI/pennylane/pull/7932)
  [(#8120)](https://github.com/PennyLaneAI/pennylane/pull/8120)

* Update JAX version used in tests to `0.6.2`
  [(#7925)](https://github.com/PennyLaneAI/pennylane/pull/7925)

* The measurement-plane attribute of the Python compiler `mbqc` dialect now uses the "opaque syntax"
  format when printing in the generic IR format. This enables usage of this attribute when IR needs
  to be passed from the python compiler to Catalyst.
  [(#7957)](https://github.com/PennyLaneAI/pennylane/pull/7957)

* An `xdsl_extras` module has been added to the Python compiler to house additional utilities and
  functionality not available upstream in xDSL.
  [(#8067)](https://github.com/PennyLaneAI/pennylane/pull/8067)
  [(#8120)](https://github.com/PennyLaneAI/pennylane/pull/8120)

* Moved `allocation.DynamicWire` from the `allocation` to the `wires` module to avoid circular dependencies.
  [(#8179)](https://github.com/PennyLaneAI/pennylane/pull/8179)

* Two new xDSL passes have been added to the Python compiler: `decompose-graph-state`, which
  decomposes `mbqc.graph_state_prep` operations to their corresponding set of quantum operations for
  execution on state simulators, and `null-decompose-graph-state`, which replaces
  `mbqc.graph_state_prep` operations with single quantum-register allocation operations for
  execution on null devices.
  [(#8090)](https://github.com/PennyLaneAI/pennylane/pull/8090)

* :func:`.transforms.decompose` and :func:`.preprocess.decompose` now have a unified internal implementation.
  [(#8193)](https://github.com/PennyLaneAI/pennylane/pull/8193)

<h3>Documentation 📝</h3>

* Rename `ancilla` to `auxiliary` in internal documentation.
  [(#8005)](https://github.com/PennyLaneAI/pennylane/pull/8005)

* Small typos in the docstring for `qml.noise.partial_wires` have been corrected.
  [(#8052)](https://github.com/PennyLaneAI/pennylane/pull/8052)

* The theoretical background section of :class:`~.BasisRotation` has been extended to explain
  the underlying Lie group/algebra homomorphism between the (dense) rotation matrix and the
  performed operations on the target qubits.
  [(#7765)](https://github.com/PennyLaneAI/pennylane/pull/7765)

* Updated the code examples in the documentation of :func:`~.specs`.
  [(#8003)](https://github.com/PennyLaneAI/pennylane/pull/8003)

* Clarifies the use case for `Operator.pow` and `Operator.adjoint`.
  [(#7999)](https://github.com/PennyLaneAI/pennylane/pull/7999)

* The docstring of the `is_hermitian` operator property has been updated to better describe its behaviour.
  [(#7946)](https://github.com/PennyLaneAI/pennylane/pull/7946)

* Improved the docstrings of all optimizers for consistency and legibility.
  [(#7891)](https://github.com/PennyLaneAI/pennylane/pull/7891)

* Updated the code example in the documentation for :func:`~.transforms.split_non_commuting`.
  [(#7892)](https://github.com/PennyLaneAI/pennylane/pull/7892)

* Fixed :math:`\LaTeX` rendering in the documentation for `qml.TrotterProduct` and `qml.trotterize`.
  [(#8014)](https://github.com/PennyLaneAI/pennylane/pull/8014)

* Updated description of `alpha` parameter in `ClassicalShadow.entropy`.
  Trimmed the outdated part of discussion regarding different choices of `alpha`.
  [(#8100)](https://github.com/PennyLaneAI/pennylane/pull/8100)

* The :doc:`Dynamic Quantum Circuits </introduction/dynamic_quantum_circuits>` page has been updated to include the latest device-dependent mid-circuit measurement method defaults.
  [(#8149)](https://github.com/PennyLaneAI/pennylane/pull/8149)

<h3>Bug fixes 🐛</h3>

* Operators queued with :func:`pennylane.apply` no longer get dequeued by subsequent dequeuing operations
  (e.g. :func:`pennylane.adjoint`).
  [(#8078)](https://github.com/PennyLaneAI/pennylane/pull/8078)

* Fixed a bug in the decomposition rules of :class:`~.Select` with the new decomposition system
  that broke the decompositions if the target ``ops`` of the ``Select`` operator were parametrized.
  This enables the new decomposition system with ``Select`` of parametrized target ``ops``.
  [(#8186)](https://github.com/PennyLaneAI/pennylane/pull/8186)
  
* `Exp` and `Evolution` now have improved decompositions, allowing them to handle more situations
  more robustly. In particular, the generator is simplified prior to decomposition. Now more
  time evolution ops can be supported on devices that do not natively support them.
  [(#8133)](https://github.com/PennyLaneAI/pennylane/pull/8133)

* A scalar product of a norm one scalar and an operator now decomposes into a `GlobalPhase` and the operator.
  For example, `-1 * qml.X(0)` now decomposes into `[qml.GlobalPhase(-np.pi), qml.X(0)]`.
  [(#8133)](https://github.com/PennyLaneAI/pennylane/pull/8133)

* Fixes a bug that made the queueing behaviour of :meth:`~.pauli.PauliWord.operation` and
  :meth:`~.pauli.PauliSentence.operation` dependent on the global state of a program due to
  a caching issue.
  [(#8135)](https://github.com/PennyLaneAI/pennylane/pull/8135)

* A more informative error is raised when extremely deep circuits are attempted to be drawn.
  [(#8139)](https://github.com/PennyLaneAI/pennylane/pull/8139)

* An error is now raised if sequences of classically processed mid circuit measurements
  are used as input to :func:`pennylane.counts` or :func:`pennylane.probs`.
  [(#8109)](https://github.com/PennyLaneAI/pennylane/pull/8109)

* Simplifying operators raised to integer powers no longer causes recursion errors.
  [(#8044)](https://github.com/PennyLaneAI/pennylane/pull/8044)

* Fixes the GPU selection issue in `qml.math` with PyTorch when multiple GPUs are present.
  [(#8008)](https://github.com/PennyLaneAI/pennylane/pull/8008)

* The `~.for_loop` function with capture enabled can now handle over indexing
  into an empty array when `start == stop`.
  [(#8026)](https://github.com/PennyLaneAI/pennylane/pull/8026)

* Plxpr primitives now only return dynamically shaped arrays if their outputs
  actually have dynamic shapes.
  [(#8004)](https://github.com/PennyLaneAI/pennylane/pull/8004)

* Fixes an issue with tree-traversal and non-sequential wire orders.
  [(#7991)](https://github.com/PennyLaneAI/pennylane/pull/7991)

* Fixes a bug in :func:`~.matrix` where an operator's
  constituents were incorrectly queued if its decomposition was requested.
  [(#7975)](https://github.com/PennyLaneAI/pennylane/pull/7975)

* An error is now raised if an `end` statement is found in a measurement conditioned branch in a QASM string being imported into PennyLane.
  [(#7872)](https://github.com/PennyLaneAI/pennylane/pull/7872)

* Fixes issue related to :func:`~.transforms.to_zx` adding the support for
  `Toffoli` and `CCZ` gates conversion into their ZX-graph representation.
  [(#7899)](https://github.com/PennyLaneAI/pennylane/pull/7899)

* `get_best_diff_method` now correctly aligns with `execute` and `construct_batch` logic in workflows.
  [(#7898)](https://github.com/PennyLaneAI/pennylane/pull/7898)

* Resolve issues with AutoGraph transforming internal PennyLane library code due to incorrect
  module attribution of wrapper functions.
  [(#7889)](https://github.com/PennyLaneAI/pennylane/pull/7889)

* Calling `QNode.update` no longer acts as if `set_shots` has been applied.
  [(#7881)](https://github.com/PennyLaneAI/pennylane/pull/7881)

* Fixes attributes and types in the quantum dialect.
  This allows for types to be inferred correctly when parsing.
  [(#7825)](https://github.com/PennyLaneAI/pennylane/pull/7825)

* Fixes `SemiAdder` to work when inputs are defined with a single wire.
  [(#7940)](https://github.com/PennyLaneAI/pennylane/pull/7940)

* Fixes a bug where `qml.prod`, `qml.matrix`, and `qml.cond` applied on a quantum function does not dequeue operators passed as arguments to the function.
  [(#8094)](https://github.com/PennyLaneAI/pennylane/pull/8094)
  [(#8119)](https://github.com/PennyLaneAI/pennylane/pull/8119)
  [(#8078)](https://github.com/PennyLaneAI/pennylane/pull/8078)

* Fixes a bug where a copy of `ShadowExpvalMP` was incorrect for a multi-term composite observable.
  [(#8078)](https://github.com/PennyLaneAI/pennylane/pull/8078)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Ali Asadi,
Utkarsh Azad,
Astral Cai,
Joey Carter,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Marcus Edwards,
Lillian Frederiksen,
Pietropaolo Frisoni,
Simone Gasperini,
David Ittah,
Soran Jahangiri,
Korbinian Kottmann,
Mehrdad Malekmohammadi
Pablo Antonio Moreno Casares
Erick Ochoa,
Mudit Pandey,
Andrija Paurevic,
Alex Preciado,
Shuli Shu,
Jay Soni,
David Wierichs,
Jake Zaia
